from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .checkpoints import load_checkpoint
from .mcts import MCTSConfig, run_mcts
from .state_codec import (
    ACTION_DIM,
    BANK_START,
    CARD_FEATURE_LEN,
    CP_BONUSES_START,
    CP_RESERVED_START,
    CP_TOKENS_START,
    FACEUP_START,
    STATE_DIM,
)


class OpponentPolicy(Protocol):
    name: str

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        ...


_COLOR_NAMES = ("white", "blue", "green", "red", "black")
_TAKE3_TRIPLETS = (
    (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
    (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
)
_TAKE2_PAIRS = (
    (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
    (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
)


def _raw_cp_tokens(norm_state: np.ndarray) -> np.ndarray:
    arr = np.zeros((6,), dtype=np.float32)
    arr[:5] = norm_state[CP_TOKENS_START : CP_TOKENS_START + 5] * 4.0
    arr[5] = norm_state[CP_TOKENS_START + 5] * 5.0
    return arr


def _raw_cp_bonuses(norm_state: np.ndarray) -> np.ndarray:
    return norm_state[CP_BONUSES_START : CP_BONUSES_START + 5] * 7.0


def _raw_bank_tokens(norm_state: np.ndarray) -> np.ndarray:
    arr = np.zeros((6,), dtype=np.float32)
    arr[:5] = norm_state[BANK_START : BANK_START + 5] * 4.0
    arr[5] = norm_state[BANK_START + 5] * 5.0
    return arr


def _card_block_at(norm_state: np.ndarray, start: int) -> np.ndarray:
    return norm_state[start : start + CARD_FEATURE_LEN]


def _decode_card(block: np.ndarray) -> tuple[np.ndarray, int, int]:
    costs = np.rint(block[:5] * 7.0).astype(np.int32)
    points = int(round(float(block[10] * 5.0)))
    bonus_one_hot = block[5:10]
    bonus_idx = int(np.argmax(bonus_one_hot)) if float(np.sum(bonus_one_hot)) > 0 else -1
    return costs, points, bonus_idx


def _visible_cards(norm_state: np.ndarray) -> list[np.ndarray]:
    cards: list[np.ndarray] = []
    for i in range(12):
        block = _card_block_at(norm_state, FACEUP_START + i * CARD_FEATURE_LEN)
        if np.any(block != 0):
            cards.append(block)
    return cards


def _reserved_cards(norm_state: np.ndarray) -> list[np.ndarray]:
    cards: list[np.ndarray] = []
    for i in range(3):
        block = _card_block_at(norm_state, CP_RESERVED_START + i * CARD_FEATURE_LEN)
        if np.any(block != 0):
            cards.append(block)
    return cards


def _card_progress_score(norm_state: np.ndarray, block: np.ndarray) -> float:
    costs, points, bonus_idx = _decode_card(block)
    tokens = _raw_cp_tokens(norm_state)
    bonuses = _raw_cp_bonuses(norm_state)
    effective_need = np.maximum(costs.astype(np.float32) - bonuses[:5], 0.0)
    shortage = np.maximum(effective_need - tokens[:5], 0.0)
    total_need = float(np.sum(effective_need))
    total_shortage = float(np.sum(shortage))
    efficiency = 0.0
    if total_need > 0:
        efficiency = (points + 1.0) / total_need
    noble_proxy = 0.0
    if bonus_idx >= 0:
        noble_proxy = 0.5 + 0.1 * float(bonuses[bonus_idx])
    return 12.0 * float(points) + 3.0 * efficiency + 1.5 * noble_proxy - 0.4 * total_shortage


def _color_usefulness(norm_state: np.ndarray) -> np.ndarray:
    tokens = _raw_cp_tokens(norm_state)
    bonuses = _raw_cp_bonuses(norm_state)
    usefulness = np.zeros((5,), dtype=np.float32)
    candidate_cards = _visible_cards(norm_state) + _reserved_cards(norm_state)
    for block in candidate_cards:
        costs, points, _bonus_idx = _decode_card(block)
        if np.all(costs == 0):
            continue
        effective_need = np.maximum(costs.astype(np.float32) - bonuses[:5], 0.0)
        shortage = np.maximum(effective_need - tokens[:5], 0.0)
        total_short = float(np.sum(shortage))
        weight = 1.0 + float(points)
        if total_short <= 0.0:
            continue
        usefulness += weight * (shortage / total_short)
    bank = _raw_bank_tokens(norm_state)
    usefulness += 0.05 * bank[:5]
    return usefulness


def _gems_taken_from_action(action_idx: int) -> np.ndarray:
    out = np.zeros((5,), dtype=np.int32)
    if 30 <= action_idx <= 39:
        for c in _TAKE3_TRIPLETS[action_idx - 30]:
            out[c] += 1
    elif 40 <= action_idx <= 44:
        out[action_idx - 40] = 2
    elif 45 <= action_idx <= 54:
        for c in _TAKE2_PAIRS[action_idx - 45]:
            out[c] += 1
    elif 55 <= action_idx <= 59:
        out[action_idx - 55] = 1
    return out


@dataclass
class RandomOpponent:
    name: str = "random"

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        legal = np.flatnonzero(state.mask)
        if legal.size == 0:
            raise RuntimeError("RandomOpponent: no legal actions")
        return int(rng.choice(legal.tolist()))


@dataclass
class GreedyHeuristicOpponent:
    name: str = "heuristic"

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        if state.state.shape != (STATE_DIM,):
            raise ValueError(f"Unexpected state shape {state.state.shape}")
        legal = np.flatnonzero(state.mask)
        if legal.size == 0:
            raise RuntimeError("GreedyHeuristicOpponent: no legal actions")

        usefulness = _color_usefulness(state.state)
        cp_tokens = _raw_cp_tokens(state.state)
        bank = _raw_bank_tokens(state.state)
        best_action = int(legal[0])
        best_score = -float("inf")

        for action in legal.tolist():
            score = self._score_action(int(action), state.state, usefulness, cp_tokens, bank)
            if score > best_score:
                best_score = score
                best_action = int(action)
        return best_action

    def _score_action(
        self,
        action: int,
        norm_state: np.ndarray,
        usefulness: np.ndarray,
        cp_tokens: np.ndarray,
        bank: np.ndarray,
    ) -> float:
        # BUY face-up (0-11)
        if 0 <= action <= 11:
            block = _card_block_at(norm_state, FACEUP_START + action * CARD_FEATURE_LEN)
            return 100.0 + _card_progress_score(norm_state, block)
        # BUY reserved (12-14)
        if 12 <= action <= 14:
            block = _card_block_at(norm_state, CP_RESERVED_START + (action - 12) * CARD_FEATURE_LEN)
            return 95.0 + _card_progress_score(norm_state, block)
        # RESERVE face-up (15-26)
        if 15 <= action <= 26:
            rel = action - 15
            block = _card_block_at(norm_state, FACEUP_START + rel * CARD_FEATURE_LEN)
            joker_bonus = 3.0 if bank[5] > 0 else 0.0
            return 25.0 + 0.55 * _card_progress_score(norm_state, block) + joker_bonus
        # RESERVE deck (27-29)
        if 27 <= action <= 29:
            joker_bonus = 4.0 if bank[5] > 0 else 0.0
            return 10.0 + joker_bonus - 0.1 * float(action - 27)
        # TAKE_GEMS (30-59)
        if 30 <= action <= 59:
            taken = _gems_taken_from_action(action).astype(np.float32)
            total_taken = float(np.sum(taken))
            weighted_use = float(np.dot(usefulness, taken))
            diversity = float(np.count_nonzero(taken))
            token_total_after = float(np.sum(cp_tokens) + total_taken)
            overcap_penalty = max(token_total_after - 10.0, 0.0) * 2.0
            single_color_penalty = 0.4 if diversity == 1 else 0.0
            return 8.0 + 1.2 * weighted_use + 0.7 * diversity + 0.2 * total_taken - overcap_penalty - single_color_penalty
        # PASS (60)
        if action == 60:
            return -100.0
        # RETURN_GEM (61-65)
        if 61 <= action <= 65:
            c = action - 61
            abundance = float(cp_tokens[c])
            return -2.0 * float(usefulness[c]) + 0.25 * abundance
        # CHOOSE_NOBLE (66-68)
        if 66 <= action <= 68:
            return -0.01 * float(action)
        return -1e6


@dataclass
class ModelMCTSOpponent:
    model: Any
    mcts_config: MCTSConfig
    device: str = "cpu"
    name: str = "mcts_model"

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        result = run_mcts(
            env,
            self.model,
            state,
            turns_taken=turns_taken,
            device=self.device,
            config=self.mcts_config,
            rng=rng,
        )
        return int(result.action)


@dataclass
class CheckpointMCTSOpponent:
    checkpoint_path: str
    mcts_config: MCTSConfig
    device: str = "cpu"
    name: str = "checkpoint_mcts"

    def __post_init__(self) -> None:
        self._model = load_checkpoint(self.checkpoint_path, device=self.device)

    def select_action(self, env, state, *, turns_taken: int, rng: random.Random) -> int:
        result = run_mcts(
            env,
            self._model,
            state,
            turns_taken=turns_taken,
            device=self.device,
            config=self.mcts_config,
            rng=rng,
        )
        return int(result.action)
