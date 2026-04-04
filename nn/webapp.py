from __future__ import annotations

import json
import os
import random
import threading
import uuid
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .alphabeta import AlphaBetaConfig, run_alphabeta
from .checkpoints import load_checkpoint
from .ismcts import ISMCTSConfig, run_ismcts
from .mcts import MCTSConfig, run_mcts
from .native_env import SplendorNativeEnv, StepState, list_standard_cards, list_standard_nobles
from .selfplay_dataset import (
    list_sessions as list_selfplay_sessions,
    load_session_npz,
    run_selfplay_session,
    run_selfplay_session_parallel,
    save_session_npz,
)
from .state_schema import (
    ACTION_DIM,
    BANK_START,
    CARD_FEATURE_LEN,
    CP_BONUSES_START,
    CP_POINTS_IDX,
    CP_RESERVED_START,
    CP_TOKENS_START,
    FACEUP_START,
    NOBLES_START,
    OP_BONUSES_START,
    OP_POINTS_IDX,
    OPPONENT_RESERVED_SLOT_LEN,
    OP_RESERVED_IS_OCCUPIED_OFFSET,
    OP_RESERVED_START,
    OP_RESERVED_TIER_OFFSET,
    OP_TOKENS_START,
    PLAYER_INDEX_IDX,
    STATE_DIM,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "nn_artifacts" / "checkpoints"
SELFPLAY_DIR = REPO_ROOT / "nn_artifacts" / "selfplay"
WEB_DIST_DIR = REPO_ROOT / "webui" / "dist"
SPENDEE_LIVE_SAVE_PATH = REPO_ROOT / "nn_artifacts" / "spendee_bridge" / "webui_save.json"

_TAKE3_TRIPLETS = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 4),
    (0, 3, 4),
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
)
_TAKE2_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)
_COLOR_NAMES = ("white", "blue", "green", "red", "black")
_REPLAY_MODEL_CACHE: dict[str, Any] = {}
_REPLAY_MODEL_CACHE_LOCK = threading.Lock()


class CheckpointDTO(BaseModel):
    id: str
    name: str
    path: str
    created_at: str
    size_bytes: int


class ActionInfoDTO(BaseModel):
    action_idx: int
    label: str


class MoveLogEntryDTO(BaseModel):
    turn_index: int
    result_turn_index: int
    result_snapshot_index: int
    actor: Literal["P0", "P1"]
    action_idx: int
    label: str


class GameConfigDTO(BaseModel):
    checkpoint_id: str
    checkpoint_path: str
    num_simulations: int
    player_seat: Literal["P0", "P1"]
    seed: int
    manual_reveal_mode: bool = False
    analysis_mode: bool = False


class ColorCountsDTO(BaseModel):
    white: int
    blue: int
    green: int
    red: int
    black: int


class TokenCountsDTO(BaseModel):
    white: int
    blue: int
    green: int
    red: int
    black: int
    gold: int


class CardDTO(BaseModel):
    points: int
    bonus_color: Literal["white", "blue", "green", "red", "black"]
    cost: ColorCountsDTO
    source: Literal["faceup", "reserved_public", "reserved_private"]
    tier: int | None = None
    slot: int | None = None
    is_placeholder: bool = False


class NobleDTO(BaseModel):
    points: int
    requirements: ColorCountsDTO
    slot: int | None = None
    is_placeholder: bool = False


class TierRowDTO(BaseModel):
    tier: int
    deck_count: int
    cards: list[CardDTO]


class PlayerBoardDTO(BaseModel):
    seat: Literal["P0", "P1"]
    display_name: str
    points: int
    tokens: TokenCountsDTO
    bonuses: ColorCountsDTO
    reserved_public: list[CardDTO]
    reserved_total: int
    is_to_move: bool


class BoardMetaDTO(BaseModel):
    target_points: int
    turn_index: int
    player_to_move: Literal["P0", "P1"]


class BoardStateDTO(BaseModel):
    meta: BoardMetaDTO
    players: list[PlayerBoardDTO]
    bank: TokenCountsDTO
    nobles: list[NobleDTO]
    tiers: list[TierRowDTO]


class GameSnapshotDTO(BaseModel):
    game_id: str
    status: str
    player_to_move: Literal["P0", "P1"]
    legal_actions: list[int]
    legal_action_details: list[ActionInfoDTO]
    winner: int
    turn_index: int
    current_snapshot_index: int | None = None
    move_log: list[MoveLogEntryDTO]
    config: GameConfigDTO | None = None
    board_state: BoardStateDTO | None = None
    pending_reveals: list["PendingRevealDTO"] = Field(default_factory=list)
    hidden_deck_card_ids_by_tier: dict[int, list[int]] = Field(default_factory=dict)
    hidden_faceup_reveal_candidates: dict[str, list[int]] = Field(default_factory=dict)
    hidden_reserved_reveal_candidates: dict[str, list[int]] = Field(default_factory=dict)
    can_undo: bool = False
    can_redo: bool = False
    determinization_seed: int | None = None


class NewGameRequest(BaseModel):
    checkpoint_id: str
    num_simulations: int = Field(ge=1, le=10000)
    player_seat: Literal["P0", "P1"]
    seed: int | None = None
    manual_reveal_mode: bool = False
    analysis_mode: bool = True


class PlayerMoveRequest(BaseModel):
    action_idx: int


class EngineApplyRequest(BaseModel):
    job_id: str


class JumpToTurnRequest(BaseModel):
    turn_index: int = Field(ge=0)


class JumpToSnapshotRequest(BaseModel):
    snapshot_index: int = Field(ge=0)


class EngineThinkRequest(BaseModel):
    num_simulations: int | None = Field(default=None, ge=1, le=500000)
    search_type: Literal["mcts", "ismcts", "alphabeta"] = "mcts"
    continuous_until_cancel: bool = False
    max_total_simulations: int | None = Field(default=None, ge=1, le=500000)
    alphabeta_max_nodes: int | None = Field(default=None, ge=0)
    alphabeta_max_depth: int | None = Field(default=None, ge=0)
    alphabeta_max_root_actions: int | None = Field(default=None, ge=0)
    alphabeta_determinization_samples: int | None = Field(default=None, ge=1)


class RevealCardRequest(BaseModel):
    tier: int = Field(ge=1, le=3)
    slot: int = Field(ge=0, le=3)
    card_id: int = Field(gt=0)


class RevealReservedCardRequest(BaseModel):
    seat: Literal["P0", "P1"]
    slot: int = Field(ge=0, le=2)
    card_id: int = Field(gt=0)


class RevealNobleRequest(BaseModel):
    slot: int = Field(ge=0, le=2)
    noble_id: int = Field(gt=0)


class EngineThinkResponse(BaseModel):
    job_id: str
    status: Literal["QUEUED", "RUNNING"]


class PlayerMoveResponse(BaseModel):
    snapshot: GameSnapshotDTO
    engine_should_move: bool


class RevealCardResponse(BaseModel):
    snapshot: GameSnapshotDTO
    engine_should_move: bool


class EngineResultDTO(BaseModel):
    action_idx: int
    search_type: Literal["mcts", "ismcts", "alphabeta"] = "mcts"
    action_details: list[ActionVizDTO] = Field(default_factory=list)
    model_action_details: list[ActionVizDTO] | None = None
    root_value: float | None = None
    total_simulations: int | None = None
    alphabeta_terminal_lines: list["AlphaBetaTerminalLineDTO"] = Field(default_factory=list)


class EngineJobStatusDTO(BaseModel):
    job_id: str
    status: Literal["QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"]
    error: str | None = None
    result: EngineResultDTO | None = None


class PendingRevealDTO(BaseModel):
    zone: Literal["faceup_card", "reserved_card", "noble"]
    tier: int
    slot: int
    reason: Literal["initial_setup", "replacement_after_buy", "replacement_after_reserve", "reserved_from_deck", "initial_noble_setup"]
    actor: Literal["P0", "P1"] | None = None
    action_idx: int | None = None


class GameEventDTO(BaseModel):
    kind: Literal["move", "reveal_card", "reveal_reserved_card", "reveal_noble", "resign"]
    actor: Literal["P0", "P1"] | None = None
    action_idx: int | None = None
    tier: int | None = None
    slot: int | None = None
    card_id: int | None = None
    noble_id: int | None = None


class SavedStateDTO(BaseModel):
    turn_index: int = Field(ge=0)
    exported_state: dict[str, Any]


class SavedGameDTO(BaseModel):
    version: int = 2
    saved_at: str
    game_id: str
    config: GameConfigDTO
    snapshots: list[SavedStateDTO] = Field(default_factory=list)
    current_index: int = Field(default=0, ge=0)


class LiveSavedGameDTO(SavedGameDTO):
    pass


class LiveSaveStatusDTO(BaseModel):
    exists: bool
    path: str
    updated_at: str | None = None


class CatalogCardDTO(BaseModel):
    id: int
    tier: int
    points: int
    bonus_color: Literal["white", "blue", "green", "red", "black"]
    cost: ColorCountsDTO


class CatalogNobleDTO(BaseModel):
    id: int
    points: int
    requirements: ColorCountsDTO


class PlacementHintDTO(BaseModel):
    zone: Literal["faceup_card", "reserved_card", "bank_token", "other"]
    tier: int | None = None
    slot: int | None = None
    color: Literal["white", "blue", "green", "red", "black"] | None = None


class ActionVizDTO(BaseModel):
    action_idx: int
    label: str
    masked: bool
    policy_prob: float
    q_value: float | None = None
    pv_preview: str | None = None
    is_selected: bool
    placement_hint: PlacementHintDTO


class AlphaBetaTerminalLineDTO(BaseModel):
    value: float
    winner: int
    plies: int
    root_action_idx: int
    actions: list[str]


class SelfPlayRunRequest(BaseModel):
    checkpoint_id: str
    num_simulations: int = Field(ge=1, le=10000, default=400)
    games: int = Field(ge=1, le=500, default=1)
    max_turns: int = Field(ge=1, le=400, default=100)
    seed: int | None = None
    workers: int | None = Field(default=None, ge=1, le=128)


class SelfPlayRunResponse(BaseModel):
    session_id: str
    path: str
    games: int
    steps: int
    created_at: str


class SelfPlaySessionDTO(BaseModel):
    session_id: str
    display_name: str
    path: str
    created_at: str
    games: int
    steps: int
    steps_per_episode: dict[str, int]
    metadata: dict[str, Any]


class SelfPlaySessionSummaryDTO(BaseModel):
    session_id: str
    path: str
    created_at: str
    games: int
    steps: int
    steps_per_episode: dict[str, int]
    metadata: dict[str, Any]
    winners_by_episode: dict[str, int]
    cutoff_by_episode: dict[str, bool]


class ReplayStepDTO(BaseModel):
    session_id: str
    episode_idx: int
    step_idx: int
    turn_idx: int
    player_id: int
    winner: int
    reached_cutoff: bool
    value_target: float
    model_value: float | None = None
    action_selected: int
    board_state: BoardStateDTO
    action_details: list[ActionVizDTO]
    model_action_details: list[ActionVizDTO] | None = None


if hasattr(GameSnapshotDTO, "model_rebuild"):
    GameSnapshotDTO.model_rebuild()
else:
    GameSnapshotDTO.update_forward_refs()


@dataclass
class GameConfig:
    checkpoint_id: str
    checkpoint_path: Path
    num_simulations: int
    player_seat: str
    seed: int
    manual_reveal_mode: bool = False
    analysis_mode: bool = False


@dataclass
class EngineJob:
    job_id: str
    game_id: str
    status: Literal["QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"]
    cancel_event: threading.Event
    future: Future[int] | None = None
    action_idx: int | None = None
    error: str | None = None
    action_details: list[ActionVizDTO] | None = None
    model_action_details: list[ActionVizDTO] | None = None
    root_value: float | None = None
    total_simulations: int = 0
    search_type: Literal["mcts", "ismcts", "alphabeta"] = "mcts"
    alphabeta_terminal_lines: list[AlphaBetaTerminalLineDTO] | None = None


@dataclass
class MoveLogEntry:
    turn_index: int
    result_turn_index: int
    result_snapshot_index: int
    actor: str
    action_idx: int
    label: str


@dataclass
class GameEvent:
    kind: Literal["move", "reveal_card", "reveal_reserved_card", "reveal_noble", "resign"]
    actor: Literal["P0", "P1"] | None = None
    action_idx: int | None = None
    tier: int | None = None
    slot: int | None = None
    card_id: int | None = None
    noble_id: int | None = None


@dataclass
class PendingReveal:
    zone: Literal["faceup_card", "reserved_card", "noble"]
    tier: int
    slot: int
    reason: Literal["initial_setup", "replacement_after_buy", "replacement_after_reserve", "reserved_from_deck", "initial_noble_setup"]
    actor: Literal["P0", "P1"] | None = None
    action_idx: int | None = None


def _seat_str(player_id: int) -> Literal["P0", "P1"]:
    return "P0" if int(player_id) == 0 else "P1"


def _seat_display_str(seat: Literal["P0", "P1"]) -> str:
    return "P1" if seat == "P0" else "P2"


def _is_blocking_pending_reveal(item: "PendingReveal") -> bool:
    return item.zone != "reserved_card"


def _describe_action(action_idx: int) -> str:
    if 0 <= action_idx <= 11:
        return f"BUY face-up tier {action_idx // 4 + 1} slot {action_idx % 4}"
    if 12 <= action_idx <= 14:
        return f"BUY reserved slot {action_idx - 12}"
    if 15 <= action_idx <= 26:
        rel = action_idx - 15
        return f"RESERVE face-up tier {rel // 4 + 1} slot {rel % 4}"
    if 27 <= action_idx <= 29:
        return f"RESERVE from deck tier {action_idx - 27 + 1}"
    if 30 <= action_idx <= 39:
        tri = _TAKE3_TRIPLETS[action_idx - 30]
        names = ", ".join(_COLOR_NAMES[i] for i in tri)
        return f"TAKE 3 gems ({names})"
    if 40 <= action_idx <= 44:
        return f"TAKE 2 gems ({_COLOR_NAMES[action_idx - 40]})"
    if 45 <= action_idx <= 54:
        pair = _TAKE2_PAIRS[action_idx - 45]
        names = ", ".join(_COLOR_NAMES[i] for i in pair)
        return f"TAKE 2 gems ({names})"
    if 55 <= action_idx <= 59:
        return f"TAKE 1 gem ({_COLOR_NAMES[action_idx - 55]})"
    if action_idx == 60:
        return "PASS"
    if 61 <= action_idx <= 65:
        return f"RETURN gem ({_COLOR_NAMES[action_idx - 61]})"
    if 66 <= action_idx <= 68:
        return f"CHOOSE noble index {action_idx - 66}"
    return f"UNKNOWN action {action_idx}"


def _is_continuation_action(action_idx: int) -> bool:
    return 61 <= int(action_idx) <= 68


def _terminal_value_for_root(winner: int, root_player_id: int) -> float:
    if winner < 0:
        return 0.0
    return 1.0 if int(winner) == int(root_player_id) else -1.0


def _collect_alphabeta_minimax_line(
    env: SplendorNativeEnv,
    *,
    max_depth: int,
    max_nodes: int = 50000,
) -> list[AlphaBetaTerminalLineDTO]:
    root_state = env.get_state()
    root_player = int(root_state.current_player_id)
    nodes_visited = 0

    def _prefer_candidate_on_tie(
        maximizing: bool,
        value: float,
        candidate_line: list[int],
        best_line: list[int],
    ) -> bool:
        del maximizing, value
        if not best_line:
            return True
        cand_len = len(candidate_line)
        best_len = len(best_line)
        # Display policy: for equal minimax value, always show the shortest witness line.
        if cand_len != best_len:
            return cand_len < best_len
        return int(candidate_line[0]) < int(best_line[0])

    def minimax(cur_env: SplendorNativeEnv, depth: int, alpha: float, beta: float) -> tuple[float, list[int], int]:
        nonlocal nodes_visited
        if nodes_visited >= max_nodes:
            return 0.0, [], -2
        nodes_visited += 1

        step = cur_env.get_state()
        if step.is_terminal:
            return _terminal_value_for_root(int(step.winner), root_player), [], int(step.winner)
        if depth >= max_depth:
            return 0.0, [], -2

        legal = np.flatnonzero(step.mask).astype(int).tolist()
        maximizing = int(step.current_player_id) == root_player
        best_value = -float("inf") if maximizing else float("inf")
        best_line: list[int] = []
        best_winner = -2

        for action_idx in legal:
            child = cur_env.clone()
            try:
                child.step(int(action_idx))
                child_value, child_line, child_winner = minimax(child, depth + 1, alpha, beta)
            finally:
                child.close()

            candidate_line = [int(action_idx)] + child_line
            better = False
            if maximizing:
                if child_value > best_value:
                    better = True
                elif child_value == best_value and _prefer_candidate_on_tie(
                    True,
                    child_value,
                    candidate_line,
                    best_line,
                ):
                    better = True
                if child_value > alpha:
                    alpha = child_value
            else:
                if child_value < best_value:
                    better = True
                elif child_value == best_value and _prefer_candidate_on_tie(
                    False,
                    child_value,
                    candidate_line,
                    best_line,
                ):
                    better = True
                if child_value < beta:
                    beta = child_value

            if better:
                best_value = child_value
                best_line = candidate_line
                best_winner = child_winner

            if beta <= alpha:
                break

        if not best_line and legal:
            return 0.0, [int(legal[0])], -2
        return best_value, best_line, best_winner

    if max_depth <= 0:
        return []

    value, line, winner = minimax(env, 0, -float("inf"), float("inf"))
    if not line:
        return []
    return [
        AlphaBetaTerminalLineDTO(
            value=float(value),
            winner=int(winner),
            plies=len(line),
            root_action_idx=int(line[0]),
            actions=[_describe_action(action_idx) for action_idx in line],
        )
    ]


def _collect_immediate_terminal_root_lines(env: SplendorNativeEnv) -> list[AlphaBetaTerminalLineDTO]:
    root = env.get_state()
    if root.is_terminal:
        return []
    root_player = int(root.current_player_id)
    lines: list[AlphaBetaTerminalLineDTO] = []
    legal = np.flatnonzero(root.mask).astype(int).tolist()
    for action_idx in legal:
        child = env.clone()
        try:
            after = child.step(int(action_idx))
        finally:
            child.close()
        if not after.is_terminal:
            continue
        value = _terminal_value_for_root(int(after.winner), root_player)
        lines.append(
            AlphaBetaTerminalLineDTO(
                value=float(value),
                winner=int(after.winner),
                plies=1,
                root_action_idx=int(action_idx),
                actions=[_describe_action(int(action_idx))],
            )
        )
    lines.sort(key=lambda item: (-float(item.value), int(item.root_action_idx)))
    return lines


def _collect_forced_opponent_reply_win_line(env: SplendorNativeEnv) -> list[AlphaBetaTerminalLineDTO]:
    """Detect a tactical forced loss in 2 plies: whatever root does, opponent can win immediately."""
    root = env.get_state()
    if root.is_terminal:
        return []
    root_player = int(root.current_player_id)
    opponent = 1 - root_player
    legal_root = np.flatnonzero(root.mask).astype(int).tolist()
    if not legal_root:
        return []

    witness_by_root: dict[int, int] = {}
    for root_action in legal_root:
        after_root = env.clone()
        try:
            step_after_root = after_root.step(int(root_action))
            if step_after_root.is_terminal:
                continue
            legal_reply = np.flatnonzero(step_after_root.mask).astype(int).tolist()
            winning_reply = None
            for reply_action in legal_reply:
                after_reply = after_root.clone()
                try:
                    step_after_reply = after_reply.step(int(reply_action))
                finally:
                    after_reply.close()
                if step_after_reply.is_terminal and int(step_after_reply.winner) == opponent:
                    winning_reply = int(reply_action)
                    break
            if winning_reply is None:
                return []
            witness_by_root[int(root_action)] = int(winning_reply)
        finally:
            after_root.close()

    # Root is losing regardless; display the lexicographically smallest witness line.
    best_root = min(witness_by_root.keys())
    best_reply = witness_by_root[best_root]
    return [
        AlphaBetaTerminalLineDTO(
            value=-1.0,
            winner=int(opponent),
            plies=2,
            root_action_idx=int(best_root),
            actions=[_describe_action(int(best_root)), _describe_action(int(best_reply))],
        )
    ]


def _collect_alphabeta_action_previews(
    env: SplendorNativeEnv,
    *,
    max_depth: int,
    max_nodes: int,
) -> dict[int, str]:
    root = env.get_state()
    if root.is_terminal or max_depth <= 0:
        return {}

    root_player = int(root.current_player_id)
    legal_root = np.flatnonzero(root.mask).astype(int).tolist()
    if not legal_root:
        return {}

    # Keep preview generation bounded; this is auxiliary UI metadata.
    per_action_nodes = max(200, int(max_nodes) // max(1, len(legal_root)))

    def _prefer_candidate_on_tie(candidate_line: list[int], best_line: list[int]) -> bool:
        if not best_line:
            return True
        if len(candidate_line) != len(best_line):
            return len(candidate_line) < len(best_line)
        return int(candidate_line[0]) < int(best_line[0])

    previews: dict[int, str] = {}
    for root_action in legal_root:
        after_root = env.clone()
        nodes_visited = 0
        try:
            step_after_root = after_root.step(int(root_action))
            if step_after_root.is_terminal:
                previews[int(root_action)] = "Terminal after this move"
                continue

            if max_depth <= 1:
                previews[int(root_action)] = "No reply shown (depth limit)"
                continue

            def minimax(cur_env: SplendorNativeEnv, depth: int, alpha: float, beta: float) -> tuple[float, list[int], int]:
                nonlocal nodes_visited
                if nodes_visited >= per_action_nodes:
                    return 0.0, [], -2
                nodes_visited += 1

                step = cur_env.get_state()
                if step.is_terminal:
                    return _terminal_value_for_root(int(step.winner), root_player), [], int(step.winner)
                if depth >= max_depth:
                    return 0.0, [], -2

                legal = np.flatnonzero(step.mask).astype(int).tolist()
                maximizing = int(step.current_player_id) == root_player
                best_value = -float("inf") if maximizing else float("inf")
                best_line: list[int] = []
                best_winner = -2

                for action_idx in legal:
                    child = cur_env.clone()
                    try:
                        child.step(int(action_idx))
                        child_value, child_line, child_winner = minimax(child, depth + 1, alpha, beta)
                    finally:
                        child.close()

                    candidate_line = [int(action_idx)] + child_line
                    better = False
                    if maximizing:
                        if child_value > best_value:
                            better = True
                        elif child_value == best_value and _prefer_candidate_on_tie(candidate_line, best_line):
                            better = True
                        if child_value > alpha:
                            alpha = child_value
                    else:
                        if child_value < best_value:
                            better = True
                        elif child_value == best_value and _prefer_candidate_on_tie(candidate_line, best_line):
                            better = True
                        if child_value < beta:
                            beta = child_value

                    if better:
                        best_value = child_value
                        best_line = candidate_line
                        best_winner = child_winner

                    if beta <= alpha:
                        break

                if not best_line and legal:
                    return 0.0, [int(legal[0])], -2
                return best_value, best_line, best_winner

            _value, child_line, _winner = minimax(after_root, 1, -float("inf"), float("inf"))
            if not child_line:
                previews[int(root_action)] = "No stable reply found"
            else:
                shown = [_describe_action(int(a)) for a in child_line[:2]]
                suffix = " -> ..." if len(child_line) > 2 else ""
                previews[int(root_action)] = f"Reply: {' -> '.join(shown)}{suffix}"
        finally:
            after_root.close()

    return previews


def _manual_reveal_for_action(action_idx: int, actor: str, step_after: StepState) -> PendingReveal | None:
    if 0 <= action_idx <= 11:
        return PendingReveal(
            zone="faceup_card",
            tier=action_idx // 4 + 1,
            slot=action_idx % 4,
            reason="replacement_after_buy",
            actor=_seat_str(0 if actor == "P0" else 1),
            action_idx=action_idx,
        )
    if 15 <= action_idx <= 26:
        rel = action_idx - 15
        return PendingReveal(
            zone="faceup_card",
            tier=rel // 4 + 1,
            slot=rel % 4,
            reason="replacement_after_reserve",
            actor=_seat_str(0 if actor == "P0" else 1),
            action_idx=action_idx,
        )
    if 27 <= action_idx <= 29:
        actor_seat = _seat_str(0 if actor == "P0" else 1)
        if step_after.current_player_id in (0, 1) and _seat_str(step_after.current_player_id) != actor_seat:
            occupied_reserved = 0
            for i in range(3):
                slot_block = _safe_slice(
                    step_after.state,
                    OP_RESERVED_START + i * OPPONENT_RESERVED_SLOT_LEN,
                    OPPONENT_RESERVED_SLOT_LEN,
                )
                occupied_reserved += int(round(float(slot_block[OP_RESERVED_IS_OCCUPIED_OFFSET])))
            slot = occupied_reserved - 1
        else:
            visible_reserved = 0
            for i in range(3):
                block = _safe_slice(step_after.state, CP_RESERVED_START + i * CARD_FEATURE_LEN, CARD_FEATURE_LEN)
                if np.any(block):
                    visible_reserved += 1
            slot = visible_reserved
        if slot < 0 or slot > 2:
            raise HTTPException(status_code=500, detail="Could not determine reserved slot for deck reserve")
        return PendingReveal(
            zone="reserved_card",
            tier=action_idx - 27 + 1,
            slot=slot,
            reason="reserved_from_deck",
            actor=actor_seat,
            action_idx=action_idx,
        )
    return None


def _initial_setup_pending_reveals() -> list[PendingReveal]:
    pending: list[PendingReveal] = []
    for tier in (1, 2, 3):
        for slot in range(4):
            pending.append(
                PendingReveal(
                    zone="faceup_card",
                    tier=tier,
                    slot=slot,
                    reason="initial_setup",
                )
            )
    for slot in range(3):
        pending.append(
            PendingReveal(
                zone="noble",
                tier=0,
                slot=slot,
                reason="initial_noble_setup",
            )
        )
    return pending


def _placement_hint_for_action(action_idx: int) -> PlacementHintDTO:
    if 0 <= action_idx <= 11:
        return PlacementHintDTO(
            zone="faceup_card",
            tier=(action_idx // 4) + 1,
            slot=(action_idx % 4),
        )
    if 12 <= action_idx <= 14:
        return PlacementHintDTO(
            zone="reserved_card",
            slot=(action_idx - 12),
        )
    if 15 <= action_idx <= 26:
        rel = action_idx - 15
        return PlacementHintDTO(
            zone="faceup_card",
            tier=(rel // 4) + 1,
            slot=(rel % 4),
        )
    if 27 <= action_idx <= 29:
        return PlacementHintDTO(zone="other", tier=(action_idx - 27 + 1))
    if 30 <= action_idx <= 39:
        # Use the first color in the TAKE-3 tuple as a stable placement anchor.
        color_idx = _TAKE3_TRIPLETS[action_idx - 30][0]
        return PlacementHintDTO(zone="bank_token", color=_COLOR_NAMES[color_idx])
    if 40 <= action_idx <= 44:
        return PlacementHintDTO(zone="bank_token", color=_COLOR_NAMES[action_idx - 40])
    if 45 <= action_idx <= 59:
        pair = _TAKE2_PAIRS[action_idx - 45] if action_idx <= 54 else (_COLOR_NAMES[action_idx - 55],)
        color = _COLOR_NAMES[pair[0]] if isinstance(pair[0], int) else pair[0]
        return PlacementHintDTO(zone="bank_token", color=color)
    if 61 <= action_idx <= 65:
        return PlacementHintDTO(zone="bank_token", color=_COLOR_NAMES[action_idx - 61])
    return PlacementHintDTO(zone="other")


def _action_viz_rows(
    mask: np.ndarray,
    policy: np.ndarray,
    selected_action: int,
    q_values: np.ndarray | None = None,
    pv_previews: dict[int, str] | None = None,
) -> list[ActionVizDTO]:
    out: list[ActionVizDTO] = []
    for action_idx in range(ACTION_DIM):
        q_value = None
        if q_values is not None:
            q_value = float(q_values[action_idx])
        out.append(
            ActionVizDTO(
                action_idx=int(action_idx),
                label=_describe_action(action_idx),
                masked=not bool(mask[action_idx]),
                policy_prob=float(policy[action_idx]),
                q_value=q_value,
                pv_preview=(None if pv_previews is None else pv_previews.get(int(action_idx))),
                is_selected=(int(selected_action) == int(action_idx)),
                placement_hint=_placement_hint_for_action(action_idx),
            )
        )
    return out


def _best_legal_action(mask: np.ndarray, policy: np.ndarray) -> int:
    legal = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    if legal.size == 0:
        raise RuntimeError("No legal actions available")
    best_idx = int(legal[0])
    best_prob = float(policy[best_idx])
    for action_idx in legal[1:]:
        prob = float(policy[int(action_idx)])
        if prob > best_prob:
            best_idx = int(action_idx)
            best_prob = prob
    return best_idx


def _masked_softmax(policy_scores: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    probs = np.zeros((ACTION_DIM,), dtype=np.float32)
    legal = np.flatnonzero(np.asarray(legal_mask, dtype=np.bool_))
    if legal.size == 0:
        return probs
    legal_scores = np.asarray(policy_scores, dtype=np.float32)[legal]
    finite = np.isfinite(legal_scores)
    if not bool(np.any(finite)):
        probs[legal] = np.float32(1.0 / float(legal.size))
        return probs
    max_score = float(np.max(legal_scores[finite]))
    weights = np.zeros((int(legal.size),), dtype=np.float64)
    for i, score in enumerate(legal_scores):
        if np.isfinite(score):
            weights[i] = np.exp(float(score) - max_score)
    weight_sum = float(weights.sum())
    if not (weight_sum > 0.0) or not np.isfinite(weight_sum):
        probs[legal] = np.float32(1.0 / float(legal.size))
        return probs
    probs[legal] = (weights / weight_sum).astype(np.float32)
    return probs


def _load_replay_model(checkpoint_path: Path):
    key = str(checkpoint_path.resolve())
    with _REPLAY_MODEL_CACHE_LOCK:
        model = _REPLAY_MODEL_CACHE.get(key)
        if model is None:
            model = load_checkpoint(checkpoint_path, device="cpu")
            _REPLAY_MODEL_CACHE[key] = model
    return model


def _evaluate_model_replay_state(
    metadata: dict[str, Any],
    state: np.ndarray,
    mask: np.ndarray,
    selected_action: int,
) -> tuple[np.ndarray, float] | None:
    checkpoint_path_value = metadata.get("checkpoint_path")
    if checkpoint_path_value is None:
        return None
    checkpoint_path = Path(str(checkpoint_path_value))
    if not checkpoint_path.exists():
        return None

    state_np = np.asarray(state, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=np.bool_)
    if state_np.shape != (STATE_DIM,) or mask_np.shape != (ACTION_DIM,):
        return None
    if selected_action < 0 or selected_action >= ACTION_DIM:
        return None

    model = _load_replay_model(checkpoint_path)
    state_t = torch.as_tensor(state_np[None, :], dtype=torch.float32)
    with torch.no_grad():
        logits_t, value_t = model(state_t)

    logits = logits_t.detach().cpu().numpy().reshape(-1)
    if logits.shape != (ACTION_DIM,):
        return None
    value_arr = value_t.detach().cpu().numpy().reshape(-1)
    if value_arr.size != 1:
        return None

    policy = _masked_softmax(logits, mask_np)
    return policy, float(value_arr[0])


def _mask_to_actions(mask: np.ndarray) -> list[int]:
    return [int(v) for v in np.flatnonzero(mask)]


def _json_safe_random_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe_random_state(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_random_state(item) for item in value]
    return value


def _random_state_from_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_random_state_from_json(item) for item in value)
    return value


def _normalize_state_for_inference(state: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(state)
    # Spendee bridge metadata contains observation timestamps and external
    # counters that are not produced by native step() and should not affect
    # action inference.
    normalized.pop("metadata", None)
    # Bridge/state exports can differ on bookkeeping-only fields that are not
    # part of the actual game mechanics transition.
    normalized.pop("move_number", None)
    # Hidden deck order is not guaranteed to align between bridge snapshots and
    # native replay state, and is not required for inferring public move intent.
    normalized.pop("deck_card_ids_by_tier", None)

    bank = normalized.get("bank")
    if isinstance(bank, dict):
        if "joker" in bank and "gold" not in bank:
            bank["gold"] = bank.pop("joker")
        bank.pop("joker", None)

    players = normalized.get("players")
    if isinstance(players, list):
        for player in players:
            if not isinstance(player, dict):
                continue
            tokens = player.get("tokens")
            if not isinstance(tokens, dict):
                continue
            if "joker" in tokens and "gold" not in tokens:
                tokens["gold"] = tokens.pop("joker")
            tokens.pop("joker", None)

    return normalized


def _canonicalize_token_map(tokens: Any) -> dict[str, int] | None:
    if not isinstance(tokens, dict):
        return None
    out = dict(tokens)
    if "joker" in out and "gold" not in out:
        out["gold"] = out.pop("joker")
    out.pop("joker", None)
    return {str(k): int(v) for k, v in out.items()}


def _observable_state_for_inference(state: dict[str, Any]) -> dict[str, Any]:
    players_raw = state.get("players") if isinstance(state.get("players"), list) else []
    players: list[dict[str, Any]] = []
    for player in players_raw:
        if not isinstance(player, dict):
            players.append({})
            continue
        players.append(
            {
                "tokens": _canonicalize_token_map(player.get("tokens")),
                "bonuses": _canonicalize_token_map(player.get("bonuses")),
                "points": int(player.get("points", 0)),
                "reserved_card_ids": list(player.get("reserved_card_ids") or []),
            }
        )

    return {
        "current_player": int(state.get("current_player", -1)),
        "phase_flags": dict(state.get("phase_flags") or {}),
        "bank": _canonicalize_token_map(state.get("bank")),
        "players": players,
        "available_noble_ids": list(state.get("available_noble_ids") or []),
    }


def _states_equal(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    return _normalize_state_for_inference(lhs) == _normalize_state_for_inference(rhs)


def _spendee_action_history_delta(start_state: dict[str, Any], end_state: dict[str, Any]) -> list[int] | None:
    start_meta = start_state.get("metadata") if isinstance(start_state, dict) else None
    end_meta = end_state.get("metadata") if isinstance(end_state, dict) else None
    if not isinstance(start_meta, dict) or not isinstance(end_meta, dict):
        return None

    start_hist = start_meta.get("spendee_action_history")
    end_hist = end_meta.get("spendee_action_history")
    if not isinstance(start_hist, list) or not isinstance(end_hist, list):
        return None
    if len(end_hist) <= len(start_hist):
        return None
    if end_hist[: len(start_hist)] != start_hist:
        return None

    delta: list[int] = []
    for value in end_hist[len(start_hist) :]:
        if not isinstance(value, int):
            return None
        if value < 0 or value >= ACTION_DIM:
            return None
        delta.append(int(value))
    return delta if delta else None


def _spendee_action_history_len(state: dict[str, Any]) -> int | None:
    meta = state.get("metadata") if isinstance(state, dict) else None
    if not isinstance(meta, dict):
        return None
    hist = meta.get("spendee_action_history")
    if not isinstance(hist, list):
        return None
    return len(hist)


def _game_event_to_dto(event: GameEvent) -> GameEventDTO:
    return GameEventDTO(
        kind=event.kind,
        actor=event.actor,
        action_idx=event.action_idx,
        tier=event.tier,
        slot=event.slot,
        card_id=event.card_id,
        noble_id=event.noble_id,
    )


def _game_event_from_dto(event: GameEventDTO) -> GameEvent:
    return GameEvent(
        kind=event.kind,
        actor=event.actor,
        action_idx=event.action_idx,
        tier=event.tier,
        slot=event.slot,
        card_id=event.card_id,
        noble_id=event.noble_id,
    )


def _resolve_checkpoint_id(checkpoint_id: str) -> Path:
    allowed = {item.id: Path(item.path) for item in _scan_checkpoints()}
    path = allowed.get(checkpoint_id)
    if path is None:
        raise HTTPException(status_code=400, detail="Invalid checkpoint_id")
    return path


def _selfplay_session_path(session_id: str) -> Path:
    return SELFPLAY_DIR / f"{session_id}.npz"


def _to_int(value: float, *, scale: float, max_hint: int | None = None) -> int:
    out = int(round(float(value) * scale))
    if out < 0:
        return 0
    if max_hint is not None:
        return min(out, int(max_hint))
    return out


def _decode_color_counts(block: np.ndarray, *, scale: float, max_hint: int | None = None) -> ColorCountsDTO:
    return ColorCountsDTO(
        white=_to_int(block[0], scale=scale, max_hint=max_hint),
        blue=_to_int(block[1], scale=scale, max_hint=max_hint),
        green=_to_int(block[2], scale=scale, max_hint=max_hint),
        red=_to_int(block[3], scale=scale, max_hint=max_hint),
        black=_to_int(block[4], scale=scale, max_hint=max_hint),
    )


def _decode_token_counts(block: np.ndarray) -> TokenCountsDTO:
    return TokenCountsDTO(
        white=_to_int(block[0], scale=4.0, max_hint=7),
        blue=_to_int(block[1], scale=4.0, max_hint=7),
        green=_to_int(block[2], scale=4.0, max_hint=7),
        red=_to_int(block[3], scale=4.0, max_hint=7),
        black=_to_int(block[4], scale=4.0, max_hint=7),
        gold=_to_int(block[5], scale=5.0, max_hint=7),
    )


def _decode_card(
    block: np.ndarray,
    *,
    source: Literal["faceup", "reserved_public", "reserved_private"],
    tier: int | None = None,
    slot: int | None = None,
) -> CardDTO | None:
    if block.shape != (CARD_FEATURE_LEN,):
        return None
    if not np.any(block):
        return None
    costs = _decode_color_counts(block[:5], scale=7.0, max_hint=7)
    bonus_slice = block[5:10]
    color_idx = int(np.argmax(bonus_slice))
    bonus_color = _COLOR_NAMES[color_idx]
    points = _to_int(block[10], scale=5.0, max_hint=5)
    return CardDTO(
        points=points,
        bonus_color=bonus_color,  # type: ignore[arg-type]
        cost=costs,
        source=source,
        tier=tier,
        slot=slot,
    )


def _private_reserved_placeholder(slot: int, tier: int | None = None) -> CardDTO:
    return CardDTO(
        points=0,
        bonus_color="white",
        cost=ColorCountsDTO(white=0, blue=0, green=0, red=0, black=0),
        source="reserved_private",
        tier=tier,
        slot=slot,
        is_placeholder=True,
    )


def _safe_slice(state: np.ndarray, start: int, length: int) -> np.ndarray:
    end = start + length
    if end > state.shape[0]:
        raise ValueError(f"decode slice out of range: start={start}, length={length}, shape={state.shape}")
    return state[start:end]


def _decode_board_state(
    step: StepState,
    *,
    turn_index: int,
    player_seat: str,
    pending_reveals: list[PendingReveal] | None = None,
    hidden_deck_card_ids_by_tier: dict[int, list[int]] | None = None,
) -> BoardStateDTO:
    state = np.asarray(step.state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Unexpected state shape for board decode: {state.shape}")

    cp_tokens = _decode_token_counts(_safe_slice(state, CP_TOKENS_START, 6))
    cp_bonuses = _decode_color_counts(_safe_slice(state, CP_BONUSES_START, 5), scale=7.0, max_hint=7)
    cp_points = _to_int(state[CP_POINTS_IDX], scale=20.0, max_hint=30)

    op_tokens = _decode_token_counts(_safe_slice(state, OP_TOKENS_START, 6))
    op_bonuses = _decode_color_counts(_safe_slice(state, OP_BONUSES_START, 5), scale=7.0, max_hint=7)
    op_points = _to_int(state[OP_POINTS_IDX], scale=20.0, max_hint=30)

    cp_reserved: list[CardDTO] = []
    for i in range(3):
        block = _safe_slice(state, CP_RESERVED_START + i * CARD_FEATURE_LEN, CARD_FEATURE_LEN)
        card = _decode_card(block, source="reserved_public", slot=i)
        if card is not None:
            cp_reserved.append(card)

    op_reserved: list[CardDTO] = []
    op_reserved_total = 0
    for i in range(3):
        slot_block = _safe_slice(state, OP_RESERVED_START + i * OPPONENT_RESERVED_SLOT_LEN, OPPONENT_RESERVED_SLOT_LEN)
        card = _decode_card(slot_block[:CARD_FEATURE_LEN], source="reserved_public", slot=i)
        is_occupied = bool(round(float(slot_block[OP_RESERVED_IS_OCCUPIED_OFFSET])))
        encoded_tier = _to_int(slot_block[OP_RESERVED_TIER_OFFSET], scale=3.0, max_hint=3)
        if is_occupied:
            op_reserved_total += 1
        if card is not None:
            op_reserved.append(card)
        elif is_occupied:
            if encoded_tier not in (1, 2, 3):
                raise ValueError(f"Opponent reserved slot {i} has invalid encoded tier {encoded_tier}")
            op_reserved.append(_private_reserved_placeholder(i, tier=encoded_tier))

    pending_reveals = pending_reveals or []
    pending_reserved_by_actor: dict[str, dict[int, int | None]] = {"P0": {}, "P1": {}}
    for item in pending_reveals:
        if item.zone == "reserved_card" and item.actor in ("P0", "P1"):
            tier_hint = int(item.tier) if int(item.tier) in (1, 2, 3) else None
            pending_reserved_by_actor[item.actor][int(item.slot)] = tier_hint

    encoded_player_index = int(round(float(state[PLAYER_INDEX_IDX])))
    if encoded_player_index not in (0, 1):
        raise ValueError(f"Unexpected player_index in state vector: {encoded_player_index}")

    cp_id = int(step.current_player_id)
    if cp_id not in (0, 1):
        raise ValueError(f"Unexpected current_player_id: {cp_id}")
    if encoded_player_index != cp_id:
        raise ValueError(
            f"State vector player_index {encoded_player_index} does not match current_player_id {cp_id}"
        )
    op_id = 1 - cp_id

    cp_seat = _seat_str(cp_id)
    op_seat = _seat_str(op_id)
    cp_pending_slots = set(pending_reserved_by_actor[cp_seat].keys())
    op_pending_slots = set(pending_reserved_by_actor[op_seat].keys())

    # Hide any still-pending reserved reveal, even if it is currently encoded
    # in the current player's private state block.
    cp_reserved = [card for card in cp_reserved if int(card.slot or 0) not in cp_pending_slots]
    op_reserved = [card for card in op_reserved if int(card.slot or 0) not in op_pending_slots]

    cp_reserved_total = len(cp_reserved) + len(cp_pending_slots)

    for slot in sorted(cp_pending_slots):
        cp_reserved.append(
            _private_reserved_placeholder(
                slot,
                tier=pending_reserved_by_actor[cp_seat].get(slot),
            )
        )

    for slot in sorted(op_pending_slots):
        if all(int(card.slot or -1) != slot for card in op_reserved):
            op_reserved.append(
                _private_reserved_placeholder(
                    slot,
                    tier=pending_reserved_by_actor[op_seat].get(slot),
                )
            )

    cp_reserved.sort(key=lambda card: int(card.slot or 0))
    op_reserved.sort(key=lambda card: int(card.slot or 0))

    players: dict[int, PlayerBoardDTO] = {
        cp_id: PlayerBoardDTO(
            seat=cp_seat,
            display_name=_seat_display_str(cp_seat),
            points=cp_points,
            tokens=cp_tokens,
            bonuses=cp_bonuses,
            reserved_public=cp_reserved,
            reserved_total=cp_reserved_total,
            is_to_move=True,
        ),
        op_id: PlayerBoardDTO(
            seat=op_seat,
            display_name=_seat_display_str(op_seat),
            points=op_points,
            tokens=op_tokens,
            bonuses=op_bonuses,
            reserved_public=op_reserved,
            reserved_total=op_reserved_total,
            is_to_move=False,
        ),
    }

    bank = _decode_token_counts(_safe_slice(state, BANK_START, 6))

    nobles: list[NobleDTO] = []
    for i in range(3):
        block = _safe_slice(state, NOBLES_START + i * 5, 5)
        if not np.any(block):
            continue
        nobles.append(NobleDTO(points=3, requirements=_decode_color_counts(block, scale=4.0, max_hint=4), slot=i))

    tiers: list[TierRowDTO] = []
    for tier in (3, 2, 1):
        cards: list[CardDTO] = []
        tier_offset = FACEUP_START + (tier - 1) * 4 * CARD_FEATURE_LEN
        for slot in range(4):
            block = _safe_slice(state, tier_offset + slot * CARD_FEATURE_LEN, CARD_FEATURE_LEN)
            card = _decode_card(block, source="faceup", tier=tier, slot=slot)
            if card is not None:
                cards.append(card)
        deck_count = len((hidden_deck_card_ids_by_tier or {}).get(int(tier), []))
        tiers.append(TierRowDTO(tier=tier, deck_count=deck_count, cards=cards))

    return BoardStateDTO(
        meta=BoardMetaDTO(
            target_points=15,
            turn_index=int(turn_index),
            player_to_move=_seat_str(step.current_player_id),
        ),
        players=[players[0], players[1]],
        bank=bank,
        nobles=nobles,
        tiers=tiers,
    )


def _scan_checkpoints() -> list[CheckpointDTO]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[CheckpointDTO] = []
    for path in CHECKPOINT_DIR.glob("*.pt"):
        st = path.stat()
        created = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        items.append(
            CheckpointDTO(
                id=str(path.resolve()),
                name=path.name,
                path=str(path.resolve()),
                created_at=created,
                size_bytes=int(st.st_size),
            )
        )
    items.sort(key=lambda item: item.created_at, reverse=True)
    return items


def _decode_replay_step(session_id: str, session_path: Path, episode_idx: int, step_idx: int) -> ReplayStepDTO:
    session = load_session_npz(session_path)
    target = None
    for step in session.steps:
        if int(step.episode_idx) == int(episode_idx) and int(step.step_idx) == int(step_idx):
            target = step
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Replay step not found")

    step_state = StepState(
        state=target.state.copy(),
        mask=target.mask.copy(),
        is_terminal=bool(target.winner != -2),
        winner=int(target.winner),
        current_player_id=int(target.current_player_id),
    )
    board_state = _decode_board_state(
        step_state,
        turn_index=int(target.turn_idx),
        player_seat=_seat_str(int(target.current_player_id)),
    )
    model_value: float | None = None
    model_action_details: list[ActionVizDTO] | None = None
    model_eval = _evaluate_model_replay_state(
        session.metadata,
        target.state,
        target.mask,
        target.action_selected,
    )
    if model_eval is not None:
        model_policy, model_value = model_eval
        model_action_details = _action_viz_rows(target.mask, model_policy, target.action_selected)
    return ReplayStepDTO(
        session_id=session_id,
        episode_idx=int(target.episode_idx),
        step_idx=int(target.step_idx),
        turn_idx=int(target.turn_idx),
        player_id=int(target.player_id),
        winner=int(target.winner),
        reached_cutoff=bool(target.reached_cutoff),
        value_target=float(target.value_target),
        model_value=model_value,
        action_selected=int(target.action_selected),
        board_state=board_state,
        action_details=_action_viz_rows(target.mask, target.policy, target.action_selected),
        model_action_details=model_action_details,
    )


def _build_selfplay_summary(session_id: str, session_path: Path) -> SelfPlaySessionSummaryDTO:
    session = load_session_npz(session_path)
    by_episode_steps: dict[str, int] = {}
    winners: dict[str, int] = {}
    cutoffs: dict[str, bool] = {}
    for step in session.steps:
        key = str(int(step.episode_idx))
        by_episode_steps[key] = by_episode_steps.get(key, 0) + 1
        winners[key] = int(step.winner)
        cutoffs[key] = bool(step.reached_cutoff)
    return SelfPlaySessionSummaryDTO(
        session_id=session_id,
        path=str(session_path.resolve()),
        created_at=session.created_at,
        games=int(session.metadata.get("games", 0)),
        steps=len(session.steps),
        steps_per_episode=by_episode_steps,
        metadata=session.metadata,
        winners_by_episode=winners,
        cutoff_by_episode=cutoffs,
    )


class GameManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-think")
        self._env: SplendorNativeEnv | None = None
        self._game_id: str | None = None
        self._config: GameConfig | None = None
        self._turn_index: int = 0
        self._move_log: list[MoveLogEntry] = []
        self._rng = random.Random(0)
        self._model_cache: dict[tuple[str, str], Any] = {}
        self._active_job_id: str | None = None
        self._jobs: dict[str, EngineJob] = {}
        self._forced_winner: int | None = None
        self._pending_reveals: list[PendingReveal] = []
        self._setup_event_log: list[GameEvent] = []
        self._event_log: list[GameEvent] = []
        self._redo_log: list[GameEvent] = []
        self._snapshot_history: list[SavedStateDTO] = []
        self._snapshot_history_index: int | None = None
        self._loaded_snapshot_history: list[SavedStateDTO] = []
        self._determinization_seed: int | None = None

    def list_checkpoints(self) -> list[CheckpointDTO]:
        return _scan_checkpoints()

    def list_standard_cards(self) -> list[CatalogCardDTO]:
        return [CatalogCardDTO(**card) for card in list_standard_cards()]

    def list_standard_nobles(self) -> list[CatalogNobleDTO]:
        return [CatalogNobleDTO(**noble) for noble in list_standard_nobles()]

    def _cancel_active_job_locked(self) -> None:
        if self._active_job_id is None:
            return
        job = self._jobs.get(self._active_job_id)
        if job is None:
            self._active_job_id = None
            return
        job.cancel_event.set()
        if job.status in ("QUEUED", "RUNNING"):
            job.status = "CANCELLED"
        self._active_job_id = None

    def _cancel_all_engine_jobs_locked(self) -> None:
        for job in self._jobs.values():
            if job.status in ("QUEUED", "RUNNING"):
                job.cancel_event.set()
                job.status = "CANCELLED"
        self._active_job_id = None

    def _ensure_env_locked(self) -> SplendorNativeEnv:
        if self._env is None:
            self._env = SplendorNativeEnv()
        return self._env

    def _require_game_locked(self) -> tuple[SplendorNativeEnv, GameConfig, str]:
        if self._game_id is None or self._config is None:
            raise HTTPException(status_code=400, detail="No active game")
        env = self._ensure_env_locked()
        return env, self._config, self._game_id

    def _resolve_checkpoint(self, checkpoint_id: str) -> Path:
        return _resolve_checkpoint_id(checkpoint_id)

    def _resolve_saved_checkpoint(self, config: GameConfigDTO) -> Path:
        try:
            return self._resolve_checkpoint(config.checkpoint_id)
        except HTTPException:
            saved_path = Path(config.checkpoint_path)
            if saved_path.exists():
                return saved_path
        raise HTTPException(
            status_code=400,
            detail="Saved checkpoint could not be resolved; ensure the original checkpoint still exists",
        )

    def _clear_snapshot_history_locked(self) -> None:
        self._snapshot_history = []
        self._snapshot_history_index = None

    def _infer_actions_between_snapshots_locked(self, start_state: dict[str, Any], end_state: dict[str, Any]) -> list[int] | None:
        hist_delta = _spendee_action_history_delta(start_state, end_state)
        if hist_delta:
            probe_hist = self._ensure_env_locked().clone()
            probe_hist.load_state(start_state)
            legal_sequence = True
            try:
                for action_idx in hist_delta:
                    step = probe_hist.get_state()
                    if action_idx < 0 or action_idx >= int(step.mask.shape[0]) or not bool(step.mask[action_idx]):
                        legal_sequence = False
                        break
                    probe_hist.step(action_idx)
                if legal_sequence and _states_equal(probe_hist.export_state(), end_state):
                    return hist_delta
                # Hidden deck ordering can differ between bridge observations and
                # native replay state. If metadata supplies a legal action delta,
                # accept it for move-log inference even when hidden state differs.
                if legal_sequence:
                    return hist_delta
            except Exception:
                pass

        probe = self._ensure_env_locked().clone()
        probe.load_state(start_state)
        start_step = probe.get_state()
        legal_first = np.flatnonzero(np.asarray(start_step.mask, dtype=np.bool_))

        for first in legal_first:
            first_idx = int(first)
            first_env = probe.clone()
            first_env.step(first_idx)
            if _states_equal(first_env.export_state(), end_state):
                return [first_idx]

            first_step = first_env.get_state()
            legal_second = np.flatnonzero(np.asarray(first_step.mask, dtype=np.bool_))
            for second in legal_second:
                second_idx = int(second)
                if not _is_continuation_action(second_idx):
                    continue
                second_env = first_env.clone()
                second_env.step(second_idx)
                if _states_equal(second_env.export_state(), end_state):
                    return [first_idx, second_idx]

        # Fallback for bridge/native hidden-state drift: match on robust
        # observable state features to recover action intent.
        target_obs = _observable_state_for_inference(end_state)
        observable_candidates: list[list[int]] = []
        for first in legal_first:
            first_idx = int(first)
            first_env = probe.clone()
            first_env.step(first_idx)
            if _observable_state_for_inference(first_env.export_state()) == target_obs:
                observable_candidates.append([first_idx])

            first_step = first_env.get_state()
            legal_second = np.flatnonzero(np.asarray(first_step.mask, dtype=np.bool_))
            for second in legal_second:
                second_idx = int(second)
                if not _is_continuation_action(second_idx):
                    continue
                second_env = first_env.clone()
                second_env.step(second_idx)
                if _observable_state_for_inference(second_env.export_state()) == target_obs:
                    observable_candidates.append([first_idx, second_idx])

        if len(observable_candidates) == 1:
            return observable_candidates[0]
        if len(observable_candidates) > 1:
            observable_candidates.sort(key=lambda seq: (len(seq), seq))
            return observable_candidates[0]
        return None

    def _refresh_move_log_from_snapshot_history_locked(self) -> None:
        if self._snapshot_history_index is None:
            return
        if not self._snapshot_history:
            self._move_log = []
            return

        upto = int(self._snapshot_history_index)
        rebuilt: list[MoveLogEntry] = []

        for idx in range(1, upto + 1):
            start_saved = self._snapshot_history[idx - 1]
            end_saved = self._snapshot_history[idx]
            start_state = start_saved.exported_state
            end_state = end_saved.exported_state

            # Ignore observations where only external metadata changed.
            if _states_equal(start_state, end_state):
                continue

            actor_env = self._ensure_env_locked().clone()
            actor_env.load_state(start_state)
            actor = _seat_str(actor_env.get_state().current_player_id)

            inferred_actions = self._infer_actions_between_snapshots_locked(start_state, end_state)
            if not inferred_actions:
                # Bridge logs can contain intermediate observation snapshots that
                # do not represent a new action (same turn index and same action
                # history length). When inference fails on those, skip them.
                start_hist_len = _spendee_action_history_len(start_state)
                end_hist_len = _spendee_action_history_len(end_state)
                if (
                    int(end_saved.turn_index) == int(start_saved.turn_index)
                    and start_hist_len is not None
                    and end_hist_len is not None
                    and int(end_hist_len) == int(start_hist_len)
                ):
                    continue
                rebuilt.append(
                    MoveLogEntry(
                        turn_index=max(0, int(end_saved.turn_index) - 1),
                        result_turn_index=int(end_saved.turn_index),
                        result_snapshot_index=int(idx),
                        actor=actor,
                        action_idx=-1,
                        label="INFERRED unknown move",
                    )
                )
                continue

            primary = int(inferred_actions[0])
            label = _describe_action(primary)
            for continuation in inferred_actions[1:]:
                label = f"{label} + {_describe_action(int(continuation))}"

            # Return-gem and noble-choice are continuations of the preceding
            # move and should not appear as standalone actions in the log.
            if _is_continuation_action(primary) and rebuilt and rebuilt[-1].actor == actor:
                rebuilt[-1].label = f"{rebuilt[-1].label} + {label}"
                rebuilt[-1].result_turn_index = int(end_saved.turn_index)
                rebuilt[-1].result_snapshot_index = int(idx)
                continue

            rebuilt.append(
                MoveLogEntry(
                    turn_index=max(0, int(end_saved.turn_index) - len(inferred_actions)),
                    result_turn_index=int(end_saved.turn_index),
                    result_snapshot_index=int(idx),
                    actor=actor,
                    action_idx=primary,
                    label=label,
                )
            )

        self._move_log = rebuilt

    def _apply_saved_snapshot_locked(
        self,
        saved: SavedStateDTO,
        *,
        game_id: str,
        config: GameConfigDTO,
        checkpoint_path: Path | None = None,
    ) -> None:
        resolved_checkpoint = checkpoint_path if checkpoint_path is not None else self._resolve_saved_checkpoint(config)
        env = self._ensure_env_locked()
        env.load_state(saved.exported_state)

        self._game_id = str(game_id or uuid.uuid4())
        self._config = GameConfig(
            checkpoint_id=config.checkpoint_id,
            checkpoint_path=resolved_checkpoint,
            num_simulations=int(config.num_simulations),
            player_seat=str(config.player_seat),
            seed=int(config.seed),
            manual_reveal_mode=bool(config.manual_reveal_mode),
            analysis_mode=bool(config.analysis_mode),
        )
        self._turn_index = int(saved.turn_index)
        self._move_log = []
        self._setup_event_log = []
        self._event_log = []
        self._redo_log = []
        self._pending_reveals = []
        self._forced_winner = None
        self._rng = random.Random(int(config.seed))
        self._determinization_seed = random.randint(0, 2**31 - 1)

    def new_game(self, req: NewGameRequest) -> GameSnapshotDTO:
        with self._lock:
            checkpoint_path = self._resolve_checkpoint(req.checkpoint_id)
            self._cancel_active_job_locked()
            env = self._ensure_env_locked()

            seed = int(req.seed) if req.seed is not None else random.randint(0, 2**31 - 1)
            env.reset(seed=seed)

            self._game_id = str(uuid.uuid4())
            self._config = GameConfig(
                checkpoint_id=req.checkpoint_id,
                checkpoint_path=checkpoint_path,
                num_simulations=int(req.num_simulations),
                player_seat=str(req.player_seat),
                seed=seed,
                manual_reveal_mode=bool(req.manual_reveal_mode),
                analysis_mode=bool(req.analysis_mode),
            )
            self._turn_index = 0
            self._move_log = []
            self._rng = random.Random(seed)
            self._determinization_seed = random.randint(0, 2**31 - 1)
            self._forced_winner = None
            self._pending_reveals = _initial_setup_pending_reveals() if bool(req.manual_reveal_mode) else []
            self._setup_event_log = []
            self._event_log = []
            self._redo_log = []
            self._loaded_snapshot_history = []
            self._clear_snapshot_history_locked()
            return self._snapshot_locked()

    def _snapshot_locked(self) -> GameSnapshotDTO:
        env, config, game_id = self._require_game_locked()
        step = env.get_state()
        hidden_deck_card_ids_by_tier = env.hidden_deck_card_ids_by_tier()
        hidden_faceup_reveal_candidates = env.hidden_faceup_reveal_candidates()
        hidden_reserved_reveal_candidates = env.hidden_reserved_reveal_candidates()

        winner = int(step.winner)
        status = "IN_PROGRESS"
        legal_actions = _mask_to_actions(step.mask) if not step.is_terminal else []

        if self._forced_winner is not None:
            winner = int(self._forced_winner)
            status = "RESIGNED"
            legal_actions = []
        elif step.is_terminal:
            status = "COMPLETED"

        return GameSnapshotDTO(
            game_id=game_id,
            status=status,
            player_to_move=_seat_str(step.current_player_id),
            legal_actions=legal_actions,
            legal_action_details=[ActionInfoDTO(action_idx=a, label=_describe_action(a)) for a in legal_actions],
            winner=winner,
            turn_index=self._turn_index,
            current_snapshot_index=(None if self._snapshot_history_index is None else int(self._snapshot_history_index)),
            move_log=[
                MoveLogEntryDTO(
                    turn_index=m.turn_index,
                    result_turn_index=m.result_turn_index,
                    result_snapshot_index=m.result_snapshot_index,
                    actor=_seat_str(0 if m.actor == "P0" else 1),
                    action_idx=m.action_idx,
                    label=m.label,
                )
                for m in self._move_log
            ],
            config=GameConfigDTO(
                checkpoint_id=config.checkpoint_id,
                checkpoint_path=str(config.checkpoint_path),
                num_simulations=config.num_simulations,
                player_seat="P0" if config.player_seat == "P0" else "P1",
                seed=config.seed,
                manual_reveal_mode=config.manual_reveal_mode,
                analysis_mode=config.analysis_mode,
            ),
            board_state=_decode_board_state(
                step,
                turn_index=self._turn_index,
                player_seat=config.player_seat,
                pending_reveals=self._pending_reveals,
                hidden_deck_card_ids_by_tier=hidden_deck_card_ids_by_tier,
            ),
            pending_reveals=[
                PendingRevealDTO(
                    zone=item.zone,
                    tier=item.tier,
                    slot=item.slot,
                    reason=item.reason,
                    actor=item.actor,
                    action_idx=item.action_idx,
                )
                for item in self._pending_reveals
            ],
            hidden_deck_card_ids_by_tier=hidden_deck_card_ids_by_tier,
            hidden_faceup_reveal_candidates=hidden_faceup_reveal_candidates,
            hidden_reserved_reveal_candidates=hidden_reserved_reveal_candidates,
            can_undo=(
                self._snapshot_history_index is not None and self._snapshot_history_index > 0
            ) or bool(self._event_log),
            can_redo=(
                self._snapshot_history_index is not None and self._snapshot_history_index < len(self._snapshot_history) - 1
            ) or bool(self._redo_log),
            determinization_seed=self._determinization_seed,
        )

    def get_state(self) -> GameSnapshotDTO:
        with self._lock:
            return self._snapshot_locked()

    def save_game(self) -> SavedGameDTO:
        with self._lock:
            env, config, game_id = self._require_game_locked()
            snapshots = list(self._snapshot_history)
            current_index = int(self._snapshot_history_index) if self._snapshot_history_index is not None else 0
            if not snapshots:
                snapshots = [SavedStateDTO(turn_index=int(self._turn_index), exported_state=env.export_state())]
            return SavedGameDTO(
                saved_at=datetime.now(timezone.utc).isoformat(),
                game_id=game_id,
                config=GameConfigDTO(
                    checkpoint_id=config.checkpoint_id,
                    checkpoint_path=str(config.checkpoint_path),
                    num_simulations=config.num_simulations,
                    player_seat="P0" if config.player_seat == "P0" else "P1",
                    seed=config.seed,
                    manual_reveal_mode=config.manual_reveal_mode,
                    analysis_mode=config.analysis_mode,
                ),
                snapshots=snapshots,
                current_index=current_index,
            )

    def load_game(self, saved: SavedGameDTO) -> GameSnapshotDTO:
        with self._lock:
            self._cancel_active_job_locked()
            self._clear_snapshot_history_locked()
            if not saved.snapshots:
                raise HTTPException(status_code=400, detail="Saved game has no snapshots")
            if saved.current_index < 0 or saved.current_index >= len(saved.snapshots):
                raise HTTPException(status_code=400, detail="Saved game current_index is out of bounds")
            checkpoint_path = self._resolve_saved_checkpoint(saved.config)
            self._snapshot_history = list(saved.snapshots)
            self._loaded_snapshot_history = list(saved.snapshots)
            self._snapshot_history_index = int(saved.current_index)
            self._apply_saved_snapshot_locked(
                self._snapshot_history[self._snapshot_history_index],
                game_id=saved.game_id,
                config=saved.config,
                checkpoint_path=checkpoint_path,
            )
            self._refresh_move_log_from_snapshot_history_locked()
            return self._snapshot_locked()

    def load_live_game(self, saved: LiveSavedGameDTO) -> GameSnapshotDTO:
        return self.load_game(saved)

    def _is_player_turn_locked(self, step: StepState, config: GameConfig) -> bool:
        return _seat_str(step.current_player_id) == config.player_seat

    def _ensure_no_pending_reveals_locked(self) -> None:
        pending = next((item for item in self._pending_reveals if _is_blocking_pending_reveal(item)), None)
        if pending is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Pending manual reveal for tier {pending.tier} slot {pending.slot}",
            )

    def _has_initial_setup_pending_locked(self) -> bool:
        return any(item.reason in ("initial_setup", "initial_noble_setup") for item in self._pending_reveals)

    def _append_move_locked(self, actor: str, action_idx: int, step_after: StepState) -> None:
        action_idx = int(action_idx)
        label = _describe_action(action_idx)
        result_turn_index = self._turn_index + 1
        if _is_continuation_action(action_idx) and self._move_log and self._move_log[-1].actor == actor:
            prior = self._move_log[-1]
            prior.label = f"{prior.label} + {label}"
            prior.result_turn_index = result_turn_index
        else:
            self._move_log.append(
                MoveLogEntry(
                    turn_index=self._turn_index,
                    result_turn_index=result_turn_index,
                    result_snapshot_index=result_turn_index,
                    actor=actor,
                    action_idx=action_idx,
                    label=label,
                )
            )
        self._turn_index += 1
        if self._config is not None and self._config.manual_reveal_mode:
            pending = _manual_reveal_for_action(int(action_idx), actor, step_after)
            if pending is not None:
                self._pending_reveals.append(pending)

    def _record_event_locked(self, event: GameEvent) -> None:
        self._clear_snapshot_history_locked()
        if event.kind in ("reveal_card", "reveal_reserved_card", "reveal_noble") and not self._move_log:
            self._setup_event_log.append(event)
            self._redo_log = []
            return
        self._event_log.append(event)
        self._redo_log = []

    def _apply_event_locked(self, env: SplendorNativeEnv, event: GameEvent) -> None:
        if event.kind == "move":
            if event.action_idx is None or event.actor is None:
                raise RuntimeError("Corrupt move event")
            step = env.get_state()
            actor = _seat_str(step.current_player_id)
            if actor != event.actor:
                raise RuntimeError("Replay actor mismatch")
            step_after = env.step(int(event.action_idx))
            self._append_move_locked(actor, int(event.action_idx), step_after)
            return
        if event.kind == "reveal_card":
            if event.tier is None or event.slot is None or event.card_id is None:
                raise RuntimeError("Corrupt reveal_card event")
            allow_setup_edit = self._has_initial_setup_pending_locked()
            pending_index = next(
                (
                    idx
                    for idx, item in enumerate(self._pending_reveals)
                    if item.zone == "faceup_card" and item.tier == event.tier and item.slot == event.slot
                ),
                None,
            )
            if allow_setup_edit or pending_index is None:
                env.set_faceup_card_any(event.tier - 1, event.slot, event.card_id)
            else:
                env.set_faceup_card(event.tier - 1, event.slot, event.card_id)
            if pending_index is not None:
                self._pending_reveals.pop(pending_index)
            return
        if event.kind == "reveal_reserved_card":
            if event.actor is None or event.slot is None or event.card_id is None:
                raise RuntimeError("Corrupt reveal_reserved_card event")
            env.set_reserved_card(0 if event.actor == "P0" else 1, event.slot, event.card_id)
            pending_index = next(
                (
                    idx
                    for idx, item in enumerate(self._pending_reveals)
                    if item.zone == "reserved_card" and item.actor == event.actor and item.slot == event.slot
                ),
                None,
            )
            if pending_index is not None:
                self._pending_reveals.pop(pending_index)
            return
        if event.kind == "reveal_noble":
            if event.slot is None or event.noble_id is None:
                raise RuntimeError("Corrupt reveal_noble event")
            allow_setup_edit = self._has_initial_setup_pending_locked()
            if allow_setup_edit:
                env.set_noble_any(event.slot, event.noble_id)
            else:
                env.set_noble(event.slot, event.noble_id)
            pending_index = next(
                (
                    idx
                    for idx, item in enumerate(self._pending_reveals)
                    if item.zone == "noble" and item.slot == event.slot
                ),
                None,
            )
            if pending_index is not None:
                self._pending_reveals.pop(pending_index)
            return
        if event.kind == "resign":
            if self._config is None:
                raise RuntimeError("Missing config during resign replay")
            self._forced_winner = 1 if self._config.player_seat == "P0" else 0
            return
        raise RuntimeError(f"Unknown event kind: {event.kind}")

    def _rebuild_from_events_locked(self, events: list[GameEvent]) -> None:
        env = self._ensure_env_locked()
        if self._config is None:
            raise HTTPException(status_code=400, detail="No active game")
        env.reset(seed=self._config.seed)
        self._turn_index = 0
        self._move_log = []
        self._rng = random.Random(self._config.seed)
        self._forced_winner = None
        self._pending_reveals = _initial_setup_pending_reveals() if self._config.manual_reveal_mode else []
        self._event_log = []
        for event in self._setup_event_log:
            self._apply_event_locked(env, event)
        for event in events:
            self._apply_event_locked(env, event)
            self._event_log.append(event)

    def player_move(self, req: PlayerMoveRequest) -> PlayerMoveResponse:
        with self._lock:
            env, config, _ = self._require_game_locked()
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")
            self._ensure_no_pending_reveals_locked()

            step = env.get_state()
            if step.is_terminal:
                raise HTTPException(status_code=400, detail="Game already finished")
            if not config.analysis_mode and not self._is_player_turn_locked(step, config):
                raise HTTPException(status_code=400, detail="Not player's turn")
            if req.action_idx < 0 or req.action_idx >= int(step.mask.shape[0]):
                raise HTTPException(status_code=400, detail="action_idx out of bounds")
            if not bool(step.mask[int(req.action_idx)]):
                raise HTTPException(status_code=400, detail="Action is not legal")

            actor = _seat_str(step.current_player_id)
            step_after = env.step(int(req.action_idx))
            self._append_move_locked(actor, int(req.action_idx), step_after)
            self._record_event_locked(GameEvent(kind="move", actor=actor, action_idx=int(req.action_idx)))
            snapshot = self._snapshot_locked()
            engine_should_move = (
                snapshot.status == "IN_PROGRESS"
                and not config.analysis_mode
                and snapshot.player_to_move != config.player_seat
                and not any(_is_blocking_pending_reveal(item) for item in self._pending_reveals)
            )
            return PlayerMoveResponse(snapshot=snapshot, engine_should_move=engine_should_move)

    def _get_model_locked(self, config: GameConfig, *, device: str):
        key = (str(config.checkpoint_path), str(device))
        model = self._model_cache.get(key)
        if model is None:
            model = load_checkpoint(config.checkpoint_path, device=device)
            self._model_cache[key] = model
        return model

    def start_engine_think(self, req: EngineThinkRequest | None = None) -> EngineThinkResponse:
        with self._lock:
            env, config, game_id = self._require_game_locked()
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")
            self._ensure_no_pending_reveals_locked()

            step = env.get_state()
            if step.is_terminal:
                raise HTTPException(status_code=400, detail="Game already finished")
            if not config.analysis_mode and self._is_player_turn_locked(step, config):
                raise HTTPException(status_code=400, detail="Engine cannot move on player's turn")

            # Latest request wins: flush any queued/running engine jobs.
            self._cancel_all_engine_jobs_locked()

            num_simulations = int(req.num_simulations) if req is not None and req.num_simulations is not None else config.num_simulations
            search_type = str(req.search_type) if req is not None else "mcts"
            alphabeta_max_nodes = int(req.alphabeta_max_nodes) if req is not None and req.alphabeta_max_nodes is not None else 0
            alphabeta_max_depth = int(req.alphabeta_max_depth) if req is not None and req.alphabeta_max_depth is not None else 0
            alphabeta_max_root_actions = int(req.alphabeta_max_root_actions) if req is not None and req.alphabeta_max_root_actions is not None else 0
            alphabeta_determinization_samples = (
                int(req.alphabeta_determinization_samples)
                if req is not None and req.alphabeta_determinization_samples is not None
                else 32
            )
            continuous_until_cancel = bool(req.continuous_until_cancel) if req is not None else False
            max_total_simulations = (
                int(req.max_total_simulations)
                if req is not None and req.max_total_simulations is not None
                else num_simulations
            )
            if search_type == "alphabeta" and alphabeta_max_nodes == 0 and alphabeta_max_depth == 0 and alphabeta_max_root_actions == 0:
                raise HTTPException(
                    status_code=400,
                    detail="AlphaBeta requires at least one limit (max_nodes, max_depth, or max_root_actions) to keep the server responsive",
                )

            job = EngineJob(
                job_id=str(uuid.uuid4()),
                game_id=game_id,
                status="QUEUED",
                cancel_event=threading.Event(),
                search_type=search_type if search_type in ("mcts", "ismcts", "alphabeta") else "mcts",
            )
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id

            search_device = "cuda" if torch.cuda.is_available() else "cpu"
            model = self._get_model_locked(config, device=search_device)
            search_env = env.clone()
            search_step = search_env.get_state()

            # Use determinization seed for consistent searches
            search_rng = random.Random(self._determinization_seed) if self._determinization_seed is not None else self._rng

            turns_taken = int(self._turn_index)

            def _run() -> int:
                with self._lock:
                    cur_job = self._jobs.get(job.job_id)
                    if cur_job is None:
                        raise RuntimeError("Engine job disappeared")
                    if self._active_job_id != job.job_id:
                        cur_job.cancel_event.set()
                        cur_job.status = "CANCELLED"
                        raise RuntimeError("Engine job superseded")
                    if cur_job.cancel_event.is_set():
                        cur_job.status = "CANCELLED"
                        raise RuntimeError("Engine job cancelled")
                    cur_job.status = "RUNNING"

                try:
                    accumulated_visits = np.zeros_like(search_step.mask, dtype=np.float64)
                    accumulated_weighted_q = np.zeros_like(search_step.mask, dtype=np.float64)
                    latest_q_values = np.zeros_like(search_step.mask, dtype=np.float32)
                    accumulated_root_value = 0.0
                    total_simulations = 0
                    alphabeta_limit_exceeded_handled = False

                    while True:
                        with self._lock:
                            cur_job = self._jobs.get(job.job_id)
                            if cur_job is None:
                                raise RuntimeError("Engine job disappeared")
                            if self._active_job_id != job.job_id:
                                cur_job.cancel_event.set()
                                cur_job.status = "CANCELLED"
                                raise RuntimeError("Engine job superseded")
                            if cur_job.cancel_event.is_set():
                                cur_job.status = "CANCELLED"
                                raise RuntimeError("Engine job cancelled")

                        remaining = max_total_simulations - total_simulations
                        if remaining <= 0:
                            break
                        chunk_simulations = min(num_simulations, remaining)
                        import time
                        start_time = time.time()
                        if search_type == "mcts":
                            result = run_mcts(
                                search_env,
                                model,
                                state=search_step,
                                turns_taken=turns_taken,
                                device=search_device,
                                config=MCTSConfig(
                                    num_simulations=chunk_simulations,
                                    c_puct=1.25,
                                    temperature_moves=0,
                                    temperature=0.0,
                                    root_dirichlet_noise=False,
                                ),
                                rng=search_rng,
                            )
                        elif search_type == "alphabeta":
                            try:
                                result = run_alphabeta(
                                    search_env,
                                    model,
                                    state=search_step,
                                    turns_taken=turns_taken,
                                    device=search_device,
                                    config=AlphaBetaConfig(
                                        max_nodes=alphabeta_max_nodes,
                                        max_depth=alphabeta_max_depth,
                                        max_root_actions=alphabeta_max_root_actions,
                                        determinization_samples=alphabeta_determinization_samples,
                                        fallback_search_type="none",
                                    ),
                                    rng=search_rng,
                                )
                            except RuntimeError as exc:
                                if not str(exc).startswith("ALPHABETA_LIMIT_EXCEEDED:"):
                                    raise
                                immediate_terminal_lines = _collect_immediate_terminal_root_lines(search_env)
                                forced_reply_lines = _collect_forced_opponent_reply_win_line(search_env)
                                forced_lines = [line for line in immediate_terminal_lines if line.winner >= 0 and abs(float(line.value)) >= 0.999]
                                if not forced_lines:
                                    forced_lines = [
                                        line
                                        for line in forced_reply_lines
                                        if line.winner >= 0 and abs(float(line.value)) >= 0.999
                                    ]
                                with self._lock:
                                    cur_job = self._jobs.get(job.job_id)
                                    if cur_job is None:
                                        raise RuntimeError("Engine job disappeared")
                                    if cur_job.cancel_event.is_set():
                                        cur_job.status = "CANCELLED"
                                        raise RuntimeError("Engine job cancelled")
                                    cur_job.search_type = "alphabeta"
                                    if forced_lines:
                                        best_forced = forced_lines[0]
                                        cur_job.action_idx = int(best_forced.root_action_idx)
                                        cur_job.root_value = float(best_forced.value)
                                        cur_job.alphabeta_terminal_lines = forced_lines
                                        cur_job.action_details = []
                                        cur_job.model_action_details = None
                                        cur_job.total_simulations = 0
                                    else:
                                        approx_result = None
                                        try:
                                            fallback_depth = int(alphabeta_max_depth) if int(alphabeta_max_depth) > 0 else 2
                                            fallback_depth = max(1, min(2, fallback_depth))
                                            fallback_nodes = int(alphabeta_max_nodes) if int(alphabeta_max_nodes) > 0 else 50000
                                            fallback_nodes = max(20000, min(100000, fallback_nodes))
                                            approx_result = run_alphabeta(
                                                search_env,
                                                model,
                                                state=search_step,
                                                turns_taken=turns_taken,
                                                device=search_device,
                                                config=AlphaBetaConfig(
                                                    max_nodes=fallback_nodes,
                                                    max_depth=fallback_depth,
                                                    max_root_actions=alphabeta_max_root_actions,
                                                    determinization_samples=1,
                                                    fallback_search_type="none",
                                                ),
                                                rng=search_rng,
                                            )
                                        except Exception:
                                            approx_result = None

                                        if approx_result is not None:
                                            approx_policy = np.asarray(approx_result.visit_probs, dtype=np.float32)
                                            approx_q = np.asarray(approx_result.q_values, dtype=np.float32)
                                            approx_action = _best_legal_action(search_step.mask, approx_policy)
                                            approx_previews = _collect_alphabeta_action_previews(
                                                search_env,
                                                max_depth=int(max(1, fallback_depth)),
                                                max_nodes=int(max(5000, fallback_nodes // 2)),
                                            )
                                            cur_job.action_idx = int(approx_action)
                                            cur_job.root_value = float(approx_result.root_best_value)
                                            cur_job.action_details = _action_viz_rows(
                                                search_step.mask,
                                                approx_policy,
                                                int(approx_action),
                                                approx_q,
                                                approx_previews,
                                            )
                                            cur_job.total_simulations = int(max(getattr(approx_result, "search_slots_evaluated", 0), 1))
                                        else:
                                            # Last-resort fallback: rank legal moves with model policy and root value.
                                            model_eval = _evaluate_model_replay_state(
                                                {"checkpoint_path": str(config.checkpoint_path)},
                                                search_step.state,
                                                search_step.mask,
                                                0,
                                            )
                                            if model_eval is not None:
                                                model_policy, model_value = model_eval
                                                approx_action = _best_legal_action(search_step.mask, model_policy)
                                                cur_job.action_idx = int(approx_action)
                                                cur_job.root_value = float(model_value)
                                                cur_job.action_details = _action_viz_rows(
                                                    search_step.mask,
                                                    model_policy,
                                                    int(approx_action),
                                                )
                                                cur_job.model_action_details = _action_viz_rows(
                                                    search_step.mask,
                                                    model_policy,
                                                    int(approx_action),
                                                )
                                                cur_job.total_simulations = 0
                                            else:
                                                cur_job.action_idx = -1
                                                cur_job.root_value = None
                                                cur_job.action_details = []
                                                cur_job.model_action_details = None
                                                cur_job.total_simulations = 0
                                        cur_job.alphabeta_terminal_lines = []
                                alphabeta_limit_exceeded_handled = True
                                break
                        else:
                            result = run_ismcts(
                                search_env,
                                model,
                                state=search_step,
                                turns_taken=turns_taken,
                                device=search_device,
                                config=ISMCTSConfig(
                                    num_simulations=chunk_simulations,
                                    c_puct=1.25,
                                    eval_batch_size=1,
                                ),
                                rng=search_rng,
                            )
                        elapsed = time.time() - start_time
                        print(f"[{search_type.upper()}] Time: {elapsed:.3f}s | Budget: {chunk_simulations} | "
                              f"Requested: {result.search_slots_requested} | "
                              f"Evaluated: {result.search_slots_evaluated} | "
                              f"Dropped (pending): {result.search_slots_drop_pending_eval} | "
                              f"Dropped (no action): {result.search_slots_drop_no_action}", flush=True)

                        if alphabeta_limit_exceeded_handled:
                            break

                        weight = float(chunk_simulations if search_type != "alphabeta" else max(result.search_slots_evaluated, 1))
                        estimated_visits = np.asarray(result.visit_probs, dtype=np.float64) * weight
                        accumulated_visits += estimated_visits
                        accumulated_weighted_q += np.asarray(result.q_values, dtype=np.float64) * estimated_visits
                        latest_q_values = np.asarray(result.q_values, dtype=np.float32)
                        total_simulations += int(weight)
                        accumulated_root_value += float(result.root_best_value) * weight

                        aggregated_policy = accumulated_visits.astype(np.float32, copy=False)
                        policy_sum = float(aggregated_policy.sum())
                        if policy_sum > 0.0:
                            aggregated_policy = aggregated_policy / policy_sum
                        aggregated_q_values = latest_q_values.copy()
                        visited = accumulated_visits > 0.0
                        aggregated_q_values[visited] = (
                            accumulated_weighted_q[visited] / accumulated_visits[visited]
                        ).astype(np.float32, copy=False)
                        aggregated_action_idx = _best_legal_action(search_step.mask, aggregated_policy)

                        if search_type == "alphabeta":
                            aggregated_root = accumulated_root_value / float(total_simulations)
                            line_depth = int(alphabeta_max_depth) if int(alphabeta_max_depth) > 0 else 6
                            line_nodes = 50000
                            if int(alphabeta_max_nodes) > 0:
                                line_nodes = max(50000, min(250000, int(alphabeta_max_nodes)))
                            preview_depth = max(1, min(line_depth, 4))
                            preview_nodes = max(5000, min(line_nodes // 2, 40000))
                            action_previews = _collect_alphabeta_action_previews(
                                search_env,
                                max_depth=preview_depth,
                                max_nodes=preview_nodes,
                            )

                            immediate_terminal_lines = _collect_immediate_terminal_root_lines(search_env)
                            forced_reply_lines = _collect_forced_opponent_reply_win_line(search_env)
                            terminal_lines = _collect_alphabeta_minimax_line(
                                search_env,
                                max_depth=line_depth,
                                max_nodes=line_nodes,
                            )

                            # If root value is decisive, prioritize proven forced lines so UI stays consistent.
                            preferred_lines: list[AlphaBetaTerminalLineDTO] = []
                            if float(aggregated_root) >= 0.999:
                                preferred_lines = [
                                    line
                                    for line in (immediate_terminal_lines + forced_reply_lines + terminal_lines)
                                    if float(line.value) >= 0.999 and int(line.winner) >= 0
                                ]
                            elif float(aggregated_root) <= -0.999:
                                preferred_lines = [
                                    line
                                    for line in (immediate_terminal_lines + forced_reply_lines + terminal_lines)
                                    if float(line.value) <= -0.999 and int(line.winner) >= 0
                                ]
                            if preferred_lines:
                                terminal_lines = preferred_lines
                            elif int(alphabeta_determinization_samples) > 1:
                                # Single deterministic PV is not representative when root value
                                # is averaged across multiple hidden-info determinizations.
                                terminal_lines = []

                            with self._lock:
                                cur_job = self._jobs.get(job.job_id)
                                if cur_job is None:
                                    raise RuntimeError("Engine job disappeared")
                                if cur_job.cancel_event.is_set():
                                    cur_job.status = "CANCELLED"
                                    raise RuntimeError("Engine job cancelled")
                                cur_job.action_idx = int(aggregated_action_idx)
                                cur_job.action_details = _action_viz_rows(
                                    search_step.mask,
                                    aggregated_policy,
                                    int(aggregated_action_idx),
                                    aggregated_q_values,
                                    action_previews,
                                )
                                cur_job.root_value = float(aggregated_root)
                                cur_job.total_simulations = int(total_simulations)
                                cur_job.search_type = "alphabeta"
                                cur_job.alphabeta_terminal_lines = terminal_lines
                                model_eval = _evaluate_model_replay_state(
                                    {"checkpoint_path": str(config.checkpoint_path)},
                                    search_step.state,
                                    search_step.mask,
                                    int(aggregated_action_idx),
                                )
                                if model_eval is not None:
                                    model_policy, _ = model_eval
                                    cur_job.model_action_details = _action_viz_rows(
                                        search_step.mask,
                                        model_policy,
                                        int(aggregated_action_idx),
                                    )
                            break
                        aggregated_root = accumulated_root_value / float(total_simulations)

                        with self._lock:
                            cur_job = self._jobs.get(job.job_id)
                            if cur_job is None:
                                raise RuntimeError("Engine job disappeared")
                            if cur_job.cancel_event.is_set():
                                cur_job.status = "CANCELLED"
                                raise RuntimeError("Engine job cancelled")
                            cur_job.action_idx = int(aggregated_action_idx)
                            cur_job.action_details = _action_viz_rows(
                                search_step.mask,
                                aggregated_policy,
                                int(aggregated_action_idx),
                                aggregated_q_values,
                            )
                            cur_job.root_value = float(aggregated_root)
                            cur_job.total_simulations = int(total_simulations)
                            cur_job.search_type = search_type if search_type in ("mcts", "ismcts", "alphabeta") else "mcts"
                            cur_job.alphabeta_terminal_lines = []
                            model_eval = _evaluate_model_replay_state(
                                {"checkpoint_path": str(config.checkpoint_path)},
                                search_step.state,
                                search_step.mask,
                                int(aggregated_action_idx),
                            )
                            if model_eval is not None:
                                model_policy, _ = model_eval
                                cur_job.model_action_details = _action_viz_rows(search_step.mask, model_policy, int(aggregated_action_idx))

                        if not continuous_until_cancel:
                            break

                    with self._lock:
                        cur_job = self._jobs.get(job.job_id)
                        if cur_job is None:
                            raise RuntimeError("Engine job disappeared")
                        if cur_job.cancel_event.is_set():
                            cur_job.status = "CANCELLED"
                            raise RuntimeError("Engine job cancelled")
                        cur_job.status = "DONE"
                        if self._active_job_id == job.job_id:
                            self._active_job_id = None
                        if cur_job.action_idx is None:
                            raise RuntimeError("Engine search finished without a result")
                        return int(cur_job.action_idx)
                except Exception as exc:
                    with self._lock:
                        cur_job = self._jobs.get(job.job_id)
                        if cur_job is not None and cur_job.status != "CANCELLED":
                            cur_job.status = "FAILED"
                            cur_job.error = str(exc)
                        if self._active_job_id == job.job_id:
                            self._active_job_id = None
                    raise

            job.future = self._executor.submit(_run)
            return EngineThinkResponse(job_id=job.job_id, status="QUEUED")

    def get_engine_job(self, job_id: str) -> EngineJobStatusDTO:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job_id")
            result = (
                EngineResultDTO(
                    action_idx=job.action_idx,
                    search_type=job.search_type,
                    action_details=job.action_details or [],
                    model_action_details=job.model_action_details,
                    root_value=job.root_value,
                    total_simulations=job.total_simulations,
                    alphabeta_terminal_lines=job.alphabeta_terminal_lines or [],
                )
                if job.action_idx is not None
                else None
            )
            return EngineJobStatusDTO(job_id=job.job_id, status=job.status, error=job.error, result=result)

    def apply_engine_move(self, req: EngineApplyRequest) -> GameSnapshotDTO:
        with self._lock:
            env, config, game_id = self._require_game_locked()
            if config.analysis_mode:
                raise HTTPException(status_code=400, detail="Engine moves are disabled in analysis mode")
            job = self._jobs.get(req.job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown job_id")
            if job.game_id != game_id:
                raise HTTPException(status_code=400, detail="Job does not belong to current game")
            if job.status != "DONE" or job.action_idx is None:
                raise HTTPException(status_code=400, detail="Engine job is not ready")
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")
            self._ensure_no_pending_reveals_locked()

            step = env.get_state()
            if step.is_terminal:
                raise HTTPException(status_code=400, detail="Game already finished")
            if self._is_player_turn_locked(step, config):
                raise HTTPException(status_code=400, detail="It is player's turn")
            if not bool(step.mask[job.action_idx]):
                raise HTTPException(status_code=400, detail="Engine produced illegal action")

            actor = _seat_str(step.current_player_id)
            step_after = env.step(int(job.action_idx))
            self._append_move_locked(actor, int(job.action_idx), step_after)
            self._record_event_locked(GameEvent(kind="move", actor=actor, action_idx=int(job.action_idx)))
            return self._snapshot_locked()

    def reveal_faceup_card(self, req: RevealCardRequest) -> RevealCardResponse:
        with self._lock:
            env, config, _ = self._require_game_locked()
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")

            pending_index = next(
                (
                    idx
                    for idx, item in enumerate(self._pending_reveals)
                    if item.zone == "faceup_card" and item.tier == req.tier and item.slot == req.slot
                ),
                None,
            )
            allow_setup_edit = self._has_initial_setup_pending_locked()
            if pending_index is None and not allow_setup_edit:
                env.set_faceup_card_any(req.tier - 1, req.slot, req.card_id)
            elif allow_setup_edit:
                env.set_faceup_card_any(req.tier - 1, req.slot, req.card_id)
            else:
                env.set_faceup_card(req.tier - 1, req.slot, req.card_id)

            if pending_index is not None:
                self._pending_reveals.pop(pending_index)
            self._record_event_locked(GameEvent(kind="reveal_card", tier=req.tier, slot=req.slot, card_id=req.card_id))
            snapshot = self._snapshot_locked()
            engine_should_move = (
                snapshot.status == "IN_PROGRESS"
                and snapshot.player_to_move != config.player_seat
                and not any(_is_blocking_pending_reveal(item) for item in self._pending_reveals)
            )
            return RevealCardResponse(snapshot=snapshot, engine_should_move=engine_should_move)

    def reveal_noble(self, req: RevealNobleRequest) -> RevealCardResponse:
        with self._lock:
            env, config, _ = self._require_game_locked()
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")
            pending_index = next(
                (
                    idx
                    for idx, item in enumerate(self._pending_reveals)
                    if item.zone == "noble" and item.slot == req.slot
                ),
                None,
            )
            allow_setup_edit = self._has_initial_setup_pending_locked()
            env.set_noble_any(req.slot, req.noble_id)
            if pending_index is not None:
                self._pending_reveals.pop(pending_index)
            self._record_event_locked(GameEvent(kind="reveal_noble", slot=req.slot, noble_id=req.noble_id))
            snapshot = self._snapshot_locked()
            engine_should_move = (
                snapshot.status == "IN_PROGRESS"
                and snapshot.player_to_move != config.player_seat
                and not any(_is_blocking_pending_reveal(item) for item in self._pending_reveals)
            )
            return RevealCardResponse(snapshot=snapshot, engine_should_move=engine_should_move)

    def reveal_reserved_card(self, req: RevealReservedCardRequest) -> RevealCardResponse:
        with self._lock:
            env, config, _ = self._require_game_locked()
            if self._forced_winner is not None:
                raise HTTPException(status_code=400, detail="Game already finished")
            pending_index = next(
                (
                    idx
                    for idx, item in enumerate(self._pending_reveals)
                    if item.zone == "reserved_card" and item.actor == req.seat and item.slot == req.slot
                ),
                None,
            )
            env.set_reserved_card(0 if req.seat == "P0" else 1, req.slot, req.card_id)
            if pending_index is not None:
                self._pending_reveals.pop(pending_index)
            self._record_event_locked(GameEvent(kind="reveal_reserved_card", actor=req.seat, slot=req.slot, card_id=req.card_id))
            snapshot = self._snapshot_locked()
            engine_should_move = (
                snapshot.status == "IN_PROGRESS"
                and snapshot.player_to_move != config.player_seat
                and not any(_is_blocking_pending_reveal(item) for item in self._pending_reveals)
            )
            return RevealCardResponse(snapshot=snapshot, engine_should_move=engine_should_move)

    def resign(self) -> GameSnapshotDTO:
        with self._lock:
            _, config, _ = self._require_game_locked()
            if self._forced_winner is not None:
                return self._snapshot_locked()
            self._cancel_active_job_locked()
            self._forced_winner = 1 if config.player_seat == "P0" else 0
            self._record_event_locked(GameEvent(kind="resign"))
            return self._snapshot_locked()

    def undo(self) -> GameSnapshotDTO:
        with self._lock:
            self._require_game_locked()
            if self._snapshot_history_index is not None:
                if self._snapshot_history_index <= 0:
                    raise HTTPException(status_code=400, detail="Nothing to undo")
                self._cancel_active_job_locked()
                self._snapshot_history_index -= 1
                self._apply_saved_snapshot_locked(
                    self._snapshot_history[self._snapshot_history_index],
                    game_id=self._game_id or "",
                    config=GameConfigDTO(
                        checkpoint_id=self._config.checkpoint_id,
                        checkpoint_path=str(self._config.checkpoint_path),
                        num_simulations=self._config.num_simulations,
                        player_seat="P0" if self._config.player_seat == "P0" else "P1",
                        seed=self._config.seed,
                        manual_reveal_mode=self._config.manual_reveal_mode,
                        analysis_mode=self._config.analysis_mode,
                    ),
                )
                self._refresh_move_log_from_snapshot_history_locked()
                return self._snapshot_locked()
            if not self._event_log:
                raise HTTPException(status_code=400, detail="Nothing to undo")
            self._cancel_active_job_locked()
            undone = self._event_log[-1]
            remaining = list(self._event_log[:-1])
            self._rebuild_from_events_locked(remaining)
            self._redo_log.insert(0, undone)
            return self._snapshot_locked()

    def redo(self) -> GameSnapshotDTO:
        with self._lock:
            self._require_game_locked()
            if self._snapshot_history_index is not None:
                if self._snapshot_history_index >= len(self._snapshot_history) - 1:
                    raise HTTPException(status_code=400, detail="Nothing to redo")
                self._cancel_active_job_locked()
                self._snapshot_history_index += 1
                self._apply_saved_snapshot_locked(
                    self._snapshot_history[self._snapshot_history_index],
                    game_id=self._game_id or "",
                    config=GameConfigDTO(
                        checkpoint_id=self._config.checkpoint_id,
                        checkpoint_path=str(self._config.checkpoint_path),
                        num_simulations=self._config.num_simulations,
                        player_seat="P0" if self._config.player_seat == "P0" else "P1",
                        seed=self._config.seed,
                        manual_reveal_mode=self._config.manual_reveal_mode,
                        analysis_mode=self._config.analysis_mode,
                    ),
                )
                self._refresh_move_log_from_snapshot_history_locked()
                return self._snapshot_locked()
            if not self._redo_log:
                raise HTTPException(status_code=400, detail="Nothing to redo")
            self._cancel_active_job_locked()
            restored = self._redo_log.pop(0)
            events = [*self._event_log, restored]
            self._rebuild_from_events_locked(events)
            return self._snapshot_locked()

    def undo_to_start(self) -> GameSnapshotDTO:
        with self._lock:
            self._require_game_locked()
            if self._snapshot_history_index is not None:
                if self._snapshot_history_index <= 0:
                    raise HTTPException(status_code=400, detail="Already at first position")
                self._cancel_active_job_locked()
                self._snapshot_history_index = 0
                self._apply_saved_snapshot_locked(
                    self._snapshot_history[self._snapshot_history_index],
                    game_id=self._game_id or "",
                    config=GameConfigDTO(
                        checkpoint_id=self._config.checkpoint_id,
                        checkpoint_path=str(self._config.checkpoint_path),
                        num_simulations=self._config.num_simulations,
                        player_seat="P0" if self._config.player_seat == "P0" else "P1",
                        seed=self._config.seed,
                        manual_reveal_mode=self._config.manual_reveal_mode,
                        analysis_mode=self._config.analysis_mode,
                    ),
                )
                self._refresh_move_log_from_snapshot_history_locked()
                return self._snapshot_locked()
            if not self._event_log:
                raise HTTPException(status_code=400, detail="Already at first position")
            self._cancel_active_job_locked()
            self._redo_log = [*self._event_log, *self._redo_log]
            self._rebuild_from_events_locked([])
            return self._snapshot_locked()

    def redo_to_end(self) -> GameSnapshotDTO:
        with self._lock:
            self._require_game_locked()
            if self._snapshot_history_index is not None:
                last_index = len(self._snapshot_history) - 1
                if self._snapshot_history_index >= last_index:
                    raise HTTPException(status_code=400, detail="Already at latest position")
                self._cancel_active_job_locked()
                self._snapshot_history_index = last_index
                self._apply_saved_snapshot_locked(
                    self._snapshot_history[self._snapshot_history_index],
                    game_id=self._game_id or "",
                    config=GameConfigDTO(
                        checkpoint_id=self._config.checkpoint_id,
                        checkpoint_path=str(self._config.checkpoint_path),
                        num_simulations=self._config.num_simulations,
                        player_seat="P0" if self._config.player_seat == "P0" else "P1",
                        seed=self._config.seed,
                        manual_reveal_mode=self._config.manual_reveal_mode,
                        analysis_mode=self._config.analysis_mode,
                    ),
                )
                self._refresh_move_log_from_snapshot_history_locked()
                return self._snapshot_locked()
            if not self._redo_log:
                raise HTTPException(status_code=400, detail="Already at latest position")
            self._cancel_active_job_locked()
            events = [*self._event_log, *self._redo_log]
            self._redo_log = []
            self._rebuild_from_events_locked(events)
            return self._snapshot_locked()

    def jump_to_turn(self, req: JumpToTurnRequest) -> GameSnapshotDTO:
        with self._lock:
            self._require_game_locked()
            target_turn = int(req.turn_index)
            self._cancel_active_job_locked()

            if self._snapshot_history_index is not None:
                if not self._snapshot_history:
                    raise HTTPException(status_code=400, detail="No saved snapshots available")
                target_index = 0
                for idx, saved in enumerate(self._snapshot_history):
                    if int(saved.turn_index) <= target_turn:
                        target_index = idx
                    else:
                        break
                self._snapshot_history_index = target_index
                self._apply_saved_snapshot_locked(
                    self._snapshot_history[self._snapshot_history_index],
                    game_id=self._game_id or "",
                    config=GameConfigDTO(
                        checkpoint_id=self._config.checkpoint_id,
                        checkpoint_path=str(self._config.checkpoint_path),
                        num_simulations=self._config.num_simulations,
                        player_seat="P0" if self._config.player_seat == "P0" else "P1",
                        seed=self._config.seed,
                        manual_reveal_mode=self._config.manual_reveal_mode,
                        analysis_mode=self._config.analysis_mode,
                    ),
                )
                self._refresh_move_log_from_snapshot_history_locked()
                return self._snapshot_locked()

            current_turn = int(self._turn_index)
            if target_turn > current_turn:
                raise HTTPException(status_code=400, detail=f"turn_index must be <= current turn ({current_turn})")

            selected_events: list[GameEvent] = []
            moves_applied = 0
            for event in self._event_log:
                if event.kind == "move":
                    if moves_applied >= target_turn:
                        break
                    selected_events.append(event)
                    moves_applied += 1
                    continue
                # Keep post-move reveal/resign events that belong to already-applied moves.
                if moves_applied > 0 and moves_applied <= target_turn:
                    selected_events.append(event)

            if moves_applied != target_turn:
                raise HTTPException(status_code=400, detail="Could not reconstruct requested turn")

            remaining_events = self._event_log[len(selected_events):]
            self._rebuild_from_events_locked(selected_events)
            self._redo_log = list(remaining_events)
            return self._snapshot_locked()

    def jump_to_snapshot(self, req: JumpToSnapshotRequest) -> GameSnapshotDTO:
        with self._lock:
            self._require_game_locked()
            history_source = self._snapshot_history if self._snapshot_history else self._loaded_snapshot_history
            if not history_source:
                raise HTTPException(status_code=400, detail="No saved snapshots available")

            target_index = int(req.snapshot_index)
            if target_index < 0 or target_index >= len(history_source):
                raise HTTPException(status_code=400, detail=f"snapshot_index must be in [0, {len(history_source) - 1}]")

            self._cancel_active_job_locked()
            # Restore active snapshot history from the originally loaded mainline
            # so users can jump back to canonical positions after branching.
            self._snapshot_history = list(history_source)
            self._snapshot_history_index = target_index
            self._apply_saved_snapshot_locked(
                self._snapshot_history[self._snapshot_history_index],
                game_id=self._game_id or "",
                config=GameConfigDTO(
                    checkpoint_id=self._config.checkpoint_id,
                    checkpoint_path=str(self._config.checkpoint_path),
                    num_simulations=self._config.num_simulations,
                    player_seat="P0" if self._config.player_seat == "P0" else "P1",
                    seed=self._config.seed,
                    manual_reveal_mode=self._config.manual_reveal_mode,
                    analysis_mode=self._config.analysis_mode,
                ),
            )
            self._refresh_move_log_from_snapshot_history_locked()
            return self._snapshot_locked()


manager = GameManager()
app = FastAPI(title="Splendor vs MCTS UI API")


@app.get("/api/checkpoints", response_model=list[CheckpointDTO])
def get_checkpoints() -> list[CheckpointDTO]:
    return manager.list_checkpoints()


@app.get("/api/cards", response_model=list[CatalogCardDTO])
def get_cards() -> list[CatalogCardDTO]:
    return manager.list_standard_cards()


@app.get("/api/nobles", response_model=list[CatalogNobleDTO])
def get_nobles() -> list[CatalogNobleDTO]:
    return manager.list_standard_nobles()


@app.post("/api/game/new", response_model=GameSnapshotDTO)
def new_game(req: NewGameRequest) -> GameSnapshotDTO:
    return manager.new_game(req)


@app.get("/api/game/state", response_model=GameSnapshotDTO)
def game_state() -> GameSnapshotDTO:
    return manager.get_state()


@app.get("/api/game/save", response_model=SavedGameDTO)
def game_save() -> SavedGameDTO:
    return manager.save_game()


@app.post("/api/game/load", response_model=GameSnapshotDTO)
def game_load(saved: SavedGameDTO) -> GameSnapshotDTO:
    return manager.load_game(saved)


@app.get("/api/game/live-save/status", response_model=LiveSaveStatusDTO)
def live_save_status() -> LiveSaveStatusDTO:
    path = SPENDEE_LIVE_SAVE_PATH
    if not path.exists():
        return LiveSaveStatusDTO(exists=False, path=str(path.resolve()))
    stat = path.stat()
    return LiveSaveStatusDTO(
        exists=True,
        path=str(path.resolve()),
        updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    )


@app.post("/api/game/live-save/load", response_model=GameSnapshotDTO)
def live_save_load() -> GameSnapshotDTO:
    path = SPENDEE_LIVE_SAVE_PATH
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Live save not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved = LiveSavedGameDTO.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse live save: {exc}") from exc
    return manager.load_live_game(saved)


@app.post("/api/game/player-move", response_model=PlayerMoveResponse)
def player_move(req: PlayerMoveRequest) -> PlayerMoveResponse:
    return manager.player_move(req)


@app.post("/api/game/reveal-card", response_model=RevealCardResponse)
def reveal_card(req: RevealCardRequest) -> RevealCardResponse:
    return manager.reveal_faceup_card(req)


@app.post("/api/game/reveal-reserved-card", response_model=RevealCardResponse)
def reveal_reserved_card(req: RevealReservedCardRequest) -> RevealCardResponse:
    return manager.reveal_reserved_card(req)


@app.post("/api/game/reveal-noble", response_model=RevealCardResponse)
def reveal_noble(req: RevealNobleRequest) -> RevealCardResponse:
    return manager.reveal_noble(req)


@app.post("/api/game/engine-think", response_model=EngineThinkResponse)
def engine_think(req: EngineThinkRequest) -> EngineThinkResponse:
    return manager.start_engine_think(req)


@app.get("/api/game/engine-job/{job_id}", response_model=EngineJobStatusDTO)
def engine_job(job_id: str) -> EngineJobStatusDTO:
    return manager.get_engine_job(job_id)


@app.post("/api/game/engine-apply", response_model=GameSnapshotDTO)
def engine_apply(req: EngineApplyRequest) -> GameSnapshotDTO:
    return manager.apply_engine_move(req)


@app.post("/api/game/resign", response_model=GameSnapshotDTO)
def game_resign() -> GameSnapshotDTO:
    return manager.resign()


@app.post("/api/game/undo", response_model=GameSnapshotDTO)
def game_undo() -> GameSnapshotDTO:
    return manager.undo()


@app.post("/api/game/redo", response_model=GameSnapshotDTO)
def game_redo() -> GameSnapshotDTO:
    return manager.redo()


@app.post("/api/game/undo-to-start", response_model=GameSnapshotDTO)
def game_undo_to_start() -> GameSnapshotDTO:
    return manager.undo_to_start()


@app.post("/api/game/redo-to-end", response_model=GameSnapshotDTO)
def game_redo_to_end() -> GameSnapshotDTO:
    return manager.redo_to_end()


@app.post("/api/game/jump-to-turn", response_model=GameSnapshotDTO)
def game_jump_to_turn(req: JumpToTurnRequest) -> GameSnapshotDTO:
    return manager.jump_to_turn(req)


@app.post("/api/game/jump-to-snapshot", response_model=GameSnapshotDTO)
def game_jump_to_snapshot(req: JumpToSnapshotRequest) -> GameSnapshotDTO:
    return manager.jump_to_snapshot(req)


@app.get("/api/selfplay/sessions", response_model=list[SelfPlaySessionDTO])
def selfplay_sessions() -> list[SelfPlaySessionDTO]:
    rows = list_selfplay_sessions(SELFPLAY_DIR)
    return [SelfPlaySessionDTO(**row) for row in rows]


@app.post("/api/selfplay/run", response_model=SelfPlayRunResponse)
def selfplay_run(req: SelfPlayRunRequest) -> SelfPlayRunResponse:
    checkpoint_path = _resolve_checkpoint_id(req.checkpoint_id)
    seed = int(req.seed) if req.seed is not None else random.randint(1, 2**31 - 1)
    games = int(req.games)
    requested_workers = int(req.workers) if req.workers is not None else None
    auto_workers = int(os.cpu_count() or 1)
    workers_used = min(games, requested_workers if requested_workers is not None else auto_workers)
    workers_used = max(1, int(workers_used))

    try:
        if workers_used > 1:
            session = run_selfplay_session_parallel(
                checkpoint_path=checkpoint_path,
                games=games,
                max_turns=int(req.max_turns),
                num_simulations=int(req.num_simulations),
                seed_base=seed,
                workers=workers_used,
            )
        else:
            model = load_checkpoint(checkpoint_path, device="cpu")
            with SplendorNativeEnv() as env:
                session = run_selfplay_session(
                    env=env,
                    model=model,
                    games=games,
                    max_turns=int(req.max_turns),
                    num_simulations=int(req.num_simulations),
                    seed_base=seed,
                )
            session.metadata["workers_requested"] = int(requested_workers) if requested_workers is not None else None
            session.metadata["workers_used"] = int(workers_used)
            session.metadata["parallelism_mode"] = "process_pool"
            session.metadata["games_per_worker"] = [int(games)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Self-play run failed: {exc}") from exc

    session.metadata["workers_requested"] = int(requested_workers) if requested_workers is not None else None
    session.metadata["workers_used"] = int(workers_used)
    session.metadata["parallelism_mode"] = "process_pool"
    if "games_per_worker" not in session.metadata:
        session.metadata["games_per_worker"] = [int(games)]
    session.metadata["checkpoint_id"] = req.checkpoint_id
    session.metadata["checkpoint_path"] = str(checkpoint_path.resolve())
    out_path = save_session_npz(session, SELFPLAY_DIR)
    return SelfPlayRunResponse(
        session_id=session.session_id,
        path=str(out_path.resolve()),
        games=int(req.games),
        steps=len(session.steps),
        created_at=session.created_at,
    )


@app.get("/api/selfplay/session/{session_id}/summary", response_model=SelfPlaySessionSummaryDTO)
def selfplay_session_summary(session_id: str) -> SelfPlaySessionSummaryDTO:
    session_path = _selfplay_session_path(session_id)
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return _build_selfplay_summary(session_id, session_path)


@app.get("/api/selfplay/session/{session_id}/step", response_model=ReplayStepDTO)
def selfplay_session_step(session_id: str, episode_idx: int, step_idx: int) -> ReplayStepDTO:
    session_path = _selfplay_session_path(session_id)
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return _decode_replay_step(session_id, session_path, episode_idx, step_idx)


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


if WEB_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST_DIR), html=True), name="web")
