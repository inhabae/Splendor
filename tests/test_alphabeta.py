from __future__ import annotations

import sys
import types
import unittest
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from nn.alphabeta import AlphaBetaConfig, run_alphabeta
from nn.mcts import MCTSResult as SharedMCTSResult
from nn.native_env import ACTION_DIM, SplendorNativeEnv, list_standard_cards


def _card_index() -> dict[int, dict]:
    return {int(card["id"]): dict(card) for card in list_standard_cards()}


def _player_payload(cards_by_id: dict[int, dict], *, tokens: dict[str, int], purchased: list[int], reserved: list[int]) -> dict:
    bonuses = {color: 0 for color in ("white", "blue", "green", "red", "black")}
    points = 0
    for card_id in purchased:
        card = cards_by_id[int(card_id)]
        bonuses[str(card["bonus_color"])] += 1
        points += int(card["points"])
    return {
        "tokens": dict(tokens),
        "bonuses": bonuses,
        "points": points,
        "purchased_card_ids": list(purchased),
        "reserved": [{"card_id": int(card_id), "is_public": True} for card_id in reserved],
        "claimed_noble_ids": [],
    }


def _winning_tiebreak_payload(cards_by_id: dict[int, dict]) -> dict:
    return {
        "current_player": 1,
        "move_number": 20,
        "players": [
            _player_payload(
                cards_by_id,
                tokens={"white": 1, "blue": 4, "green": 4, "red": 1, "black": 4, "joker": 5},
                purchased=[73, 78, 86],
                reserved=[2, 10, 18],
            ),
            _player_payload(
                cards_by_id,
                tokens={"white": 3, "blue": 0, "green": 0, "red": 3, "black": 0, "joker": 0},
                purchased=[72, 82, 90],
                reserved=[40, 16, 74],
            ),
        ],
        "faceup_card_ids": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "available_noble_ids": [],
        "bank": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0, "joker": 0},
        "phase_flags": {"is_return_phase": False, "is_noble_choice_phase": False},
    }


def _draw_payload(cards_by_id: dict[int, dict]) -> dict:
    return {
        "current_player": 1,
        "move_number": 20,
        "players": [
            _player_payload(
                cards_by_id,
                tokens={"white": 4, "blue": 4, "green": 4, "red": 4, "black": 2, "joker": 5},
                purchased=[72, 78, 82, 8],
                reserved=[2, 10, 18],
            ),
            _player_payload(
                cards_by_id,
                tokens={"white": 0, "blue": 0, "green": 0, "red": 0, "black": 2, "joker": 0},
                purchased=[73, 74, 86],
                reserved=[32, 41, 90],
            ),
        ],
        "faceup_card_ids": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "available_noble_ids": [],
        "bank": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0, "joker": 0},
        "phase_flags": {"is_return_phase": False, "is_noble_choice_phase": False},
    }


def _loss_payload(cards_by_id: dict[int, dict]) -> dict:
    return {
        "current_player": 1,
        "move_number": 20,
        "players": [
            _player_payload(
                cards_by_id,
                tokens={"white": 4, "blue": 4, "green": 4, "red": 4, "black": 2, "joker": 5},
                purchased=[78, 82, 86],
                reserved=[2, 10, 18],
            ),
            _player_payload(
                cards_by_id,
                tokens={"white": 0, "blue": 0, "green": 0, "red": 0, "black": 2, "joker": 0},
                purchased=[72, 74, 90],
                reserved=[32, 41, 73],
            ),
        ],
        "faceup_card_ids": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "available_noble_ids": [],
        "bank": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0, "joker": 0},
        "phase_flags": {"is_return_phase": False, "is_noble_choice_phase": False},
    }


def _return_phase_payload(cards_by_id: dict[int, dict]) -> dict:
    return {
        "current_player": 0,
        "move_number": 5,
        "players": [
            _player_payload(
                cards_by_id,
                tokens={"white": 4, "blue": 4, "green": 3, "red": 0, "black": 0, "joker": 0},
                purchased=[],
                reserved=[72, 78, 82],
            ),
            _player_payload(
                cards_by_id,
                tokens={"white": 0, "blue": 0, "green": 1, "red": 0, "black": 4, "joker": 5},
                purchased=[],
                reserved=[73, 74, 86],
            ),
        ],
        "faceup_card_ids": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "available_noble_ids": [],
        "bank": {"white": 0, "blue": 0, "green": 0, "red": 4, "black": 0, "joker": 0},
        "phase_flags": {"is_return_phase": True, "is_noble_choice_phase": False},
    }


def _noble_choice_payload(cards_by_id: dict[int, dict]) -> dict:
    return {
        "current_player": 0,
        "move_number": 7,
        "players": [
            _player_payload(
                cards_by_id,
                tokens={"white": 4, "blue": 4, "green": 4, "red": 0, "black": 0, "joker": 5},
                purchased=[1, 2, 3, 4, 33, 34, 35, 36],
                reserved=[],
            ),
            _player_payload(
                cards_by_id,
                tokens={"white": 0, "blue": 0, "green": 0, "red": 4, "black": 4, "joker": 0},
                purchased=[],
                reserved=[],
            ),
        ],
        "faceup_card_ids": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "available_noble_ids": [6],
        "bank": {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0, "joker": 0},
        "phase_flags": {"is_return_phase": False, "is_noble_choice_phase": True},
    }


class AlphaBetaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards_by_id = _card_index()

    def setUp(self) -> None:
        try:
            self.env = SplendorNativeEnv()
        except Exception as exc:
            self.skipTest(f"splendor_native unavailable: {exc}")

    def tearDown(self) -> None:
        if hasattr(self, "env"):
            self.env.close()

    def test_run_alphabeta_prefers_lowest_index_among_equal_wins(self) -> None:
        state = self.env.load_state(_winning_tiebreak_payload(self.cards_by_id))
        result = run_alphabeta(self.env, None, state, turns_taken=20, config=AlphaBetaConfig())
        legal = np.flatnonzero(state.mask)

        self.assertIsInstance(result, SharedMCTSResult)
        self.assertEqual(int(result.chosen_action_idx), 12)
        self.assertEqual(float(result.root_best_value), 1.0)
        self.assertEqual(result.visit_probs.shape, (ACTION_DIM,))
        self.assertEqual(result.q_values.shape, (ACTION_DIM,))
        self.assertAlmostEqual(float(np.sum(result.visit_probs)), 1.0, places=6)
        self.assertEqual(float(result.visit_probs[12]), 1.0)
        self.assertEqual(float(result.q_values[12]), 1.0)
        self.assertEqual(float(result.q_values[13]), 1.0)
        self.assertTrue(np.all(result.q_values[np.setdiff1d(np.arange(ACTION_DIM), legal)] == 0.0))

    def test_run_alphabeta_reports_exact_draw(self) -> None:
        state = self.env.load_state(_draw_payload(self.cards_by_id))
        result = run_alphabeta(self.env, None, state, turns_taken=20, config=AlphaBetaConfig())

        self.assertEqual(int(result.chosen_action_idx), 12)
        self.assertEqual(float(result.root_best_value), 0.0)
        self.assertEqual(float(result.q_values[12]), 0.0)

    def test_run_alphabeta_reports_exact_loss(self) -> None:
        state = self.env.load_state(_loss_payload(self.cards_by_id))
        result = run_alphabeta(self.env, None, state, turns_taken=20, config=AlphaBetaConfig())

        self.assertEqual(int(result.chosen_action_idx), 12)
        self.assertEqual(float(result.root_best_value), -1.0)
        self.assertEqual(float(result.q_values[12]), -1.0)

    def test_phase_states_preserve_expected_legal_actions(self) -> None:
        return_state = self.env.load_state(_return_phase_payload(self.cards_by_id))
        return_legal = np.flatnonzero(return_state.mask).tolist()
        self.assertEqual(return_legal, [61, 62, 63])

        noble_state = self.env.load_state(_noble_choice_payload(self.cards_by_id))
        noble_legal = np.flatnonzero(noble_state.mask).tolist()
        self.assertEqual(noble_legal, [66])

    def test_run_alphabeta_falls_back_to_mcts_when_limits_hit(self) -> None:
        state = self.env.load_state(_winning_tiebreak_payload(self.cards_by_id))
        seen: dict[str, object] = {}

        def _raise_limit(**_: object) -> object:
            raise RuntimeError("ALPHABETA_LIMIT_EXCEEDED: max_nodes")

        self.env.run_alphabeta_native = _raise_limit  # type: ignore[method-assign]

        fake_mcts = types.ModuleType("nn.mcts")

        class _DummyMCTSConfig:
            def __init__(self) -> None:
                self.num_simulations = 64

        def _run_mcts(env, model, state, *, turns_taken, device="cpu", config=None, rng=None):
            del env, model, turns_taken, device, config
            seen["rng"] = rng
            visit_probs = np.zeros((ACTION_DIM,), dtype=np.float32)
            visit_probs[13] = 1.0
            q_values = np.zeros((ACTION_DIM,), dtype=np.float32)
            q_values[13] = 0.25
            return SimpleNamespace(
                chosen_action_idx=13,
                visit_probs=visit_probs,
                q_values=q_values,
                root_best_value=0.25,
                search_slots_requested=7,
                search_slots_evaluated=7,
                search_slots_drop_pending_eval=0,
                search_slots_drop_no_action=0,
            )

        fake_mcts.MCTSConfig = _DummyMCTSConfig
        fake_mcts.run_mcts = _run_mcts

        seeded_rng = random.Random(123)
        with patch.dict(sys.modules, {"nn.mcts": fake_mcts}):
            result = run_alphabeta(
                self.env,
                object(),
                state,
                turns_taken=20,
                config=AlphaBetaConfig(max_nodes=1, fallback_search_type="mcts"),
                rng=seeded_rng,
            )

        self.assertEqual(int(result.chosen_action_idx), 13)
        self.assertEqual(float(result.root_best_value), 0.25)
        self.assertEqual(float(result.visit_probs[13]), 1.0)
        self.assertIs(seen.get("rng"), seeded_rng)

    def test_run_alphabeta_rejects_hidden_info_without_determinization(self) -> None:
        state = self.env.reset(seed=5)

        with self.assertRaisesRegex(ValueError, "determinize_root_hidden_info=True"):
            run_alphabeta(
                self.env,
                None,
                state,
                turns_taken=0,
                config=AlphaBetaConfig(determinize_root_hidden_info=False),
            )


if __name__ == "__main__":
    unittest.main()
