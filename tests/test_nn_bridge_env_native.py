from __future__ import annotations

import random
import unittest

import numpy as np

from nn.native_env import SplendorNativeEnv, StepState
from nn.state_schema import ACTION_DIM, STATE_DIM
from tests.codec_reference import encode_state


class TestSplendorNativeEnv(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.env = SplendorNativeEnv()
        except Exception as exc:
            self.skipTest(f"splendor_native unavailable: {exc}")

    def tearDown(self) -> None:
        if hasattr(self, "env"):
            self.env.close()

    def _first_legal_action(self, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        self.assertGreater(len(legal), 0, "Expected at least one legal action")
        return int(legal[0])

    def _random_legal_action(self, mask: np.ndarray, rng: random.Random) -> int:
        legal = np.flatnonzero(mask)
        self.assertGreater(len(legal), 0, "Expected at least one legal action")
        return int(rng.choice(legal.tolist()))

    def _assert_step_state_contract(self, step: StepState) -> None:
        self.assertEqual(step.state.shape, (STATE_DIM,))
        self.assertEqual(step.mask.shape, (ACTION_DIM,))
        self.assertEqual(step.state.dtype, np.float32)
        self.assertEqual(step.mask.dtype, np.bool_)
        self.assertTrue(np.isfinite(step.state).all())
        self.assertTrue(np.isin(step.mask, [False, True]).all())
        self.assertIsInstance(step.is_terminal, bool)
        self.assertIsInstance(step.current_player_id, int)

        if not step.is_terminal:
            self.assertEqual(int(step.winner), -2)
            self.assertTrue(bool(step.mask.any()), "Non-terminal state should have at least one legal action")
        else:
            self.assertIn(int(step.winner), (-1, 0, 1))

    def _collect_rollout(self, seed: int, max_steps: int, *, rng_seed: int | None = None) -> list[StepState]:
        state = self.env.reset(seed=seed)
        out = [state]
        rng = random.Random(seed if rng_seed is None else rng_seed)
        for _ in range(max_steps):
            self._assert_step_state_contract(state)
            if state.is_terminal:
                break
            action = self._random_legal_action(state.mask, rng)
            self.assertTrue(bool(state.mask[action]))
            state = self.env.step(action)
            out.append(state)
        return out

    def _assert_raises_msg(self, exc_type: type[BaseException], pattern: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(exc_type) as ctx:
            fn(*args, **kwargs)
        self.assertIn(pattern.lower(), str(ctx.exception).lower())

    def test_reset_shapes_and_dtypes(self) -> None:
        state = self.env.reset(seed=123)
        self._assert_step_state_contract(state)
        self.assertEqual(state.current_player_id, self.env.current_player_id)

    def test_get_state_before_reset_raises(self) -> None:
        self._assert_raises_msg(RuntimeError, "reset", self.env.get_state)

    def test_step_before_reset_raises(self) -> None:
        self._assert_raises_msg(RuntimeError, "reset", self.env.step, 0)

    def test_step_invalid_action_index_negative_raises(self) -> None:
        self.env.reset(seed=123)
        with self.assertRaises(Exception):
            self.env.step(-1)

    def test_step_invalid_action_index_too_large_raises(self) -> None:
        self.env.reset(seed=123)
        with self.assertRaises(Exception):
            self.env.step(ACTION_DIM)

    def test_step_illegal_in_range_action_raises(self) -> None:
        state = self.env.reset(seed=123)
        illegal = np.flatnonzero(~state.mask)
        if illegal.size == 0:
            self.skipTest("No illegal action available in this state")
        with self.assertRaises(Exception) as ctx:
            self.env.step(int(illegal[0]))
        msg = str(ctx.exception).lower()
        self.assertTrue(("valid" in msg) or ("action" in msg))

    def test_cpp_encoder_matches_python_codec(self) -> None:
        native_step = self.env.reset(seed=123)
        raw = np.asarray(self.env.debug_raw_state(), dtype=np.int32)
        self.assertEqual(raw.shape, (STATE_DIM,))
        py_encoded = encode_state(raw.tolist())
        np.testing.assert_allclose(native_step.state, py_encoded, rtol=0.0, atol=0.0)

    def test_raw_state_shape_and_dtype_from_debug_raw_state(self) -> None:
        self.env.reset(seed=456)
        raw = np.asarray(self.env.debug_raw_state())
        self.assertEqual(raw.shape, (STATE_DIM,))
        self.assertTrue(np.issubdtype(raw.dtype, np.integer))
        self.assertTrue(np.isfinite(raw.astype(np.float64)).all())

    def test_random_rollout_contract_many_seeds(self) -> None:
        for seed in [1, 2, 3, 7, 13, 123]:
            states = self._collect_rollout(seed=seed, max_steps=200)
            self.assertGreaterEqual(len(states), 1)
            for step in states:
                self._assert_step_state_contract(step)

    def test_terminal_winner_semantics_on_rollout(self) -> None:
        saw_terminal = False
        for seed in [1, 2, 3, 7, 13, 123]:
            states = self._collect_rollout(seed=seed, max_steps=400)
            for step in states:
                if step.is_terminal:
                    saw_terminal = True
                    self.assertIn(int(step.winner), (-1, 0, 1))
                else:
                    self.assertEqual(int(step.winner), -2)
        self.assertTrue(saw_terminal, "Expected at least one terminal state across rollout seeds")

    def test_current_player_id_updates_across_steps(self) -> None:
        state = self.env.reset(seed=123)
        self.assertEqual(state.current_player_id, self.env.current_player_id)
        for _ in range(10):
            self._assert_step_state_contract(state)
            if state.is_terminal:
                break
            prev = self.env.current_player_id
            action = self._first_legal_action(state.mask)
            state = self.env.step(action)
            self.assertEqual(state.current_player_id, self.env.current_player_id)
            self.assertIn(self.env.current_player_id, (0, 1))
            self.assertIn(prev, (0, 1))

    def test_cpp_encoder_matches_python_codec_across_random_rollout_states(self) -> None:
        for seed in [3, 7, 11]:
            state = self.env.reset(seed=seed)
            rng = random.Random(seed)
            for _ in range(50):
                self._assert_step_state_contract(state)
                raw = np.asarray(self.env.debug_raw_state(), dtype=np.int32)
                self.assertEqual(raw.shape, (STATE_DIM,))
                py_encoded = encode_state(raw.tolist())
                np.testing.assert_allclose(state.state, py_encoded, rtol=0.0, atol=0.0)

                if state.is_terminal:
                    break
                action = self._random_legal_action(state.mask, rng)
                state = self.env.step(action)


if __name__ == "__main__":
    unittest.main()
