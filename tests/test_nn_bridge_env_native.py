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
        self.assertIsInstance(step.is_return_phase, bool)
        self.assertIsInstance(step.is_noble_choice_phase, bool)
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

    def test_snapshot_before_reset_raises(self) -> None:
        self._assert_raises_msg(RuntimeError, "reset", self.env.snapshot)

    def test_restore_snapshot_before_reset_raises(self) -> None:
        self._assert_raises_msg(RuntimeError, "reset", self.env.restore_snapshot, 1)

    def test_drop_snapshot_before_reset_raises(self) -> None:
        self._assert_raises_msg(RuntimeError, "reset", self.env.drop_snapshot, 1)

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

    def test_snapshot_restore_round_trip(self) -> None:
        state0 = self.env.reset(seed=7)
        snap = self.env.snapshot()
        action = self._first_legal_action(state0.mask)
        state1 = self.env.step(action)
        self.assertFalse(np.array_equal(state0.state, state1.state))
        restored = self.env.restore_snapshot(snap)
        np.testing.assert_allclose(restored.state, state0.state, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(restored.mask, state0.mask)
        self.assertEqual(restored.current_player_id, state0.current_player_id)
        self.env.drop_snapshot(snap)
        with self.assertRaises(Exception):
            self.env.restore_snapshot(snap)

    def test_restore_invalid_snapshot_id_raises(self) -> None:
        self.env.reset(seed=123)
        with self.assertRaises(Exception):
            self.env.restore_snapshot(999999)

    def test_drop_invalid_snapshot_id_raises(self) -> None:
        self.env.reset(seed=123)
        with self.assertRaises(Exception):
            self.env.drop_snapshot(999999)

    def test_reset_clears_snapshot_table(self) -> None:
        self.env.reset(seed=123)
        snap = self.env.snapshot()
        self.env.reset(seed=123)
        with self.assertRaises(Exception):
            self.env.restore_snapshot(snap)

    def test_multiple_snapshots_restore_correct_branches(self) -> None:
        root = self.env.reset(seed=123)
        root_snap = self.env.snapshot()
        legal = np.flatnonzero(root.mask)
        self.assertGreaterEqual(legal.size, 2, "Expected at least two legal actions at root")
        a0, a1 = int(legal[0]), int(legal[1])

        state_a = self.env.step(a0)
        snap_a = self.env.snapshot()

        self.env.restore_snapshot(root_snap)
        state_b = self.env.step(a1)
        snap_b = self.env.snapshot()

        self.assertFalse(np.array_equal(state_a.state, state_b.state))

        restored_a1 = self.env.restore_snapshot(snap_a)
        restored_a2 = self.env.restore_snapshot(snap_a)
        np.testing.assert_allclose(restored_a1.state, state_a.state, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(restored_a2.state, state_a.state, rtol=0.0, atol=0.0)
        self.assertEqual(restored_a1.current_player_id, state_a.current_player_id)

        restored_b = self.env.restore_snapshot(snap_b)
        np.testing.assert_allclose(restored_b.state, state_b.state, rtol=0.0, atol=0.0)
        self.assertEqual(restored_b.current_player_id, state_b.current_player_id)

        for sid in (snap_b, snap_a, root_snap):
            self.env.drop_snapshot(sid)

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
        # Not a hard requirement, but useful signal if rollout horizon becomes too short.
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
            # Current player may stay the same during return/noble phases; this is just a type/consistency check.
            self.assertIn(prev, (0, 1))

    def test_restore_snapshot_restores_current_player_id(self) -> None:
        state0 = self.env.reset(seed=321)
        snap = self.env.snapshot()
        state1 = self.env.step(self._first_legal_action(state0.mask))
        self.assertEqual(state1.current_player_id, self.env.current_player_id)
        restored = self.env.restore_snapshot(snap)
        self.assertEqual(restored.current_player_id, state0.current_player_id)
        self.assertEqual(self.env.current_player_id, state0.current_player_id)

    def test_phase_flags_are_boolean_fields(self) -> None:
        state = self.env.reset(seed=123)
        self.assertIsInstance(state.is_return_phase, bool)
        self.assertIsInstance(state.is_noble_choice_phase, bool)
        for _ in range(20):
            if state.is_terminal:
                break
            action = self._first_legal_action(state.mask)
            state = self.env.step(action)
            self.assertIsInstance(state.is_return_phase, bool)
            self.assertIsInstance(state.is_noble_choice_phase, bool)

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

                snap = self.env.snapshot()
                restored = self.env.restore_snapshot(snap)
                raw_restored = np.asarray(self.env.debug_raw_state(), dtype=np.int32)
                np.testing.assert_allclose(restored.state, encode_state(raw_restored.tolist()), rtol=0.0, atol=0.0)
                self.env.drop_snapshot(snap)

                if state.is_terminal:
                    break
                action = self._random_legal_action(state.mask, rng)
                state = self.env.step(action)


if __name__ == "__main__":
    unittest.main()
