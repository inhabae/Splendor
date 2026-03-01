from __future__ import annotations

import random
import unittest
from dataclasses import dataclass

import numpy as np

from nn.native_env import SplendorNativeEnv, StepState
from nn.state_schema import ACTION_DIM
from tests.mcts_reference import ReferenceMCTSResult, run_reference_mcts


def _state_batch_contract(states: np.ndarray, masks: np.ndarray) -> tuple[int, np.ndarray]:
    states = np.asarray(states, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.bool_)
    if states.ndim != 2:
        raise ValueError("states must have shape (B, STATE_DIM)")
    if masks.ndim != 2 or masks.shape[1] != ACTION_DIM:
        raise ValueError("masks must have shape (B, ACTION_DIM)")
    if states.shape[0] != masks.shape[0]:
        raise ValueError("states/masks batch mismatch")
    return int(states.shape[0]), masks


@dataclass
class ZeroEvaluator:
    def __call__(self, states: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch, _ = _state_batch_contract(states, masks)
        logits = np.zeros((batch, ACTION_DIM), dtype=np.float32)
        values = np.zeros((batch,), dtype=np.float32)
        return logits, values


@dataclass
class BiasedEvaluator:
    preferred_action: int = 30
    secondary_action: int = 31

    def __call__(self, states: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch, masks = _state_batch_contract(states, masks)
        logits = np.full((batch, ACTION_DIM), -10.0, dtype=np.float32)
        logits[:, self.preferred_action] = 8.0
        logits[:, self.secondary_action] = 6.0
        # Keep values bounded and deterministic from encoded state.
        values = np.tanh(np.sum(states[:, :16], axis=1) * 0.05).astype(np.float32)
        # Discourage impossible preferred/secondary actions to avoid brittle behavior.
        for i in range(batch):
            if not bool(masks[i, self.preferred_action]):
                logits[i, self.preferred_action] = -20.0
            if not bool(masks[i, self.secondary_action]):
                logits[i, self.secondary_action] = -20.0
        return logits, values


@dataclass
class FiniteButExtremeEvaluator:
    anchor_action: int = 30

    def __call__(self, states: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch, masks = _state_batch_contract(states, masks)
        logits = np.full((batch, ACTION_DIM), -1_000.0, dtype=np.float32)
        logits[:, self.anchor_action] = 1_000.0
        for i in range(batch):
            if not bool(masks[i, self.anchor_action]):
                logits[i, self.anchor_action] = -1_000.0
                legal = np.flatnonzero(masks[i])
                if legal.size > 0:
                    logits[i, int(legal[0])] = 1_000.0
        values = np.tanh(np.mean(states[:, :32], axis=1)).astype(np.float32)
        return logits, values


@dataclass
class DegeneratePriorEvaluator:
    value: float = 0.0

    def __call__(self, states: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch, _ = _state_batch_contract(states, masks)
        logits = np.full((batch, ACTION_DIM), np.nan, dtype=np.float32)
        values = np.full((batch,), self.value, dtype=np.float32)
        return logits, values


@dataclass
class _ReplayRoot:
    seed: int
    root_actions: list[int]
    env: SplendorNativeEnv
    root_state: StepState

    def transition(self, actions_from_root: list[int]) -> StepState:
        state = self.env.reset(seed=int(self.seed))
        for action in self.root_actions:
            state = self.env.step(int(action))
        for action in actions_from_root:
            state = self.env.step(int(action))
        return state


def _build_replay_root(seed: int, root_actions: list[int] | None = None) -> _ReplayRoot:
    root_actions = [] if root_actions is None else [int(a) for a in root_actions]
    env = SplendorNativeEnv()
    state = env.reset(seed=int(seed))
    for action in root_actions:
        state = env.step(int(action))
    return _ReplayRoot(seed=seed, root_actions=root_actions, env=env, root_state=state)


def _run_native(
    root: _ReplayRoot,
    evaluator,
    *,
    turns_taken: int,
    rng_seed: int,
    num_simulations: int = 64,
    c_puct: float = 1.25,
    temperature_moves: int = 10,
    temperature: float = 1.0,
    eps: float = 1e-8,
    root_dirichlet_noise: bool = False,
    root_dirichlet_epsilon: float = 0.25,
    root_dirichlet_alpha_total: float = 10.0,
    eval_batch_size: int = 16,
):
    return root.env.run_mcts_native(
        evaluator=evaluator,
        turns_taken=int(turns_taken),
        num_simulations=int(num_simulations),
        c_puct=float(c_puct),
        temperature_moves=int(temperature_moves),
        temperature=float(temperature),
        eps=float(eps),
        root_dirichlet_noise=bool(root_dirichlet_noise),
        root_dirichlet_epsilon=float(root_dirichlet_epsilon),
        root_dirichlet_alpha_total=float(root_dirichlet_alpha_total),
        eval_batch_size=int(eval_batch_size),
        rng_seed=int(rng_seed),
    )


def _run_reference(
    root: _ReplayRoot,
    evaluator,
    *,
    turns_taken: int,
    rng_seed: int,
    num_simulations: int = 64,
    c_puct: float = 1.25,
    temperature_moves: int = 10,
    temperature: float = 1.0,
    eps: float = 1e-8,
    root_dirichlet_noise: bool = False,
    root_dirichlet_epsilon: float = 0.25,
    root_dirichlet_alpha_total: float = 10.0,
    eval_batch_size: int = 16,
) -> ReferenceMCTSResult:
    return run_reference_mcts(
        root_state=root.root_state,
        transition_from_root=root.transition,
        evaluator=evaluator,
        turns_taken=int(turns_taken),
        num_simulations=int(num_simulations),
        c_puct=float(c_puct),
        temperature_moves=int(temperature_moves),
        temperature=float(temperature),
        eps=float(eps),
        root_dirichlet_noise=bool(root_dirichlet_noise),
        root_dirichlet_epsilon=float(root_dirichlet_epsilon),
        root_dirichlet_alpha_total=float(root_dirichlet_alpha_total),
        eval_batch_size=int(eval_batch_size),
        rng_seed=int(rng_seed),
    )


class TestNNMCTSDifferential(unittest.TestCase):
    def tearDown(self) -> None:
        for attr in ("_root_a", "_root_b", "_root_c", "_root_d"):
            root = getattr(self, attr, None)
            if root is not None:
                root.env.close()

    def _assert_probs_contract(self, probs: np.ndarray, mask: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=np.float32)
        self.assertEqual(probs.shape, (ACTION_DIM,))
        self.assertTrue(np.isfinite(probs).all())
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)
        illegal = np.flatnonzero(~mask)
        if illegal.size > 0:
            self.assertTrue(np.allclose(probs[illegal], 0.0))

    def test_native_matches_reference_no_noise_argmax(self) -> None:
        self._root_a = _build_replay_root(seed=41)
        evaluator = BiasedEvaluator()
        rng_seed = random.Random(123).getrandbits(64)
        native = _run_native(
            self._root_a,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=48,
            c_puct=1.25,
            temperature_moves=0,
            temperature=0.0,
            root_dirichlet_noise=False,
            eval_batch_size=12,
        )
        ref = _run_reference(
            self._root_a,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=48,
            c_puct=1.25,
            temperature_moves=0,
            temperature=0.0,
            root_dirichlet_noise=False,
            eval_batch_size=12,
        )
        native_probs = np.asarray(native.visit_probs, dtype=np.float32)
        self._assert_probs_contract(native_probs, self._root_a.root_state.mask)
        self._assert_probs_contract(ref.visit_probs, self._root_a.root_state.mask)
        self.assertEqual(int(native.chosen_action_idx), int(ref.chosen_action_idx))
        np.testing.assert_allclose(native_probs, ref.visit_probs, rtol=0.0, atol=1e-7)

    def test_native_matches_reference_with_root_noise_seeded(self) -> None:
        self._root_b = _build_replay_root(seed=77)
        evaluator = FiniteButExtremeEvaluator()
        rng_seed = random.Random(321).getrandbits(64)
        # Keep epsilon=0 for deterministic parity while still exercising the noise-gated code path.
        native = _run_native(
            self._root_b,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=32,
            c_puct=1.25,
            temperature_moves=0,
            temperature=0.0,
            root_dirichlet_noise=True,
            root_dirichlet_epsilon=0.0,
            root_dirichlet_alpha_total=10.0,
            eval_batch_size=8,
        )
        ref = _run_reference(
            self._root_b,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=32,
            c_puct=1.25,
            temperature_moves=0,
            temperature=0.0,
            root_dirichlet_noise=True,
            root_dirichlet_epsilon=0.0,
            root_dirichlet_alpha_total=10.0,
            eval_batch_size=8,
        )
        native_probs = np.asarray(native.visit_probs, dtype=np.float32)
        self.assertEqual(int(native.chosen_action_idx), int(ref.chosen_action_idx))
        np.testing.assert_allclose(native_probs, ref.visit_probs, rtol=0.0, atol=1e-7)

    def test_backup_sign_parity_on_player_switch(self) -> None:
        self._root_c = _build_replay_root(seed=133)
        evaluator = BiasedEvaluator(preferred_action=30, secondary_action=45)
        rng_seed = random.Random(999).getrandbits(64)
        native = _run_native(
            self._root_c,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=64,
            c_puct=0.6,
            temperature_moves=0,
            temperature=0.0,
            eval_batch_size=8,
        )
        ref = _run_reference(
            self._root_c,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=64,
            c_puct=0.6,
            temperature_moves=0,
            temperature=0.0,
            eval_batch_size=8,
        )
        self.assertEqual(int(native.chosen_action_idx), int(ref.chosen_action_idx))
        np.testing.assert_allclose(
            np.asarray(native.visit_probs, dtype=np.float32),
            ref.visit_probs,
            rtol=0.0,
            atol=1e-7,
        )
        self.assertAlmostEqual(float(native.root_value), float(ref.root_value), places=6)

    def test_temperature_sampling_parity_seeded(self) -> None:
        self._root_d = _build_replay_root(seed=171)
        evaluator = FiniteButExtremeEvaluator(anchor_action=30)
        rng_seed = random.Random(777).getrandbits(64)
        native = _run_native(
            self._root_d,
            evaluator,
            turns_taken=0,
            rng_seed=rng_seed,
            num_simulations=40,
            c_puct=0.0,
            temperature_moves=10,
            temperature=1.0,
            eval_batch_size=10,
        )
        ref = _run_reference(
            self._root_d,
            evaluator,
            turns_taken=0,
            rng_seed=rng_seed,
            num_simulations=40,
            c_puct=0.0,
            temperature_moves=10,
            temperature=1.0,
            eval_batch_size=10,
        )
        self.assertEqual(int(native.chosen_action_idx), int(ref.chosen_action_idx))
        np.testing.assert_allclose(np.asarray(native.visit_probs, dtype=np.float32), ref.visit_probs, atol=1e-7)

    def test_uniform_fallback_parity_when_logits_nonfinite_or_degenerate(self) -> None:
        root = _build_replay_root(seed=207)
        self.addCleanup(root.env.close)
        evaluator = DegeneratePriorEvaluator(value=0.0)
        rng_seed = random.Random(2024).getrandbits(64)
        native = _run_native(
            root,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=36,
            c_puct=1.0,
            temperature_moves=0,
            temperature=0.0,
            eval_batch_size=9,
        )
        ref = _run_reference(
            root,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=36,
            c_puct=1.0,
            temperature_moves=0,
            temperature=0.0,
            eval_batch_size=9,
        )
        np.testing.assert_allclose(np.asarray(native.visit_probs, dtype=np.float32), ref.visit_probs, atol=1e-7)
        self.assertEqual(int(native.chosen_action_idx), int(ref.chosen_action_idx))

    def test_visit_probs_contract_and_root_value_bounded(self) -> None:
        root = _build_replay_root(seed=251)
        self.addCleanup(root.env.close)
        evaluator = BiasedEvaluator()
        rng_seed = 123456
        native = _run_native(
            root,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=32,
            c_puct=1.25,
            temperature_moves=0,
            temperature=0.0,
            eval_batch_size=8,
        )
        probs = np.asarray(native.visit_probs, dtype=np.float32)
        self._assert_probs_contract(probs, root.root_state.mask)
        self.assertGreaterEqual(float(native.root_value), -1.0)
        self.assertLessEqual(float(native.root_value), 1.0)

    def test_monotonicity_sanity_with_more_simulations(self) -> None:
        root = _build_replay_root(seed=303)
        self.addCleanup(root.env.close)
        evaluator = BiasedEvaluator()
        rng_seed = 42
        res_small = _run_native(
            root,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=24,
            c_puct=1.25,
            temperature_moves=0,
            temperature=0.0,
            eval_batch_size=8,
        )
        res_large = _run_native(
            root,
            evaluator,
            turns_taken=99,
            rng_seed=rng_seed,
            num_simulations=96,
            c_puct=1.25,
            temperature_moves=0,
            temperature=0.0,
            eval_batch_size=16,
        )
        p_small = np.asarray(res_small.visit_probs, dtype=np.float32)
        p_large = np.asarray(res_large.visit_probs, dtype=np.float32)
        self._assert_probs_contract(p_small, root.root_state.mask)
        self._assert_probs_contract(p_large, root.root_state.mask)
        self.assertGreaterEqual(int(np.count_nonzero(p_small > 0.0)), 1)
        self.assertGreaterEqual(int(np.count_nonzero(p_large > 0.0)), 1)
        self.assertEqual(int(np.argmax(p_small)), int(np.argmax(p_large)))


if __name__ == "__main__":
    unittest.main()
