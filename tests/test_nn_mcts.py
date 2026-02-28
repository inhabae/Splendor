#!/usr/bin/env python3
import random
import unittest

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

if np is not None and torch is not None:
    from nn.bridge_env import StepState
    from nn.mcts import MCTSConfig, _apply_dirichlet_root_noise, _sample_action_from_visits, run_mcts
    from nn.state_schema import ACTION_DIM, STATE_DIM
else:
    StepState = None
    MCTSConfig = None
    _apply_dirichlet_root_noise = None
    _sample_action_from_visits = None
    run_mcts = None
    ACTION_DIM = 69
    STATE_DIM = 246


def _mk_state(state_id: int, legal_actions, *, current_player: int, terminal: bool = False, winner: int = -2):
    state = np.zeros((STATE_DIM,), dtype=np.float32)
    state[0] = float(state_id)
    mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
    for a in legal_actions:
        mask[a] = True
    return StepState(
        state=state,
        mask=mask,
        is_terminal=terminal,
        winner=winner,
        current_player_id=current_player,
    )


class _FakeSnapshotEnv:
    def __init__(self, states: dict[int, object], transitions: dict[tuple[int, int], int], initial_state_id: int):
        self._states = states
        self._transitions = transitions
        self._current_state_id = initial_state_id
        self._snapshots: dict[int, int] = {}
        self._next_snapshot_id = 1

    @property
    def current_player_id(self):
        return int(self._states[self._current_state_id].current_player_id)

    def snapshot(self) -> int:
        sid = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots[sid] = self._current_state_id
        return sid

    def restore_snapshot(self, snapshot_id: int):
        if snapshot_id not in self._snapshots:
            raise RuntimeError("unknown snapshot")
        self._current_state_id = self._snapshots[snapshot_id]
        return self._states[self._current_state_id]

    def drop_snapshot(self, snapshot_id: int) -> None:
        if snapshot_id not in self._snapshots:
            raise RuntimeError("unknown snapshot")
        del self._snapshots[snapshot_id]

    def step(self, action_idx: int):
        key = (self._current_state_id, int(action_idx))
        if key not in self._transitions:
            raise RuntimeError(f"missing transition for {key}")
        self._current_state_id = self._transitions[key]
        return self._states[self._current_state_id]


if nn is not None:
    class _FixedModel(nn.Module):
        def __init__(self, priors_by_state_id: dict[int, dict[int, float]], value_by_state_id: dict[int, float]):
            super().__init__()
            self._priors = priors_by_state_id
            self._values = value_by_state_id

        def forward(self, x):
            batch = x.shape[0]
            logits = torch.full((batch, ACTION_DIM), -5.0, dtype=torch.float32, device=x.device)
            values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
            for i in range(batch):
                state_id = int(round(float(x[i, 0].item())))
                for action, prob_like in self._priors.get(state_id, {}).items():
                    logits[i, int(action)] = float(prob_like)
                values[i] = float(self._values.get(state_id, 0.0))
            return logits, values
else:  # pragma: no cover
    _FixedModel = None


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
class TestNNMCTS(unittest.TestCase):
    def test_apply_dirichlet_root_noise_invariants_and_determinism(self):
        priors = np.zeros((ACTION_DIM,), dtype=np.float32)
        mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        mask[[3, 4, 12]] = True
        priors[3] = 0.7
        priors[4] = 0.2
        priors[12] = 0.1

        out_a = _apply_dirichlet_root_noise(
            priors,
            mask,
            epsilon=0.25,
            alpha_total=6.0,
            rng=random.Random(123),
        )
        out_b = _apply_dirichlet_root_noise(
            priors,
            mask,
            epsilon=0.25,
            alpha_total=6.0,
            rng=random.Random(123),
        )
        self.assertEqual(out_a.shape, (ACTION_DIM,))
        self.assertTrue(np.isfinite(out_a).all())
        self.assertAlmostEqual(float(out_a.sum()), 1.0, places=6)
        self.assertEqual(float(out_a[0]), 0.0)
        np.testing.assert_allclose(out_a, out_b, rtol=0.0, atol=0.0)

        no_noise = _apply_dirichlet_root_noise(
            priors,
            mask,
            epsilon=0.0,
            alpha_total=6.0,
            rng=random.Random(1),
        )
        np.testing.assert_allclose(no_noise, priors, rtol=0.0, atol=0.0)

        single_mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        single_mask[3] = True
        single_priors = np.zeros((ACTION_DIM,), dtype=np.float32)
        single_priors[3] = 1.0
        single_out = _apply_dirichlet_root_noise(
            single_priors,
            single_mask,
            epsilon=0.25,
            alpha_total=6.0,
            rng=random.Random(1),
        )
        np.testing.assert_allclose(single_out, single_priors, rtol=0.0, atol=0.0)

    def test_run_mcts_returns_legal_action_and_normalized_visit_probs(self):
        states = {
            0: _mk_state(0, [3, 4], current_player=0),
            1: _mk_state(1, [], current_player=1, terminal=True, winner=0),
            2: _mk_state(2, [], current_player=1, terminal=True, winner=1),
        }
        transitions = {(0, 3): 1, (0, 4): 2}
        env = _FakeSnapshotEnv(states, transitions, initial_state_id=0)
        model = _FixedModel({0: {3: 3.0, 4: 1.0}}, {0: 0.0})

        result = run_mcts(
            env,
            model,
            states[0],
            turns_taken=99,
            device="cpu",
            config=MCTSConfig(num_simulations=12, c_puct=1.25, temperature_moves=0, temperature=1.0),
            rng=random.Random(0),
        )
        self.assertIn(result.action, [3, 4])
        self.assertEqual(result.visit_probs.shape, (ACTION_DIM,))
        self.assertTrue(np.isfinite(result.visit_probs).all())
        self.assertAlmostEqual(float(result.visit_probs.sum()), 1.0, places=6)
        self.assertEqual(float(result.visit_probs[0]), 0.0)
        self.assertGreaterEqual(float(result.visit_probs[3]), 0.0)
        self.assertGreaterEqual(float(result.visit_probs[4]), 0.0)

    def test_backup_sign_respects_same_player_vs_player_change(self):
        # Action 3 keeps same player and wins for player 0 (should back up positive).
        # Action 4 changes to player 1 and wins for player 1 (should back up negative for root/player 0).
        states = {
            0: _mk_state(0, [3, 4], current_player=0),
            1: _mk_state(1, [], current_player=0, terminal=True, winner=0),
            2: _mk_state(2, [], current_player=1, terminal=True, winner=1),
        }
        transitions = {(0, 3): 1, (0, 4): 2}
        env = _FakeSnapshotEnv(states, transitions, initial_state_id=0)
        model = _FixedModel({0: {3: 1.0, 4: 1.0}}, {0: 0.0})

        result = run_mcts(
            env,
            model,
            states[0],
            turns_taken=99,
            device="cpu",
            config=MCTSConfig(num_simulations=24, c_puct=0.5, temperature_moves=0, temperature=1.0),
            rng=random.Random(0),
        )
        self.assertEqual(result.action, 3)
        self.assertGreater(float(result.visit_probs[3]), float(result.visit_probs[4]))

    def test_visit_sampling_helper_samples_early_and_argmax_late(self):
        visit_probs = np.zeros((ACTION_DIM,), dtype=np.float32)
        legal_mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        legal_mask[[3, 4]] = True
        visit_probs[3] = 0.2
        visit_probs[4] = 0.8
        cfg = MCTSConfig(num_simulations=4, temperature_moves=5, temperature=1.0)

        late_action = _sample_action_from_visits(
            visit_probs, legal_mask, turns_taken=5, config=cfg, rng=random.Random(0)
        )
        self.assertEqual(late_action, 4)

        early_samples = {
            _sample_action_from_visits(visit_probs, legal_mask, turns_taken=0, config=cfg, rng=random.Random(seed))
            for seed in range(10)
        }
        self.assertTrue(early_samples.issubset({3, 4}))
        self.assertIn(4, early_samples)

    def test_run_mcts_root_dirichlet_noise_changes_root_visit_distribution(self):
        states = {
            0: _mk_state(0, [3, 4], current_player=0),
            1: _mk_state(1, [], current_player=1, terminal=True, winner=-1),
            2: _mk_state(2, [], current_player=1, terminal=True, winner=-1),
        }
        transitions = {(0, 3): 1, (0, 4): 2}
        model = _FixedModel({0: {3: 1.0, 4: 1.0}}, {0: 0.0})

        env_plain = _FakeSnapshotEnv(states, transitions, initial_state_id=0)
        plain = run_mcts(
            env_plain,
            model,
            states[0],
            turns_taken=99,
            device="cpu",
            config=MCTSConfig(num_simulations=32, c_puct=1.25, temperature_moves=0, temperature=1.0),
            rng=random.Random(7),
        )

        env_noisy = _FakeSnapshotEnv(states, transitions, initial_state_id=0)
        noisy = run_mcts(
            env_noisy,
            model,
            states[0],
            turns_taken=99,
            device="cpu",
            config=MCTSConfig(
                num_simulations=32,
                c_puct=1.25,
                temperature_moves=0,
                temperature=1.0,
                root_dirichlet_noise=True,
                root_dirichlet_epsilon=0.25,
                root_dirichlet_alpha_total=2.0,
            ),
            rng=random.Random(7),
        )

        self.assertAlmostEqual(float(plain.visit_probs.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(noisy.visit_probs.sum()), 1.0, places=6)
        self.assertTrue(np.isfinite(noisy.visit_probs).all())
        plain_legal = plain.visit_probs[[3, 4]]
        noisy_legal = noisy.visit_probs[[3, 4]]
        self.assertFalse(np.allclose(plain_legal, noisy_legal, atol=1e-6, rtol=0.0))

    def test_run_mcts_validates_root_dirichlet_params(self):
        states = {
            0: _mk_state(0, [3, 4], current_player=0),
            1: _mk_state(1, [], current_player=1, terminal=True, winner=0),
            2: _mk_state(2, [], current_player=1, terminal=True, winner=1),
        }
        transitions = {(0, 3): 1, (0, 4): 2}
        env = _FakeSnapshotEnv(states, transitions, initial_state_id=0)
        model = _FixedModel({0: {3: 1.0, 4: 1.0}}, {0: 0.0})

        with self.assertRaises(ValueError):
            run_mcts(
                env,
                model,
                states[0],
                turns_taken=0,
                device="cpu",
                config=MCTSConfig(num_simulations=4, root_dirichlet_epsilon=1.5),
                rng=random.Random(0),
            )
        with self.assertRaises(ValueError):
            run_mcts(
                env,
                model,
                states[0],
                turns_taken=0,
                device="cpu",
                config=MCTSConfig(num_simulations=4, root_dirichlet_alpha_total=0.0),
                rng=random.Random(0),
            )


if __name__ == "__main__":
    unittest.main()
