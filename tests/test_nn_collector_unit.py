#!/usr/bin/env python3
import random
import unittest
from dataclasses import dataclass
from unittest import mock

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

if np is not None and torch is not None:
    from nn.native_env import StepState
    from nn.model import MaskedPolicyValueNet
    from nn.replay import ReplayBuffer
    from nn.state_schema import ACTION_DIM, STATE_DIM
    from nn.train import CollectorStats, collect_episode
else:
    StepState = None
    MaskedPolicyValueNet = None
    ReplayBuffer = None
    CollectorStats = None
    collect_episode = None
    ACTION_DIM = 69
    STATE_DIM = 246


def _mk_state(mask_indices, *, terminal=False, winner=-2):
    state = np.zeros((STATE_DIM,), dtype=np.float32)
    mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
    for i in mask_indices:
        mask[i] = True
    return StepState(
        state=state,
        mask=mask,
        is_terminal=terminal,
        winner=winner,
    )


@dataclass
class _Transition:
    next_state: object
    next_player_id: int


class FakeEnv:
    def __init__(self, initial_state, transitions):
        self._initial_state = initial_state
        self._transitions = list(transitions)
        self._idx = 0
        self._current_player_id = 0

    @property
    def current_player_id(self):
        return self._current_player_id

    def reset(self, seed=0):
        self._idx = 0
        self._current_player_id = 0
        return self._initial_state

    def step(self, action_idx):
        if self._idx >= len(self._transitions):
            raise RuntimeError("No more fake transitions")
        t = self._transitions[self._idx]
        self._idx += 1
        self._current_player_id = t.next_player_id
        return t.next_state


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
class TestNNCollectorUnit(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(123)

    def test_collect_episode_random_mode_stores_replay_and_counts_random_actions(self):
        initial = _mk_state([3])
        s1 = _mk_state([4])
        terminal = _mk_state([], terminal=True, winner=0)
        env = FakeEnv(initial, [_Transition(s1, 1), _Transition(terminal, 0)])
        replay = ReplayBuffer()
        stats = CollectorStats()

        summary = collect_episode(
            env,
            replay,
            seed=1,
            max_turns=2,
            rng=self.rng,
            collector_policy="random",
            collector_stats=stats,
        )
        self.assertEqual(summary.num_steps, 2)
        self.assertEqual(summary.num_turns, 2)
        self.assertFalse(summary.reached_cutoff)
        self.assertEqual(summary.winner, 0)
        self.assertEqual(len(replay), 2)
        self.assertEqual(stats.random_actions, 2)
        self.assertEqual(stats.model_actions, 0)
        self.assertEqual(stats.mcts_actions, 0)
        self.assertEqual(float(replay._samples[0].policy_target.sum()), 1.0)
        self.assertEqual(float(replay._samples[0].policy_target[replay._samples[0].action_target]), 1.0)

    def test_collect_episode_model_sample_requires_model(self):
        initial = _mk_state([3])
        env = FakeEnv(initial, [])
        replay = ReplayBuffer()
        with self.assertRaises(ValueError):
            collect_episode(
                env,
                replay,
                seed=1,
                max_turns=1,
                rng=self.rng,
                collector_policy="model-sample",
                model=None,
            )

    def test_collect_episode_unknown_policy_raises(self):
        initial = _mk_state([3])
        env = FakeEnv(initial, [])
        replay = ReplayBuffer()
        with self.assertRaises(ValueError):
            collect_episode(
                env,
                replay,
                seed=1,
                max_turns=1,
                rng=self.rng,
                collector_policy="unknown",
            )

    def test_terminal_on_final_allowed_turn_is_not_cutoff_draw(self):
        # max_turns == 1, terminal reached after the first action that also ends the turn.
        initial = _mk_state([3])
        terminal = _mk_state([], terminal=True, winner=0)
        env = FakeEnv(initial, [_Transition(terminal, 1)])
        replay = ReplayBuffer()
        stats = CollectorStats()

        summary = collect_episode(
            env,
            replay,
            seed=1,
            max_turns=1,
            rng=self.rng,
            collector_policy="random",
            collector_stats=stats,
        )
        self.assertFalse(summary.reached_cutoff)
        self.assertEqual(summary.winner, 0)
        self.assertEqual(summary.num_turns, 1)
        self.assertEqual(len(replay), 1)
        sample = replay._samples[0]
        self.assertEqual(sample.value_target, 1.0)

    def test_turn_count_only_increments_on_player_change(self):
        initial = _mk_state([3])
        same_player_continues = _mk_state([4])
        terminal = _mk_state([], terminal=True, winner=0)
        env = FakeEnv(
            initial,
            [
                _Transition(same_player_continues, 0),  # same player, no turn increment
                _Transition(terminal, 1),  # player changes, one turn counted
            ],
        )
        replay = ReplayBuffer()

        summary = collect_episode(
            env,
            replay,
            seed=1,
            max_turns=1,
            rng=self.rng,
            collector_policy="random",
        )
        self.assertEqual(summary.num_steps, 2)
        self.assertEqual(summary.num_turns, 1)
        self.assertFalse(summary.reached_cutoff)

    def test_value_targets_assign_signs_by_player_and_draw(self):
        # p0 acts, then p1 acts, then terminal winner=0
        initial = _mk_state([3])
        p1_state = _mk_state([4])
        terminal_win_p0 = _mk_state([], terminal=True, winner=0)
        env = FakeEnv(initial, [_Transition(p1_state, 1), _Transition(terminal_win_p0, 0)])
        replay = ReplayBuffer()

        summary = collect_episode(
            env,
            replay,
            seed=1,
            max_turns=3,
            rng=self.rng,
            collector_policy="random",
        )
        self.assertEqual(summary.winner, 0)
        self.assertEqual([s.value_target for s in replay._samples], [1.0, -1.0])

        # Draw path via cutoff -> all zeros
        looping_initial = _mk_state([3])
        looping_next = _mk_state([4])
        # Provide enough transitions so the loop stops due to turn cutoff, not fake-env exhaustion.
        env2 = FakeEnv(looping_initial, [_Transition(looping_next, 1), _Transition(looping_initial, 0)])
        replay2 = ReplayBuffer()
        summary2 = collect_episode(
            env2,
            replay2,
            seed=1,
            max_turns=2,
            rng=self.rng,
            collector_policy="random",
        )
        self.assertTrue(summary2.reached_cutoff)
        self.assertEqual(summary2.winner, -1)
        self.assertTrue(all(s.value_target == 0.0 for s in replay2._samples))

    def test_cutoff_labels_draw_when_never_terminal_before_limit(self):
        initial = _mk_state([3])
        next_state = _mk_state([4])
        env = FakeEnv(initial, [_Transition(next_state, 1)])
        replay = ReplayBuffer()

        summary = collect_episode(
            env,
            replay,
            seed=1,
            max_turns=1,
            rng=self.rng,
            collector_policy="random",
        )
        self.assertTrue(summary.reached_cutoff)
        self.assertEqual(summary.winner, -1)
        self.assertEqual(summary.num_turns, 1)
        self.assertTrue(all(s.value_target == 0.0 for s in replay._samples))

    def test_collect_episode_model_sample_mode_counts_model_actions(self):
        model = MaskedPolicyValueNet()
        initial = _mk_state([3, 12, 31])
        terminal = _mk_state([], terminal=True, winner=0)
        env = FakeEnv(initial, [_Transition(terminal, 1)])
        replay = ReplayBuffer()
        stats = CollectorStats()

        summary = collect_episode(
            env,
            replay,
            seed=1,
            max_turns=1,
            rng=self.rng,
            collector_policy="model-sample",
            model=model,
            device="cpu",
            collector_stats=stats,
        )
        self.assertEqual(summary.num_steps, 1)
        self.assertEqual(stats.random_actions, 0)
        self.assertEqual(stats.model_actions, 1)
        self.assertEqual(stats.mcts_actions, 0)
        self.assertEqual(len(replay), 1)
        sample = replay._samples[0]
        self.assertEqual(float(sample.policy_target.sum()), 1.0)
        self.assertEqual(float(sample.policy_target[sample.action_target]), 1.0)

    def test_collect_episode_mcts_requires_model(self):
        initial = _mk_state([3])
        env = FakeEnv(initial, [])
        replay = ReplayBuffer()
        with self.assertRaises(ValueError):
            collect_episode(
                env,
                replay,
                seed=1,
                max_turns=1,
                rng=self.rng,
                collector_policy="mcts",
                model=None,
            )

    def test_collect_episode_mcts_mode_stores_visit_policy_and_counts_mcts_actions(self):
        model = MaskedPolicyValueNet()
        initial = _mk_state([3, 12, 31])
        terminal = _mk_state([], terminal=True, winner=0)
        env = FakeEnv(initial, [_Transition(terminal, 1)])
        replay = ReplayBuffer()
        stats = CollectorStats()

        visit_probs = np.zeros((ACTION_DIM,), dtype=np.float32)
        visit_probs[3] = 0.25
        visit_probs[12] = 0.75

        fake_result = mock.Mock(
            chosen_action_idx=12,
            visit_probs=visit_probs,
            root_value=0.0,
        )
        with mock.patch("nn.train.run_mcts", return_value=fake_result) as run_mcts_mock:
            summary = collect_episode(
                env,
                replay,
                seed=1,
                max_turns=1,
                rng=self.rng,
                collector_policy="mcts",
                model=model,
                device="cpu",
                collector_stats=stats,
            )

        self.assertEqual(summary.num_steps, 1)
        self.assertEqual(stats.random_actions, 0)
        self.assertEqual(stats.model_actions, 0)
        self.assertEqual(stats.mcts_actions, 1)
        self.assertEqual(len(replay), 1)
        run_mcts_mock.assert_called_once()
        sample = replay._samples[0]
        self.assertAlmostEqual(float(sample.policy_target[3]), 0.25, places=6)
        self.assertAlmostEqual(float(sample.policy_target[12]), 0.75, places=6)
        self.assertEqual(sample.action_target, 12)


if __name__ == "__main__":
    unittest.main()
