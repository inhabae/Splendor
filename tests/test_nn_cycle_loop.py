#!/usr/bin/env python3
import random
import unittest
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
    import nn.train as train_mod
    from nn.replay import ReplayBuffer, ReplaySample
    from nn.state_codec import ACTION_DIM, STATE_DIM
else:
    train_mod = None
    ReplayBuffer = None
    ReplaySample = None
    ACTION_DIM = 69
    STATE_DIM = 246


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
class TestNNCycleLoopHelpers(unittest.TestCase):
    def _make_valid_sample(self, action=3):
        state = np.zeros((STATE_DIM,), dtype=np.float32)
        mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        mask[action] = True
        policy = np.zeros((ACTION_DIM,), dtype=np.float32)
        policy[action] = 1.0
        return ReplaySample(state=state, mask=mask, action_target=action, value_target=1.0, policy_target=policy)

    def test_train_on_replay_aggregates_metrics_and_preserves_last_step(self):
        class _FakeReplay:
            def __len__(self):
                return 5

            def sample_batch(self, batch_size, device="cpu"):
                return {"dummy": True, "batch_size": batch_size, "device": device}

        replay = _FakeReplay()
        fake_model = object()
        fake_optimizer = object()
        side_effect = [
            {"policy_loss": 2.0, "value_loss": 0.5, "total_loss": 2.5, "grad_norm": 1.0, "legal_target_ok": 1.0},
            {"policy_loss": 1.0, "value_loss": 0.3, "total_loss": 1.3, "grad_norm": 2.0, "legal_target_ok": 1.0},
            {"policy_loss": 3.0, "value_loss": 0.1, "total_loss": 3.1, "grad_norm": 3.0, "legal_target_ok": 1.0},
        ]

        with mock.patch.object(train_mod, "train_one_step", side_effect=side_effect):
            metrics = train_mod._train_on_replay(
                fake_model,
                fake_optimizer,
                replay,
                batch_size=4,
                train_steps=3,
                log_every=10,
                device="cpu",
            )

        self.assertEqual(metrics["train_steps"], 3.0)
        self.assertAlmostEqual(metrics["avg_policy_loss"], (2.0 + 1.0 + 3.0) / 3.0)
        self.assertAlmostEqual(metrics["avg_value_loss"], (0.5 + 0.3 + 0.1) / 3.0)
        self.assertAlmostEqual(metrics["avg_total_loss"], (2.5 + 1.3 + 3.1) / 3.0)
        self.assertAlmostEqual(metrics["avg_grad_norm"], (1.0 + 2.0 + 3.0) / 3.0)
        # Last-step metrics preserved
        self.assertEqual(metrics["policy_loss"], 3.0)
        self.assertEqual(metrics["value_loss"], 0.1)
        self.assertEqual(metrics["total_loss"], 3.1)
        self.assertEqual(metrics["grad_norm"], 3.0)

    def test_collect_replay_seed_progression_is_monotonic(self):
        replay = ReplayBuffer()
        rng = random.Random(0)
        observed_seeds = []

        def _fake_collect_episode(env, replay, *, seed, max_turns, rng, collector_policy, model=None, device="cpu", collector_stats=None, mcts_config=None):
            observed_seeds.append(seed)
            replay.add(self._make_valid_sample(action=3))
            if collector_stats is not None:
                collector_stats.random_actions += 1
            return train_mod.EpisodeSummary(num_steps=1, num_turns=1, reached_cutoff=False, winner=0)

        with mock.patch.object(train_mod, "collect_episode", side_effect=_fake_collect_episode):
            metrics = train_mod._collect_replay(
                env=object(),
                replay=replay,
                episodes=3,
                max_turns=10,
                rng=rng,
                collector_policy="random",
                model=object(),
                device="cpu",
                seed_start=7,
            )

        self.assertEqual(observed_seeds, [7, 8, 9])
        self.assertEqual(metrics["next_seed"], 10)
        self.assertEqual(metrics["episodes"], 3.0)
        self.assertEqual(metrics["terminal_episodes"], 3.0)
        self.assertEqual(metrics["cutoff_episodes"], 0.0)
        self.assertEqual(metrics["replay_samples"], 3.0)
        self.assertEqual(metrics["collector_random_actions"], 3.0)
        self.assertEqual(metrics["collector_model_actions"], 0.0)
        self.assertEqual(metrics["collector_mcts_actions"], 0.0)

    def test_run_cycles_argument_validation(self):
        with self.assertRaises(ValueError):
            train_mod.run_cycles(cycles=0)
        with self.assertRaises(ValueError):
            train_mod.run_cycles(cycles=1, episodes_per_cycle=0)
        with self.assertRaises(ValueError):
            train_mod.run_cycles(cycles=1, episodes_per_cycle=1, train_steps_per_cycle=0)
        with self.assertRaises(ValueError):
            train_mod.run_cycles(cycles=1, episodes_per_cycle=1, train_steps_per_cycle=1, collector_policy="bad")

    def test_mcts_root_dirichlet_noise_config_wiring(self):
        class _StopBeforeEnv(Exception):
            pass

        class _FailEnterCtx:
            def __enter__(self):
                raise _StopBeforeEnv()

            def __exit__(self, exc_type, exc, tb):
                return False

        smoke_cfg_calls = []

        def _capture_smoke_cfg(**kwargs):
            smoke_cfg_calls.append(kwargs)
            return object()

        with mock.patch.object(train_mod, "MCTSConfig", side_effect=_capture_smoke_cfg):
            with mock.patch.object(train_mod, "SplendorBridgeEnv", return_value=_FailEnterCtx()):
                with self.assertRaises(_StopBeforeEnv):
                    train_mod.run_smoke(
                        episodes=1,
                        train_steps=1,
                        collector_policy="mcts",
                        mcts_root_dirichlet_noise=True,
                        mcts_root_dirichlet_epsilon=0.3,
                        mcts_root_dirichlet_alpha_total=7.5,
                    )

        self.assertEqual(len(smoke_cfg_calls), 1)
        self.assertTrue(smoke_cfg_calls[0]["root_dirichlet_noise"])
        self.assertAlmostEqual(smoke_cfg_calls[0]["root_dirichlet_epsilon"], 0.3)
        self.assertAlmostEqual(smoke_cfg_calls[0]["root_dirichlet_alpha_total"], 7.5)

        cycle_cfg_calls = []

        def _capture_cycle_cfg(**kwargs):
            cycle_cfg_calls.append(kwargs)
            return object()

        with mock.patch.object(train_mod, "MCTSConfig", side_effect=_capture_cycle_cfg):
            with mock.patch.object(train_mod, "SplendorBridgeEnv", return_value=_FailEnterCtx()):
                with self.assertRaises(_StopBeforeEnv):
                    train_mod.run_cycles(
                        cycles=1,
                        episodes_per_cycle=1,
                        train_steps_per_cycle=1,
                        collector_policy="mcts",
                        mcts_root_dirichlet_noise=True,
                        mcts_root_dirichlet_epsilon=0.4,
                        mcts_root_dirichlet_alpha_total=9.0,
                    )

        self.assertEqual(len(cycle_cfg_calls), 3)
        self.assertTrue(cycle_cfg_calls[0]["root_dirichlet_noise"])
        self.assertAlmostEqual(cycle_cfg_calls[0]["root_dirichlet_epsilon"], 0.4)
        self.assertAlmostEqual(cycle_cfg_calls[0]["root_dirichlet_alpha_total"], 9.0)
        self.assertFalse(cycle_cfg_calls[1]["root_dirichlet_noise"])
        self.assertFalse(cycle_cfg_calls[2]["root_dirichlet_noise"])

    def test_run_checkpoint_benchmark_uses_eval_mcts_without_root_noise(self):
        cfg_calls = []

        def _capture_cfg(**kwargs):
            cfg_calls.append(kwargs)
            return object()

        fake_suite = train_mod.BenchmarkSuiteResult(
            candidate_checkpoint="ckpt.pt",
            matchups=[],
            suite_candidate_wins=0,
            suite_candidate_losses=0,
            suite_draws=0,
            suite_avg_turns_per_game=0.0,
            warnings=[],
        )

        with mock.patch.object(train_mod, "MCTSConfig", side_effect=_capture_cfg):
            with mock.patch.object(train_mod, "_build_suite_opponents_from_registry", return_value=([], False)):
                with mock.patch.object(train_mod, "CheckpointMCTSOpponent", return_value=object()) as cand_mock:
                    with mock.patch.object(train_mod, "run_benchmark_suite", return_value=fake_suite) as suite_mock:
                        with mock.patch.object(train_mod, "_print_benchmark_suite"):
                            metrics = train_mod.run_checkpoint_benchmark(
                                candidate_checkpoint="ckpt.pt",
                                benchmark_games_per_opponent=4,
                                benchmark_mcts_sims=32,
                                mcts_c_puct=1.5,
                                benchmark_seed=123,
                                benchmark_cycle_idx=9,
                            )

        self.assertEqual(len(cfg_calls), 1)
        self.assertFalse(cfg_calls[0]["root_dirichlet_noise"])
        self.assertEqual(cfg_calls[0]["temperature_moves"], 0)
        self.assertEqual(cfg_calls[0]["temperature"], 0.0)
        cand_mock.assert_called_once()
        suite_mock.assert_called_once()
        self.assertEqual(metrics["mode"], "benchmark")
        self.assertEqual(metrics["benchmark_candidate_checkpoint"], "ckpt.pt")


if __name__ == "__main__":
    unittest.main()
