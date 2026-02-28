#!/usr/bin/env python3
from __future__ import annotations

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
    from nn.mcts import MCTSConfig, run_mcts
    from nn.native_env import SplendorNativeEnv, StepState
    from nn.state_schema import ACTION_DIM, STATE_DIM
else:  # pragma: no cover
    MCTSConfig = None
    run_mcts = None
    SplendorNativeEnv = None
    StepState = None
    ACTION_DIM = 69
    STATE_DIM = 246


if nn is not None:
    class _GoodModel(nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros((batch, ACTION_DIM), dtype=torch.float32, device=x.device)
            values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
            return logits, values


    class _BadPolicyShapeModel(nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros((batch, ACTION_DIM - 1), dtype=torch.float32, device=x.device)
            values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
            return logits, values


    class _BadValueShapeModel(nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros((batch, ACTION_DIM), dtype=torch.float32, device=x.device)
            values = torch.zeros((batch, 2), dtype=torch.float32, device=x.device)
            return logits, values


    class _NanPolicyModel(nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros((batch, ACTION_DIM), dtype=torch.float32, device=x.device)
            logits[:, 0] = float("nan")
            values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
            return logits, values


    class _NanValueModel(nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros((batch, ACTION_DIM), dtype=torch.float32, device=x.device)
            values = torch.zeros((batch,), dtype=torch.float32, device=x.device)
            values[0] = float("nan")
            return logits, values
else:  # pragma: no cover
    _GoodModel = object
    _BadPolicyShapeModel = object
    _BadValueShapeModel = object
    _NanPolicyModel = object
    _NanValueModel = object


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
class TestNNMCTS(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.env = SplendorNativeEnv()
        except Exception as exc:
            self.skipTest(f"splendor_native unavailable: {exc}")

    def tearDown(self) -> None:
        if hasattr(self, "env"):
            self.env.close()

    def test_run_mcts_requires_native_env(self):
        state = StepState(
            state=np.zeros((STATE_DIM,), dtype=np.float32),
            mask=np.ones((ACTION_DIM,), dtype=np.bool_),
            is_terminal=False,
            winner=-2,
            current_player_id=0,
        )
        with self.assertRaises(TypeError):
            run_mcts(
                object(),
                _GoodModel(),
                state=state,
                turns_taken=0,
                device="cpu",
                config=MCTSConfig(num_simulations=1),
                rng=random.Random(0),
            )

    def test_run_mcts_rejects_terminal_root(self):
        _ = self.env.reset(seed=123)
        terminal = StepState(
            state=np.zeros((STATE_DIM,), dtype=np.float32),
            mask=np.zeros((ACTION_DIM,), dtype=np.bool_),
            is_terminal=True,
            winner=0,
            current_player_id=0,
        )
        with self.assertRaises(ValueError):
            run_mcts(
                self.env,
                _GoodModel(),
                state=terminal,
                turns_taken=0,
                device="cpu",
                config=MCTSConfig(num_simulations=1),
                rng=random.Random(0),
            )

    def test_run_mcts_enforces_model_output_shapes(self):
        state = self.env.reset(seed=123)
        with self.assertRaises(ValueError):
            run_mcts(
                self.env,
                _BadPolicyShapeModel(),
                state=state,
                turns_taken=0,
                device="cpu",
                config=MCTSConfig(num_simulations=2),
                rng=random.Random(0),
            )
        with self.assertRaises(ValueError):
            run_mcts(
                self.env,
                _BadValueShapeModel(),
                state=state,
                turns_taken=0,
                device="cpu",
                config=MCTSConfig(num_simulations=2),
                rng=random.Random(0),
            )

    def test_run_mcts_enforces_finite_model_outputs(self):
        state = self.env.reset(seed=321)
        with self.assertRaises(ValueError):
            run_mcts(
                self.env,
                _NanPolicyModel(),
                state=state,
                turns_taken=0,
                device="cpu",
                config=MCTSConfig(num_simulations=2),
                rng=random.Random(0),
            )
        with self.assertRaises(ValueError):
            run_mcts(
                self.env,
                _NanValueModel(),
                state=state,
                turns_taken=0,
                device="cpu",
                config=MCTSConfig(num_simulations=2),
                rng=random.Random(0),
            )

    def test_run_mcts_result_contract(self):
        state = self.env.reset(seed=222)
        result = run_mcts(
            self.env,
            _GoodModel(),
            state=state,
            turns_taken=99,
            device="cpu",
            config=MCTSConfig(num_simulations=16, temperature_moves=0, temperature=0.0),
            rng=random.Random(7),
        )
        self.assertEqual(result.visit_probs.shape, (ACTION_DIM,))
        self.assertTrue(np.isfinite(result.visit_probs).all())
        self.assertAlmostEqual(float(result.visit_probs.sum()), 1.0, places=5)
        self.assertTrue(bool(state.mask[int(result.chosen_action_idx)]))


if __name__ == "__main__":
    unittest.main()
