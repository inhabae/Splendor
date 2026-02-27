from __future__ import annotations

import random
import unittest

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from nn.mcts import MCTSConfig, run_mcts
from nn.native_env import SplendorNativeEnv
from nn.state_schema import ACTION_DIM


if nn is not None:
    class _DeckSensitiveModel(nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros((batch, ACTION_DIM), dtype=torch.float32, device=x.device)
            # Emphasize face-up reserve actions (15-26), where deck refill effects appear.
            logits[:, 15:27] = 1.5

            faceup_start = 91
            card_feature_len = 11
            slot0_points = x[:, faceup_start + 10]
            slot1_points = x[:, faceup_start + card_feature_len + 10]
            values = torch.tanh(3.0 * (slot0_points - slot1_points))
            return logits, values
else:  # pragma: no cover
    _DeckSensitiveModel = object


@unittest.skipIf(torch is None, "torch not installed")
class TestNativeISMCTSDeckSampling(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.env = SplendorNativeEnv()
        except Exception as exc:
            self.skipTest(f"splendor_native unavailable: {exc}")
        self.model = _DeckSensitiveModel()

    def tearDown(self) -> None:
        if hasattr(self, "env"):
            self.env.close()

    def test_native_mcts_result_contract_still_holds(self) -> None:
        state = self.env.reset(seed=123)
        result = run_mcts(
            self.env,
            self.model,
            state=state,
            turns_taken=99,
            device="cpu",
            config=MCTSConfig(num_simulations=48, temperature_moves=0, temperature=0.0),
            rng=random.Random(11),
        )
        self.assertEqual(result.visit_probs.shape, (ACTION_DIM,))
        self.assertTrue(np.isfinite(result.visit_probs).all())
        self.assertAlmostEqual(float(result.visit_probs.sum()), 1.0, places=5)
        illegal = np.flatnonzero(~state.mask)
        if illegal.size > 0:
            self.assertTrue(np.allclose(result.visit_probs[illegal], 0.0))
        self.assertTrue(bool(state.mask[int(result.action)]))

    def test_per_sim_root_sampling_changes_visits_across_rng_seeds(self) -> None:
        state = self.env.reset(seed=321)
        snap = self.env.snapshot()
        cfg = MCTSConfig(num_simulations=160, c_puct=1.25, temperature_moves=0, temperature=0.0)

        result_a = run_mcts(
            self.env,
            self.model,
            state=state,
            turns_taken=99,
            device="cpu",
            config=cfg,
            rng=random.Random(1),
        )

        restored = self.env.restore_snapshot(snap)
        result_b = run_mcts(
            self.env,
            self.model,
            state=restored,
            turns_taken=99,
            device="cpu",
            config=cfg,
            rng=random.Random(2),
        )

        self.env.drop_snapshot(snap)

        self.assertTrue(np.isfinite(result_a.visit_probs).all())
        self.assertTrue(np.isfinite(result_b.visit_probs).all())
        self.assertAlmostEqual(float(result_a.visit_probs.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(result_b.visit_probs.sum()), 1.0, places=5)
        self.assertTrue(bool(state.mask[int(result_a.action)]))
        self.assertTrue(bool(state.mask[int(result_b.action)]))

        legal = np.flatnonzero(state.mask)
        l1_diff = float(np.sum(np.abs(result_a.visit_probs[legal] - result_b.visit_probs[legal])))
        self.assertGreater(l1_diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
