#!/usr/bin/env python3
import os
import unittest
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

if np is not None:
    from nn.state_schema import ACTION_DIM, STATE_DIM
    from tests.codec_reference import encode_state
else:
    ACTION_DIM = 69
    STATE_DIM = 246
    encode_state = None

if np is not None and torch is not None:
    from nn.model import MaskedPolicyValueNet
    from nn.train import _model_sample_legal_action, masked_logits
else:
    MaskedPolicyValueNet = None
    _model_sample_legal_action = None
    masked_logits = None


REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from nn.native_env import SplendorNativeEnv as _SmokeEnv

    _ENV_AVAILABLE = True
except Exception:
    _SmokeEnv = None
    _ENV_AVAILABLE = False


class TestStateCodec(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy not installed")
    def test_encode_state_shape_and_finite(self):
        raw = [0] * STATE_DIM
        out = encode_state(raw)
        self.assertEqual(out.shape, (STATE_DIM,))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
class TestModelAndMask(unittest.TestCase):
    def test_model_output_shapes(self):
        model = MaskedPolicyValueNet()
        x = torch.zeros((4, STATE_DIM), dtype=torch.float32)
        logits, value = model(x)
        self.assertEqual(tuple(logits.shape), (4, ACTION_DIM))
        self.assertEqual(tuple(value.shape), (4,))

    def test_masked_logits_excludes_illegal_actions(self):
        logits = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
        mask = torch.zeros((1, ACTION_DIM), dtype=torch.bool)
        mask[0, [3, 12]] = True
        out = masked_logits(logits, mask)
        self.assertTrue(torch.isfinite(out[0, 3]))
        self.assertTrue(torch.isfinite(out[0, 12]))
        self.assertLess(float(out[0, 0].item()), -1e8)

    def test_model_sample_returns_legal_action(self):
        model = MaskedPolicyValueNet()
        state = np.zeros((STATE_DIM,), dtype=np.float32)
        mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        legal_idxs = [3, 12, 31, 60]
        mask[legal_idxs] = True
        for _ in range(20):
            action = _model_sample_legal_action(model, state, mask, device="cpu")
            self.assertIn(action, legal_idxs)


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
@unittest.skipIf(not _ENV_AVAILABLE, "native environment backend not available")
class TestIntegrationSmoke(unittest.TestCase):
    def test_one_train_step_smoke(self):
        from nn.train import run_smoke

        metrics = run_smoke(episodes=1, max_turns=20, batch_size=8, seed=123)
        self.assertGreaterEqual(metrics["replay_samples"], 1.0)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))
        self.assertTrue(np.isfinite(metrics["value_loss"]))
        self.assertEqual(metrics["legal_target_ok"], 1.0)
        self.assertEqual(metrics["collector_policy"], "random")
        self.assertGreater(metrics["collector_random_actions"], 0.0)
        self.assertEqual(metrics["collector_model_actions"], 0.0)
        self.assertEqual(metrics["collector_mcts_actions"], 0.0)

    def test_one_train_step_smoke_model_sample(self):
        from nn.train import run_smoke

        metrics = run_smoke(episodes=1, max_turns=20, batch_size=8, seed=123, collector_policy="model-sample")
        self.assertGreaterEqual(metrics["replay_samples"], 1.0)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))
        self.assertTrue(np.isfinite(metrics["value_loss"]))
        self.assertEqual(metrics["legal_target_ok"], 1.0)
        self.assertEqual(metrics["collector_policy"], "model-sample")
        self.assertGreater(metrics["collector_model_actions"], 0.0)
        self.assertEqual(metrics["collector_mcts_actions"], 0.0)

    def test_run_cycles_tiny_smoke_random(self):
        from nn.train import run_cycles

        metrics = run_cycles(
            cycles=2,
            episodes_per_cycle=1,
            train_steps_per_cycle=1,
            max_turns=10,
            batch_size=4,
            collector_policy="random",
            seed=123,
        )
        self.assertEqual(metrics["mode"], "cycles")
        self.assertEqual(metrics["collector_policy"], "random")
        self.assertEqual(metrics["episodes"], 2.0)
        self.assertEqual(metrics["train_steps_per_cycle"], 1.0)
        self.assertEqual(metrics["collector_model_actions"], 0.0)
        self.assertEqual(metrics["collector_mcts_actions"], 0.0)
        self.assertGreaterEqual(metrics["collector_random_actions"], 1.0)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))
        self.assertTrue(np.isfinite(metrics["value_loss"]))
        self.assertEqual(metrics["legal_target_ok"], 1.0)


if __name__ == "__main__":
    unittest.main()
