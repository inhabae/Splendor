#!/usr/bin/env python3
import unittest

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

if np is not None and torch is not None:
    from nn.replay import ReplayBuffer, ReplaySample
    from nn.state_schema import ACTION_DIM, STATE_DIM
else:
    ReplayBuffer = None
    ReplaySample = None
    ACTION_DIM = 69
    STATE_DIM = 246


@unittest.skipIf(np is None, "numpy not installed")
@unittest.skipIf(torch is None, "torch not installed")
class TestNNReplay(unittest.TestCase):
    def _valid_sample(self, action: int = 3, *, policy_target=None):
        state = np.zeros((STATE_DIM,), dtype=np.float32)
        mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        mask[action] = True
        if policy_target is None:
            policy_target = np.zeros((ACTION_DIM,), dtype=np.float32)
            policy_target[action] = 1.0
        return ReplaySample(state=state, mask=mask, action_target=action, value_target=1.0, policy_target=policy_target)

    def test_add_valid_sample_succeeds(self):
        buf = ReplayBuffer()
        buf.add(self._valid_sample())
        self.assertEqual(len(buf), 1)

    def test_reject_invalid_state_shape(self):
        buf = ReplayBuffer()
        s = self._valid_sample()
        s.state = np.zeros((STATE_DIM - 1,), dtype=np.float32)
        with self.assertRaises(ValueError):
            buf.add(s)

    def test_reject_invalid_mask_shape(self):
        buf = ReplayBuffer()
        s = self._valid_sample()
        s.mask = np.zeros((ACTION_DIM - 1,), dtype=np.bool_)
        with self.assertRaises(ValueError):
            buf.add(s)

    def test_reject_illegal_target_action(self):
        buf = ReplayBuffer()
        s = self._valid_sample()
        s.mask[:] = False
        with self.assertRaises(ValueError):
            buf.add(s)

    def test_reject_negative_action_target_regression(self):
        buf = ReplayBuffer()
        s = self._valid_sample(action=68)
        s.action_target = -1
        # Regression: numpy negative indexing used to allow this.
        with self.assertRaises(ValueError):
            buf.add(s)

    def test_reject_out_of_range_action_target(self):
        buf = ReplayBuffer()
        s = self._valid_sample(action=3)
        s.action_target = ACTION_DIM
        with self.assertRaises(ValueError):
            buf.add(s)

    def test_sample_batch_on_empty_raises(self):
        with self.assertRaises(ValueError):
            ReplayBuffer().sample_batch(1)

    def test_sample_batch_invalid_batch_size_raises(self):
        buf = ReplayBuffer()
        buf.add(self._valid_sample())
        with self.assertRaises(ValueError):
            buf.sample_batch(0)

    def test_sample_batch_shapes_and_dtypes(self):
        buf = ReplayBuffer()
        for action in [1, 2, 3]:
            buf.add(self._valid_sample(action=action))
        batch = buf.sample_batch(2, device="cpu")
        self.assertEqual(tuple(batch["state"].shape), (2, STATE_DIM))
        self.assertEqual(tuple(batch["mask"].shape), (2, ACTION_DIM))
        self.assertEqual(tuple(batch["action_target"].shape), (2,))
        self.assertEqual(tuple(batch["policy_target"].shape), (2, ACTION_DIM))
        self.assertEqual(tuple(batch["value_target"].shape), (2,))
        self.assertEqual(batch["state"].dtype, torch.float32)
        self.assertEqual(batch["mask"].dtype, torch.bool)
        self.assertEqual(batch["action_target"].dtype, torch.long)
        self.assertEqual(batch["policy_target"].dtype, torch.float32)
        self.assertEqual(batch["value_target"].dtype, torch.float32)

    def test_add_accepts_soft_policy_target_over_legal_actions(self):
        buf = ReplayBuffer()
        mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        mask[[3, 4]] = True
        policy = np.zeros((ACTION_DIM,), dtype=np.float32)
        policy[3] = 0.25
        policy[4] = 0.75
        sample = ReplaySample(
            state=np.zeros((STATE_DIM,), dtype=np.float32),
            mask=mask,
            action_target=4,
            value_target=1.0,
            policy_target=policy,
        )
        buf.add(sample)
        self.assertEqual(len(buf), 1)

    def test_reject_policy_target_illegal_mass(self):
        buf = ReplayBuffer()
        policy = np.zeros((ACTION_DIM,), dtype=np.float32)
        policy[3] = 0.5
        policy[4] = 0.5
        sample = self._valid_sample(action=3, policy_target=policy)
        with self.assertRaises(ValueError):
            buf.add(sample)

    def test_reject_policy_target_not_normalized(self):
        buf = ReplayBuffer()
        policy = np.zeros((ACTION_DIM,), dtype=np.float32)
        policy[3] = 0.8
        sample = self._valid_sample(action=3, policy_target=policy)
        with self.assertRaises(ValueError):
            buf.add(sample)


if __name__ == "__main__":
    unittest.main()
