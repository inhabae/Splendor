#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest

import numpy as np

from nn.selfplay_dataset import (
    SelfPlaySession,
    SelfPlayStep,
    list_sessions,
    load_session_npz,
    save_session_npz,
)
from nn.state_schema import ACTION_DIM, STATE_DIM


class TestSelfPlayDataset(unittest.TestCase):
    def _make_step(self, *, episode_idx: int, step_idx: int, action: int, winner: int) -> SelfPlayStep:
        state = np.zeros((STATE_DIM,), dtype=np.float32)
        mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
        mask[action] = True
        policy = np.zeros((ACTION_DIM,), dtype=np.float32)
        policy[action] = 1.0
        return SelfPlayStep(
            state=state,
            mask=mask,
            policy=policy,
            value_target=1.0 if winner == 0 else 0.0,
            value_root=0.25,
            action_selected=action,
            episode_idx=episode_idx,
            step_idx=step_idx,
            turn_idx=step_idx,
            player_id=0,
            winner=winner,
            reached_cutoff=False,
            current_player_id=0,
        )

    def test_save_load_and_list_roundtrip(self) -> None:
        session = SelfPlaySession(
            session_id="selfplay_test",
            created_at="2026-02-27T00:00:00+00:00",
            metadata={
                "games": 1,
                "max_turns": 100,
                "num_simulations": 16,
                "seed_base": 123,
                "checkpoint_path": "/tmp/model.pt",
            },
            steps=[
                self._make_step(episode_idx=0, step_idx=0, action=3, winner=0),
                self._make_step(episode_idx=0, step_idx=1, action=4, winner=0),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = save_session_npz(session, tmp)
            self.assertTrue(out_path.exists())

            loaded = load_session_npz(out_path)
            self.assertEqual(loaded.session_id, session.session_id)
            self.assertEqual(len(loaded.steps), 2)
            self.assertEqual(loaded.steps[0].state.shape, (STATE_DIM,))
            self.assertEqual(loaded.steps[0].mask.shape, (ACTION_DIM,))
            self.assertEqual(loaded.steps[0].policy.shape, (ACTION_DIM,))
            self.assertTrue(bool(loaded.steps[0].mask[3]))
            self.assertAlmostEqual(float(loaded.steps[0].policy.sum()), 1.0, places=5)

            listed = list_sessions(tmp)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["session_id"], session.session_id)
            self.assertEqual(int(listed[0]["steps"]), 2)
            self.assertEqual(int(listed[0]["steps_per_episode"]["0"]), 2)


if __name__ == "__main__":
    unittest.main()
