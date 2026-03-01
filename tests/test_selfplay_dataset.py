#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import Future
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nn.selfplay_dataset import (
    SelfPlaySession,
    SelfPlayStep,
    list_sessions,
    load_session_npz,
    run_selfplay_session_parallel,
    save_session_npz,
)
from nn.state_schema import ACTION_DIM, STATE_DIM


class _FakeProcessPoolExecutor:
    def __init__(self, *args, **kwargs) -> None:
        self._futures: list[Future] = []

    def __enter__(self) -> _FakeProcessPoolExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover
            fut.set_exception(exc)
        self._futures.append(fut)
        return fut


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

    def test_parallel_merge_determinism_for_episode_mapping(self) -> None:
        def fake_worker(
            *,
            worker_idx: int,
            checkpoint_path: str,
            games_for_worker: int,
            episode_start_idx: int,
            max_turns: int,
            num_simulations: int,
            seed_base: int,
        ) -> dict:
            steps: list[SelfPlayStep] = []
            for i in range(int(games_for_worker)):
                episode_idx = int(episode_start_idx + i)
                assigned_seed = int(seed_base + episode_idx)
                action = episode_idx % ACTION_DIM
                state = np.full((STATE_DIM,), float(episode_idx), dtype=np.float32)
                mask = np.zeros((ACTION_DIM,), dtype=np.bool_)
                mask[action] = True
                policy = np.zeros((ACTION_DIM,), dtype=np.float32)
                policy[action] = 1.0
                steps.append(
                    SelfPlayStep(
                        state=state,
                        mask=mask,
                        policy=policy,
                        value_target=0.0,
                        value_root=float(assigned_seed),
                        action_selected=action,
                        episode_idx=episode_idx,
                        step_idx=0,
                        turn_idx=0,
                        player_id=0,
                        winner=-1,
                        reached_cutoff=False,
                        current_player_id=0,
                    )
                )
            return {
                "worker_idx": int(worker_idx),
                "episode_start_idx": int(episode_start_idx),
                "games": int(games_for_worker),
                "steps": steps,
            }

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "fake.pt"
            ckpt.write_bytes(b"checkpoint")
            with patch("nn.selfplay_dataset.ProcessPoolExecutor", _FakeProcessPoolExecutor), patch(
                "nn.selfplay_dataset._run_selfplay_worker_task", side_effect=fake_worker
            ):
                session_one = run_selfplay_session_parallel(
                    checkpoint_path=ckpt,
                    games=4,
                    max_turns=10,
                    num_simulations=1,
                    seed_base=100,
                    workers=1,
                )
                session_two = run_selfplay_session_parallel(
                    checkpoint_path=ckpt,
                    games=4,
                    max_turns=10,
                    num_simulations=1,
                    seed_base=100,
                    workers=2,
                )

        episodes_one = [int(step.episode_idx) for step in session_one.steps]
        episodes_two = [int(step.episode_idx) for step in session_two.steps]
        self.assertEqual(episodes_one, [0, 1, 2, 3])
        self.assertEqual(episodes_two, [0, 1, 2, 3])

        seed_map_one = {int(step.episode_idx): int(step.value_root) for step in session_one.steps}
        seed_map_two = {int(step.episode_idx): int(step.value_root) for step in session_two.steps}
        self.assertEqual(seed_map_one, seed_map_two)
        self.assertEqual(seed_map_one, {0: 100, 1: 101, 2: 102, 3: 103})

        self.assertEqual([int(x) for x in session_one.metadata["games_per_worker"]], [4])
        self.assertEqual([int(x) for x in session_two.metadata["games_per_worker"]], [2, 2])
        self.assertEqual(int(session_two.metadata["workers_used"]), 2)
        self.assertEqual(str(session_two.metadata["parallelism_mode"]), "process_pool")

    def test_parallel_invalid_checkpoint_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_selfplay_session_parallel(
                checkpoint_path="/tmp/not/a/real/checkpoint.pt",
                games=2,
                max_turns=10,
                num_simulations=1,
                seed_base=1,
                workers=2,
            )


if __name__ == "__main__":
    unittest.main()
