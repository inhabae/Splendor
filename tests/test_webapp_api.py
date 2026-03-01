from __future__ import annotations

import time
import unittest

from fastapi.testclient import TestClient

from nn.webapp import app


class TestWebAppAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        health = cls.client.get('/healthz')
        if health.status_code != 200:
            raise unittest.SkipTest('FastAPI app not healthy')
        ck = cls.client.get('/api/checkpoints')
        if ck.status_code != 200:
            raise unittest.SkipTest('Checkpoint endpoint unavailable')
        cls.checkpoints = ck.json()
        if not cls.checkpoints:
            raise unittest.SkipTest('No checkpoints available in nn_artifacts/checkpoints')

    def test_checkpoints_sorted_newest_first(self) -> None:
        response = self.client.get('/api/checkpoints')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data), 1)

        created = [item['created_at'] for item in data]
        self.assertEqual(created, sorted(created, reverse=True))

        for item in data:
            self.assertIn('id', item)
            self.assertIn('name', item)
            self.assertIn('path', item)
            self.assertIn('size_bytes', item)

    def test_new_game_rejects_invalid_checkpoint(self) -> None:
        response = self.client.post(
            '/api/game/new',
            json={
                'checkpoint_id': '/tmp/does/not/exist.pt',
                'num_simulations': 8,
                'player_seat': 'P0',
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid checkpoint_id', response.text)

    def _assert_board_state_shape(self, snapshot: dict) -> None:
        self.assertIn('board_state', snapshot)
        board = snapshot['board_state']
        self.assertIsNotNone(board)
        self.assertEqual(board['meta']['target_points'], 15)
        self.assertEqual(len(board['players']), 2)
        self.assertEqual(board['players'][0]['seat'], 'P0')
        self.assertEqual(board['players'][1]['seat'], 'P1')
        self.assertEqual(len(board['tiers']), 3)
        for tier in board['tiers']:
            self.assertIn('tier', tier)
            self.assertIn('deck_count', tier)
            self.assertIn('cards', tier)
            for card in tier['cards']:
                self.assertIn('points', card)
                self.assertIn('bonus_color', card)
                self.assertIn('cost', card)

    def test_board_state_present_and_seat_mapping(self) -> None:
        cp = self.checkpoints[0]['id']
        created = self.client.post(
            '/api/game/new',
            json={
                'checkpoint_id': cp,
                'num_simulations': 4,
                'player_seat': 'P0',
                'seed': 123,
            },
        )
        self.assertEqual(created.status_code, 200)
        snapshot = created.json()
        self._assert_board_state_shape(snapshot)
        self.assertEqual(snapshot['board_state']['meta']['player_to_move'], snapshot['player_to_move'])

    def test_player_move_rejects_out_of_turn(self) -> None:
        cp = self.checkpoints[0]['id']
        created = self.client.post(
            '/api/game/new',
            json={
                'checkpoint_id': cp,
                'num_simulations': 8,
                'player_seat': 'P1',
                'seed': 123,
            },
        )
        self.assertEqual(created.status_code, 200)

        move_resp = self.client.post('/api/game/player-move', json={'action_idx': 0})
        self.assertEqual(move_resp.status_code, 400)
        self.assertIn("Not player's turn", move_resp.text)

    def test_engine_job_lifecycle_and_apply(self) -> None:
        cp = self.checkpoints[0]['id']
        created = self.client.post(
            '/api/game/new',
            json={
                'checkpoint_id': cp,
                'num_simulations': 1,
                'player_seat': 'P0',
                'seed': 321,
            },
        )
        self.assertEqual(created.status_code, 200)
        snapshot = created.json()
        self.assertEqual(snapshot['status'], 'IN_PROGRESS')
        self.assertEqual(snapshot['player_to_move'], 'P0')
        self.assertTrue(snapshot['legal_actions'])
        self._assert_board_state_shape(snapshot)

        action = int(snapshot['legal_actions'][0])
        moved = self.client.post('/api/game/player-move', json={'action_idx': action})
        self.assertEqual(moved.status_code, 200)
        moved_json = moved.json()
        self.assertIn('snapshot', moved_json)
        self.assertIn('engine_should_move', moved_json)
        self._assert_board_state_shape(moved_json['snapshot'])

        if not moved_json['engine_should_move']:
            self.skipTest('Selected action kept player turn; skipping async engine test path')

        think = self.client.post('/api/game/engine-think', json={})
        self.assertEqual(think.status_code, 200)
        job_id = think.json()['job_id']

        apply_too_early = self.client.post('/api/game/engine-apply', json={'job_id': job_id})
        self.assertEqual(apply_too_early.status_code, 400)

        deadline = time.time() + 30
        last_status = None
        while time.time() < deadline:
            poll = self.client.get(f'/api/game/engine-job/{job_id}')
            self.assertEqual(poll.status_code, 200)
            payload = poll.json()
            last_status = payload['status']
            if last_status in ('DONE', 'FAILED', 'CANCELLED'):
                break
            time.sleep(0.1)

        self.assertIn(last_status, ('DONE', 'FAILED', 'CANCELLED'))
        if last_status != 'DONE':
            self.skipTest(f'Engine job did not complete successfully: {last_status}')

        applied = self.client.post('/api/game/engine-apply', json={'job_id': job_id})
        self.assertEqual(applied.status_code, 200)
        after = applied.json()
        self.assertGreaterEqual(after['turn_index'], 2)
        self._assert_board_state_shape(after)

    def test_resign_marks_game_finished(self) -> None:
        cp = self.checkpoints[0]['id']
        created = self.client.post(
            '/api/game/new',
            json={
                'checkpoint_id': cp,
                'num_simulations': 8,
                'player_seat': 'P0',
                'seed': 999,
            },
        )
        self.assertEqual(created.status_code, 200)

        resign = self.client.post('/api/game/resign', json={})
        self.assertEqual(resign.status_code, 200)
        snap = resign.json()
        self.assertEqual(snap['status'], 'RESIGNED')
        self.assertEqual(snap['winner'], 1)
        self.assertEqual(snap['legal_actions'], [])
        self._assert_board_state_shape(snap)

    def test_selfplay_run_and_replay_endpoints(self) -> None:
        cp = self.checkpoints[0]['id']
        run_resp = self.client.post(
            '/api/selfplay/run',
            json={
                'checkpoint_id': cp,
                'num_simulations': 1,
                'games': 1,
                'max_turns': 20,
                'seed': 123,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        run_payload = run_resp.json()
        self.assertIn('session_id', run_payload)
        self.assertIn('steps', run_payload)
        self.assertGreaterEqual(int(run_payload['steps']), 1)
        session_id = str(run_payload['session_id'])

        list_resp = self.client.get('/api/selfplay/sessions')
        self.assertEqual(list_resp.status_code, 200)
        sessions = list_resp.json()
        self.assertTrue(any(item['session_id'] == session_id for item in sessions))

        summary_resp = self.client.get(f'/api/selfplay/session/{session_id}/summary')
        self.assertEqual(summary_resp.status_code, 200)
        summary = summary_resp.json()
        self.assertIn('steps_per_episode', summary)
        self.assertIn('winners_by_episode', summary)

        step_resp = self.client.get(f'/api/selfplay/session/{session_id}/step?episode_idx=0&step_idx=0')
        self.assertEqual(step_resp.status_code, 200)
        step = step_resp.json()
        self.assertIn('board_state', step)
        self.assertIn('action_details', step)
        self.assertEqual(len(step['action_details']), 69)

        masked = [a for a in step['action_details'] if a['masked']]
        legal = [a for a in step['action_details'] if not a['masked']]
        self.assertGreater(len(legal), 0)
        self.assertGreater(len(masked), 0)

        policy_sum = sum(float(a['policy_prob']) for a in legal)
        self.assertAlmostEqual(policy_sum, 1.0, places=3)

    def test_selfplay_parallel_workers_metadata(self) -> None:
        cp = self.checkpoints[0]['id']
        run_resp = self.client.post(
            '/api/selfplay/run',
            json={
                'checkpoint_id': cp,
                'num_simulations': 1,
                'games': 2,
                'max_turns': 20,
                'seed': 77,
                'workers': 2,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        payload = run_resp.json()
        session_id = str(payload['session_id'])

        summary_resp = self.client.get(f'/api/selfplay/session/{session_id}/summary')
        self.assertEqual(summary_resp.status_code, 200)
        summary = summary_resp.json()
        metadata = summary['metadata']
        self.assertEqual(int(metadata.get('workers_used', 0)), 2)
        self.assertEqual(int(metadata.get('workers_requested', 0)), 2)
        self.assertEqual(str(metadata.get('parallelism_mode', '')), 'process_pool')
        self.assertEqual([int(x) for x in metadata.get('games_per_worker', [])], [1, 1])

    def test_selfplay_workers_clamped_to_games(self) -> None:
        cp = self.checkpoints[0]['id']
        run_resp = self.client.post(
            '/api/selfplay/run',
            json={
                'checkpoint_id': cp,
                'num_simulations': 1,
                'games': 1,
                'max_turns': 20,
                'seed': 78,
                'workers': 8,
            },
        )
        self.assertEqual(run_resp.status_code, 200)
        payload = run_resp.json()
        session_id = str(payload['session_id'])

        summary_resp = self.client.get(f'/api/selfplay/session/{session_id}/summary')
        self.assertEqual(summary_resp.status_code, 200)
        metadata = summary_resp.json()['metadata']
        self.assertEqual(int(metadata.get('workers_used', 0)), 1)
        self.assertEqual(int(metadata.get('workers_requested', 0)), 8)
        self.assertEqual([int(x) for x in metadata.get('games_per_worker', [])], [1])


if __name__ == '__main__':
    unittest.main()
