import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  CheckpointDTO,
  EngineJobStatusDTO,
  EngineThinkResponse,
  GameSnapshotDTO,
  PlayerMoveResponse,
  Seat,
} from './types';
import { GameBoard } from './components/board/GameBoard';

type UiStatus = 'IDLE' | 'WAITING_ENGINE' | 'WAITING_PLAYER' | 'GAME_OVER';

const POLL_MS = 400;

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // Ignore parse errors and keep status text.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

function winnerLabel(winner: number): string {
  if (winner === -2) return 'In progress';
  if (winner === -1) return 'Draw';
  return winner === 0 ? 'Winner: P0' : 'Winner: P1';
}

export function App() {
  const [checkpoints, setCheckpoints] = useState<CheckpointDTO[]>([]);
  const [checkpointId, setCheckpointId] = useState('');
  const [numSimulations, setNumSimulations] = useState(400);
  const [playerSeat, setPlayerSeat] = useState<Seat>('P0');
  const [seed, setSeed] = useState('');

  const [snapshot, setSnapshot] = useState<GameSnapshotDTO | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<EngineJobStatusDTO | null>(null);
  const [uiStatus, setUiStatus] = useState<UiStatus>('IDLE');
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<number | null>(null);

  const selectedCheckpoint = useMemo(
    () => checkpoints.find((item) => item.id === checkpointId) ?? null,
    [checkpoints, checkpointId],
  );

  useEffect(() => {
    void (async () => {
      try {
        const list = await fetchJSON<CheckpointDTO[]>('/api/checkpoints');
        setCheckpoints(list);
        if (!checkpointId && list.length > 0) {
          setCheckpointId(list[0].id);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    })();
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
      }
    };
  }, []);

  function clearPolling(): void {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  function deriveUiStatus(nextSnapshot: GameSnapshotDTO): UiStatus {
    if (nextSnapshot.status !== 'IN_PROGRESS') {
      return 'GAME_OVER';
    }
    return nextSnapshot.player_to_move === nextSnapshot.config?.player_seat
      ? 'WAITING_PLAYER'
      : 'WAITING_ENGINE';
  }

  async function startEngineThink(): Promise<void> {
    setError(null);
    const think = await fetchJSON<EngineThinkResponse>('/api/game/engine-think', {
      method: 'POST',
      body: '{}',
    });
    setJobId(think.job_id);
    setUiStatus('WAITING_ENGINE');
    clearPolling();

    pollRef.current = window.setInterval(() => {
      void pollEngineJob(think.job_id);
    }, POLL_MS);
  }

  async function pollEngineJob(nextJobId: string): Promise<void> {
    try {
      const status = await fetchJSON<EngineJobStatusDTO>(`/api/game/engine-job/${nextJobId}`);
      setJobStatus(status);
      if (status.status === 'DONE') {
        clearPolling();
        const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/engine-apply', {
          method: 'POST',
          body: JSON.stringify({ job_id: nextJobId }),
        });
        setSnapshot(nextSnapshot);
        const nextUiStatus = deriveUiStatus(nextSnapshot);
        setUiStatus(nextUiStatus);
        if (nextUiStatus === 'WAITING_ENGINE') {
          await startEngineThink();
        }
      } else if (status.status === 'FAILED' || status.status === 'CANCELLED') {
        clearPolling();
        setUiStatus('WAITING_PLAYER');
      }
    } catch (err) {
      clearPolling();
      setError((err as Error).message);
      setUiStatus('WAITING_PLAYER');
    }
  }

  async function onStartGame(event: FormEvent): Promise<void> {
    event.preventDefault();
    setError(null);
    clearPolling();
    setJobId(null);
    setJobStatus(null);

    try {
      if (!checkpointId) {
        throw new Error('Please choose a checkpoint');
      }
      const payload = {
        checkpoint_id: checkpointId,
        num_simulations: Number(numSimulations),
        player_seat: playerSeat,
        ...(seed.trim().length > 0 ? { seed: Number(seed) } : {}),
      };

      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/new', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      setSnapshot(nextSnapshot);
      const status = deriveUiStatus(nextSnapshot);
      setUiStatus(status);
      if (status === 'WAITING_ENGINE') {
        await startEngineThink();
      }
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onPlayerMove(actionIdx: number): Promise<void> {
    setError(null);
    try {
      const result = await fetchJSON<PlayerMoveResponse>('/api/game/player-move', {
        method: 'POST',
        body: JSON.stringify({ action_idx: actionIdx }),
      });
      setSnapshot(result.snapshot);
      if (result.snapshot.status !== 'IN_PROGRESS') {
        setUiStatus('GAME_OVER');
        return;
      }
      if (result.engine_should_move) {
        await startEngineThink();
      } else {
        setUiStatus('WAITING_PLAYER');
      }
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function onResign(): Promise<void> {
    setError(null);
    clearPolling();
    try {
      const nextSnapshot = await fetchJSON<GameSnapshotDTO>('/api/game/resign', {
        method: 'POST',
        body: '{}',
      });
      setSnapshot(nextSnapshot);
      setUiStatus('GAME_OVER');
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const canStart = Boolean(selectedCheckpoint) && numSimulations > 0 && numSimulations <= 5000;
  const canMove = uiStatus === 'WAITING_PLAYER' && snapshot?.status === 'IN_PROGRESS';

  return (
    <main className="app-shell">
      <header>
        <h1>Splendor vs MCTS</h1>
      </header>

      <section className="panel">
        <h2>Setup</h2>
        <form onSubmit={(event) => void onStartGame(event)} className="grid-form">
          <label>
            Checkpoint
            <select
              value={checkpointId}
              onChange={(event) => setCheckpointId(event.target.value)}
              disabled={checkpoints.length === 0}
            >
              {checkpoints.length === 0 && <option value="">No checkpoints found</option>}
              {checkpoints.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            MCTS sims per move
            <input
              type="number"
              min={1}
              max={5000}
              value={numSimulations}
              onChange={(event) => setNumSimulations(Number(event.target.value))}
            />
          </label>

          <label>
            Play as
            <select value={playerSeat} onChange={(event) => setPlayerSeat(event.target.value as Seat)}>
              <option value="P0">P0 (first)</option>
              <option value="P1">P1 (second)</option>
            </select>
          </label>

          <label>
            Seed (optional)
            <input value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="Random if blank" />
          </label>

          <button type="submit" disabled={!canStart}>
            Start Game
          </button>
        </form>
      </section>

      {snapshot && (
        <section className="panel game-layout">
          <div className="board-column">
            <h2>Game Board</h2>
            {snapshot.board_state ? (
              <GameBoard board={snapshot.board_state} />
            ) : (
              <div className="empty-note">Board data unavailable</div>
            )}
          </div>

          <aside className="side-column">
            <h2>Game Controls</h2>
            <p>
              Status: <strong>{snapshot.status}</strong> | Winner: <strong>{winnerLabel(snapshot.winner)}</strong>
            </p>
            <p>
              Phase flags: return={String(snapshot.phase_flags.is_return_phase)} noble_choice={String(snapshot.phase_flags.is_noble_choice_phase)}
            </p>

            <div className="engine-box">
              <h3>Engine</h3>
              <p>UI status: {uiStatus}</p>
              {uiStatus === 'WAITING_ENGINE' && <p className="spinner">Engine thinking...</p>}
              {jobId && <p>Job ID: {jobId}</p>}
              {jobStatus && <p>Job status: {jobStatus.status}</p>}
              {jobStatus?.error && <p className="error">Engine error: {jobStatus.error}</p>}
              {uiStatus !== 'WAITING_ENGINE' &&
                snapshot.status === 'IN_PROGRESS' &&
                snapshot.player_to_move !== snapshot.config?.player_seat && (
                  <button onClick={() => void startEngineThink()}>Retry Engine Move</button>
                )}
            </div>

            <div className="actions">
              <h3>Legal actions</h3>
              <ul>
                {snapshot.legal_action_details.map((action) => (
                  <li key={action.action_idx}>
                    <button disabled={!canMove} onClick={() => void onPlayerMove(action.action_idx)}>
                      [{action.action_idx}] {action.label}
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            <div className="controls">
              <button onClick={() => void onResign()} disabled={snapshot.status !== 'IN_PROGRESS'}>
                Resign
              </button>
            </div>

            <div className="history">
              <h3>Move History</h3>
              <ol>
                {snapshot.move_log.map((entry) => (
                  <li key={`${entry.turn_index}-${entry.action_idx}`}>
                    {entry.turn_index}. {entry.actor} {'->'} [{entry.action_idx}] {entry.label}
                  </li>
                ))}
              </ol>
            </div>
          </aside>
        </section>
      )}

      {error && <section className="panel error">Error: {error}</section>}
    </main>
  );
}
