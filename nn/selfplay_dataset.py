from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .mcts import MCTSConfig, run_mcts
from .native_env import SplendorNativeEnv
from .state_schema import ACTION_DIM, STATE_DIM


@dataclass
class SelfPlayStep:
    state: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    value_target: float
    value_root: float
    action_selected: int
    episode_idx: int
    step_idx: int
    turn_idx: int
    player_id: int
    winner: int
    reached_cutoff: bool
    current_player_id: int


@dataclass
class SelfPlaySession:
    session_id: str
    created_at: str
    metadata: dict[str, Any]
    steps: list[SelfPlayStep]

    @property
    def games(self) -> int:
        value = self.metadata.get("games", 0)
        return int(value)


def _winner_to_value_for_player(winner: int, player_id: int) -> float:
    if winner == -1:
        return 0.0
    return 1.0 if int(winner) == int(player_id) else -1.0


def run_selfplay_session(
    *,
    env: SplendorNativeEnv,
    model: Any,
    games: int,
    max_turns: int,
    num_simulations: int,
    seed_base: int,
) -> SelfPlaySession:
    if games <= 0:
        raise ValueError("games must be positive")
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")

    rng = random.Random(int(seed_base))
    created_at = datetime.now(timezone.utc).isoformat()
    session_id = f"selfplay_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    config = MCTSConfig(
        num_simulations=int(num_simulations),
        c_puct=1.25,
        temperature_moves=10,
        temperature=1.0,
        root_dirichlet_noise=False,
    )

    all_steps: list[SelfPlayStep] = []

    for episode_idx in range(games):
        seed = int(seed_base + episode_idx)
        state = env.reset(seed=seed)
        episode_steps: list[SelfPlayStep] = []
        winner = -1
        reached_cutoff = False
        turns_taken = 0

        while turns_taken < max_turns:
            if state.is_terminal:
                winner = int(state.winner)
                break
            if state.state.shape != (STATE_DIM,):
                raise RuntimeError(f"Unexpected state shape {state.state.shape}")
            if state.mask.shape != (ACTION_DIM,):
                raise RuntimeError(f"Unexpected mask shape {state.mask.shape}")
            if not bool(state.mask.any()):
                raise RuntimeError("Encountered non-terminal state with no legal actions")

            player_id = int(env.current_player_id)
            mcts_result = run_mcts(
                env,
                model,
                state,
                turns_taken=int(turns_taken),
                device="cpu",
                config=config,
                rng=rng,
            )

            action = int(mcts_result.chosen_action_idx)
            policy = np.asarray(mcts_result.visit_probs, dtype=np.float32)
            if policy.shape != (ACTION_DIM,):
                raise RuntimeError(f"Unexpected visit_probs shape {policy.shape}")
            if not bool(state.mask[action]):
                raise RuntimeError(f"MCTS produced illegal action {action}")

            episode_steps.append(
                SelfPlayStep(
                    state=state.state.copy(),
                    mask=state.mask.copy(),
                    policy=policy.copy(),
                    value_target=0.0,  # Filled after episode outcome is known.
                    value_root=float(mcts_result.root_value),
                    action_selected=action,
                    episode_idx=int(episode_idx),
                    step_idx=len(episode_steps),
                    turn_idx=int(turns_taken),
                    player_id=player_id,
                    winner=-1,
                    reached_cutoff=False,
                    current_player_id=int(state.current_player_id),
                )
            )

            prev_player_id = int(env.current_player_id)
            state = env.step(action)
            if int(env.current_player_id) != prev_player_id:
                turns_taken += 1
            if state.is_terminal:
                winner = int(state.winner)
                break
        else:
            reached_cutoff = True
            winner = -1

        if not reached_cutoff and not state.is_terminal:
            reached_cutoff = True
            winner = -1

        for step in episode_steps:
            step.value_target = _winner_to_value_for_player(winner, step.player_id)
            step.winner = int(winner)
            step.reached_cutoff = bool(reached_cutoff)
            all_steps.append(step)

    metadata = {
        "session_id": session_id,
        "created_at": created_at,
        "games": int(games),
        "max_turns": int(max_turns),
        "num_simulations": int(num_simulations),
        "seed_base": int(seed_base),
    }
    return SelfPlaySession(
        session_id=session_id,
        created_at=created_at,
        metadata=metadata,
        steps=all_steps,
    )


def save_session_npz(session: SelfPlaySession, out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{session.session_id}.npz"

    n = len(session.steps)
    states = np.zeros((n, STATE_DIM), dtype=np.float32)
    masks = np.zeros((n, ACTION_DIM), dtype=np.bool_)
    policies = np.zeros((n, ACTION_DIM), dtype=np.float32)
    value_target = np.zeros((n,), dtype=np.float32)
    value_root = np.zeros((n,), dtype=np.float32)
    action_selected = np.zeros((n,), dtype=np.int32)
    episode_idx = np.zeros((n,), dtype=np.int32)
    step_idx = np.zeros((n,), dtype=np.int32)
    turn_idx = np.zeros((n,), dtype=np.int32)
    player_id = np.zeros((n,), dtype=np.int32)
    winner = np.zeros((n,), dtype=np.int32)
    reached_cutoff = np.zeros((n,), dtype=np.bool_)
    current_player_id = np.zeros((n,), dtype=np.int32)

    for i, step in enumerate(session.steps):
        states[i] = step.state
        masks[i] = step.mask
        policies[i] = step.policy
        value_target[i] = float(step.value_target)
        value_root[i] = float(step.value_root)
        action_selected[i] = int(step.action_selected)
        episode_idx[i] = int(step.episode_idx)
        step_idx[i] = int(step.step_idx)
        turn_idx[i] = int(step.turn_idx)
        player_id[i] = int(step.player_id)
        winner[i] = int(step.winner)
        reached_cutoff[i] = bool(step.reached_cutoff)
        current_player_id[i] = int(step.current_player_id)

    metadata = dict(session.metadata)
    metadata["session_id"] = session.session_id
    metadata["created_at"] = session.created_at

    np.savez_compressed(
        file_path,
        metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
        state=states,
        mask=masks,
        policy=policies,
        value_target=value_target,
        value_root=value_root,
        action_selected=action_selected,
        episode_idx=episode_idx,
        step_idx=step_idx,
        turn_idx=turn_idx,
        player_id=player_id,
        winner=winner,
        reached_cutoff=reached_cutoff,
        current_player_id=current_player_id,
    )
    return file_path


def load_session_npz(path: str | Path) -> SelfPlaySession:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Session file not found: {data_path}")
    with np.load(data_path, allow_pickle=False) as npz:
        metadata_raw = npz["metadata_json"]
        if metadata_raw.ndim == 0:
            metadata_json = str(metadata_raw.item())
        else:
            metadata_json = str(metadata_raw.tolist())
        metadata = json.loads(metadata_json)

        states = np.asarray(npz["state"], dtype=np.float32)
        masks = np.asarray(npz["mask"], dtype=np.bool_)
        policies = np.asarray(npz["policy"], dtype=np.float32)
        value_target = np.asarray(npz["value_target"], dtype=np.float32)
        value_root = np.asarray(npz["value_root"], dtype=np.float32)
        action_selected = np.asarray(npz["action_selected"], dtype=np.int32)
        episode_idx = np.asarray(npz["episode_idx"], dtype=np.int32)
        step_idx = np.asarray(npz["step_idx"], dtype=np.int32)
        turn_idx = np.asarray(npz["turn_idx"], dtype=np.int32)
        player_id = np.asarray(npz["player_id"], dtype=np.int32)
        winner = np.asarray(npz["winner"], dtype=np.int32)
        reached_cutoff = np.asarray(npz["reached_cutoff"], dtype=np.bool_)
        current_player_id = np.asarray(npz["current_player_id"], dtype=np.int32)

    n = int(states.shape[0])
    steps: list[SelfPlayStep] = []
    for i in range(n):
        steps.append(
            SelfPlayStep(
                state=states[i].copy(),
                mask=masks[i].copy(),
                policy=policies[i].copy(),
                value_target=float(value_target[i]),
                value_root=float(value_root[i]),
                action_selected=int(action_selected[i]),
                episode_idx=int(episode_idx[i]),
                step_idx=int(step_idx[i]),
                turn_idx=int(turn_idx[i]),
                player_id=int(player_id[i]),
                winner=int(winner[i]),
                reached_cutoff=bool(reached_cutoff[i]),
                current_player_id=int(current_player_id[i]),
            )
        )

    created_at = str(metadata.get("created_at", ""))
    session_id = str(metadata.get("session_id", data_path.stem))
    return SelfPlaySession(
        session_id=session_id,
        created_at=created_at,
        metadata=metadata,
        steps=steps,
    )


def list_sessions(out_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for path in root.glob("*.npz"):
        try:
            session = load_session_npz(path)
        except Exception:
            continue
        by_episode: dict[int, int] = {}
        for step in session.steps:
            by_episode[step.episode_idx] = by_episode.get(step.episode_idx, 0) + 1
        checkpoint_path = str(session.metadata.get("checkpoint_path", ""))
        checkpoint_name = Path(checkpoint_path).name if checkpoint_path else "unknown.pt"
        seed_base = int(session.metadata.get("seed_base", 0))
        sims = int(session.metadata.get("num_simulations", 0))
        games = int(session.metadata.get("games", 0))
        display_name = f"{checkpoint_name}_seed{seed_base}_sims{sims}_games{games}"
        items.append(
            {
                "session_id": session.session_id,
                "display_name": display_name,
                "path": str(path.resolve()),
                "created_at": session.created_at,
                "games": int(session.metadata.get("games", 0)),
                "steps": int(len(session.steps)),
                "steps_per_episode": {str(k): int(v) for k, v in sorted(by_episode.items())},
                "metadata": session.metadata,
            }
        )
    items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return items
