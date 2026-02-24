from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .bridge_env import StepState
from .state_codec import ACTION_DIM, STATE_DIM


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.25
    temperature_moves: int = 10
    temperature: float = 1.0
    eps: float = 1e-8


@dataclass
class MCTSResult:
    action: int
    visit_probs: np.ndarray  # (69,) float32
    root_value: float
    num_simulations: int
    root_total_visits: int
    root_nonzero_visit_actions: int
    root_legal_actions: int


@dataclass
class MCTSNode:
    snapshot_id: int
    state: np.ndarray
    mask: np.ndarray
    is_terminal: bool
    winner: int
    to_play_abs: int
    priors: np.ndarray = field(default_factory=lambda: np.zeros((ACTION_DIM,), dtype=np.float32))
    visit_count: np.ndarray = field(default_factory=lambda: np.zeros((ACTION_DIM,), dtype=np.int32))
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros((ACTION_DIM,), dtype=np.float32))
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    expanded: bool = False


def _winner_to_value_for_player(winner: int, player_id: int) -> float:
    if winner == -1:
        return 0.0
    if winner not in (0, 1):
        raise ValueError(f"Unexpected winner value {winner}")
    return 1.0 if winner == player_id else -1.0


def _evaluate_leaf(model: Any, node: MCTSNode, *, device: str) -> float:
    if node.is_terminal:
        return _winner_to_value_for_player(node.winner, node.to_play_abs)
    if node.expanded:
        raise RuntimeError("Leaf evaluation called on already-expanded node")
    if node.state.shape != (STATE_DIM,):
        raise ValueError(f"Unexpected state shape {node.state.shape}")
    if node.mask.shape != (ACTION_DIM,):
        raise ValueError(f"Unexpected mask shape {node.mask.shape}")
    if not bool(node.mask.any()):
        raise ValueError("Non-terminal MCTS node has no legal actions")

    state_t = torch.as_tensor(node.state[None, :], dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(node.mask[None, :], dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        logits, value_t = model(state_t)
        logits = logits.clone()
        logits[~mask_t] = -1e9
        priors = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        value = float(value_t.item())

    priors[~node.mask] = 0.0
    prior_sum = float(priors.sum())
    if prior_sum <= 0.0 or not np.isfinite(prior_sum):
        legal = np.flatnonzero(node.mask)
        priors[:] = 0.0
        priors[legal] = 1.0 / float(len(legal))
    else:
        priors /= prior_sum

    node.priors = priors
    node.expanded = True
    return value


def _select_puct_action(node: MCTSNode, *, c_puct: float, eps: float) -> int:
    legal = np.flatnonzero(node.mask)
    if legal.size == 0:
        raise RuntimeError("No legal actions to select in MCTS node")
    parent_n = float(node.visit_count.sum())
    sqrt_parent = math.sqrt(parent_n + eps)

    best_action = int(legal[0])
    best_score = -float("inf")
    for action in legal.tolist():
        n = float(node.visit_count[action])
        q = 0.0 if n <= 0.0 else float(node.value_sum[action] / n)
        u = c_puct * float(node.priors[action]) * sqrt_parent / (1.0 + n)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = int(action)
    return best_action


def _sample_action_from_visits(
    visit_probs: np.ndarray,
    legal_mask: np.ndarray,
    *,
    turns_taken: int,
    config: MCTSConfig,
    rng: random.Random | None,
) -> int:
    legal = np.flatnonzero(legal_mask)
    if legal.size == 0:
        raise RuntimeError("No legal actions for final MCTS action selection")
    if turns_taken >= config.temperature_moves:
        legal_visits = visit_probs[legal]
        return int(legal[int(np.argmax(legal_visits))])

    temp = float(config.temperature)
    base = visit_probs[legal].astype(np.float64, copy=False)
    if temp <= 0:
        return int(legal[int(np.argmax(base))])
    if temp != 1.0:
        base = np.power(base, 1.0 / temp)
    weight_sum = float(base.sum())
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        probs = np.full((legal.size,), 1.0 / float(legal.size), dtype=np.float64)
    else:
        probs = base / weight_sum

    py_rng = rng or random
    # random.Random.choices expects Python sequences.
    return int(py_rng.choices(legal.tolist(), weights=probs.tolist(), k=1)[0])


def run_mcts(
    env: Any,
    model: Any,
    state: StepState,
    *,
    turns_taken: int,
    device: str = "cpu",
    config: MCTSConfig | None = None,
    rng: random.Random | None = None,
) -> MCTSResult:
    cfg = config or MCTSConfig()
    if cfg.num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if state.is_terminal:
        raise ValueError("run_mcts called on terminal state")
    if state.state.shape != (STATE_DIM,):
        raise ValueError(f"Unexpected root state shape {state.state.shape}")
    if state.mask.shape != (ACTION_DIM,):
        raise ValueError(f"Unexpected root mask shape {state.mask.shape}")
    if not bool(state.mask.any()):
        raise ValueError("MCTS root has no legal actions")

    snapshot_ids: set[int] = set()
    root_snapshot_id = env.snapshot()
    snapshot_ids.add(root_snapshot_id)
    root = MCTSNode(
        snapshot_id=root_snapshot_id,
        state=state.state.copy(),
        mask=state.mask.copy(),
        is_terminal=state.is_terminal,
        winner=state.winner,
        to_play_abs=int(state.current_player_id),
    )

    try:
        for _ in range(cfg.num_simulations):
            node = root
            path: list[tuple[MCTSNode, int, bool]] = []

            while True:
                if node.is_terminal or not node.expanded:
                    value = _evaluate_leaf(model, node, device=device)
                    break

                action = _select_puct_action(node, c_puct=cfg.c_puct, eps=cfg.eps)
                child = node.children.get(action)
                if child is None:
                    env.restore_snapshot(node.snapshot_id)
                    child_state = env.step(int(action))
                    child_snapshot_id = env.snapshot()
                    snapshot_ids.add(child_snapshot_id)
                    child = MCTSNode(
                        snapshot_id=child_snapshot_id,
                        state=child_state.state.copy(),
                        mask=child_state.mask.copy(),
                        is_terminal=child_state.is_terminal,
                        winner=child_state.winner,
                        to_play_abs=int(child_state.current_player_id),
                    )
                    node.children[int(action)] = child
                same_player = (child.to_play_abs == node.to_play_abs)
                path.append((node, int(action), same_player))
                node = child

            for parent, action, same_player in reversed(path):
                backed_value = value if same_player else -value
                parent.visit_count[action] += 1
                parent.value_sum[action] += float(backed_value)
                value = backed_value

        visit_counts = root.visit_count.astype(np.float64, copy=False)
        total_visits = float(visit_counts.sum())
        visit_probs = np.zeros((ACTION_DIM,), dtype=np.float32)
        if total_visits > 0:
            visit_probs = (visit_counts / total_visits).astype(np.float32, copy=False)
        else:
            legal = np.flatnonzero(root.mask)
            visit_probs[legal] = 1.0 / float(len(legal))

        visit_probs[~root.mask] = 0.0
        prob_sum = float(visit_probs.sum())
        if prob_sum > 0:
            visit_probs /= prob_sum

        action = _sample_action_from_visits(
            visit_probs,
            root.mask,
            turns_taken=turns_taken,
            config=cfg,
            rng=rng,
        )

        legal = np.flatnonzero(root.mask)
        q_vals = []
        for a in legal.tolist():
            n = float(root.visit_count[a])
            q_vals.append(0.0 if n <= 0 else float(root.value_sum[a] / n))
        root_value = float(np.mean(q_vals)) if q_vals else 0.0

        return MCTSResult(
            action=action,
            visit_probs=visit_probs,
            root_value=root_value,
            num_simulations=cfg.num_simulations,
            root_total_visits=int(root.visit_count.sum()),
            root_nonzero_visit_actions=int(np.count_nonzero(root.visit_count)),
            root_legal_actions=int(np.count_nonzero(root.mask)),
        )
    finally:
        try:
            env.restore_snapshot(root_snapshot_id)
        except Exception:
            pass
        for sid in list(snapshot_ids):
            try:
                env.drop_snapshot(int(sid))
            except Exception:
                pass
