from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .bridge_env import SplendorBridgeEnv, StepState
from .mcts import MCTSConfig, run_mcts
from .model import MaskedPolicyValueNet
from .replay import ReplayBuffer, ReplaySample
from .state_codec import ACTION_DIM, STATE_DIM


MASK_FILL_VALUE = -1e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"logits shape {logits.shape} must match mask shape {mask.shape}")
    if logits.ndim != 2 or logits.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected logits/mask shape (B,{ACTION_DIM}), got {logits.shape}")
    if not mask.dtype == torch.bool:
        raise ValueError("mask must be torch.bool")
    if not mask.any(dim=1).all():
        raise ValueError("Each sample must have at least one legal action")
    return logits.masked_fill(~mask, MASK_FILL_VALUE)


def masked_cross_entropy_loss(logits: torch.Tensor, mask: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    if target_idx.ndim != 1:
        raise ValueError(f"target_idx must be shape (B,), got {target_idx.shape}")
    if target_idx.shape[0] != logits.shape[0]:
        raise ValueError("target_idx batch size mismatch")
    row_idx = torch.arange(target_idx.shape[0], device=target_idx.device)
    if not mask[row_idx, target_idx].all():
        raise ValueError("All target actions must be legal under mask")
    return F.cross_entropy(masked_logits(logits, mask), target_idx)


def masked_soft_cross_entropy_loss(logits: torch.Tensor, mask: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    if target_probs.shape != logits.shape:
        raise ValueError(f"target_probs shape {target_probs.shape} must match logits shape {logits.shape}")
    if target_probs.ndim != 2 or target_probs.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected target_probs shape (B,{ACTION_DIM}), got {target_probs.shape}")
    if not torch.isfinite(target_probs).all():
        raise ValueError("target_probs contains non-finite values")
    if (target_probs < 0).any():
        raise ValueError("target_probs cannot contain negative values")
    if (target_probs[~mask] != 0).any():
        raise ValueError("target_probs must assign zero probability to illegal actions")
    row_sums = target_probs.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0.0):
        raise ValueError("Each target_probs row must sum to 1")
    masked = masked_logits(logits, mask)
    log_probs = F.log_softmax(masked, dim=-1)
    return -(target_probs * log_probs).sum(dim=1).mean()


def select_masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.argmax(masked_logits(logits, mask), dim=-1)


def select_masked_sample(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = masked_logits(logits, mask)
    return torch.distributions.Categorical(logits=masked).sample()


def _uniform_random_legal_action(mask: np.ndarray, rng: random.Random) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("Mask contains no legal actions")
    return int(rng.choice(legal.tolist()))


def _model_sample_legal_action(
    model: MaskedPolicyValueNet,
    state_np: np.ndarray,
    mask_np: np.ndarray,
    *,
    device: str,
) -> int:
    if state_np.shape != (STATE_DIM,):
        raise ValueError(f"Expected state shape ({STATE_DIM},), got {state_np.shape}")
    if mask_np.shape != (ACTION_DIM,):
        raise ValueError(f"Expected mask shape ({ACTION_DIM},), got {mask_np.shape}")
    if not bool(mask_np.any()):
        raise ValueError("Model-sample collector requires at least one legal action")

    state_t = torch.as_tensor(state_np[None, :], dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_np[None, :], dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        logits, _value = model(state_t)
        action_t = select_masked_sample(logits, mask_t)

    action = int(action_t.item())
    if not (0 <= action < ACTION_DIM):
        raise RuntimeError(f"Model-sampled action out of range: {action}")
    if not bool(mask_np[action]):
        raise RuntimeError(f"Model-sampled illegal action {action}")
    return action


@dataclass
class _EpisodeStep:
    state: np.ndarray
    mask: np.ndarray
    action_target: int
    policy_target: np.ndarray
    player_id: int


@dataclass
class EpisodeSummary:
    num_steps: int
    num_turns: int
    reached_cutoff: bool
    winner: int  # -1 draw, 0/1 winner


def _winner_to_value_for_player(winner: int, player_id: int) -> float:
    if winner == -1:
        return 0.0
    if winner not in (0, 1):
        raise ValueError(f"Unexpected winner value {winner}")
    return 1.0 if winner == player_id else -1.0


@dataclass
class CollectorStats:
    random_actions: int = 0
    model_actions: int = 0
    mcts_actions: int = 0
    mcts_sum_search_seconds: float = 0.0
    mcts_sum_root_entropy: float = 0.0
    mcts_sum_root_top1_prob: float = 0.0
    mcts_sum_selected_visit_prob: float = 0.0
    mcts_sum_root_value: float = 0.0
    mcts_sum_root_total_visits: float = 0.0
    mcts_sum_root_nonzero_visit_actions: float = 0.0
    mcts_sum_root_legal_actions: float = 0.0


def _policy_entropy(policy: np.ndarray, mask: np.ndarray) -> float:
    if policy.shape != (ACTION_DIM,) or mask.shape != (ACTION_DIM,):
        raise ValueError("Unexpected policy/mask shape for entropy")
    p = policy[mask].astype(np.float64, copy=False)
    if p.size == 0:
        return 0.0
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _validate_collector_policy(collector_policy: str) -> None:
    if collector_policy not in ("random", "model-sample", "mcts"):
        raise ValueError(f"Unsupported collector_policy: {collector_policy}")


def _one_hot_policy_target(mask: np.ndarray, action: int) -> np.ndarray:
    if mask.shape != (ACTION_DIM,):
        raise ValueError(f"Expected mask shape ({ACTION_DIM},), got {mask.shape}")
    if not (0 <= int(action) < ACTION_DIM):
        raise ValueError(f"Action out of range for one-hot policy target: {action}")
    if not bool(mask[int(action)]):
        raise ValueError("One-hot policy target action must be legal")
    target = np.zeros((ACTION_DIM,), dtype=np.float32)
    target[int(action)] = 1.0
    return target


def collect_episode(
    env: SplendorBridgeEnv,
    replay: ReplayBuffer,
    *,
    seed: int,
    max_turns: int,
    rng: random.Random,
    collector_policy: str,
    model: Optional[MaskedPolicyValueNet] = None,
    device: str = "cpu",
    collector_stats: Optional[CollectorStats] = None,
    mcts_config: Optional[MCTSConfig] = None,
) -> EpisodeSummary:
    state = env.reset(seed=seed)
    episode_steps: List[_EpisodeStep] = []

    reached_cutoff = False
    winner = -1
    turns_taken = 0

    while turns_taken < max_turns:
        if state.is_terminal:
            winner = state.winner
            break

        if state.state.shape != (STATE_DIM,):
            raise AssertionError(f"State shape mismatch: {state.state.shape}")
        if state.mask.shape != (ACTION_DIM,):
            raise AssertionError(f"Mask shape mismatch: {state.mask.shape}")
        if not state.mask.any():
            raise AssertionError("No legal actions in non-terminal state")

        player_id = env.current_player_id
        if collector_policy == "random":
            action = _uniform_random_legal_action(state.mask, rng)
            policy_target = _one_hot_policy_target(state.mask, action)
            if collector_stats is not None:
                collector_stats.random_actions += 1
        elif collector_policy == "model-sample":
            if model is None:
                raise ValueError("collector_policy='model-sample' requires a model")
            action = _model_sample_legal_action(model, state.state, state.mask, device=device)
            policy_target = _one_hot_policy_target(state.mask, action)
            if collector_stats is not None:
                collector_stats.model_actions += 1
        elif collector_policy == "mcts":
            if model is None:
                raise ValueError("collector_policy='mcts' requires a model")
            t0 = time.perf_counter()
            mcts_result = run_mcts(
                env,
                model,
                state,
                turns_taken=turns_taken,
                device=device,
                config=mcts_config,
                rng=rng,
            )
            elapsed = time.perf_counter() - t0
            action = int(mcts_result.action)
            policy_target = mcts_result.visit_probs.astype(np.float32, copy=True)
            if collector_stats is not None:
                collector_stats.mcts_actions += 1
                collector_stats.mcts_sum_search_seconds += float(elapsed)
                collector_stats.mcts_sum_root_entropy += _policy_entropy(policy_target, state.mask)
                legal_probs = policy_target[state.mask]
                collector_stats.mcts_sum_root_top1_prob += float(np.max(legal_probs)) if legal_probs.size > 0 else 0.0
                collector_stats.mcts_sum_selected_visit_prob += float(policy_target[action])
                collector_stats.mcts_sum_root_value += float(mcts_result.root_value)
                collector_stats.mcts_sum_root_total_visits += float(mcts_result.root_total_visits)
                collector_stats.mcts_sum_root_nonzero_visit_actions += float(mcts_result.root_nonzero_visit_actions)
                collector_stats.mcts_sum_root_legal_actions += float(mcts_result.root_legal_actions)
        else:
            raise ValueError(f"Unknown collector_policy: {collector_policy}")
        if not bool(state.mask[action]):
            raise AssertionError("Sampled action is not legal")

        prev_player_id = env.current_player_id
        episode_steps.append(
            _EpisodeStep(
                state=state.state.copy(),
                mask=state.mask.copy(),
                action_target=action,
                policy_target=policy_target.copy(),
                player_id=player_id,
            )
        )
        state = env.step(action)
        if env.current_player_id != prev_player_id:
            turns_taken += 1
        if state.is_terminal:
            winner = state.winner
            break
    else:
        reached_cutoff = True
        winner = -1

    if not reached_cutoff and not state.is_terminal:
        # Defensive fallback for unusual protocol behavior.
        winner = -1
        reached_cutoff = True

    for step in episode_steps:
        replay.add(
            ReplaySample(
                state=step.state,
                mask=step.mask,
                action_target=step.action_target,
                value_target=_winner_to_value_for_player(winner, step.player_id),
                policy_target=step.policy_target,
            )
        )

    return EpisodeSummary(
        num_steps=len(episode_steps),
        num_turns=turns_taken,
        reached_cutoff=reached_cutoff,
        winner=winner,
    )


def _collect_replay(
    env: SplendorBridgeEnv,
    replay: ReplayBuffer,
    *,
    episodes: int,
    max_turns: int,
    rng: random.Random,
    collector_policy: str,
    model: MaskedPolicyValueNet,
    device: str,
    seed_start: int,
    mcts_config: Optional[MCTSConfig] = None,
) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    _validate_collector_policy(collector_policy)

    cutoff_count = 0
    total_steps = 0
    total_turns = 0
    terminal_episodes = 0
    collector_stats = CollectorStats()
    next_seed = seed_start

    for _ in range(episodes):
        summary = collect_episode(
            env,
            replay,
            seed=next_seed,
            max_turns=max_turns,
            rng=rng,
            collector_policy=collector_policy,
            model=model,
            device=device,
            collector_stats=collector_stats,
            mcts_config=mcts_config,
        )
        next_seed += 1
        total_steps += summary.num_steps
        total_turns += summary.num_turns
        if summary.reached_cutoff:
            cutoff_count += 1
        else:
            terminal_episodes += 1

    mcts_n = max(collector_stats.mcts_actions, 1)
    has_mcts = collector_stats.mcts_actions > 0

    return {
        "episodes": float(episodes),
        "terminal_episodes": float(terminal_episodes),
        "cutoff_episodes": float(cutoff_count),
        "replay_samples": float(len(replay)),
        "total_steps": float(total_steps),
        "total_turns": float(total_turns),
        "collector_random_actions": float(collector_stats.random_actions),
        "collector_model_actions": float(collector_stats.model_actions),
        "collector_mcts_actions": float(collector_stats.mcts_actions),
        "mcts_avg_search_ms": (1000.0 * collector_stats.mcts_sum_search_seconds / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_entropy": (collector_stats.mcts_sum_root_entropy / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_top1_visit_prob": (collector_stats.mcts_sum_root_top1_prob / mcts_n) if has_mcts else 0.0,
        "mcts_avg_selected_visit_prob": (collector_stats.mcts_sum_selected_visit_prob / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_value": (collector_stats.mcts_sum_root_value / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_total_visits": (collector_stats.mcts_sum_root_total_visits / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_nonzero_visit_actions": (collector_stats.mcts_sum_root_nonzero_visit_actions / mcts_n) if has_mcts else 0.0,
        "mcts_avg_root_legal_actions": (collector_stats.mcts_sum_root_legal_actions / mcts_n) if has_mcts else 0.0,
        "next_seed": int(next_seed),
    }


def train_one_step(
    model: MaskedPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    value_loss_weight: float = 1.0,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    states = batch["state"]
    masks = batch["mask"]
    action_target = batch["action_target"]
    policy_target = batch["policy_target"]
    value_target = batch["value_target"]

    logits, value_pred = model(states)
    if not torch.isfinite(logits).all() or not torch.isfinite(value_pred).all():
        raise RuntimeError("Model produced non-finite outputs")

    policy_loss = masked_soft_cross_entropy_loss(logits, masks, policy_target)
    value_loss = F.mse_loss(value_pred, value_target)
    total_loss = policy_loss + value_loss_weight * value_loss

    if not torch.isfinite(total_loss):
        raise RuntimeError("Non-finite total loss")

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    with torch.no_grad():
        row_idx = torch.arange(action_target.shape[0], device=action_target.device)
        legal_target_ok = bool(masks[row_idx, action_target].all().item())

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "total_loss": float(total_loss.item()),
        "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
        "legal_target_ok": 1.0 if legal_target_ok else 0.0,
    }


def _train_on_replay(
    model: MaskedPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    *,
    batch_size: int,
    train_steps: int,
    log_every: int,
    device: str,
    value_loss_weight: float = 1.0,
    grad_clip_norm: float = 1.0,
    log_prefix: str = "",
) -> dict[str, object]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if len(replay) == 0:
        raise RuntimeError("Replay buffer is empty")

    metrics: dict[str, object] = {}
    sum_policy_loss = 0.0
    sum_value_loss = 0.0
    sum_total_loss = 0.0
    sum_grad_norm = 0.0

    for step in range(1, train_steps + 1):
        batch = replay.sample_batch(min(batch_size, len(replay)), device=device)
        metrics = train_one_step(
            model,
            optimizer,
            batch,
            value_loss_weight=value_loss_weight,
            grad_clip_norm=grad_clip_norm,
        )

        sum_policy_loss += float(metrics["policy_loss"])
        sum_value_loss += float(metrics["value_loss"])
        sum_total_loss += float(metrics["total_loss"])
        sum_grad_norm += float(metrics["grad_norm"])

        if step == 1 or step % log_every == 0 or step == train_steps:
            print(
                f"{log_prefix}train_step={step}/{train_steps} "
                f"policy_loss={metrics['policy_loss']:.6f} "
                f"value_loss={metrics['value_loss']:.6f} "
                f"total_loss={metrics['total_loss']:.6f} "
                f"grad_norm={metrics['grad_norm']:.6f}"
            )

    metrics.update(
        {
            "train_steps": float(train_steps),
            "avg_policy_loss": sum_policy_loss / train_steps,
            "avg_value_loss": sum_value_loss / train_steps,
            "avg_total_loss": sum_total_loss / train_steps,
            "avg_grad_norm": sum_grad_norm / train_steps,
        }
    )
    return metrics


def run_smoke(
    *,
    episodes: int = 5,
    max_turns: int = 80,
    batch_size: int = 32,
    collector_policy: str = "random",
    train_steps: int = 1,
    log_every: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    mcts_sims: int = 64,
    mcts_c_puct: float = 1.25,
    mcts_temperature_moves: int = 10,
    mcts_temperature: float = 1.0,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    _validate_collector_policy(collector_policy)

    replay = ReplayBuffer()
    rng = random.Random(seed)
    mcts_config = MCTSConfig(
        num_simulations=mcts_sims,
        c_puct=mcts_c_puct,
        temperature_moves=mcts_temperature_moves,
        temperature=mcts_temperature,
    )

    model = MaskedPolicyValueNet().to(device)

    with SplendorBridgeEnv() as env:
        collection_metrics = _collect_replay(
            env,
            replay,
            episodes=episodes,
            max_turns=max_turns,
            rng=rng,
            collector_policy=collector_policy,
            model=model,
            device=device,
            seed_start=seed,
            mcts_config=mcts_config,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    metrics = _train_on_replay(
        model,
        optimizer,
        replay,
        batch_size=batch_size,
        train_steps=train_steps,
        log_every=log_every,
        device=device,
    )

    metrics.update(
        {
            "mode": "smoke",
            "collector_policy": collector_policy,
            "collector_random_actions": collection_metrics["collector_random_actions"],
            "collector_model_actions": collection_metrics["collector_model_actions"],
            "collector_mcts_actions": collection_metrics["collector_mcts_actions"],
            "mcts_avg_search_ms": collection_metrics["mcts_avg_search_ms"],
            "mcts_avg_root_entropy": collection_metrics["mcts_avg_root_entropy"],
            "mcts_avg_root_top1_visit_prob": collection_metrics["mcts_avg_root_top1_visit_prob"],
            "mcts_avg_selected_visit_prob": collection_metrics["mcts_avg_selected_visit_prob"],
            "mcts_avg_root_value": collection_metrics["mcts_avg_root_value"],
            "mcts_avg_root_total_visits": collection_metrics["mcts_avg_root_total_visits"],
            "mcts_avg_root_nonzero_visit_actions": collection_metrics["mcts_avg_root_nonzero_visit_actions"],
            "mcts_avg_root_legal_actions": collection_metrics["mcts_avg_root_legal_actions"],
            "episodes": collection_metrics["episodes"],
            "terminal_episodes": collection_metrics["terminal_episodes"],
            "cutoff_episodes": collection_metrics["cutoff_episodes"],
            "replay_samples": collection_metrics["replay_samples"],
            "total_steps": collection_metrics["total_steps"],
            "total_turns": collection_metrics["total_turns"],
        }
    )
    return metrics


def run_cycles(
    *,
    cycles: int = 3,
    episodes_per_cycle: int = 5,
    train_steps_per_cycle: int = 50,
    max_turns: int = 80,
    batch_size: int = 32,
    collector_policy: str = "random",
    log_every: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    mcts_sims: int = 64,
    mcts_c_puct: float = 1.25,
    mcts_temperature_moves: int = 10,
    mcts_temperature: float = 1.0,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if episodes_per_cycle <= 0:
        raise ValueError("episodes_per_cycle must be positive")
    if train_steps_per_cycle <= 0:
        raise ValueError("train_steps_per_cycle must be positive")
    _validate_collector_policy(collector_policy)

    model = MaskedPolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    rng = random.Random(seed)
    next_episode_seed = seed
    mcts_config = MCTSConfig(
        num_simulations=mcts_sims,
        c_puct=mcts_c_puct,
        temperature_moves=mcts_temperature_moves,
        temperature=mcts_temperature,
    )

    total_episodes = 0.0
    total_terminal_episodes = 0.0
    total_cutoff_episodes = 0.0
    total_replay_samples = 0.0
    total_steps = 0.0
    total_turns = 0.0
    total_random_actions = 0.0
    total_model_actions = 0.0
    total_mcts_actions = 0.0
    weighted_sum_mcts_avg_search_ms = 0.0
    weighted_sum_mcts_avg_root_entropy = 0.0
    weighted_sum_mcts_avg_root_top1_visit_prob = 0.0
    weighted_sum_mcts_avg_selected_visit_prob = 0.0
    weighted_sum_mcts_avg_root_value = 0.0
    weighted_sum_mcts_avg_root_total_visits = 0.0
    weighted_sum_mcts_avg_root_nonzero_visit_actions = 0.0
    weighted_sum_mcts_avg_root_legal_actions = 0.0

    total_train_steps = 0.0
    weighted_sum_avg_policy_loss = 0.0
    weighted_sum_avg_value_loss = 0.0
    weighted_sum_avg_total_loss = 0.0
    weighted_sum_avg_grad_norm = 0.0

    last_train_metrics: dict[str, object] = {}

    with SplendorBridgeEnv() as env:
        for cycle_idx in range(1, cycles + 1):
            replay = ReplayBuffer()
            collection_metrics = _collect_replay(
                env,
                replay,
                episodes=episodes_per_cycle,
                max_turns=max_turns,
                rng=rng,
                collector_policy=collector_policy,
                model=model,
                device=device,
                seed_start=next_episode_seed,
                mcts_config=mcts_config,
            )
            next_episode_seed = int(collection_metrics["next_seed"])

            train_metrics = _train_on_replay(
                model,
                optimizer,
                replay,
                batch_size=batch_size,
                train_steps=train_steps_per_cycle,
                log_every=log_every,
                device=device,
                log_prefix=f"cycle={cycle_idx}/{cycles} ",
            )
            last_train_metrics = train_metrics

            print(
                f"cycle_summary={cycle_idx}/{cycles} "
                f"collector_policy={collector_policy} "
                f"replay_samples={collection_metrics['replay_samples']} "
                f"terminal_episodes={collection_metrics['terminal_episodes']} "
                f"cutoff_episodes={collection_metrics['cutoff_episodes']} "
                f"total_steps={collection_metrics['total_steps']} "
                f"total_turns={collection_metrics['total_turns']} "
                f"avg_policy_loss={train_metrics['avg_policy_loss']:.6f} "
                f"avg_value_loss={train_metrics['avg_value_loss']:.6f} "
                f"avg_total_loss={train_metrics['avg_total_loss']:.6f} "
                f"avg_grad_norm={train_metrics['avg_grad_norm']:.6f} "
                f"final_total_loss={train_metrics['total_loss']:.6f}"
            )
            if collector_policy == "mcts" and float(collection_metrics["collector_mcts_actions"]) > 0:
                print(
                    f"cycle_mcts={cycle_idx}/{cycles} "
                    f"avg_search_ms={float(collection_metrics['mcts_avg_search_ms']):.3f} "
                    f"avg_root_entropy={float(collection_metrics['mcts_avg_root_entropy']):.4f} "
                    f"avg_root_top1={float(collection_metrics['mcts_avg_root_top1_visit_prob']):.4f} "
                    f"avg_selected_visit={float(collection_metrics['mcts_avg_selected_visit_prob']):.4f} "
                    f"avg_root_value={float(collection_metrics['mcts_avg_root_value']):.4f} "
                    f"avg_root_visits={float(collection_metrics['mcts_avg_root_total_visits']):.2f} "
                    f"avg_root_nonzero_actions={float(collection_metrics['mcts_avg_root_nonzero_visit_actions']):.2f} "
                    f"avg_root_legal_actions={float(collection_metrics['mcts_avg_root_legal_actions']):.2f}"
                )

            total_episodes += float(collection_metrics["episodes"])
            total_terminal_episodes += float(collection_metrics["terminal_episodes"])
            total_cutoff_episodes += float(collection_metrics["cutoff_episodes"])
            total_replay_samples += float(collection_metrics["replay_samples"])
            total_steps += float(collection_metrics["total_steps"])
            total_turns += float(collection_metrics["total_turns"])
            total_random_actions += float(collection_metrics["collector_random_actions"])
            total_model_actions += float(collection_metrics["collector_model_actions"])
            cycle_mcts_actions = float(collection_metrics["collector_mcts_actions"])
            total_mcts_actions += cycle_mcts_actions
            if cycle_mcts_actions > 0:
                weighted_sum_mcts_avg_search_ms += float(collection_metrics["mcts_avg_search_ms"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_entropy += float(collection_metrics["mcts_avg_root_entropy"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_top1_visit_prob += float(collection_metrics["mcts_avg_root_top1_visit_prob"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_selected_visit_prob += float(collection_metrics["mcts_avg_selected_visit_prob"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_value += float(collection_metrics["mcts_avg_root_value"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_total_visits += float(collection_metrics["mcts_avg_root_total_visits"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_nonzero_visit_actions += float(collection_metrics["mcts_avg_root_nonzero_visit_actions"]) * cycle_mcts_actions
                weighted_sum_mcts_avg_root_legal_actions += float(collection_metrics["mcts_avg_root_legal_actions"]) * cycle_mcts_actions

            cycle_train_steps = float(train_metrics["train_steps"])
            total_train_steps += cycle_train_steps
            weighted_sum_avg_policy_loss += float(train_metrics["avg_policy_loss"]) * cycle_train_steps
            weighted_sum_avg_value_loss += float(train_metrics["avg_value_loss"]) * cycle_train_steps
            weighted_sum_avg_total_loss += float(train_metrics["avg_total_loss"]) * cycle_train_steps
            weighted_sum_avg_grad_norm += float(train_metrics["avg_grad_norm"]) * cycle_train_steps

    if total_train_steps <= 0:
        raise RuntimeError("No training steps executed in cycle run")

    result: dict[str, object] = {
        "mode": "cycles",
        "collector_policy": collector_policy,
        "cycles": float(cycles),
        "episodes_per_cycle": float(episodes_per_cycle),
        "train_steps_per_cycle": float(train_steps_per_cycle),
        "episodes": total_episodes,
        "terminal_episodes": total_terminal_episodes,
        "cutoff_episodes": total_cutoff_episodes,
        "replay_samples_total": total_replay_samples,
        "total_steps": total_steps,
        "total_turns": total_turns,
        "collector_random_actions": total_random_actions,
        "collector_model_actions": total_model_actions,
        "collector_mcts_actions": total_mcts_actions,
        "mcts_avg_search_ms": (weighted_sum_mcts_avg_search_ms / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_entropy": (weighted_sum_mcts_avg_root_entropy / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_top1_visit_prob": (weighted_sum_mcts_avg_root_top1_visit_prob / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_selected_visit_prob": (weighted_sum_mcts_avg_selected_visit_prob / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_value": (weighted_sum_mcts_avg_root_value / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_total_visits": (weighted_sum_mcts_avg_root_total_visits / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_nonzero_visit_actions": (weighted_sum_mcts_avg_root_nonzero_visit_actions / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "mcts_avg_root_legal_actions": (weighted_sum_mcts_avg_root_legal_actions / total_mcts_actions) if total_mcts_actions > 0 else 0.0,
        "avg_policy_loss": weighted_sum_avg_policy_loss / total_train_steps,
        "avg_value_loss": weighted_sum_avg_value_loss / total_train_steps,
        "avg_total_loss": weighted_sum_avg_total_loss / total_train_steps,
        "avg_grad_norm": weighted_sum_avg_grad_norm / total_train_steps,
    }
    result.update(
        {
            "policy_loss": last_train_metrics.get("policy_loss"),
            "value_loss": last_train_metrics.get("value_loss"),
            "total_loss": last_train_metrics.get("total_loss"),
            "grad_norm": last_train_metrics.get("grad_norm"),
            "legal_target_ok": last_train_metrics.get("legal_target_ok"),
        }
    )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Splendor NN smoke pipeline (random self-play + one train step)")
    p.add_argument("--mode", type=str, choices=["smoke", "cycles"], default="smoke")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--collector-policy", type=str, choices=["random", "model-sample", "mcts"], default="random")
    p.add_argument("--train-steps", type=int, default=1)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--episodes-per-cycle", type=int, default=5)
    p.add_argument("--train-steps-per-cycle", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mcts-sims", type=int, default=64)
    p.add_argument("--mcts-c-puct", type=float, default=1.25)
    p.add_argument("--mcts-temperature-moves", type=int, default=10)
    p.add_argument("--mcts-temperature", type=float, default=1.0)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.mode == "smoke":
        metrics = run_smoke(
            episodes=args.episodes,
            max_turns=args.max_turns,
            batch_size=args.batch_size,
            collector_policy=args.collector_policy,
            train_steps=args.train_steps,
            log_every=args.log_every,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            mcts_sims=args.mcts_sims,
            mcts_c_puct=args.mcts_c_puct,
            mcts_temperature_moves=args.mcts_temperature_moves,
            mcts_temperature=args.mcts_temperature,
        )
    else:
        metrics = run_cycles(
            cycles=args.cycles,
            episodes_per_cycle=args.episodes_per_cycle,
            train_steps_per_cycle=args.train_steps_per_cycle,
            max_turns=args.max_turns,
            batch_size=args.batch_size,
            collector_policy=args.collector_policy,
            log_every=args.log_every,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            mcts_sims=args.mcts_sims,
            mcts_c_puct=args.mcts_c_puct,
            mcts_temperature_moves=args.mcts_temperature_moves,
            mcts_temperature=args.mcts_temperature,
        )
    print("Run complete")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")


if __name__ == "__main__":
    main()
