from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .bridge_env import SplendorBridgeEnv, StepState
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


@dataclass
class _EpisodeStep:
    state: np.ndarray
    mask: np.ndarray
    action_target: int
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


def collect_random_episode(
    env: SplendorBridgeEnv,
    replay: ReplayBuffer,
    *,
    seed: int,
    max_turns: int,
    rng: random.Random,
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
        action = _uniform_random_legal_action(state.mask, rng)
        if not bool(state.mask[action]):
            raise AssertionError("Sampled action is not legal")

        prev_player_id = env.current_player_id
        episode_steps.append(
            _EpisodeStep(
                state=state.state.copy(),
                mask=state.mask.copy(),
                action_target=action,
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
            )
        )

    return EpisodeSummary(
        num_steps=len(episode_steps),
        num_turns=turns_taken,
        reached_cutoff=reached_cutoff,
        winner=winner,
    )


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
    value_target = batch["value_target"]

    logits, value_pred = model(states)
    if not torch.isfinite(logits).all() or not torch.isfinite(value_pred).all():
        raise RuntimeError("Model produced non-finite outputs")

    policy_loss = masked_cross_entropy_loss(logits, masks, action_target)
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


def run_smoke(
    *,
    episodes: int = 5,
    max_turns: int = 80,
    batch_size: int = 32,
    train_steps: int = 1,
    log_every: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    replay = ReplayBuffer()
    rng = random.Random(seed)
    cutoff_count = 0
    total_steps = 0
    total_turns = 0
    terminal_episodes = 0

    with SplendorBridgeEnv() as env:
        for ep in range(episodes):
            summary = collect_random_episode(
                env,
                replay,
                seed=seed + ep,
                max_turns=max_turns,
                rng=rng,
            )
            total_steps += summary.num_steps
            total_turns += summary.num_turns
            if summary.reached_cutoff:
                cutoff_count += 1
            else:
                terminal_episodes += 1

    if len(replay) == 0:
        raise RuntimeError("Replay buffer is empty after collection")

    model = MaskedPolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    metrics: dict[str, float] = {}
    sum_policy_loss = 0.0
    sum_value_loss = 0.0
    sum_total_loss = 0.0
    sum_grad_norm = 0.0

    for step in range(1, train_steps + 1):
        batch = replay.sample_batch(min(batch_size, len(replay)), device=device)
        metrics = train_one_step(model, optimizer, batch, value_loss_weight=1.0, grad_clip_norm=1.0)

        sum_policy_loss += metrics["policy_loss"]
        sum_value_loss += metrics["value_loss"]
        sum_total_loss += metrics["total_loss"]
        sum_grad_norm += metrics["grad_norm"]

        if step == 1 or step % log_every == 0 or step == train_steps:
            print(
                f"train_step={step}/{train_steps} "
                f"policy_loss={metrics['policy_loss']:.6f} "
                f"value_loss={metrics['value_loss']:.6f} "
                f"total_loss={metrics['total_loss']:.6f} "
                f"grad_norm={metrics['grad_norm']:.6f}"
            )

    metrics.update(
        {
            "episodes": float(episodes),
            "terminal_episodes": float(terminal_episodes),
            "cutoff_episodes": float(cutoff_count),
            "replay_samples": float(len(replay)),
            "total_steps": float(total_steps),
            "total_turns": float(total_turns),
            "train_steps": float(train_steps),
            "avg_policy_loss": sum_policy_loss / train_steps,
            "avg_value_loss": sum_value_loss / train_steps,
            "avg_total_loss": sum_total_loss / train_steps,
            "avg_grad_norm": sum_grad_norm / train_steps,
        }
    )
    return metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Splendor NN smoke pipeline (random self-play + one train step)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-steps", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    metrics = run_smoke(
        episodes=args.episodes,
        max_turns=args.max_turns,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        log_every=args.log_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    print("Smoke run complete")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")


if __name__ == "__main__":
    main()
