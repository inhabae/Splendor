from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .native_env import SplendorNativeEnv, StepState
from .state_schema import ACTION_DIM, STATE_DIM


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.25
    temperature_moves: int = 10
    temperature: float = 1.0
    eps: float = 1e-8
    root_dirichlet_noise: bool = False
    root_dirichlet_epsilon: float = 0.25
    root_dirichlet_alpha_total: float = 10.0
    eval_batch_size: int = 8


@dataclass
class MCTSResult:
    action: int
    visit_probs: np.ndarray  # (69,) float32
    root_value: float
    num_simulations: int
    root_total_visits: int
    root_nonzero_visit_actions: int
    root_legal_actions: int


def _normalize_priors_rows(priors: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = np.asarray(priors, dtype=np.float32).copy()
    out[~masks] = 0.0
    for i in range(out.shape[0]):
        legal = masks[i]
        if not bool(np.any(legal)):
            raise ValueError("Batched evaluator received a row with no legal actions")
        row = out[i]
        row_sum = float(np.sum(row[legal], dtype=np.float64))
        if row_sum <= 0.0 or not np.isfinite(row_sum):
            row[:] = 0.0
            row[legal] = 1.0 / float(np.count_nonzero(legal))
        else:
            row[legal] /= row_sum
            row[~legal] = 0.0
    return out


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
    model.eval()

    if not isinstance(env, SplendorNativeEnv):
        raise TypeError("run_mcts requires nn.native_env.SplendorNativeEnv (native-env-only implementation)")
    if cfg.num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if not (0.0 <= float(cfg.root_dirichlet_epsilon) <= 1.0):
        raise ValueError("root_dirichlet_epsilon must be in [0,1]")
    if float(cfg.root_dirichlet_alpha_total) <= 0.0:
        raise ValueError("root_dirichlet_alpha_total must be positive")
    if int(cfg.eval_batch_size) <= 0:
        raise ValueError("eval_batch_size must be positive")
    if state.is_terminal:
        raise ValueError("run_mcts called on terminal state")
    if state.state.shape != (STATE_DIM,):
        raise ValueError(f"Unexpected root state shape {state.state.shape}")
    if state.mask.shape != (ACTION_DIM,):
        raise ValueError(f"Unexpected root mask shape {state.mask.shape}")
    if not bool(state.mask.any()):
        raise ValueError("MCTS root has no legal actions")

    def evaluator(states_np: np.ndarray, masks_np: np.ndarray):
        states_np = np.asarray(states_np, dtype=np.float32)
        masks_np = np.asarray(masks_np, dtype=np.bool_)
        if states_np.ndim != 2 or states_np.shape[1] != STATE_DIM:
            raise ValueError(f"evaluator states shape must be (B,{STATE_DIM}), got {states_np.shape}")
        if masks_np.ndim != 2 or masks_np.shape[1] != ACTION_DIM:
            raise ValueError(f"evaluator masks shape must be (B,{ACTION_DIM}), got {masks_np.shape}")
        if states_np.shape[0] != masks_np.shape[0]:
            raise ValueError("evaluator batch size mismatch between states and masks")
        if states_np.shape[0] == 0:
            raise ValueError("evaluator requires non-empty batch")

        state_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(masks_np, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, value_t = model(state_t)

        if tuple(logits.shape) != (states_np.shape[0], ACTION_DIM):
            raise ValueError(f"Model logits shape must be (B,{ACTION_DIM}), got {tuple(logits.shape)}")
        if value_t.ndim == 2 and value_t.shape[1] == 1:
            value_t = value_t.squeeze(1)
        if value_t.ndim != 1 or value_t.shape[0] != states_np.shape[0]:
            raise ValueError(f"Model values shape must be (B,), got {tuple(value_t.shape)}")

        logits = logits.clone()
        logits[~mask_t] = -1e9
        priors_t = torch.softmax(logits, dim=-1)
        priors = priors_t.detach().cpu().numpy().astype(np.float32, copy=False)
        values = value_t.detach().cpu().numpy().astype(np.float32, copy=False)
        priors = _normalize_priors_rows(priors, masks_np)
        if not np.isfinite(values).all():
            raise ValueError("Model returned non-finite values")
        return priors, values

    py_rng = rng if rng is not None else random
    rng_seed = int(py_rng.getrandbits(64))

    native_result = env.run_mcts_native(
        evaluator=evaluator,
        turns_taken=int(turns_taken),
        num_simulations=int(cfg.num_simulations),
        c_puct=float(cfg.c_puct),
        temperature_moves=int(cfg.temperature_moves),
        temperature=float(cfg.temperature),
        eps=float(cfg.eps),
        root_dirichlet_noise=bool(cfg.root_dirichlet_noise),
        root_dirichlet_epsilon=float(cfg.root_dirichlet_epsilon),
        root_dirichlet_alpha_total=float(cfg.root_dirichlet_alpha_total),
        eval_batch_size=int(cfg.eval_batch_size),
        rng_seed=rng_seed,
    )

    visit_probs = np.asarray(native_result.visit_probs, dtype=np.float32)
    if visit_probs.shape != (ACTION_DIM,):
        raise RuntimeError(f"Unexpected native visit_probs shape {visit_probs.shape}")

    return MCTSResult(
        action=int(native_result.action),
        visit_probs=visit_probs,
        root_value=float(native_result.root_value),
        num_simulations=int(native_result.num_simulations),
        root_total_visits=int(native_result.root_total_visits),
        root_nonzero_visit_actions=int(native_result.root_nonzero_visit_actions),
        root_legal_actions=int(native_result.root_legal_actions),
    )
