from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .imperfect_info import acting_player_has_hidden_uncertainty
from .mcts import MCTSConfig, MCTSResult
from .native_env import SplendorNativeEnv, StepState

if TYPE_CHECKING:
    from .ismcts import ISMCTSConfig


@dataclass
class AlphaBetaConfig:
    max_nodes: int = 0
    max_depth: int = 0
    max_root_actions: int = 0
    determinize_root_hidden_info: bool = True
    fallback_search_type: Literal["mcts", "ismcts"] = "mcts"
    fallback_mcts_config: MCTSConfig | None = None
    fallback_ismcts_config: "ISMCTSConfig | None" = None


def _is_limit_exceeded_error(exc: Exception) -> bool:
    return "ALPHABETA_LIMIT_EXCEEDED:" in str(exc)


def run_alphabeta(
    env: Any,
    model: Any,
    state: StepState,
    *,
    turns_taken: int,
    device: str = "cpu",
    config: AlphaBetaConfig | None = None,
    rng: random.Random | None = None,
) -> MCTSResult:
    cfg = config or AlphaBetaConfig()
    if not isinstance(env, SplendorNativeEnv):
        raise TypeError("run_alphabeta requires nn.native_env.SplendorNativeEnv")
    if state.is_terminal:
        raise ValueError("run_alphabeta called on terminal state")
    if int(cfg.max_nodes) < 0:
        raise ValueError("max_nodes must be non-negative")
    if int(cfg.max_depth) < 0:
        raise ValueError("max_depth must be non-negative")
    if int(cfg.max_root_actions) < 0:
        raise ValueError("max_root_actions must be non-negative")
    if not bool(cfg.determinize_root_hidden_info):
        exported = env.export_state()
        if acting_player_has_hidden_uncertainty(exported):
            raise ValueError(
                "run_alphabeta requires determinize_root_hidden_info=True when the acting player has hidden uncertainty"
            )

    py_rng = rng if rng is not None else random
    try:
        native_result = env.run_alphabeta_native(
            max_nodes=int(cfg.max_nodes),
            max_depth=int(cfg.max_depth),
            max_root_actions=int(cfg.max_root_actions),
            rng_seed=int(py_rng.getrandbits(64)),
            determinize_root_hidden_info=bool(cfg.determinize_root_hidden_info),
        )
    except RuntimeError as exc:
        if not _is_limit_exceeded_error(exc):
            raise
        if cfg.fallback_search_type == "ismcts":
            from .ismcts import ISMCTSConfig, run_ismcts

            return run_ismcts(
                env,
                model,
                state=state,
                turns_taken=int(turns_taken),
                device=device,
                config=(cfg.fallback_ismcts_config or ISMCTSConfig()),
                rng=py_rng,
            )
        if cfg.fallback_search_type == "mcts":
            from .mcts import run_mcts

            return run_mcts(
                env,
                model,
                state=state,
                turns_taken=int(turns_taken),
                device=device,
                config=(cfg.fallback_mcts_config or MCTSConfig()),
                rng=py_rng,
            )
        raise ValueError(f"Unsupported fallback_search_type: {cfg.fallback_search_type}")

    visit_probs = np.asarray(native_result.visit_probs, dtype=np.float32)
    q_values = np.asarray(native_result.q_values, dtype=np.float32)
    if visit_probs.shape != q_values.shape:
        raise RuntimeError("Native alpha-beta returned mismatched visit_probs/q_values shapes")

    return MCTSResult(
        chosen_action_idx=int(native_result.chosen_action_idx),
        visit_probs=visit_probs,
        q_values=q_values,
        root_best_value=float(native_result.root_best_value),
        search_slots_requested=int(getattr(native_result, "search_slots_requested", 0)),
        search_slots_evaluated=int(getattr(native_result, "search_slots_evaluated", 0)),
        search_slots_drop_pending_eval=int(getattr(native_result, "search_slots_drop_pending_eval", 0)),
        search_slots_drop_no_action=int(getattr(native_result, "search_slots_drop_no_action", 0)),
    )
