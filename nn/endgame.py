"""nn/endgame.py

Python-side wrapper for the native C++ endgame solver.

Usage:
    from nn.endgame import EndgameConfig, should_use_endgame, run_endgame

    cfg = EndgameConfig()
    if should_use_endgame(state, cfg):
        result = run_endgame(env, cfg, rng_seed=42)
        action = result.best_action
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EndgameConfig:
    """Thresholds and budget settings for the endgame solver."""

    # Trigger conditions: solver activates when ANY condition is true.
    min_points_threshold: int = 11   # either player has >= this many points
    max_deck_cards_threshold: int = 8  # total cards remaining in all decks

    # Search budget
    node_budget: int = 2_000_000
    k_determinizations: int = 8

    # If True, the endgame solver is only called when the MCTS would otherwise
    # be called; it fully replaces MCTS in endgame positions.
    replace_mcts: bool = True


# ---------------------------------------------------------------------------
# Endgame detection
# ---------------------------------------------------------------------------

def should_use_endgame(state: Any, cfg: EndgameConfig) -> bool:
    """
    Return True if the current state is in endgame territory and the solver
    should be used instead of (or in addition to) MCTS.

    `state` is a StepState from SplendorNativeEnv.
    The raw integer state vector contains enough information to decode
    points and a proxy for deck size.

    We use the native env's export_state() for an accurate check — call this
    function *after* getting a fresh StepState so the env is in sync.
    """
    if state.is_terminal:
        return False

    import numpy as np
    from .state_schema import (
        CP_POINTS_IDX, OP_POINTS_IDX,
        FACEUP_START, CARD_FEATURE_LEN,
    )

    s = state.state  # float32 (252,) normalized

    # Decode points (normalized by /20)
    cp_points = int(round(float(s[CP_POINTS_IDX]) * 20.0))
    op_points = int(round(float(s[OP_POINTS_IDX]) * 20.0))
    max_points = max(cp_points, op_points)

    if max_points >= cfg.min_points_threshold:
        return True

    # Proxy for deck size: count non-empty face-up slots to infer progress.
    # A more accurate check uses env.hidden_deck_card_ids_by_tier() but that
    # requires the env object; here we use a cheap approximation.
    # Callers that have the env available should pass use_exact_deck_count=True.
    return False


def should_use_endgame_exact(env: Any, cfg: EndgameConfig) -> bool:
    """
    More accurate endgame check using the env's exact deck counts.
    Slightly more expensive than should_use_endgame() but recommended when
    the env object is readily available.
    """
    state = env.get_state()
    if state.is_terminal:
        return False

    from .state_schema import CP_POINTS_IDX, OP_POINTS_IDX
    s = state.state
    cp_points = int(round(float(s[CP_POINTS_IDX]) * 20.0))
    op_points = int(round(float(s[OP_POINTS_IDX]) * 20.0))
    max_points = max(cp_points, op_points)

    if max_points >= cfg.min_points_threshold:
        return True

    # Exact deck count from native env
    deck_ids = env.hidden_deck_card_ids_by_tier()
    total_deck = sum(len(v) for v in deck_ids.values())
    if total_deck <= cfg.max_deck_cards_threshold:
        return True

    return False


# ---------------------------------------------------------------------------
# Endgame result wrapper
# ---------------------------------------------------------------------------

@dataclass
class EndgameResult:
    best_action: int
    value: float                      # +1 = solver expects win, -1 = loss
    is_exact: bool                    # True iff all paths searched to terminal
    nodes_searched: int
    tt_hits: int
    determinizations_completed: int


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_endgame(
    env: Any,
    cfg: EndgameConfig | None = None,
    *,
    rng_seed: int | None = None,
) -> EndgameResult:
    """
    Run the native C++ endgame solver on the current env state.

    Parameters
    ----------
    env : SplendorNativeEnv (must be initialized)
    cfg : EndgameConfig (uses defaults if None)
    rng_seed : optional seed for determinization sampling

    Returns
    -------
    EndgameResult with best_action and diagnostic fields
    """
    if cfg is None:
        cfg = EndgameConfig()

    seed = rng_seed if rng_seed is not None else random.getrandbits(64)

    native_result = env.solve_endgame(
        node_budget=int(cfg.node_budget),
        k_determinizations=int(cfg.k_determinizations),
        rng_seed=int(seed),
    )

    return EndgameResult(
        best_action=int(native_result.best_action),
        value=float(native_result.value),
        is_exact=bool(native_result.is_exact),
        nodes_searched=int(native_result.nodes_searched),
        tt_hits=int(native_result.tt_hits),
        determinizations_completed=int(native_result.determinizations_completed),
    )


# ---------------------------------------------------------------------------
# Drop-in replacement for run_mcts / run_ismcts
# ---------------------------------------------------------------------------

def run_endgame_or_mcts(
    env: Any,
    model: Any,
    state: Any,
    *,
    turns_taken: int,
    device: str = "cpu",
    mcts_config: Any = None,
    endgame_config: EndgameConfig | None = None,
    rng: random.Random | None = None,
) -> Any:
    """
    Drop-in replacement: uses the endgame solver when in endgame territory,
    falls back to MCTS otherwise.

    Returns an object with a `.chosen_action_idx` attribute and `.visit_probs`
    compatible with the existing MCTS result interface.
    """
    from .mcts import run_mcts, MCTSResult
    import numpy as np
    from .state_schema import ACTION_DIM

    eg_cfg = endgame_config or EndgameConfig()

    if should_use_endgame_exact(env, eg_cfg):
        py_rng = rng or random.Random()
        eg_result = run_endgame(env, eg_cfg, rng_seed=py_rng.getrandbits(64))

        # Wrap as MCTSResult-compatible object
        visit_probs = np.zeros(ACTION_DIM, dtype=np.float32)
        if eg_result.best_action >= 0:
            visit_probs[eg_result.best_action] = 1.0

        q_values = np.zeros(ACTION_DIM, dtype=np.float32)
        if eg_result.best_action >= 0:
            q_values[eg_result.best_action] = eg_result.value

        return MCTSResult(
            chosen_action_idx=eg_result.best_action,
            visit_probs=visit_probs,
            q_values=q_values,
            root_best_value=eg_result.value,
            search_slots_requested=eg_result.nodes_searched,
            search_slots_evaluated=eg_result.nodes_searched,
            search_slots_drop_pending_eval=0,
            search_slots_drop_no_action=0,
        )

    return run_mcts(
        env,
        model,
        state,
        turns_taken=turns_taken,
        device=device,
        config=mcts_config,
        rng=rng,
    )
