from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .determinize import sample_hydrated_states
from .replay_log import build_replay_move_log
from .shadow_state import ShadowState


def _deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def build_webui_save_payload(
    shadow: ShadowState,
    *,
    checkpoint_path: str,
    num_simulations: int,
    player_seat: str | None,
    analysis_mode: bool = True,
) -> dict[str, Any]:
    observed = shadow.last_observation
    if observed is None:
        raise RuntimeError("ShadowState has not been bootstrapped")

    rng = random.Random(
        _deterministic_seed(
            observed.game_id,
            observed.board_version,
            observed.turns_count,
            observed.current_turn_seat,
            checkpoint_path,
            player_seat or "auto",
        )
    )
    exported_state = sample_hydrated_states(shadow, samples=1, rng=rng)[0]
    replay_log = build_replay_move_log(observed)
    metadata = dict(exported_state.get("metadata", {}))
    metadata.update(
        {
            "source": "spendee_bridge",
            "spendee_game_id": observed.game_id,
            "spendee_board_version": observed.board_version,
            "spendee_observed_at": observed.observed_at,
            "spendee_player_seat": player_seat,
            "spendee_current_turn_seat": observed.current_turn_seat,
            "spendee_current_job": observed.current_job,
            "spendee_action_items_count": len(observed.raw_action_items),
            "spendee_move_log_complete_action_indices": replay_log.complete_action_indices,
        }
    )
    exported_state["metadata"] = metadata

    checkpoint_abs = str(Path(checkpoint_path).resolve())
    saved_at = datetime.now(timezone.utc).isoformat()
    return {
        "version": 1,
        "saved_at": saved_at,
        "game_id": observed.game_id,
        "config": {
            "checkpoint_id": checkpoint_abs,
            "checkpoint_path": checkpoint_abs,
            "num_simulations": int(num_simulations),
            "player_seat": "P0" if player_seat not in ("P0", "P1") else player_seat,
            "seed": 0,
            "manual_reveal_mode": False,
            "analysis_mode": bool(analysis_mode),
        },
        "exported_state": exported_state,
        "move_log": [
            {
                "turn_index": entry.turn_index,
                "actor": entry.actor,
                "action_idx": entry.action_idx,
                "label": entry.label,
            }
            for entry in replay_log.entries
        ],
        "setup_event_log": [],
        "event_log": [],
        "redo_log": [],
        "pending_reveals": [],
        "forced_winner": None,
        "rng_state": None,
    }
