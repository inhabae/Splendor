from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .observer import ObservedBoardState

COLORS = ("white", "blue", "green", "red", "black")
TAKE3_TRIPLETS = (
    ("white", "blue", "green"),
    ("white", "blue", "red"),
    ("white", "blue", "black"),
    ("white", "green", "red"),
    ("white", "green", "black"),
    ("white", "red", "black"),
    ("blue", "green", "red"),
    ("blue", "green", "black"),
    ("blue", "red", "black"),
    ("green", "red", "black"),
)
TAKE2_PAIRS = (
    ("white", "blue"),
    ("white", "green"),
    ("white", "red"),
    ("white", "black"),
    ("blue", "green"),
    ("blue", "red"),
    ("blue", "black"),
    ("green", "red"),
    ("green", "black"),
    ("red", "black"),
)


@dataclass(frozen=True)
class ReplayMoveLogEntry:
    turn_index: int
    actor: str
    action_idx: int
    label: str


@dataclass(frozen=True)
class ReplayMoveLog:
    entries: list[ReplayMoveLogEntry]
    complete_action_indices: bool


def _describe_action(action_idx: int) -> str:
    if 0 <= action_idx <= 11:
        return f"BUY face-up tier {action_idx // 4 + 1} slot {action_idx % 4}"
    if 12 <= action_idx <= 14:
        return f"BUY reserved slot {action_idx - 12}"
    if 15 <= action_idx <= 26:
        rel = action_idx - 15
        return f"RESERVE face-up tier {rel // 4 + 1} slot {rel % 4}"
    if 27 <= action_idx <= 29:
        return f"RESERVE from deck tier {action_idx - 27 + 1}"
    if 30 <= action_idx <= 39:
        names = ", ".join(TAKE3_TRIPLETS[action_idx - 30])
        return f"TAKE 3 gems ({names})"
    if 40 <= action_idx <= 44:
        return f"TAKE 2 gems ({COLORS[action_idx - 40]})"
    if 45 <= action_idx <= 54:
        names = ", ".join(TAKE2_PAIRS[action_idx - 45])
        return f"TAKE 2 gems ({names})"
    if 55 <= action_idx <= 59:
        return f"TAKE 1 gem ({COLORS[action_idx - 55]})"
    if action_idx == 60:
        return "PASS"
    if 61 <= action_idx <= 65:
        return f"RETURN gem ({COLORS[action_idx - 61]})"
    if 66 <= action_idx <= 68:
        return f"CHOOSE noble index {action_idx - 66}"
    return f"UNKNOWN action {action_idx}"


def _seat_from_player_index(observed: ObservedBoardState, player_index: int) -> str:
    for seat in ("P0", "P1"):
        if observed.players[seat].spendee_player_index == player_index:
            return seat
    return "P0" if int(player_index) == 0 else "P1"


def _chips_to_action_idx(chips: list[int]) -> int | None:
    if len(chips) != 5:
        return None
    total = sum(int(value) for value in chips)
    if total == 3 and all(int(value) in (0, 1) for value in chips):
        picked = tuple(color for color, value in zip(COLORS, chips) if int(value) == 1)
        try:
            return 30 + TAKE3_TRIPLETS.index(picked)
        except ValueError:
            return None
    if total == 2:
        if chips.count(2) == 1 and chips.count(1) == 0:
            return 40 + chips.index(2)
        if chips.count(1) == 2 and all(int(value) in (0, 1) for value in chips):
            picked = tuple(color for color, value in zip(COLORS, chips) if int(value) == 1)
            try:
                return 45 + TAKE2_PAIRS.index(picked)
            except ValueError:
                return None
    if total == 1 and chips.count(1) == 1:
        return 55 + chips.index(1)
    return None


def _format_chip_label(prefix: str, chips: list[int]) -> str:
    parts = [f"{count} {color}" for color, count in zip(COLORS, chips) if int(count) > 0]
    suffix = ", ".join(parts) if parts else "none"
    return f"{prefix} ({suffix})"


def _pick_noble_action_idx(observed: ObservedBoardState, noble_index: int) -> int | None:
    for slot, noble in enumerate(observed.visible_nobles):
        if int(noble.spendee_noble_index) == int(noble_index):
            return 66 + slot
    return None


def _entries_from_action_item(
    observed: ObservedBoardState,
    item: dict[str, Any],
    *,
    turn_index: int,
) -> list[ReplayMoveLogEntry]:
    action = item.get("action")
    if not isinstance(action, dict):
        return []
    player_index = action.get("playerIndex")
    if not isinstance(player_index, int):
        return []
    actor = _seat_from_player_index(observed, player_index)
    action_type = str(action.get("type", ""))

    if action_type == "reserveHiddenCard":
        level = action.get("level")
        if isinstance(level, int) and 0 <= level <= 2:
            action_idx = 27 + level
            return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=action_idx, label=_describe_action(action_idx))]
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=-1, label="RESERVE from deck")]

    if action_type == "pickChips":
        chips = [int(value) for value in action.get("chips", [])]
        action_idx = _chips_to_action_idx(chips)
        label = _describe_action(action_idx) if action_idx is not None else _format_chip_label("TAKE gems", chips)
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=action_idx if action_idx is not None else -1, label=label)]

    if action_type == "returnChips":
        chips = [int(value) for value in action.get("chips", [])]
        entries: list[ReplayMoveLogEntry] = []
        current_turn = turn_index
        for color_index, count in enumerate(chips[:5]):
            for _ in range(max(count, 0)):
                action_idx = 61 + color_index
                entries.append(
                    ReplayMoveLogEntry(
                        turn_index=current_turn,
                        actor=actor,
                        action_idx=action_idx,
                        label=_describe_action(action_idx),
                    )
                )
                current_turn += 1
        if entries:
            return entries
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=-1, label="RETURN gems")]

    if action_type == "passRegular":
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=60, label=_describe_action(60))]

    if action_type == "pickNoble":
        noble_index = action.get("nobleIndex")
        if isinstance(noble_index, int):
            action_idx = _pick_noble_action_idx(observed, noble_index)
            if action_idx is not None:
                return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=action_idx, label=_describe_action(action_idx))]
            return [
                ReplayMoveLogEntry(
                    turn_index=turn_index,
                    actor=actor,
                    action_idx=-1,
                    label=f"CHOOSE noble #{noble_index}",
                )
            ]
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=-1, label="CHOOSE noble")]

    if action_type == "buyCard":
        card_index = action.get("cardIndex")
        label = f"BUY face-up card #{card_index}" if isinstance(card_index, int) else "BUY face-up card"
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=-1, label=label)]

    if action_type == "buyReservedCard":
        card_index = action.get("cardIndex")
        label = f"BUY reserved card #{card_index}" if isinstance(card_index, int) else "BUY reserved card"
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=-1, label=label)]

    if action_type == "reserveShowedCard":
        card_index = action.get("cardIndex")
        label = f"RESERVE face-up card #{card_index}" if isinstance(card_index, int) else "RESERVE face-up card"
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=-1, label=label)]

    if action_type:
        return [ReplayMoveLogEntry(turn_index=turn_index, actor=actor, action_idx=-1, label=f"SPENDEE {action_type}")]
    return []


def build_replay_move_log(observed: ObservedBoardState) -> ReplayMoveLog:
    entries: list[ReplayMoveLogEntry] = []
    turn_index = 0
    for item in observed.raw_action_items:
        action_entries = _entries_from_action_item(observed, dict(item), turn_index=turn_index)
        entries.extend(action_entries)
        turn_index += len(action_entries)
    complete_action_indices = bool(entries) and all(entry.action_idx >= 0 for entry in entries)
    return ReplayMoveLog(entries=entries, complete_action_indices=complete_action_indices)
