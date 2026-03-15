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
class ActionExecutionPlan:
    action_idx: int
    kind: str
    payload: dict[str, Any]


def plan_return_actions(
    action_indices: list[int] | tuple[int, ...],
    *,
    player_seat: str,
    observation: ObservedBoardState,
) -> ActionExecutionPlan:
    if not action_indices:
        raise ValueError("Expected at least one return action index")
    player_index = observation.players[player_seat].spendee_player_index
    chips = [0, 0, 0, 0, 0]
    for action_idx in action_indices:
        if action_idx < 61 or action_idx > 65:
            raise ValueError(f"Non-return action index in return batch: {action_idx}")
        chips[action_idx - 61] += 1
    return ActionExecutionPlan(
        int(action_indices[0]),
        "return_gem",
        {"type": "returnChips", "playerIndex": player_index, "goldChips": 0, "chips": chips},
    )


def plan_action(action_idx: int, *, player_seat: str, observation: ObservedBoardState) -> ActionExecutionPlan:
    player_index = observation.players[player_seat].spendee_player_index
    if 0 <= action_idx <= 11:
        tier = action_idx // 4 + 1
        slot = action_idx % 4
        card = observation.faceup[tier - 1][slot]
        if card is None:
            raise RuntimeError(f"No visible face-up card for action {action_idx}")
        return ActionExecutionPlan(
            action_idx,
            "buy_faceup",
            {"type": "buyCard", "playerIndex": player_index, "cardIndex": card.spendee_card_index},
        )
    if 12 <= action_idx <= 14:
        slot = action_idx - 12
        reserved_slots = observation.players[player_seat].reserved_slots
        if slot >= len(reserved_slots):
            raise RuntimeError(f"No reserved slot {slot} for action {action_idx}")
        reserved = reserved_slots[slot]
        if reserved.card is None:
            raise RuntimeError(f"Reserved slot {slot} has no visible card for action {action_idx}")
        return ActionExecutionPlan(
            action_idx,
            "buy_reserved",
            {"type": "buyReservedCard", "playerIndex": player_index, "cardIndex": reserved.card.spendee_card_index},
        )
    if 15 <= action_idx <= 26:
        rel = action_idx - 15
        tier = rel // 4 + 1
        slot = rel % 4
        card = observation.faceup[tier - 1][slot]
        if card is None:
            raise RuntimeError(f"No visible face-up card to reserve for action {action_idx}")
        return ActionExecutionPlan(
            action_idx,
            "reserve_faceup",
            {"type": "reserveShowedCard", "playerIndex": player_index, "cardIndex": card.spendee_card_index},
        )
    if 27 <= action_idx <= 29:
        level = action_idx - 27
        return ActionExecutionPlan(
            action_idx,
            "reserve_deck",
            {"type": "reserveHiddenCard", "playerIndex": player_index, "level": level},
        )
    if 30 <= action_idx <= 39:
        colors = TAKE3_TRIPLETS[action_idx - 30]
        chips = [1 if color in colors else 0 for color in COLORS]
        return ActionExecutionPlan(action_idx, "take_gems", {"type": "pickChips", "playerIndex": player_index, "chips": chips})
    if 40 <= action_idx <= 44:
        color = COLORS[action_idx - 40]
        chips = [2 if candidate == color else 0 for candidate in COLORS]
        return ActionExecutionPlan(action_idx, "take_gems", {"type": "pickChips", "playerIndex": player_index, "chips": chips})
    if 45 <= action_idx <= 54:
        colors = TAKE2_PAIRS[action_idx - 45]
        chips = [1 if color in colors else 0 for color in COLORS]
        return ActionExecutionPlan(action_idx, "take_gems", {"type": "pickChips", "playerIndex": player_index, "chips": chips})
    if 55 <= action_idx <= 59:
        color = COLORS[action_idx - 55]
        chips = [1 if candidate == color else 0 for candidate in COLORS]
        return ActionExecutionPlan(action_idx, "take_gems", {"type": "pickChips", "playerIndex": player_index, "chips": chips})
    if action_idx == 60:
        return ActionExecutionPlan(
            action_idx,
            "pass_regular",
            {"type": "passRegular", "playerIndex": player_index},
        )
    if 61 <= action_idx <= 65:
        return plan_return_actions([action_idx], player_seat=player_seat, observation=observation)
    if 66 <= action_idx <= 68:
        slot = action_idx - 66
        if slot >= len(observation.visible_nobles):
            raise RuntimeError(f"No visible noble slot {slot} for action {action_idx}")
        noble = observation.visible_nobles[slot]
        return ActionExecutionPlan(
            action_idx,
            "choose_noble",
            {"type": "pickNoble", "playerIndex": player_index, "nobleIndex": noble.spendee_noble_index},
        )
    raise ValueError(f"Unknown action index: {action_idx}")


class SpendeeExecutor:
    def __init__(self) -> None:
        pass

    async def execute_plan(
        self,
        page: Any,
        plan: ActionExecutionPlan,
        *,
        dry_run: bool = False,
    ) -> ActionExecutionPlan:
        if dry_run:
            return plan
        result = await page.evaluate(
            """
            async ({payload}) => {
              if (typeof Games === "undefined" || !Games || typeof Games.findOne !== "function") {
                throw new Error("Games Minimongo collection is not available");
              }
              const game = Games.findOne();
              if (!game || typeof game.clientAction !== "function") {
                throw new Error("Current game does not expose clientAction");
              }
              return await new Promise((resolve, reject) => {
                try {
                  game.clientAction(payload, (...args) => {
                    resolve(args);
                  });
                } catch (err) {
                  reject(String(err));
                }
              });
            }
            """,
            {"payload": plan.payload},
        )
        if result and len(result) > 0 and result[0]:
            raise RuntimeError(f"Spendee clientAction failed: {result}")
        return plan
