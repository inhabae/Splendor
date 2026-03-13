from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nn.native_env import list_standard_cards, list_standard_nobles

COLORS = ("white", "blue", "green", "red", "black")

SPENDEE_CARD_CONSTANTS: tuple[dict[str, object], ...] = (
    {"index": 0, "level": 0, "costs": (0, 3, 0, 0, 0), "discount": 0, "score": 0},
    {"index": 1, "level": 0, "costs": (0, 0, 0, 2, 1), "discount": 0, "score": 0},
    {"index": 2, "level": 0, "costs": (0, 1, 1, 1, 1), "discount": 0, "score": 0},
    {"index": 3, "level": 0, "costs": (0, 2, 0, 0, 2), "discount": 0, "score": 0},
    {"index": 4, "level": 0, "costs": (0, 0, 4, 0, 0), "discount": 0, "score": 1},
    {"index": 5, "level": 0, "costs": (0, 1, 2, 1, 1), "discount": 0, "score": 0},
    {"index": 6, "level": 0, "costs": (0, 2, 2, 0, 1), "discount": 0, "score": 0},
    {"index": 7, "level": 0, "costs": (3, 1, 0, 0, 1), "discount": 0, "score": 0},
    {"index": 8, "level": 0, "costs": (1, 0, 0, 0, 2), "discount": 1, "score": 0},
    {"index": 9, "level": 0, "costs": (0, 0, 0, 0, 3), "discount": 1, "score": 0},
    {"index": 10, "level": 0, "costs": (1, 0, 1, 1, 1), "discount": 1, "score": 0},
    {"index": 11, "level": 0, "costs": (0, 0, 2, 0, 2), "discount": 1, "score": 0},
    {"index": 12, "level": 0, "costs": (0, 0, 0, 4, 0), "discount": 1, "score": 1},
    {"index": 13, "level": 0, "costs": (1, 0, 1, 2, 1), "discount": 1, "score": 0},
    {"index": 14, "level": 0, "costs": (1, 0, 2, 2, 0), "discount": 1, "score": 0},
    {"index": 15, "level": 0, "costs": (0, 1, 3, 1, 0), "discount": 1, "score": 0},
    {"index": 16, "level": 0, "costs": (2, 1, 0, 0, 0), "discount": 2, "score": 0},
    {"index": 17, "level": 0, "costs": (0, 0, 0, 3, 0), "discount": 2, "score": 0},
    {"index": 18, "level": 0, "costs": (1, 1, 0, 1, 1), "discount": 2, "score": 0},
    {"index": 19, "level": 0, "costs": (0, 2, 0, 2, 0), "discount": 2, "score": 0},
    {"index": 20, "level": 0, "costs": (0, 0, 0, 0, 4), "discount": 2, "score": 1},
    {"index": 21, "level": 0, "costs": (1, 1, 0, 1, 2), "discount": 2, "score": 0},
    {"index": 22, "level": 0, "costs": (0, 1, 0, 2, 2), "discount": 2, "score": 0},
    {"index": 23, "level": 0, "costs": (1, 3, 1, 0, 0), "discount": 2, "score": 0},
    {"index": 24, "level": 0, "costs": (0, 2, 1, 0, 0), "discount": 3, "score": 0},
    {"index": 25, "level": 0, "costs": (3, 0, 0, 0, 0), "discount": 3, "score": 0},
    {"index": 26, "level": 0, "costs": (1, 1, 1, 0, 1), "discount": 3, "score": 0},
    {"index": 27, "level": 0, "costs": (2, 0, 0, 2, 0), "discount": 3, "score": 0},
    {"index": 28, "level": 0, "costs": (4, 0, 0, 0, 0), "discount": 3, "score": 1},
    {"index": 29, "level": 0, "costs": (2, 1, 1, 0, 1), "discount": 3, "score": 0},
    {"index": 30, "level": 0, "costs": (2, 0, 1, 0, 2), "discount": 3, "score": 0},
    {"index": 31, "level": 0, "costs": (1, 0, 0, 1, 3), "discount": 3, "score": 0},
    {"index": 32, "level": 0, "costs": (0, 0, 2, 1, 0), "discount": 4, "score": 0},
    {"index": 33, "level": 0, "costs": (0, 0, 3, 0, 0), "discount": 4, "score": 0},
    {"index": 34, "level": 0, "costs": (1, 1, 1, 1, 0), "discount": 4, "score": 0},
    {"index": 35, "level": 0, "costs": (2, 0, 2, 0, 0), "discount": 4, "score": 0},
    {"index": 36, "level": 0, "costs": (0, 4, 0, 0, 0), "discount": 4, "score": 1},
    {"index": 37, "level": 0, "costs": (1, 2, 1, 1, 0), "discount": 4, "score": 0},
    {"index": 38, "level": 0, "costs": (2, 2, 0, 1, 0), "discount": 4, "score": 0},
    {"index": 39, "level": 0, "costs": (0, 0, 1, 3, 1), "discount": 4, "score": 0},
    {"index": 40, "level": 1, "costs": (0, 0, 0, 5, 0), "discount": 0, "score": 2},
    {"index": 41, "level": 1, "costs": (6, 0, 0, 0, 0), "discount": 0, "score": 3},
    {"index": 42, "level": 1, "costs": (0, 0, 3, 2, 2), "discount": 0, "score": 1},
    {"index": 43, "level": 1, "costs": (0, 0, 1, 4, 2), "discount": 0, "score": 2},
    {"index": 44, "level": 1, "costs": (2, 3, 0, 3, 0), "discount": 0, "score": 1},
    {"index": 45, "level": 1, "costs": (0, 0, 0, 5, 3), "discount": 0, "score": 2},
    {"index": 46, "level": 1, "costs": (0, 5, 0, 0, 0), "discount": 1, "score": 2},
    {"index": 47, "level": 1, "costs": (0, 6, 0, 0, 0), "discount": 1, "score": 3},
    {"index": 48, "level": 1, "costs": (0, 2, 2, 3, 0), "discount": 1, "score": 1},
    {"index": 49, "level": 1, "costs": (2, 0, 0, 1, 4), "discount": 1, "score": 2},
    {"index": 50, "level": 1, "costs": (0, 2, 3, 0, 3), "discount": 1, "score": 1},
    {"index": 51, "level": 1, "costs": (5, 3, 0, 0, 0), "discount": 1, "score": 2},
    {"index": 52, "level": 1, "costs": (0, 0, 5, 0, 0), "discount": 2, "score": 2},
    {"index": 53, "level": 1, "costs": (0, 0, 6, 0, 0), "discount": 2, "score": 3},
    {"index": 54, "level": 1, "costs": (2, 3, 0, 0, 2), "discount": 2, "score": 1},
    {"index": 55, "level": 1, "costs": (3, 0, 2, 3, 0), "discount": 2, "score": 1},
    {"index": 56, "level": 1, "costs": (4, 2, 0, 0, 1), "discount": 2, "score": 2},
    {"index": 57, "level": 1, "costs": (0, 5, 3, 0, 0), "discount": 2, "score": 2},
    {"index": 58, "level": 1, "costs": (0, 0, 0, 0, 5), "discount": 3, "score": 2},
    {"index": 59, "level": 1, "costs": (0, 0, 0, 6, 0), "discount": 3, "score": 3},
    {"index": 60, "level": 1, "costs": (2, 0, 0, 2, 3), "discount": 3, "score": 1},
    {"index": 61, "level": 1, "costs": (1, 4, 2, 0, 0), "discount": 3, "score": 2},
    {"index": 62, "level": 1, "costs": (0, 3, 0, 2, 3), "discount": 3, "score": 1},
    {"index": 63, "level": 1, "costs": (3, 0, 0, 0, 5), "discount": 3, "score": 2},
    {"index": 64, "level": 1, "costs": (5, 0, 0, 0, 0), "discount": 4, "score": 2},
    {"index": 65, "level": 1, "costs": (0, 0, 0, 0, 6), "discount": 4, "score": 3},
    {"index": 66, "level": 1, "costs": (3, 2, 2, 0, 0), "discount": 4, "score": 1},
    {"index": 67, "level": 1, "costs": (0, 1, 4, 2, 0), "discount": 4, "score": 2},
    {"index": 68, "level": 1, "costs": (3, 0, 3, 0, 2), "discount": 4, "score": 1},
    {"index": 69, "level": 1, "costs": (0, 0, 5, 3, 0), "discount": 4, "score": 2},
    {"index": 70, "level": 2, "costs": (0, 0, 0, 0, 7), "discount": 0, "score": 4},
    {"index": 71, "level": 2, "costs": (3, 0, 0, 0, 7), "discount": 0, "score": 5},
    {"index": 72, "level": 2, "costs": (3, 0, 0, 3, 6), "discount": 0, "score": 4},
    {"index": 73, "level": 2, "costs": (0, 3, 3, 5, 3), "discount": 0, "score": 3},
    {"index": 74, "level": 2, "costs": (7, 0, 0, 0, 0), "discount": 1, "score": 4},
    {"index": 75, "level": 2, "costs": (7, 3, 0, 0, 0), "discount": 1, "score": 5},
    {"index": 76, "level": 2, "costs": (6, 3, 0, 0, 3), "discount": 1, "score": 4},
    {"index": 77, "level": 2, "costs": (3, 0, 3, 3, 5), "discount": 1, "score": 3},
    {"index": 78, "level": 2, "costs": (0, 7, 0, 0, 0), "discount": 2, "score": 4},
    {"index": 79, "level": 2, "costs": (0, 7, 3, 0, 0), "discount": 2, "score": 5},
    {"index": 80, "level": 2, "costs": (3, 6, 3, 0, 0), "discount": 2, "score": 4},
    {"index": 81, "level": 2, "costs": (5, 3, 0, 3, 3), "discount": 2, "score": 3},
    {"index": 82, "level": 2, "costs": (0, 0, 7, 0, 0), "discount": 3, "score": 4},
    {"index": 83, "level": 2, "costs": (0, 0, 7, 3, 0), "discount": 3, "score": 5},
    {"index": 84, "level": 2, "costs": (0, 3, 6, 3, 0), "discount": 3, "score": 4},
    {"index": 85, "level": 2, "costs": (3, 5, 3, 0, 3), "discount": 3, "score": 3},
    {"index": 86, "level": 2, "costs": (0, 0, 0, 7, 0), "discount": 4, "score": 4},
    {"index": 87, "level": 2, "costs": (0, 0, 0, 7, 3), "discount": 4, "score": 5},
    {"index": 88, "level": 2, "costs": (0, 0, 3, 6, 3), "discount": 4, "score": 4},
    {"index": 89, "level": 2, "costs": (3, 3, 5, 3, 0), "discount": 4, "score": 3},
)

SPENDEE_NOBLE_CONSTANTS: tuple[dict[str, object], ...] = (
    {"index": 0, "costs": (3, 3, 0, 0, 3), "score": 3},
    {"index": 1, "costs": (3, 3, 3, 0, 0), "score": 3},
    {"index": 2, "costs": (0, 3, 3, 3, 0), "score": 3},
    {"index": 3, "costs": (0, 0, 3, 3, 3), "score": 3},
    {"index": 4, "costs": (3, 0, 0, 3, 3), "score": 3},
    {"index": 5, "costs": (4, 0, 0, 0, 4), "score": 3},
    {"index": 6, "costs": (4, 4, 0, 0, 0), "score": 3},
    {"index": 7, "costs": (0, 4, 4, 0, 0), "score": 3},
    {"index": 8, "costs": (0, 0, 4, 4, 0), "score": 3},
    {"index": 9, "costs": (0, 0, 0, 4, 4), "score": 3},
)


def _cost_tuple(cost: dict[str, int]) -> tuple[int, int, int, int, int]:
    return tuple(int(cost.get(color, 0)) for color in COLORS)


@dataclass(frozen=True)
class CardSignature:
    tier: int
    points: int
    bonus_color: str
    cost: tuple[int, int, int, int, int]

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "CardSignature":
        cost = payload.get("cost", {})
        if not isinstance(cost, dict):
            raise TypeError("card payload cost must be a dict")
        return cls(
            tier=int(payload["tier"]),
            points=int(payload["points"]),
            bonus_color=str(payload["bonus_color"]),
            cost=_cost_tuple(cost),
        )


@dataclass(frozen=True)
class NobleSignature:
    points: int
    requirements: tuple[int, int, int, int, int]

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "NobleSignature":
        reqs = payload.get("requirements", {})
        if not isinstance(reqs, dict):
            raise TypeError("noble payload requirements must be a dict")
        return cls(points=int(payload["points"]), requirements=_cost_tuple(reqs))


@dataclass
class SpendeeCatalog:
    cards_by_id: dict[int, dict[str, object]]
    nobles_by_id: dict[int, dict[str, object]]
    card_ids_by_signature: dict[CardSignature, int]
    noble_ids_by_signature: dict[NobleSignature, int]
    card_ids_by_tier: dict[int, tuple[int, ...]]
    spendee_card_to_engine_id: dict[int, int]
    engine_card_to_spendee_index: dict[int, int]
    spendee_noble_to_engine_id: dict[int, int]
    engine_noble_to_spendee_index: dict[int, int]

    @classmethod
    def load(cls) -> "SpendeeCatalog":
        cards = [dict(item) for item in list_standard_cards()]
        nobles = [dict(item) for item in list_standard_nobles()]

        card_ids_by_signature: dict[CardSignature, int] = {}
        cards_by_tier: dict[int, list[int]] = {1: [], 2: [], 3: []}
        for card in cards:
            signature = CardSignature.from_payload(card)
            if signature in card_ids_by_signature:
                raise ValueError(f"Duplicate card signature detected: {signature}")
            card_id = int(card["id"])
            card_ids_by_signature[signature] = card_id
            cards_by_tier[int(card["tier"])].append(card_id)

        noble_ids_by_signature: dict[NobleSignature, int] = {}
        for noble in nobles:
            signature = NobleSignature.from_payload(noble)
            if signature in noble_ids_by_signature:
                raise ValueError(f"Duplicate noble signature detected: {signature}")
            noble_ids_by_signature[signature] = int(noble["id"])

        spendee_card_to_engine_id: dict[int, int] = {}
        engine_card_to_spendee_index: dict[int, int] = {}
        for entry in SPENDEE_CARD_CONSTANTS:
            spendee_index = int(entry["index"])
            signature = CardSignature(
                tier=int(entry["level"]) + 1,
                points=int(entry["score"]),
                bonus_color=COLORS[int(entry["discount"])],
                cost=tuple(int(v) for v in entry["costs"]),
            )
            try:
                engine_id = card_ids_by_signature[signature]
            except KeyError as exc:
                raise KeyError(f"Unable to map Spendee card index {spendee_index} with signature {signature}") from exc
            spendee_card_to_engine_id[spendee_index] = engine_id
            engine_card_to_spendee_index[engine_id] = spendee_index

        spendee_noble_to_engine_id: dict[int, int] = {}
        engine_noble_to_spendee_index: dict[int, int] = {}
        for entry in SPENDEE_NOBLE_CONSTANTS:
            spendee_index = int(entry["index"])
            signature = NobleSignature(points=int(entry["score"]), requirements=tuple(int(v) for v in entry["costs"]))
            try:
                engine_id = noble_ids_by_signature[signature]
            except KeyError as exc:
                raise KeyError(f"Unable to map Spendee noble index {spendee_index} with signature {signature}") from exc
            spendee_noble_to_engine_id[spendee_index] = engine_id
            engine_noble_to_spendee_index[engine_id] = spendee_index

        return cls(
            cards_by_id={int(card["id"]): card for card in cards},
            nobles_by_id={int(noble["id"]): noble for noble in nobles},
            card_ids_by_signature=card_ids_by_signature,
            noble_ids_by_signature=noble_ids_by_signature,
            card_ids_by_tier={tier: tuple(ids) for tier, ids in cards_by_tier.items()},
            spendee_card_to_engine_id=spendee_card_to_engine_id,
            engine_card_to_spendee_index=engine_card_to_spendee_index,
            spendee_noble_to_engine_id=spendee_noble_to_engine_id,
            engine_noble_to_spendee_index=engine_noble_to_spendee_index,
        )

    def resolve_card_id(self, *, tier: int, points: int, bonus_color: str, cost: dict[str, int]) -> int:
        signature = CardSignature(tier=tier, points=points, bonus_color=bonus_color, cost=_cost_tuple(cost))
        try:
            return self.card_ids_by_signature[signature]
        except KeyError as exc:
            raise KeyError(f"Unknown visible card signature: {signature}") from exc

    def resolve_noble_id(self, *, points: int, requirements: dict[str, int]) -> int:
        signature = NobleSignature(points=points, requirements=_cost_tuple(requirements))
        try:
            return self.noble_ids_by_signature[signature]
        except KeyError as exc:
            raise KeyError(f"Unknown noble signature: {signature}") from exc

    def remaining_card_ids_by_tier(self, used_card_ids: Iterable[int]) -> dict[int, list[int]]:
        used = {int(card_id) for card_id in used_card_ids}
        return {
            tier: [card_id for card_id in card_ids if card_id not in used]
            for tier, card_ids in self.card_ids_by_tier.items()
        }

    def spendee_card_index_to_engine(self, spendee_index: int) -> int:
        return self.spendee_card_to_engine_id[int(spendee_index)]

    def engine_card_id_to_spendee(self, engine_id: int) -> int:
        return self.engine_card_to_spendee_index[int(engine_id)]

    def spendee_noble_index_to_engine(self, spendee_index: int) -> int:
        return self.spendee_noble_to_engine_id[int(spendee_index)]

    def engine_noble_id_to_spendee(self, engine_id: int) -> int:
        return self.engine_noble_to_spendee_index[int(engine_id)]
