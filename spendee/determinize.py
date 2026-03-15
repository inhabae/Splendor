from __future__ import annotations

import random

from .shadow_state import ShadowState


def _sample_unknown_card_id(
    remaining_by_tier: dict[int, list[int]],
    *,
    observed_deck_counts: dict[int, int],
    tier_hint: int | None,
    rng: random.Random,
) -> int:
    if tier_hint is not None:
        pool = remaining_by_tier[int(tier_hint)]
        if len(pool) <= int(observed_deck_counts[int(tier_hint)]):
            raise RuntimeError(f"No surplus cards available for hidden tier {tier_hint}")
        if not pool:
            raise RuntimeError(f"No remaining cards available for hidden tier {tier_hint}")
        choice_index = rng.randrange(len(pool))
        return pool.pop(choice_index)

    candidates: list[tuple[int, int]] = []
    for tier in (1, 2, 3):
        pool = remaining_by_tier[tier]
        surplus = max(len(pool) - int(observed_deck_counts[tier]), 0)
        candidates.extend((tier, idx) for idx in range(surplus))
    if not candidates:
        raise RuntimeError("No remaining cards available for hidden reserved slot")
    tier, idx = candidates[rng.randrange(len(candidates))]
    return remaining_by_tier[tier].pop(idx)


def sample_hydrated_states(
    shadow: ShadowState,
    *,
    samples: int = 8,
    rng: random.Random | None = None,
) -> list[dict[str, object]]:
    observed = shadow.last_observation
    if observed is None:
        raise RuntimeError("ShadowState has not been bootstrapped")

    random_source = rng or random.Random()
    unresolved = shadow.unresolved_hidden_slots(observed)

    base = shadow.build_base_payload(observed)
    used = shadow.used_card_ids(observed)
    remaining = shadow.catalog.remaining_card_ids_by_tier(used)

    has_hidden_information = any(int(count) > 0 for count in observed.deck_counts)
    has_hidden_information = has_hidden_information or any(item.tier > 0 for item in unresolved)
    if not has_hidden_information:
        return [base]
    sample_count = int(samples) if has_hidden_information else 1

    out: list[dict[str, object]] = []
    for _ in range(max(sample_count, 1)):
        remaining_by_tier = {tier: list(card_ids) for tier, card_ids in remaining.items()}
        observed_deck_counts = {1: int(observed.deck_counts[0]), 2: int(observed.deck_counts[1]), 3: int(observed.deck_counts[2])}
        sampled_players = []
        for player_payload in base["players"]:  # type: ignore[index]
            player_dict = dict(player_payload)
            reserved = []
            for entry in player_dict["reserved"]:  # type: ignore[index]
                reserved_entry = dict(entry)
                tier_hint = reserved_entry.pop("tier_hint", None)
                if "card_id" not in reserved_entry:
                    reserved_entry["card_id"] = _sample_unknown_card_id(
                        remaining_by_tier,
                        observed_deck_counts=observed_deck_counts,
                        tier_hint=(int(tier_hint) if tier_hint is not None else None),
                        rng=random_source,
                    )
                reserved.append(reserved_entry)
            player_dict["reserved"] = reserved
            sampled_players.append(player_dict)

        decks = []
        for tier in (1, 2, 3):
            deck_size = int(observed.deck_counts[tier - 1])
            pool = list(remaining_by_tier[tier])
            random_source.shuffle(pool)
            if deck_size > len(pool):
                raise RuntimeError(
                    f"Not enough unknown cards available for tier {tier}: need {deck_size}, have {len(pool)}"
                )
            decks.append(pool[:deck_size])

        payload = dict(base)
        payload["players"] = sampled_players
        payload["deck_card_ids_by_tier"] = decks
        out.append(payload)
    return out


def build_root_determinized_payload(
    shadow: ShadowState,
    *,
    rng: random.Random | None = None,
) -> dict[str, object]:
    observed = shadow.last_observation
    if observed is None:
        raise RuntimeError("ShadowState has not been bootstrapped")

    random_source = rng or random.Random()
    base = shadow.build_base_payload(observed)
    used = shadow.used_card_ids(observed)
    remaining = shadow.catalog.remaining_card_ids_by_tier(used)
    observed_deck_counts = {1: int(observed.deck_counts[0]), 2: int(observed.deck_counts[1]), 3: int(observed.deck_counts[2])}

    remaining_by_tier = {tier: list(card_ids) for tier, card_ids in remaining.items()}
    sampled_players = []
    for player_payload in base["players"]:  # type: ignore[index]
        player_dict = dict(player_payload)
        reserved = []
        for entry in player_dict["reserved"]:  # type: ignore[index]
            reserved_entry = dict(entry)
            tier_hint = reserved_entry.pop("tier_hint", None)
            if "card_id" not in reserved_entry:
                reserved_entry["card_id"] = _sample_unknown_card_id(
                    remaining_by_tier,
                    observed_deck_counts=observed_deck_counts,
                    tier_hint=(int(tier_hint) if tier_hint is not None else None),
                    rng=random_source,
                )
            reserved.append(reserved_entry)
        player_dict["reserved"] = reserved
        sampled_players.append(player_dict)

    payload = dict(base)
    payload["players"] = sampled_players
    payload.pop("deck_card_ids_by_tier", None)
    return payload
