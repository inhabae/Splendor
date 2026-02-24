from __future__ import annotations

from typing import Sequence

import numpy as np

STATE_DIM = 246
ACTION_DIM = 69
CARD_FEATURE_LEN = 11


# State layout must match splendor_bridge.cpp exactly.
CP_TOKENS_START = 0
CP_BONUSES_START = 6
CP_POINTS_IDX = 11
CP_RESERVED_START = 12
OP_TOKENS_START = 45
OP_BONUSES_START = 51
OP_POINTS_IDX = 56
OP_RESERVED_START = 57
OP_RESERVED_COUNT_IDX = 90
FACEUP_START = 91
BANK_START = 223
NOBLES_START = 229
PHASE_FLAGS_START = 244


def _normalize_token_block(arr: np.ndarray, start: int) -> None:
    # white, blue, green, red, black /4; joker /5
    arr[start : start + 5] /= 4.0
    arr[start + 5] /= 5.0


def _normalize_bonus_block(arr: np.ndarray, start: int) -> None:
    arr[start : start + 5] /= 7.0


def _normalize_card_block(arr: np.ndarray, start: int) -> None:
    # costs (5) /7
    arr[start : start + 5] /= 7.0
    # bonus one-hot (5) unchanged
    # points (1) /5
    arr[start + 10] /= 5.0


def encode_state(raw_state: Sequence[int]) -> np.ndarray:
    """Normalize the 246-length raw bridge state into float32 features."""
    if len(raw_state) != STATE_DIM:
        raise ValueError(f"Expected state length {STATE_DIM}, got {len(raw_state)}")

    out = np.asarray(raw_state, dtype=np.float32).copy()

    # Current player
    _normalize_token_block(out, CP_TOKENS_START)
    _normalize_bonus_block(out, CP_BONUSES_START)
    out[CP_POINTS_IDX] /= 20.0
    for i in range(3):
        _normalize_card_block(out, CP_RESERVED_START + i * CARD_FEATURE_LEN)

    # Opponent
    _normalize_token_block(out, OP_TOKENS_START)
    _normalize_bonus_block(out, OP_BONUSES_START)
    out[OP_POINTS_IDX] /= 20.0
    for i in range(3):
        _normalize_card_block(out, OP_RESERVED_START + i * CARD_FEATURE_LEN)
    out[OP_RESERVED_COUNT_IDX] /= 3.0

    # Board face-up cards (12 x 11)
    for i in range(12):
        _normalize_card_block(out, FACEUP_START + i * CARD_FEATURE_LEN)

    # Bank tokens
    _normalize_token_block(out, BANK_START)

    # Nobles (3 x 5)
    out[NOBLES_START:PHASE_FLAGS_START] /= 4.0

    # Phase flags unchanged (244, 245)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite value encountered during state encoding")

    return out
