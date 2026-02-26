from __future__ import annotations

from typing import Sequence

import numpy as np

from nn.state_schema import (
    BANK_START,
    CARD_FEATURE_LEN,
    CP_BONUSES_START,
    CP_POINTS_IDX,
    CP_RESERVED_START,
    CP_TOKENS_START,
    FACEUP_START,
    NOBLES_START,
    OP_BONUSES_START,
    OP_POINTS_IDX,
    OP_RESERVED_COUNT_IDX,
    OP_RESERVED_START,
    OP_TOKENS_START,
    PHASE_FLAGS_START,
    STATE_DIM,
)


def _normalize_token_block(arr: np.ndarray, start: int) -> None:
    arr[start : start + 5] /= 4.0
    arr[start + 5] /= 5.0


def _normalize_bonus_block(arr: np.ndarray, start: int) -> None:
    arr[start : start + 5] /= 7.0


def _normalize_card_block(arr: np.ndarray, start: int) -> None:
    arr[start : start + 5] /= 7.0
    arr[start + 10] /= 5.0


def encode_state(raw_state: Sequence[int]) -> np.ndarray:
    """Reference Python encoder for parity tests against the native C++ encoder."""
    if len(raw_state) != STATE_DIM:
        raise ValueError(f"Expected state length {STATE_DIM}, got {len(raw_state)}")

    out = np.asarray(raw_state, dtype=np.float32).copy()

    _normalize_token_block(out, CP_TOKENS_START)
    _normalize_bonus_block(out, CP_BONUSES_START)
    out[CP_POINTS_IDX] /= 20.0
    for i in range(3):
        _normalize_card_block(out, CP_RESERVED_START + i * CARD_FEATURE_LEN)

    _normalize_token_block(out, OP_TOKENS_START)
    _normalize_bonus_block(out, OP_BONUSES_START)
    out[OP_POINTS_IDX] /= 20.0
    for i in range(3):
        _normalize_card_block(out, OP_RESERVED_START + i * CARD_FEATURE_LEN)
    out[OP_RESERVED_COUNT_IDX] /= 3.0

    for i in range(12):
        _normalize_card_block(out, FACEUP_START + i * CARD_FEATURE_LEN)

    _normalize_token_block(out, BANK_START)
    out[NOBLES_START:PHASE_FLAGS_START] /= 4.0

    if not np.isfinite(out).all():
        raise ValueError("Non-finite value encountered during state encoding")
    return out

