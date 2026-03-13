from __future__ import annotations

# Canonical state/action schema shared across Python runtime code and tests.
# The production encoder lives in py_splendor.cpp and must match these constants.

STATE_DIM = 252
ACTION_DIM = 69
CARD_FEATURE_LEN = 11
OPPONENT_RESERVED_SLOT_LEN = 13

# State layout offsets (side-to-move canonical perspective)
CP_TOKENS_START = 0
CP_BONUSES_START = 6
CP_POINTS_IDX = 11
CP_RESERVED_START = 12
OP_TOKENS_START = 45
OP_BONUSES_START = 51
OP_POINTS_IDX = 56
PLAYER_INDEX_IDX = 57
OP_RESERVED_START = 58
OP_RESERVED_IS_OCCUPIED_OFFSET = 11
OP_RESERVED_TIER_OFFSET = 12
FACEUP_START = 97
BANK_START = 229
NOBLES_START = 235
PHASE_FLAGS_START = 250
