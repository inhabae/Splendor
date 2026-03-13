#pragma once

#include <array>
#include <cstdint>

#include "game_logic.h"

namespace state_encoder {

inline constexpr int ACTION_DIM = 69;
inline constexpr int STATE_DIM = 252;
inline constexpr int CARD_FEATURE_LEN = 11;

inline constexpr int CP_TOKENS_START = 0;
inline constexpr int CP_BONUSES_START = 6;
inline constexpr int CP_POINTS_IDX = 11;
inline constexpr int CP_RESERVED_START = 12;
inline constexpr int OP_TOKENS_START = 45;
inline constexpr int OP_BONUSES_START = 51;
inline constexpr int OP_POINTS_IDX = 56;
inline constexpr int PLAYER_INDEX_IDX = 57;
inline constexpr int OPPONENT_RESERVED_SLOT_LEN = 13;
inline constexpr int OPPONENT_RESERVED_CARD_LEN = 11;
inline constexpr int OPPONENT_RESERVED_OCCUPIED_OFFSET = 11;
inline constexpr int OPPONENT_RESERVED_TIER_OFFSET = 12;
inline constexpr int OP_RESERVED_START = 58;
inline constexpr int FACEUP_START = 97;
inline constexpr int BANK_START = 229;
inline constexpr int NOBLES_START = 235;
inline constexpr int PHASE_FLAGS_START = 250;

struct TerminalMetadata {
    bool is_terminal = false;
    int winner = -2;
    int current_player_id = 0;
};

std::array<int, STATE_DIM> build_raw_state(const GameState& state);
std::array<float, STATE_DIM> encode_state(const GameState& state);
std::array<std::uint8_t, ACTION_DIM> build_legal_mask(const GameState& state);
TerminalMetadata build_terminal_metadata(const GameState& state);

}  // namespace state_encoder
