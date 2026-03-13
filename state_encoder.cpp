#include "state_encoder.h"

#include <cstddef>
#include <stdexcept>

namespace state_encoder {
namespace {

void append_card_raw(std::array<int, STATE_DIM>& raw, int& idx, const Card& c) {
    if (idx + CARD_FEATURE_LEN > STATE_DIM) {
        throw std::runtime_error("State encoder overflow while appending card");
    }
    if (c.id == 0) {
        for (int i = 0; i < CARD_FEATURE_LEN; ++i) {
            raw[static_cast<std::size_t>(idx++)] = 0;
        }
        return;
    }

    raw[static_cast<std::size_t>(idx++)] = c.cost.white;
    raw[static_cast<std::size_t>(idx++)] = c.cost.blue;
    raw[static_cast<std::size_t>(idx++)] = c.cost.green;
    raw[static_cast<std::size_t>(idx++)] = c.cost.red;
    raw[static_cast<std::size_t>(idx++)] = c.cost.black;

    raw[static_cast<std::size_t>(idx++)] = (c.color == Color::White) ? 1 : 0;
    raw[static_cast<std::size_t>(idx++)] = (c.color == Color::Blue) ? 1 : 0;
    raw[static_cast<std::size_t>(idx++)] = (c.color == Color::Green) ? 1 : 0;
    raw[static_cast<std::size_t>(idx++)] = (c.color == Color::Red) ? 1 : 0;
    raw[static_cast<std::size_t>(idx++)] = (c.color == Color::Black) ? 1 : 0;

    raw[static_cast<std::size_t>(idx++)] = c.points;
}

void normalize_token_block(std::array<float, STATE_DIM>& out, int start) {
    out[static_cast<std::size_t>(start + 0)] /= 4.0f;
    out[static_cast<std::size_t>(start + 1)] /= 4.0f;
    out[static_cast<std::size_t>(start + 2)] /= 4.0f;
    out[static_cast<std::size_t>(start + 3)] /= 4.0f;
    out[static_cast<std::size_t>(start + 4)] /= 4.0f;
    out[static_cast<std::size_t>(start + 5)] /= 5.0f;
}

void normalize_bonus_block(std::array<float, STATE_DIM>& out, int start) {
    for (int i = 0; i < 5; ++i) {
        out[static_cast<std::size_t>(start + i)] /= 7.0f;
    }
}

void normalize_card_block(std::array<float, STATE_DIM>& out, int start) {
    for (int i = 0; i < 5; ++i) {
        out[static_cast<std::size_t>(start + i)] /= 7.0f;
    }
    out[static_cast<std::size_t>(start + 10)] /= 5.0f;
}

void append_opponent_reserved_slot_raw(std::array<int, STATE_DIM>& raw, int& idx, const ReservedCard* reserved) {
    const Card kEmptyCard{};
    if (idx + OPPONENT_RESERVED_SLOT_LEN > STATE_DIM) {
        throw std::runtime_error("State encoder overflow while appending opponent reserved slot");
    }
    if (reserved == nullptr) {
        append_card_raw(raw, idx, kEmptyCard);
        raw[static_cast<std::size_t>(idx++)] = 0;
        raw[static_cast<std::size_t>(idx++)] = 0;
        return;
    }

    if (reserved->is_public) {
        append_card_raw(raw, idx, reserved->card);
    } else {
        append_card_raw(raw, idx, kEmptyCard);
    }
    raw[static_cast<std::size_t>(idx++)] = 1;
    const int level = reserved->card.level;
    if (level < 1 || level > 3) {
        throw std::runtime_error("State encoder encountered opponent reserved card with invalid tier");
    }
    raw[static_cast<std::size_t>(idx++)] = level;
}

}  // namespace

std::array<int, STATE_DIM> build_raw_state(const GameState& state) {
    std::array<int, STATE_DIM> raw{};
    const Card kEmptyCard{};
    int idx = 0;

    const int cur = state.current_player;
    const int opp = 1 - cur;
    const Player& cp = state.players[cur];
    const Player& op = state.players[opp];

    raw[static_cast<std::size_t>(idx++)] = cp.tokens.white;
    raw[static_cast<std::size_t>(idx++)] = cp.tokens.blue;
    raw[static_cast<std::size_t>(idx++)] = cp.tokens.green;
    raw[static_cast<std::size_t>(idx++)] = cp.tokens.red;
    raw[static_cast<std::size_t>(idx++)] = cp.tokens.black;
    raw[static_cast<std::size_t>(idx++)] = cp.tokens.joker;

    raw[static_cast<std::size_t>(idx++)] = cp.bonuses.white;
    raw[static_cast<std::size_t>(idx++)] = cp.bonuses.blue;
    raw[static_cast<std::size_t>(idx++)] = cp.bonuses.green;
    raw[static_cast<std::size_t>(idx++)] = cp.bonuses.red;
    raw[static_cast<std::size_t>(idx++)] = cp.bonuses.black;

    raw[static_cast<std::size_t>(idx++)] = cp.points;

    for (int i = 0; i < 3; ++i) {
        if (i < static_cast<int>(cp.reserved.size())) {
            append_card_raw(raw, idx, cp.reserved[static_cast<std::size_t>(i)].card);
        } else {
            append_card_raw(raw, idx, kEmptyCard);
        }
    }

    raw[static_cast<std::size_t>(idx++)] = op.tokens.white;
    raw[static_cast<std::size_t>(idx++)] = op.tokens.blue;
    raw[static_cast<std::size_t>(idx++)] = op.tokens.green;
    raw[static_cast<std::size_t>(idx++)] = op.tokens.red;
    raw[static_cast<std::size_t>(idx++)] = op.tokens.black;
    raw[static_cast<std::size_t>(idx++)] = op.tokens.joker;

    raw[static_cast<std::size_t>(idx++)] = op.bonuses.white;
    raw[static_cast<std::size_t>(idx++)] = op.bonuses.blue;
    raw[static_cast<std::size_t>(idx++)] = op.bonuses.green;
    raw[static_cast<std::size_t>(idx++)] = op.bonuses.red;
    raw[static_cast<std::size_t>(idx++)] = op.bonuses.black;

    raw[static_cast<std::size_t>(idx++)] = op.points;
    raw[static_cast<std::size_t>(idx++)] = cur;

    for (int i = 0; i < 3; ++i) {
        if (i < static_cast<int>(op.reserved.size())) {
            append_opponent_reserved_slot_raw(raw, idx, &op.reserved[static_cast<std::size_t>(i)]);
        } else {
            append_opponent_reserved_slot_raw(raw, idx, nullptr);
        }
    }

    for (int tier = 0; tier < 3; ++tier) {
        for (int slot = 0; slot < 4; ++slot) {
            append_card_raw(raw, idx, state.faceup[tier][static_cast<std::size_t>(slot)]);
        }
    }

    raw[static_cast<std::size_t>(idx++)] = state.bank.white;
    raw[static_cast<std::size_t>(idx++)] = state.bank.blue;
    raw[static_cast<std::size_t>(idx++)] = state.bank.green;
    raw[static_cast<std::size_t>(idx++)] = state.bank.red;
    raw[static_cast<std::size_t>(idx++)] = state.bank.black;
    raw[static_cast<std::size_t>(idx++)] = state.bank.joker;

    for (int i = 0; i < 3; ++i) {
        if (i < state.noble_count) {
            const Noble& n = state.available_nobles[static_cast<std::size_t>(i)];
            raw[static_cast<std::size_t>(idx++)] = n.requirements.white;
            raw[static_cast<std::size_t>(idx++)] = n.requirements.blue;
            raw[static_cast<std::size_t>(idx++)] = n.requirements.green;
            raw[static_cast<std::size_t>(idx++)] = n.requirements.red;
            raw[static_cast<std::size_t>(idx++)] = n.requirements.black;
        } else {
            for (int j = 0; j < 5; ++j) {
                raw[static_cast<std::size_t>(idx++)] = 0;
            }
        }
    }

    raw[static_cast<std::size_t>(idx++)] = state.is_return_phase ? 1 : 0;
    raw[static_cast<std::size_t>(idx++)] = state.is_noble_choice_phase ? 1 : 0;

    if (idx != STATE_DIM) {
        throw std::runtime_error("State encoder produced unexpected length");
    }
    return raw;
}

std::array<float, STATE_DIM> encode_state(const GameState& state) {
    const std::array<int, STATE_DIM> raw = build_raw_state(state);
    std::array<float, STATE_DIM> out{};
    for (int i = 0; i < STATE_DIM; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<float>(raw[static_cast<std::size_t>(i)]);
    }

    normalize_token_block(out, CP_TOKENS_START);
    normalize_bonus_block(out, CP_BONUSES_START);
    out[static_cast<std::size_t>(CP_POINTS_IDX)] /= 20.0f;
    for (int i = 0; i < 3; ++i) {
        normalize_card_block(out, CP_RESERVED_START + i * CARD_FEATURE_LEN);
    }

    normalize_token_block(out, OP_TOKENS_START);
    normalize_bonus_block(out, OP_BONUSES_START);
    out[static_cast<std::size_t>(OP_POINTS_IDX)] /= 20.0f;
    for (int i = 0; i < 3; ++i) {
        normalize_card_block(out, OP_RESERVED_START + i * OPPONENT_RESERVED_SLOT_LEN);
        out[static_cast<std::size_t>(OP_RESERVED_START + i * OPPONENT_RESERVED_SLOT_LEN + OPPONENT_RESERVED_TIER_OFFSET)] /= 3.0f;
    }

    for (int i = 0; i < 12; ++i) {
        normalize_card_block(out, FACEUP_START + i * CARD_FEATURE_LEN);
    }

    normalize_token_block(out, BANK_START);
    for (int i = 0; i < 15; ++i) {
        out[static_cast<std::size_t>(NOBLES_START + i)] /= 4.0f;
    }

    out[static_cast<std::size_t>(PHASE_FLAGS_START)] =
        raw[static_cast<std::size_t>(PHASE_FLAGS_START)] != 0 ? 1.0f : 0.0f;
    out[static_cast<std::size_t>(PHASE_FLAGS_START + 1)] =
        raw[static_cast<std::size_t>(PHASE_FLAGS_START + 1)] != 0 ? 1.0f : 0.0f;

    return out;
}

std::array<std::uint8_t, ACTION_DIM> build_legal_mask(const GameState& state) {
    std::array<std::uint8_t, ACTION_DIM> out{};
    const auto mask = getValidMoveMask(state);
    for (int i = 0; i < ACTION_DIM; ++i) {
        out[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(mask[static_cast<std::size_t>(i)] != 0 ? 1 : 0);
    }
    return out;
}

TerminalMetadata build_terminal_metadata(const GameState& state) {
    TerminalMetadata meta;
    meta.is_terminal = isGameOver(state);
    meta.winner = meta.is_terminal ? determineWinner(state) : -2;
    meta.current_player_id = state.current_player;
    return meta;
}

}  // namespace state_encoder
