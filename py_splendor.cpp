#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include "game_logic.h"

namespace py = pybind11;

namespace {

constexpr int kActionDim = 69;
constexpr int kStateDim = 246;
constexpr int kCardFeatureLen = 11;

constexpr int kCpTokensStart = 0;
constexpr int kCpBonusesStart = 6;
constexpr int kCpPointsIdx = 11;
constexpr int kCpReservedStart = 12;
constexpr int kOpTokensStart = 45;
constexpr int kOpBonusesStart = 51;
constexpr int kOpPointsIdx = 56;
constexpr int kOpReservedStart = 57;
constexpr int kOpReservedCountIdx = 90;
constexpr int kFaceupStart = 91;
constexpr int kBankStart = 223;
constexpr int kNoblesStart = 229;
constexpr int kPhaseFlagsStart = 244;

void append_card_raw(std::array<int, kStateDim>& raw, int& idx, const Card& c) {
    if (idx + kCardFeatureLen > kStateDim) {
        throw std::runtime_error("State encoder overflow while appending card");
    }
    if (c.id == 0) {
        for (int i = 0; i < kCardFeatureLen; ++i) {
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

void normalize_token_block(std::array<float, kStateDim>& out, int start) {
    out[static_cast<std::size_t>(start + 0)] /= 4.0f;
    out[static_cast<std::size_t>(start + 1)] /= 4.0f;
    out[static_cast<std::size_t>(start + 2)] /= 4.0f;
    out[static_cast<std::size_t>(start + 3)] /= 4.0f;
    out[static_cast<std::size_t>(start + 4)] /= 4.0f;
    out[static_cast<std::size_t>(start + 5)] /= 5.0f;
}

void normalize_bonus_block(std::array<float, kStateDim>& out, int start) {
    for (int i = 0; i < 5; ++i) {
        out[static_cast<std::size_t>(start + i)] /= 7.0f;
    }
}

void normalize_card_block(std::array<float, kStateDim>& out, int start) {
    for (int i = 0; i < 5; ++i) {
        out[static_cast<std::size_t>(start + i)] /= 7.0f;
    }
    out[static_cast<std::size_t>(start + 10)] /= 5.0f;
}

std::array<int, kStateDim> build_raw_state(const GameState& state) {
    std::array<int, kStateDim> raw{};
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

    for (int i = 0; i < 3; ++i) {
        if (i < static_cast<int>(op.reserved.size()) &&
            op.reserved[static_cast<std::size_t>(i)].is_public) {
            append_card_raw(raw, idx, op.reserved[static_cast<std::size_t>(i)].card);
        } else {
            append_card_raw(raw, idx, kEmptyCard);
        }
    }
    raw[static_cast<std::size_t>(idx++)] = static_cast<int>(op.reserved.size());

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

    if (idx != kStateDim) {
        throw std::runtime_error("State encoder produced unexpected length");
    }
    return raw;
}

std::array<float, kStateDim> encode_raw_state(const std::array<int, kStateDim>& raw) {
    std::array<float, kStateDim> out{};
    for (int i = 0; i < kStateDim; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<float>(raw[static_cast<std::size_t>(i)]);
    }

    normalize_token_block(out, kCpTokensStart);
    normalize_bonus_block(out, kCpBonusesStart);
    out[static_cast<std::size_t>(kCpPointsIdx)] /= 20.0f;
    for (int i = 0; i < 3; ++i) {
        normalize_card_block(out, kCpReservedStart + i * kCardFeatureLen);
    }

    normalize_token_block(out, kOpTokensStart);
    normalize_bonus_block(out, kOpBonusesStart);
    out[static_cast<std::size_t>(kOpPointsIdx)] /= 20.0f;
    for (int i = 0; i < 3; ++i) {
        normalize_card_block(out, kOpReservedStart + i * kCardFeatureLen);
    }
    out[static_cast<std::size_t>(kOpReservedCountIdx)] /= 3.0f;

    for (int i = 0; i < 12; ++i) {
        normalize_card_block(out, kFaceupStart + i * kCardFeatureLen);
    }

    normalize_token_block(out, kBankStart);

    for (int i = kNoblesStart; i < kPhaseFlagsStart; ++i) {
        out[static_cast<std::size_t>(i)] /= 4.0f;
    }

    return out;
}

struct StepResult {
    std::array<float, kStateDim> state{};
    std::array<int, kStateDim> raw_state{};
    std::array<std::uint8_t, kActionDim> mask{};
    bool is_terminal = false;
    int winner = -2;
    bool is_return_phase = false;
    bool is_noble_choice_phase = false;
    int current_player_id = 0;

    py::array_t<float> state_array() const {
        py::array_t<float> arr(kStateDim);
        auto buf = arr.mutable_data();
        std::memcpy(buf, state.data(), sizeof(float) * static_cast<std::size_t>(kStateDim));
        return arr;
    }

    py::array_t<int> raw_state_array() const {
        py::array_t<int> arr(kStateDim);
        auto buf = arr.mutable_data();
        std::memcpy(buf, raw_state.data(), sizeof(int) * static_cast<std::size_t>(kStateDim));
        return arr;
    }

    py::array_t<bool> mask_array() const {
        py::array_t<bool> arr(kActionDim);
        auto view = arr.mutable_unchecked<1>();
        for (int i = 0; i < kActionDim; ++i) {
            view(i) = (mask[static_cast<std::size_t>(i)] != 0);
        }
        return arr;
    }
};

class NativeEnv {
public:
    NativeEnv() = default;

    StepResult reset(unsigned int seed = 0) {
        initializeGame(state_, seed);
        initialized_ = true;
        snapshots_.clear();
        next_snapshot_id_ = 1;
        return make_step_result();
    }

    StepResult get_state() const {
        ensure_initialized();
        return make_step_result();
    }

    StepResult step(int action_idx) {
        ensure_initialized();
        validate_action_idx(action_idx);
        const auto mask = getValidMoveMask(state_);
        if (!mask[static_cast<std::size_t>(action_idx)]) {
            throw std::invalid_argument("action is not valid in current state");
        }
        applyMove(state_, actionIndexToMove(action_idx));
        return make_step_result();
    }

    int snapshot() {
        ensure_initialized();
        const int id = next_snapshot_id_++;
        snapshots_[id] = state_;
        return id;
    }

    StepResult restore_snapshot(int snapshot_id) {
        ensure_initialized();
        const auto it = snapshots_.find(snapshot_id);
        if (it == snapshots_.end()) {
            throw std::out_of_range("Unknown snapshot_id");
        }
        state_ = it->second;
        return make_step_result();
    }

    void drop_snapshot(int snapshot_id) {
        ensure_initialized();
        const auto erased = snapshots_.erase(snapshot_id);
        if (erased == 0U) {
            throw std::out_of_range("Unknown snapshot_id");
        }
    }

    py::array_t<int> debug_raw_state() const {
        ensure_initialized();
        const auto raw = build_raw_state(state_);
        py::array_t<int> arr(kStateDim);
        std::memcpy(arr.mutable_data(), raw.data(), sizeof(int) * static_cast<std::size_t>(kStateDim));
        return arr;
    }

private:
    void ensure_initialized() const {
        if (!initialized_) {
            throw std::runtime_error("Game not initialized; call reset() first");
        }
    }

    static void validate_action_idx(int action_idx) {
        if (action_idx < 0 || action_idx >= kActionDim) {
            throw std::out_of_range("action_idx must be in [0, 68]");
        }
    }

    StepResult make_step_result() const {
        StepResult out;
        out.raw_state = build_raw_state(state_);
        out.state = encode_raw_state(out.raw_state);
        const auto mask = getValidMoveMask(state_);
        for (int i = 0; i < kActionDim; ++i) {
            out.mask[static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>(mask[static_cast<std::size_t>(i)] != 0 ? 1 : 0);
        }
        out.is_terminal = isGameOver(state_);
        out.winner = out.is_terminal ? determineWinner(state_) : -2;
        out.is_return_phase = state_.is_return_phase;
        out.is_noble_choice_phase = state_.is_noble_choice_phase;
        out.current_player_id = state_.current_player;
        return out;
    }

    GameState state_{};
    bool initialized_ = false;
    std::unordered_map<int, GameState> snapshots_;
    int next_snapshot_id_ = 1;
};

}  // namespace

PYBIND11_MODULE(splendor_native, m) {
    m.doc() = "High-throughput pybind11 bindings for Splendor game logic";

    m.attr("ACTION_DIM") = py::int_(kActionDim);
    m.attr("STATE_DIM") = py::int_(kStateDim);

    py::class_<StepResult>(m, "StepResult")
        .def_property_readonly("state", &StepResult::state_array)
        .def_property_readonly("raw_state", &StepResult::raw_state_array)
        .def_property_readonly("mask", &StepResult::mask_array)
        .def_readonly("is_terminal", &StepResult::is_terminal)
        .def_readonly("winner", &StepResult::winner)
        .def_readonly("is_return_phase", &StepResult::is_return_phase)
        .def_readonly("is_noble_choice_phase", &StepResult::is_noble_choice_phase)
        .def_readonly("current_player_id", &StepResult::current_player_id);

    py::class_<NativeEnv>(m, "NativeEnv")
        .def(py::init<>())
        .def("reset", &NativeEnv::reset, py::arg("seed") = 0)
        .def("get_state", &NativeEnv::get_state)
        .def("step", &NativeEnv::step, py::arg("action_idx"))
        .def("snapshot", &NativeEnv::snapshot)
        .def("restore_snapshot", &NativeEnv::restore_snapshot, py::arg("snapshot_id"))
        .def("drop_snapshot", &NativeEnv::drop_snapshot, py::arg("snapshot_id"))
        .def("debug_raw_state", &NativeEnv::debug_raw_state);

    m.attr("Game") = m.attr("NativeEnv");
}
