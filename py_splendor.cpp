#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "game_logic.h"
#include "native_mcts.h"
#include "state_encoder.h"

namespace py = pybind11;

namespace {

constexpr int kActionDim = state_encoder::ACTION_DIM;
constexpr int kStateDim = state_encoder::STATE_DIM;

struct StepResult {
    std::array<float, kStateDim> state{};
    std::array<std::uint8_t, kActionDim> mask{};
    bool is_terminal = false;
    int winner = -2;
    int current_player_id = 0;

    py::array_t<float> state_array() const {
        py::array_t<float> arr(kStateDim);
        std::memcpy(arr.mutable_data(), state.data(), sizeof(float) * static_cast<std::size_t>(kStateDim));
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

    py::array_t<int> debug_raw_state() const {
        ensure_initialized();
        const auto raw = state_encoder::build_raw_state(state_);
        py::array_t<int> arr(kStateDim);
        std::memcpy(arr.mutable_data(), raw.data(), sizeof(int) * static_cast<std::size_t>(kStateDim));
        return arr;
    }

    NativeMCTSResult run_mcts(
        py::function evaluator,
        int turns_taken,
        int num_simulations = 64,
        float c_puct = 1.25f,
        int temperature_moves = 10,
        float temperature = 1.0f,
        float eps = 1e-8f,
        bool root_dirichlet_noise = false,
        float root_dirichlet_epsilon = 0.25f,
        float root_dirichlet_alpha_total = 10.0f,
        int eval_batch_size = 32,
        std::uint64_t rng_seed = 0
    ) const {
        ensure_initialized();
        return run_native_mcts(
            state_,
            std::move(evaluator),
            turns_taken,
            num_simulations,
            c_puct,
            temperature_moves,
            temperature,
            eps,
            root_dirichlet_noise,
            root_dirichlet_epsilon,
            root_dirichlet_alpha_total,
            eval_batch_size,
            rng_seed
        );
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
        out.state = state_encoder::encode_state(state_);
        out.mask = state_encoder::build_legal_mask(state_);
        const auto terminal = state_encoder::build_terminal_metadata(state_);
        out.is_terminal = terminal.is_terminal;
        out.winner = terminal.winner;
        out.current_player_id = terminal.current_player_id;
        return out;
    }

    GameState state_{};
    bool initialized_ = false;
};

}  // namespace

PYBIND11_MODULE(splendor_native, m) {
    m.doc() = "High-throughput pybind11 bindings for Splendor game logic";

    m.attr("ACTION_DIM") = py::int_(kActionDim);
    m.attr("STATE_DIM") = py::int_(kStateDim);

    py::class_<StepResult>(m, "StepResult")
        .def_property_readonly("state", &StepResult::state_array)
        .def_property_readonly("mask", &StepResult::mask_array)
        .def_readonly("is_terminal", &StepResult::is_terminal)
        .def_readonly("winner", &StepResult::winner)
        .def_readonly("current_player_id", &StepResult::current_player_id);

    py::class_<NativeMCTSResult>(m, "NativeMCTSResult")
        .def_property_readonly("visit_probs", &NativeMCTSResult::visit_probs_array)
        .def_readonly("chosen_action_idx", &NativeMCTSResult::chosen_action_idx)
        .def_readonly("root_value", &NativeMCTSResult::root_value);

    py::class_<NativeEnv>(m, "NativeEnv")
        .def(py::init<>())
        .def("reset", &NativeEnv::reset, py::arg("seed") = 0)
        .def("get_state", &NativeEnv::get_state)
        .def("step", &NativeEnv::step, py::arg("action_idx"))
        .def("debug_raw_state", &NativeEnv::debug_raw_state)
        .def(
            "run_mcts",
            &NativeEnv::run_mcts,
            py::arg("evaluator"),
            py::arg("turns_taken"),
            py::arg("num_simulations") = 64,
            py::arg("c_puct") = 1.25f,
            py::arg("temperature_moves") = 10,
            py::arg("temperature") = 1.0f,
            py::arg("eps") = 1e-8f,
            py::arg("root_dirichlet_noise") = false,
            py::arg("root_dirichlet_epsilon") = 0.25f,
            py::arg("root_dirichlet_alpha_total") = 10.0f,
            py::arg("eval_batch_size") = 32,
            py::arg("rng_seed") = static_cast<std::uint64_t>(0)
        );
}
