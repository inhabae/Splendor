#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "game_logic.h"

struct NativeMCTSResult {
    int action = 0;
    std::array<float, 69> visit_probs{};
    float root_value = 0.0f;
    int num_simulations = 0;
    int root_total_visits = 0;
    int root_nonzero_visit_actions = 0;
    int root_legal_actions = 0;

    pybind11::array_t<float> visit_probs_array() const;
};

struct NativeMCTSNodeData {
    std::array<float, 246> state{};
    std::array<std::uint8_t, 69> mask{};
    bool is_terminal = false;
    int winner = -2;
    int current_player_id = 0;
};

using NativeMCTSNodeDataFn = std::function<NativeMCTSNodeData(const GameState&)>;

NativeMCTSResult run_native_mcts(
    const GameState& root_state,
    const NativeMCTSNodeDataFn& make_node_data,
    pybind11::function evaluator,
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
);
