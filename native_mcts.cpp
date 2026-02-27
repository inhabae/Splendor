#include "native_mcts.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kActionDim = 69;
constexpr int kStateDim = 246;

struct MCTSNode {
    std::array<float, kActionDim> priors{};
    std::array<int, kActionDim> visit_count{};
    std::array<float, kActionDim> value_sum{};
    std::array<int, kActionDim> child_index{};
    bool expanded = false;
    bool pending_eval = false;

    MCTSNode() {
        child_index.fill(-1);
    }
};

struct PathStep {
    int node_index = -1;
    int action = -1;
    bool same_player = false;
};

struct PendingLeafEval {
    int node_index = -1;
    std::array<float, kStateDim> state{};
    std::array<std::uint8_t, kActionDim> mask{};
    std::vector<PathStep> path;
};

struct ReadyBackup {
    float value = 0.0f;
    std::vector<PathStep> path;
};

float winner_to_value_for_player(int winner, int player_id) {
    if (winner == -1) {
        return 0.0f;
    }
    if (winner != 0 && winner != 1) {
        throw std::runtime_error("Unexpected winner value in native MCTS");
    }
    return winner == player_id ? 1.0f : -1.0f;
}

template <typename Rng>
void apply_dirichlet_root_noise(
    std::array<float, kActionDim>& priors,
    const std::array<std::uint8_t, kActionDim>& mask,
    float epsilon,
    float alpha_total,
    Rng& rng
) {
    if (epsilon <= 0.0f) {
        return;
    }
    std::vector<int> legal;
    legal.reserve(kActionDim);
    for (int i = 0; i < kActionDim; ++i) {
        if (mask[static_cast<std::size_t>(i)] != 0) {
            legal.push_back(i);
        } else {
            priors[static_cast<std::size_t>(i)] = 0.0f;
        }
    }
    if (legal.size() < 2U) {
        return;
    }

    const double alpha = static_cast<double>(alpha_total) / static_cast<double>(legal.size());
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise(legal.size(), 0.0);
    double noise_sum = 0.0;
    for (std::size_t i = 0; i < legal.size(); ++i) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }
    if (!(noise_sum > 0.0) || !std::isfinite(noise_sum)) {
        for (double& x : noise) {
            x = 1.0 / static_cast<double>(legal.size());
        }
    } else {
        for (double& x : noise) {
            x /= noise_sum;
        }
    }

    double mixed_sum = 0.0;
    for (std::size_t i = 0; i < legal.size(); ++i) {
        const int a = legal[i];
        const double mixed =
            (1.0 - static_cast<double>(epsilon)) * static_cast<double>(priors[static_cast<std::size_t>(a)]) +
            static_cast<double>(epsilon) * noise[i];
        priors[static_cast<std::size_t>(a)] = static_cast<float>(mixed);
        mixed_sum += mixed;
    }
    if (!(mixed_sum > 0.0) || !std::isfinite(mixed_sum)) {
        const float u = 1.0f / static_cast<float>(legal.size());
        for (int a : legal) {
            priors[static_cast<std::size_t>(a)] = u;
        }
    } else {
        for (int a : legal) {
            priors[static_cast<std::size_t>(a)] = static_cast<float>(
                static_cast<double>(priors[static_cast<std::size_t>(a)]) / mixed_sum
            );
        }
    }
}

int select_puct_action(
    const std::vector<MCTSNode>& nodes,
    int node_index,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    float c_puct,
    float eps
) {
    const MCTSNode& node = nodes[static_cast<std::size_t>(node_index)];
    float parent_n = 0.0f;
    for (int i = 0; i < kActionDim; ++i) {
        parent_n += static_cast<float>(node.visit_count[static_cast<std::size_t>(i)]);
    }
    const float sqrt_parent = std::sqrt(parent_n + eps);

    int best_action = -1;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int action = 0; action < kActionDim; ++action) {
        if (legal_mask[static_cast<std::size_t>(action)] == 0) {
            continue;
        }
        const int child_idx = node.child_index[static_cast<std::size_t>(action)];
        if (child_idx >= 0 && nodes[static_cast<std::size_t>(child_idx)].pending_eval) {
            continue;
        }
        const float n = static_cast<float>(node.visit_count[static_cast<std::size_t>(action)]);
        const float q = (n <= 0.0f) ? 0.0f : (node.value_sum[static_cast<std::size_t>(action)] / n);
        const float u = c_puct * node.priors[static_cast<std::size_t>(action)] * sqrt_parent / (1.0f + n);
        const float score = q + u;
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    return best_action;
}

template <typename Rng>
void sample_root_decks(GameState& state, Rng& rng) {
    for (int tier = 0; tier < 3; ++tier) {
        std::shuffle(state.deck[tier].begin(), state.deck[tier].end(), rng);
    }
}

template <typename Rng>
int sample_action_from_visits(
    const std::array<float, kActionDim>& visit_probs,
    const std::array<std::uint8_t, kActionDim>& legal_mask,
    int turns_taken,
    int temperature_moves,
    float temperature,
    Rng& rng
) {
    std::vector<int> legal;
    legal.reserve(kActionDim);
    for (int i = 0; i < kActionDim; ++i) {
        if (legal_mask[static_cast<std::size_t>(i)] != 0) {
            legal.push_back(i);
        }
    }
    if (legal.empty()) {
        throw std::runtime_error("No legal actions for final native MCTS action selection");
    }
    if (turns_taken >= temperature_moves) {
        int best_action = legal.front();
        float best_prob = visit_probs[static_cast<std::size_t>(best_action)];
        for (int a : legal) {
            const float p = visit_probs[static_cast<std::size_t>(a)];
            if (p > best_prob) {
                best_prob = p;
                best_action = a;
            }
        }
        return best_action;
    }

    std::vector<double> weights;
    weights.reserve(legal.size());
    if (temperature <= 0.0f) {
        for (int a : legal) {
            weights.push_back(static_cast<double>(visit_probs[static_cast<std::size_t>(a)]));
        }
    } else {
        for (int a : legal) {
            const double base = static_cast<double>(visit_probs[static_cast<std::size_t>(a)]);
            weights.push_back(temperature == 1.0f ? base : std::pow(base, 1.0 / static_cast<double>(temperature)));
        }
    }
    double weight_sum = 0.0;
    for (double w : weights) {
        weight_sum += w;
    }
    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        weights.assign(legal.size(), 1.0);
    }
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return legal[static_cast<std::size_t>(dist(rng))];
}

}  // namespace

pybind11::array_t<float> NativeMCTSResult::visit_probs_array() const {
    pybind11::array_t<float> arr(kActionDim);
    std::memcpy(arr.mutable_data(), visit_probs.data(), sizeof(float) * static_cast<std::size_t>(kActionDim));
    return arr;
}

NativeMCTSResult run_native_mcts(
    const GameState& root_state,
    const NativeMCTSNodeDataFn& make_node_data,
    pybind11::function evaluator,
    int turns_taken,
    int num_simulations,
    float c_puct,
    int temperature_moves,
    float temperature,
    float eps,
    bool root_dirichlet_noise,
    float root_dirichlet_epsilon,
    float root_dirichlet_alpha_total,
    int eval_batch_size,
    std::uint64_t rng_seed
) {
    if (num_simulations <= 0) {
        throw std::invalid_argument("num_simulations must be positive");
    }
    if (!(root_dirichlet_epsilon >= 0.0f && root_dirichlet_epsilon <= 1.0f)) {
        throw std::invalid_argument("root_dirichlet_epsilon must be in [0,1]");
    }
    if (!(root_dirichlet_alpha_total > 0.0f)) {
        throw std::invalid_argument("root_dirichlet_alpha_total must be positive");
    }
    if (eval_batch_size <= 0) {
        throw std::invalid_argument("eval_batch_size must be positive");
    }

    const NativeMCTSNodeData root_data = make_node_data(root_state);
    if (root_data.is_terminal) {
        throw std::invalid_argument("run_mcts called on terminal state");
    }

    bool has_legal = false;
    for (int i = 0; i < kActionDim; ++i) {
        if (root_data.mask[static_cast<std::size_t>(i)] != 0) {
            has_legal = true;
            break;
        }
    }
    if (!has_legal) {
        throw std::invalid_argument("MCTS root has no legal actions");
    }

    std::vector<MCTSNode> nodes;
    nodes.reserve(static_cast<std::size_t>(num_simulations + 1));
    nodes.emplace_back();

    std::mt19937_64 rng(rng_seed);
    bool root_noise_applied = false;
    int completed = 0;

    while (completed < num_simulations) {
        const int target_batch = std::min(eval_batch_size, num_simulations - completed);
        std::vector<PendingLeafEval> pending;
        pending.reserve(static_cast<std::size_t>(target_batch));
        std::vector<ReadyBackup> backups;
        backups.reserve(static_cast<std::size_t>(target_batch));

        {
            pybind11::gil_scoped_release release;
            for (int slot = 0; slot < target_batch; ++slot) {
                GameState sim_state = root_state;
                sample_root_decks(sim_state, rng);
                int node_index = 0;
                std::vector<PathStep> path;
                path.reserve(64);

                while (true) {
                    const NativeMCTSNodeData node_data = make_node_data(sim_state);
                    MCTSNode& node = nodes[static_cast<std::size_t>(node_index)];
                    if (node_data.is_terminal) {
                        ReadyBackup ready;
                        ready.value = winner_to_value_for_player(node_data.winner, node_data.current_player_id);
                        ready.path = std::move(path);
                        backups.push_back(std::move(ready));
                        break;
                    }
                    if (!node.expanded) {
                        if (node.pending_eval) {
                            break;
                        }
                        node.pending_eval = true;
                        PendingLeafEval req;
                        req.node_index = node_index;
                        req.state = node_data.state;
                        req.mask = node_data.mask;
                        req.path = std::move(path);
                        pending.push_back(std::move(req));
                        break;
                    }

                    const int action = select_puct_action(
                        nodes,
                        node_index,
                        node_data.mask,
                        c_puct,
                        eps
                    );
                    if (action < 0) {
                        break;
                    }

                    int child_idx = node.child_index[static_cast<std::size_t>(action)];
                    if (child_idx < 0) {
                        child_idx = static_cast<int>(nodes.size());
                        nodes.emplace_back();
                        nodes[static_cast<std::size_t>(node_index)].child_index[static_cast<std::size_t>(action)] =
                            child_idx;
                    }

                    const int parent_to_play = node_data.current_player_id;
                    applyMove(sim_state, actionIndexToMove(action));
                    const NativeMCTSNodeData child_data = make_node_data(sim_state);
                    const bool same_player = child_data.current_player_id == parent_to_play;
                    path.push_back(PathStep{node_index, action, same_player});
                    node_index = child_idx;
                }
            }
        }

        if (!pending.empty()) {
            const pybind11::ssize_t batch = static_cast<pybind11::ssize_t>(pending.size());
            pybind11::array_t<float> states({batch, static_cast<pybind11::ssize_t>(kStateDim)});
            pybind11::array_t<bool> masks({batch, static_cast<pybind11::ssize_t>(kActionDim)});
            auto states_view = states.mutable_unchecked<2>();
            auto masks_view = masks.mutable_unchecked<2>();
            for (pybind11::ssize_t i = 0; i < batch; ++i) {
                const PendingLeafEval& req = pending[static_cast<std::size_t>(i)];
                for (int j = 0; j < kStateDim; ++j) {
                    states_view(i, j) = req.state[static_cast<std::size_t>(j)];
                }
                for (int j = 0; j < kActionDim; ++j) {
                    masks_view(i, j) = (req.mask[static_cast<std::size_t>(j)] != 0);
                }
            }

            pybind11::object out_obj = evaluator(states, masks);
            pybind11::tuple out = out_obj.cast<pybind11::tuple>();
            if (out.size() != 2) {
                throw std::runtime_error("MCTS evaluator must return (priors, values)");
            }

            auto priors_arr =
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>::ensure(out[0]);
            auto values_arr =
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>::ensure(out[1]);
            if (!priors_arr || !values_arr) {
                throw std::runtime_error("MCTS evaluator outputs must be float arrays");
            }
            if (priors_arr.ndim() != 2 ||
                priors_arr.shape(0) != batch ||
                priors_arr.shape(1) != static_cast<pybind11::ssize_t>(kActionDim)) {
                throw std::runtime_error("MCTS evaluator priors must have shape (B, ACTION_DIM)");
            }
            if (values_arr.ndim() != 1 || values_arr.shape(0) != batch) {
                throw std::runtime_error("MCTS evaluator values must have shape (B,)");
            }

            auto priors_view = priors_arr.unchecked<2>();
            auto values_view = values_arr.unchecked<1>();

            for (pybind11::ssize_t i = 0; i < batch; ++i) {
                PendingLeafEval& req = pending[static_cast<std::size_t>(i)];
                MCTSNode& node = nodes[static_cast<std::size_t>(req.node_index)];
                int legal_count = 0;
                bool has_finite_legal_score = false;
                float max_score = -std::numeric_limits<float>::infinity();
                for (int a = 0; a < kActionDim; ++a) {
                    if (req.mask[static_cast<std::size_t>(a)] == 0) {
                        node.priors[static_cast<std::size_t>(a)] = 0.0f;
                        continue;
                    }
                    ++legal_count;
                    const float s = priors_view(i, a);  // Python returns policy scores/logits.
                    if (std::isfinite(static_cast<double>(s))) {
                        if (!has_finite_legal_score || s > max_score) {
                            max_score = s;
                        }
                        has_finite_legal_score = true;
                    }
                }

                double sum = 0.0;
                if (has_finite_legal_score) {
                    for (int a = 0; a < kActionDim; ++a) {
                        if (req.mask[static_cast<std::size_t>(a)] == 0) {
                            node.priors[static_cast<std::size_t>(a)] = 0.0f;
                            continue;
                        }
                        const float s = priors_view(i, a);
                        if (!std::isfinite(static_cast<double>(s))) {
                            node.priors[static_cast<std::size_t>(a)] = 0.0f;
                            continue;
                        }
                        const float w = std::exp(s - max_score);
                        node.priors[static_cast<std::size_t>(a)] = w;
                        sum += static_cast<double>(w);
                    }
                }

                if (!has_finite_legal_score || !(sum > 0.0) || !std::isfinite(sum)) {
                    const float u = 1.0f / static_cast<float>(legal_count);
                    for (int a = 0; a < kActionDim; ++a) {
                        node.priors[static_cast<std::size_t>(a)] =
                            (req.mask[static_cast<std::size_t>(a)] != 0) ? u : 0.0f;
                    }
                } else {
                    for (int a = 0; a < kActionDim; ++a) {
                        if (req.mask[static_cast<std::size_t>(a)] != 0) {
                            node.priors[static_cast<std::size_t>(a)] = static_cast<float>(
                                static_cast<double>(node.priors[static_cast<std::size_t>(a)]) / sum
                            );
                        } else {
                            node.priors[static_cast<std::size_t>(a)] = 0.0f;
                        }
                    }
                }

                const float value = values_view(i);
                if (!std::isfinite(static_cast<double>(value))) {
                    throw std::runtime_error("MCTS evaluator values contain non-finite entries");
                }
                node.expanded = true;
                node.pending_eval = false;

                if (req.node_index == 0 && !root_noise_applied && root_dirichlet_noise) {
                    apply_dirichlet_root_noise(
                        node.priors,
                        req.mask,
                        root_dirichlet_epsilon,
                        root_dirichlet_alpha_total,
                        rng
                    );
                    root_noise_applied = true;
                }

                ReadyBackup ready;
                ready.value = value;
                ready.path = std::move(req.path);
                backups.push_back(std::move(ready));
            }
        }

        if (backups.empty()) {
            throw std::runtime_error("Native MCTS made no progress while gathering/evaluating leaves");
        }

        {
            pybind11::gil_scoped_release release;
            for (ReadyBackup& ready : backups) {
                float value = ready.value;
                for (auto it = ready.path.rbegin(); it != ready.path.rend(); ++it) {
                    const float backed = it->same_player ? value : -value;
                    MCTSNode& parent = nodes[static_cast<std::size_t>(it->node_index)];
                    parent.visit_count[static_cast<std::size_t>(it->action)] += 1;
                    parent.value_sum[static_cast<std::size_t>(it->action)] += backed;
                    value = backed;
                }
            }
        }

        completed += static_cast<int>(backups.size());
    }

    const MCTSNode& root = nodes[0];
    NativeMCTSResult result;
    result.num_simulations = num_simulations;

    double total_visits = 0.0;
    for (int a = 0; a < kActionDim; ++a) {
        total_visits += static_cast<double>(root.visit_count[static_cast<std::size_t>(a)]);
    }
    if (total_visits > 0.0) {
        for (int a = 0; a < kActionDim; ++a) {
            result.visit_probs[static_cast<std::size_t>(a)] = static_cast<float>(
                static_cast<double>(root.visit_count[static_cast<std::size_t>(a)]) / total_visits
            );
        }
    } else {
        int legal_count = 0;
        for (int a = 0; a < kActionDim; ++a) {
            if (root_data.mask[static_cast<std::size_t>(a)] != 0) {
                ++legal_count;
            }
        }
        const float u = 1.0f / static_cast<float>(legal_count);
        for (int a = 0; a < kActionDim; ++a) {
            result.visit_probs[static_cast<std::size_t>(a)] =
                (root_data.mask[static_cast<std::size_t>(a)] != 0) ? u : 0.0f;
        }
    }
    double prob_sum = 0.0;
    for (int a = 0; a < kActionDim; ++a) {
        if (root_data.mask[static_cast<std::size_t>(a)] == 0) {
            result.visit_probs[static_cast<std::size_t>(a)] = 0.0f;
        }
        prob_sum += static_cast<double>(result.visit_probs[static_cast<std::size_t>(a)]);
    }
    if (prob_sum > 0.0 && std::isfinite(prob_sum)) {
        for (int a = 0; a < kActionDim; ++a) {
            result.visit_probs[static_cast<std::size_t>(a)] = static_cast<float>(
                static_cast<double>(result.visit_probs[static_cast<std::size_t>(a)]) / prob_sum
            );
        }
    }

    result.action = sample_action_from_visits(
        result.visit_probs, root_data.mask, turns_taken, temperature_moves, temperature, rng
    );

    double q_sum = 0.0;
    int q_count = 0;
    for (int a = 0; a < kActionDim; ++a) {
        if (root_data.mask[static_cast<std::size_t>(a)] == 0) {
            continue;
        }
        const int n = root.visit_count[static_cast<std::size_t>(a)];
        const float q = (n <= 0) ? 0.0f : (root.value_sum[static_cast<std::size_t>(a)] / static_cast<float>(n));
        q_sum += static_cast<double>(q);
        ++q_count;
    }
    result.root_value = q_count > 0 ? static_cast<float>(q_sum / static_cast<double>(q_count)) : 0.0f;
    for (int a = 0; a < kActionDim; ++a) {
        result.root_total_visits += root.visit_count[static_cast<std::size_t>(a)];
        if (root.visit_count[static_cast<std::size_t>(a)] > 0) {
            result.root_nonzero_visit_actions += 1;
        }
        if (root_data.mask[static_cast<std::size_t>(a)] != 0) {
            result.root_legal_actions += 1;
        }
    }
    return result;
}
