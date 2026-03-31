#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "game_logic.h"

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

struct EndgameSolverResult {
    int   best_action    = -1;   // action index [0,68], -1 if no legal actions
    float value          = 0.0f; // +1 = current player wins, -1 = loses, 0 = draw
    bool  is_exact       = false; // true iff all lines searched to terminal
    int   nodes_searched = 0;
    int   tt_hits        = 0;
    int   determinizations_completed = 0;
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

// Run k-determinization alpha-beta endgame solver on `root_state`.
// node_budget is split evenly across k_determinizations.
// Returns the action with highest average minimax value across determinizations.
EndgameSolverResult run_endgame_solver(
    const GameState& root_state,
    int   node_budget        = 2'000'000,
    int   k_determinizations = 8,
    uint64_t rng_seed        = 0
);
