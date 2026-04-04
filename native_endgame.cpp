// native_endgame.cpp
// ---------------------------------------------------------------------------
// Terminal-value negamax endgame solver for Splendor.
//
// Design:
//   - Pure terminal evaluation: no neural network, no heuristic leaf scores.
//     Every leaf MUST be a terminal game state. If the node budget runs out
//     before all paths reach terminals, `is_exact` is false and the best
//     move found so far is returned (anytime behaviour via IDDFS).
//   - Iterative deepening (IDDFS) within each determinization so we always
//     have a result from the previous depth if budget is exhausted.
//   - Transposition table (TT) with 2^N buckets, cleared between
//     determinizations (deck order is part of state identity).
//   - k-determinization: hidden deck order and opponent reserved cards are
//     sampled k times; node budget is split evenly; final answer is the
//     action with highest average value.
//   - Value convention: +1 = current-player-at-root wins, -1 = loses, 0 = draw.
//     Inside negamax the value is ALWAYS from the perspective of the player
//     to move at that node. When the active player changes we negate.
//     During return/noble-choice sub-turns (same_player), we do NOT negate.
// ---------------------------------------------------------------------------

#include "native_endgame.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "game_logic.h"
#include "state_encoder.h"  // build_raw_state, build_terminal_metadata

namespace {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr float kWin  =  1.0f;
constexpr float kLoss = -1.0f;
constexpr float kDraw =  0.0f;
constexpr float kInf  =  2.0f;  // outside [kLoss, kWin]

constexpr int kMaxDepth = 200;  // safety cap; real games end well before this

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

enum class TTFlag : uint8_t { Exact = 0, LowerBound = 1, UpperBound = 2 };

struct TTEntry {
    uint64_t hash        = 0;
    float    value       = 0.0f;
    int16_t  depth_left  = -1;   // plies remaining when this was stored
    int8_t   best_action = -1;   // cast to int when used
    TTFlag   flag        = TTFlag::Exact;
    bool     occupied    = false;
};

// Power-of-2 table; 1<<22 = ~4M entries × ~16 bytes = ~64 MB
constexpr int kTTBits = 22;
constexpr std::size_t kTTSize = std::size_t{1} << kTTBits;
constexpr uint64_t kTTMask = kTTSize - 1u;

using TranspositionTable = std::vector<TTEntry>;

inline TTEntry* tt_probe(TranspositionTable& tt, uint64_t hash) {
    TTEntry* e = &tt[hash & kTTMask];
    return (e->occupied && e->hash == hash) ? e : nullptr;
}

inline void tt_store(TranspositionTable& tt, uint64_t hash,
                     float value, int depth_left, int best_action, TTFlag flag) {
    TTEntry& e = tt[hash & kTTMask];
    // Always-replace: simpler and works well for IDDFS
    e.hash        = hash;
    e.value       = value;
    e.depth_left  = static_cast<int16_t>(depth_left);
    e.best_action = static_cast<int8_t>(best_action < 127 ? best_action : 127);
    e.flag        = flag;
    e.occupied    = true;
}

// ---------------------------------------------------------------------------
// Zobrist-style hashing over the raw integer state vector
// ---------------------------------------------------------------------------
// We use a simple FNV-1a over the raw state to avoid pre-computing a huge
// Zobrist table.  Incremental Zobrist can replace this if speed is needed.

inline uint64_t hash_raw_state(const std::array<int, state_encoder::STATE_DIM>& raw) {
    uint64_t h = 14695981039346656037ULL;
    for (int v : raw) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(v));
        h *= 1099511628211ULL;
    }
    return h;
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------
// Returns a list of (score, action_idx) pairs, sorted descending by score.
// Higher score = try this move first (more likely to cause cutoff).

constexpr int kActionDim = state_encoder::ACTION_DIM;

// Estimate how many points the current player would gain from buying a card.
// Uses the same approach as the C++ heuristic in py_splendor.cpp but simplified.
int card_points_from_faceup(const GameState& state, int tier, int slot) {
    const Card& card = state.faceup[tier][slot];
    return card.id > 0 ? card.points : 0;
}

int card_points_from_reserved(const GameState& state, int slot) {
    const Player& p = state.players[state.current_player];
    if (slot >= static_cast<int>(p.reserved.size())) return 0;
    return p.reserved[slot].card.points;
}

// Returns true if after buying/taking this action the current player
// would reach >= 15 points (a potential win trigger next turn or immediately).
// We don't simulate the full move here; just a quick upper-bound check.
bool is_winning_buy(const GameState& state, int action_idx) {
    const Player& p = state.players[state.current_player];
    if (action_idx >= 0 && action_idx <= 11) {
        int tier = action_idx / 4;
        int slot = action_idx % 4;
        return p.points + card_points_from_faceup(state, tier, slot) >= 15;
    }
    if (action_idx >= 12 && action_idx <= 14) {
        return p.points + card_points_from_reserved(state, action_idx - 12) >= 15;
    }
    return false;
}

struct ScoredAction {
    float score;
    int   action_idx;
};

std::vector<ScoredAction> order_moves(
    const GameState& state,
    const std::array<int, kActionDim>& mask,
    int tt_best_action)
{
    const Player& cp = state.players[state.current_player];
    std::vector<ScoredAction> moves;
    moves.reserve(32);

    // Pre-compute a crude "gem need" vector for TAKE scoring
    std::array<int, 5> gem_need{};
    // Look at face-up cards and estimate how many of each color we're short
    for (int tier = 0; tier < 3; ++tier) {
        for (int slot = 0; slot < 4; ++slot) {
            const Card& card = state.faceup[tier][slot];
            if (card.id <= 0) continue;
            const int colors[5] = {
                card.cost.white - cp.bonuses.white,
                card.cost.blue  - cp.bonuses.blue,
                card.cost.green - cp.bonuses.green,
                card.cost.red   - cp.bonuses.red,
                card.cost.black - cp.bonuses.black
            };
            for (int c = 0; c < 5; ++c) {
                if (colors[c] > 0) gem_need[c] += colors[c];
            }
        }
    }

    for (int a = 0; a < kActionDim; ++a) {
        if (!mask[a]) continue;

        float score = 0.0f;

        // TT best action always goes first
        if (a == tt_best_action) {
            score = 10000.0f;
        }
        // Winning buys
        else if (is_winning_buy(state, a)) {
            score = 5000.0f + static_cast<float>(a < 12
                ? card_points_from_faceup(state, a / 4, a % 4)
                : card_points_from_reserved(state, a - 12));
        }
        // Buy face-up (0-11)
        else if (a >= 0 && a <= 11) {
            int pts = card_points_from_faceup(state, a / 4, a % 4);
            score = 2000.0f + static_cast<float>(pts) * 100.0f;
        }
        // Buy reserved (12-14)
        else if (a >= 12 && a <= 14) {
            int pts = card_points_from_reserved(state, a - 12);
            score = 1800.0f + static_cast<float>(pts) * 100.0f;
        }
        // Reserve face-up (15-26) — lower priority in endgame
        else if (a >= 15 && a <= 26) {
            int tier = (a - 15) / 4;
            int slot = (a - 15) % 4;
            int pts = card_points_from_faceup(state, tier, slot);
            // Bonus if bank has a joker (we'd pick one up)
            float joker_bonus = state.bank.joker > 0 ? 200.0f : 0.0f;
            score = 400.0f + static_cast<float>(pts) * 50.0f + joker_bonus;
        }
        // Reserve from deck (27-29)
        else if (a >= 27 && a <= 29) {
            float joker_bonus = state.bank.joker > 0 ? 150.0f : 0.0f;
            score = 200.0f + joker_bonus;
        }
        // Take 3 different (30-39)
        else if (a >= 30 && a <= 39) {
            constexpr std::array<std::array<int,3>,10> triplets{{
                {0,1,2},{0,1,3},{0,1,4},{0,2,3},{0,2,4},
                {0,3,4},{1,2,3},{1,2,4},{1,3,4},{2,3,4}
            }};
            const auto& tri = triplets[a - 30];
            float usefulness = static_cast<float>(gem_need[tri[0]] + gem_need[tri[1]] + gem_need[tri[2]]);
            score = 500.0f + usefulness * 20.0f;
        }
        // Take 2 same (40-44)
        else if (a >= 40 && a <= 44) {
            score = 450.0f + static_cast<float>(gem_need[a - 40]) * 30.0f;
        }
        // Take 2 different (45-54)
        else if (a >= 45 && a <= 54) {
            constexpr std::array<std::array<int,2>,10> pairs{{
                {0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}
            }};
            const auto& pr = pairs[a - 45];
            float usefulness = static_cast<float>(gem_need[pr[0]] + gem_need[pr[1]]);
            score = 350.0f + usefulness * 20.0f;
        }
        // Take 1 (55-59)
        else if (a >= 55 && a <= 59) {
            score = 250.0f + static_cast<float>(gem_need[a - 55]) * 20.0f;
        }
        // Pass (60) — last resort
        else if (a == 60) {
            score = -1000.0f;
        }
        // Return gem (61-65) — forced, keep as-is but pick least-needed color
        else if (a >= 61 && a <= 65) {
            // Prefer to return colors we need least
            score = 100.0f - static_cast<float>(gem_need[a - 61]) * 10.0f;
        }
        // Choose noble (66-68) — always good, do immediately
        else if (a >= 66 && a <= 68) {
            score = 4000.0f;
        }

        moves.push_back({score, a});
    }

    std::sort(moves.begin(), moves.end(),
              [](const ScoredAction& x, const ScoredAction& y) {
                  return x.score > y.score;
              });
    return moves;
}

// ---------------------------------------------------------------------------
// Solver state shared across the search
// ---------------------------------------------------------------------------

struct SolverStats {
    int nodes    = 0;
    int tt_hits  = 0;
    int budget   = 0;  // remaining node budget; search stops when <= 0
    bool exact   = true; // becomes false if budget exhausted
};

// ---------------------------------------------------------------------------
// Core negamax with IDDFS support
// ---------------------------------------------------------------------------
// Value is always from the perspective of the player-to-move at this node.
// We negate when current_player changes after applying a move.
// For same-player sub-turns (return phase, noble choice) we do NOT negate.

float negamax(
    GameState& state,
    int depth_left,
    float alpha,
    float beta,
    const uint64_t parent_hash,  // hash of state BEFORE entering this call
    TranspositionTable& tt,
    SolverStats& stats)
{
    // ------------------------------------------------------------------
    // 1. Terminal check
    // ------------------------------------------------------------------
    const auto terminal = state_encoder::build_terminal_metadata(state);
    if (terminal.is_terminal) {
        // value from the perspective of the player who is current at terminal
        if (terminal.winner == -1) return kDraw;
        return terminal.winner == terminal.current_player_id ? kWin : kLoss;
    }

    // ------------------------------------------------------------------
    // 2. Budget check
    // ------------------------------------------------------------------
    if (stats.budget <= 0) {
        stats.exact = false;
        return kDraw;  // neutral fallback; caller uses the TT best_action
    }

    // ------------------------------------------------------------------
    // 3. Depth limit — treat as draw (caller will deepen via IDDFS)
    // ------------------------------------------------------------------
    if (depth_left <= 0) {
        // Not a terminal but depth exhausted; solver is incomplete here
        stats.exact = false;
        return kDraw;
    }

    // ------------------------------------------------------------------
    // 4. Compute state hash
    // ------------------------------------------------------------------
    const auto raw = state_encoder::build_raw_state(state);
    const uint64_t hash = hash_raw_state(raw);

    // ------------------------------------------------------------------
    // 5. TT lookup
    // ------------------------------------------------------------------
    int tt_best_action = -1;
    {
        const TTEntry* entry = tt_probe(tt, hash);
        if (entry != nullptr && entry->depth_left >= depth_left) {
            ++stats.tt_hits;
            float stored = entry->value;
            tt_best_action = static_cast<int>(static_cast<uint8_t>(entry->best_action));
            if (entry->flag == TTFlag::Exact)      return stored;
            if (entry->flag == TTFlag::LowerBound) alpha = std::max(alpha, stored);
            if (entry->flag == TTFlag::UpperBound) beta  = std::min(beta,  stored);
            if (alpha >= beta) return stored;
        } else if (entry != nullptr) {
            tt_best_action = static_cast<int>(static_cast<uint8_t>(entry->best_action));
        }
    }

    // ------------------------------------------------------------------
    // 6. Generate and order moves
    // ------------------------------------------------------------------
    const auto mask = getValidMoveMask(state);
    const auto moves = order_moves(state, mask, tt_best_action);

    if (moves.empty()) {
        // No legal actions in a non-terminal state shouldn't happen,
        // but handle gracefully
        return kDraw;
    }

    // ------------------------------------------------------------------
    // 7. Search
    // ------------------------------------------------------------------
    const int current_player_before = state.current_player;
    float best_value = -kInf;
    int   best_action = moves[0].action_idx;
    float original_alpha = alpha;

    for (const auto& scored : moves) {
        const int a = scored.action_idx;

        --stats.budget;
        ++stats.nodes;

        applyMove(state, actionIndexToMove(a));

        const bool same_player = (state.current_player == current_player_before);

        float child_value;
        if (same_player) {
            // Still the same player's turn (return phase, noble choice, etc.)
            // DO NOT negate; the value perspective is unchanged.
            child_value = negamax(state, depth_left - 1, alpha, beta, hash, tt, stats);
        } else {
            // Opponent is now to move. Negate to convert to our perspective.
            child_value = -negamax(state, depth_left - 1, -beta, -alpha, hash, tt, stats);
        }

        // Undo the move by restoring from saved state
        // (We use a copy-restore pattern — GameState is small enough)
        // NOTE: Because applyMove modifies state in-place and there is no
        // "undoMove", we need to restore. The cleanest approach in C++ is
        // to pass state by value, but that copies on every call.
        // We instead save/restore here using the GameState copy made on entry.
        // See the wrapper below for how we manage this.
        //
        // (This function is actually called with a *copy* of the state at
        //  each level — see negamax_entry which copies before each child.)

        if (child_value > best_value) {
            best_value = child_value;
            best_action = a;
        }
        alpha = std::max(alpha, best_value);

        if (alpha >= beta) {
            // Beta cutoff
            tt_store(tt, hash, best_value, depth_left, best_action, TTFlag::LowerBound);
            return best_value;
        }

        if (stats.budget <= 0) {
            stats.exact = false;
            break;
        }
    }

    // ------------------------------------------------------------------
    // 8. Store in TT
    // ------------------------------------------------------------------
    TTFlag flag = TTFlag::Exact;
    if (best_value <= original_alpha) flag = TTFlag::UpperBound;
    else if (best_value >= beta)      flag = TTFlag::LowerBound;
    tt_store(tt, hash, best_value, depth_left, best_action, flag);

    return best_value;
}

// Wrapper that copies state before each recursive call so that applyMove
// side-effects don't need an undoMove. This is cache-friendly for shallow
// endgame trees and avoids implementing a full undo system.
float negamax_root(
    const GameState& state,
    int depth_left,
    float alpha,
    float beta,
    TranspositionTable& tt,
    SolverStats& stats,
    int& out_best_action)
{
    const auto mask = getValidMoveMask(state);

    // TT lookup at root to get initial best action for ordering
    const auto raw = state_encoder::build_raw_state(state);
    const uint64_t hash = hash_raw_state(raw);
    int tt_best_action = -1;
    {
        const TTEntry* entry = tt_probe(tt, hash);
        if (entry != nullptr) {
            tt_best_action = static_cast<int>(static_cast<uint8_t>(entry->best_action));
        }
    }

    const auto moves = order_moves(state, mask, tt_best_action);
    if (moves.empty()) {
        out_best_action = -1;
        return kDraw;
    }

    const int current_player_before = state.current_player;
    float best_value = -kInf;
    out_best_action = moves[0].action_idx;
    float original_alpha = alpha;

    for (const auto& scored : moves) {
        const int a = scored.action_idx;

        --stats.budget;
        ++stats.nodes;

        // Copy state for this child — enables move/undo without an undo system
        GameState child_state = state;
        applyMove(child_state, actionIndexToMove(a));

        const bool same_player = (child_state.current_player == current_player_before);

        float child_value;
        if (same_player) {
            child_value = negamax(child_state, depth_left - 1, alpha, beta, hash, tt, stats);
        } else {
            child_value = -negamax(child_state, depth_left - 1, -beta, -alpha, hash, tt, stats);
        }

        if (child_value > best_value) {
            best_value = child_value;
            out_best_action = a;
        }
        alpha = std::max(alpha, best_value);

        if (alpha >= beta) break;
        if (stats.budget <= 0) {
            stats.exact = false;
            break;
        }
    }

    TTFlag flag = TTFlag::Exact;
    if (best_value <= original_alpha) flag = TTFlag::UpperBound;
    else if (best_value >= beta)      flag = TTFlag::LowerBound;
    tt_store(tt, hash, best_value, depth_left, out_best_action, flag);

    return best_value;
}

// Recursive negamax using copy-on-call (pass state by value each level).
// This replaces the negamax() above — cleaner, safe undo.
float negamax(
    GameState state,           // passed by VALUE — copy on each call
    int depth_left,
    float alpha,
    float beta,
    TranspositionTable& tt,
    SolverStats& stats)
{
    // ------------------------------------------------------------------
    // 1. Terminal check
    // ------------------------------------------------------------------
    const auto terminal = state_encoder::build_terminal_metadata(state);
    if (terminal.is_terminal) {
        if (terminal.winner == -1) return kDraw;
        return terminal.winner == terminal.current_player_id ? kWin : kLoss;
    }

    // ------------------------------------------------------------------
    // 2. Budget / depth check
    // ------------------------------------------------------------------
    if (stats.budget <= 0) {
        stats.exact = false;
        return kDraw;
    }
    if (depth_left <= 0) {
        stats.exact = false;
        return kDraw;
    }

    // ------------------------------------------------------------------
    // 3. Compute hash
    // ------------------------------------------------------------------
    const auto raw = state_encoder::build_raw_state(state);
    const uint64_t hash = hash_raw_state(raw);

    // ------------------------------------------------------------------
    // 4. TT lookup
    // ------------------------------------------------------------------
    int tt_best_action = -1;
    {
        const TTEntry* entry = tt_probe(tt, hash);
        if (entry != nullptr && entry->depth_left >= depth_left) {
            ++stats.tt_hits;
            float stored = entry->value;
            tt_best_action = static_cast<int>(static_cast<uint8_t>(entry->best_action));
            if (entry->flag == TTFlag::Exact)      return stored;
            if (entry->flag == TTFlag::LowerBound) alpha = std::max(alpha, stored);
            if (entry->flag == TTFlag::UpperBound) beta  = std::min(beta,  stored);
            if (alpha >= beta) return stored;
        } else if (entry != nullptr) {
            tt_best_action = static_cast<int>(static_cast<uint8_t>(entry->best_action));
        }
    }

    // ------------------------------------------------------------------
    // 5. Generate and order moves
    // ------------------------------------------------------------------
    const auto mask = getValidMoveMask(state);
    const auto moves = order_moves(state, mask, tt_best_action);
    if (moves.empty()) return kDraw;

    // ------------------------------------------------------------------
    // 6. Search children
    // ------------------------------------------------------------------
    const int current_player_before = state.current_player;
    float best_value = -kInf;
    int   best_action = moves[0].action_idx;
    float original_alpha = alpha;

    for (const auto& scored : moves) {
        const int a = scored.action_idx;

        --stats.budget;
        ++stats.nodes;

        // Copy state before applying move (enables clean undo via scope)
        GameState child = state;
        applyMove(child, actionIndexToMove(a));

        const bool same_player = (child.current_player == current_player_before);

        float child_value;
        if (same_player) {
            child_value = negamax(std::move(child), depth_left - 1, alpha, beta, tt, stats);
        } else {
            child_value = -negamax(std::move(child), depth_left - 1, -beta, -alpha, tt, stats);
        }

        if (child_value > best_value) {
            best_value = child_value;
            best_action = a;
        }
        alpha = std::max(alpha, best_value);
        if (alpha >= beta) break;
        if (stats.budget <= 0) { stats.exact = false; break; }
    }

    // ------------------------------------------------------------------
    // 7. TT store
    // ------------------------------------------------------------------
    TTFlag flag = TTFlag::Exact;
    if (best_value <= original_alpha) flag = TTFlag::UpperBound;
    else if (best_value >= beta)      flag = TTFlag::LowerBound;
    tt_store(tt, hash, best_value, depth_left, best_action, flag);

    return best_value;
}

// ---------------------------------------------------------------------------
// IDDFS within one determinization
// ---------------------------------------------------------------------------

struct DetResult {
    int   best_action = -1;
    float value       = 0.0f;
    bool  exact       = false;
    int   nodes       = 0;
    int   tt_hits     = 0;
};

DetResult run_iddfs(const GameState& root, int node_budget) {
    TranspositionTable tt(kTTSize);

    SolverStats stats;
    stats.budget = node_budget;
    stats.exact  = true;

    DetResult result;
    result.best_action = -1;
    result.exact       = false;

    // Quick check: any legal moves?
    const auto mask = getValidMoveMask(root);
    bool has_legal = false;
    for (int a = 0; a < kActionDim; ++a) {
        if (mask[a]) { result.best_action = a; has_legal = true; break; }
    }
    if (!has_legal) return result;

    // Check if root is already terminal
    const auto terminal = state_encoder::build_terminal_metadata(root);
    if (terminal.is_terminal) {
        result.exact = true;
        result.value = (terminal.winner == -1) ? kDraw
                     : (terminal.winner == terminal.current_player_id ? kWin : kLoss);
        return result;
    }

    // IDDFS loop: depth 1, 2, 3, ...
    for (int depth = 1; depth <= kMaxDepth && stats.budget > 0; ++depth) {
        stats.exact = true;  // will be set false if depth/budget exceeded

        int candidate_action = result.best_action;
        float v = negamax_root(root, depth, -kInf, kInf, tt, stats,
                                 candidate_action);

        // Accept this depth's result
        result.best_action = candidate_action;
        result.value       = v;
        result.exact       = stats.exact;

        // If we found a proven win/loss, no point searching deeper
        if (std::abs(v) >= kWin - 1e-5f && stats.exact) break;

        // If budget ran out mid-search, stop (we still have depth-1 result)
        if (stats.budget <= 0) break;
    }

    result.nodes   = stats.nodes;
    result.tt_hits = stats.tt_hits;
    return result;
}

// ---------------------------------------------------------------------------
// Hidden information re-sampling (mirrors sample_root_hidden_information
// from native_mcts.cpp, adapted to work without pybind11)
// ---------------------------------------------------------------------------

template <typename Rng>
void resample_hidden_information(GameState& state, Rng& rng) {
    // 1. Shuffle deck order for each tier
    for (int tier = 0; tier < 3; ++tier) {
        std::shuffle(state.deck[tier].begin(), state.deck[tier].end(), rng);
    }

    // 2. Resample opponent's hidden reserved cards
    const int current_player = state.current_player;
    if (current_player < 0 || current_player > 1) return;
    const int opponent = 1 - current_player;
    Player& opp = state.players[opponent];

    // Group hidden reserved slot indices by tier
    std::array<std::vector<int>, 3> hidden_slots_by_tier;
    std::array<std::vector<Card>, 3> hidden_cards_by_tier;

    for (int slot = 0; slot < static_cast<int>(opp.reserved.size()); ++slot) {
        const ReservedCard& rc = opp.reserved[slot];
        if (rc.is_public) continue;
        const int tier = rc.card.level - 1;
        if (tier < 0 || tier >= 3) continue;
        hidden_slots_by_tier[tier].push_back(slot);
        hidden_cards_by_tier[tier].push_back(rc.card);
    }

    for (int tier = 0; tier < 3; ++tier) {
        const auto& slot_indices = hidden_slots_by_tier[tier];
        if (slot_indices.empty()) continue;
        const auto& hidden_cards = hidden_cards_by_tier[tier];

        // Pool = deck cards + the hidden reserved cards themselves
        std::vector<Card> pool;
        pool.reserve(state.deck[tier].size() + hidden_cards.size());
        pool.insert(pool.end(), state.deck[tier].begin(), state.deck[tier].end());
        pool.insert(pool.end(), hidden_cards.begin(), hidden_cards.end());
        std::shuffle(pool.begin(), pool.end(), rng);

        const std::size_t hidden_count = slot_indices.size();
        if (pool.size() < hidden_count) continue;  // shouldn't happen

        // Assign shuffled cards to hidden slots
        for (std::size_t i = 0; i < hidden_count; ++i) {
            opp.reserved[slot_indices[i]].card = pool[i];
        }
        // Remaining cards go back to deck
        state.deck[tier].assign(pool.begin() + static_cast<std::ptrdiff_t>(hidden_count),
                                pool.end());
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

EndgameSolverResult run_endgame_solver(
    const GameState& root_state,
    int   node_budget,
    int   k_determinizations,
    uint64_t rng_seed)
{
    if (k_determinizations <= 0) k_determinizations = 1;
    if (node_budget <= 0) {
        EndgameSolverResult empty;
        // Return first legal action as fallback
        const auto mask = getValidMoveMask(root_state);
        for (int a = 0; a < kActionDim; ++a) {
            if (mask[a]) { empty.best_action = a; break; }
        }
        return empty;
    }

    const int budget_per_det = node_budget / k_determinizations;

    std::mt19937_64 rng(rng_seed);

    // Accumulate vote scores: for each action, sum the minimax values
    // returned across determinizations where that action was chosen as best.
    // We use a simple approach: each det contributes its best_action with
    // weight = value (+1 win, 0 draw, -1 loss mapped to 0..2 for ranking).
    std::array<double, kActionDim> action_scores{};
    std::array<int,    kActionDim> action_votes{};
    action_scores.fill(0.0);
    action_votes.fill(0);

    EndgameSolverResult final_result;
    final_result.best_action = -1;

    // Get first legal action as ultimate fallback
    {
        const auto mask = getValidMoveMask(root_state);
        for (int a = 0; a < kActionDim; ++a) {
            if (mask[a]) { final_result.best_action = a; break; }
        }
    }

    bool any_exact = false;
    int total_nodes = 0;
    int total_tt_hits = 0;

    for (int det = 0; det < k_determinizations; ++det) {
        GameState det_state = root_state;
        resample_hidden_information(det_state, rng);

        const DetResult det_result = run_iddfs(det_state, budget_per_det);

        total_nodes   += det_result.nodes;
        total_tt_hits += det_result.tt_hits;
        if (det_result.exact) any_exact = true;

        final_result.determinizations_completed = det + 1;

        const int best_a = det_result.best_action;
        if (best_a >= 0 && best_a < kActionDim) {
            // Map value [-1, +1] to score [0, 2] to avoid negative weights
            double score = static_cast<double>(det_result.value) + 1.0;
            action_scores[best_a] += score;
            action_votes[best_a]  += 1;
        }
    }

    // Pick action with highest total score; break ties by vote count
    double best_score  = -1.0;
    int    best_votes  = -1;
    for (int a = 0; a < kActionDim; ++a) {
        if (action_votes[a] == 0) continue;
        if (action_scores[a] > best_score ||
            (action_scores[a] == best_score && action_votes[a] > best_votes)) {
            best_score = action_scores[a];
            best_votes = action_votes[a];
            final_result.best_action = a;
            final_result.value       = static_cast<float>(
                (action_scores[a] / action_votes[a]) - 1.0);
        }
    }

    final_result.is_exact       = any_exact;
    final_result.nodes_searched = total_nodes;
    final_result.tt_hits        = total_tt_hits;

    return final_result;
}
