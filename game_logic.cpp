// game_logic.cpp
#include "game_logic.h"
#include "simple_json_parse.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <random>
#include <ctime>

// ─── Helpers ──────────────────────────────────────────

static Color parseColor(const std::string& s) {
    if (s == "white") return Color::White;
    if (s == "blue")  return Color::Blue;
    if (s == "green") return Color::Green;
    if (s == "red")   return Color::Red;
    if (s == "black") return Color::Black;
    throw std::runtime_error("Invalid card color: " + s);
}

// Parse a Tokens object from a JSON sub-object
// e.g. {"blue": 2, "green": 1} → Tokens{blue=2, green=1}
static Tokens parseTokens(const std::string& json) {
    Tokens t;
    t.white = simple_json::extractInt(json, "white");
    t.blue  = simple_json::extractInt(json, "blue");
    t.green = simple_json::extractInt(json, "green");
    t.red   = simple_json::extractInt(json, "red");
    t.black = simple_json::extractInt(json, "black");
    t.joker = simple_json::extractInt(json, "joker");
    return t;
}

// ─── loadCards ────────────────────────────────────────
std::vector<Card> loadCards(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open cards file: " + path);

    // Read entire file
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();

    std::vector<Card> cards;

    // Each card is a JSON object {...} in the array
    size_t pos = content.find('{');
    while (pos != std::string::npos) {
        // Find matching closing brace
        int depth = 1;
        size_t cur = pos + 1;
        while (cur < content.size() && depth > 0) {
            if (content[cur] == '{') depth++;
            else if (content[cur] == '}') depth--;
            cur++;
        }

        std::string card_json = content.substr(pos, cur - pos);

        Card card;
        card.id     = simple_json::extractInt(card_json, "id");
        card.level  = simple_json::extractInt(card_json, "level");
        card.points = simple_json::extractInt(card_json, "points");
        card.color  = parseColor(simple_json::extractStr(card_json, "color"));
        card.cost   = parseTokens(simple_json::extractObject(card_json, "cost"));

        // Only add valid cards (id > 0)
        if (card.id > 0)
            cards.push_back(card);

        pos = content.find('{', cur);
    }

    return cards;
}

// ─── loadNobles ───────────────────────────────────────
std::vector<Noble> loadNobles(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open nobles file: " + path);

    std::ostringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();

    std::vector<Noble> nobles;

    size_t pos = content.find('{');
    while (pos != std::string::npos) {
        int depth = 1;
        size_t cur = pos + 1;
        while (cur < content.size() && depth > 0) {
            if (content[cur] == '{') depth++;
            else if (content[cur] == '}') depth--;
            cur++;
        }

        std::string noble_json = content.substr(pos, cur - pos);

        Noble noble;
        noble.id           = simple_json::extractInt(noble_json, "id");
        noble.points       = simple_json::extractInt(noble_json, "points");
        noble.requirements = parseTokens(simple_json::extractObject(noble_json, "requirements"));

        if (noble.id > 0)
            nobles.push_back(noble);

        pos = content.find('{', cur);
    }

    return nobles;
}

// ─── initializeGame ───────────────────────────────────
void initializeGame(GameState&                state,
                    const std::vector<Card>&  all_cards,
                    const std::vector<Noble>& all_nobles,
                    unsigned int              seed) {
    // Reset state
    state = GameState();

    // Seed RNG
    if (seed == 0) seed = static_cast<unsigned int>(std::time(nullptr));
    std::mt19937 rng(seed);

    // ── Separate cards by tier ──
    std::vector<Card> tier[3];
    for (const Card& c : all_cards) {
        if (c.level >= 1 && c.level <= 3)
            tier[c.level - 1].push_back(c);
    }

    // ── Shuffle each tier ──
    for (int t = 0; t < 3; t++)
        std::shuffle(tier[t].begin(), tier[t].end(), rng);

    // ── Deal 4 face-up cards per tier, rest go to deck ──
    for (int t = 0; t < 3; t++) {
        for (int slot = 0; slot < 4; slot++) {
            if (slot < (int)tier[t].size())
                state.faceup[t][slot] = tier[t][slot];
            // else stays as default Card{id=0} — empty slot
        }
        // Remaining cards go to deck (index 4 onwards)
        for (int i = 4; i < (int)tier[t].size(); i++)
            state.deck[t].push_back(tier[t][i]);
    }

    // ── Shuffle and pick 3 nobles ──
    std::vector<Noble> shuffled_nobles = all_nobles;
    std::shuffle(shuffled_nobles.begin(), shuffled_nobles.end(), rng);

    state.noble_count = 0;
    for (int i = 0; i < 3 && i < (int)shuffled_nobles.size(); i++) {
        state.available_nobles[i] = shuffled_nobles[i];
        state.noble_count++;
    }

    // ── Initialize bank ──
    // 2-player game: 4 of each color, 5 jokers
    state.bank.white = 4;
    state.bank.blue  = 4;
    state.bank.green = 4;
    state.bank.red   = 4;
    state.bank.black = 4;
    state.bank.joker = 5;

    // ── Players start with nothing ──
    state.current_player  = 0;
    state.move_number     = 0;
    state.is_return_phase = false;
}

// ─── Helper: replace faceup slot from deck ────────────
static void refillSlot(GameState& state, int tier, int slot) {
    if (!state.deck[tier].empty()) {
        state.faceup[tier][slot] = state.deck[tier].back();
        state.deck[tier].pop_back();
    } else {
        state.faceup[tier][slot] = Card{}; // empty slot id=0
    }
}

// ─── Helper: check and assign nobles ──────────────────
static void checkNobles(GameState& state, int player_idx) {
    Player& player = state.players[player_idx];
    for (int i = 0; i < state.noble_count; i++) {
        Noble& noble = state.available_nobles[i];
        if (player.bonuses.white >= noble.requirements.white &&
            player.bonuses.blue  >= noble.requirements.blue  &&
            player.bonuses.green >= noble.requirements.green &&
            player.bonuses.red   >= noble.requirements.red   &&
            player.bonuses.black >= noble.requirements.black) {
            // assign noble
            player.nobles.push_back(noble);
            player.points += noble.points;
            // remove from available by shifting left
            for (int j = i; j < state.noble_count - 1; j++)
                state.available_nobles[j] = state.available_nobles[j+1];
            state.available_nobles[state.noble_count-1] = Noble{};
            state.noble_count--;
            break; // only one noble per turn
        }
    }
}

// ─── Helper: effective cost of card after bonuses ─────
static Tokens effectiveCost(const Card& card, const Player& player) {
    Tokens cost;
    cost.white = std::max(0, card.cost.white - player.bonuses.white);
    cost.blue  = std::max(0, card.cost.blue  - player.bonuses.blue);
    cost.green = std::max(0, card.cost.green - player.bonuses.green);
    cost.red   = std::max(0, card.cost.red   - player.bonuses.red);
    cost.black = std::max(0, card.cost.black - player.bonuses.black);
    return cost;
}

// ─── Helper: can player afford card ───────────────────
static bool canAfford(const Card& card, const Player& player) {
    Tokens cost = effectiveCost(card, player);
    int needed = cost.white + cost.blue + cost.green + cost.red + cost.black;
    int jokers_available = player.tokens.joker;
    // check each color
    int shortfall = 0;
    shortfall += std::max(0, cost.white - player.tokens.white);
    shortfall += std::max(0, cost.blue  - player.tokens.blue);
    shortfall += std::max(0, cost.green - player.tokens.green);
    shortfall += std::max(0, cost.red   - player.tokens.red);
    shortfall += std::max(0, cost.black - player.tokens.black);
    return shortfall <= jokers_available;
}

// ─── applyMove ────────────────────────────────────────
void applyMove(GameState& state, const Move& move) {
    int p = state.current_player;
    Player& player = state.players[p];

    switch (move.type) {

        case BUY_CARD: {
            Card* card = nullptr;
            int tier = move.card_tier;
            int slot = move.card_slot;

            if (move.from_reserved) {
                card = &player.reserved[slot];
            } else {
                card = &state.faceup[tier][slot];
            }

            // Calculate payment
            Tokens cost = effectiveCost(*card, player);
            Tokens payment;
            payment.white = std::min(cost.white, player.tokens.white);
            payment.blue  = std::min(cost.blue,  player.tokens.blue);
            payment.green = std::min(cost.green,  player.tokens.green);
            payment.red   = std::min(cost.red,   player.tokens.red);
            payment.black = std::min(cost.black,  player.tokens.black);
            int shortfall = (cost.white - payment.white) +
                            (cost.blue  - payment.blue)  +
                            (cost.green - payment.green) +
                            (cost.red   - payment.red)   +
                            (cost.black - payment.black);
            payment.joker = shortfall;

            // Apply payment
            player.tokens -= payment;
            state.bank    += payment;

            // Add card to player
            player.bonuses[card->color]++;
            player.points += card->points;
            player.cards.push_back(*card);

            // Remove card from source
            if (move.from_reserved) {
                player.reserved.erase(player.reserved.begin() + slot);
            } else {
                refillSlot(state, tier, slot);
            }

            // Check nobles
            checkNobles(state, p);

            // End return phase check and switch player
            state.is_return_phase = false;
            state.current_player  = 1 - p;
            state.move_number++;
            break;
        }

        case RESERVE_CARD: {
            Card card;
            if (move.from_deck) {
                int tier = move.card_tier;
                card = state.deck[tier].back();
                state.deck[tier].pop_back();
            } else {
                int tier = move.card_tier;
                int slot = move.card_slot;
                card = state.faceup[tier][slot];
                refillSlot(state, tier, slot);
            }

            player.reserved.push_back(card);

            // Give joker if available
            if (state.bank.joker > 0) {
                player.tokens.joker++;
                state.bank.joker--;
            }

            // Check if return phase needed
            if (player.tokens.total() > 10) {
                state.is_return_phase = true;
            } else {
                state.is_return_phase = false;
                state.current_player  = 1 - p;
                state.move_number++;
            }
            break;
        }

        case TAKE_GEMS: {
            player.tokens += move.gems_taken;
            state.bank    -= move.gems_taken;

            // Check if return phase needed
            if (player.tokens.total() > 10) {
                state.is_return_phase = true;
            } else {
                state.is_return_phase = false;
                state.current_player  = 1 - p;
                state.move_number++;
            }
            break;
        }

        case RETURN_GEM: {
            // Return exactly 1 token of specified color
            player.tokens -= move.gem_returned;
            state.bank    += move.gem_returned;

            // Check if still over 10
            if (player.tokens.total() > 10) {
                state.is_return_phase = true;
            } else {
                state.is_return_phase = false;
                state.current_player  = 1 - p;
                state.move_number++;
            }
            break;
        }

        case PASS_TURN: {
            state.is_return_phase = false;
            state.current_player  = 1 - p;
            state.move_number++;
            break;
        }
    }
}

// ─── findAllValidMoves ────────────────────────────────
std::vector<Move> findAllValidMoves(const GameState& state) {
    std::vector<Move> moves;
    int p = state.current_player;
    const Player& player = state.players[p];

    // ── Return phase: only return moves ──
    if (state.is_return_phase) {
        const Color colors[] = {Color::White, Color::Blue, Color::Green, Color::Red, Color::Black};
        for (int i = 0; i < 5; i++) {
            if (player.tokens[colors[i]] > 0) {
                Move m;
                m.type = RETURN_GEM;
                m.gem_returned[colors[i]] = 1;
                moves.push_back(m);
            }
        }
        return moves;
    }

    // ── BUY face-up ──
    for (int t = 0; t < 3; t++) {
        for (int s = 0; s < 4; s++) {
            const Card& card = state.faceup[t][s];
            if (card.id > 0 && canAfford(card, player)) {
                Move m;
                m.type      = BUY_CARD;
                m.card_tier = t;
                m.card_slot = s;
                moves.push_back(m);
            }
        }
    }

    // ── BUY reserved ──
    for (int s = 0; s < (int)player.reserved.size(); s++) {
        if (canAfford(player.reserved[s], player)) {
            Move m;
            m.type         = BUY_CARD;
            m.card_slot    = s;
            m.from_reserved = true;
            moves.push_back(m);
        }
    }

    // ── RESERVE face-up ──
    if ((int)player.reserved.size() < 3) {
        for (int t = 0; t < 3; t++) {
            for (int s = 0; s < 4; s++) {
                if (state.faceup[t][s].id > 0) {
                    Move m;
                    m.type      = RESERVE_CARD;
                    m.card_tier = t;
                    m.card_slot = s;
                    moves.push_back(m);
                }
            }
        }
        // ── RESERVE from deck ──
        for (int t = 0; t < 3; t++) {
            if (!state.deck[t].empty()) {
                Move m;
                m.type      = RESERVE_CARD;
                m.card_tier = t;
                m.from_deck = true;
                moves.push_back(m);
            }
        }
    }

    // ── TAKE GEMS ──
    int colors_available = 0;
    const Color colors[] = {Color::White, Color::Blue, Color::Green, Color::Red, Color::Black};
    bool avail[5];
    for (int i = 0; i < 5; i++) {
        avail[i] = state.bank[colors[i]] > 0;
        if (avail[i]) colors_available++;
    }

    // Take 3 different
    if (colors_available >= 3) {
        for (int i = 0; i < 5; i++) if (avail[i])
        for (int j = i+1; j < 5; j++) if (avail[j])
        for (int k = j+1; k < 5; k++) if (avail[k]) {
            Move m; m.type = TAKE_GEMS;
            m.gems_taken[colors[i]] = 1;
            m.gems_taken[colors[j]] = 1;
            m.gems_taken[colors[k]] = 1;
            moves.push_back(m);
        }
    }
    // Take 2 same
    for (int i = 0; i < 5; i++) {
        if (state.bank[colors[i]] >= 4) {
            Move m; m.type = TAKE_GEMS;
            m.gems_taken[colors[i]] = 2;
            moves.push_back(m);
        }
    }
    // Take 2 different (only when < 3 colors available)
    if (colors_available == 2) {
        for (int i = 0; i < 5; i++) if (avail[i])
        for (int j = i+1; j < 5; j++) if (avail[j]) {
            Move m; m.type = TAKE_GEMS;
            m.gems_taken[colors[i]] = 1;
            m.gems_taken[colors[j]] = 1;
            moves.push_back(m);
        }
    }
    // Take 1 (only when exactly 1 color available)
    if (colors_available == 1) {
        for (int i = 0; i < 5; i++) {
            if (avail[i]) {
                Move m; m.type = TAKE_GEMS;
                m.gems_taken[colors[i]] = 1;
                moves.push_back(m);
            }
        }
    }

    // ── PASS (only if no other moves) ──
    if (moves.empty()) {
        Move m; m.type = PASS_TURN;
        moves.push_back(m);
    }

    return moves;
}

// ─── isGameOver ───────────────────────────────────────
bool isGameOver(const GameState& state) {
    // Don't end mid return phase
    if (state.is_return_phase) return false;

    bool p0_has_15 = state.players[0].points >= 15;
    bool p1_has_15 = state.players[1].points >= 15;

    if (!p0_has_15 && !p1_has_15) return false;

    // Player 1 (second) reached 15 — game ends immediately
    if (p1_has_15) return true;

    // Player 0 (first) reached 15 — player 1 gets last turn
    // current_player==0 means player 1 just finished their turn
    if (p0_has_15 && state.current_player == 0) return true;

    return false;
}

// ─── determineWinner ──────────────────────────────────
int determineWinner(const GameState& state) {
    const Player& p0 = state.players[0];
    const Player& p1 = state.players[1];

    // Higher points wins
    if (p0.points > p1.points) return 0;
    if (p1.points > p0.points) return 1;

    // Tiebreaker: fewer purchased cards
    if ((int)p0.cards.size() < (int)p1.cards.size()) return 0;
    if ((int)p1.cards.size() < (int)p0.cards.size()) return 1;

    return -1; // draw
}

// ─── moveToActionIndex ────────────────────────────────
int moveToActionIndex(const Move& move, const GameState& state) {
    switch (move.type) {

        case BUY_CARD:
            if (move.from_reserved)
                return 12 + move.card_slot;          // 12-14
            return move.card_tier * 4 + move.card_slot; // 0-11

        case RESERVE_CARD:
            if (move.from_deck)
                return 27 + move.card_tier;           // 27-29
            return 15 + move.card_tier * 4 + move.card_slot; // 15-26

        case TAKE_GEMS: {
            const Tokens& t = move.gems_taken;
            int total = t.white + t.blue + t.green + t.red + t.black;

            if (total == 2 && (t.white==2||t.blue==2||t.green==2||
                               t.red==2||t.black==2)) {
                // Take 2 same (40-44)
                if (t.white==2) return 40;
                if (t.blue ==2) return 41;
                if (t.green==2) return 42;
                if (t.red  ==2) return 43;
                if (t.black==2) return 44;
            }

            // Encode which colors taken as bitmask w=1,b=2,g=4,r=8,k=16
            int mask = (t.white?1:0)|(t.blue?2:0)|(t.green?4:0)|
                       (t.red?8:0)|(t.black?16:0);

            if (total == 3) {
                // Take 3 different (30-39)
                // Order: wbg=7,wbr=11,wbk=19,wgr=13,wgk=21,wrk=25,
                //        bgr=14,bgk=22,brk=26,grk=28
                switch(mask) {
                    case 7:  return 30; // w+b+g
                    case 11: return 31; // w+b+r
                    case 19: return 32; // w+b+k
                    case 13: return 33; // w+g+r
                    case 21: return 34; // w+g+k
                    case 25: return 35; // w+r+k
                    case 14: return 36; // b+g+r
                    case 22: return 37; // b+g+k
                    case 26: return 38; // b+r+k
                    case 28: return 39; // g+r+k
                }
            }

            if (total == 2) {
                // Take 2 different (45-54)
                switch(mask) {
                    case 3:  return 45; // w+b
                    case 5:  return 46; // w+g
                    case 9:  return 47; // w+r
                    case 17: return 48; // w+k
                    case 6:  return 49; // b+g
                    case 10: return 50; // b+r
                    case 18: return 51; // b+k
                    case 12: return 52; // g+r
                    case 20: return 53; // g+k
                    case 24: return 54; // r+k
                }
            }

            if (total == 1) {
                // Take 1 (55-59)
                if (t.white) return 55;
                if (t.blue)  return 56;
                if (t.green) return 57;
                if (t.red)   return 58;
                if (t.black) return 59;
            }
            return -1;
        }

        case PASS_TURN:
            return 60;

        case RETURN_GEM: {
            const Tokens& r = move.gem_returned;
            if (r.white) return 61;
            if (r.blue)  return 62;
            if (r.green) return 63;
            if (r.red)   return 64;
            if (r.black) return 65;
            return -1;
        }
    }
    return -1;
}

// ─── actionIndexToMove ────────────────────────────────
Move actionIndexToMove(int idx, const GameState& state) {
    Move m;
    const Player& player = state.players[state.current_player];

    // Buy face-up (0-11)
    if (idx >= 0 && idx <= 11) {
        m.type      = BUY_CARD;
        m.card_tier = idx / 4;
        m.card_slot = idx % 4;
        return m;
    }
    // Buy reserved (12-14)
    if (idx >= 12 && idx <= 14) {
        m.type         = BUY_CARD;
        m.card_slot    = idx - 12;
        m.from_reserved = true;
        return m;
    }
    // Reserve face-up (15-26)
    if (idx >= 15 && idx <= 26) {
        m.type      = RESERVE_CARD;
        int rel     = idx - 15;
        m.card_tier = rel / 4;
        m.card_slot = rel % 4;
        return m;
    }
    // Reserve from deck (27-29)
    if (idx >= 27 && idx <= 29) {
        m.type      = RESERVE_CARD;
        m.card_tier = idx - 27;
        m.from_deck = true;
        return m;
    }

    const Color colors[] = {Color::White, Color::Blue, Color::Green, Color::Red, Color::Black};

    // Take 3 different (30-39)
    if (idx >= 30 && idx <= 39) {
        m.type = TAKE_GEMS;
        // Map index to color triplets
        const int triplets[10][3] = {
            {0,1,2},{0,1,3},{0,1,4},{0,2,3},{0,2,4},
            {0,3,4},{1,2,3},{1,2,4},{1,3,4},{2,3,4}
        };
        int rel = idx - 30;
        m.gems_taken[colors[triplets[rel][0]]] = 1;
        m.gems_taken[colors[triplets[rel][1]]] = 1;
        m.gems_taken[colors[triplets[rel][2]]] = 1;
        return m;
    }
    // Take 2 same (40-44)
    if (idx >= 40 && idx <= 44) {
        m.type = TAKE_GEMS;
        m.gems_taken[colors[idx-40]] = 2;
        return m;
    }
    // Take 2 different (45-54)
    if (idx >= 45 && idx <= 54) {
        m.type = TAKE_GEMS;
        const int pairs[10][2] = {
            {0,1},{0,2},{0,3},{0,4},{1,2},
            {1,3},{1,4},{2,3},{2,4},{3,4}
        };
        int rel = idx - 45;
        m.gems_taken[colors[pairs[rel][0]]] = 1;
        m.gems_taken[colors[pairs[rel][1]]] = 1;
        return m;
    }
    // Take 1 (55-59)
    if (idx >= 55 && idx <= 59) {
        m.type = TAKE_GEMS;
        m.gems_taken[colors[idx-55]] = 1;
        return m;
    }
    // Pass (60)
    if (idx == 60) {
        m.type = PASS_TURN;
        return m;
    }
    // Return gem (61-65)
    if (idx >= 61 && idx <= 65) {
        m.type = RETURN_GEM;
        m.gem_returned[colors[idx-61]] = 1;
        return m;
    }

    return m; // fallback PASS
}

// ─── getValidMoveMask ─────────────────────────────────
std::array<int, 66> getValidMoveMask(const GameState& state) {
    std::array<int, 66> mask = {};
    std::vector<Move> valid = findAllValidMoves(state);
    for (const Move& move : valid) {
        int idx = moveToActionIndex(move, state);
        if (idx >= 0 && idx < 66)
            mask[idx] = 1;
    }
    return mask;
}
