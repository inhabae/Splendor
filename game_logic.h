// game_logic.h
#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <stdexcept>

// ─── Tokens ───────────────────────────────────────────
enum class Color { White, Blue, Green, Red, Black, Joker };

struct Tokens {
    int white = 0, blue = 0, green = 0,
        red   = 0, black = 0, joker = 0;

    int total() const {
        return white + blue + green + red + black + joker;
    }
    int& operator[](Color color) {
        switch (color) {
            case Color::White: return white;
            case Color::Blue:  return blue;
            case Color::Green: return green;
            case Color::Red:   return red;
            case Color::Black: return black;
            case Color::Joker: return joker;
        }
        throw std::out_of_range("Invalid token color");
    }
    const int& operator[](Color color) const {
        switch (color) {
            case Color::White: return white;
            case Color::Blue:  return blue;
            case Color::Green: return green;
            case Color::Red:   return red;
            case Color::Black: return black;
            case Color::Joker: return joker;
        }
        throw std::out_of_range("Invalid token color");
    }
    Tokens operator+(const Tokens& o) const {
        return {white+o.white, blue+o.blue, green+o.green,
                red+o.red,     black+o.black, joker+o.joker};
    }
    Tokens operator-(const Tokens& o) const {
        return {white-o.white, blue-o.blue, green-o.green,
                red-o.red,     black-o.black, joker-o.joker};
    }
    Tokens& operator+=(const Tokens& o) {
        white+=o.white; blue+=o.blue; green+=o.green;
        red+=o.red;     black+=o.black; joker+=o.joker;
        return *this;
    }
    Tokens& operator-=(const Tokens& o) {
        white-=o.white; blue-=o.blue; green-=o.green;
        red-=o.red;     black-=o.black; joker-=o.joker;
        return *this;
    }
};

// ─── Card ─────────────────────────────────────────────
struct Card {
    int         id     = 0;
    int         level  = 0;
    int         points = 0;
    Color       color = Color::White;
    Tokens      cost;
};

// ─── Noble ────────────────────────────────────────────
struct Noble {
    int    id     = 0;
    int    points = 3;
    Tokens requirements;
};

// ─── Reserved Card (tracks public visibility provenance) ───────────────
struct ReservedCard {
    Card card;
    bool is_public = false; // true if reserved from face-up board; false if reserved from deck
};

// ─── Player ───────────────────────────────────────────
struct Player {
    Tokens             tokens;
    Tokens             bonuses;
    int                points = 0;
    std::vector<Card>  cards;     // purchased
    std::vector<ReservedCard> reserved;  // max 3, positional
    std::vector<Noble> nobles;
};

// ─── GameState ────────────────────────────────────────
struct GameState {
    Player              players[2];
    Tokens              bank;
    std::array<Card, 4> faceup[3];    // faceup[0]=tier1,[1]=tier2,[2]=tier3
    std::vector<Card>   deck[3];      // deck[0]=tier1, etc.
    std::array<Noble,3> available_nobles;
    int                 noble_count    = 0;
    int                 current_player = 0;
    int                 move_number    = 0;
    bool                is_return_phase = false;
    bool                is_noble_choice_phase = false;
};

// ─── Move ─────────────────────────────────────────────
enum MoveType { BUY_CARD, RESERVE_CARD, TAKE_GEMS, RETURN_GEM, PASS_TURN, CHOOSE_NOBLE };

struct Move {
    MoveType type       = PASS_TURN;
    int  card_tier      = -1;   // 0,1,2 for tiers
    int  card_slot      = -1;   // 0-3 faceup, 0-2 reserved
    bool from_deck      = false;
    bool from_reserved  = false;
    Tokens gems_taken;
    Tokens gem_returned;        // single color, used in return phase
    int  noble_idx      = -1;
};

// ─── Function Declarations ────────────────────────────

// Built-in standard Splendor dataset
const std::vector<Card>&  standardCards();
const std::vector<Noble>& standardNobles();

// Convenience overload for standard dataset
void initializeGame(GameState& state, unsigned int seed = 0);

// Game setup
void initializeGame(GameState&                state,
                    const std::vector<Card>&  all_cards,
                    const std::vector<Noble>& all_nobles,
                    unsigned int              seed = 0);

// Game logic
void              applyMove        (GameState& state, const Move& move);
std::vector<Move> findAllValidMoves(const GameState& state);
bool              isGameOver       (const GameState& state);
int               determineWinner  (const GameState& state);

// Action space mapping
int                  moveToActionIndex(const Move& move);
Move                 actionIndexToMove(int action_idx);
std::array<int, 69>  getValidMoveMask (const GameState& state);

#ifdef SPLENDOR_TEST_HOOKS
// Test-only wrapper for internal face-up refill helper.
void testHook_refillSlot(GameState& state, int tier, int slot);
bool testHook_canClaimNoble(const Player& player, const Noble& noble);
std::vector<int> testHook_getClaimableNobleIndices(const GameState& state, int player_idx);
void testHook_claimNobleByIndex(GameState& state, int player_idx, int noble_idx);
void testHook_validateMoveForApply(const GameState& state, const Move& move);
#endif
