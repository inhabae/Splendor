// game_logic.h
#pragma once
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <random>

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
        return joker;
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
        return joker;
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

// Loading
std::vector<Card>  loadCards (const std::string& path);
std::vector<Noble> loadNobles(const std::string& path);

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
int                  moveToActionIndex(const Move& move, const GameState& state);
Move                 actionIndexToMove(int action_idx,   const GameState& state);
std::array<int, 69>  getValidMoveMask (const GameState& state);
