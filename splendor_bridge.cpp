// splendor_bridge.cpp
// Stdin/stdout interface between C++ game logic and Python neural network
#include "game_logic.h"
#include "simple_json_parse.h"
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

// ─── Flat state array ─────────────────────────────────
// Outputs raw integers Python needs for get_encoded_state()
// Python applies normalization — C++ just sends raw values
// Order matches encode_state.py exactly (246 values total):
//   current player: tokens(6) bonuses(5) points(1) reserved(3x11)
//   opponent:       tokens(6) bonuses(5) points(1) reserved(3x11; face-up reserves visible,
//                   deck reserves/empty slots zero-masked) reserved_count(1)
//   board:          faceup(12x11) bank(6) nobles(3x5)
//   phase:          is_return_phase(1) is_noble_choice_phase(1)

static void appendCard(std::ostringstream& ss, const Card& c, bool& first) {
    auto sep = [&]() { if (!first) ss << ","; first = false; };

    if (c.id == 0) {
        for (int i = 0; i < 11; i++) { sep(); ss << 0; }
        return;
    }
    // cost: white, blue, green, red, black
    sep(); ss << c.cost.white;
    sep(); ss << c.cost.blue;
    sep(); ss << c.cost.green;
    sep(); ss << c.cost.red;
    sep(); ss << c.cost.black;
    // bonus one-hot
    sep(); ss << (c.color == Color::White ? 1 : 0);
    sep(); ss << (c.color == Color::Blue  ? 1 : 0);
    sep(); ss << (c.color == Color::Green ? 1 : 0);
    sep(); ss << (c.color == Color::Red   ? 1 : 0);
    sep(); ss << (c.color == Color::Black ? 1 : 0);
    // points
    sep(); ss << c.points;
}

static std::string buildStateArray(const GameState& state) {
    std::ostringstream ss;
    ss << "[";
    bool first = true;
    auto sep = [&]() { if (!first) ss << ","; first = false; };

    int cur = state.current_player;
    int opp = 1 - cur;
    const Player& cp = state.players[cur];
    const Player& op = state.players[opp];

    // current player tokens (6)
    sep(); ss << cp.tokens.white;
    sep(); ss << cp.tokens.blue;
    sep(); ss << cp.tokens.green;
    sep(); ss << cp.tokens.red;
    sep(); ss << cp.tokens.black;
    sep(); ss << cp.tokens.joker;

    // current player bonuses (5)
    sep(); ss << cp.bonuses.white;
    sep(); ss << cp.bonuses.blue;
    sep(); ss << cp.bonuses.green;
    sep(); ss << cp.bonuses.red;
    sep(); ss << cp.bonuses.black;

    // current player points (1)
    sep(); ss << cp.points;

    // current player reserved (3 x 11)
    static const Card kEmptyCard{};
    for (int i = 0; i < 3; i++) {
        if (i < (int)cp.reserved.size()) {
            appendCard(ss, cp.reserved[i].card, first);
        } else {
            appendCard(ss, kEmptyCard, first);
        }
    }

    // opponent tokens (6)
    sep(); ss << op.tokens.white;
    sep(); ss << op.tokens.blue;
    sep(); ss << op.tokens.green;
    sep(); ss << op.tokens.red;
    sep(); ss << op.tokens.black;
    sep(); ss << op.tokens.joker;

    // opponent bonuses (5)
    sep(); ss << op.bonuses.white;
    sep(); ss << op.bonuses.blue;
    sep(); ss << op.bonuses.green;
    sep(); ss << op.bonuses.red;
    sep(); ss << op.bonuses.black;

    // opponent points (1)
    sep(); ss << op.points;

    // Opponent reserved cards: show only publicly known face-up reservations.
    for (int i = 0; i < 3; i++) {
        if (i < (int)op.reserved.size() && op.reserved[i].is_public) {
            appendCard(ss, op.reserved[i].card, first);
        } else {
            appendCard(ss, kEmptyCard, first);
        }
    }
    // Opponent reserved count is public information, so expose it explicitly.
    sep(); ss << (int)op.reserved.size();

    // face-up cards tier1, tier2, tier3 (12 x 11)
    for (int t = 0; t < 3; t++)
        for (int s = 0; s < 4; s++)
            appendCard(ss, state.faceup[t][s], first);

    // bank tokens (6)
    sep(); ss << state.bank.white;
    sep(); ss << state.bank.blue;
    sep(); ss << state.bank.green;
    sep(); ss << state.bank.red;
    sep(); ss << state.bank.black;
    sep(); ss << state.bank.joker;

    // nobles (3 x 5)
    for (int i = 0; i < 3; i++) {
        if (i < state.noble_count) {
            const Noble& n = state.available_nobles[i];
            sep(); ss << n.requirements.white;
            sep(); ss << n.requirements.blue;
            sep(); ss << n.requirements.green;
            sep(); ss << n.requirements.red;
            sep(); ss << n.requirements.black;
        } else {
            for (int j = 0; j < 5; j++) { sep(); ss << 0; }
        }
    }

    // phase flags (2)
    sep(); ss << (state.is_return_phase ? 1 : 0);
    sep(); ss << (state.is_noble_choice_phase ? 1 : 0);

    ss << "]";
    return ss.str();
}

static std::string maskToJson(const std::array<int, 69>& mask) {
    std::ostringstream ss;
    ss << "[";
    for (int i = 0; i < 69; i++) {
        if (i > 0) ss << ",";
        ss << mask[i];
    }
    ss << "]";
    return ss.str();
}

// ─── Response Builders ────────────────────────────────

static std::string okResponse(const GameState& state) {
    auto mask     = getValidMoveMask(state);
    bool terminal = isGameOver(state);
    int  winner   = terminal ? determineWinner(state) : -2;

    std::ostringstream ss;
    ss << "{"
       << "\"status\":\"ok\","
       << "\"state\":"           << buildStateArray(state) << ","
       << "\"mask\":"            << maskToJson(mask) << ","
       << "\"is_return_phase\":" << (state.is_return_phase ? "true" : "false") << ","
       << "\"is_noble_choice_phase\":" << (state.is_noble_choice_phase ? "true" : "false") << ","
       << "\"is_terminal\":"     << (terminal ? "true" : "false") << ","
       << "\"winner\":"          << winner
       << "}";
    return ss.str();
}

static std::string errorResponse(const std::string& msg) {
    return "{\"status\":\"error\",\"message\":\"" + msg + "\"}";
}

// ─── Main Interface Loop ──────────────────────────────

int main(int argc, char* argv[]) {
    std::string cards_path  = "cards.json";
    std::string nobles_path = "nobles.json";

    if (argc >= 3) {
        cards_path  = argv[1];
        nobles_path = argv[2];
    }

    std::vector<Card>  all_cards;
    std::vector<Noble> all_nobles;
    try {
        all_cards  = loadCards (cards_path);
        all_nobles = loadNobles(nobles_path);
    } catch (const std::exception& e) {
        std::cout << errorResponse(e.what()) << std::endl;
        return 1;
    }

    GameState state;
    bool game_initialized = false;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::string cmd = simple_json::extractStr(line, "cmd");

        // init / reset
        if (cmd == "init" || cmd == "reset") {
            int seed = simple_json::extractInt(line, "seed", -1);
            if (seed < 0) seed = 0;
            initializeGame(state, all_cards, all_nobles, (unsigned int)seed);
            game_initialized = true;
            std::cout << okResponse(state) << std::endl;
        }

        // apply
        else if (cmd == "apply") {
            if (!game_initialized) {
                std::cout << errorResponse("Game not initialized") << std::endl;
                continue;
            }
            if (isGameOver(state)) {
                std::cout << errorResponse("Game is already over") << std::endl;
                continue;
            }
            int action_idx = simple_json::extractInt(line, "action", -1);
            if (action_idx < 0 || action_idx >= 69) {
                std::cout << errorResponse("Invalid action index") << std::endl;
                continue;
            }
            auto mask = getValidMoveMask(state);
            if (!mask[action_idx]) {
                std::cout << errorResponse("Action is not valid in current state") << std::endl;
                continue;
            }
            try {
                Move move = actionIndexToMove(action_idx, state);
                applyMove(state, move);
                std::cout << okResponse(state) << std::endl;
            } catch (const std::exception& e) {
                std::cout << errorResponse(e.what()) << std::endl;
            }
        }

        // get_state
        else if (cmd == "get_state") {
            if (!game_initialized) {
                std::cout << errorResponse("Game not initialized") << std::endl;
                continue;
            }
            std::cout << okResponse(state) << std::endl;
        }

        // quit
        else if (cmd == "quit") {
            break;
        }

        else {
            std::cout << errorResponse("Unknown command: " + cmd) << std::endl;
        }
    }

    return 0;
}
