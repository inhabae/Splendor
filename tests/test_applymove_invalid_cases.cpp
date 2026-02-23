#include "../game_logic.h"

#include <iostream>
#include <string>

static int run_scenario(const std::string& scenario) {
    GameState state;
    Move move;

    if (scenario == "control_valid_pass") {
        move.type = PASS_TURN;
        applyMove(state, move);
        return 0;
    }

    if (scenario == "buy_reserved_out_of_range") {
        move.type = BUY_CARD;
        move.from_reserved = true;
        move.card_slot = 0;
        applyMove(state, move); // Expected current behavior: may crash/abort (unsafe input).
        return 0;
    }

    if (scenario == "reserve_from_empty_deck") {
        move.type = RESERVE_CARD;
        move.from_deck = true;
        move.card_tier = 0;
        applyMove(state, move); // Expected current behavior: may crash/abort (unsafe input).
        return 0;
    }

    std::cerr << "Unknown scenario: " << scenario << std::endl;
    return 2;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <scenario>" << std::endl;
        return 2;
    }

    return run_scenario(argv[1]);
}
