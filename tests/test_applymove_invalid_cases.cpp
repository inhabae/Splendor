#include "../game_logic.h"

#include <exception>
#include <iostream>
#include <string>

static int run_scenario(const std::string& scenario) {
    GameState state;
    Move move;

    if (scenario == "control_valid_pass") {
        try {
            move.type = PASS_TURN;
            applyMove(state, move);
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Unexpected exception in control_valid_pass: " << e.what() << std::endl;
            return 1;
        }
    }

    if (scenario == "buy_reserved_out_of_range") {
        try {
            move.type = BUY_CARD;
            move.from_reserved = true;
            move.card_slot = 0;
            applyMove(state, move);
            std::cerr << "Expected exception for buy_reserved_out_of_range, but none was thrown" << std::endl;
            return 1;
        } catch (const std::exception&) {
            return 0;
        }
    }

    if (scenario == "reserve_from_empty_deck") {
        try {
            move.type = RESERVE_CARD;
            move.from_deck = true;
            move.card_tier = 0;
            applyMove(state, move);
            std::cerr << "Expected exception for reserve_from_empty_deck, but none was thrown" << std::endl;
            return 1;
        } catch (const std::exception&) {
            return 0;
        }
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
