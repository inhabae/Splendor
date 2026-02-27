#include "../game_logic.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

static const char* colorName(Color c) {
    switch (c) {
        case Color::White: return "W";
        case Color::Blue:  return "B";
        case Color::Green: return "G";
        case Color::Red:   return "R";
        case Color::Black: return "K";
        case Color::Joker: return "J";
    }
    return "?";
}

static void printTokens(const Tokens& t) {
    std::cout << "{W:" << t.white
              << " B:" << t.blue
              << " G:" << t.green
              << " R:" << t.red
              << " K:" << t.black
              << " J:" << t.joker << "}";
}

static void printCardShort(const Card& c) {
    if (c.id == 0) {
        std::cout << "[empty]";
        return;
    }
    std::cout << "[id=" << c.id
              << " t" << c.level
              << " " << colorName(c.color)
              << " p" << c.points
              << " cost=";
    printTokens(c.cost);
    std::cout << "]";
}

static void printNobleShort(const Noble& n) {
    if (n.id == 0) {
        std::cout << "[empty noble]";
        return;
    }
    std::cout << "[id=" << n.id << " p" << n.points << " req=";
    printTokens(n.requirements);
    std::cout << "]";
}

static std::string describeMove(const Move& m) {
    switch (m.type) {
        case BUY_CARD:
            if (m.from_reserved) return "BUY reserved slot " + std::to_string(m.card_slot);
            return "BUY faceup t" + std::to_string(m.card_tier + 1) +
                   " s" + std::to_string(m.card_slot);
        case RESERVE_CARD:
            if (m.from_deck) return "RESERVE from deck t" + std::to_string(m.card_tier + 1);
            return "RESERVE faceup t" + std::to_string(m.card_tier + 1) +
                   " s" + std::to_string(m.card_slot);
        case TAKE_GEMS: {
            std::string out = "TAKE ";
            bool first = true;
            auto add = [&](const char* name, int n) {
                if (n <= 0) return;
                if (!first) out += ", ";
                first = false;
                out += name;
                out += "x";
                out += std::to_string(n);
            };
            add("W", m.gems_taken.white);
            add("B", m.gems_taken.blue);
            add("G", m.gems_taken.green);
            add("R", m.gems_taken.red);
            add("K", m.gems_taken.black);
            if (first) out += "(none)";
            return out;
        }
        case RETURN_GEM: {
            const Color colors[] = {Color::White, Color::Blue, Color::Green, Color::Red, Color::Black};
            for (Color c : colors) {
                if (m.gem_returned[c] > 0) {
                    return std::string("RETURN ") + colorName(c);
                }
            }
            return "RETURN (invalid payload)";
        }
        case PASS_TURN:
            return "PASS";
        case CHOOSE_NOBLE:
            return "CHOOSE_NOBLE idx " + std::to_string(m.noble_idx);
    }
    return "UNKNOWN";
}

static void printPlayerView(const GameState& state) {
    const int cur = state.current_player;
    const int opp = 1 - cur;
    const Player& cp = state.players[cur];
    const Player& op = state.players[opp];

    std::cout << "\n=== Turn " << state.move_number
              << " | Current Player: P" << cur << " ===\n";
    std::cout << "Phase: return=" << (state.is_return_phase ? 1 : 0)
              << " noble_choice=" << (state.is_noble_choice_phase ? 1 : 0) << "\n";

    std::cout << "Bank ";
    printTokens(state.bank);
    std::cout << "\n";

    std::cout << "P" << cur << " tokens ";
    printTokens(cp.tokens);
    std::cout << " bonuses ";
    printTokens(cp.bonuses);
    std::cout << " points " << cp.points
              << " cards " << cp.cards.size()
              << " reserved " << cp.reserved.size()
              << " nobles " << cp.nobles.size() << "\n";

    std::cout << "P" << opp << " tokens ";
    printTokens(op.tokens);
    std::cout << " bonuses ";
    printTokens(op.bonuses);
    std::cout << " points " << op.points
              << " cards " << op.cards.size()
              << " reserved_count " << op.reserved.size()
              << " nobles " << op.nobles.size() << "\n";

    std::cout << "Available nobles (" << state.noble_count << "):\n";
    for (int i = 0; i < state.noble_count; ++i) {
        std::cout << "  [" << i << "] ";
        printNobleShort(state.available_nobles[static_cast<std::size_t>(i)]);
        std::cout << "\n";
    }

    std::cout << "Face-up cards:\n";
    for (int t = 0; t < 3; ++t) {
        std::cout << "  Tier " << (t + 1) << ": ";
        for (int s = 0; s < 4; ++s) {
            if (s > 0) std::cout << " | ";
            printCardShort(state.faceup[t][static_cast<std::size_t>(s)]);
        }
        std::cout << "   (deck size=" << state.deck[t].size() << ")\n";
    }

    std::cout << "Your reserved:\n";
    for (std::size_t i = 0; i < cp.reserved.size(); ++i) {
        std::cout << "  [" << i << "] ";
        printCardShort(cp.reserved[i].card);
        std::cout << " public=" << (cp.reserved[i].is_public ? 1 : 0) << "\n";
    }

    std::cout << "Opponent reserved (public only):\n";
    for (std::size_t i = 0; i < op.reserved.size(); ++i) {
        std::cout << "  [" << i << "] ";
        if (op.reserved[i].is_public) {
            printCardShort(op.reserved[i].card);
        } else {
            std::cout << "[hidden]";
        }
        std::cout << "\n";
    }
}

int main() {
    GameState state;
    initializeGame(state, 123);

    std::cout << "Manual game loop (testing). Enter action index, 'r' to reset, 'q' to quit.\n";
    std::cout << "Using built-in standard dataset. Initial seed = 123.\n";

    while (true) {
        printPlayerView(state);

        if (isGameOver(state)) {
            int winner = determineWinner(state);
            std::cout << "\nGAME OVER. Winner: " << winner << "\n";
            std::cout << "Enter 'r' to reset or 'q' to quit: ";
            std::string cmd;
            if (!std::getline(std::cin, cmd)) break;
            if (cmd == "q") break;
            if (cmd == "r") {
                initializeGame(state, 123);
                continue;
            }
            continue;
        }

        std::vector<Move> moves = findAllValidMoves(state);
        std::vector<std::pair<int, Move>> indexed_moves;
        indexed_moves.reserve(moves.size());
        for (const Move& m : moves) {
            indexed_moves.push_back({moveToActionIndex(m), m});
        }

        std::sort(indexed_moves.begin(), indexed_moves.end(),
                  [](const std::pair<int, Move>& a, const std::pair<int, Move>& b) {
                      return a.first < b.first;
                  });

        std::cout << "\nValid actions (" << indexed_moves.size() << "):\n";
        for (const auto& entry : indexed_moves) {
            std::cout << "  [" << entry.first << "] " << describeMove(entry.second) << "\n";
        }

        std::cout << "Choose action index (or 'r'/'q'): ";
        std::string input;
        if (!std::getline(std::cin, input)) break;
        if (input == "q") break;
        if (input == "r") {
            initializeGame(state, 123);
            continue;
        }

        int action_idx = -1;
        try {
            action_idx = std::stoi(input);
        } catch (const std::exception&) {
            std::cout << "Invalid input.\n";
            continue;
        }

        bool valid = false;
        for (const auto& entry : indexed_moves) {
            if (entry.first == action_idx) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            std::cout << "Action is not valid in current state.\n";
            continue;
        }

        try {
            applyMove(state, actionIndexToMove(action_idx));
        } catch (const std::exception& e) {
            std::cout << "applyMove error: " << e.what() << "\n";
        }
    }

    return 0;
}
