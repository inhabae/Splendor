#include "../game_logic.h"

#include <cstdlib>
#include <iostream>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "CHECK failed at " << __FILE__ << ":" << __LINE__ << ": "  \
                      << #cond << std::endl;                                         \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

static void test_initialize_game_has_three_nobles() {
    GameState state;
    initializeGame(state, 123);

    CHECK(state.noble_count == 3);
    CHECK(state.available_nobles[0].id != 0);
    CHECK(state.available_nobles[1].id != 0);
    CHECK(state.available_nobles[2].id != 0);
}

static void test_initialize_game_nobles_are_unique() {
    GameState state;
    initializeGame(state, 123);

    CHECK(state.available_nobles[0].id != state.available_nobles[1].id);
    CHECK(state.available_nobles[0].id != state.available_nobles[2].id);
    CHECK(state.available_nobles[1].id != state.available_nobles[2].id);
}

static void test_initialize_game_bank_has_standard_2p_counts() {
    GameState state;
    initializeGame(state, 123);

    CHECK(state.bank.white == 4);
    CHECK(state.bank.blue == 4);
    CHECK(state.bank.green == 4);
    CHECK(state.bank.red == 4);
    CHECK(state.bank.black == 4);
    CHECK(state.bank.joker == 5);
}

static void test_initialize_game_faceup_tiers_are_full_and_correct() {
    GameState state;
    initializeGame(state, 123);

    for (int tier = 0; tier < 3; ++tier) {
        for (int slot = 0; slot < 4; ++slot) {
            const Card& c = state.faceup[tier][slot];
            CHECK(c.id != 0);              // no empty slot
            CHECK(c.level == tier + 1);    // card belongs to this tier
        }
    }
}

static void test_initialize_game_decks_have_expected_sizes_and_tiers() {
    GameState state;
    initializeGame(state, 123);

    CHECK(state.deck[0].size() == 36);
    CHECK(state.deck[1].size() == 26);
    CHECK(state.deck[2].size() == 16);

    for (const Card& c : state.deck[0]) CHECK(c.level == 1);
    for (const Card& c : state.deck[1]) CHECK(c.level == 2);
    for (const Card& c : state.deck[2]) CHECK(c.level == 3);
}

static void test_initialize_game_contains_all_90_unique_cards() {
    GameState state;
    initializeGame(state, 123);

    bool seen_ids[91] = {};
    int total_cards = 0;

    auto visit_card = [&](const Card& c) {
        CHECK(c.id > 0);
        CHECK(c.id <= 90);
        CHECK(!seen_ids[c.id]);
        seen_ids[c.id] = true;
        total_cards++;
    };

    for (int tier = 0; tier < 3; ++tier) {
        for (int slot = 0; slot < 4; ++slot) {
            visit_card(state.faceup[tier][slot]);
        }
    }

    for (int tier = 0; tier < 3; ++tier) {
        for (const Card& c : state.deck[tier]) {
            visit_card(c);
        }
    }

    CHECK(total_cards == 90);
    for (int id = 1; id <= 90; ++id) CHECK(seen_ids[id]);
}

int main() {
    // Nobles
    test_initialize_game_has_three_nobles();
    test_initialize_game_nobles_are_unique();
    test_initialize_game_bank_has_standard_2p_counts();

    // Face-up tiers
    test_initialize_game_faceup_tiers_are_full_and_correct();

    // Decks
    test_initialize_game_decks_have_expected_sizes_and_tiers();

    // Whole-card-set consistency
    test_initialize_game_contains_all_90_unique_cards();

    std::cout << "inhabae_test_game_logic passed." << std::endl;
    return 0;
}
