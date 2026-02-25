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

static void test_standard_cards_have_expected_total_and_tier_counts() {
    const auto& cards = standardCards();
    int tier1 = 0, tier2 = 0, tier3 = 0;

    for (const Card& c : cards) {
        if (c.level == 1) tier1++;
        else if (c.level == 2) tier2++;
        else if (c.level == 3) tier3++;
        else CHECK(false);
    }

    CHECK(cards.size() == 90);
    CHECK(tier1 == 40);
    CHECK(tier2 == 30);
    CHECK(tier3 == 20);
}

static void test_standard_nobles_have_expected_total() {
    const auto& nobles = standardNobles();
    CHECK(nobles.size() == 10);
}

static void check_initialized_nobles_basic(const GameState& state) {
    CHECK(state.noble_count == 3);
    CHECK(state.available_nobles[0].id != 0);
    CHECK(state.available_nobles[1].id != 0);
    CHECK(state.available_nobles[2].id != 0);
}

static void check_initialized_turn_state(const GameState& state) {
    CHECK(state.current_player == 0);
    CHECK(state.move_number == 0);
    CHECK(state.is_return_phase == false);
    CHECK(state.is_noble_choice_phase == false);
}

static void check_initialized_players_zeroed(const GameState& state) {
    for (int p = 0; p < 2; ++p) {
        const Player& player = state.players[p];
        CHECK(player.tokens.white == 0);
        CHECK(player.tokens.blue == 0);
        CHECK(player.tokens.green == 0);
        CHECK(player.tokens.red == 0);
        CHECK(player.tokens.black == 0);
        CHECK(player.tokens.joker == 0);

        CHECK(player.bonuses.white == 0);
        CHECK(player.bonuses.blue == 0);
        CHECK(player.bonuses.green == 0);
        CHECK(player.bonuses.red == 0);
        CHECK(player.bonuses.black == 0);
        CHECK(player.bonuses.joker == 0);

        CHECK(player.points == 0);
        CHECK(player.cards.empty());
        CHECK(player.nobles.empty());
        CHECK(player.reserved.empty());
    }
}

static void check_initialized_faceup_tiers(const GameState& state) {
    for (int tier = 0; tier < 3; ++tier) {
        for (int slot = 0; slot < 4; ++slot) {
            const Card& c = state.faceup[tier][slot];
            CHECK(c.id != 0);
            CHECK(c.level == tier + 1);
        }
    }
}

static void check_initialized_deck_sizes_and_tiers(const GameState& state) {
    CHECK(state.deck[0].size() == 36);
    CHECK(state.deck[1].size() == 26);
    CHECK(state.deck[2].size() == 16);

    for (const Card& c : state.deck[0]) CHECK(c.level == 1);
    for (const Card& c : state.deck[1]) CHECK(c.level == 2);
    for (const Card& c : state.deck[2]) CHECK(c.level == 3);
}

static void test_initialize_game_has_three_nobles() {
    GameState state;
    initializeGame(state, 123);
    check_initialized_nobles_basic(state);
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

static void test_initialize_game_basic_turn_state_is_reset() {
    GameState state;
    initializeGame(state, 123);
    check_initialized_turn_state(state);
}

static void test_initialize_game_players_start_empty_and_zeroed() {
    GameState state;
    initializeGame(state, 123);
    check_initialized_players_zeroed(state);
}

static void test_initialize_game_clears_dirty_state() {
    GameState state;

    // Deliberately dirty many fields to ensure initializeGame fully resets state.
    state.current_player = 1;
    state.move_number = 99;
    state.is_return_phase = true;
    state.is_noble_choice_phase = true;
    state.noble_count = 2;
    state.available_nobles[0].id = 999;
    state.faceup[0][0].id = 777;
    state.faceup[0][0].level = 3;
    state.deck[0].push_back(Card{1234, 1, 0, Color::White, {}});
    state.players[0].tokens.white = 5;
    state.players[0].bonuses.blue = 2;
    state.players[0].points = 10;
    state.players[0].cards.push_back(Card{1, 1, 0, Color::White, {}});
    state.players[0].reserved.push_back(ReservedCard{Card{2, 1, 0, Color::Blue, {}}, true});
    state.players[0].nobles.push_back(Noble{1, 3, {}});

    initializeGame(state, 123);

    check_initialized_turn_state(state);
    check_initialized_nobles_basic(state);
    check_initialized_players_zeroed(state);
    check_initialized_faceup_tiers(state);
    check_initialized_deck_sizes_and_tiers(state);
}

static void test_initialize_game_faceup_tiers_are_full_and_correct() {
    GameState state;
    initializeGame(state, 123);
    check_initialized_faceup_tiers(state);
}

static void test_initialize_game_decks_have_expected_sizes_and_tiers() {
    GameState state;
    initializeGame(state, 123);
    check_initialized_deck_sizes_and_tiers(state);
}

static void test_initialize_game_different_seeds_change_state_smoke() {
    GameState a, b;
    initializeGame(a, 123);
    initializeGame(b, 456);

    bool nobles_differ = false;
    bool faceup_differs = false;
    bool decks_differ = false;

    for (int i = 0; i < a.noble_count && !nobles_differ; ++i) {
        if (a.available_nobles[i].id != b.available_nobles[i].id) nobles_differ = true;
    }

    for (int tier = 0; tier < 3 && !faceup_differs; ++tier) {
        for (int slot = 0; slot < 4; ++slot) {
            if (a.faceup[tier][slot].id != b.faceup[tier][slot].id) {
                faceup_differs = true;
                break;
            }
        }
    }

    for (int tier = 0; tier < 3 && !decks_differ; ++tier) {
        if (a.deck[tier].size() != b.deck[tier].size()) {
            decks_differ = true;
            break;
        }
        for (size_t i = 0; i < a.deck[tier].size(); ++i) {
            if (a.deck[tier][i].id != b.deck[tier][i].id) {
                decks_differ = true;
                break;
            }
        }
    }

    CHECK(faceup_differs || decks_differ || nobles_differ);
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

#ifdef SPLENDOR_TEST_HOOKS
static Noble make_test_noble(const Tokens& req) {
    Noble n;
    n.id = 1;
    n.points = 3;
    n.requirements = req;
    return n;
}

static Player make_test_player_with_bonuses(const Tokens& bonuses) {
    Player p;
    p.bonuses = bonuses;
    return p;
}

static void test_can_claim_noble_exact_match_passes() {
    Noble noble = make_test_noble(Tokens{3, 3, 0, 0, 3, 0});
    Player player = make_test_player_with_bonuses(Tokens{3, 3, 0, 0, 3, 0});
    CHECK(testHook_canClaimNoble(player, noble));
}

static void test_can_claim_noble_exceeding_bonuses_passes() {
    Noble noble = make_test_noble(Tokens{3, 0, 3, 3, 0, 0});
    Player player = make_test_player_with_bonuses(Tokens{4, 1, 5, 5, 3, 0});
    CHECK(testHook_canClaimNoble(player, noble));
}

static void test_can_claim_noble_one_color_short_fails() {
    Noble noble = make_test_noble(Tokens{3, 3, 3, 0, 0, 0});
    Player player = make_test_player_with_bonuses(Tokens{3, 2, 3, 0, 0, 0});
    CHECK(!testHook_canClaimNoble(player, noble));
}

static void test_can_claim_noble_multiple_colors_short_fails() {
    Noble noble = make_test_noble(Tokens{0, 4, 4, 0, 0, 0});
    Player player = make_test_player_with_bonuses(Tokens{1, 2, 0, 0, 0, 0});
    CHECK(!testHook_canClaimNoble(player, noble));
}

static void test_can_claim_noble_tokens_do_not_substitute_for_bonuses() {
    Noble noble = make_test_noble(Tokens{3, 3, 3, 0, 0, 0});
    Player player = make_test_player_with_bonuses(Tokens{0, 0, 0, 0, 0, 0});
    player.tokens.white = 4;
    player.tokens.blue = 4;
    player.tokens.green = 4;
    player.tokens.red = 4;
    player.tokens.black = 4;
    CHECK(!testHook_canClaimNoble(player, noble));
}

static void test_can_claim_noble_joker_does_not_cover_bonus_shortfall() {
    Noble noble = make_test_noble(Tokens{0, 0, 0, 3, 3, 3});
    Player player = make_test_player_with_bonuses(Tokens{0, 0, 0, 2, 3, 3});
    player.tokens.joker = 1;
    CHECK(!testHook_canClaimNoble(player, noble));
}

static void test_refill_slot_nonempty_deck_replaces_from_deck_and_only_mutates_target() {
    GameState state;
    initializeGame(state, 123);

    const int tier = 0;
    const int slot = 0;

    CHECK(!state.deck[tier].empty());

    const int old_slot_id = state.faceup[tier][slot].id;
    const Card expected_new_card = state.deck[tier].back();

    int faceup_ids_before[3][4];
    for (int t = 0; t < 3; ++t) {
        for (int s = 0; s < 4; ++s) {
            faceup_ids_before[t][s] = state.faceup[t][s].id;
        }
    }

    const size_t deck_sizes_before[3] = {
        state.deck[0].size(), state.deck[1].size(), state.deck[2].size()
    };

    bool expected_new_card_was_in_deck = false;
    for (const Card& c : state.deck[tier]) {
        if (c.id == expected_new_card.id) {
            expected_new_card_was_in_deck = true;
            break;
        }
    }
    CHECK(expected_new_card_was_in_deck);

    testHook_refillSlot(state, tier, slot);

    // Basic replacement behavior on non-empty deck.
    CHECK(state.faceup[tier][slot].id != old_slot_id);
    CHECK(state.faceup[tier][slot].id == expected_new_card.id);
    CHECK(state.faceup[tier][slot].level == tier + 1);

    // Deck size behavior.
    CHECK(state.deck[tier].size() == deck_sizes_before[tier] - 1);
    CHECK(state.deck[1].size() == deck_sizes_before[1]);
    CHECK(state.deck[2].size() == deck_sizes_before[2]);

    // Card conservation: replacement came from prior deck contents and was removed from deck.
    bool new_card_still_in_deck = false;
    for (const Card& c : state.deck[tier]) {
        if (c.id == expected_new_card.id) {
            new_card_still_in_deck = true;
            break;
        }
    }
    CHECK(!new_card_still_in_deck);

    // No unintended mutation: all other face-up slots remain unchanged.
    for (int t = 0; t < 3; ++t) {
        for (int s = 0; s < 4; ++s) {
            if (t == tier && s == slot) continue;
            CHECK(state.faceup[t][s].id == faceup_ids_before[t][s]);
        }
    }
}

static void test_refill_slot_empty_deck_clears_slot_then_rejects_empty_slot_refill() {
    GameState state;
    initializeGame(state, 123);

    const int tier = 1;
    const int slot = 2;

    // Make only the target tier deck empty while keeping the rest of the initialized state intact.
    state.deck[tier].clear();

    const size_t deck_sizes_before[3] = {
        state.deck[0].size(), state.deck[1].size(), state.deck[2].size()
    };

    int faceup_ids_before[3][4];
    for (int t = 0; t < 3; ++t) {
        for (int s = 0; s < 4; ++s) {
            faceup_ids_before[t][s] = state.faceup[t][s].id;
        }
    }

    // First call on empty deck should be safe and clear target slot.
    testHook_refillSlot(state, tier, slot);

    CHECK(state.deck[tier].size() == 0);
    CHECK(state.deck[0].size() == deck_sizes_before[0]);
    CHECK(state.deck[1].size() == deck_sizes_before[1]);
    CHECK(state.deck[2].size() == deck_sizes_before[2]);

    CHECK(state.faceup[tier][slot].id == 0);
    CHECK(state.faceup[tier][slot].level == 0);
    CHECK(state.faceup[tier][slot].points == 0);
    CHECK(state.faceup[tier][slot].cost.white == 0);
    CHECK(state.faceup[tier][slot].cost.blue == 0);
    CHECK(state.faceup[tier][slot].cost.green == 0);
    CHECK(state.faceup[tier][slot].cost.red == 0);
    CHECK(state.faceup[tier][slot].cost.black == 0);
    CHECK(state.faceup[tier][slot].cost.joker == 0);

    for (int t = 0; t < 3; ++t) {
        for (int s = 0; s < 4; ++s) {
            if (t == tier && s == slot) continue;
            CHECK(state.faceup[t][s].id == faceup_ids_before[t][s]);
        }
    }

    // Calling refill again on an already-empty slot should now be rejected.
    bool threw = false;
    try {
        testHook_refillSlot(state, tier, slot);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(state.deck[tier].size() == 0);
    CHECK(state.faceup[tier][slot].id == 0);
}
#endif

int main() {
    // Standard dataset accessors
    test_standard_cards_have_expected_total_and_tier_counts();
    test_standard_nobles_have_expected_total();

    // Nobles
    test_initialize_game_has_three_nobles();
    test_initialize_game_nobles_are_unique();
    test_initialize_game_bank_has_standard_2p_counts();
    test_initialize_game_basic_turn_state_is_reset();
    test_initialize_game_players_start_empty_and_zeroed();
    test_initialize_game_clears_dirty_state();

    // Face-up tiers
    test_initialize_game_faceup_tiers_are_full_and_correct();

    // Decks
    test_initialize_game_decks_have_expected_sizes_and_tiers();
    test_initialize_game_different_seeds_change_state_smoke();

    // Whole-card-set consistency
    test_initialize_game_contains_all_90_unique_cards();

#ifdef SPLENDOR_TEST_HOOKS
    // canClaimNoble test hook
    test_can_claim_noble_exact_match_passes();
    test_can_claim_noble_exceeding_bonuses_passes();
    test_can_claim_noble_one_color_short_fails();
    test_can_claim_noble_multiple_colors_short_fails();
    test_can_claim_noble_tokens_do_not_substitute_for_bonuses();
    test_can_claim_noble_joker_does_not_cover_bonus_shortfall();

    // refillSlot test hook
    test_refill_slot_nonempty_deck_replaces_from_deck_and_only_mutates_target();
    test_refill_slot_empty_deck_clears_slot_then_rejects_empty_slot_refill();
#endif

    std::cout << "inhabae_test_game_logic passed." << std::endl;
    return 0;
}
