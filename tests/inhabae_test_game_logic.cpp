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
            const Card& c = state.faceup[tier][static_cast<std::size_t>(slot)];
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
        if (a.available_nobles[static_cast<std::size_t>(i)].id !=
            b.available_nobles[static_cast<std::size_t>(i)].id) nobles_differ = true;
    }

    for (int tier = 0; tier < 3 && !faceup_differs; ++tier) {
        for (int slot = 0; slot < 4; ++slot) {
            if (a.faceup[tier][static_cast<std::size_t>(slot)].id !=
                b.faceup[tier][static_cast<std::size_t>(slot)].id) {
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
            visit_card(state.faceup[tier][static_cast<std::size_t>(slot)]);
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

static void check_indices_eq(const std::vector<int>& actual, std::initializer_list<int> expected) {
    CHECK(actual.size() == expected.size());
    size_t i = 0;
    for (int v : expected) {
        CHECK(actual[i] == v);
        ++i;
    }
}

static void test_get_claimable_noble_indices_none_claimable() {
    GameState state{};
    state.noble_count = 3;
    state.available_nobles[0] = make_test_noble(Tokens{3, 3, 3, 0, 0, 0});
    state.available_nobles[1] = make_test_noble(Tokens{0, 4, 4, 0, 0, 0});
    state.available_nobles[2] = make_test_noble(Tokens{0, 3, 3, 3, 0, 0});
    check_indices_eq(testHook_getClaimableNobleIndices(state, 0), {});
}

static void test_get_claimable_noble_indices_one_claimable() {
    GameState state{};
    state.noble_count = 3;
    state.available_nobles[0] = make_test_noble(Tokens{3, 0, 0, 3, 3, 0});
    state.available_nobles[1] = make_test_noble(Tokens{3, 3, 3, 0, 0, 0});
    state.available_nobles[2] = make_test_noble(Tokens{4, 4, 0, 0, 0, 0});
    state.players[0].bonuses.white = 3;
    state.players[0].bonuses.blue = 3;
    state.players[0].bonuses.green = 5;
    check_indices_eq(testHook_getClaimableNobleIndices(state, 0), {1});
}

static void test_get_claimable_noble_indices_multiple_claimable_in_order() {
    GameState state{};
    state.noble_count = 3;
    state.available_nobles[0] = make_test_noble(Tokens{4, 4, 0, 0, 0, 0});
    state.available_nobles[1] = make_test_noble(Tokens{0, 3, 3, 3, 0, 0});
    state.available_nobles[2] = make_test_noble(Tokens{3, 3, 0, 0, 3, 0});
    state.players[0].bonuses = Tokens{4, 4, 0, 0, 3, 0};
    check_indices_eq(testHook_getClaimableNobleIndices(state, 0), {0, 2});
}

static void test_get_claimable_noble_indices_all_claimable() {
    GameState state{};
    state.noble_count = 3;
    state.available_nobles[0] = make_test_noble(Tokens{3, 3, 3, 0, 0, 0});
    state.available_nobles[1] = make_test_noble(Tokens{0, 3, 3, 3, 0, 0});
    state.available_nobles[2] = make_test_noble(Tokens{4, 0, 0, 0, 4, 0});
    state.players[0].bonuses = Tokens{4, 3, 3, 3, 4, 0};
    check_indices_eq(testHook_getClaimableNobleIndices(state, 0), {0, 1, 2});
}

static void test_get_claimable_noble_indices_respects_noble_count() {
    GameState state{};
    state.noble_count = 2;
    state.available_nobles[0] = make_test_noble(Tokens{3, 3, 3, 0, 0, 0});
    state.available_nobles[1] = make_test_noble(Tokens{0, 3, 3, 3, 0, 0});
    state.available_nobles[2] = make_test_noble(Tokens{0, 0, 3, 3, 3, 0});
    state.players[0].bonuses.blue = 4;
    state.players[0].bonuses.red = 5;
    state.players[0].bonuses.black = 6;
    check_indices_eq(testHook_getClaimableNobleIndices(state, 0), {});
}

static void test_get_claimable_noble_indices_uses_requested_player() {
    GameState state{};
    state.noble_count = 2;
    state.available_nobles[0] = make_test_noble(Tokens{3, 3, 3, 0, 0, 0});
    state.available_nobles[1] = make_test_noble(Tokens{3, 3, 0, 0, 3, 0});
    state.players[0].bonuses.white = 3;
    state.players[0].bonuses.blue = 4;
    state.players[0].bonuses.black = 3;
    state.players[1].bonuses.white = 3;
    state.players[1].bonuses.blue = 3;
    state.players[1].bonuses.green = 5;
    check_indices_eq(testHook_getClaimableNobleIndices(state, 0), {1});
    check_indices_eq(testHook_getClaimableNobleIndices(state, 1), {0});
}

static void test_get_claimable_noble_indices_invalid_player_idx_throws() {
    GameState state{};
    bool threw_neg = false;
    try {
        (void)testHook_getClaimableNobleIndices(state, -1);
    } catch (const std::invalid_argument&) {
        threw_neg = true;
    }
    CHECK(threw_neg);

    bool threw_big = false;
    try {
        (void)testHook_getClaimableNobleIndices(state, 2);
    } catch (const std::invalid_argument&) {
        threw_big = true;
    }
    CHECK(threw_big);
}

static void test_claim_noble_by_index_success_middle_compacts_and_updates_player() {
    GameState state{};
    state.noble_count = 3;

    Noble n0 = make_test_noble(Tokens{9, 0, 0, 0, 0, 0}); n0.id = 10; n0.points = 3;
    Noble n1 = make_test_noble(Tokens{0, 2, 0, 0, 0, 0}); n1.id = 20; n1.points = 3;
    Noble n2 = make_test_noble(Tokens{0, 0, 3, 0, 0, 0}); n2.id = 30; n2.points = 3;
    state.available_nobles[0] = n0;
    state.available_nobles[1] = n1;
    state.available_nobles[2] = n2;

    state.players[0].bonuses.blue = 2;  // claim index 1 only
    state.players[1].bonuses.green = 3; // should remain untouched
    state.players[1].points = 7;

    testHook_claimNobleByIndex(state, 0, 1);

    CHECK(state.players[0].nobles.size() == 1);
    CHECK(state.players[0].nobles[0].id == 20);
    CHECK(state.players[0].points == 3);
    CHECK(state.noble_count == 2);

    // Compaction: old index 2 shifts into index 1.
    CHECK(state.available_nobles[0].id == 10);
    CHECK(state.available_nobles[1].id == 30);
    CHECK(state.available_nobles[2].id == 0);

    // No unintended mutation to other player.
    CHECK(state.players[1].nobles.empty());
    CHECK(state.players[1].points == 7);
}

static void test_claim_noble_by_index_success_last_index_clears_tail() {
    GameState state{};
    state.noble_count = 3;

    Noble n0 = make_test_noble(Tokens{1, 0, 0, 0, 0, 0}); n0.id = 11;
    Noble n1 = make_test_noble(Tokens{0, 1, 0, 0, 0, 0}); n1.id = 22;
    Noble n2 = make_test_noble(Tokens{0, 0, 1, 0, 0, 0}); n2.id = 33;
    state.available_nobles[0] = n0;
    state.available_nobles[1] = n1;
    state.available_nobles[2] = n2;
    state.players[0].bonuses.green = 1;

    testHook_claimNobleByIndex(state, 0, 2);

    CHECK(state.noble_count == 2);
    CHECK(state.available_nobles[0].id == 11);
    CHECK(state.available_nobles[1].id == 22);
    CHECK(state.available_nobles[2].id == 0);
    CHECK(state.players[0].nobles.size() == 1);
    CHECK(state.players[0].nobles[0].id == 33);
}

static void test_claim_noble_by_index_single_noble_claim_clears_slot() {
    GameState state{};
    state.noble_count = 1;
    Noble n0 = make_test_noble(Tokens{0, 0, 0, 1, 0, 0}); n0.id = 99;
    state.available_nobles[0] = n0;
    state.players[0].bonuses.red = 1;

    testHook_claimNobleByIndex(state, 0, 0);

    CHECK(state.noble_count == 0);
    CHECK(state.available_nobles[0].id == 0);
    CHECK(state.players[0].nobles.size() == 1);
    CHECK(state.players[0].nobles[0].id == 99);
}

static void test_claim_noble_by_index_invalid_noble_index_throws() {
    GameState state{};
    state.noble_count = 1;
    state.available_nobles[0] = make_test_noble(Tokens{0, 0, 0, 0, 0, 0});

    bool threw_neg = false;
    try {
        testHook_claimNobleByIndex(state, 0, -1);
    } catch (const std::runtime_error&) {
        threw_neg = true;
    }
    CHECK(threw_neg);

    bool threw_big = false;
    try {
        testHook_claimNobleByIndex(state, 0, 1);
    } catch (const std::runtime_error&) {
        threw_big = true;
    }
    CHECK(threw_big);
}

static void test_claim_noble_by_index_unclaimable_throws() {
    GameState state{};
    state.noble_count = 1;
    Noble noble = make_test_noble(Tokens{0, 0, 0, 0, 4, 0});
    noble.id = 44;
    state.available_nobles[0] = noble;
    state.players[0].bonuses.black = 3;

    bool threw = false;
    try {
        testHook_claimNobleByIndex(state, 0, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(state.noble_count == 1);
    CHECK(state.players[0].nobles.empty());
    CHECK(state.players[0].points == 0);
}

static void test_claim_noble_by_index_invalid_player_idx_throws() {
    GameState state{};
    state.noble_count = 1;
    state.available_nobles[0] = make_test_noble(Tokens{0, 0, 0, 0, 0, 0});

    bool threw_neg = false;
    try {
        testHook_claimNobleByIndex(state, -1, 0);
    } catch (const std::invalid_argument&) {
        threw_neg = true;
    }
    CHECK(threw_neg);

    bool threw_big = false;
    try {
        testHook_claimNobleByIndex(state, 2, 0);
    } catch (const std::invalid_argument&) {
        threw_big = true;
    }
    CHECK(threw_big);
}

static void expect_validate_move_passes(const GameState& state, const Move& move) {
    testHook_validateMoveForApply(state, move);
}

static void expect_validate_move_throws(const GameState& state, const Move& move) {
    bool threw = false;
    try {
        testHook_validateMoveForApply(state, move);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

static GameState make_validate_state() {
    GameState state{};
    state.current_player = 0;
    return state;
}

static void test_validate_move_for_apply_invalid_current_player_throws() {
    GameState state = make_validate_state();
    Move m;
    m.type = PASS_TURN;

    state.current_player = -1;
    expect_validate_move_throws(state, m);

    state.current_player = 2;
    expect_validate_move_throws(state, m);
}

static void test_validate_move_for_apply_invalid_move_type_throws() {
    GameState state = make_validate_state();
    Move m;
    m.type = static_cast<MoveType>(999);
    expect_validate_move_throws(state, m);
}

static void test_validate_move_for_apply_pass_turn_phase_rules() {
    GameState state = make_validate_state();
    Move m;
    m.type = PASS_TURN;

    expect_validate_move_passes(state, m);

    state.is_return_phase = true;
    expect_validate_move_throws(state, m);
    state.is_return_phase = false;

    state.is_noble_choice_phase = true;
    expect_validate_move_throws(state, m);
}

static void test_validate_move_for_apply_buy_valid_faceup_passes() {
    GameState state = make_validate_state();
    state.faceup[0][0] = Card{1, 1, 0, Color::Blue, Tokens{1, 0, 0, 0, 0, 0}};
    state.players[0].tokens.white = 1;

    Move m;
    m.type = BUY_CARD;
    m.card_tier = 0;
    m.card_slot = 0;
    expect_validate_move_passes(state, m);
}

static void test_validate_move_for_apply_buy_uses_current_player() {
    GameState state = make_validate_state();
    state.current_player = 1;
    state.faceup[0][0] = Card{1, 1, 0, Color::Blue, Tokens{1, 0, 0, 0, 0, 0}};
    state.players[0].tokens.white = 0;
    state.players[1].tokens.white = 1;

    Move m;
    m.type = BUY_CARD;
    m.card_tier = 0;
    m.card_slot = 0;
    expect_validate_move_passes(state, m);
}

static void test_validate_move_for_apply_buy_invalid_shape_and_phase_cases_throw() {
    GameState state = make_validate_state();
    state.faceup[0][0] = Card{1, 1, 0, Color::Blue, Tokens{}};
    Move m;
    m.type = BUY_CARD;

    m.from_deck = true;
    expect_validate_move_throws(state, m);
    m.from_deck = false;

    m.card_tier = 9;
    m.card_slot = 0;
    expect_validate_move_throws(state, m);

    m.card_tier = 0;
    m.card_slot = 9;
    expect_validate_move_throws(state, m);

    state.is_return_phase = true;
    m.card_tier = 0;
    m.card_slot = 0;
    expect_validate_move_throws(state, m);
    state.is_return_phase = false;

    state.is_noble_choice_phase = true;
    expect_validate_move_throws(state, m);
}

static void test_validate_move_for_apply_buy_empty_slot_and_reserved_slot_cases() {
    GameState state = make_validate_state();
    Move m;
    m.type = BUY_CARD;
    m.card_tier = 0;
    m.card_slot = 0;

    // Face-up empty slot.
    expect_validate_move_throws(state, m);

    // Reserved slot out of range.
    m.from_reserved = true;
    expect_validate_move_throws(state, m);

    // Reserved valid slot passes when affordable.
    state.players[0].reserved.push_back(ReservedCard{Card{9, 1, 0, Color::White, Tokens{1, 0, 0, 0, 0, 0}}, true});
    state.players[0].tokens.white = 1;
    expect_validate_move_passes(state, m);
}

static void test_validate_move_for_apply_reserve_valid_cases_pass() {
    GameState state = make_validate_state();
    state.faceup[0][0] = Card{1, 1, 0, Color::Blue, Tokens{}};
    state.deck[1].push_back(Card{20, 2, 0, Color::Red, Tokens{}});

    Move faceup;
    faceup.type = RESERVE_CARD;
    faceup.card_tier = 0;
    faceup.card_slot = 0;
    expect_validate_move_passes(state, faceup);

    Move from_deck;
    from_deck.type = RESERVE_CARD;
    from_deck.from_deck = true;
    from_deck.card_tier = 1;
    expect_validate_move_passes(state, from_deck);
}

static void test_validate_move_for_apply_reserve_invalid_cases_throw() {
    GameState state = make_validate_state();
    Move m;
    m.type = RESERVE_CARD;
    m.card_tier = 0;
    m.card_slot = 0;

    // Empty face-up slot.
    expect_validate_move_throws(state, m);

    // Invalid tier / slot.
    m.card_tier = -1;
    expect_validate_move_throws(state, m);
    m.card_tier = 0;
    m.card_slot = 4;
    expect_validate_move_throws(state, m);

    // Reserve from empty deck.
    m.from_deck = true;
    m.card_tier = 2;
    expect_validate_move_throws(state, m);
    m.from_deck = false;

    // Reserved full.
    state.faceup[0][0] = Card{1, 1, 0, Color::Blue, Tokens{}};
    state.players[0].reserved.resize(3);
    m.card_tier = 0;
    m.card_slot = 0;
    expect_validate_move_throws(state, m);

    // Phase gated.
    state.players[0].reserved.clear();
    state.is_return_phase = true;
    expect_validate_move_throws(state, m);
    state.is_return_phase = false;
    state.is_noble_choice_phase = true;
    expect_validate_move_throws(state, m);
}

static void test_validate_move_for_apply_take_gems_valid_patterns_pass() {
    GameState state = make_validate_state();
    Move m;
    m.type = TAKE_GEMS;

    // total=3, three different, >=3 colors available
    state.bank = Tokens{4, 4, 4, 0, 0, 5};
    m.gems_taken = Tokens{1, 1, 1, 0, 0, 0};
    expect_validate_move_passes(state, m);

    // total=2, same color x2, bank has >=4
    state.bank = Tokens{4, 0, 0, 0, 0, 5};
    m.gems_taken = Tokens{2, 0, 0, 0, 0, 0};
    expect_validate_move_passes(state, m);

    // total=2, two different, exactly two colors available
    state.bank = Tokens{1, 1, 0, 0, 0, 5};
    m.gems_taken = Tokens{1, 1, 0, 0, 0, 0};
    expect_validate_move_passes(state, m);

    // total=1, exactly one color available
    state.bank = Tokens{0, 0, 0, 1, 0, 5};
    m.gems_taken = Tokens{0, 0, 0, 1, 0, 0};
    expect_validate_move_passes(state, m);
}

static void test_validate_move_for_apply_take_gems_invalid_cases_throw() {
    GameState state = make_validate_state();
    Move m;
    m.type = TAKE_GEMS;

    // Negative payload.
    state.bank = Tokens{4, 4, 4, 4, 4, 5};
    m.gems_taken = Tokens{-1, 0, 0, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // Joker in payload.
    m.gems_taken = Tokens{0, 0, 0, 0, 0, 1};
    expect_validate_move_throws(state, m);

    // More than available in bank.
    state.bank = Tokens{1, 0, 0, 0, 0, 5};
    m.gems_taken = Tokens{2, 0, 0, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // More than 2 of one color.
    state.bank = Tokens{4, 4, 4, 4, 4, 5};
    m.gems_taken = Tokens{3, 0, 0, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // total=3 but not three different.
    m.gems_taken = Tokens{2, 1, 0, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // total=3 but fewer than 3 colors available in bank.
    state.bank = Tokens{4, 4, 0, 0, 0, 5};
    m.gems_taken = Tokens{1, 1, 1, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // total=2 same color but bank has <4.
    state.bank = Tokens{3, 0, 0, 0, 0, 5};
    m.gems_taken = Tokens{2, 0, 0, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // total=2 different but not exactly two colors available in bank.
    state.bank = Tokens{1, 1, 1, 0, 0, 5};
    m.gems_taken = Tokens{1, 1, 0, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // total=1 but not exactly one color available in bank.
    state.bank = Tokens{1, 1, 0, 0, 0, 5};
    m.gems_taken = Tokens{1, 0, 0, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // Invalid total.
    state.bank = Tokens{4, 4, 4, 4, 4, 5};
    m.gems_taken = Tokens{};
    expect_validate_move_throws(state, m);

    // Phase gated.
    state.is_return_phase = true;
    m.gems_taken = Tokens{1, 1, 1, 0, 0, 0};
    expect_validate_move_throws(state, m);
    state.is_return_phase = false;
    state.is_noble_choice_phase = true;
    expect_validate_move_throws(state, m);
}

static void test_validate_move_for_apply_return_gem_valid_and_invalid_cases() {
    GameState state = make_validate_state();
    Move m;
    m.type = RETURN_GEM;

    // Not in return phase.
    m.gem_returned = Tokens{1, 0, 0, 0, 0, 0};
    expect_validate_move_throws(state, m);

    // Valid return.
    state.is_return_phase = true;
    state.players[0].tokens.white = 1;
    expect_validate_move_passes(state, m);

    // Player lacks the gem.
    state.players[0].tokens.white = 0;
    expect_validate_move_throws(state, m);

    // Invalid shape (joker).
    m.gem_returned = Tokens{0, 0, 0, 0, 0, 1};
    expect_validate_move_throws(state, m);

    // Noble-choice phase blocks return gem even if return phase is true.
    state.is_noble_choice_phase = true;
    m.gem_returned = Tokens{1, 0, 0, 0, 0, 0};
    state.players[0].tokens.white = 1;
    expect_validate_move_throws(state, m);
}

static void test_validate_move_for_apply_choose_noble_valid_and_invalid_cases() {
    GameState state = make_validate_state();
    Move m;
    m.type = CHOOSE_NOBLE;
    m.noble_idx = 0;

    // Not in noble choice phase.
    expect_validate_move_throws(state, m);

    // Setup noble choice phase with one claimable noble at index 1.
    state.is_noble_choice_phase = true;
    state.noble_count = 2;
    state.available_nobles[0] = make_test_noble(Tokens{3, 0, 0, 0, 0, 0});
    state.available_nobles[1] = make_test_noble(Tokens{0, 2, 0, 0, 0, 0});
    state.players[0].bonuses.blue = 2;

    // Invalid index.
    m.noble_idx = -1;
    expect_validate_move_throws(state, m);
    m.noble_idx = 2;
    expect_validate_move_throws(state, m);

    // In-range but not claimable.
    m.noble_idx = 0;
    expect_validate_move_throws(state, m);

    // Claimable index passes.
    m.noble_idx = 1;
    expect_validate_move_passes(state, m);
}

static void test_validate_move_for_apply_non_choose_actions_rejected_in_noble_choice_phase() {
    GameState state = make_validate_state();
    state.is_noble_choice_phase = true;

    Move buy;
    buy.type = BUY_CARD;
    buy.card_tier = 0;
    buy.card_slot = 0;
    state.faceup[0][0] = Card{1, 1, 0, Color::Blue, Tokens{}};
    expect_validate_move_throws(state, buy);

    Move reserve;
    reserve.type = RESERVE_CARD;
    reserve.card_tier = 0;
    reserve.card_slot = 0;
    expect_validate_move_throws(state, reserve);

    Move take;
    take.type = TAKE_GEMS;
    take.gems_taken = Tokens{1, 1, 1, 0, 0, 0};
    state.bank = Tokens{4, 4, 4, 4, 4, 5};
    expect_validate_move_throws(state, take);

    Move pass;
    pass.type = PASS_TURN;
    expect_validate_move_throws(state, pass);
}

static int count_moves_of_type_local(const std::vector<Move>& moves, MoveType type) {
    int count = 0;
    for (const Move& m : moves) {
        if (m.type == type) count++;
    }
    return count;
}

static bool contains_return_color(const std::vector<Move>& moves, Color color) {
    for (const Move& m : moves) {
        if (m.type == RETURN_GEM && m.gem_returned[color] == 1) return true;
    }
    return false;
}

static bool contains_choose_noble_index(const std::vector<Move>& moves, int noble_idx) {
    for (const Move& m : moves) {
        if (m.type == CHOOSE_NOBLE && m.noble_idx == noble_idx) return true;
    }
    return false;
}

static bool contains_buy_faceup(const std::vector<Move>& moves, int tier, int slot) {
    for (const Move& m : moves) {
        if (m.type == BUY_CARD && !m.from_reserved && m.card_tier == tier && m.card_slot == slot) return true;
    }
    return false;
}

static bool contains_buy_reserved(const std::vector<Move>& moves, int slot) {
    for (const Move& m : moves) {
        if (m.type == BUY_CARD && m.from_reserved && m.card_slot == slot) return true;
    }
    return false;
}

static bool contains_reserve_faceup(const std::vector<Move>& moves, int tier, int slot) {
    for (const Move& m : moves) {
        if (m.type == RESERVE_CARD && !m.from_deck && m.card_tier == tier && m.card_slot == slot) return true;
    }
    return false;
}

static bool contains_reserve_deck(const std::vector<Move>& moves, int tier) {
    for (const Move& m : moves) {
        if (m.type == RESERVE_CARD && m.from_deck && m.card_tier == tier) return true;
    }
    return false;
}

static bool contains_take_gems(const std::vector<Move>& moves, const Tokens& gems) {
    for (const Move& m : moves) {
        if (m.type != TAKE_GEMS) continue;
        if (m.gems_taken.white == gems.white &&
            m.gems_taken.blue  == gems.blue &&
            m.gems_taken.green == gems.green &&
            m.gems_taken.red   == gems.red &&
            m.gems_taken.black == gems.black &&
            m.gems_taken.joker == gems.joker) {
            return true;
        }
    }
    return false;
}

static void test_find_all_valid_moves_return_phase_only_returns_colored_gems() {
    GameState state{};
    state.current_player = 0;
    state.is_return_phase = true;
    state.players[0].tokens.white = 1;
    state.players[0].tokens.green = 2;
    state.players[0].tokens.joker = 3; // should not create RETURN_GEM move in current action space

    std::vector<Move> moves = findAllValidMoves(state);
    CHECK(count_moves_of_type_local(moves, RETURN_GEM) == 2);
    CHECK(static_cast<int>(moves.size()) == 2);
    CHECK(contains_return_color(moves, Color::White));
    CHECK(contains_return_color(moves, Color::Green));
    CHECK(!contains_return_color(moves, Color::Blue));
    CHECK(!contains_return_color(moves, Color::Joker));
}

static void test_find_all_valid_moves_noble_choice_phase_only_claimable_choices() {
    GameState state{};
    state.current_player = 0;
    state.is_noble_choice_phase = true;
    state.noble_count = 3;
    state.available_nobles[0] = make_test_noble(Tokens{3, 0, 0, 0, 0, 0});
    state.available_nobles[1] = make_test_noble(Tokens{0, 2, 0, 0, 0, 0});
    state.available_nobles[2] = make_test_noble(Tokens{0, 0, 3, 0, 0, 0});
    state.players[0].bonuses.white = 3;
    state.players[0].bonuses.green = 3;

    std::vector<Move> moves = findAllValidMoves(state);
    CHECK(count_moves_of_type_local(moves, CHOOSE_NOBLE) == 2);
    CHECK(static_cast<int>(moves.size()) == 2);
    CHECK(contains_choose_noble_index(moves, 0));
    CHECK(!contains_choose_noble_index(moves, 1));
    CHECK(contains_choose_noble_index(moves, 2));
}

static void test_find_all_valid_moves_generates_buy_faceup_and_reserved_when_affordable() {
    GameState state{};
    state.current_player = 0;
    state.faceup[0][0] = Card{1, 1, 0, Color::White, Tokens{1, 0, 0, 0, 0, 0}};
    state.faceup[0][1] = Card{2, 1, 0, Color::Blue, Tokens{0, 2, 0, 0, 0, 0}};
    state.players[0].reserved.push_back(ReservedCard{Card{3, 1, 0, Color::Red, Tokens{0, 0, 1, 0, 0, 0}}, true});
    state.players[0].reserved.push_back(ReservedCard{Card{4, 1, 0, Color::Green, Tokens{0, 0, 0, 2, 0, 0}}, true});
    state.players[0].tokens.white = 1;
    state.players[0].tokens.green = 1;

    std::vector<Move> moves = findAllValidMoves(state);

    CHECK(contains_buy_faceup(moves, 0, 0));
    CHECK(!contains_buy_faceup(moves, 0, 1)); // unaffordable
    CHECK(contains_buy_reserved(moves, 0));
    CHECK(!contains_buy_reserved(moves, 1)); // unaffordable
}

static void test_find_all_valid_moves_reserve_generation_respects_capacity_and_sources() {
    GameState state{};
    state.current_player = 0;
    state.faceup[0][0] = Card{1, 1, 0, Color::White, {}};
    state.faceup[2][3] = Card{2, 3, 0, Color::Blue, {}};
    state.deck[0].push_back(Card{11, 1, 0, Color::Green, {}});
    state.deck[2].push_back(Card{12, 3, 0, Color::Red, {}});

    std::vector<Move> moves = findAllValidMoves(state);
    CHECK(contains_reserve_faceup(moves, 0, 0));
    CHECK(contains_reserve_faceup(moves, 2, 3));
    CHECK(contains_reserve_deck(moves, 0));
    CHECK(!contains_reserve_deck(moves, 1)); // empty deck
    CHECK(contains_reserve_deck(moves, 2));

    state.players[0].reserved.resize(3);
    moves = findAllValidMoves(state);
    CHECK(!contains_reserve_faceup(moves, 0, 0));
    CHECK(!contains_reserve_deck(moves, 0));
    CHECK(!contains_reserve_deck(moves, 2));
}

static void test_find_all_valid_moves_take_gems_three_different_when_three_plus_colors_available() {
    GameState state{};
    state.current_player = 0;
    state.bank = Tokens{1, 1, 1, 1, 1, 5}; // 5 available colors

    std::vector<Move> moves = findAllValidMoves(state);

    CHECK(contains_take_gems(moves, Tokens{1, 1, 1, 0, 0, 0}));
    CHECK(contains_take_gems(moves, Tokens{1, 0, 1, 0, 1, 0}));
    CHECK(count_moves_of_type_local(moves, TAKE_GEMS) >= 10); // at least the 10 three-color combos
}

static void test_find_all_valid_moves_take_gems_two_same_requires_bank_four() {
    GameState state{};
    state.current_player = 0;
    state.bank = Tokens{4, 3, 4, 0, 5, 5};

    std::vector<Move> moves = findAllValidMoves(state);

    CHECK(contains_take_gems(moves, Tokens{2, 0, 0, 0, 0, 0})); // white (4)
    CHECK(!contains_take_gems(moves, Tokens{0, 2, 0, 0, 0, 0})); // blue (3)
    CHECK(contains_take_gems(moves, Tokens{0, 0, 2, 0, 0, 0})); // green (4)
    CHECK(contains_take_gems(moves, Tokens{0, 0, 0, 0, 2, 0})); // black (5)
}

static void test_find_all_valid_moves_take_gems_two_different_only_when_exactly_two_colors_available() {
    GameState state{};
    state.current_player = 0;
    state.bank = Tokens{1, 0, 2, 0, 0, 5}; // exactly white+green available

    std::vector<Move> moves = findAllValidMoves(state);
    CHECK(contains_take_gems(moves, Tokens{1, 0, 1, 0, 0, 0}));

    // No three-different possible with only 2 colors.
    CHECK(!contains_take_gems(moves, Tokens{1, 1, 1, 0, 0, 0}));
}

static void test_find_all_valid_moves_take_one_only_when_exactly_one_color_available() {
    GameState state{};
    state.current_player = 0;
    state.bank = Tokens{0, 0, 0, 1, 0, 5}; // exactly one color available, count <4 so no two-same

    std::vector<Move> moves = findAllValidMoves(state);
    CHECK(contains_take_gems(moves, Tokens{0, 0, 0, 1, 0, 0}));
    CHECK(!contains_take_gems(moves, Tokens{0, 0, 0, 2, 0, 0}));
}

static void test_find_all_valid_moves_returns_pass_only_when_no_other_moves_exist() {
    GameState state{};
    state.current_player = 0;
    state.bank = Tokens{}; // no gems at all
    // no face-up cards, no reserved cards, no decks

    std::vector<Move> moves = findAllValidMoves(state);
    CHECK(static_cast<int>(moves.size()) == 1);
    CHECK(moves[0].type == PASS_TURN);
}

static void test_find_all_valid_moves_does_not_add_pass_when_other_moves_exist() {
    GameState state{};
    state.current_player = 0;
    state.bank = Tokens{1, 1, 1, 0, 0, 5};

    std::vector<Move> moves = findAllValidMoves(state);
    CHECK(count_moves_of_type_local(moves, PASS_TURN) == 0);
    CHECK(count_moves_of_type_local(moves, TAKE_GEMS) > 0);
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
            faceup_ids_before[t][s] = state.faceup[t][static_cast<std::size_t>(s)].id;
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
            CHECK(state.faceup[t][static_cast<std::size_t>(s)].id == faceup_ids_before[t][s]);
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
            faceup_ids_before[t][s] = state.faceup[t][static_cast<std::size_t>(s)].id;
        }
    }

    // First call on empty deck should be safe and clear target slot.
    testHook_refillSlot(state, tier, slot);

    CHECK(state.deck[tier].size() == 0);
    CHECK(state.deck[0].size() == deck_sizes_before[0]);
    CHECK(state.deck[1].size() == deck_sizes_before[1]);
    CHECK(state.deck[2].size() == deck_sizes_before[2]);

    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].id == 0);
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].level == 0);
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].points == 0);
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].cost.white == 0);
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].cost.blue == 0);
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].cost.green == 0);
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].cost.red == 0);
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].cost.black == 0);
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].cost.joker == 0);

    for (int t = 0; t < 3; ++t) {
        for (int s = 0; s < 4; ++s) {
            if (t == tier && s == slot) continue;
            CHECK(state.faceup[t][static_cast<std::size_t>(s)].id == faceup_ids_before[t][s]);
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
    CHECK(state.faceup[tier][static_cast<std::size_t>(slot)].id == 0);
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

    // getClaimableNobleIndices test hook
    test_get_claimable_noble_indices_none_claimable();
    test_get_claimable_noble_indices_one_claimable();
    test_get_claimable_noble_indices_multiple_claimable_in_order();
    test_get_claimable_noble_indices_all_claimable();
    test_get_claimable_noble_indices_respects_noble_count();
    test_get_claimable_noble_indices_uses_requested_player();
    test_get_claimable_noble_indices_invalid_player_idx_throws();

    // claimNobleByIndex test hook
    test_claim_noble_by_index_success_middle_compacts_and_updates_player();
    test_claim_noble_by_index_success_last_index_clears_tail();
    test_claim_noble_by_index_single_noble_claim_clears_slot();
    test_claim_noble_by_index_invalid_noble_index_throws();
    test_claim_noble_by_index_unclaimable_throws();
    test_claim_noble_by_index_invalid_player_idx_throws();

    // validateMoveForApply test hook
    test_validate_move_for_apply_invalid_current_player_throws();
    test_validate_move_for_apply_invalid_move_type_throws();
    test_validate_move_for_apply_pass_turn_phase_rules();
    test_validate_move_for_apply_buy_valid_faceup_passes();
    test_validate_move_for_apply_buy_uses_current_player();
    test_validate_move_for_apply_buy_invalid_shape_and_phase_cases_throw();
    test_validate_move_for_apply_buy_empty_slot_and_reserved_slot_cases();
    test_validate_move_for_apply_reserve_valid_cases_pass();
    test_validate_move_for_apply_reserve_invalid_cases_throw();
    test_validate_move_for_apply_take_gems_valid_patterns_pass();
    test_validate_move_for_apply_take_gems_invalid_cases_throw();
    test_validate_move_for_apply_return_gem_valid_and_invalid_cases();
    test_validate_move_for_apply_choose_noble_valid_and_invalid_cases();
    test_validate_move_for_apply_non_choose_actions_rejected_in_noble_choice_phase();

    // findAllValidMoves (public) branch matrix
    test_find_all_valid_moves_return_phase_only_returns_colored_gems();
    test_find_all_valid_moves_noble_choice_phase_only_claimable_choices();
    test_find_all_valid_moves_generates_buy_faceup_and_reserved_when_affordable();
    test_find_all_valid_moves_reserve_generation_respects_capacity_and_sources();
    test_find_all_valid_moves_take_gems_three_different_when_three_plus_colors_available();
    test_find_all_valid_moves_take_gems_two_same_requires_bank_four();
    test_find_all_valid_moves_take_gems_two_different_only_when_exactly_two_colors_available();
    test_find_all_valid_moves_take_one_only_when_exactly_one_color_available();
    test_find_all_valid_moves_returns_pass_only_when_no_other_moves_exist();
    test_find_all_valid_moves_does_not_add_pass_when_other_moves_exist();

    // refillSlot test hook
    test_refill_slot_nonempty_deck_replaces_from_deck_and_only_mutates_target();
    test_refill_slot_empty_deck_clears_slot_then_rejects_empty_slot_refill();
#endif

    std::cout << "inhabae_test_game_logic passed." << std::endl;
    return 0;
}
