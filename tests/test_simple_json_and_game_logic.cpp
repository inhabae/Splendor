#include "../game_logic.h"
#include "../simple_json_parse.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "CHECK failed at " << __FILE__ << ":" << __LINE__ << ": "  \
                      << #cond << std::endl;                                         \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

static Card make_card(int id, int level, int points, Color color, Tokens cost = {}) {
    Card c;
    c.id = id;
    c.level = level;
    c.points = points;
    c.color = color;
    c.cost = cost;
    return c;
}

static void clear_state(GameState& state) {
    state = GameState{};
}

static int count_moves_of_type(const std::vector<Move>& moves, MoveType type) {
    int count = 0;
    for (const auto& m : moves) {
        if (m.type == type) count++;
    }
    return count;
}

static bool same_tokens(const Tokens& a, const Tokens& b) {
    return a.white == b.white && a.blue == b.blue && a.green == b.green &&
           a.red == b.red && a.black == b.black && a.joker == b.joker;
}

static int count_mask_bits(const std::array<int, 66>& mask) {
    int count = 0;
    for (int v : mask) count += (v != 0);
    return count;
}

static bool has_take_move(const std::vector<Move>& moves, const Tokens& expected) {
    for (const auto& m : moves) {
        if (m.type == TAKE_GEMS && same_tokens(m.gems_taken, expected)) return true;
    }
    return false;
}

static bool has_single_pass_only(const std::vector<Move>& moves) {
    return moves.size() == 1 && moves[0].type == PASS_TURN;
}

static void check_no_negative_tokens(const GameState& state) {
    const Tokens players[] = {state.players[0].tokens, state.players[1].tokens};
    for (const Tokens& t : players) {
        CHECK(t.white >= 0);
        CHECK(t.blue >= 0);
        CHECK(t.green >= 0);
        CHECK(t.red >= 0);
        CHECK(t.black >= 0);
        CHECK(t.joker >= 0);
    }
    CHECK(state.bank.white >= 0);
    CHECK(state.bank.blue >= 0);
    CHECK(state.bank.green >= 0);
    CHECK(state.bank.red >= 0);
    CHECK(state.bank.black >= 0);
    CHECK(state.bank.joker >= 0);
}

static void check_token_conservation_2p(const GameState& state) {
    CHECK(state.players[0].tokens.white + state.players[1].tokens.white + state.bank.white == 4);
    CHECK(state.players[0].tokens.blue + state.players[1].tokens.blue + state.bank.blue == 4);
    CHECK(state.players[0].tokens.green + state.players[1].tokens.green + state.bank.green == 4);
    CHECK(state.players[0].tokens.red + state.players[1].tokens.red + state.bank.red == 4);
    CHECK(state.players[0].tokens.black + state.players[1].tokens.black + state.bank.black == 4);
    CHECK(state.players[0].tokens.joker + state.players[1].tokens.joker + state.bank.joker == 5);
}

static Move choose_deterministic_valid_move(const std::vector<Move>& moves) {
    const MoveType order[] = {RESERVE_CARD, TAKE_GEMS, RETURN_GEM, BUY_CARD, PASS_TURN};
    for (MoveType type : order) {
        for (const Move& m : moves) {
            if (m.type == type) return m;
        }
    }
    CHECK(false && "No valid move found");
    return Move{};
}

static void test_simple_json_extractors() {
    CHECK(simple_json::extractInt(R"({"seed":123})", "seed", -1) == 123);
    CHECK(simple_json::extractInt(R"({"cmd":"init"})", "seed", -1) == -1);
    CHECK(simple_json::extractInt(R"({"action":   42  })", "action", -1) == 42);
    CHECK(simple_json::extractInt(R"({"action":"abc"})", "action", -1) == -1);

    CHECK(simple_json::extractStr(R"({"cmd":"get_state"})", "cmd") == "get_state");
    CHECK(simple_json::extractStr(R"({"x":1})", "cmd").empty());

    std::string obj = simple_json::extractObject(
        R"({"id":1,"cost":{"blue":2,"green":1},"points":0})", "cost");
    CHECK(!obj.empty());
    CHECK(obj.front() == '{');
    CHECK(obj.find("\"blue\"") != std::string::npos);
    CHECK(obj.find("\"green\"") != std::string::npos);
    CHECK(simple_json::extractObject(R"({"x":1})", "cost") == "{}");
}

static void test_tokens_enum_indexing() {
    Tokens t;
    t[Color::White] = 1;
    t[Color::Blue] = 2;
    t[Color::Green] = 3;
    t[Color::Red] = 4;
    t[Color::Black] = 5;
    t[Color::Joker] = 6;

    CHECK(t.white == 1);
    CHECK(t.blue == 2);
    CHECK(t.green == 3);
    CHECK(t.red == 4);
    CHECK(t.black == 5);
    CHECK(t.joker == 6);

    const Tokens& ct = t;
    CHECK(ct[Color::Red] == 4);
    CHECK(ct[Color::Joker] == 6);
}

static void test_tokens_arithmetic() {
    Tokens a{1, 2, 3, 4, 5, 1};
    Tokens b{5, 4, 3, 2, 1, 2};

    Tokens sum = a + b;
    CHECK(sum.white == 6 && sum.blue == 6 && sum.green == 6);
    CHECK(sum.red == 6 && sum.black == 6 && sum.joker == 3);

    Tokens diff = a - b;
    CHECK(diff.white == -4 && diff.blue == -2 && diff.green == 0);
    CHECK(diff.red == 2 && diff.black == 4 && diff.joker == -1);

    a += b;
    CHECK(a.white == 6 && a.blue == 6 && a.green == 6);
    CHECK(a.red == 6 && a.black == 6 && a.joker == 3);

    a -= b;
    CHECK(a.white == 1 && a.blue == 2 && a.green == 3);
    CHECK(a.red == 4 && a.black == 5 && a.joker == 1);
}

static void test_file_loading_and_enum_conversion() {
    auto cards = loadCards("cards.json");
    CHECK(!cards.empty());
    for (const auto& c : cards) {
        CHECK(c.id > 0);
        CHECK(c.level >= 1 && c.level <= 3);
        CHECK(c.color != Color::Joker);
    }

    auto nobles = loadNobles("nobles.json");
    CHECK(!nobles.empty());
    for (const auto& n : nobles) {
        CHECK(n.id > 0);
        CHECK(n.points >= 0);
    }
}

static void test_invalid_card_color_throws() {
    const char* path = "tests/tmp_invalid_cards.json";
    {
        std::ofstream out(path);
        CHECK(out.is_open());
        out << R"([{"id":1,"level":1,"points":0,"color":"purple","cost":{"white":1}}])";
    }

    bool threw = false;
    try {
        (void)loadCards(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::remove(path);
    CHECK(threw);
}

static void test_find_all_valid_moves_initial_state() {
    GameState state;
    auto cards = loadCards("cards.json");
    auto nobles = loadNobles("nobles.json");
    initializeGame(state, cards, nobles, 123);

    auto moves = findAllValidMoves(state);
    CHECK(!moves.empty());

    bool found_take_or_reserve = false;
    for (const auto& m : moves) {
        if (m.type == TAKE_GEMS || m.type == RESERVE_CARD) {
            found_take_or_reserve = true;
            break;
        }
    }
    CHECK(found_take_or_reserve);
}

static void test_return_phase_move_generation() {
    GameState state;
    state.current_player = 0;
    state.is_return_phase = true;
    state.players[0].tokens.white = 2;
    state.players[0].tokens.blue = 1;
    state.players[0].tokens.black = 1;
    state.players[0].tokens.joker = 1; // joker should not generate return move in current action space

    auto moves = findAllValidMoves(state);
    CHECK(moves.size() == 3);
    for (const auto& m : moves) {
        CHECK(m.type == RETURN_GEM);
        CHECK(m.gem_returned.total() == 1);
        CHECK(m.gem_returned.joker == 0);
    }
}

static void test_action_roundtrip_for_gem_moves() {
    GameState state;

    std::vector<Move> samples;

    Move take3;
    take3.type = TAKE_GEMS;
    take3.gems_taken[Color::White] = 1;
    take3.gems_taken[Color::Blue] = 1;
    take3.gems_taken[Color::Green] = 1;
    samples.push_back(take3);

    Move take2same;
    take2same.type = TAKE_GEMS;
    take2same.gems_taken[Color::Red] = 2;
    samples.push_back(take2same);

    Move take2diff;
    take2diff.type = TAKE_GEMS;
    take2diff.gems_taken[Color::Green] = 1;
    take2diff.gems_taken[Color::Black] = 1;
    samples.push_back(take2diff);

    Move take1;
    take1.type = TAKE_GEMS;
    take1.gems_taken[Color::Blue] = 1;
    samples.push_back(take1);

    Move ret;
    ret.type = RETURN_GEM;
    ret.gem_returned[Color::Black] = 1;
    samples.push_back(ret);

    for (const auto& m : samples) {
        int idx = moveToActionIndex(m, state);
        CHECK(idx >= 0 && idx < 66);
        Move decoded = actionIndexToMove(idx, state);
        CHECK(moveToActionIndex(decoded, state) == idx);
    }
}

static void test_buy_card_updates_bonus_via_enum_color() {
    GameState state;
    state.current_player = 0;
    state.faceup[0][0] = Card{};
    state.faceup[0][0].id = 99;
    state.faceup[0][0].level = 1;
    state.faceup[0][0].points = 1;
    state.faceup[0][0].color = Color::Blue;
    state.faceup[0][0].cost.white = 1;

    state.players[0].tokens.white = 1;
    state.bank.white = 4;

    Move buy;
    buy.type = BUY_CARD;
    buy.card_tier = 0;
    buy.card_slot = 0;

    applyMove(state, buy);

    CHECK(state.players[0].bonuses.blue == 1);
    CHECK(state.players[0].points == 1);
    CHECK(state.players[0].cards.size() == 1);
    CHECK(state.players[0].cards[0].id == 99);
    CHECK(state.faceup[0][0].id == 0); // no deck to refill from
}

static void test_take_gems_enters_return_phase_when_exceeding_ten() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.players[0].tokens.white = 9;
    state.bank.white = 4;

    Move m;
    m.type = TAKE_GEMS;
    m.gems_taken.white = 2;

    applyMove(state, m);

    CHECK(state.players[0].tokens.white == 11);
    CHECK(state.bank.white == 2);
    CHECK(state.is_return_phase == true);
    CHECK(state.current_player == 0);
    CHECK(state.move_number == 0);
}

static void test_take_gems_ends_turn_when_total_at_most_ten() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.players[0].tokens.white = 7;
    state.bank.white = 1;
    state.bank.blue = 1;
    state.bank.green = 1;

    Move m;
    m.type = TAKE_GEMS;
    m.gems_taken.white = 1;
    m.gems_taken.blue = 1;
    m.gems_taken.green = 1;

    applyMove(state, m);

    CHECK(state.players[0].tokens.total() == 10);
    CHECK(state.is_return_phase == false);
    CHECK(state.current_player == 1);
    CHECK(state.move_number == 1);
}

static void test_return_gem_stays_in_return_phase_above_ten() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.is_return_phase = true;
    state.players[0].tokens.white = 12;

    Move m;
    m.type = RETURN_GEM;
    m.gem_returned.white = 1;

    applyMove(state, m);

    CHECK(state.players[0].tokens.white == 11);
    CHECK(state.bank.white == 1);
    CHECK(state.is_return_phase == true);
    CHECK(state.current_player == 0);
    CHECK(state.move_number == 0);
}

static void test_return_gem_exits_return_phase_at_ten() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.is_return_phase = true;
    state.players[0].tokens.white = 11;

    Move m;
    m.type = RETURN_GEM;
    m.gem_returned.white = 1;

    applyMove(state, m);

    CHECK(state.players[0].tokens.white == 10);
    CHECK(state.bank.white == 1);
    CHECK(state.is_return_phase == false);
    CHECK(state.current_player == 1);
    CHECK(state.move_number == 1);
}

static void test_reserve_faceup_refills_and_gives_joker() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.bank.joker = 5;

    Card face = make_card(10, 1, 0, Color::White);
    Card refill = make_card(11, 1, 1, Color::Blue);
    state.faceup[1][2] = face;
    state.deck[1].push_back(refill);

    Move m;
    m.type = RESERVE_CARD;
    m.card_tier = 1;
    m.card_slot = 2;

    applyMove(state, m);

    CHECK(state.players[0].reserved.size() == 1);
    CHECK(state.players[0].reserved[0].id == 10);
    CHECK(state.faceup[1][2].id == 11);
    CHECK(state.deck[1].empty());
    CHECK(state.players[0].tokens.joker == 1);
    CHECK(state.bank.joker == 4);
    CHECK(state.is_return_phase == false);
    CHECK(state.current_player == 1);
    CHECK(state.move_number == 1);
}

static void test_reserve_faceup_no_joker_when_bank_empty() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.bank.joker = 0;
    state.faceup[0][0] = make_card(21, 1, 0, Color::Green);

    Move m;
    m.type = RESERVE_CARD;
    m.card_tier = 0;
    m.card_slot = 0;

    applyMove(state, m);

    CHECK(state.players[0].reserved.size() == 1);
    CHECK(state.players[0].reserved[0].id == 21);
    CHECK(state.players[0].tokens.joker == 0);
    CHECK(state.bank.joker == 0);
}

static void test_reserve_from_deck_does_not_touch_faceup() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.faceup[2][1] = make_card(31, 3, 3, Color::Red);
    state.deck[2].push_back(make_card(32, 3, 4, Color::Black));
    state.bank.joker = 0;

    Move m;
    m.type = RESERVE_CARD;
    m.card_tier = 2;
    m.from_deck = true;

    applyMove(state, m);

    CHECK(state.players[0].reserved.size() == 1);
    CHECK(state.players[0].reserved[0].id == 32);
    CHECK(state.deck[2].empty());
    CHECK(state.faceup[2][1].id == 31);
}

static void test_buy_reserved_card_updates_state_and_transfers_payment() {
    GameState state;
    clear_state(state);
    state.current_player = 0;

    Tokens cost;
    cost.red = 2;
    Card reserved = make_card(40, 2, 2, Color::Green, cost);
    state.players[0].reserved.push_back(reserved);
    state.players[0].tokens.red = 2;
    state.bank.red = 4;

    Move m;
    m.type = BUY_CARD;
    m.from_reserved = true;
    m.card_slot = 0;

    applyMove(state, m);

    CHECK(state.players[0].reserved.empty());
    CHECK(state.players[0].cards.size() == 1);
    CHECK(state.players[0].cards[0].id == 40);
    CHECK(state.players[0].points == 2);
    CHECK(state.players[0].bonuses.green == 1);
    CHECK(state.players[0].tokens.red == 0);
    CHECK(state.bank.red == 6);
    CHECK(state.is_return_phase == false);
    CHECK(state.current_player == 1);
    CHECK(state.move_number == 1);
}

static void test_buy_faceup_uses_joker_for_shortfall() {
    GameState state;
    clear_state(state);
    state.current_player = 0;

    Tokens cost;
    cost.white = 1;
    cost.blue = 1;
    cost.black = 1;
    state.faceup[0][0] = make_card(50, 1, 1, Color::Red, cost);

    state.players[0].tokens.white = 1;
    state.players[0].tokens.black = 1;
    state.players[0].tokens.joker = 1;

    Move m;
    m.type = BUY_CARD;
    m.card_tier = 0;
    m.card_slot = 0;

    applyMove(state, m);

    CHECK(state.players[0].tokens.white == 0);
    CHECK(state.players[0].tokens.black == 0);
    CHECK(state.players[0].tokens.joker == 0);
    CHECK(state.bank.white == 1);
    CHECK(state.bank.black == 1);
    CHECK(state.bank.joker == 1);
    CHECK(state.players[0].bonuses.red == 1);
    CHECK(state.faceup[0][0].id == 0); // empty deck, slot cleared
}

static void test_pass_turn_flips_player_and_increments_move() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.is_return_phase = true;

    Move m;
    m.type = PASS_TURN;

    applyMove(state, m);

    CHECK(state.is_return_phase == false);
    CHECK(state.current_player == 1);
    CHECK(state.move_number == 1);
}

static void test_noble_awarded_when_requirements_met_after_buy() {
    GameState state;
    clear_state(state);
    state.current_player = 0;

    Noble noble;
    noble.id = 1;
    noble.points = 3;
    noble.requirements.blue = 1;
    state.available_nobles[0] = noble;
    state.noble_count = 1;

    state.faceup[0][0] = make_card(60, 1, 1, Color::Blue);

    Move m;
    m.type = BUY_CARD;
    m.card_tier = 0;
    m.card_slot = 0;
    applyMove(state, m);

    CHECK(state.players[0].nobles.size() == 1);
    CHECK(state.players[0].nobles[0].id == 1);
    CHECK(state.players[0].points == 4); // 1 card point + 3 noble points
    CHECK(state.noble_count == 0);
    CHECK(state.available_nobles[0].id == 0);
}

static void test_only_one_noble_awarded_when_multiple_match() {
    GameState state;
    clear_state(state);
    state.current_player = 0;

    Noble n1;
    n1.id = 11;
    n1.requirements.blue = 1;
    Noble n2;
    n2.id = 12;
    n2.requirements.blue = 1;
    state.available_nobles[0] = n1;
    state.available_nobles[1] = n2;
    state.noble_count = 2;

    state.faceup[0][0] = make_card(61, 1, 0, Color::Blue);

    Move m;
    m.type = BUY_CARD;
    m.card_tier = 0;
    m.card_slot = 0;
    applyMove(state, m);

    CHECK(state.players[0].nobles.size() == 1);
    CHECK(state.noble_count == 1);
    // TODO: Current implementation awards the first matching noble by available_nobles order.
    CHECK(state.players[0].nobles[0].id == 11);
    CHECK(state.available_nobles[0].id == 12);
    CHECK(state.players[0].points == 3);
}

static void test_no_noble_awarded_when_requirements_not_met() {
    GameState state;
    clear_state(state);
    state.current_player = 0;

    Noble noble;
    noble.id = 21;
    noble.requirements.green = 2;
    state.available_nobles[0] = noble;
    state.noble_count = 1;

    state.faceup[0][0] = make_card(62, 1, 1, Color::Blue);

    Move m;
    m.type = BUY_CARD;
    m.card_tier = 0;
    m.card_slot = 0;
    applyMove(state, m);

    CHECK(state.players[0].nobles.empty());
    CHECK(state.players[0].points == 1);
    CHECK(state.noble_count == 1);
}

static void test_find_all_valid_moves_only_pass_when_no_actions_exist() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.bank = Tokens{};
    auto moves = findAllValidMoves(state);
    CHECK(has_single_pass_only(moves));
}

static void test_find_all_valid_moves_gem_rules_by_bank_availability() {
    // Variant A: >=3 colors available and one color >=4 -> 3-different + 2-same (no 2-diff / single)
    {
        GameState state;
        clear_state(state);
        state.bank.white = 4;
        state.bank.blue = 1;
        state.bank.green = 1;
        auto moves = findAllValidMoves(state);

        Tokens take3;
        take3.white = take3.blue = take3.green = 1;
        Tokens take2same;
        take2same.white = 2;
        Tokens take2diff;
        take2diff.white = 1;
        take2diff.blue = 1;
        Tokens take1;
        take1.white = 1;

        CHECK(has_take_move(moves, take3));
        CHECK(has_take_move(moves, take2same));
        CHECK(!has_take_move(moves, take2diff));
        CHECK(!has_take_move(moves, take1));
    }

    // Variant B: exactly 2 colors available -> only 2-different combination (plus reserve/buy if present; here none)
    {
        GameState state;
        clear_state(state);
        state.bank.white = 1;
        state.bank.blue = 1;
        auto moves = findAllValidMoves(state);

        Tokens take2diff;
        take2diff.white = 1;
        take2diff.blue = 1;
        Tokens take3;
        take3.white = take3.blue = take3.green = 1;
        Tokens take1;
        take1.white = 1;

        CHECK(has_take_move(moves, take2diff));
        CHECK(!has_take_move(moves, take3));
        CHECK(!has_take_move(moves, take1));
    }

    // Variant C: exactly 1 color available -> only single take for that color
    {
        GameState state;
        clear_state(state);
        state.bank.red = 3; // <4 avoids 2-same action
        auto moves = findAllValidMoves(state);

        Tokens take1;
        take1.red = 1;
        Tokens take2same;
        take2same.red = 2;

        CHECK(has_take_move(moves, take1));
        CHECK(!has_take_move(moves, take2same));
        CHECK(count_moves_of_type(moves, TAKE_GEMS) == 1);
    }
}

static void test_is_game_over_during_return_phase_false_even_if_player_has_15() {
    GameState state;
    clear_state(state);
    state.is_return_phase = true;
    state.players[0].points = 15;
    CHECK(isGameOver(state) == false);
}

static void test_is_game_over_false_when_no_player_has_15() {
    GameState state;
    clear_state(state);
    state.players[0].points = 14;
    state.players[1].points = 14;
    CHECK(isGameOver(state) == false);
}

static void test_is_game_over_true_when_player1_has_15() {
    GameState state;
    clear_state(state);
    state.players[1].points = 15;
    CHECK(isGameOver(state) == true);
}

static void test_is_game_over_player0_waits_for_final_turn_semantics() {
    GameState state;
    clear_state(state);
    state.players[0].points = 15;

    state.current_player = 1;
    CHECK(isGameOver(state) == false);

    state.current_player = 0;
    CHECK(isGameOver(state) == true);
}

static void test_determine_winner_by_points() {
    GameState state;
    clear_state(state);
    state.players[0].points = 16;
    state.players[1].points = 12;
    CHECK(determineWinner(state) == 0);

    state.players[0].points = 10;
    state.players[1].points = 17;
    CHECK(determineWinner(state) == 1);
}

static void test_determine_winner_tiebreaker_fewer_cards() {
    GameState state;
    clear_state(state);
    state.players[0].points = 15;
    state.players[1].points = 15;
    state.players[0].cards.push_back(make_card(1, 1, 0, Color::White));
    state.players[1].cards.push_back(make_card(2, 1, 0, Color::White));
    state.players[1].cards.push_back(make_card(3, 1, 0, Color::White));
    CHECK(determineWinner(state) == 0);
}

static void test_determine_winner_draw_on_exact_tie() {
    GameState state;
    clear_state(state);
    state.players[0].points = 15;
    state.players[1].points = 15;
    state.players[0].cards.push_back(make_card(1, 1, 0, Color::White));
    state.players[1].cards.push_back(make_card(2, 1, 0, Color::White));
    CHECK(determineWinner(state) == -1);
}

static void test_valid_move_mask_matches_find_all_valid_moves() {
    GameState state;
    auto cards = loadCards("cards.json");
    auto nobles = loadNobles("nobles.json");
    initializeGame(state, cards, nobles, 123);

    auto moves = findAllValidMoves(state);
    auto mask = getValidMoveMask(state);

    bool seen[66];
    std::memset(seen, 0, sizeof(seen));
    int unique_count = 0;
    for (const auto& m : moves) {
        int idx = moveToActionIndex(m, state);
        CHECK(idx >= 0 && idx < 66);
        CHECK(mask[idx] == 1);
        if (!seen[idx]) {
            seen[idx] = true;
            unique_count++;
        }
    }
    CHECK(count_mask_bits(mask) == unique_count);
}

static void test_valid_move_sequence_preserves_token_invariants() {
    GameState state;
    auto cards = loadCards("cards.json");
    auto nobles = loadNobles("nobles.json");
    initializeGame(state, cards, nobles, 123);

    bool saw_nontrivial = false;
    for (int step = 0; step < 8 && !isGameOver(state); ++step) {
        auto moves = findAllValidMoves(state);
        CHECK(!moves.empty());
        Move chosen = choose_deterministic_valid_move(moves);
        if (chosen.type == RESERVE_CARD || chosen.type == TAKE_GEMS || chosen.type == RETURN_GEM) {
            saw_nontrivial = true;
        }
        applyMove(state, chosen);

        check_no_negative_tokens(state);
        check_token_conservation_2p(state);
        CHECK(state.players[0].reserved.size() <= 3);
        CHECK(state.players[1].reserved.size() <= 3);
        CHECK(state.current_player == 0 || state.current_player == 1);
        CHECK(state.noble_count >= 0 && state.noble_count <= 3);
    }

    CHECK(saw_nontrivial);
}

static void test_multiturn_reserve_return_then_buy_reserved_flow() {
    GameState state;
    clear_state(state);
    state.current_player = 0;
    state.move_number = 0;

    // Standard 2-player token totals split between player 0 and bank for conservation checks.
    state.players[0].tokens.white = 3;
    state.players[0].tokens.blue = 3;
    state.players[0].tokens.green = 4; // total 10 before reserve

    state.bank.white = 1;
    state.bank.blue = 1;
    state.bank.green = 0;
    state.bank.red = 4;
    state.bank.black = 4;
    state.bank.joker = 5;

    Tokens cost;
    cost.white = 1;
    cost.blue = 1;
    cost.black = 1; // will be covered by joker gained from reserve
    state.faceup[0][0] = make_card(70, 1, 1, Color::Red, cost);

    check_token_conservation_2p(state);
    check_no_negative_tokens(state);

    Move reserve;
    reserve.type = RESERVE_CARD;
    reserve.card_tier = 0;
    reserve.card_slot = 0;
    applyMove(state, reserve);

    CHECK(state.players[0].reserved.size() == 1);
    CHECK(state.players[0].reserved[0].id == 70);
    CHECK(state.players[0].tokens.joker == 1);
    CHECK(state.is_return_phase == true);
    CHECK(state.current_player == 0);
    CHECK(state.move_number == 0);
    check_token_conservation_2p(state);
    check_no_negative_tokens(state);

    Move ret;
    ret.type = RETURN_GEM;
    ret.gem_returned.green = 1;
    applyMove(state, ret);

    CHECK(state.players[0].tokens.total() == 10);
    CHECK(state.is_return_phase == false);
    CHECK(state.current_player == 1);
    CHECK(state.move_number == 1);
    check_token_conservation_2p(state);
    check_no_negative_tokens(state);

    Move pass;
    pass.type = PASS_TURN;
    applyMove(state, pass);

    CHECK(state.current_player == 0);
    CHECK(state.move_number == 2);
    CHECK(state.is_return_phase == false);
    check_token_conservation_2p(state);
    check_no_negative_tokens(state);

    Move buy_reserved;
    buy_reserved.type = BUY_CARD;
    buy_reserved.from_reserved = true;
    buy_reserved.card_slot = 0;
    applyMove(state, buy_reserved);

    CHECK(state.players[0].reserved.empty());
    CHECK(state.players[0].cards.size() == 1);
    CHECK(state.players[0].cards[0].id == 70);
    CHECK(state.players[0].bonuses.red == 1);
    CHECK(state.players[0].points == 1);
    CHECK(state.players[0].tokens.white == 2);
    CHECK(state.players[0].tokens.blue == 2);
    CHECK(state.players[0].tokens.joker == 0); // joker paid black shortfall
    CHECK(state.bank.white == 2);
    CHECK(state.bank.blue == 2);
    CHECK(state.bank.joker == 5);
    CHECK(state.current_player == 1);
    CHECK(state.move_number == 3);
    CHECK(state.is_return_phase == false);
    check_token_conservation_2p(state);
    check_no_negative_tokens(state);
}

static void test_invalid_move_cases_fail_in_subprocess() {
    // TODO: This locks in current unsafe behavior until applyMove gains validation.
    // Invalid moves may crash/abort; we assert non-zero exit in isolated subprocesses.
    int build_rc = std::system(
        "c++ -std=c++17 -O0 -g tests/test_applymove_invalid_cases.cpp game_logic.cpp "
        "-o tests/test_applymove_invalid_cases_bin");
    CHECK(build_rc == 0);

    int control_rc = std::system("./tests/test_applymove_invalid_cases_bin control_valid_pass");
    CHECK(control_rc == 0);

    int bad_reserved_rc = std::system("./tests/test_applymove_invalid_cases_bin buy_reserved_out_of_range");
    CHECK(bad_reserved_rc != 0);

    int bad_deck_rc = std::system("./tests/test_applymove_invalid_cases_bin reserve_from_empty_deck");
    CHECK(bad_deck_rc != 0);
}

int main() {
    test_simple_json_extractors();
    test_tokens_enum_indexing();
    test_tokens_arithmetic();
    test_file_loading_and_enum_conversion();
    test_invalid_card_color_throws();
    test_find_all_valid_moves_initial_state();
    test_return_phase_move_generation();
    test_action_roundtrip_for_gem_moves();
    test_buy_card_updates_bonus_via_enum_color();
    test_take_gems_enters_return_phase_when_exceeding_ten();
    test_take_gems_ends_turn_when_total_at_most_ten();
    test_return_gem_stays_in_return_phase_above_ten();
    test_return_gem_exits_return_phase_at_ten();
    test_reserve_faceup_refills_and_gives_joker();
    test_reserve_faceup_no_joker_when_bank_empty();
    test_reserve_from_deck_does_not_touch_faceup();
    test_buy_reserved_card_updates_state_and_transfers_payment();
    test_buy_faceup_uses_joker_for_shortfall();
    test_pass_turn_flips_player_and_increments_move();
    test_noble_awarded_when_requirements_met_after_buy();
    test_only_one_noble_awarded_when_multiple_match();
    test_no_noble_awarded_when_requirements_not_met();
    test_find_all_valid_moves_only_pass_when_no_actions_exist();
    test_find_all_valid_moves_gem_rules_by_bank_availability();
    // TODO: Current action space does not support returning jokers during return phase.
    // Tests intentionally lock current behavior until action-space/rules are expanded.
    test_is_game_over_during_return_phase_false_even_if_player_has_15();
    test_is_game_over_false_when_no_player_has_15();
    test_is_game_over_true_when_player1_has_15();
    test_is_game_over_player0_waits_for_final_turn_semantics();
    test_determine_winner_by_points();
    test_determine_winner_tiebreaker_fewer_cards();
    test_determine_winner_draw_on_exact_tie();
    test_valid_move_mask_matches_find_all_valid_moves();
    test_valid_move_sequence_preserves_token_invariants();
    test_multiturn_reserve_return_then_buy_reserved_flow();
    test_invalid_move_cases_fail_in_subprocess();

    std::cout << "All C++ unit tests passed." << std::endl;
    return 0;
}
