# Tests

Lightweight regression tests for:

- centralized JSON parsing helpers (`simple_json_parse.h`)
- enum-based color handling (`Color`, `Tokens::operator[](Color)`)
- bridge protocol behavior (`splendor_bridge`)

## Run from repo root

### 1. C++ unit tests

```bash
c++ -std=c++17 -O0 -g tests/test_simple_json_and_game_logic.cpp game_logic.cpp -o tests/test_game_logic
./tests/test_game_logic
```

### 2. Bridge protocol tests

```bash
c++ -std=c++17 -O0 -g splendor_bridge.cpp game_logic.cpp -o tests/splendor_bridge_test_bin
python3 tests/test_bridge_protocol.py
```

You can override the bridge binary path (useful for sanitizer builds):

```bash
SPLENDOR_BRIDGE_TEST_BIN=tests/splendor_bridge_test_bin python3 tests/test_bridge_protocol.py
```

### 3. Invalid `applyMove` robustness helper (optional/manual)

```bash
c++ -std=c++17 -O0 -g tests/test_applymove_invalid_cases.cpp game_logic.cpp -o tests/test_applymove_invalid_cases_bin
./tests/test_applymove_invalid_cases_bin control_valid_pass
```

Notes:

- `tests/test_bridge_protocol.py` will auto-build `tests/splendor_bridge_test_bin` if it does not exist.
- `tests/test_simple_json_and_game_logic.cpp` now compiles/runs `tests/test_applymove_invalid_cases_bin` internally for isolated invalid-move subprocess checks.
- Tests expect `cards.json` and `nobles.json` in the repository root.
- Bridge `ok` responses now encode `state` length `246` (was `244` originally), including appended `opponent_reserved_count` and `is_noble_choice_phase`.
- Downstream models/parsers expecting length `244` must update input dimensions and index constants.

## Sanitizer Runs (ASan/UBSan)

### C++ unit tests under sanitizers

```bash
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/test_simple_json_and_game_logic.cpp game_logic.cpp -o tests/test_game_logic_san
ASAN_OPTIONS=detect_leaks=1 ./tests/test_game_logic_san
```

### Bridge protocol tests with sanitized bridge binary

```bash
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  splendor_bridge.cpp game_logic.cpp -o tests/splendor_bridge_test_bin_san
SPLENDOR_BRIDGE_TEST_BIN=tests/splendor_bridge_test_bin_san python3 tests/test_bridge_protocol.py
```

Notes:

- Invalid `applyMove()` cases are now expected to throw exceptions (not crash) in the helper test binary.
- Run sanitizer builds against normal unit/protocol tests; sanitizer output is most useful for valid execution paths.
- On some macOS toolchains, `ASAN_OPTIONS=detect_leaks=1` is unsupported; if so, run `./tests/test_game_logic_san` without that env var.

## Self-Play Perspective Notes

- Bridge `state` is always encoded from the **current-player (side-to-move)** perspective.
- Bridge `mask` is the valid action mask for that same perspective and action space (69 actions).
- `winner` in bridge `ok` responses is an **absolute seat index** (`0` or `1`), `-1` for draw, `-2` for non-terminal.
- For self-play training, store `to_play_abs` (`0`/`1`) with each sample and convert value labels at terminal:
  - draw -> `z = 0`
  - win for `to_play_abs` -> `z = +1`
  - loss for `to_play_abs` -> `z = -1`
- Do not manually re-swap/flip bridge states between moves; `apply` responses are already side-to-move canonical.

## Opponent Reserved Visibility (State Semantics)

- `state` length is `246`.
- Opponent reserved block (`3 x 11` features) now encodes:
  - face-up-board-reserved cards as visible card features (public information)
  - deck-reserved cards as zeros (hidden)
  - empty reserved slots as zeros
- Appended `opponent_reserved_count` remains a public feature, and the state now also includes explicit phase bits for `is_return_phase` and `is_noble_choice_phase` at the tail of the 246-length vector.
