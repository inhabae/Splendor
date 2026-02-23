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

### 3. Invalid `applyMove` robustness helper (optional/manual)

```bash
c++ -std=c++17 -O0 -g tests/test_applymove_invalid_cases.cpp game_logic.cpp -o tests/test_applymove_invalid_cases_bin
./tests/test_applymove_invalid_cases_bin control_valid_pass
```

Notes:

- `tests/test_bridge_protocol.py` will auto-build `tests/splendor_bridge_test_bin` if it does not exist.
- `tests/test_simple_json_and_game_logic.cpp` now compiles/runs `tests/test_applymove_invalid_cases_bin` internally for isolated invalid-move subprocess checks.
- Tests expect `cards.json` and `nobles.json` in the repository root.
