# Tests

Regression tests for the native Splendor engine and Python training stack.

## Coverage areas

- core game logic (`game_logic.cpp`)
- invalid `applyMove` robustness helper binary
- pybind11 native environment (`splendor_native`)
- Python NN/MCTS/training utilities

## Run From Repo Root

### 0. Install Python dependencies for NN training/tests

```bash
.venv/bin/pip install -r requirements-nn.txt
```

### 1. Build the native module and C++ test binaries (CMake)

```bash
cmake -S . -B build
cmake --build build --target splendor_native test_game_logic test_applymove_invalid_cases_bin
```

### 2. C++ game-logic unit tests

```bash
./build/test_game_logic
```

### 3. Invalid `applyMove` robustness helper (optional/manual)

```bash
./build/test_applymove_invalid_cases_bin control_valid_pass
```

### 4. Native pybind11 environment tests

```bash
.venv/bin/python -m unittest -v tests.test_nn_bridge_env_native
```

### 5. Python smoke / training tests

```bash
.venv/bin/python -m unittest -v tests.test_nn_smoke
```

## Notes

- The runtime path is native-only (`splendor_native`); the legacy subprocess bridge has been decommissioned.
- `nn.state_codec` remains the Python reference spec for state layout and normalization, and must match the C++ encoder in `py_splendor.cpp`.
- `SplendorBridgeEnv` is now a native backend compatibility wrapper (name retained to avoid large call-site churn).

## Sanitizer Example (C++ core)

```bash
c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/test_game_logic_core.cpp game_logic.cpp -o tests/test_game_logic_san
ASAN_OPTIONS=detect_leaks=1 ./tests/test_game_logic_san
```

On some macOS toolchains, `ASAN_OPTIONS=detect_leaks=1` is unsupported; if so, run without it.
