from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .state_schema import ACTION_DIM, STATE_DIM

_NATIVE_MODULE: Any | None = None
_NATIVE_MODULE_LOAD_ATTEMPTED = False


@dataclass
class StepState:
    state: np.ndarray  # normalized float32, shape (246,)
    mask: np.ndarray  # bool, shape (69,)
    is_terminal: bool
    winner: int
    current_player_id: int = 0


def _try_load_native_module() -> Any | None:
    global _NATIVE_MODULE, _NATIVE_MODULE_LOAD_ATTEMPTED
    if _NATIVE_MODULE_LOAD_ATTEMPTED:
        return _NATIVE_MODULE
    _NATIVE_MODULE_LOAD_ATTEMPTED = True

    try:
        _NATIVE_MODULE = importlib.import_module("splendor_native")
        return _NATIVE_MODULE
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "build"
    if not build_dir.exists():
        return None

    # Try to load the compiled native module from the build directory.
    # Use recursive search so multi-config generators (e.g. Visual Studio on Windows)
    # can load from build/Release or build/Debug without extra copying.
    patterns = ("splendor_native*.so", "splendor_native*.pyd", "splendor_native*.dylib")
    for pattern in patterns:
        for candidate in sorted(build_dir.rglob(pattern)):
            spec = importlib.util.spec_from_file_location("splendor_native", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["splendor_native"] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop("splendor_native", None)
                continue
            _NATIVE_MODULE = module
            return _NATIVE_MODULE

    return None


class SplendorNativeEnv:
    """Python adapter over the native pybind11 Splendor environment."""

    def __init__(self) -> None:
        native = _try_load_native_module()
        if native is None:
            raise ImportError(
                "splendor_native module not available. Build it (e.g. `cmake --build build --target splendor_native`)."
            )
        self._native = native
        env_cls = getattr(native, "NativeEnv", None)
        if env_cls is None:
            raise RuntimeError("splendor_native module missing NativeEnv class")
        self._env = env_cls()
        self._closed = False
        self._initialized = False
        self._current_player_id = 0

    @property
    def current_player_id(self) -> int:
        return self._current_player_id

    # Convert the raw result from c++ into a clean Python StepState dataclass.
    def _to_step_state(self, result: Any) -> StepState:
        state = np.asarray(result.state, dtype=np.float32)
        mask = np.asarray(result.mask, dtype=np.bool_)
        if state.shape != (STATE_DIM,):
            raise RuntimeError(f"Unexpected native state shape {state.shape}")
        if mask.shape != (ACTION_DIM,):
            raise RuntimeError(f"Unexpected native mask shape {mask.shape}")
        step = StepState(
            state=state,
            mask=mask,
            is_terminal=bool(result.is_terminal),
            winner=int(result.winner),
            current_player_id=int(result.current_player_id),
        )
        self._current_player_id = step.current_player_id
        return step

    def reset(self, seed: int = 0) -> StepState:
        result = self._env.reset(int(seed))
        self._initialized = True
        return self._to_step_state(result)

    def get_state(self) -> StepState:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return self._to_step_state(self._env.get_state())

    def step(self, action_idx: int) -> StepState:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return self._to_step_state(self._env.step(int(action_idx)))

    def snapshot(self) -> int:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return int(self._env.snapshot())

    def restore_snapshot(self, snapshot_id: int) -> StepState:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return self._to_step_state(self._env.restore_snapshot(int(snapshot_id)))

    def drop_snapshot(self, snapshot_id: int) -> None:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        self._env.drop_snapshot(int(snapshot_id))

    def debug_raw_state(self) -> np.ndarray:
        """Test/debug helper exposing the native pre-normalized state vector."""
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        raw = np.asarray(self._env.debug_raw_state())
        if raw.shape != (STATE_DIM,):
            raise RuntimeError(f"Unexpected native raw state shape {raw.shape}")
        return raw

    def run_mcts_native(self, evaluator, **kwargs):
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        return self._env.run_mcts(evaluator, **kwargs)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "SplendorNativeEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

__all__ = [
    "StepState",
    "SplendorNativeEnv",
]
