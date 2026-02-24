from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .state_codec import ACTION_DIM, STATE_DIM, encode_state


@dataclass
class StepState:
    state: np.ndarray  # normalized float32, shape (246,)
    mask: np.ndarray  # bool, shape (69,)
    is_terminal: bool
    winner: int
    is_return_phase: bool
    is_noble_choice_phase: bool
    current_player_id: int = 0


def _default_bridge_binary(repo_root: Path) -> Path:
    env_bin = os.environ.get("SPLENDOR_BRIDGE_BIN") or os.environ.get("SPLENDOR_BRIDGE_TEST_BIN")
    if env_bin:
        return Path(env_bin)
    return repo_root / "tests" / "splendor_bridge_test_bin"


class SplendorBridgeEnv:
    """Subprocess JSON wrapper around splendor_bridge.cpp protocol."""

    def __init__(
        self,
        *,
        repo_root: Optional[Path] = None,
        binary: Optional[Path] = None,
        cards_path: Optional[Path] = None,
        nobles_path: Optional[Path] = None,
    ) -> None:
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
        self.binary = Path(binary) if binary is not None else _default_bridge_binary(self.repo_root)
        self.cards_path = Path(cards_path) if cards_path is not None else self.repo_root / "cards.json"
        self.nobles_path = Path(nobles_path) if nobles_path is not None else self.repo_root / "nobles.json"

        if not self.binary.exists():
            raise FileNotFoundError(
                f"Bridge binary not found at {self.binary}. "
                "Build it first or set SPLENDOR_BRIDGE_BIN / SPLENDOR_BRIDGE_TEST_BIN."
            )

        self.proc = subprocess.Popen(
            [str(self.binary), str(self.cards_path), str(self.nobles_path)],
            cwd=str(self.repo_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._closed = False
        self._initialized = False
        self._current_player_id = 0  # source of truth is bridge responses
        self._snapshot_ids: set[int] = set()

    @property
    def current_player_id(self) -> int:
        return self._current_player_id

    def _send(self, payload: dict) -> dict:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Bridge subprocess pipes are unavailable")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if line == "":
            stderr_text = ""
            if self.proc.stderr is not None:
                try:
                    stderr_text = self.proc.stderr.read()
                except Exception:
                    stderr_text = ""
            raise RuntimeError(f"No response from bridge for payload {payload!r}. stderr={stderr_text!r}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Bridge returned invalid JSON: {line!r}") from exc

    def _parse_ok_response(self, resp: dict) -> StepState:
        if resp.get("status") != "ok":
            raise RuntimeError(f"Bridge error response: {resp}")
        raw_state = resp.get("state")
        raw_mask = resp.get("mask")
        if not isinstance(raw_state, list) or len(raw_state) != STATE_DIM:
            raise RuntimeError(f"Expected state length {STATE_DIM}, got {type(raw_state)} len={len(raw_state) if isinstance(raw_state, list) else 'n/a'}")
        if not isinstance(raw_mask, list) or len(raw_mask) != ACTION_DIM:
            raise RuntimeError(f"Expected mask length {ACTION_DIM}, got {type(raw_mask)} len={len(raw_mask) if isinstance(raw_mask, list) else 'n/a'}")
        mask = np.asarray(raw_mask, dtype=np.bool_)
        state = encode_state(raw_state)
        if state.shape != (STATE_DIM,) or mask.shape != (ACTION_DIM,):
            raise RuntimeError("Unexpected normalized state/mask shapes")
        return StepState(
            state=state,
            mask=mask,
            is_terminal=bool(resp.get("is_terminal", False)),
            winner=int(resp.get("winner", -2)),
            is_return_phase=bool(resp.get("is_return_phase", False)),
            is_noble_choice_phase=bool(resp.get("is_noble_choice_phase", False)),
            current_player_id=int(resp.get("current_player", self._current_player_id)),
        )

    def reset(self, seed: int = 0) -> StepState:
        resp = self._send({"cmd": "reset", "seed": int(seed)})
        state = self._parse_ok_response(resp)
        self._current_player_id = state.current_player_id
        self._snapshot_ids.clear()
        self._initialized = True
        return state

    def get_state(self) -> StepState:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        resp = self._send({"cmd": "get_state"})
        return self._parse_ok_response(resp)

    def step(self, action_idx: int) -> StepState:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        action_idx = int(action_idx)
        if not (0 <= action_idx < ACTION_DIM):
            raise ValueError(f"Action index out of range: {action_idx}")
        resp = self._send({"cmd": "apply", "action": action_idx})
        next_state = self._parse_ok_response(resp)
        self._current_player_id = next_state.current_player_id
        return next_state

    def snapshot(self) -> int:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        resp = self._send({"cmd": "snapshot"})
        if resp.get("status") != "ok":
            raise RuntimeError(f"Bridge error response: {resp}")
        snapshot_id = int(resp.get("snapshot_id", -1))
        if snapshot_id < 0:
            raise RuntimeError(f"Invalid snapshot response: {resp}")
        self._snapshot_ids.add(snapshot_id)
        return snapshot_id

    def restore_snapshot(self, snapshot_id: int) -> StepState:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        resp = self._send({"cmd": "restore_snapshot", "snapshot_id": int(snapshot_id)})
        state = self._parse_ok_response(resp)
        self._current_player_id = state.current_player_id
        return state

    def drop_snapshot(self, snapshot_id: int) -> None:
        if not self._initialized:
            raise RuntimeError("Game not initialized; call reset() first")
        resp = self._send({"cmd": "drop_snapshot", "snapshot_id": int(snapshot_id)})
        if resp.get("status") != "ok":
            raise RuntimeError(f"Bridge error response: {resp}")
        self._snapshot_ids.discard(int(snapshot_id))

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.proc.poll() is None:
                for snapshot_id in list(self._snapshot_ids):
                    try:
                        self._send({"cmd": "drop_snapshot", "snapshot_id": int(snapshot_id)})
                    except Exception:
                        break
                    finally:
                        self._snapshot_ids.discard(snapshot_id)
                try:
                    self._send({"cmd": "quit"})
                except Exception:
                    pass
        finally:
            self._closed = True
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def __enter__(self) -> "SplendorBridgeEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
