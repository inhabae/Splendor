#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BINARY = REPO_ROOT / "tests" / "splendor_bridge_test_bin"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def compile_bridge_if_needed():
    if BINARY.exists():
        return
    cmd = [
        "c++",
        "-std=c++17",
        "-O0",
        "-g",
        "splendor_bridge.cpp",
        "game_logic.cpp",
        "-o",
        str(BINARY),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


class BridgeClient:
    def __init__(self):
        self.proc = subprocess.Popen(
            [str(BINARY), "cards.json", "nobles.json"],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def send(self, payload):
        line = json.dumps(payload)
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline()
        check(out != "", f"No response for payload: {payload}")
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Invalid JSON response: {out!r}") from e

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send({"cmd": "quit"})
            except Exception:
                pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc.wait(timeout=2)


def assert_ok_shape(resp):
    check(resp.get("status") == "ok", f"Expected ok response, got {resp}")
    check(isinstance(resp.get("state"), list), "state must be a list")
    check(len(resp["state"]) == 244, f"state length must be 244, got {len(resp['state'])}")
    check(isinstance(resp.get("mask"), list), "mask must be a list")
    check(len(resp["mask"]) == 66, f"mask length must be 66, got {len(resp['mask'])}")
    check(isinstance(resp.get("is_return_phase"), bool), "is_return_phase must be bool")
    check(isinstance(resp.get("is_terminal"), bool), "is_terminal must be bool")
    check(isinstance(resp.get("winner"), int), "winner must be int")


def test_get_state_before_init():
    c = BridgeClient()
    try:
        resp = c.send({"cmd": "get_state"})
        check(resp.get("status") == "error", f"Expected error, got {resp}")
        check("initialized" in resp.get("message", "").lower(), f"Unexpected message: {resp}")
    finally:
        c.close()


def test_apply_before_init():
    c = BridgeClient()
    try:
        resp = c.send({"cmd": "apply", "action": 0})
        check(resp.get("status") == "error", f"Expected error, got {resp}")
    finally:
        c.close()


def test_init_success_shape():
    c = BridgeClient()
    try:
        resp = c.send({"cmd": "init", "seed": 123})
        assert_ok_shape(resp)
    finally:
        c.close()


def test_invalid_action_indices():
    c = BridgeClient()
    try:
        c.send({"cmd": "init", "seed": 123})
        resp_hi = c.send({"cmd": "apply", "action": 66})
        resp_lo = c.send({"cmd": "apply", "action": -1})
        check(resp_hi.get("status") == "error", f"Expected error, got {resp_hi}")
        check(resp_lo.get("status") == "error", f"Expected error, got {resp_lo}")
    finally:
        c.close()


def test_invalid_but_in_range_action_rejected():
    c = BridgeClient()
    try:
        init_resp = c.send({"cmd": "init", "seed": 123})
        mask = init_resp["mask"]
        invalid_idx = next((i for i, v in enumerate(mask) if v == 0), None)
        check(invalid_idx is not None, "Expected at least one invalid action in mask")
        resp = c.send({"cmd": "apply", "action": invalid_idx})
        check(resp.get("status") == "error", f"Expected error, got {resp}")
        check("not valid" in resp.get("message", "").lower(), f"Unexpected message: {resp}")
    finally:
        c.close()


def test_valid_action_apply_roundtrip():
    c = BridgeClient()
    try:
        init_resp = c.send({"cmd": "init", "seed": 123})
        valid_idx = next((i for i, v in enumerate(init_resp["mask"]) if v == 1), None)
        check(valid_idx is not None, "Expected at least one valid action")
        resp = c.send({"cmd": "apply", "action": valid_idx})
        assert_ok_shape(resp)
    finally:
        c.close()


def test_unknown_command_valid_json():
    c = BridgeClient()
    try:
        resp = c.send({"cmd": "nope"})
        check(resp.get("status") == "error", f"Expected error, got {resp}")
    finally:
        c.close()


def test_serialization_regression_reserved_slots_stable_state_length():
    c = BridgeClient()
    try:
        init_resp = c.send({"cmd": "init", "seed": 1})
        reserve_idx = next((i for i, v in enumerate(init_resp["mask"]) if v == 1 and 15 <= i <= 29), None)
        check(reserve_idx is not None, "Expected at least one valid reserve action")

        apply_resp = c.send({"cmd": "apply", "action": reserve_idx})
        assert_ok_shape(apply_resp)

        state_resp = c.send({"cmd": "get_state"})
        assert_ok_shape(state_resp)
    finally:
        c.close()


def main():
    compile_bridge_if_needed()

    tests = [
        test_get_state_before_init,
        test_apply_before_init,
        test_init_success_shape,
        test_invalid_action_indices,
        test_invalid_but_in_range_action_rejected,
        test_valid_action_apply_roundtrip,
        test_unknown_command_valid_json,
        test_serialization_regression_reserved_slots_stable_state_length,
    ]

    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print("All bridge protocol tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
