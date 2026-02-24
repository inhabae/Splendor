#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "tests" / "splendor_bridge_test_bin"
BINARY = Path(os.environ.get("SPLENDOR_BRIDGE_TEST_BIN", str(DEFAULT_BINARY)))
CARD_FEATURE_LEN = 11
OPP_RESERVED_BLOCK_START = 57
OPP_RESERVED_COUNT_IDX = 90  # After current player (45) + opponent tokens/bonuses/points (12) + hidden reserved zeros (33)
IS_RETURN_PHASE_STATE_IDX = 244
IS_NOBLE_CHOICE_PHASE_STATE_IDX = 245


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
    check(len(resp["state"]) == 246, f"state length must be 246, got {len(resp['state'])}")
    check(isinstance(resp.get("mask"), list), "mask must be a list")
    check(len(resp["mask"]) == 69, f"mask length must be 69, got {len(resp['mask'])}")
    check(isinstance(resp.get("is_return_phase"), bool), "is_return_phase must be bool")
    check(isinstance(resp.get("is_noble_choice_phase"), bool), "is_noble_choice_phase must be bool")
    check(isinstance(resp.get("is_terminal"), bool), "is_terminal must be bool")
    check(isinstance(resp.get("winner"), int), "winner must be int")
    check(isinstance(resp.get("current_player"), int), "current_player must be int")


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


def test_state_includes_noble_phase_flag_and_matches_top_level_phase():
    c = BridgeClient()
    try:
        resp = c.send({"cmd": "init", "seed": 123})
        assert_ok_shape(resp)
        check(resp["state"][IS_RETURN_PHASE_STATE_IDX] == 0, "Expected state is_return_phase bit = 0 at init")
        check(resp["state"][IS_NOBLE_CHOICE_PHASE_STATE_IDX] == 0, "Expected state is_noble_choice_phase bit = 0 at init")
        check(resp["is_return_phase"] is False, "Expected top-level is_return_phase = false at init")
        check(resp["is_noble_choice_phase"] is False, "Expected top-level is_noble_choice_phase = false at init")
    finally:
        c.close()


def test_invalid_action_indices():
    c = BridgeClient()
    try:
        c.send({"cmd": "init", "seed": 123})
        resp_hi = c.send({"cmd": "apply", "action": 69})
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


def test_snapshot_restore_roundtrip_restores_state_and_mask():
    c = BridgeClient()
    try:
        init_resp = c.send({"cmd": "init", "seed": 123})
        assert_ok_shape(init_resp)

        snap_resp = c.send({"cmd": "snapshot"})
        check(snap_resp.get("status") == "ok", f"Expected ok snapshot response, got {snap_resp}")
        snapshot_id = snap_resp.get("snapshot_id")
        check(isinstance(snapshot_id, int), f"Expected int snapshot_id, got {snap_resp}")

        valid_idx = next((i for i, v in enumerate(init_resp["mask"]) if v == 1), None)
        check(valid_idx is not None, "Expected at least one valid action")
        after_apply = c.send({"cmd": "apply", "action": valid_idx})
        assert_ok_shape(after_apply)

        restored = c.send({"cmd": "restore_snapshot", "snapshot_id": snapshot_id})
        assert_ok_shape(restored)
        check(restored["state"] == init_resp["state"], "restore_snapshot did not restore state exactly")
        check(restored["mask"] == init_resp["mask"], "restore_snapshot did not restore mask exactly")
        check(restored["winner"] == init_resp["winner"], "restore_snapshot winner mismatch")
        check(restored["current_player"] == init_resp["current_player"], "restore_snapshot current_player mismatch")
    finally:
        c.close()


def test_restore_snapshot_invalid_id_rejected():
    c = BridgeClient()
    try:
        c.send({"cmd": "init", "seed": 123})
        resp = c.send({"cmd": "restore_snapshot", "snapshot_id": 999999})
        check(resp.get("status") == "error", f"Expected error, got {resp}")
    finally:
        c.close()


def test_drop_snapshot_then_restore_fails():
    c = BridgeClient()
    try:
        c.send({"cmd": "init", "seed": 123})
        snap_resp = c.send({"cmd": "snapshot"})
        snapshot_id = snap_resp.get("snapshot_id")
        check(isinstance(snapshot_id, int), f"Expected int snapshot_id, got {snap_resp}")
        drop_resp = c.send({"cmd": "drop_snapshot", "snapshot_id": snapshot_id})
        check(drop_resp.get("status") == "ok", f"Expected ok, got {drop_resp}")
        restore_resp = c.send({"cmd": "restore_snapshot", "snapshot_id": snapshot_id})
        check(restore_resp.get("status") == "error", f"Expected error, got {restore_resp}")
    finally:
        c.close()


def test_reset_clears_snapshot_table():
    c = BridgeClient()
    try:
        c.send({"cmd": "init", "seed": 123})
        snap_resp = c.send({"cmd": "snapshot"})
        snapshot_id = snap_resp.get("snapshot_id")
        check(isinstance(snapshot_id, int), f"Expected int snapshot_id, got {snap_resp}")
        reset_resp = c.send({"cmd": "reset", "seed": 123})
        assert_ok_shape(reset_resp)
        restore_resp = c.send({"cmd": "restore_snapshot", "snapshot_id": snapshot_id})
        check(restore_resp.get("status") == "error", f"Expected error, got {restore_resp}")
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


def test_opponent_reserved_count_feature_present_and_updates():
    c = BridgeClient()
    try:
        init_resp = c.send({"cmd": "init", "seed": 123})
        assert_ok_shape(init_resp)
        check(
            init_resp["state"][OPP_RESERVED_COUNT_IDX] == 0,
            f"Expected initial opponent_reserved_count=0, got {init_resp['state'][OPP_RESERVED_COUNT_IDX]}",
        )

        # Make any valid move to hand turn to the opponent.
        first_valid = next((i for i, v in enumerate(init_resp["mask"]) if v == 1), None)
        check(first_valid is not None, "Expected at least one valid action at init")
        after_first = c.send({"cmd": "apply", "action": first_valid})
        assert_ok_shape(after_first)

        # On opponent's turn, choose a reserve action (15..29) so the next response should show opponent count = 1.
        reserve_idx = next((i for i, v in enumerate(after_first["mask"]) if v == 1 and 15 <= i <= 29), None)
        check(reserve_idx is not None, "Expected at least one valid reserve action for opponent")
        after_opponent_reserve = c.send({"cmd": "apply", "action": reserve_idx})
        assert_ok_shape(after_opponent_reserve)
        observed_count = after_opponent_reserve["state"][OPP_RESERVED_COUNT_IDX]
        check(
            observed_count == 1,
            f"Expected opponent_reserved_count=1 after opponent reserve, got {observed_count}",
        )
    finally:
        c.close()


def test_opponent_faceup_reserved_card_is_visible_in_reserved_block():
    c = BridgeClient()
    try:
        init_resp = c.send({"cmd": "init", "seed": 123})
        assert_ok_shape(init_resp)

        first_valid = next((i for i, v in enumerate(init_resp["mask"]) if v == 1), None)
        check(first_valid is not None, "Expected at least one valid action at init")
        after_first = c.send({"cmd": "apply", "action": first_valid})
        assert_ok_shape(after_first)

        faceup_reserve_idx = next((i for i, v in enumerate(after_first["mask"]) if v == 1 and 15 <= i <= 26), None)
        check(faceup_reserve_idx is not None, "Expected at least one valid face-up reserve action")
        after_opp_reserve = c.send({"cmd": "apply", "action": faceup_reserve_idx})
        assert_ok_shape(after_opp_reserve)

        check(after_opp_reserve["state"][OPP_RESERVED_COUNT_IDX] == 1, "Expected opponent reserved count = 1")
        first_reserved_slot = after_opp_reserve["state"][OPP_RESERVED_BLOCK_START:OPP_RESERVED_BLOCK_START + CARD_FEATURE_LEN]
        check(any(v != 0 for v in first_reserved_slot), f"Expected visible opponent reserved card features, got {first_reserved_slot}")
    finally:
        c.close()


def test_opponent_deck_reserved_card_remains_hidden_in_reserved_block():
    c = BridgeClient()
    try:
        init_resp = c.send({"cmd": "init", "seed": 123})
        assert_ok_shape(init_resp)

        first_valid = next((i for i, v in enumerate(init_resp["mask"]) if v == 1), None)
        check(first_valid is not None, "Expected at least one valid action at init")
        after_first = c.send({"cmd": "apply", "action": first_valid})
        assert_ok_shape(after_first)

        deck_reserve_idx = next((i for i, v in enumerate(after_first["mask"]) if v == 1 and 27 <= i <= 29), None)
        check(deck_reserve_idx is not None, "Expected at least one valid deck reserve action")
        after_opp_reserve = c.send({"cmd": "apply", "action": deck_reserve_idx})
        assert_ok_shape(after_opp_reserve)

        check(after_opp_reserve["state"][OPP_RESERVED_COUNT_IDX] == 1, "Expected opponent reserved count = 1")
        first_reserved_slot = after_opp_reserve["state"][OPP_RESERVED_BLOCK_START:OPP_RESERVED_BLOCK_START + CARD_FEATURE_LEN]
        check(all(v == 0 for v in first_reserved_slot), f"Expected hidden deck-reserved card to be zero-masked, got {first_reserved_slot}")
    finally:
        c.close()


def main():
    compile_bridge_if_needed()

    tests = [
        test_get_state_before_init,
        test_apply_before_init,
        test_init_success_shape,
        test_state_includes_noble_phase_flag_and_matches_top_level_phase,
        test_invalid_action_indices,
        test_invalid_but_in_range_action_rejected,
        test_valid_action_apply_roundtrip,
        test_unknown_command_valid_json,
        test_snapshot_restore_roundtrip_restores_state_and_mask,
        test_restore_snapshot_invalid_id_rejected,
        test_drop_snapshot_then_restore_fails,
        test_reset_clears_snapshot_table,
        test_serialization_regression_reserved_slots_stable_state_length,
        test_opponent_reserved_count_feature_present_and_updates,
        test_opponent_faceup_reserved_card_is_visible_in_reserved_block,
        test_opponent_deck_reserved_card_remains_hidden_in_reserved_block,
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
