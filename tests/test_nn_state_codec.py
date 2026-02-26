#!/usr/bin/env python3
import unittest

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

if np is not None:
    from nn.state_schema import (
        BANK_START,
        CP_BONUSES_START,
        CP_POINTS_IDX,
        CP_RESERVED_START,
        FACEUP_START,
        NOBLES_START,
        OP_BONUSES_START,
        OP_POINTS_IDX,
        OP_RESERVED_COUNT_IDX,
        OP_RESERVED_START,
        OP_TOKENS_START,
        PHASE_FLAGS_START,
        STATE_DIM,
    )
    from tests.codec_reference import encode_state
else:
    STATE_DIM = 246
    encode_state = None


@unittest.skipIf(np is None, "numpy not installed")
class TestNNStateCodec(unittest.TestCase):
    def test_encode_state_shape_dtype_finite(self):
        raw = [0] * STATE_DIM
        out = encode_state(raw)
        self.assertEqual(out.shape, (STATE_DIM,))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())

    def test_encode_state_length_validation(self):
        with self.assertRaises(ValueError):
            encode_state([0] * (STATE_DIM - 1))
        with self.assertRaises(ValueError):
            encode_state([0] * (STATE_DIM + 1))

    def test_token_normalization_exactness(self):
        raw = [0] * STATE_DIM
        # Current player tokens
        raw[0:6] = [4, 2, 0, 1, 3, 5]
        # Opponent tokens
        raw[OP_TOKENS_START : OP_TOKENS_START + 6] = [1, 4, 2, 0, 3, 5]
        # Bank tokens
        raw[BANK_START : BANK_START + 6] = [4, 3, 2, 1, 0, 5]
        out = encode_state(raw)

        self.assertTrue(np.allclose(out[0:5], np.array([1.0, 0.5, 0.0, 0.25, 0.75], dtype=np.float32)))
        self.assertAlmostEqual(float(out[5]), 1.0)
        self.assertTrue(np.allclose(out[OP_TOKENS_START : OP_TOKENS_START + 5], np.array([0.25, 1.0, 0.5, 0.0, 0.75], dtype=np.float32)))
        self.assertAlmostEqual(float(out[OP_TOKENS_START + 5]), 1.0)
        self.assertTrue(np.allclose(out[BANK_START : BANK_START + 5], np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32)))
        self.assertAlmostEqual(float(out[BANK_START + 5]), 1.0)

    def test_bonus_and_point_normalization_exactness(self):
        raw = [0] * STATE_DIM
        raw[CP_BONUSES_START : CP_BONUSES_START + 5] = [7, 0, 1, 3, 6]
        raw[CP_POINTS_IDX] = 20
        raw[OP_BONUSES_START : OP_BONUSES_START + 5] = [2, 4, 7, 1, 0]
        raw[OP_POINTS_IDX] = 10
        out = encode_state(raw)

        self.assertTrue(np.allclose(out[CP_BONUSES_START : CP_BONUSES_START + 5], np.array([1.0, 0.0, 1.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0], dtype=np.float32)))
        self.assertAlmostEqual(float(out[CP_POINTS_IDX]), 1.0)
        self.assertTrue(np.allclose(out[OP_BONUSES_START : OP_BONUSES_START + 5], np.array([2.0 / 7.0, 4.0 / 7.0, 1.0, 1.0 / 7.0, 0.0], dtype=np.float32)))
        self.assertAlmostEqual(float(out[OP_POINTS_IDX]), 0.5)

    def test_card_block_normalization_exactness_across_sections(self):
        raw = [0] * STATE_DIM
        starts = [CP_RESERVED_START, OP_RESERVED_START, FACEUP_START]
        for start in starts:
            # costs
            raw[start : start + 5] = [7, 0, 3, 1, 6]
            # bonus one-hot
            raw[start + 5 : start + 10] = [0, 1, 0, 0, 0]
            # points
            raw[start + 10] = 5
        out = encode_state(raw)

        for start in starts:
            self.assertTrue(np.allclose(out[start : start + 5], np.array([1.0, 0.0, 3.0 / 7.0, 1.0 / 7.0, 6.0 / 7.0], dtype=np.float32)))
            self.assertTrue(np.allclose(out[start + 5 : start + 10], np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)))
            self.assertAlmostEqual(float(out[start + 10]), 1.0)

    def test_reserved_count_nobles_and_phase_flags(self):
        raw = [0] * STATE_DIM
        raw[OP_RESERVED_COUNT_IDX] = 3
        raw[NOBLES_START:PHASE_FLAGS_START] = [4] * (PHASE_FLAGS_START - NOBLES_START)
        raw[PHASE_FLAGS_START : PHASE_FLAGS_START + 2] = [1, 0]
        out = encode_state(raw)

        self.assertAlmostEqual(float(out[OP_RESERVED_COUNT_IDX]), 1.0)
        self.assertTrue(np.allclose(out[NOBLES_START:PHASE_FLAGS_START], np.ones((PHASE_FLAGS_START - NOBLES_START,), dtype=np.float32)))
        self.assertTrue(np.allclose(out[PHASE_FLAGS_START : PHASE_FLAGS_START + 2], np.array([1.0, 0.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
