"""Minimal neural-network training scaffold for Splendor."""

from .state_codec import ACTION_DIM, STATE_DIM, encode_state

__all__ = ["ACTION_DIM", "STATE_DIM", "encode_state"]

try:
    from .bridge_env import SplendorBridgeEnv, StepState
    from .model import MaskedPolicyValueNet
    from .replay import ReplayBuffer, ReplaySample

    __all__.extend(
        [
            "MaskedPolicyValueNet",
            "ReplayBuffer",
            "ReplaySample",
            "SplendorBridgeEnv",
            "StepState",
        ]
    )
except Exception:
    # Allow importing nn.state_codec without torch/numpy runtime deps for test discovery.
    pass
