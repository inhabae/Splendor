"""Minimal neural-network training scaffold for Splendor."""

from .state_schema import ACTION_DIM, STATE_DIM

__all__ = ["ACTION_DIM", "STATE_DIM"]

try:
    from .native_env import SplendorNativeEnv, StepState
    from .model import MaskedPolicyValueNet
    from .replay import ReplayBuffer, ReplaySample

    __all__.extend(
        [
            "MaskedPolicyValueNet",
            "ReplayBuffer",
            "ReplaySample",
            "SplendorNativeEnv",
            "StepState",
        ]
    )
except Exception:
    # Allow importing schema/constants without full runtime deps for test discovery.
    pass
