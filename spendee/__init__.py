"""Browser automation bridge for playing Spendee with the local Splendor engine."""

from .catalog import SpendeeCatalog
from .determinize import sample_hydrated_states
from .engine_policy import DeterminizedMCTSPolicy, DeterminizedPolicyResult
from .executor import ActionExecutionPlan, SpendeeExecutor, plan_action
from .observer import (
    ObservedBoardState,
    ObservedCard,
    ObservedModalState,
    ObservedNoble,
    ObservedPlayerState,
    ObservedReservedSlot,
    SpendeeObserver,
    normalize_probe_payload,
)
from .runner import SpendeeBridgeConfig, SpendeeBridgeRunner
from .selectors import SpendeeSelectorConfig, build_probe_script
from .shadow_state import ShadowState

__all__ = [
    "ActionExecutionPlan",
    "DeterminizedMCTSPolicy",
    "DeterminizedPolicyResult",
    "ObservedBoardState",
    "ObservedCard",
    "ObservedModalState",
    "ObservedNoble",
    "ObservedPlayerState",
    "ObservedReservedSlot",
    "ShadowState",
    "SpendeeBridgeConfig",
    "SpendeeBridgeRunner",
    "SpendeeCatalog",
    "SpendeeExecutor",
    "SpendeeObserver",
    "SpendeeSelectorConfig",
    "build_probe_script",
    "normalize_probe_payload",
    "plan_action",
    "sample_hydrated_states",
]
