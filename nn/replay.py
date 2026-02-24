from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from .state_codec import ACTION_DIM, STATE_DIM


@dataclass
class ReplaySample:
    state: np.ndarray  # (246,) float32
    mask: np.ndarray  # (69,) bool
    action_target: int  # int64-compatible
    value_target: float  # -1,0,+1


class ReplayBuffer:
    def __init__(self) -> None:
        self._samples: List[ReplaySample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: ReplaySample) -> None:
        if sample.state.shape != (STATE_DIM,):
            raise ValueError(f"Invalid state shape {sample.state.shape}")
        if sample.mask.shape != (ACTION_DIM,):
            raise ValueError(f"Invalid mask shape {sample.mask.shape}")
        action_idx = int(sample.action_target)
        if not (0 <= action_idx < ACTION_DIM):
            raise ValueError(f"action_target out of range: {sample.action_target}")
        if not bool(sample.mask[action_idx]):
            raise ValueError("action_target must be legal under sample.mask")
        self._samples.append(sample)

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        for s in samples:
            self.add(s)

    def sample_batch(self, batch_size: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if not self._samples:
            raise ValueError("ReplayBuffer is empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        picks = random.sample(self._samples, k=min(batch_size, len(self._samples)))
        states = np.stack([s.state for s in picks], axis=0).astype(np.float32, copy=False)
        masks = np.stack([s.mask for s in picks], axis=0).astype(np.bool_, copy=False)
        actions = np.asarray([s.action_target for s in picks], dtype=np.int64)
        values = np.asarray([s.value_target for s in picks], dtype=np.float32)
        batch = {
            "state": torch.as_tensor(states, dtype=torch.float32, device=device),
            "mask": torch.as_tensor(masks, dtype=torch.bool, device=device),
            "action_target": torch.as_tensor(actions, dtype=torch.long, device=device),
            "value_target": torch.as_tensor(values, dtype=torch.float32, device=device),
        }
        if not torch.isfinite(batch["state"]).all():
            raise ValueError("Non-finite values in replay batch state")
        return batch
