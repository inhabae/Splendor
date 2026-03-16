from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .state_schema import ACTION_DIM, STATE_DIM


@dataclass
class ReplaySample:
    state: np.ndarray  # (STATE_DIM,) float32
    mask: np.ndarray  # (69,) bool
    action_target: int  # int64-compatible
    value_target: float  # blended value target, typically in [-1, +1]
    policy_target: np.ndarray | None = None  # (69,) float32, sums to 1 over legal actions
    generation_idx: int | None = None


@dataclass
class ReplayGeneration:
    generation_idx: int
    sample_count: int = 0
    replay_games_added: int = 0


class ReplayBuffer:
    def __init__(self) -> None:
        self._samples: deque[ReplaySample] = deque()
        self._generations: deque[ReplayGeneration] = deque()
        self._active_generation_idx: int | None = None

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def generation_count(self) -> int:
        return len(self._generations)

    @property
    def active_generation_idx(self) -> int | None:
        return self._active_generation_idx

    def start_generation(self, generation_idx: int) -> None:
        generation_idx = int(generation_idx)
        if self._active_generation_idx is not None:
            raise RuntimeError("Cannot start a new replay generation before finalizing the current generation")
        if self._generations and generation_idx <= int(self._generations[-1].generation_idx):
            raise ValueError("generation_idx must be strictly increasing")
        self._generations.append(ReplayGeneration(generation_idx=generation_idx))
        self._active_generation_idx = generation_idx

    def finalize_generation(self, *, replay_games_added: int = 0) -> None:
        if self._active_generation_idx is None:
            raise RuntimeError("No active replay generation to finalize")
        if replay_games_added < 0:
            raise ValueError("replay_games_added must be non-negative")
        self._generations[-1].replay_games_added = int(replay_games_added)
        self._active_generation_idx = None

    def trim_generations(self, max_generations: int) -> int:
        if max_generations <= 0:
            raise ValueError("max_generations must be positive")
        if self._active_generation_idx is not None:
            raise RuntimeError("Cannot trim replay generations while a generation is active")
        removed = 0
        while len(self._generations) > int(max_generations):
            oldest = self._generations.popleft()
            for _ in range(int(oldest.sample_count)):
                self._samples.popleft()
            removed += 1
        return removed

    @staticmethod
    def _validate_sample(sample: ReplaySample) -> None:
        if sample.state.shape != (STATE_DIM,):
            raise ValueError(f"Invalid state shape {sample.state.shape}")
        if sample.mask.shape != (ACTION_DIM,):
            raise ValueError(f"Invalid mask shape {sample.mask.shape}")
        action_idx = int(sample.action_target)
        if not (0 <= action_idx < ACTION_DIM):
            raise ValueError(f"action_target out of range: {sample.action_target}")
        if not bool(sample.mask[action_idx]):
            raise ValueError("action_target must be legal under sample.mask")
        if sample.policy_target is None:
            policy_target = np.zeros((ACTION_DIM,), dtype=np.float32)
            policy_target[action_idx] = 1.0
            sample.policy_target = policy_target
        if sample.policy_target.shape != (ACTION_DIM,):
            raise ValueError(f"Invalid policy_target shape {sample.policy_target.shape}")
        if not np.isfinite(sample.policy_target).all():
            raise ValueError("Non-finite values in policy_target")
        if (sample.policy_target < 0).any():
            raise ValueError("policy_target cannot contain negative values")
        if (sample.policy_target[~sample.mask] != 0).any():
            raise ValueError("policy_target must assign zero probability to illegal actions")
        prob_sum = float(sample.policy_target.sum())
        if abs(prob_sum - 1.0) > 1e-5:
            raise ValueError(f"policy_target must sum to 1 (got {prob_sum})")

    def add(self, sample: ReplaySample) -> None:
        self._validate_sample(sample)
        if self._active_generation_idx is not None:
            sample.generation_idx = int(self._active_generation_idx)
        elif sample.generation_idx is not None:
            sample.generation_idx = int(sample.generation_idx)
        self._samples.append(sample)
        if self._active_generation_idx is not None:
            self._generations[-1].sample_count += 1

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        for s in samples:
            self.add(s)

    def sample_batch(self, batch_size: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if not self._samples:
            raise ValueError("ReplayBuffer is empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        picks = random.sample(tuple(self._samples), k=min(batch_size, len(self._samples)))
        states = np.stack([s.state for s in picks], axis=0).astype(np.float32, copy=False)
        masks = np.stack([s.mask for s in picks], axis=0).astype(np.bool_, copy=False)
        actions = np.asarray([s.action_target for s in picks], dtype=np.int64)
        values = np.asarray([s.value_target for s in picks], dtype=np.float32)
        policies = np.stack([s.policy_target for s in picks], axis=0).astype(np.float32, copy=False)
        batch = {
            "state": torch.as_tensor(states, dtype=torch.float32, device=device),
            "mask": torch.as_tensor(masks, dtype=torch.bool, device=device),
            "action_target": torch.as_tensor(actions, dtype=torch.long, device=device),
            "value_target": torch.as_tensor(values, dtype=torch.float32, device=device),
            "policy_target": torch.as_tensor(policies, dtype=torch.float32, device=device),
        }
        if not torch.isfinite(batch["state"]).all():
            raise ValueError("Non-finite values in replay batch state")
        return batch

    def save_npz(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self._samples)
        states = np.zeros((n, STATE_DIM), dtype=np.float32)
        masks = np.zeros((n, ACTION_DIM), dtype=np.bool_)
        actions = np.zeros((n,), dtype=np.int64)
        values = np.zeros((n,), dtype=np.float32)
        policies = np.zeros((n, ACTION_DIM), dtype=np.float32)
        for i, sample in enumerate(self._samples):
            states[i] = sample.state
            masks[i] = sample.mask
            actions[i] = int(sample.action_target)
            values[i] = float(sample.value_target)
            if sample.policy_target is None:
                policy = np.zeros((ACTION_DIM,), dtype=np.float32)
                policy[int(sample.action_target)] = 1.0
                policies[i] = policy
            else:
                policies[i] = sample.policy_target

        metadata = {
            "count": int(n),
            "generations": [asdict(generation) for generation in self._generations],
        }
        np.savez_compressed(
            out_path,
            metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
            state=states,
            mask=masks,
            action_target=actions,
            value_target=values,
            policy_target=policies,
            generation_idx=np.asarray(
                [(-1 if sample.generation_idx is None else int(sample.generation_idx)) for sample in self._samples],
                dtype=np.int64,
            ),
        )
        return out_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "ReplayBuffer":
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Replay buffer file not found: {in_path}")
        with np.load(in_path, allow_pickle=False) as data:
            metadata_raw = data["metadata_json"]
            if metadata_raw.ndim == 0:
                metadata_json = str(metadata_raw.item())
            else:
                metadata_json = str(metadata_raw.tolist())
            metadata = json.loads(metadata_json)
            states = np.asarray(data["state"], dtype=np.float32)
            masks = np.asarray(data["mask"], dtype=np.bool_)
            actions = np.asarray(data["action_target"], dtype=np.int64)
            values = np.asarray(data["value_target"], dtype=np.float32)
            policies = np.asarray(data["policy_target"], dtype=np.float32)
            generation_indices = (
                np.asarray(data["generation_idx"], dtype=np.int64)
                if "generation_idx" in data.files
                else np.full((states.shape[0],), -1, dtype=np.int64)
            )

        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"Invalid replay state shape {states.shape}")
        if masks.ndim != 2 or masks.shape[1] != ACTION_DIM:
            raise ValueError(f"Invalid replay mask shape {masks.shape}")
        if policies.ndim != 2 or policies.shape[1] != ACTION_DIM:
            raise ValueError(f"Invalid replay policy shape {policies.shape}")
        n = int(states.shape[0])
        if (
            masks.shape[0] != n
            or actions.shape[0] != n
            or values.shape[0] != n
            or policies.shape[0] != n
            or generation_indices.shape[0] != n
        ):
            raise ValueError("Replay arrays have mismatched leading dimensions")

        out = cls()
        samples = [
            ReplaySample(
                state=states[i].copy(),
                mask=masks[i].copy(),
                action_target=int(actions[i]),
                value_target=float(values[i]),
                policy_target=policies[i].copy(),
                generation_idx=(None if int(generation_indices[i]) < 0 else int(generation_indices[i])),
            )
            for i in range(n)
        ]
        out.extend(samples)
        generation_entries = metadata.get("generations")
        if isinstance(generation_entries, list):
            out._generations = deque(
                ReplayGeneration(
                    generation_idx=int(entry["generation_idx"]),
                    sample_count=int(entry["sample_count"]),
                    replay_games_added=int(entry.get("replay_games_added", 0)),
                )
                for entry in generation_entries
            )
        elif n > 0:
            unique_generation_indices = [int(idx) for idx in generation_indices.tolist() if int(idx) >= 0]
            if unique_generation_indices:
                counts: dict[int, int] = {}
                for generation_idx in unique_generation_indices:
                    counts[generation_idx] = counts.get(generation_idx, 0) + 1
                out._generations = deque(
                    ReplayGeneration(generation_idx=int(generation_idx), sample_count=int(sample_count))
                    for generation_idx, sample_count in sorted(counts.items())
                )
            else:
                out._generations = deque([ReplayGeneration(generation_idx=0, sample_count=n)])
        return out
