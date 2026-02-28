from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .native_env import SplendorNativeEnv

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

try:
    from .mcts import MCTSConfig, run_mcts
except Exception:  # pragma: no cover
    MCTSConfig = None
    run_mcts = None

from .state_schema import ACTION_DIM


@dataclass
class NativeEnvBenchmarkResult:
    reset_ops_per_sec: float
    step_ops_per_sec: float
    step_iterations: int
    mcts_enabled: bool
    mcts_wall_time_sec: float | None
    mcts_sims_per_sec: float | None
    mcts_num_simulations: int | None
    python_version: str
    platform: str
    numpy_version: str
    torch_version: str | None


def _timeit(fn, iterations: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    return time.perf_counter() - t0


def _first_legal_action(mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("No legal actions available")
    return int(legal[0])


def _random_legal_action(mask: np.ndarray, rng: random.Random) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("No legal actions available")
    return int(rng.choice(legal.tolist()))


if nn is not None:
    class _ZeroModel(nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            logits = torch.zeros((batch, ACTION_DIM), dtype=torch.float32, device=x.device)
            value = torch.zeros((batch,), dtype=torch.float32, device=x.device)
            return logits, value
else:  # pragma: no cover
    _ZeroModel = None


def benchmark_native_env(
    *,
    seed: int = 123,
    reset_iterations: int = 500,
    step_iterations: int = 5000,
    warmup_steps: int = 256,
    include_mcts: bool = False,
    mcts_num_simulations: int = 64,
    mcts_turns_taken: int = 0,
    mcts_device: str = "cpu",
) -> NativeEnvBenchmarkResult:
    rng = random.Random(seed)

    with SplendorNativeEnv() as env:
        env.reset(seed)

        for _ in range(max(0, warmup_steps)):
            state = env.get_state()
            if state.is_terminal:
                env.reset(seed)
                continue
            env.step(_random_legal_action(state.mask, rng))

        reset_elapsed = _timeit(lambda: env.reset(seed), reset_iterations)

        env.reset(seed)

        def do_step_once() -> None:
            state = env.get_state()
            if state.is_terminal:
                env.reset(seed)
                return
            env.step(_random_legal_action(state.mask, rng))

        step_elapsed = _timeit(do_step_once, step_iterations)

        mcts_wall_time_sec: float | None = None
        mcts_sims_per_sec: float | None = None
        mcts_enabled = False

        if include_mcts and run_mcts is not None and MCTSConfig is not None and torch is not None and _ZeroModel is not None:
            env.reset(seed)
            root = env.get_state()
            if not root.is_terminal:
                model = _ZeroModel()
                if mcts_device != "cpu":
                    model = model.to(mcts_device)
                cfg = MCTSConfig(num_simulations=int(mcts_num_simulations))
                t0 = time.perf_counter()
                _ = run_mcts(
                    env,
                    model,
                    root,
                    turns_taken=int(mcts_turns_taken),
                    device=mcts_device,
                    config=cfg,
                    rng=random.Random(seed),
                )
                mcts_wall_time_sec = time.perf_counter() - t0
                if mcts_wall_time_sec > 0:
                    mcts_sims_per_sec = float(mcts_num_simulations) / mcts_wall_time_sec
                mcts_enabled = True

    return NativeEnvBenchmarkResult(
        reset_ops_per_sec=(float(reset_iterations) / reset_elapsed) if reset_elapsed > 0 else 0.0,
        step_ops_per_sec=(float(step_iterations) / step_elapsed) if step_elapsed > 0 else 0.0,
        step_iterations=int(step_iterations),
        mcts_enabled=bool(mcts_enabled),
        mcts_wall_time_sec=mcts_wall_time_sec,
        mcts_sims_per_sec=mcts_sims_per_sec,
        mcts_num_simulations=int(mcts_num_simulations) if mcts_enabled else None,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        numpy_version=np.__version__,
        torch_version=(torch.__version__ if torch is not None else None),
    )


def _print_pretty(result: NativeEnvBenchmarkResult) -> None:
    print("native_env_benchmark")
    print(f"  reset_ops_per_sec: {result.reset_ops_per_sec:.2f}")
    print(f"  step_ops_per_sec: {result.step_ops_per_sec:.2f}")
    if result.mcts_enabled:
        print(f"  mcts_wall_time_sec: {result.mcts_wall_time_sec:.6f}")
        print(f"  mcts_sims_per_sec: {result.mcts_sims_per_sec:.2f}")
        print(f"  mcts_num_simulations: {result.mcts_num_simulations}")
    else:
        print("  mcts: skipped")
    print(f"  python: {result.python_version}")
    print(f"  numpy: {result.numpy_version}")
    print(f"  torch: {result.torch_version}")
    print(f"  platform: {result.platform}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Manual benchmark for native Splendor pybind11 env")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--reset-iters", type=int, default=500)
    p.add_argument("--step-iters", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=256)
    p.add_argument("--include-mcts", action="store_true")
    p.add_argument("--mcts-sims", type=int, default=64)
    p.add_argument("--mcts-turns-taken", type=int, default=0)
    p.add_argument("--mcts-device", type=str, default="cpu")
    p.add_argument("--json", action="store_true", help="Print JSON instead of human-readable output")
    args = p.parse_args(argv)

    result = benchmark_native_env(
        seed=int(args.seed),
        reset_iterations=int(args.reset_iters),
        step_iterations=int(args.step_iters),
        warmup_steps=int(args.warmup_steps),
        include_mcts=bool(args.include_mcts),
        mcts_num_simulations=int(args.mcts_sims),
        mcts_turns_taken=int(args.mcts_turns_taken),
        mcts_device=str(args.mcts_device),
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        _print_pretty(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
