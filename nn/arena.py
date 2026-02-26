from __future__ import annotations

import argparse
import math

from .benchmark import run_matchup
from .native_env import SplendorNativeEnv
from .mcts import MCTSConfig
from .opponents import CheckpointMCTSOpponent


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4 * total * total)))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct checkpoint-vs-checkpoint arena match")
    p.add_argument("--candidate-checkpoint", type=str, required=True)
    p.add_argument("--opponent-checkpoint", type=str, required=True)
    p.add_argument("--candidate-name", type=str, default="candidate")
    p.add_argument("--opponent-name", type=str, default="opponent")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mcts-sims", type=int, default=64)
    p.add_argument("--mcts-c-puct", type=float, default=1.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cycle-idx", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.mcts_sims <= 0:
        raise ValueError("--mcts-sims must be positive")

    cfg = MCTSConfig(
        num_simulations=int(args.mcts_sims),
        c_puct=float(args.mcts_c_puct),
        temperature_moves=0,
        temperature=0.0,
        root_dirichlet_noise=False,
    )
    candidate = CheckpointMCTSOpponent(
        checkpoint_path=str(args.candidate_checkpoint),
        mcts_config=cfg,
        device=str(args.device),
        name=str(args.candidate_name),
    )
    opponent = CheckpointMCTSOpponent(
        checkpoint_path=str(args.opponent_checkpoint),
        mcts_config=cfg,
        device=str(args.device),
        name=str(args.opponent_name),
    )

    with SplendorNativeEnv() as env:
        result = run_matchup(
            env,
            candidate,
            opponent,
            games=int(args.games),
            max_turns=int(args.max_turns),
            seed_base=int(args.seed),
            cycle_idx=int(args.cycle_idx),
        )

    decisive_games = result.candidate_wins + result.candidate_losses
    wl_lo, wl_hi = _wilson_interval(result.candidate_wins, max(1, decisive_games))
    wr_lo, wr_hi = _wilson_interval(result.candidate_wins, result.games)

    print("arena_result")
    print(f"candidate_checkpoint={args.candidate_checkpoint}")
    print(f"opponent_checkpoint={args.opponent_checkpoint}")
    print(f"candidate_name={args.candidate_name}")
    print(f"opponent_name={result.opponent_name}")
    print(f"games={result.games}")
    print(f"candidate_wins={result.candidate_wins}")
    print(f"candidate_losses={result.candidate_losses}")
    print(f"draws={result.draws}")
    print(f"candidate_win_rate={result.candidate_win_rate:.4f}")
    print(f"candidate_nonloss_rate={result.candidate_nonloss_rate:.4f}")
    print(f"candidate_win_rate_95ci=[{wr_lo:.4f},{wr_hi:.4f}]")
    print(f"decisive_win_rate={(result.candidate_wins / decisive_games) if decisive_games else 0.0:.4f}")
    print(f"decisive_win_rate_95ci=[{wl_lo:.4f},{wl_hi:.4f}]")
    print(f"seat0_games={result.seat0_games}")
    print(f"seat1_games={result.seat1_games}")
    print(f"avg_turns_per_game={result.avg_turns_per_game:.2f}")
    print(f"cutoff_rate={result.cutoff_rate:.4f}")


if __name__ == "__main__":
    main()
