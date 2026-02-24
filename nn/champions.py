from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ChampionEntry:
    checkpoint_path: str
    accepted_at: str
    run_id: str = ""
    cycle_idx: int = 0
    notes: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChampionRegistry:
    version: int = 1
    champions: list[ChampionEntry] = field(default_factory=list)


def load_champion_registry(path: str | Path) -> ChampionRegistry:
    p = Path(path)
    if not p.exists():
        return ChampionRegistry()
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Champion registry must be a JSON object")
    version = int(raw.get("version", 1))
    raw_entries = raw.get("champions", [])
    if not isinstance(raw_entries, list):
        raise ValueError("Champion registry 'champions' must be a list")
    champs: list[ChampionEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ValueError("Champion entry must be an object")
        champs.append(
            ChampionEntry(
                checkpoint_path=str(item.get("checkpoint_path", "")),
                accepted_at=str(item.get("accepted_at", "")),
                run_id=str(item.get("run_id", "")),
                cycle_idx=int(item.get("cycle_idx", 0)),
                notes=str(item.get("notes", "")),
                metrics=dict(item.get("metrics", {}) or {}),
            )
        )
    return ChampionRegistry(version=version, champions=champs)


def save_champion_registry(path: str | Path, registry: ChampionRegistry) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": int(registry.version), "champions": [asdict(c) for c in registry.champions]}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def get_current_and_previous_champions(registry: ChampionRegistry) -> list[ChampionEntry]:
    if not registry.champions:
        return []
    if len(registry.champions) == 1:
        return [registry.champions[-1]]
    return [registry.champions[-1], registry.champions[-2]]


def append_accepted_champion(registry: ChampionRegistry, entry: ChampionEntry) -> None:
    registry.champions.append(entry)


def has_any_champion(registry: ChampionRegistry) -> bool:
    return len(registry.champions) > 0


def build_champion_entry_from_promotion(
    *,
    checkpoint_path: str,
    run_id: str,
    cycle_idx: int,
    metrics: dict[str, Any],
    notes: str = "",
) -> ChampionEntry:
    accepted_at = datetime.now(timezone.utc).isoformat()
    return ChampionEntry(
        checkpoint_path=str(checkpoint_path),
        accepted_at=accepted_at,
        run_id=str(run_id),
        cycle_idx=int(cycle_idx),
        notes=str(notes),
        metrics=dict(metrics),
    )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Champion registry helper")
    p.add_argument("--registry", type=str, default="nn_artifacts/champions.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List champion entries")

    add = sub.add_parser("add", help="Append a champion checkpoint")
    add.add_argument("checkpoint_path", type=str)
    add.add_argument("--run-id", type=str, default="")
    add.add_argument("--cycle-idx", type=int, default=0)
    add.add_argument("--notes", type=str, default="")
    return p


def main() -> None:
    args = _build_cli().parse_args()
    registry = load_champion_registry(args.registry)
    if args.cmd == "list":
        if not registry.champions:
            print("No champions in registry")
            return
        for i, c in enumerate(registry.champions, start=1):
            print(f"{i}. {c.checkpoint_path} run_id={c.run_id} cycle_idx={c.cycle_idx} accepted_at={c.accepted_at}")
        return

    accepted_at = datetime.now(timezone.utc).isoformat()
    append_accepted_champion(
        registry,
        ChampionEntry(
            checkpoint_path=str(args.checkpoint_path),
            accepted_at=accepted_at,
            run_id=str(args.run_id),
            cycle_idx=int(args.cycle_idx),
            notes=str(args.notes),
        ),
    )
    save_champion_registry(args.registry, registry)
    print(f"Added champion: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
