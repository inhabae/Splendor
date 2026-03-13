from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BridgeArtifactLogger:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    async def capture_failure(self, page: Any, name: str, payload: Any) -> None:
        self.write_json(name, payload)
        screenshot_path = self.root_dir / f"{name}.png"
        html_path = self.root_dir / f"{name}.html"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        html_path.write_text(await page.content(), encoding="utf-8")
