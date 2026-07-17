from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    path: Path

    def load(self) -> dict[str, dict[str, Any]]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding='utf-8'))
        
        return {}
    
    def save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, key: str) -> tuple[str, int] | None:
        data = self.load().get(key)
        if not data:
            return None
        return data.get("updated_at"), int(data.get("last_id", 0))

    def put(self, key: str, updated_at: str, last_id: int) -> None:
        allc = self.load()
        allc[key] = {"updated_at": updated_at, "last_id": last_id, "ts": int(time.time())}
        self.save(allc)

