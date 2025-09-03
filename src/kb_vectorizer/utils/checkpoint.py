from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class CheckpointStore:
    path: Path
    _data: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def open(cls, path: str | Path) -> CheckpointStore:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        return cls(path=p, _data=data)

    def get(self, key: str) -> datetime | None:
        raw = self._data.get(key)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    def update_if_greater(self, key: str, dt: datetime | None) -> bool:
        if dt is None:
            return False
        with self._lock:
            cur = self.get(key)
            if cur is None or dt > cur:
                self._data[key] = dt.isoformat()
                self._atomic_write()
                return True
        return False

    def _atomic_write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)
        # Atomic replace on Windows/Unix where supported
        os.replace(tmp, self.path)  # write-then-replace pattern.
