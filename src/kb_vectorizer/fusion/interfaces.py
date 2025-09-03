from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable


class BaseFusor(ABC):

    @abstractmethod
    def fuse(rankings: Iterable[list[str]], k: int = 60) -> list[tuple[str, float]]:
        ...