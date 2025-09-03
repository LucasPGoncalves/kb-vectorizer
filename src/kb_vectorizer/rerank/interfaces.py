from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Reranks a list of candidates for a given query. Returns new order of indices."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int | None = None) -> list[int]:
        ...
