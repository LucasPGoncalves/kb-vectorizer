from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable


class BaseFusor(ABC):
    """Combines multiple ranked ID lists into a single fused ranking.

    Implementations operate purely on IDs and their rank within each
    input list — they never see the underlying documents, vectors, or
    metadata. That's what makes a fusor reusable regardless of which
    :class:`~kb_vectorizer.storage.interfaces.BaseVectorStore` or
    :class:`~kb_vectorizer.retrieval.interfaces.BaseKeywordIndex`
    produced each ranking: both expose an ``id`` on every hit, and that's
    all a fusor needs.
    """

    @abstractmethod
    def fuse(self, rankings: Iterable[list[str]], k: int = 60) -> list[tuple[str, float]]:
        """Fuse multiple rankings of IDs into one combined ranking.

        Args:
            rankings: One ranked list of IDs per retriever/lane, each
                ordered best-first (rank 1 = most relevant).
            k: Fusion constant; meaning is algorithm-specific.

        Returns:
            ``(id, fused_score)`` pairs, sorted best-first.

        """
        ...
