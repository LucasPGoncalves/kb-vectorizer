from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from kb_vectorizer.storage.interfaces import StoredRecord


class BaseReranker(ABC):
    """Reorders a list of retrieved candidates for a given query.

    Implementations return a *permutation* — a list of indices into
    *candidates*, ordered best-first — rather than reordered candidate
    objects. This lets a caller apply the same ordering to a parallel
    list (e.g. the original :class:`StoredRecord` objects, or any other
    per-candidate data) without the reranker needing to know about or
    preserve every field::

        order = reranker.rerank(query, candidates, top_n=5)
        reranked = [candidates[i] for i in order]

    Every implementation accepts *query*, even ones that don't use the
    query text directly (e.g.
    :class:`~kb_vectorizer.rerank.mmr_reranker.MMRReranker`, which only
    needs precomputed embeddings/scores) — keeping the call site uniform
    across implementations is the entire point of a shared interface.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[StoredRecord],
        top_n: int | None = None,
    ) -> list[int]:
        """Return indices into *candidates*, reordered best-first.

        Args:
            query: The original query string.
            candidates: Retrieved candidates to rerank, typically one
                inner list from a
                :meth:`~kb_vectorizer.storage.interfaces.BaseVectorStore.query`
                call.
            top_n: Maximum number of indices to return. ``None`` returns
                all of them, reordered.

        Returns:
            Indices into *candidates*, best match first.

        """
        ...
