from __future__ import annotations

from collections.abc import Iterable

from .interfaces import BaseFusor


class RRFFusor(BaseFusor):
    """Reciprocal Rank Fusion (RRF) — combines rankings without needing their raw scores.

    RRF only looks at each ID's *rank* within a ranking, not its
    similarity score or distance — which is exactly why it can combine a
    dense-vector ranking (Chroma-style distances, lower is better;
    Qdrant-style scores, higher is better) with a keyword ranking
    (BM25 score, higher is better) without first having to normalize or
    reconcile those different conventions.
    """

    @staticmethod
    def fuse(rankings: Iterable[list[str]], k: int = 60) -> list[tuple[str, float]]:
        """Fuse multiple rankings of IDs via Reciprocal Rank Fusion.

        ``score(d) = sum over rankings r of 1 / (k + rank_r(d))``, where
        ``rank_r(d)`` is ``d``'s 1-based position in ranking ``r`` (an ID
        absent from a given ranking simply contributes nothing from it).

        Args:
            rankings: One ranked list of IDs per retriever/lane, each
                ordered best-first.
            k: Fusion constant that dampens the influence of low ranks;
                60 is the standard default from the original RRF paper.

        Returns:
            ``(id, fused_score)`` pairs, sorted by fused score descending.

        """
        scores: dict[str, float] = {}
        for run in rankings:
            for r, doc_id in enumerate(run, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
