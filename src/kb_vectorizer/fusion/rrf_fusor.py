from __future__ import annotations

from collections.abc import Iterable

from .interfaces import BaseFusor


class RRFFusor(BaseFusor):

    @staticmethod
    def fuse(rankings: Iterable[list[str]], k: int = 60) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion (RRF).
        rankings: list of ranked ID lists (best first) from different retrievers.
        Returns: list of (id, score) sorted by fused score desc.

        score(d) = sum_{runs r} 1 / (k + rank_r(d)), rank_r(d) starts at 1.
        """
        scores: dict[str, float] = {}
        for run in rankings:
            for r, doc_id in enumerate(run, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
