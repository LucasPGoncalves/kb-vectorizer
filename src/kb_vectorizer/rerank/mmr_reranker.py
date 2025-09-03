from __future__ import annotations

import math
from typing import Any

from .interfaces import BaseReranker


class MMRReranker(BaseReranker):

    def __init__(self, lambda_mult: float = 0.5, top_n: int = 8):
        self.lambda_mult = float(lambda_mult)
        self.top_n = int(top_n)

    @staticmethod
    def _cosine(u: list[float], v: list[float]) -> float:
        num = sum(a*b for a,b in zip(u, v, strict=False))
        du = math.sqrt(sum(a*a for a in u))
        dv = math.sqrt(sum(b*b for b in v))
        return 0.0 if du == 0 or dv == 0 else num / (du * dv)

    def _mmr_rerank(
        self,
        *,
        query_similarity: list[float],          # similarity of each candidate to the query (e.g., 1 - cosine_distance)
        candidate_embeddings: list[list[float]],# embeddings of candidates (for intra-set similarity)
        lambda_mult: float = 0.5,               # 0..1 (higher => more relevance, lower => more diversity)
        top_n: int = 5
    ) -> list[int]:
        """Return indices of selected candidates after MMR re-ranking.
        
        Assumes query_similarity[i] corresponds to candidate_embeddings[i].
        """
        n = len(candidate_embeddings)
        if n == 0:
            return []
        top_n = min(top_n, n)
        selected: list[int] = []
        remaining = set(range(n))

        # seed with the most relevant item
        seed = max(remaining, key=lambda i: query_similarity[i])
        selected.append(seed)
        remaining.remove(seed)

        while len(selected) < top_n and remaining:
            def mmr_score(i: int) -> float:
                max_sim_to_selected = max(self._cosine(candidate_embeddings[i], candidate_embeddings[j]) for j in selected) if selected else 0.0
                return lambda_mult * query_similarity[i] - (1 - lambda_mult) * max_sim_to_selected

            next_i = max(remaining, key=mmr_score)
            selected.append(next_i)
            remaining.remove(next_i)

        return selected

    def rerank(self, candidates: list[dict[str, Any]], top_n: int | None = None) -> list[int]:
        # Expect: candidates have "distance" (cosine distance) and "embedding"
        sims = [1 - (c.get("distance", 1.0) or 1.0) for c in candidates]
        embs = [c.get("embedding") for c in candidates]
        keep = self._mmr_rerank(
            query_similarity=sims,
            candidate_embeddings=embs,
            lambda_mult=self.lambda_mult,
            top_n=min(top_n or self.top_n, len(candidates)),
        )
        return keep
