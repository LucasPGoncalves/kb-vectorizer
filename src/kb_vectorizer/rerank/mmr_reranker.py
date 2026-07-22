from __future__ import annotations

import math
from collections.abc import Sequence

from kb_vectorizer.storage.interfaces import StoredRecord

from .interfaces import BaseReranker


class MMRReranker(BaseReranker):
    """Maximal Marginal Relevance reranker for diversifying retrieval results.

    Greedily selects candidates that balance relevance to the query against
    redundancy with already-selected candidates, controlled by
    *lambda_mult*:

    - ``lambda_mult=1.0`` → pure relevance ranking (no diversity benefit).
    - ``lambda_mult=0.0`` → pure diversity (ignores relevance entirely).

    Operates entirely on precomputed data (``StoredRecord.score`` and
    ``StoredRecord.vector``) rather than the query text itself — *query* is
    still accepted (and ignored) to keep
    :class:`~kb_vectorizer.rerank.interfaces.BaseReranker`'s call site
    uniform across every reranker.

    **Score convention:** every candidate's ``score`` must mean "higher is
    more similar to the query" for the relevance term to work correctly.
    Set *higher_is_better=False* if your candidates instead carry a
    distance (e.g. Chroma's native query results, where lower is more
    similar) — the reranker then treats ``score`` as a **cosine distance**
    and inverts it via ``1 - score``, which only produces a meaningful
    ``[0, 1]``-ish relevance value for that specific metric. Getting this
    flag backwards silently inverts the relevance term. See
    :class:`~kb_vectorizer.storage.chromadb_store.ChromaStore` and
    :class:`~kb_vectorizer.storage.qdrant_store.QdrantStore`'s docstrings
    for which convention each backend's ``query()`` actually uses.

    Args:
        lambda_mult: Relevance/diversity trade-off in ``[0, 1]``.
        top_n: Default number of candidates to keep; overridable per call.
        higher_is_better: Whether ``StoredRecord.score`` is already a
            similarity (``True``, e.g. Qdrant) or a cosine distance
            (``False``, e.g. Chroma).

    """

    def __init__(
        self,
        lambda_mult: float = 0.5,
        top_n: int = 8,
        higher_is_better: bool = True,
    ) -> None:
        """Initialize the MMR reranker.

        Args:
            lambda_mult: Relevance/diversity trade-off in ``[0, 1]``; higher
                favors relevance, lower favors diversity.
            top_n: Default number of candidates to keep; overridable per call.
            higher_is_better: Whether candidate scores are similarities
                (``True``) or cosine distances (``False``, inverted
                internally via ``1 - score``).

        """
        self.lambda_mult = float(lambda_mult)
        self.top_n = int(top_n)
        self.higher_is_better = higher_is_better

    @staticmethod
    def _cosine_matrix(vectors: list[list[float]]) -> list[list[float]]:
        """Precompute the full pairwise cosine-similarity matrix for *vectors*.

        MMR's greedy loop repeatedly needs "similarity of candidate i to
        every already-selected candidate" across many iterations — without
        precomputing, the same (i, j) pair gets recomputed from scratch on
        every outer-loop iteration until one of the two is selected or
        exhausted. Precomputing once, and normalizing each vector's norm
        only once, avoids that redundant work.

        Args:
            vectors: One embedding per candidate, all the same dimension.

        Returns:
            An ``n x n`` matrix where ``matrix[i][j]`` is the cosine
            similarity between ``vectors[i]`` and ``vectors[j]``.

        """
        n = len(vectors)
        norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in vectors]
        normalized = [[x / norms[i] for x in v] for i, v in enumerate(vectors)]

        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = sum(a * b for a, b in zip(normalized[i], normalized[j], strict=True))
                matrix[i][j] = sim
                matrix[j][i] = sim
        return matrix

    def _mmr_select(
        self,
        *,
        relevance: list[float],
        similarity_matrix: list[list[float]],
        top_n: int,
    ) -> list[int]:
        """Greedily select indices balancing *relevance* against redundancy.

        Args:
            relevance: Query-similarity for each candidate (higher = better).
            similarity_matrix: Precomputed pairwise cosine similarities,
                from :meth:`_cosine_matrix`.
            top_n: Maximum number of indices to select.

        Returns:
            Selected indices, in selection order (most relevant first).

        """
        n = len(relevance)
        if n == 0:
            return []
        top_n = min(top_n, n)
        selected: list[int] = []
        remaining = set(range(n))

        seed = max(remaining, key=lambda i: relevance[i])
        selected.append(seed)
        remaining.remove(seed)

        while len(selected) < top_n and remaining:
            def mmr_score(i: int) -> float:
                max_sim_to_selected = max(similarity_matrix[i][j] for j in selected)
                return self.lambda_mult * relevance[i] - (1 - self.lambda_mult) * max_sim_to_selected

            next_i = max(remaining, key=mmr_score)
            selected.append(next_i)
            remaining.remove(next_i)

        return selected

    def rerank(
        self,
        query: str,
        candidates: Sequence[StoredRecord],
        top_n: int | None = None,
    ) -> list[int]:
        """Return indices into *candidates*, reordered by MMR.

        Args:
            query: Unused — accepted for interface consistency with
                :class:`~kb_vectorizer.rerank.interfaces.BaseReranker`.
            candidates: Candidates to rerank. Every candidate must have
                both ``vector`` and ``score`` populated — query the store
                with ``include_vectors=True`` to get vectors on hits.
            top_n: Maximum number of indices to return. Defaults to the
                ``top_n`` passed at construction time.

        Returns:
            Indices into *candidates*, in MMR selection order (most
            relevant first, subsequent picks balanced against diversity).

        Raises:
            ValueError: If any candidate is missing ``vector`` or ``score``.

        """
        if not candidates:
            return []

        vectors: list[list[float]] = []
        relevance: list[float] = []
        for i, c in enumerate(candidates):
            if c.vector is None:
                raise ValueError(
                    f"MMRReranker requires every candidate to have a vector "
                    f"(candidate at index {i}, id={c.id!r} has none). "
                    "Query the store with include_vectors=True."
                )
            if c.score is None:
                raise ValueError(
                    f"MMRReranker requires every candidate to have a score "
                    f"(candidate at index {i}, id={c.id!r} has none)."
                )
            vectors.append(c.vector)
            relevance.append(c.score if self.higher_is_better else (1.0 - c.score))

        similarity_matrix = self._cosine_matrix(vectors)
        resolved_top_n = top_n if top_n is not None else self.top_n
        return self._mmr_select(
            relevance=relevance,
            similarity_matrix=similarity_matrix,
            top_n=resolved_top_n,
        )
