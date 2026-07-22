from __future__ import annotations

from collections.abc import Sequence

from kb_vectorizer.retrieval.interfaces import BaseKeywordIndex, KeywordMatch, SupportsKeywordSearch


class NativeKeywordIndex(BaseKeywordIndex):
    """Keyword index that delegates search to a store's own native sparse-vector support.

    Wraps any object implementing
    :class:`~kb_vectorizer.retrieval.interfaces.SupportsKeywordSearch` —
    structurally, not by inheritance, so this class has no dependency on
    any concrete store. Today that means
    :class:`~kb_vectorizer.storage.qdrant_store.QdrantStore` constructed
    with ``enable_bm25=True``, but the same wrapper works unchanged with
    any future backend that implements the same method.

    Since the wrapped store computes and stores the sparse (keyword) vector
    automatically as part of its own ``upsert``/``delete`` — the same point
    carries both the dense and sparse vectors together — :meth:`upsert` and
    :meth:`delete` here are no-ops: indexing and removal already happened
    when you called those methods on the store itself. Only :meth:`search`
    does real work.

    Args:
        store: A store implementing ``keyword_search()``.
        collection: Name of the collection to search within.

    """

    def __init__(self, store: SupportsKeywordSearch, collection: str) -> None:
        """Wrap *store* for keyword search scoped to *collection*.

        Args:
            store: A store implementing ``keyword_search()``.
            collection: Name of the collection to search within.

        """
        self._store = store
        self._collection = collection

    def upsert(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        """No-op — indexing already happens via the wrapped store's own upsert().

        The store computes and attaches the sparse (keyword) vector
        alongside the dense one in the same call, so there is nothing
        additional to do here.

        Args:
            ids: Ignored.
            texts: Ignored.

        """

    def search(self, query: str, k: int = 50) -> list[KeywordMatch]:
        """Return the top-*k* documents matching *query* via the store's native search.

        Args:
            query: Query string.
            k: Maximum number of matches to return.

        Returns:
            Matches ordered best-first, as returned by the wrapped store.

        """
        hits = self._store.keyword_search(collection=self._collection, query_text=query, k=k)
        return [KeywordMatch(id=hit.id, score=hit.score if hit.score is not None else 0.0) for hit in hits]

    def delete(self, ids: Sequence[str]) -> None:
        """No-op — deletion already happens via the wrapped store's own delete().

        Since a single point carries both the dense and sparse vector
        together, deleting it from the store removes both at once.

        Args:
            ids: Ignored.

        """

    def close(self) -> None:
        """No-op — the wrapped store's lifecycle is managed by its owner, not this wrapper."""
