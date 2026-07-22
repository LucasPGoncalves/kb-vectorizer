from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class KeywordMatch:
    """A single keyword-search hit.

    Attributes:
        id: The application-level document/chunk ID that matched.
        score: Relevance score. Higher is more relevant, regardless of
            which :class:`BaseKeywordIndex` implementation produced it —
            unlike :class:`~kb_vectorizer.storage.interfaces.StoredRecord`,
            this is not tied to any one backend's native scoring convention.

    """

    id: str
    score: float


class BaseKeywordIndex(ABC):
    """Backend-agnostic interface for keyword (lexical/BM25-style) search.

    Two implementations ship with this package, chosen for the deployment
    you're in — same interface either way, so calling code never changes:

    - :class:`~kb_vectorizer.retrieval.inmemory_keyword_index.InMemoryKeywordIndex`
      — holds the whole tokenized corpus in memory and rebuilds a
      ``rank_bm25.BM25Okapi`` index on every mutation. Simple and exact, but
      doesn't scale to large corpora. Good for prototyping.
    - :class:`~kb_vectorizer.retrieval.native_keyword_index.NativeKeywordIndex`
      — delegates to a vector store's own native sparse-vector search (e.g.
      :class:`~kb_vectorizer.storage.qdrant_store.QdrantStore` configured
      with ``enable_bm25=True``), which computes term frequencies per
      document independently and lets the backend maintain corpus-wide IDF
      statistics incrementally, server-side — no in-memory corpus required.
      Use this in production.
    """

    @abstractmethod
    def upsert(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        """Index or re-index a batch of documents.

        Args:
            ids: Application-level IDs, one per text.
            texts: Raw text content, one per ID.

        """
        ...

    @abstractmethod
    def search(self, query: str, k: int = 50) -> list[KeywordMatch]:
        """Return the top-*k* documents matching *query* by keyword relevance.

        Args:
            query: Query string.
            k: Maximum number of matches to return.

        Returns:
            Matches ordered best-first (highest score first).

        """
        ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Remove documents from the index by ID.

        Args:
            ids: Application-level IDs to remove.

        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this index (connections, memory, …)."""
        ...


@runtime_checkable
class KeywordSearchHit(Protocol):
    """Structural type for a single hit returned by a native keyword search.

    Deliberately minimal — :class:`~kb_vectorizer.storage.interfaces.StoredRecord`
    satisfies this today, but any object exposing ``id``/``score`` works,
    so :class:`~kb_vectorizer.retrieval.native_keyword_index.NativeKeywordIndex`
    never needs to import anything from ``kb_vectorizer.storage``.
    """

    id: str
    score: float | None


@runtime_checkable
class SupportsKeywordSearch(Protocol):
    """Structural type for a vector store with native keyword-search support.

    Any store implementing this method — regardless of which vector
    database backs it — can be wrapped by
    :class:`~kb_vectorizer.retrieval.native_keyword_index.NativeKeywordIndex`.
    Today only :class:`~kb_vectorizer.storage.qdrant_store.QdrantStore`
    (constructed with ``enable_bm25=True``) implements it, but nothing here
    is Qdrant-specific — a future backend with its own native sparse/lexical
    search need only implement this same method to plug into the same
    :class:`~kb_vectorizer.retrieval.interfaces.BaseKeywordIndex` machinery.
    """

    def keyword_search(
        self, *, collection: str, query_text: str, k: int
    ) -> Sequence[KeywordSearchHit]:
        """Return the top-*k* records matching *query_text* by native keyword relevance.

        Args:
            collection: Collection/index name to search within.
            query_text: Raw query string.
            k: Maximum number of hits to return.

        Returns:
            Hits ordered best-first, each with ``id`` and ``score`` populated.

        """
        ...
