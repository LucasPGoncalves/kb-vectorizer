from __future__ import annotations

from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from kb_vectorizer.retrieval.interfaces import BaseKeywordIndex, KeywordMatch
from kb_vectorizer.text.tokenizer import tokenize


class InMemoryKeywordIndex(BaseKeywordIndex):
    """In-process BM25 keyword index, backed by ``rank_bm25.BM25Okapi``.

    Holds every document's raw text and tokenized form in memory, and
    rebuilds the full ``BM25Okapi`` index from scratch on every
    :meth:`upsert`/:meth:`delete` call — ``BM25Okapi`` has no incremental
    update API, and computing its corpus-wide statistics (average document
    length, per-term document frequency) requires seeing the whole corpus
    at once.

    This is fine for prototyping and small-to-medium corpora, but the full
    in-memory corpus and full-rebuild-per-mutation are exactly the "memory
    problem" that :class:`~kb_vectorizer.retrieval.native_keyword_index.NativeKeywordIndex`
    exists to avoid for production use.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory keyword index."""
        self._docs: dict[str, str] = {}
        self._order: list[str] = []
        self._bm25: BM25Okapi | None = None

    def _rebuild(self) -> None:
        """Retokenize every stored document and rebuild the BM25Okapi index."""
        self._order = list(self._docs.keys())
        if not self._order:
            self._bm25 = None
            return
        tokenized_corpus = [tokenize(self._docs[doc_id]) for doc_id in self._order]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def upsert(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        """Insert or update documents, then rebuild the index over the full corpus.

        Args:
            ids: Application-level IDs, one per text.
            texts: Raw text content, one per ID.

        """
        for doc_id, text in zip(ids, texts, strict=True):
            self._docs[doc_id] = text
        self._rebuild()

    def search(self, query: str, k: int = 50) -> list[KeywordMatch]:
        """Return the top-*k* documents matching *query* by BM25 score.

        Args:
            query: Query string.
            k: Maximum number of matches to return.

        Returns:
            Matches ordered best-first (highest BM25 score first). Empty
            if the index has no documents.

        """
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [KeywordMatch(id=self._order[idx], score=float(score)) for idx, score in ranked]

    def delete(self, ids: Sequence[str]) -> None:
        """Remove documents from the index by ID, then rebuild.

        Args:
            ids: Application-level IDs to remove. IDs not present are ignored.

        """
        for doc_id in ids:
            self._docs.pop(doc_id, None)
        self._rebuild()

    def close(self) -> None:
        """Clear the in-memory corpus and index.

        Safe to call multiple times.
        """
        self._docs.clear()
        self._order = []
        self._bm25 = None
