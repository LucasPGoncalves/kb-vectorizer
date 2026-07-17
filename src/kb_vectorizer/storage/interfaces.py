from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from kb_vectorizer.embedding.interfaces import BaseEmbedder


@dataclass
class StoredRecord:
    """A single record returned from a vector store.

    Used both for plain lookups (:meth:`BaseVectorStore.get`, where
    ``score`` is always ``None``) and for search hits
    (:meth:`BaseVectorStore.query`, where ``score`` is populated).

    Attributes:
        id: The original application-level identifier for this record.
        vector: The embedding vector, if it was fetched.
        document: The raw text content stored alongside the vector.
        metadata: Arbitrary key-value metadata attached to the record.
        score: Similarity/distance assigned by a query; ``None`` when the
            record came from :meth:`get` rather than :meth:`query`. The sign
            convention (higher-is-better vs lower-is-better) is
            backend-specific — see each store's :meth:`query` docstring.

    """

    id: str
    vector: list[float] | None = None
    document: str | None = None
    metadata: dict[str, Any] | None = None
    score: float | None = None


class BaseVectorStore(ABC):
    """Backend-agnostic interface for a vector store.

    All concrete implementations (Chroma, Qdrant, …) must implement every
    abstract method.  The interface is intentionally synchronous.

    **Async usage:** run store operations via ``asyncio.to_thread()`` at the
    pipeline level rather than adding ``async def`` methods here.  Chroma has
    no native async client, and mixing sync/async in the same base class
    doubles the API surface area.  If a fully-async Qdrant store is needed
    later, create a separate ``AsyncQdrantStore`` that wraps
    ``AsyncQdrantClient`` — same interface, different execution model.

    **Embedding:** pass an *embedder* to let the store turn text into
    vectors itself.  Any :class:`~kb_vectorizer.embedding.interfaces.BaseEmbedder`
    works — e.g. ``SentenceTransformerEmbedder`` or ``CloudEmbedder`` — and
    the same embedder produces identical vectors regardless of which store
    it's attached to.  Passing vectors directly (via ``vectors=`` /
    ``query_vectors=``) always takes priority and never touches the
    embedder.  Whether an embedder is *required* when text is given but no
    vectors are supplied is backend-specific: see each store's docstring.

    **Context manager:** All implementations automatically support the
    ``with`` statement because :meth:`__enter__` and :meth:`__exit__` are
    provided here and delegate to :meth:`close`.
    """

    embedder: BaseEmbedder | None

    def __init__(self, embedder: BaseEmbedder | None = None) -> None:
        """Store the optional embedder shared by :meth:`upsert` and :meth:`query`.

        Args:
            embedder: Used to embed *documents*/*query_texts* whenever
                vectors aren't supplied directly.  Pass ``None`` to require
                pre-computed vectors for every call.

        """
        self.embedder = embedder

    def _resolve_vectors(
        self,
        *,
        texts: Sequence[str] | None,
        vectors: Sequence[Sequence[float]] | None,
    ) -> list[list[float]] | None:
        """Return *vectors* as-is, or embed *texts* via the configured embedder.

        Args:
            texts: Raw strings to embed if *vectors* is ``None``.
            vectors: Pre-computed vectors, which always take priority.

        Returns:
            A list of vectors, or ``None`` if neither *vectors* nor *texts*
            (with an embedder configured) could produce one. Callers decide
            whether ``None`` is an error or a valid backend-native fallback.

        """
        if vectors is not None:
            return [list(v) for v in vectors]
        if texts is not None and self.embedder is not None:
            return self.embedder.embed(list(texts)).vectors
        return None

    # ---- context manager ----

    def __enter__(self) -> BaseVectorStore:
        """Return self so the store can be used as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Call :meth:`close` on exit, regardless of whether an exception occurred."""
        self.close()

    # ---- collection management ----

    @abstractmethod
    def create_collection(self, name: str) -> None:
        """Create the named collection, or do nothing if it already exists.

        Args:
            name: Collection name to create.

        """
        ...

    @abstractmethod
    def get_collection(self, name: str) -> Any:
        """Return the backend-native collection object for *name*.

        Args:
            name: Name of the collection to retrieve.

        Returns:
            The backend's native collection handle (type varies by backend).

        """
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Permanently delete the named collection and all its data.

        Args:
            name: Collection name to delete.

        """
        ...

    # ---- data operations ----

    @abstractmethod
    def upsert(
        self,
        *,
        collection: str,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]] | None = None,
        documents: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update records in *collection*.

        Each positional slot corresponds to one record: ``ids[i]``,
        ``vectors[i]``, ``documents[i]``, and ``metadatas[i]`` all describe
        the same item.

        Args:
            collection: Target collection name.
            ids: Unique application-level IDs for each record.
            vectors: Embedding vectors, one per record.  If ``None`` and
                *documents* is given, the store embeds *documents* itself —
                see the concrete class for what happens without a
                configured embedder.
            documents: Raw text to store alongside each vector.
            metadatas: Arbitrary metadata dicts, one per record.

        """
        ...

    @abstractmethod
    def delete(
        self,
        *,
        collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> None:
        """Delete records from *collection* by ID and/or filter.

        Args:
            collection: Target collection name.
            ids: Delete specific records by their application-level IDs.
            where: A plain ``{field: value}`` metadata filter, translated to
                whatever native filter representation the backend uses.
            where_document: Document content filter (Chroma only).

        """
        ...

    @abstractmethod
    def get(
        self,
        *,
        collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[StoredRecord]:
        """Fetch records from *collection* without a similarity query.

        Args:
            collection: Source collection name.
            ids: Retrieve only these specific IDs.  If ``None``, apply
                *where* and *limit* to browse the full collection.
            where: A plain ``{field: value}`` metadata filter.
            limit: Maximum number of records to return when *ids* is
                ``None``.

        Returns:
            A list of :class:`StoredRecord`, each with ``score=None``.

        """
        ...

    @abstractmethod
    def query(
        self,
        *,
        collection: str,
        query_texts: Sequence[str] | None = None,
        query_vectors: Sequence[Sequence[float]] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[list[StoredRecord]]:
        """Run a nearest-neighbour search against *collection*.

        Exactly one of *query_texts* or *query_vectors* should be provided.
        Passing texts without pre-computed vectors requires the store to
        have an embedder configured — behavior when it doesn't is
        backend-specific, see the concrete class.

        Args:
            collection: Source collection name.
            query_texts: Query strings, embedded via the store's configured
                embedder (or the backend's own default, if it has one).
            query_vectors: Pre-computed query embeddings; takes priority
                over *query_texts* if both are given.
            n_results: Number of nearest neighbours to return per query.
            where: A plain ``{field: value}`` metadata filter.
            where_document: Document content filter (Chroma only).
            include_vectors: Fetch and populate ``StoredRecord.vector`` on
                each hit.  Skipped by default since vectors are the most
                expensive field to transfer.

        Returns:
            One list of :class:`StoredRecord` per query, ordered best match
            first, with ``score`` populated on every record.  If multiple
            queries were given, the outer list has one entry per query, in
            the same order.

        """
        ...

    @abstractmethod
    def count(self, *, collection: str) -> int:
        """Return the total number of records in *collection*.

        Args:
            collection: Target collection name.

        Returns:
            Record count as a non-negative integer.

        """
        ...

    @abstractmethod
    def persist(self) -> None:
        """Flush any in-memory state to durable storage, if applicable.

        Implementations that auto-persist (Qdrant, Chroma HTTP) may treat
        this as a no-op.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this store (connections, file handles, …).

        Called automatically by :meth:`__exit__` when using the store as a
        context manager.
        """
        ...

    # ---- shared presentation helpers ----

    def build_context_window(
        self,
        records: Sequence[StoredRecord],
        *,
        title_field: str = "title",
    ) -> str:
        """Format search results into an LLM-ready context string.

        Most relevant record is placed first, since LLMs attend more
        strongly to content at the start of the context window (the "lost
        in the middle" effect). Shared across every backend — it only reads
        generic :class:`StoredRecord` fields.

        Args:
            records: Results to format, typically one inner list from
                :meth:`query`'s return value, already ordered best-first.
            title_field: Metadata key to pull a human-readable title from,
                if present.

        Returns:
            A formatted multi-line string ready to inject into an LLM
            prompt.

        """
        lines = [f"=== RETRIEVED CONTEXT (top {len(records)}) ===\n"]
        for i, r in enumerate(records, start=1):
            title = (r.metadata or {}).get(title_field, "")
            score_str = f"{r.score:.4f}" if r.score is not None else "n/a"
            header = f"[{i}] id={r.id}"
            if title:
                header += f"  title={title}"
            header += f"  score={score_str}"
            lines.append(f"{header}\n{r.document or ''}\n")
        return "\n".join(lines)

    def print_results(
        self,
        records: Sequence[StoredRecord],
        *,
        label: str = "RESULTS",
        snippet_width: int = 88,
    ) -> None:
        """Pretty-print search results to stdout for interactive debugging.

        Args:
            records: Results to print, typically one inner list from
                :meth:`query`'s return value.
            label: Heading printed above the result list.
            snippet_width: Maximum characters of ``document`` shown per row
                before truncation.

        """
        print(f"\n{'═' * 70}")
        print(f"  {label} — {len(records)} result(s)")
        print("═" * 70)
        for i, r in enumerate(records, start=1):
            snippet = textwrap.shorten(r.document or "", width=snippet_width, placeholder="…")
            score_str = f"{r.score:.4f}" if r.score is not None else "n/a"
            print(f"  [{i}] id={r.id:<20} score={score_str:<10} {snippet}")
        print("═" * 70 + "\n")
