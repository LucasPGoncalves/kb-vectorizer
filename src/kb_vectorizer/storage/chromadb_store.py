from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kb_vectorizer.embedding.interfaces import BaseEmbedder
from kb_vectorizer.storage.interfaces import BaseVectorStore, StoredRecord

# Fields always requested from Chroma so returned StoredRecords are fully
# populated. "embeddings" is added on top of this only when include_vectors=True,
# since it's the one field expensive enough to skip by default.
_BASE_INCLUDE = ["metadatas", "documents"]


class ChromaStore(BaseVectorStore):
    """Thin adapter over an injected Chroma client.

    The client is injected at construction time so the store is portable
    across every Chroma deployment mode:

    - ``chromadb.EphemeralClient()``          — in-memory, no persistence
    - ``chromadb.PersistentClient(path=…)``   — on-disk persistence
    - ``chromadb.HttpClient(host=…, port=…)`` — standalone server / Docker
    - ``chromadb.CloudClient()``              — Chroma Cloud

    Use :func:`~kb_vectorizer.storage.chroma_client_factory.make_chroma_client`
    to construct the appropriate client from a config string.

    **Embedding:** pass *embedder* to control exactly which model turns text
    into vectors (e.g. ``SentenceTransformerEmbedder`` for any HuggingFace
    sentence-transformers checkpoint).  When *embedder* is ``None`` and
    *documents*/*query_texts* are given without vectors, this store falls
    back to Chroma's own built-in default embedding function (an ONNX
    MiniLM model) — Chroma's native behavior, unchanged from a plain
    ``chromadb.Collection``.

    Args:
        client: Any Chroma client instance.
        embedder: Optional embedder used instead of Chroma's built-in default.

    """

    def __init__(self, client: Any, embedder: BaseEmbedder | None = None) -> None:
        """Initialise the store with an already-constructed Chroma client.

        Args:
            client: An instantiated Chroma client (EphemeralClient,
                PersistentClient, HttpClient, or CloudClient).
            embedder: Optional embedder for text-in calls. If omitted,
                Chroma's own default embedding function is used whenever
                vectors aren't supplied directly.

        """
        super().__init__(embedder=embedder)
        self._client = client
        self._closed = False

    # ---- collection management ----

    def create_collection(self, name: str) -> None:
        """Create *name* if it does not already exist.

        Uses ``get_or_create_collection`` so the call is idempotent.

        Args:
            name: Collection name to create.

        """
        self._client.get_or_create_collection(name=name)

    def get_collection(self, name: str) -> Any:
        """Return (or lazily create) the Chroma collection for *name*.

        Args:
            name: Collection name.

        Returns:
            A ``chromadb.Collection`` object.

        """
        return self._client.get_or_create_collection(name=name)

    def delete_collection(self, name: str) -> None:
        """Permanently delete the named collection and all its data.

        Args:
            name: Collection name to delete.

        """
        self._client.delete_collection(name)

    # ---- data operations ----

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

        Calls Chroma's ``upsert`` (not ``add``) to ensure idempotency on
        reruns — existing records are updated rather than rejected.

        Args:
            collection: Target collection name.
            ids: Unique IDs for each record.
            vectors: Pre-computed embeddings.  If ``None`` and *documents*
                is given, embeds via the configured *embedder*; with no
                embedder configured, Chroma's own default embedding
                function embeds *documents* instead.
            documents: Raw text content, one per record.
            metadatas: Metadata dicts, one per record.

        """
        resolved = self._resolve_vectors(texts=documents, vectors=vectors)
        col = self.get_collection(collection)
        col.upsert(
            ids=list(ids),
            embeddings=resolved,
            documents=list(documents) if documents else None,
            metadatas=list(metadatas) if metadatas else None,
        )

    def delete(
        self,
        *,
        collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> None:
        """Delete records from *collection* by ID and/or metadata filter.

        Args:
            collection: Target collection name.
            ids: Delete specific records by ID.
            where: Chroma metadata filter expression, e.g.
                ``{"source": {"$eq": "wiki"}}``.
            where_document: Chroma document content filter, e.g.
                ``{"$contains": "keyword"}``.

        """
        col = self.get_collection(collection)
        col.delete(
            ids=list(ids) if ids else None,
            where=where,
            where_document=where_document,
        )

    def get(
        self,
        *,
        collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[StoredRecord]:
        """Fetch records without a similarity search.

        Args:
            collection: Source collection name.
            ids: Retrieve only these specific record IDs.
            where: Chroma metadata filter.
            limit: Maximum records to return when *ids* is ``None``.

        Returns:
            A list of :class:`~kb_vectorizer.storage.interfaces.StoredRecord`,
            each with ``vector`` populated and ``score=None``.

        """
        col = self.get_collection(collection)
        res = col.get(
            ids=list(ids) if ids else None,
            where=where,
            limit=limit,
            include=["metadatas", "documents", "embeddings"],
        )
        out: list[StoredRecord] = []
        for i, record_id in enumerate(res.get("ids", [])):
            out.append(
                StoredRecord(
                    id=record_id,
                    vector=(res["embeddings"][i] if res.get("embeddings") is not None else None),
                    document=(res["documents"][i] if res.get("documents") else None),
                    metadata=(res["metadatas"][i] if res.get("metadatas") else None),
                )
            )
        return out

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

        If *query_vectors* is omitted and *query_texts* is given, embeds via
        the configured *embedder*; with no embedder configured, Chroma's own
        default embedding function embeds the query texts instead.

        Args:
            collection: Source collection name.
            query_texts: Query strings for text-based search.
            query_vectors: Pre-computed query embeddings.
            n_results: Number of nearest neighbours per query.
            where: Chroma metadata filter.
            where_document: Chroma document content filter.
            include_vectors: Fetch and populate ``StoredRecord.vector`` on
                each hit.

        Returns:
            One list of :class:`StoredRecord` per query, ordered nearest
            first, with ``score`` set to Chroma's native distance
            (**lower is more similar**).

        """
        resolved = self._resolve_vectors(texts=query_texts, vectors=query_vectors)
        include = [*_BASE_INCLUDE, "distances"]
        if include_vectors:
            include.append("embeddings")

        col = self.get_collection(collection)
        res = col.query(
            query_texts=list(query_texts) if resolved is None and query_texts else None,
            query_embeddings=resolved,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )

        results: list[list[StoredRecord]] = []
        ids_rows = res.get("ids") or []
        for row_idx, row_ids in enumerate(ids_rows):
            row: list[StoredRecord] = []
            for col_idx, record_id in enumerate(row_ids):
                embeddings = res.get("embeddings")
                row.append(
                    StoredRecord(
                        id=record_id,
                        vector=(embeddings[row_idx][col_idx] if embeddings is not None else None),
                        document=(res["documents"][row_idx][col_idx] if res.get("documents") else None),
                        metadata=(res["metadatas"][row_idx][col_idx] if res.get("metadatas") else None),
                        score=(res["distances"][row_idx][col_idx] if res.get("distances") else None),
                    )
                )
            results.append(row)
        return results

    def count(self, *, collection: str) -> int:
        """Return the total number of records in *collection*.

        Args:
            collection: Target collection name.

        Returns:
            Record count.

        """
        return self.get_collection(collection).count()

    def persist(self) -> None:
        """Flush in-memory state to disk for ``PersistentClient``.

        This is a no-op for ``EphemeralClient`` and ``HttpClient``.
        """
        if hasattr(self._client, "persist"):
            self._client.persist()

    def close(self) -> None:
        """Flush pending writes and release the client reference.

        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.persist()
        finally:
            self._client = None  # type: ignore[assignment]
