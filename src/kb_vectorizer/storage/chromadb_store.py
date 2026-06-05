from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kb_vectorizer.storage.interfaces import BaseVectorStore, StoredRecord


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

    Args:
        client: Any Chroma client instance.

    """

    def __init__(self, client: Any) -> None:
        """Initialise the store with an already-constructed Chroma client.

        Args:
            client: An instantiated Chroma client (EphemeralClient,
                PersistentClient, HttpClient, or CloudClient).

        """
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
            vectors: Pre-computed embeddings.  Pass ``None`` to let Chroma
                embed *documents* using its configured embedding function.
            documents: Raw text content, one per record.
            metadatas: Metadata dicts, one per record.

        """
        col = self.get_collection(collection)
        col.upsert(
            ids=list(ids),
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
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
            A list of :class:`~kb_vectorizer.storage.interfaces.StoredRecord`.

        """
        col = self.get_collection(collection)
        res = col.get(
            ids=list(ids) if ids else None,
            where=where,
            limit=limit,
        )
        out: list[StoredRecord] = []
        for i, record_id in enumerate(res.get("ids", [])):
            out.append(
                StoredRecord(
                    id=record_id,
                    vector=(res["embeddings"][i] if res.get("embeddings") else None),
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
        include: Sequence[str] = ("metadatas", "documents", "distances", "embeddings"),
    ) -> dict[str, Any]:
        """Run a nearest-neighbour search against *collection*.

        Passes *query_texts* or *query_vectors* directly to Chroma's
        ``Collection.query``.  When *query_texts* is used, Chroma embeds them
        server-side with the collection's configured embedding function.

        Args:
            collection: Source collection name.
            query_texts: Query strings for text-based search.
            query_vectors: Pre-computed query embeddings.
            n_results: Number of nearest neighbours per query.
            where: Chroma metadata filter.
            where_document: Chroma document content filter.
            include: Fields to include in the response.

        Returns:
            Chroma's raw query response dict with keys ``"ids"``,
            ``"documents"``, ``"metadatas"``, ``"distances"``, and
            optionally ``"embeddings"``.

        """
        col = self.get_collection(collection)
        return col.query(
            query_texts=list(query_texts) if query_texts else None,
            query_embeddings=list(query_vectors) if query_vectors else None,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=list(include),
        )

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
