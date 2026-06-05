from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class StoredRecord:
    """A single record returned from a vector store.

    Attributes:
        id: The original application-level identifier for this record.
        vector: The embedding vector, if the store was asked to return it.
        document: The raw text content stored alongside the vector.
        metadata: Arbitrary key-value metadata attached to the record.

    """

    id: str
    vector: list[float] | None = None
    document: str | None = None
    metadata: dict[str, Any] | None = None


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

    **Context manager:** All implementations automatically support the
    ``with`` statement because :meth:`__enter__` and :meth:`__exit__` are
    provided here and delegate to :meth:`close`.
    """

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
            vectors: Embedding vectors, one per record.  Pass ``None`` if the
                backend can embed documents itself (e.g. Chroma with an
                embedding function configured).
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
            where: Backend-specific metadata filter.  For Chroma this is a
                ``{"field": {"$eq": value}}`` dict; for Qdrant pass a
                ``qdrant_client.models.Filter`` object.
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
            where: Backend-specific metadata filter.
            limit: Maximum number of records to return when *ids* is
                ``None``.

        Returns:
            A list of :class:`StoredRecord` objects.

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
        include: Sequence[str] = ("metadatas", "documents", "distances", "embeddings"),
    ) -> dict[str, Any]:
        """Run a nearest-neighbour search against *collection*.

        Exactly one of *query_texts* or *query_vectors* must be provided.
        The response format mirrors Chroma's: keys are ``"ids"``,
        ``"documents"``, ``"metadatas"``, ``"distances"``, and
        ``"embeddings"``, each a list-of-lists (one inner list per query).

        Args:
            collection: Source collection name.
            query_texts: Query strings to embed and search (backend must
                have an embedding function configured).
            query_vectors: Pre-computed query embeddings.
            n_results: Number of nearest neighbours to return per query.
            where: Backend-specific metadata filter.
            where_document: Document content filter (Chroma only).
            include: Fields to include in the response.

        Returns:
            A ``dict`` whose keys are the field names requested in *include*,
            each mapping to a list-of-lists (one per query).

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
