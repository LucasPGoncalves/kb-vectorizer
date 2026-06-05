from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from kb_vectorizer.storage.interfaces import BaseVectorStore, StoredRecord

# Deterministic namespace for converting arbitrary string IDs to UUID5.
# Qdrant point IDs must be unsigned integers or UUID strings.
_KB_NS = uuid.UUID("b3c9d7e8-4f2a-5b6c-8d9e-0a1b2c3d4e5f")

# Payload keys reserved for internal bookkeeping — excluded from user metadata.
_KEY_KB_ID = "_kb_id"
_KEY_KB_DOC = "_kb_document"
_RESERVED = frozenset({_KEY_KB_ID, _KEY_KB_DOC})

_DISTANCE_MAP: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
    "manhattan": Distance.MANHATTAN,
}


def _build_qdrant_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Convert a plain ``{field: value}`` dict into a Qdrant :class:`Filter`.

    Each key/value pair becomes a ``MatchValue`` condition; all conditions are
    combined with ``must`` (logical AND).  The filter is applied at the HNSW
    graph level during search, so it adds negligible latency compared to
    post-processing.

    Args:
        filters: A mapping of payload field names to their required values,
            e.g. ``{"zone": "NORTH", "doc_type": "incident_report"}``.
            Pass ``None`` or an empty dict to match all records.

    Returns:
        A :class:`qdrant_client.models.Filter` with one ``FieldCondition``
        per entry, or ``None`` if *filters* is empty or ``None``.

    Examples:
        >>> _build_qdrant_filter({"zone": "NORTH"})
        Filter(must=[FieldCondition(key='zone', match=MatchValue(value='NORTH'))])

        >>> _build_qdrant_filter(None)
        None

    """
    if not filters:
        return None
    return Filter(
        must=[
            FieldCondition(key=field, match=MatchValue(value=value))
            for field, value in filters.items()
        ]
    )


def _to_point_id(kb_id: str) -> str:
    """Convert an arbitrary string ID to a deterministic UUID5 string.

    Qdrant requires point IDs to be unsigned integers or UUID-formatted strings.
    This function maps any string to a stable UUID so callers can use their
    own opaque IDs (e.g. ``"doc-1:000042"``) without collisions.

    Args:
        kb_id: The application-level record identifier.

    Returns:
        A UUID5 string derived from *kb_id*.

    """
    return str(uuid.uuid5(_KB_NS, kb_id))


class QdrantStore(BaseVectorStore):
    """Thin adapter over an injected :class:`qdrant_client.QdrantClient`.

    The client is injected at construction time so the store works with every
    Qdrant deployment mode:

    - ``QdrantClient(":memory:")``           — in-process ephemeral
    - ``QdrantClient(path="…")``             — on-disk persistence
    - ``QdrantClient(url="…", api_key="…")`` — remote server / Qdrant Cloud

    Use :func:`~kb_vectorizer.storage.qdrant_client_factory.make_qdrant_client`
    to construct the appropriate client from a config string.

    **ID handling:** Qdrant requires point IDs to be UUIDs or unsigned
    integers.  Arbitrary application IDs (e.g. ``"doc-1:000042"``) are
    converted to UUID5 hashes internally.  The original ID is stored in the
    payload under ``"_kb_id"`` and restored transparently on every read.

    **Text queries:** Qdrant is a pure vector store — it has no built-in
    embedding function.  Passing *query_texts* to :meth:`query` raises
    ``ValueError``; callers must pre-embed texts and pass *query_vectors*.

    **Distance scores:** :meth:`query` returns ``1.0 − score`` in the
    ``"distances"`` field (Chroma-compatible convention: lower = more
    similar).  This is accurate for cosine similarity; for Euclidean or dot
    metrics the value is still monotonically ordered but not a true distance.

    Args:
        client: An instantiated :class:`qdrant_client.QdrantClient`.
        vector_size: Dimensionality of the vectors stored in this instance.
            Required when creating new collections.
        distance: Distance metric to use for new collections.  One of
            ``"cosine"`` (default), ``"dot"``, ``"euclid"``, ``"manhattan"``.

    Raises:
        ValueError: If *distance* is not a recognised metric name.

    """

    def __init__(
        self,
        client: QdrantClient,
        vector_size: int,
        distance: str = "cosine",
    ) -> None:
        """Initialise the store with a Qdrant client and vector configuration.

        Args:
            client: An instantiated :class:`qdrant_client.QdrantClient`.
            vector_size: Number of dimensions in the stored vectors.
            distance: Similarity metric for new collections.  Accepted values:
                ``"cosine"``, ``"dot"``, ``"euclid"``, ``"manhattan"``.

        Raises:
            ValueError: If *distance* is not recognised.

        """
        dist = _DISTANCE_MAP.get(distance.lower())
        if dist is None:
            raise ValueError(
                f"Unknown distance '{distance}'. Choose: {list(_DISTANCE_MAP)}."
            )
        self._client = client
        self._vector_size = vector_size
        self._distance = dist
        self._closed = False

    # ---- collection management ----

    def create_collection(self, name: str) -> None:
        """Create *name* if it does not already exist.

        Args:
            name: Collection name to create.

        """
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )

    def get_collection(self, name: str) -> Any:
        """Return the Qdrant ``CollectionInfo`` for *name*.

        Args:
            name: Collection name.

        Returns:
            A ``qdrant_client.models.CollectionInfo`` object.

        """
        return self._client.get_collection(collection_name=name)

    def delete_collection(self, name: str) -> None:
        """Permanently delete the named collection and all its data.

        Args:
            name: Collection name to delete.

        """
        self._client.delete_collection(collection_name=name)

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

        Application-level IDs are stored in payload under ``"_kb_id"``; they
        are restored transparently on reads.  Document text is stored under
        ``"_kb_document"``.

        Args:
            collection: Target collection name.
            ids: Unique application-level IDs.
            vectors: Embedding vectors, one per record.  If ``None``, a
                zero vector of length ``vector_size`` is stored — not useful
                for search, but lets you inspect payloads without embeddings.
            documents: Raw text content, one per record.
            metadatas: Metadata dicts, one per record.

        """
        points: list[PointStruct] = []
        for i, kb_id in enumerate(ids):
            payload: dict[str, Any] = {_KEY_KB_ID: kb_id}
            if documents:
                payload[_KEY_KB_DOC] = documents[i]
            if metadatas and metadatas[i]:
                payload.update(metadatas[i])
            vec = list(vectors[i]) if vectors else [0.0] * self._vector_size
            points.append(
                PointStruct(id=_to_point_id(kb_id), vector=vec, payload=payload)
            )
        self._client.upsert(collection_name=collection, points=points)

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
            where: A ``qdrant_client.models.Filter`` object for payload-based
                deletion.  Raw dicts are not supported by Qdrant.
            where_document: Not supported by Qdrant — raises ``ValueError``.

        Raises:
            ValueError: If *where_document* is provided.

        """
        if where_document is not None:
            raise ValueError("QdrantStore does not support where_document filters.")
        if ids is not None:
            self._client.delete(
                collection_name=collection,
                points_selector=PointIdsList(
                    points=[_to_point_id(i) for i in ids]
                ),
            )
        elif where is not None:
            self._client.delete(
                collection_name=collection,
                points_selector=FilterSelector(filter=_build_qdrant_filter(where)),
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
            ids: Retrieve only these specific application-level IDs.
            where: A ``qdrant_client.models.Filter`` for payload filtering
                when *ids* is ``None``.
            limit: Maximum records returned when scrolling (default 100).

        Returns:
            A list of :class:`~kb_vectorizer.storage.interfaces.StoredRecord`.

        """
        if ids is not None:
            points = self._client.retrieve(
                collection_name=collection,
                ids=[_to_point_id(i) for i in ids],
                with_vectors=True,
                with_payload=True,
            )
        else:
            scroll_result, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=_build_qdrant_filter(where),
                limit=limit or 100,
                with_vectors=True,
                with_payload=True,
            )
            points = scroll_result
        return [_build_record(p) for p in points]

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

        The response format mirrors Chroma's for drop-in interoperability:
        ``"ids"``, ``"documents"``, ``"metadatas"``, ``"distances"``, and
        ``"embeddings"`` — each a list-of-lists (one inner list per query
        vector).

        Args:
            collection: Source collection name.
            query_texts: Not supported — raises ``ValueError``.  Pre-embed
                your texts and pass *query_vectors* instead.
            query_vectors: Pre-computed query embeddings, one per query.
            n_results: Number of nearest neighbours per query.
            where: A ``qdrant_client.models.Filter`` for payload filtering.
            where_document: Not supported — raises ``ValueError``.
            include: Fields to include.  ``"embeddings"`` fetches stored
                vectors; all others control payload fields in the response.

        Returns:
            A dict with keys from *include*, each mapping to a list-of-lists.

        Raises:
            ValueError: If *query_texts* or *where_document* is provided.
            ValueError: If neither *query_texts* nor *query_vectors* is given.

        """
        if query_texts is not None:
            raise ValueError(
                "QdrantStore does not support query_texts. "
                "Pre-embed your texts and pass query_vectors instead."
            )
        if not query_vectors:
            raise ValueError(
                "query_vectors must be provided for QdrantStore."
            )
        if where_document is not None:
            raise ValueError("QdrantStore does not support where_document filters.")

        inc = set(include)
        with_vectors = "embeddings" in inc

        all_ids: list[list[str]] = []
        all_docs: list[list[str | None]] = []
        all_metas: list[list[dict[str, Any] | None]] = []
        all_dists: list[list[float]] = []
        all_embeds: list[list[list[float]] | None] = []

        for qvec in query_vectors:
            response = self._client.query_points(
                collection_name=collection,
                query=list(qvec),
                limit=n_results,
                query_filter=_build_qdrant_filter(where),
                with_vectors=with_vectors,
                with_payload=True,
            )
            hits = response.points

            row_ids, row_docs, row_metas, row_dists, row_embeds = [], [], [], [], []
            for pt in hits:
                payload = pt.payload or {}
                row_ids.append(payload.get(_KEY_KB_ID, str(pt.id)))
                row_docs.append(payload.get(_KEY_KB_DOC))
                meta = {k: v for k, v in payload.items() if k not in _RESERVED}
                row_metas.append(meta or None)
                # 1.0 - score converts cosine similarity to Chroma-style distance.
                row_dists.append(1.0 - pt.score)
                if with_vectors:
                    vec = pt.vector
                    row_embeds.append(list(vec) if isinstance(vec, (list, tuple)) else None)

            all_ids.append(row_ids)
            all_docs.append(row_docs)
            all_metas.append(row_metas)
            all_dists.append(row_dists)
            all_embeds.append(row_embeds if with_vectors else None)

        result: dict[str, Any] = {"ids": all_ids}
        if "documents" in inc:
            result["documents"] = all_docs
        if "metadatas" in inc:
            result["metadatas"] = all_metas
        if "distances" in inc:
            result["distances"] = all_dists
        if "embeddings" in inc:
            result["embeddings"] = all_embeds
        return result

    def count(self, *, collection: str) -> int:
        """Return the total number of records in *collection*.

        Args:
            collection: Target collection name.

        Returns:
            Exact record count.

        """
        return self._client.count(collection_name=collection, exact=True).count

    def persist(self) -> None:
        """No-op — Qdrant handles persistence automatically."""

    def close(self) -> None:
        """Close the underlying Qdrant client connection.

        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        self._client.close()


def _build_record(point: Any) -> StoredRecord:
    """Construct a :class:`StoredRecord` from a Qdrant ``Record`` or ``ScoredPoint``.

    Args:
        point: A Qdrant point object with ``.id``, ``.vector``, and
            ``.payload`` attributes.

    Returns:
        A :class:`~kb_vectorizer.storage.interfaces.StoredRecord` with the
        original application ID restored from the payload.

    """
    payload = point.payload or {}
    original_id = payload.get(_KEY_KB_ID, str(point.id))
    document = payload.get(_KEY_KB_DOC)
    meta = {k: v for k, v in payload.items() if k not in _RESERVED}
    vec = point.vector
    vector = list(vec) if isinstance(vec, (list, tuple)) else None
    return StoredRecord(
        id=original_id,
        vector=vector,
        document=document,
        metadata=meta or None,
    )
