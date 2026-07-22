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
    Modifier,
    PointIdsList,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from kb_vectorizer.embedding.interfaces import BaseEmbedder
from kb_vectorizer.storage.interfaces import BaseVectorStore, StoredRecord
from kb_vectorizer.text.tokenizer import term_frequency_vector

# Deterministic namespace for converting arbitrary string IDs to UUID5.
# Qdrant point IDs must be unsigned integers or UUID strings, so an
# application ID like "doc-1:000042" cannot be used directly.
_KB_NS = uuid.UUID("b3c9d7e8-4f2a-5b6c-8d9e-0a1b2c3d4e5f")

# The only payload key this store reserves for itself: the original
# application-level ID, needed to translate back from the internal UUID5.
# Everything else in payload is the caller's data, untouched.
_KEY_KB_ID = "_kb_id"

# Name of the named sparse vector field used for native BM25-style search
# when enable_bm25=True. The dense vector stays unnamed/default either way.
_SPARSE_VECTOR_NAME = "bm25"

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


def _build_record(point: Any, score: float | None = None) -> StoredRecord:
    """Construct a :class:`StoredRecord` from a Qdrant ``Record`` or ``ScoredPoint``.

    Qdrant has no first-class "document" concept — payload is just JSON.  So
    ``document`` is always ``None`` here; if the caller stored text, it lives
    in ``metadata`` like any other field.

    Args:
        point: A Qdrant point object with ``.id``, ``.vector``, and
            ``.payload`` attributes.
        score: Similarity score to attach, for query hits. ``None`` for
            plain ``get``/``scroll`` results.

    Returns:
        A :class:`~kb_vectorizer.storage.interfaces.StoredRecord` with the
        original application ID restored from the payload.

    """
    payload = point.payload or {}
    original_id = payload.get(_KEY_KB_ID, str(point.id))
    metadata = {k: v for k, v in payload.items() if k != _KEY_KB_ID}
    vec = point.vector
    vector = list(vec) if isinstance(vec, (list, tuple)) else None
    return StoredRecord(
        id=original_id,
        vector=vector,
        document=None,
        metadata=metadata or None,
        score=score,
    )


class QdrantStore(BaseVectorStore):
    """Qdrant-native adapter over an injected :class:`qdrant_client.QdrantClient`.

    This store is built around Qdrant's own model — points with a vector and
    a JSON payload — rather than imitating another backend's conventions.
    It satisfies :class:`~kb_vectorizer.storage.interfaces.BaseVectorStore`
    and returns :class:`~kb_vectorizer.storage.interfaces.StoredRecord`
    objects, but the *values* inside those objects reflect what Qdrant
    actually returns.

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
    payload under the single reserved key ``"_kb_id"`` and restored
    transparently on every read.

    **Payload, not document/metadata:** Qdrant payload is one flat JSON
    object — there's no separate "document" slot.  ``upsert(documents=...)``
    merges the text into payload under a plain ``"document"`` key (no special
    treatment), and reads never split it back out: :attr:`StoredRecord.document`
    is always ``None``, and any text you stored is simply a field inside
    :attr:`StoredRecord.metadata`, exactly as you'd find it in the payload.

    **Scores:** :meth:`query` populates ``StoredRecord.score`` with Qdrant's
    native similarity score — **higher is more similar**, the opposite
    convention from Chroma's distance (lower is more similar). Do not assume
    one backend's ordering convention when swapping stores.

    **Embedding:** unlike Chroma, a plain ``QdrantClient`` has no built-in
    embedding capability (that would require the separate FastEmbed
    integration, which this store does not use, to keep model choice
    identical to :class:`~kb_vectorizer.storage.chromadb_store.ChromaStore`).
    Pass *embedder* to enable ``documents=``/``query_texts=`` calls; without
    one, those calls raise ``ValueError`` and you must supply
    pre-computed vectors instead.

    **Native BM25 keyword search:** pass ``enable_bm25=True`` to also
    configure a named sparse vector field (with Qdrant's ``Modifier.IDF``)
    on every collection this store creates. When enabled, every
    :meth:`upsert` call with *documents* computes a term-frequency sparse
    vector via :func:`~kb_vectorizer.text.tokenizer.term_frequency_vector`
    and attaches it to the same point as the dense vector — no separate
    indexing step. Qdrant then maintains corpus-wide IDF statistics
    incrementally, server-side, so — unlike an in-memory ``BM25Okapi``
    index — the client never needs to hold the whole corpus in memory at
    once. Call :meth:`keyword_search` to search it directly, or wrap this
    store in :class:`~kb_vectorizer.retrieval.native_keyword_index.NativeKeywordIndex`
    to use it through the same interface as
    :class:`~kb_vectorizer.retrieval.inmemory_keyword_index.InMemoryKeywordIndex`.

    Args:
        client: An instantiated :class:`qdrant_client.QdrantClient`.
        vector_size: Dimensionality of the vectors stored in this instance.
            Required when creating new collections.
        distance: Distance metric to use for new collections.  One of
            ``"cosine"`` (default), ``"dot"``, ``"euclid"``, ``"manhattan"``.
        embedder: Optional embedder used to turn *documents*/*query_texts*
            into vectors. Required for text-in calls, since Qdrant itself
            has no built-in embedding here.
        enable_bm25: Configure a native sparse-vector BM25 field on every
            collection this store creates, and populate it automatically
            from *documents* on every :meth:`upsert`.

    Raises:
        ValueError: If *distance* is not a recognised metric name.

    """

    def __init__(
        self,
        client: QdrantClient,
        vector_size: int,
        distance: str = "cosine",
        embedder: BaseEmbedder | None = None,
        enable_bm25: bool = False,
    ) -> None:
        """Initialise the store with a Qdrant client and vector configuration.

        Args:
            client: An instantiated :class:`qdrant_client.QdrantClient`.
            vector_size: Number of dimensions in the stored vectors.
            distance: Similarity metric for new collections.  Accepted values:
                ``"cosine"``, ``"dot"``, ``"euclid"``, ``"manhattan"``.
            embedder: Optional embedder for text-in calls.
            enable_bm25: Configure and maintain a native sparse-vector BM25
                field alongside the dense vector.

        Raises:
            ValueError: If *distance* is not recognised.

        """
        dist = _DISTANCE_MAP.get(distance.lower())
        if dist is None:
            raise ValueError(
                f"Unknown distance '{distance}'. Choose: {list(_DISTANCE_MAP)}."
            )
        super().__init__(embedder=embedder)
        self._client = client
        self._vector_size = vector_size
        self._distance = dist
        self._enable_bm25 = enable_bm25
        self._closed = False

    # ---- collection management ----

    def create_collection(self, name: str) -> None:
        """Create *name* if it does not already exist.

        When ``enable_bm25=True`` was passed to the constructor, also
        configures a named sparse vector field (``"bm25"``) with Qdrant's
        IDF modifier, so the collection supports native keyword search
        immediately.

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
                sparse_vectors_config=(
                    {_SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)}
                    if self._enable_bm25
                    else None
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

        Each record's payload is built from *metadatas* (if given) plus a
        plain ``"document"`` field (if *documents* is given) — a flat JSON
        object, Qdrant-style.  The application ID is added last, under the
        reserved key ``"_kb_id"``, so it always wins over a same-named
        metadata field.

        When ``enable_bm25=True`` was passed to the constructor and
        *documents* is given, each record also gets a term-frequency sparse
        vector attached automatically — no separate keyword-indexing call
        needed.

        Args:
            collection: Target collection name.
            ids: Unique application-level IDs.
            vectors: Pre-computed embeddings.  If ``None`` and *documents*
                is given, embeds via the configured *embedder*.
            documents: Raw text content, one per record, stored under the
                payload key ``"document"``. Also used to compute the sparse
                BM25 vector when ``enable_bm25=True``.
            metadatas: Metadata dicts, one per record.

        Raises:
            ValueError: If vectors can't be resolved — *vectors* is
                ``None`` and either *documents* is ``None`` too, or no
                embedder is configured.

        """
        resolved = self._resolve_vectors(texts=documents, vectors=vectors)
        if resolved is None:
            raise ValueError(
                "QdrantStore.upsert requires vectors, or documents plus a "
                "configured embedder (pass embedder=... to the constructor)."
            )
        points: list[PointStruct] = []
        for i, kb_id in enumerate(ids):
            payload: dict[str, Any] = {}
            if metadatas and metadatas[i]:
                payload.update(metadatas[i])
            if documents:
                payload["document"] = documents[i]
            payload[_KEY_KB_ID] = kb_id

            vector: dict[str, Any] | list[float] = list(resolved[i])
            if self._enable_bm25:
                vector = {"": list(resolved[i])}
                if documents:
                    tf = term_frequency_vector(documents[i])
                    if tf:
                        vector[_SPARSE_VECTOR_NAME] = SparseVector(
                            indices=list(tf.keys()), values=list(tf.values())
                        )

            points.append(PointStruct(id=_to_point_id(kb_id), vector=vector, payload=payload))
        self._client.upsert(collection_name=collection, points=points)

    def delete(
        self,
        *,
        collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> None:
        """Delete records from *collection* by ID and/or payload filter.

        Args:
            collection: Target collection name.
            ids: Delete specific records by their application-level IDs.
            where: A plain ``{field: value}`` dict, converted to a native
                Qdrant :class:`~qdrant_client.models.Filter` via
                :func:`_build_qdrant_filter`.
            where_document: Not a Qdrant concept — raises ``ValueError``.

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
            where: A plain ``{field: value}`` dict, converted to a native
                Qdrant :class:`~qdrant_client.models.Filter`, used only when
                *ids* is ``None``.
            limit: Maximum records returned when scrolling (default 100).

        Returns:
            A list of :class:`~kb_vectorizer.storage.interfaces.StoredRecord`.
            ``document`` is always ``None``; stored text (if any) appears in
            ``metadata`` instead.

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
        include_vectors: bool = False,
    ) -> list[list[StoredRecord]]:
        """Run a nearest-neighbour search against *collection*.

        Args:
            collection: Source collection name.
            query_texts: Query strings.  Requires a configured embedder —
                raises ``ValueError`` otherwise.
            query_vectors: Pre-computed query embeddings, one per query.
                Takes priority over *query_texts* if both are given.
            n_results: Number of nearest neighbours per query.
            where: A plain ``{field: value}`` dict, converted to a native
                Qdrant :class:`~qdrant_client.models.Filter`.
            where_document: Not a Qdrant concept — raises ``ValueError``.
            include_vectors: Fetch and populate ``StoredRecord.vector`` on
                each hit.

        Returns:
            One list of :class:`StoredRecord` per query vector, ordered
            most-similar first, with ``score`` set to Qdrant's native
            similarity (**higher is more similar**).

        Raises:
            ValueError: If *where_document* is provided.
            ValueError: If vectors can't be resolved — *query_vectors* is
                ``None`` and either *query_texts* is ``None`` too, or no
                embedder is configured.

        """
        if where_document is not None:
            raise ValueError("QdrantStore does not support where_document filters.")

        resolved = self._resolve_vectors(texts=query_texts, vectors=query_vectors)
        if resolved is None:
            raise ValueError(
                "QdrantStore.query requires query_vectors, or query_texts "
                "plus a configured embedder (pass embedder=... to the constructor)."
            )

        results: list[list[StoredRecord]] = []
        for qvec in resolved:
            response = self._client.query_points(
                collection_name=collection,
                query=list(qvec),
                limit=n_results,
                query_filter=_build_qdrant_filter(where),
                with_vectors=include_vectors,
                with_payload=True,
            )
            results.append([_build_record(pt, score=pt.score) for pt in response.points])
        return results

    def keyword_search(
        self, *, collection: str, query_text: str, k: int = 50
    ) -> list[StoredRecord]:
        """Run a native BM25-style keyword search against *collection*'s sparse vector field.

        Tokenizes *query_text* into term frequencies the same way
        :meth:`upsert` does for documents, then searches the sparse vector
        field — Qdrant applies its IDF modifier and length normalization
        server-side, using corpus statistics it has maintained
        incrementally since ingestion.

        Implements :class:`~kb_vectorizer.retrieval.interfaces.SupportsKeywordSearch`,
        so this store can be wrapped directly in
        :class:`~kb_vectorizer.retrieval.native_keyword_index.NativeKeywordIndex`.

        Args:
            collection: Source collection name.
            query_text: Raw query string.
            k: Maximum number of hits to return.

        Returns:
            Matches ordered best-first (highest score first), with ``score``
            set to Qdrant's native BM25 relevance score. Empty if
            *query_text* tokenizes to no terms.

        Raises:
            ValueError: If ``enable_bm25=False`` on this store instance.

        """
        if not self._enable_bm25:
            raise ValueError(
                "keyword_search requires enable_bm25=True on this QdrantStore instance."
            )
        tf = term_frequency_vector(query_text)
        if not tf:
            return []
        sparse_query = SparseVector(indices=list(tf.keys()), values=list(tf.values()))
        response = self._client.query_points(
            collection_name=collection,
            query=sparse_query,
            using=_SPARSE_VECTOR_NAME,
            limit=k,
            with_payload=True,
        )
        return [_build_record(pt, score=pt.score) for pt in response.points]

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
