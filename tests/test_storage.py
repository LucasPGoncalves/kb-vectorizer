"""Unit tests for the kb_vectorizer.storage module.

Covers:
  - StoredRecord                          (interfaces.py)
  - BaseVectorStore                       (interfaces.py) — context manager,
    embedder resolution, build_context_window, print_results
  - ChromaStore                           (chromadb_store.py)
  - make_chroma_client                    (chroma_client_factory.py)
  - QdrantStore                           (qdrant_store.py)
  - make_qdrant_client                    (qdrant_client_factory.py)

All tests run fully in-memory — no external servers, no real ML models
(a deterministic FakeEmbedder stands in for SentenceTransformerEmbedder /
CloudEmbedder so embedder-integration tests stay fast and network-free).
"""

from __future__ import annotations

import chromadb
import pytest
from qdrant_client import QdrantClient

from kb_vectorizer.embedding.interfaces import BaseEmbedder, EmbedResponse
from kb_vectorizer.storage import ChromaStore, StoredRecord
from kb_vectorizer.storage.chroma_client_factory import make_chroma_client
from kb_vectorizer.storage.qdrant_client_factory import make_qdrant_client
from kb_vectorizer.storage.qdrant_store import QdrantStore

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

COLLECTION = "test_collection"
DIM = 3

# Three orthogonal unit vectors — cosine similarity between any distinct pair = 0
V1 = [1.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0]
V3 = [0.0, 0.0, 1.0]

# Maps known strings to fixed vectors, so FakeEmbedder-driven tests can make
# precise similarity assertions without any real embedding model.
_TEXT_TO_VECTOR = {
    "vector one": V1,
    "vector two": V2,
    "vector three": V3,
}


class FakeEmbedder(BaseEmbedder):
    """Deterministic test double for BaseEmbedder — no ML model involved.

    Looks up each text in a fixed table; unknown text embeds to the zero
    vector. Stands in for SentenceTransformerEmbedder/CloudEmbedder in tests
    that only need to prove the store <-> embedder wiring works.
    """

    def __init__(self) -> None:
        self.model_name = "fake-embedder"
        self.max_batch_size = 64
        self.dimension = DIM

    def embed(self, texts: list[str]) -> EmbedResponse:
        vectors = [_TEXT_TO_VECTOR.get(t, [0.0, 0.0, 0.0]) for t in texts]
        return EmbedResponse(vectors=vectors, model=self.model_name, dimension=DIM)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def chroma():
    """ChromaStore backed by a guaranteed-empty test collection.

    chromadb.EphemeralClient() is a singleton in v1 — state persists across
    fixture invocations.  We delete then recreate the collection to ensure
    every test starts with zero records.
    """
    store = ChromaStore(chromadb.EphemeralClient())
    try:
        store.delete_collection(COLLECTION)
    except Exception:
        pass
    store.create_collection(COLLECTION)
    yield store
    try:
        store.delete_collection(COLLECTION)
    except Exception:
        pass
    store.close()


@pytest.fixture()
def chroma_with_embedder():
    """ChromaStore configured with FakeEmbedder, empty test collection."""
    store = ChromaStore(chromadb.EphemeralClient(), embedder=FakeEmbedder())
    try:
        store.delete_collection(COLLECTION)
    except Exception:
        pass
    store.create_collection(COLLECTION)
    yield store
    store.close()


@pytest.fixture()
def qdrant():
    """QdrantStore backed by a guaranteed-empty test collection."""
    store = QdrantStore(QdrantClient(":memory:"), vector_size=DIM)
    try:
        store.delete_collection(COLLECTION)
    except Exception:
        pass
    store.create_collection(COLLECTION)
    yield store
    store.close()


@pytest.fixture()
def qdrant_with_embedder():
    """QdrantStore configured with FakeEmbedder, empty test collection."""
    store = QdrantStore(QdrantClient(":memory:"), vector_size=DIM, embedder=FakeEmbedder())
    try:
        store.delete_collection(COLLECTION)
    except Exception:
        pass
    store.create_collection(COLLECTION)
    yield store
    store.close()


# ---------------------------------------------------------------------------
# StoredRecord
# ---------------------------------------------------------------------------


def test_stored_record_defaults():
    """Optional fields, including score, default to None when not supplied."""
    rec = StoredRecord(id="x")

    assert rec.id == "x"
    assert rec.vector is None
    assert rec.document is None
    assert rec.metadata is None
    assert rec.score is None


def test_stored_record_full_construction():
    """All fields, including score, are stored and accessible."""
    rec = StoredRecord(id="r1", vector=[0.1, 0.2], document="hello", metadata={"k": "v"}, score=0.5)

    assert rec.vector == [0.1, 0.2]
    assert rec.document == "hello"
    assert rec.metadata == {"k": "v"}
    assert rec.score == 0.5


# ---------------------------------------------------------------------------
# BaseVectorStore — context manager
# ---------------------------------------------------------------------------


def test_context_manager_calls_close():
    """__exit__ triggers close(), marking the store as closed."""
    store = ChromaStore(chromadb.EphemeralClient())

    with store:
        assert not store._closed

    assert store._closed


# ---------------------------------------------------------------------------
# BaseVectorStore — embedder resolution
# ---------------------------------------------------------------------------


def test_resolve_vectors_prefers_vectors_over_texts():
    """When both vectors and texts are given, vectors win and the embedder is not consulted."""
    store = ChromaStore(chromadb.EphemeralClient(), embedder=FakeEmbedder())

    resolved = store._resolve_vectors(texts=["vector one"], vectors=[V3])

    assert resolved == [V3]


def test_resolve_vectors_embeds_texts_when_no_vectors():
    """With no vectors given, texts are embedded via the configured embedder."""
    store = ChromaStore(chromadb.EphemeralClient(), embedder=FakeEmbedder())

    resolved = store._resolve_vectors(texts=["vector two"], vectors=None)

    assert resolved == [V2]


def test_resolve_vectors_returns_none_without_embedder_or_vectors():
    """With no vectors and no embedder, resolution yields None (caller decides how to react)."""
    store = ChromaStore(chromadb.EphemeralClient(), embedder=None)

    resolved = store._resolve_vectors(texts=["vector one"], vectors=None)

    assert resolved is None


# ---------------------------------------------------------------------------
# BaseVectorStore — shared presentation helpers
# ---------------------------------------------------------------------------


def test_build_context_window_includes_document_and_score():
    """build_context_window formats id, title, score, and document text."""
    store = ChromaStore(chromadb.EphemeralClient())
    records = [
        StoredRecord(id="d1", document="Hello world", metadata={"title": "Doc One"}, score=0.12),
    ]

    text = store.build_context_window(records)

    assert "Doc One" in text
    assert "Hello world" in text
    assert "0.1200" in text
    assert "d1" in text


def test_build_context_window_handles_missing_title_and_score():
    """build_context_window doesn't error when title/score/document are absent."""
    store = ChromaStore(chromadb.EphemeralClient())
    records = [StoredRecord(id="d1")]

    text = store.build_context_window(records)

    assert "d1" in text
    assert "n/a" in text


def test_print_results_writes_to_stdout(capsys):
    """print_results prints a summary line per record without raising."""
    store = ChromaStore(chromadb.EphemeralClient())
    records = [StoredRecord(id="d1", document="some text", score=0.42)]

    store.print_results(records, label="TEST RESULTS")

    captured = capsys.readouterr()
    assert "TEST RESULTS" in captured.out
    assert "d1" in captured.out


# ---------------------------------------------------------------------------
# ChromaStore — collection management
# ---------------------------------------------------------------------------


def test_chroma_create_collection_idempotent():
    """Creating a collection twice raises no error."""
    store = ChromaStore(chromadb.EphemeralClient())
    store.create_collection(COLLECTION)
    store.create_collection(COLLECTION)  # must not raise


def test_chroma_delete_collection(chroma: ChromaStore):
    """Deleted collections are no longer accessible."""
    chroma.delete_collection(COLLECTION)
    # After deletion, re-creating should succeed (confirms it was gone)
    chroma.create_collection(COLLECTION)


# ---------------------------------------------------------------------------
# ChromaStore — data operations
# ---------------------------------------------------------------------------


def test_chroma_upsert_and_count(chroma: ChromaStore):
    """Upserted records increase the collection count."""
    chroma.upsert(
        collection=COLLECTION,
        ids=["a", "b", "c"],
        vectors=[V1, V2, V3],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"src": "x"}, {"src": "y"}, {"src": "z"}],
    )

    assert chroma.count(collection=COLLECTION) == 3


def test_chroma_upsert_is_idempotent(chroma: ChromaStore):
    """Upserting the same ID twice updates the record, not duplicates it."""
    chroma.upsert(collection=COLLECTION, ids=["a"], vectors=[V1], documents=["v1"])
    chroma.upsert(collection=COLLECTION, ids=["a"], vectors=[V2], documents=["v2"])

    assert chroma.count(collection=COLLECTION) == 1


def test_chroma_get_by_ids(chroma: ChromaStore):
    """get() retrieves the correct StoredRecord for each ID, vector included."""
    chroma.upsert(
        collection=COLLECTION,
        ids=["doc-1"],
        vectors=[V1],
        documents=["hello"],
        metadatas=[{"lang": "en"}],
    )

    records = chroma.get(collection=COLLECTION, ids=["doc-1"])

    assert len(records) == 1
    assert records[0].id == "doc-1"
    assert records[0].document == "hello"
    assert records[0].metadata == {"lang": "en"}
    assert records[0].vector == pytest.approx(V1)
    assert records[0].score is None


def test_chroma_delete_by_ids(chroma: ChromaStore):
    """Deleted records are no longer returned by count or get."""
    chroma.upsert(collection=COLLECTION, ids=["x", "y"], vectors=[V1, V2])
    chroma.delete(collection=COLLECTION, ids=["x"])

    assert chroma.count(collection=COLLECTION) == 1
    remaining = chroma.get(collection=COLLECTION, ids=["y"])
    assert remaining[0].id == "y"


def test_chroma_query_returns_nearest_neighbour(chroma: ChromaStore):
    """query() returns the most similar vector first, as StoredRecord objects."""
    chroma.upsert(
        collection=COLLECTION,
        ids=["close", "far1", "far2"],
        vectors=[V1, V2, V3],
        documents=["close doc", "far doc 1", "far doc 2"],
    )

    results = chroma.query(collection=COLLECTION, query_vectors=[V1], n_results=1)

    assert len(results) == 1  # one inner list, for the one query vector
    top = results[0][0]
    assert isinstance(top, StoredRecord)
    assert top.id == "close"
    assert top.document == "close doc"
    assert top.score == pytest.approx(0.0, abs=1e-5)


def test_chroma_query_returns_ordered_results(chroma: ChromaStore):
    """query() results are ordered nearest-first (Chroma distance ascending)."""
    chroma.upsert(collection=COLLECTION, ids=["v1", "v2", "v3"], vectors=[V1, V2, V3])

    results = chroma.query(collection=COLLECTION, query_vectors=[V1], n_results=3)

    scores: list[float] = [r.score for r in results[0] if r.score is not None]
    assert len(scores) == 3
    assert scores == sorted(scores)


def test_chroma_query_include_vectors(chroma: ChromaStore):
    """include_vectors=True populates StoredRecord.vector on query hits."""
    chroma.upsert(collection=COLLECTION, ids=["v1"], vectors=[V1])

    results = chroma.query(collection=COLLECTION, query_vectors=[V1], n_results=1, include_vectors=True)

    assert results[0][0].vector == pytest.approx(V1)


def test_chroma_query_omits_vectors_by_default(chroma: ChromaStore):
    """include_vectors defaults to False, so StoredRecord.vector stays None."""
    chroma.upsert(collection=COLLECTION, ids=["v1"], vectors=[V1])

    results = chroma.query(collection=COLLECTION, query_vectors=[V1], n_results=1)

    assert results[0][0].vector is None


def test_chroma_persist_noop_for_ephemeral_client(chroma: ChromaStore):
    """persist() does not raise for in-memory clients."""
    chroma.persist()  # must not raise


def test_chroma_close_is_idempotent():
    """close() can be called multiple times without error."""
    store = ChromaStore(chromadb.EphemeralClient())
    store.close()
    store.close()  # must not raise


# ---------------------------------------------------------------------------
# ChromaStore — embedder integration
# ---------------------------------------------------------------------------


def test_chroma_upsert_with_embedder_no_vectors(chroma_with_embedder: ChromaStore):
    """upsert() without vectors embeds documents via the configured embedder."""
    chroma_with_embedder.upsert(
        collection=COLLECTION,
        ids=["a"],
        documents=["vector one"],
    )

    records = chroma_with_embedder.get(collection=COLLECTION, ids=["a"])
    assert records[0].vector == pytest.approx(V1)


def test_chroma_query_with_embedder_no_query_vectors(chroma_with_embedder: ChromaStore):
    """query() without query_vectors embeds query_texts via the configured embedder."""
    chroma_with_embedder.upsert(
        collection=COLLECTION,
        ids=["a", "b"],
        documents=["vector one", "vector two"],
    )

    results = chroma_with_embedder.query(
        collection=COLLECTION,
        query_texts=["vector one"],
        n_results=1,
    )

    assert results[0][0].id == "a"
    assert results[0][0].score == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# make_chroma_client factory
# ---------------------------------------------------------------------------


def test_make_chroma_client_memory():
    """engine='memory' returns a working EphemeralClient."""
    client = make_chroma_client(engine="memory")
    # Basic sanity — can create a collection
    client.get_or_create_collection("smoke")


def test_make_chroma_client_unknown_engine_raises():
    """An unrecognised engine string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        make_chroma_client(engine="cassandra")


# ---------------------------------------------------------------------------
# QdrantStore — init validation
# ---------------------------------------------------------------------------


def test_qdrant_invalid_distance_raises():
    """Constructing QdrantStore with an unknown distance metric raises ValueError."""
    with pytest.raises(ValueError, match="Unknown distance"):
        QdrantStore(QdrantClient(":memory:"), vector_size=DIM, distance="hamming")


# ---------------------------------------------------------------------------
# QdrantStore — collection management
# ---------------------------------------------------------------------------


def test_qdrant_create_collection_idempotent():
    """Creating a collection twice raises no error."""
    store = QdrantStore(QdrantClient(":memory:"), vector_size=DIM)
    store.create_collection(COLLECTION)
    store.create_collection(COLLECTION)  # must not raise


def test_qdrant_delete_collection(qdrant: QdrantStore):
    """Deleted collections can be re-created from scratch."""
    qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(COLLECTION)  # must not raise


# ---------------------------------------------------------------------------
# QdrantStore — data operations
# ---------------------------------------------------------------------------


def test_qdrant_upsert_and_count(qdrant: QdrantStore):
    """Upserted records are reflected in the collection count."""
    qdrant.upsert(collection=COLLECTION, ids=["a", "b", "c"], vectors=[V1, V2, V3])

    assert qdrant.count(collection=COLLECTION) == 3


def test_qdrant_upsert_is_idempotent(qdrant: QdrantStore):
    """Upserting the same ID twice updates the record, not duplicates it."""
    qdrant.upsert(collection=COLLECTION, ids=["a"], vectors=[V1])
    qdrant.upsert(collection=COLLECTION, ids=["a"], vectors=[V2])

    assert qdrant.count(collection=COLLECTION) == 1


def test_qdrant_upsert_without_vectors_or_embedder_raises(qdrant: QdrantStore):
    """upsert() with neither vectors nor a configured embedder raises ValueError."""
    with pytest.raises(ValueError, match="embedder"):
        qdrant.upsert(collection=COLLECTION, ids=["a"], documents=["some text"])


def test_qdrant_get_by_ids_restores_original_id(qdrant: QdrantStore):
    """get() returns the original application-level ID, not the internal UUID."""
    qdrant.upsert(
        collection=COLLECTION,
        ids=["doc-99:000042"],
        vectors=[V1],
        documents=["hello qdrant"],
        metadatas=[{"lang": "pt"}],
    )

    records = qdrant.get(collection=COLLECTION, ids=["doc-99:000042"])

    assert len(records) == 1
    assert records[0].id == "doc-99:000042"
    # Qdrant has no document/metadata split: document is always None,
    # and any stored text lives inside metadata like any other field.
    assert records[0].document is None
    assert records[0].metadata == {"lang": "pt", "document": "hello qdrant"}
    assert records[0].score is None


def test_qdrant_metadata_does_not_contain_reserved_id_key(qdrant: QdrantStore):
    """The internal _kb_id payload key is stripped from returned metadata."""
    qdrant.upsert(
        collection=COLLECTION,
        ids=["m1"],
        vectors=[V1],
        documents=["text"],
        metadatas=[{"score": 0.9}],
    )

    records = qdrant.get(collection=COLLECTION, ids=["m1"])
    meta = records[0].metadata or {}

    assert "_kb_id" not in meta
    assert meta.get("score") == 0.9
    assert meta.get("document") == "text"


def test_qdrant_delete_by_ids(qdrant: QdrantStore):
    """Deleted records are no longer counted."""
    qdrant.upsert(collection=COLLECTION, ids=["x", "y"], vectors=[V1, V2])
    qdrant.delete(collection=COLLECTION, ids=["x"])

    assert qdrant.count(collection=COLLECTION) == 1


def test_qdrant_query_returns_nearest_neighbour(qdrant: QdrantStore):
    """query() returns the most similar vector first, as StoredRecord objects."""
    qdrant.upsert(
        collection=COLLECTION,
        ids=["close", "far1", "far2"],
        vectors=[V1, V2, V3],
        documents=["close doc", "far doc 1", "far doc 2"],
    )

    results = qdrant.query(collection=COLLECTION, query_vectors=[V1], n_results=1)

    assert len(results) == 1  # one inner list, for the one query vector
    top = results[0][0]
    assert isinstance(top, StoredRecord)
    assert top.id == "close"
    assert (top.metadata or {})["document"] == "close doc"
    # cosine similarity of a vector against itself is 1.0 (higher = more similar)
    assert top.score == pytest.approx(1.0, abs=1e-5)


def test_qdrant_query_returns_ordered_results(qdrant: QdrantStore):
    """query() results are ordered most-similar-first (Qdrant score descending)."""
    qdrant.upsert(collection=COLLECTION, ids=["v1", "v2", "v3"], vectors=[V1, V2, V3])

    results = qdrant.query(collection=COLLECTION, query_vectors=[V1], n_results=3)

    scores: list[float] = [r.score for r in results[0] if r.score is not None]
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)


def test_qdrant_query_multiple_queries(qdrant: QdrantStore):
    """Passing multiple query vectors returns one result list per query, in order."""
    qdrant.upsert(collection=COLLECTION, ids=["v1", "v2"], vectors=[V1, V2])

    results = qdrant.query(collection=COLLECTION, query_vectors=[V1, V2], n_results=1)

    assert len(results) == 2
    assert results[0][0].id == "v1"
    assert results[1][0].id == "v2"


def test_qdrant_query_include_vectors(qdrant: QdrantStore):
    """include_vectors=True populates StoredRecord.vector on query hits."""
    qdrant.upsert(collection=COLLECTION, ids=["v1"], vectors=[V1])

    results = qdrant.query(collection=COLLECTION, query_vectors=[V1], n_results=1, include_vectors=True)

    assert results[0][0].vector == pytest.approx(V1)


def test_qdrant_query_omits_vectors_by_default(qdrant: QdrantStore):
    """include_vectors defaults to False, so StoredRecord.vector stays None."""
    qdrant.upsert(collection=COLLECTION, ids=["v1"], vectors=[V1])

    results = qdrant.query(collection=COLLECTION, query_vectors=[V1], n_results=1)

    assert results[0][0].vector is None


def test_qdrant_query_without_vectors_or_embedder_raises(qdrant: QdrantStore):
    """query() with neither query_vectors nor a configured embedder raises ValueError."""
    with pytest.raises(ValueError, match="embedder"):
        qdrant.query(collection=COLLECTION, query_texts=["some text"], n_results=1)


def test_qdrant_query_where_document_raises(qdrant: QdrantStore):
    """Passing where_document to QdrantStore raises ValueError."""
    with pytest.raises(ValueError, match="where_document"):
        qdrant.query(
            collection=COLLECTION,
            query_vectors=[V1],
            where_document={"$contains": "text"},
        )


def test_qdrant_delete_where_document_raises(qdrant: QdrantStore):
    """Passing where_document to delete() raises ValueError."""
    with pytest.raises(ValueError, match="where_document"):
        qdrant.delete(collection=COLLECTION, where_document={"$contains": "x"})


def test_qdrant_persist_is_noop(qdrant: QdrantStore):
    """persist() does not raise — it is a documented no-op for Qdrant."""
    qdrant.persist()  # must not raise


def test_qdrant_close_is_idempotent():
    """close() can be called multiple times without error."""
    store = QdrantStore(QdrantClient(":memory:"), vector_size=DIM)
    store.close()
    store.close()  # must not raise


# ---------------------------------------------------------------------------
# QdrantStore — embedder integration
# ---------------------------------------------------------------------------


def test_qdrant_upsert_with_embedder_no_vectors(qdrant_with_embedder: QdrantStore):
    """upsert() without vectors embeds documents via the configured embedder."""
    qdrant_with_embedder.upsert(
        collection=COLLECTION,
        ids=["a"],
        documents=["vector one"],
    )

    records = qdrant_with_embedder.get(collection=COLLECTION, ids=["a"])
    assert records[0].vector == pytest.approx(V1)


def test_qdrant_query_with_embedder_no_query_vectors(qdrant_with_embedder: QdrantStore):
    """query() without query_vectors embeds query_texts via the configured embedder."""
    qdrant_with_embedder.upsert(
        collection=COLLECTION,
        ids=["a", "b"],
        documents=["vector one", "vector two"],
    )

    results = qdrant_with_embedder.query(
        collection=COLLECTION,
        query_texts=["vector one"],
        n_results=1,
    )

    assert results[0][0].id == "a"
    assert results[0][0].score == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# make_qdrant_client factory
# ---------------------------------------------------------------------------


def test_make_qdrant_client_memory():
    """engine='memory' returns a working in-process QdrantClient."""
    client = make_qdrant_client(engine="memory")
    # Basic sanity — store can be built and a collection created
    store = QdrantStore(client, vector_size=DIM)
    store.create_collection("smoke")


def test_make_qdrant_client_unknown_engine_raises():
    """An unrecognised engine string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        make_qdrant_client(engine="cassandra")
