"""Unit tests for the kb_vectorizer.storage module.

Covers:
  - StoredRecord             (interfaces.py)
  - BaseVectorStore          (interfaces.py) — context manager
  - ChromaStore              (chromadb_store.py)
  - make_chroma_client       (chroma_client_factory.py)
  - QdrantStore              (qdrant_store.py)
  - make_qdrant_client       (qdrant_client_factory.py)

All tests run fully in-memory — no external servers required.
"""

from __future__ import annotations

import chromadb
import pytest
from qdrant_client import QdrantClient

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


# ---------------------------------------------------------------------------
# StoredRecord
# ---------------------------------------------------------------------------


def test_stored_record_defaults():
    """Optional fields default to None when not supplied."""
    rec = StoredRecord(id="x")

    assert rec.id == "x"
    assert rec.vector is None
    assert rec.document is None
    assert rec.metadata is None


def test_stored_record_full_construction():
    """All fields are stored and accessible."""
    rec = StoredRecord(id="r1", vector=[0.1, 0.2], document="hello", metadata={"k": "v"})

    assert rec.vector == [0.1, 0.2]
    assert rec.document == "hello"
    assert rec.metadata == {"k": "v"}


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
    """get() retrieves the correct StoredRecord for each ID."""
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


def test_chroma_delete_by_ids(chroma: ChromaStore):
    """Deleted records are no longer returned by count or get."""
    chroma.upsert(collection=COLLECTION, ids=["x", "y"], vectors=[V1, V2])
    chroma.delete(collection=COLLECTION, ids=["x"])

    assert chroma.count(collection=COLLECTION) == 1
    remaining = chroma.get(collection=COLLECTION, ids=["y"])
    assert remaining[0].id == "y"


def test_chroma_query_returns_nearest_neighbour(chroma: ChromaStore):
    """query() returns the most similar vector first."""
    chroma.upsert(
        collection=COLLECTION,
        ids=["close", "far1", "far2"],
        vectors=[V1, V2, V3],
        documents=["close doc", "far doc 1", "far doc 2"],
    )

    res = chroma.query(
        collection=COLLECTION,
        query_vectors=[V1],
        n_results=1,
        include=["documents", "distances"],
    )

    # The closest document to V1 is V1 itself
    assert res["ids"][0][0] == "close"
    assert res["distances"][0][0] == pytest.approx(0.0, abs=1e-5)


def test_chroma_query_returns_ordered_results(chroma: ChromaStore):
    """query() results are ordered nearest-first (distances ascending)."""
    chroma.upsert(
        collection=COLLECTION,
        ids=["v1", "v2", "v3"],
        vectors=[V1, V2, V3],
    )

    res = chroma.query(
        collection=COLLECTION,
        query_vectors=[V1],
        n_results=3,
        include=["distances"],
    )

    dists = res["distances"][0]
    assert dists == sorted(dists)


def test_chroma_persist_noop_for_ephemeral_client(chroma: ChromaStore):
    """persist() does not raise for in-memory clients."""
    chroma.persist()  # must not raise


def test_chroma_close_is_idempotent():
    """close() can be called multiple times without error."""
    store = ChromaStore(chromadb.EphemeralClient())
    store.close()
    store.close()  # must not raise


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
    qdrant.upsert(
        collection=COLLECTION,
        ids=["a", "b", "c"],
        vectors=[V1, V2, V3],
    )

    assert qdrant.count(collection=COLLECTION) == 3


def test_qdrant_upsert_is_idempotent(qdrant: QdrantStore):
    """Upserting the same ID twice updates the record, not duplicates it."""
    qdrant.upsert(collection=COLLECTION, ids=["a"], vectors=[V1])
    qdrant.upsert(collection=COLLECTION, ids=["a"], vectors=[V2])

    assert qdrant.count(collection=COLLECTION) == 1


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
    assert records[0].document == "hello qdrant"
    assert records[0].metadata == {"lang": "pt"}


def test_qdrant_metadata_does_not_contain_reserved_keys(qdrant: QdrantStore):
    """Internal payload keys (_kb_id, _kb_document) are stripped from metadata."""
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
    assert "_kb_document" not in meta
    assert meta.get("score") == 0.9


def test_qdrant_delete_by_ids(qdrant: QdrantStore):
    """Deleted records are no longer counted."""
    qdrant.upsert(collection=COLLECTION, ids=["x", "y"], vectors=[V1, V2])
    qdrant.delete(collection=COLLECTION, ids=["x"])

    assert qdrant.count(collection=COLLECTION) == 1


def test_qdrant_query_returns_nearest_neighbour(qdrant: QdrantStore):
    """query() returns the most similar vector first."""
    qdrant.upsert(
        collection=COLLECTION,
        ids=["close", "far1", "far2"],
        vectors=[V1, V2, V3],
        documents=["close doc", "far doc 1", "far doc 2"],
    )

    res = qdrant.query(
        collection=COLLECTION,
        query_vectors=[V1],
        n_results=1,
        include=["documents", "distances"],
    )

    assert res["ids"][0][0] == "close"
    assert res["distances"][0][0] == pytest.approx(0.0, abs=1e-5)


def test_qdrant_query_returns_ordered_results(qdrant: QdrantStore):
    """query() results are ordered nearest-first (distances ascending)."""
    qdrant.upsert(
        collection=COLLECTION,
        ids=["v1", "v2", "v3"],
        vectors=[V1, V2, V3],
    )

    res = qdrant.query(
        collection=COLLECTION,
        query_vectors=[V1],
        n_results=3,
        include=["distances"],
    )

    dists = res["distances"][0]
    assert dists == sorted(dists)


def test_qdrant_query_multiple_queries(qdrant: QdrantStore):
    """Passing multiple query vectors returns one result list per query."""
    qdrant.upsert(collection=COLLECTION, ids=["v1", "v2"], vectors=[V1, V2])

    res = qdrant.query(
        collection=COLLECTION,
        query_vectors=[V1, V2],
        n_results=1,
        include=["documents"],
    )

    assert len(res["ids"]) == 2
    assert res["ids"][0][0] == "v1"
    assert res["ids"][1][0] == "v2"


def test_qdrant_query_texts_raises(qdrant: QdrantStore):
    """Passing query_texts to QdrantStore raises ValueError."""
    with pytest.raises(ValueError, match="query_texts"):
        qdrant.query(
            collection=COLLECTION,
            query_texts=["some text"],
            n_results=1,
        )


def test_qdrant_query_no_vectors_raises(qdrant: QdrantStore):
    """Calling query() without query_vectors or query_texts raises ValueError."""
    with pytest.raises(ValueError, match="query_vectors"):
        qdrant.query(collection=COLLECTION, n_results=1)


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
