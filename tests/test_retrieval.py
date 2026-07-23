"""Unit tests for kb_vectorizer.retrieval.

Covers:
  - KeywordMatch                       (interfaces.py)
  - InMemoryKeywordIndex               (inmemory_keyword_index.py) — rank_bm25-backed
  - NativeKeywordIndex                 (native_keyword_index.py) — delegates to a
    SupportsKeywordSearch-conforming store (QdrantStore with enable_bm25=True here)

Both implementations are exercised through the exact same BaseKeywordIndex
surface, proving they're truly swappable.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from kb_vectorizer.retrieval import InMemoryKeywordIndex, KeywordMatch, NativeKeywordIndex
from kb_vectorizer.retrieval.interfaces import BaseKeywordIndex, SupportsKeywordSearch
from kb_vectorizer.storage.qdrant_store import QdrantStore

COLLECTION = "kw_test_collection"
DIM = 3
V1 = [1.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0]
V3 = [0.0, 0.0, 1.0]

DOCS = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "python is a great programming language for data science",
    "d3": "the dog barked at the quick fox in the yard",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def inmemory_index():
    """Fresh InMemoryKeywordIndex, closed after the test."""
    idx = InMemoryKeywordIndex()
    yield idx
    idx.close()


@pytest.fixture()
def qdrant_bm25_store():
    """QdrantStore with enable_bm25=True, backed by a fresh in-memory collection."""
    store = QdrantStore(QdrantClient(":memory:"), vector_size=DIM, enable_bm25=True)
    try:
        store.delete_collection(COLLECTION)
    except Exception:
        pass
    store.create_collection(COLLECTION)
    yield store
    store.close()


@pytest.fixture()
def native_index(qdrant_bm25_store: QdrantStore):
    """NativeKeywordIndex wrapping a pre-populated QdrantStore collection."""
    qdrant_bm25_store.upsert(
        collection=COLLECTION,
        ids=list(DOCS.keys()),
        vectors=[V1, V2, V3],
        documents=list(DOCS.values()),
    )
    idx = NativeKeywordIndex(qdrant_bm25_store, collection=COLLECTION)
    yield idx
    idx.close()


# ---------------------------------------------------------------------------
# KeywordMatch
# ---------------------------------------------------------------------------


def test_keyword_match_fields():
    """KeywordMatch stores id and score as given."""
    match = KeywordMatch(id="d1", score=1.5)
    assert match.id == "d1"
    assert match.score == 1.5


# ---------------------------------------------------------------------------
# SupportsKeywordSearch protocol
# ---------------------------------------------------------------------------


def test_qdrant_store_satisfies_keyword_search_protocol(qdrant_bm25_store: QdrantStore):
    """QdrantStore(enable_bm25=True) structurally satisfies SupportsKeywordSearch."""
    assert isinstance(qdrant_bm25_store, SupportsKeywordSearch)


# ---------------------------------------------------------------------------
# InMemoryKeywordIndex
# ---------------------------------------------------------------------------


def test_inmemory_is_a_base_keyword_index():
    """InMemoryKeywordIndex conforms to BaseKeywordIndex."""
    assert isinstance(InMemoryKeywordIndex(), BaseKeywordIndex)


def test_inmemory_search_empty_index_returns_empty(inmemory_index: InMemoryKeywordIndex):
    """Searching an index with no documents returns no matches."""
    assert inmemory_index.search("anything") == []


def test_inmemory_upsert_and_search_finds_relevant_docs(inmemory_index: InMemoryKeywordIndex):
    """A query matching document content returns that document, ranked first."""
    inmemory_index.upsert(list(DOCS.keys()), list(DOCS.values()))

    results = inmemory_index.search("quick fox dog", k=5)

    assert results
    assert all(isinstance(r, KeywordMatch) for r in results)
    result_ids = {r.id for r in results}
    assert "d1" in result_ids
    assert "d3" in result_ids


def test_inmemory_results_ordered_by_score_descending(inmemory_index: InMemoryKeywordIndex):
    """Results come back best-match-first."""
    inmemory_index.upsert(list(DOCS.keys()), list(DOCS.values()))

    results = inmemory_index.search("quick fox dog", k=5)
    scores = [r.score for r in results]

    assert scores == sorted(scores, reverse=True)


def test_inmemory_respects_k_limit(inmemory_index: InMemoryKeywordIndex):
    """search() never returns more than k matches."""
    inmemory_index.upsert(list(DOCS.keys()), list(DOCS.values()))

    results = inmemory_index.search("the dog fox python", k=1)

    assert len(results) == 1


def test_inmemory_upsert_is_idempotent(inmemory_index: InMemoryKeywordIndex):
    """Upserting the same ID twice updates its text rather than duplicating it."""
    inmemory_index.upsert(["a"], ["original text about cats"])
    inmemory_index.upsert(["a"], ["updated text about dogs"])

    results = inmemory_index.search("dogs", k=10)
    assert len(results) == 1
    assert results[0].id == "a"


def test_inmemory_delete_removes_document(inmemory_index: InMemoryKeywordIndex):
    """A deleted document no longer appears in search results."""
    inmemory_index.upsert(list(DOCS.keys()), list(DOCS.values()))
    inmemory_index.delete(["d1"])

    results = inmemory_index.search("quick fox dog", k=10)

    assert all(r.id != "d1" for r in results)


def test_inmemory_delete_unknown_id_does_not_raise(inmemory_index: InMemoryKeywordIndex):
    """Deleting an ID that was never indexed is a silent no-op."""
    inmemory_index.upsert(["a"], ["some text"])
    inmemory_index.delete(["does-not-exist"])  # must not raise


def test_inmemory_close_clears_state(inmemory_index: InMemoryKeywordIndex):
    """close() clears the corpus so subsequent searches return nothing."""
    inmemory_index.upsert(["a"], ["some searchable text"])
    inmemory_index.close()

    assert inmemory_index.search("searchable") == []


# ---------------------------------------------------------------------------
# NativeKeywordIndex
# ---------------------------------------------------------------------------


def test_native_is_a_base_keyword_index(qdrant_bm25_store: QdrantStore):
    """NativeKeywordIndex conforms to BaseKeywordIndex."""
    idx = NativeKeywordIndex(qdrant_bm25_store, collection=COLLECTION)
    assert isinstance(idx, BaseKeywordIndex)


def test_native_search_finds_relevant_docs(native_index: NativeKeywordIndex):
    """A query matching document content returns that document via Qdrant's sparse search."""
    results = native_index.search("quick fox dog", k=5)

    assert results
    assert all(isinstance(r, KeywordMatch) for r in results)
    result_ids = {r.id for r in results}
    assert "d1" in result_ids
    assert "d3" in result_ids


def test_native_results_ordered_by_score_descending(native_index: NativeKeywordIndex):
    """Results come back best-match-first (Qdrant's native BM25 score, higher is better)."""
    results = native_index.search("quick fox dog", k=5)
    scores = [r.score for r in results]

    assert scores == sorted(scores, reverse=True)


def test_native_respects_k_limit(native_index: NativeKeywordIndex):
    """search() never returns more than k matches."""
    results = native_index.search("the dog fox python", k=1)

    assert len(results) == 1


def test_native_upsert_is_a_noop(native_index: NativeKeywordIndex):
    """upsert() does not raise and does not add unrelated documents — indexing already happened via the store."""
    native_index.upsert(["new-id"], ["completely unrelated content"])

    results = native_index.search("completely unrelated content", k=10)
    assert all(r.id != "new-id" for r in results)


def test_native_delete_is_a_noop_but_store_delete_reflects_in_search(
    native_index: NativeKeywordIndex, qdrant_bm25_store: QdrantStore
):
    """delete() on the index itself is a no-op; deleting via the underlying store removes the hit."""
    native_index.delete(["d1"])  # no-op, must not raise
    results_before = native_index.search("quick fox dog", k=10)
    assert any(r.id == "d1" for r in results_before)

    qdrant_bm25_store.delete(collection=COLLECTION, ids=["d1"])
    results_after = native_index.search("quick fox dog", k=10)
    assert all(r.id != "d1" for r in results_after)


def test_native_close_is_a_noop(native_index: NativeKeywordIndex):
    """close() does not raise and does not affect the underlying store."""
    native_index.close()  # must not raise
    # Store should still be usable after
    assert native_index.search("quick fox dog", k=5)


def test_qdrant_keyword_search_requires_enable_bm25():
    """keyword_search() raises ValueError on a store constructed without enable_bm25."""
    store = QdrantStore(QdrantClient(":memory:"), vector_size=DIM, enable_bm25=False)
    store.create_collection(COLLECTION)

    with pytest.raises(ValueError, match="enable_bm25"):
        store.keyword_search(collection=COLLECTION, query_text="anything", k=5)


def test_qdrant_keyword_search_where_filter_scopes_results(qdrant_bm25_store: QdrantStore):
    """where= restricts keyword search to matching payload fields, at the HNSW level."""
    qdrant_bm25_store.upsert(
        collection=COLLECTION,
        ids=["north-doc", "south-doc"],
        vectors=[V1, V2],
        documents=["quick fox jumps", "quick fox jumps"],
        metadatas=[{"zone": "NORTH"}, {"zone": "SOUTH"}],
    )

    north_only = qdrant_bm25_store.keyword_search(
        collection=COLLECTION, query_text="quick fox", k=5, where={"zone": "NORTH"}
    )
    south_only = qdrant_bm25_store.keyword_search(
        collection=COLLECTION, query_text="quick fox", k=5, where={"zone": "SOUTH"}
    )

    assert {r.id for r in north_only} == {"north-doc"}
    assert {r.id for r in south_only} == {"south-doc"}


def test_qdrant_keyword_search_no_where_returns_all_matches(qdrant_bm25_store: QdrantStore):
    """Without a where filter, keyword search considers the whole collection."""
    qdrant_bm25_store.upsert(
        collection=COLLECTION,
        ids=["north-doc", "south-doc"],
        vectors=[V1, V2],
        documents=["quick fox jumps", "quick fox jumps"],
        metadatas=[{"zone": "NORTH"}, {"zone": "SOUTH"}],
    )

    results = qdrant_bm25_store.keyword_search(collection=COLLECTION, query_text="quick fox", k=5)

    assert {r.id for r in results} == {"north-doc", "south-doc"}
