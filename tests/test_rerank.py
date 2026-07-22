"""Unit tests for kb_vectorizer.rerank.

Covers:
  - MMRReranker           (mmr_reranker.py) — pure Python, no ML model needed
  - HFCrossEncoderReranker (hf_crossencoder_reranker.py) — real small model
  - _candidate_text        bridges StoredRecord.document (Chroma) and
    StoredRecord.metadata["document"] (Qdrant) conventions
"""

from __future__ import annotations

import pytest

from kb_vectorizer.rerank.hf_crossencoder_reranker import HFCrossEncoderReranker, _candidate_text
from kb_vectorizer.rerank.interfaces import BaseReranker
from kb_vectorizer.rerank.mmr_reranker import MMRReranker
from kb_vectorizer.storage.interfaces import StoredRecord

V_A = [1.0, 0.0, 0.0]
V_B = [0.9, 0.1, 0.0]  # near-duplicate of A
V_C = [0.0, 1.0, 0.0]  # orthogonal to A/B — the "diverse" option


# ---------------------------------------------------------------------------
# MMRReranker
# ---------------------------------------------------------------------------


def test_mmr_is_a_base_reranker():
    """MMRReranker conforms to BaseReranker."""
    assert isinstance(MMRReranker(), BaseReranker)


def test_mmr_empty_candidates_returns_empty():
    """Reranking an empty candidate list returns an empty order."""
    reranker = MMRReranker()
    assert reranker.rerank("query", []) == []


def test_mmr_pure_relevance_matches_score_order():
    """lambda_mult=1.0 ignores diversity entirely and just sorts by score."""
    candidates = [
        StoredRecord(id="a", vector=V_A, score=0.95),
        StoredRecord(id="b", vector=V_B, score=0.90),
        StoredRecord(id="c", vector=V_C, score=0.5),
    ]
    reranker = MMRReranker(lambda_mult=1.0, higher_is_better=True)

    order = reranker.rerank("query", candidates, top_n=3)

    assert order == [0, 1, 2]


def test_mmr_diversity_prefers_dissimilar_candidate():
    """A lower lambda_mult favors a diverse pick over a near-duplicate, even with a lower score."""
    candidates = [
        StoredRecord(id="a", vector=V_A, score=0.95),
        StoredRecord(id="b", vector=V_B, score=0.90),  # near-duplicate of a
        StoredRecord(id="c", vector=V_C, score=0.5),  # diverse from a/b
    ]
    reranker = MMRReranker(lambda_mult=0.3, higher_is_better=True)

    order = reranker.rerank("query", candidates, top_n=3)

    assert order[0] == 0  # most relevant still picked first
    assert order[1] == 2  # diverse "c" preferred over redundant "b" next


def test_mmr_respects_top_n_override():
    """A top_n passed to rerank() overrides the constructor default."""
    candidates = [
        StoredRecord(id="a", vector=V_A, score=0.9),
        StoredRecord(id="b", vector=V_B, score=0.8),
        StoredRecord(id="c", vector=V_C, score=0.5),
    ]
    reranker = MMRReranker(top_n=10)

    order = reranker.rerank("query", candidates, top_n=1)

    assert len(order) == 1


def test_mmr_uses_constructor_top_n_by_default():
    """Without an explicit top_n argument, the constructor's default applies."""
    candidates = [
        StoredRecord(id="a", vector=V_A, score=0.9),
        StoredRecord(id="b", vector=V_B, score=0.8),
        StoredRecord(id="c", vector=V_C, score=0.5),
    ]
    reranker = MMRReranker(top_n=2)

    order = reranker.rerank("query", candidates)

    assert len(order) == 2


def test_mmr_missing_vector_raises():
    """A candidate with no vector raises a clear ValueError."""
    candidates = [StoredRecord(id="x", vector=None, score=0.5)]
    reranker = MMRReranker()

    with pytest.raises(ValueError, match="vector"):
        reranker.rerank("query", candidates)


def test_mmr_missing_score_raises():
    """A candidate with no score raises a clear ValueError."""
    candidates = [StoredRecord(id="x", vector=V_A, score=None)]
    reranker = MMRReranker()

    with pytest.raises(ValueError, match="score"):
        reranker.rerank("query", candidates)


def test_mmr_higher_is_better_false_inverts_distance():
    """With higher_is_better=False, a small score (distance) ranks as more relevant."""
    candidates = [
        StoredRecord(id="close", vector=V_A, score=0.05),  # small distance = similar
        StoredRecord(id="far", vector=V_C, score=0.9),  # large distance = dissimilar
    ]
    reranker = MMRReranker(lambda_mult=1.0, higher_is_better=False)

    order = reranker.rerank("query", candidates, top_n=2)

    assert order[0] == 0


def test_mmr_cosine_matrix_self_similarity_is_one():
    """A vector's similarity to itself is exactly 1.0 in the precomputed matrix."""
    matrix = MMRReranker._cosine_matrix([V_A, V_C])
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][1] == pytest.approx(1.0)


def test_mmr_cosine_matrix_orthogonal_vectors_zero():
    """Orthogonal vectors have zero cosine similarity."""
    matrix = MMRReranker._cosine_matrix([V_A, V_C])
    assert matrix[0][1] == pytest.approx(0.0, abs=1e-9)
    assert matrix[1][0] == pytest.approx(0.0, abs=1e-9)


def test_mmr_query_argument_is_ignored_without_error():
    """MMRReranker never inspects the query string — any value works, including empty."""
    candidates = [StoredRecord(id="a", vector=V_A, score=0.9)]
    reranker = MMRReranker()

    order = reranker.rerank("", candidates)

    assert order == [0]


# ---------------------------------------------------------------------------
# _candidate_text (bridges Chroma/Qdrant document conventions)
# ---------------------------------------------------------------------------


def test_candidate_text_prefers_document_field():
    """Chroma-style candidates expose text directly via .document."""
    record = StoredRecord(id="a", document="chroma text")
    assert _candidate_text(record) == "chroma text"


def test_candidate_text_falls_back_to_metadata():
    """Qdrant-style candidates have document=None; text lives in metadata instead."""
    record = StoredRecord(id="b", document=None, metadata={"document": "qdrant text"})
    assert _candidate_text(record) == "qdrant text"


def test_candidate_text_defaults_to_empty_string():
    """A candidate with neither .document nor metadata text yields an empty string."""
    record = StoredRecord(id="c")
    assert _candidate_text(record) == ""


# ---------------------------------------------------------------------------
# HFCrossEncoderReranker
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def crossencoder():
    """A real, small cross-encoder model, loaded once per test module."""
    return HFCrossEncoderReranker(
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", batch_size=4
    )


def test_crossencoder_is_a_base_reranker(crossencoder: HFCrossEncoderReranker):
    """HFCrossEncoderReranker conforms to BaseReranker."""
    assert isinstance(crossencoder, BaseReranker)


def test_crossencoder_empty_candidates_returns_empty(crossencoder: HFCrossEncoderReranker):
    """Reranking an empty candidate list returns an empty order without invoking the model."""
    assert crossencoder.rerank("query", []) == []


def test_crossencoder_ranks_relevant_document_first(crossencoder: HFCrossEncoderReranker):
    """A document semantically related to the query ranks above an unrelated one."""
    candidates = [
        StoredRecord(id="irrelevant", document="The weather today is sunny with a chance of rain."),
        StoredRecord(id="relevant", document="Python is a popular programming language for data science."),
    ]

    order = crossencoder.rerank("What is Python used for?", candidates)

    assert order[0] == 1


def test_crossencoder_respects_top_n(crossencoder: HFCrossEncoderReranker):
    """top_n limits the number of returned indices."""
    candidates = [
        StoredRecord(id="a", document="Cats are popular pets."),
        StoredRecord(id="b", document="Dogs are loyal companions."),
        StoredRecord(id="c", document="Python is used for data science."),
    ]

    order = crossencoder.rerank("What is Python used for?", candidates, top_n=1)

    assert len(order) == 1
    assert order[0] == 2


def test_crossencoder_works_with_qdrant_style_metadata_text(crossencoder: HFCrossEncoderReranker):
    """Candidates with document=None and text in metadata still rerank correctly."""
    candidates = [
        StoredRecord(id="irrelevant", document=None, metadata={"document": "The weather today is sunny."}),
        StoredRecord(id="relevant", document=None, metadata={"document": "Python is used for data science."}),
    ]

    order = crossencoder.rerank("What is Python used for?", candidates)

    assert order[0] == 1


def test_crossencoder_missing_sentence_transformers_raises(monkeypatch):
    """Constructing the reranker without sentence-transformers installed raises RuntimeError."""
    import kb_vectorizer.rerank.hf_crossencoder_reranker as mod

    monkeypatch.setattr(mod, "CrossEncoder", None)

    with pytest.raises(RuntimeError, match="sentence-transformers not installed"):
        HFCrossEncoderReranker()
