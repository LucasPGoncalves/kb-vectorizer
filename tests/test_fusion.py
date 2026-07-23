"""Unit tests for kb_vectorizer.fusion.rrf_fusor."""

from __future__ import annotations

import pytest

from kb_vectorizer.fusion.interfaces import BaseFusor
from kb_vectorizer.fusion.rrf_fusor import RRFFusor


def test_rrf_fusor_is_a_base_fusor():
    """RRFFusor conforms to BaseFusor."""
    assert isinstance(RRFFusor(), BaseFusor)


def test_rrf_single_ranking_preserves_order():
    """Fusing a single ranking just reflects its own rank order."""
    fused = RRFFusor.fuse([["a", "b", "c"]])
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["a", "b", "c"]


def test_rrf_agreement_across_rankings_boosts_score():
    """An ID ranked highly in every input ranking outranks one seen in only one."""
    dense_ranking = ["a", "b", "c"]
    keyword_ranking = ["a", "c", "b"]

    fused = RRFFusor.fuse([dense_ranking, keyword_ranking])

    assert fused[0][0] == "a"  # ranked #1 in both


def test_rrf_score_matches_formula():
    """Fused score exactly matches sum(1 / (k + rank)) across rankings."""
    k = 60
    fused = RRFFusor.fuse([["a", "b"]], k=k)
    scores = dict(fused)

    assert scores["a"] == pytest.approx(1.0 / (k + 1))
    assert scores["b"] == pytest.approx(1.0 / (k + 2))


def test_rrf_id_missing_from_one_ranking_still_included():
    """An ID that only appears in one of several rankings is still present in the fused result."""
    fused = RRFFusor.fuse([["a", "b"], ["c"]])
    ids = {doc_id for doc_id, _ in fused}

    assert ids == {"a", "b", "c"}


def test_rrf_results_sorted_descending():
    """Fused results are always ordered by score, highest first."""
    fused = RRFFusor.fuse([["a", "b", "c"], ["c", "a", "b"]])
    scores = [score for _, score in fused]

    assert scores == sorted(scores, reverse=True)


def test_rrf_empty_rankings_returns_empty():
    """Fusing zero rankings returns an empty list."""
    assert RRFFusor.fuse([]) == []


def test_rrf_k_dampens_low_rank_contributions():
    """A larger k reduces the score gap between a top rank and a low rank."""
    small_k = dict(RRFFusor.fuse([["a", "b"]], k=1))
    large_k = dict(RRFFusor.fuse([["a", "b"]], k=1000))

    gap_small_k = small_k["a"] - small_k["b"]
    gap_large_k = large_k["a"] - large_k["b"]

    assert gap_small_k > gap_large_k
