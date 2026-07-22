"""Unit tests for kb_vectorizer.text.tokenizer.

Covers the fuzzy multi-lingual tokenizer, accent normalization, stable
token hashing, and term-frequency vector construction used by both
InMemoryKeywordIndex (rank_bm25) and QdrantStore's native BM25 support.
"""

from __future__ import annotations

from collections import Counter

from kb_vectorizer.text.tokenizer import (
    normalize_accents,
    term_frequency_vector,
    token_to_index,
    tokenize,
)


def test_normalize_accents_strips_diacritics():
    """Accented characters are converted to their unaccented ASCII form."""
    assert normalize_accents("verificação") == "verificacao"
    assert normalize_accents("São Paulo") == "Sao Paulo"


def test_normalize_accents_leaves_ascii_untouched():
    """Plain ASCII text passes through unchanged."""
    assert normalize_accents("hello world") == "hello world"


def test_tokenize_simple_words():
    """Plain words tokenize to themselves, lowercased."""
    tokens = tokenize("Hello World")
    assert tokens == ["hello", "world"]


def test_tokenize_expands_hyphenated_tokens():
    """A hyphenated token also yields a flattened (no-separator) variant."""
    tokens = tokenize("SN-X2047-QR")
    assert "sn-x2047-qr" in tokens
    assert "snx2047qr" in tokens


def test_tokenize_expands_accented_tokens():
    """An accented token also yields an accent-stripped variant."""
    tokens = tokenize("verificação")
    assert "verificação" in tokens
    assert "verificacao" in tokens


def test_tokenize_accented_and_hyphenated_combines_all_variants():
    """A token with both a separator and accents yields all four variants."""
    tokens = tokenize("ação-teste")
    assert "ação-teste" in tokens
    assert "acao-teste" in tokens
    assert "açãoteste" in tokens
    assert "acaoteste" in tokens


def test_tokenize_preserves_repeat_occurrences():
    """Repeated words produce repeated tokens, not deduplicated ones.

    This matters because BM25/term-frequency consumers need accurate
    counts, not a set of unique tokens.
    """
    tokens = tokenize("hello world hello")
    assert Counter(tokens)["hello"] == 2
    assert Counter(tokens)["world"] == 1


def test_tokenize_empty_string_returns_empty_list():
    """Tokenizing an empty string yields no tokens."""
    assert tokenize("") == []


def test_token_to_index_is_stable_across_calls():
    """The same token always maps to the same index, across repeated calls."""
    a = token_to_index("hello")
    b = token_to_index("hello")
    assert a == b


def test_token_to_index_differs_for_different_tokens():
    """Distinct tokens map to distinct indices (no trivial collision)."""
    assert token_to_index("hello") != token_to_index("world")


def test_token_to_index_within_vocab_size():
    """The returned index always falls within [0, vocab_size)."""
    idx = token_to_index("some-token", vocab_size=1000)
    assert 0 <= idx < 1000


def test_term_frequency_vector_counts_occurrences():
    """Each unique token's frequency reflects its true occurrence count."""
    tf = term_frequency_vector("hello world hello")
    assert sum(tf.values()) == 3.0
    assert len(tf) == 2  # two distinct base tokens: "hello", "world"


def test_term_frequency_vector_empty_text():
    """Empty text produces an empty term-frequency vector."""
    assert term_frequency_vector("") == {}


def test_term_frequency_vector_keys_match_token_to_index():
    """Vector keys are exactly the token_to_index() values for tokenize() output."""
    text = "alpha beta alpha"
    tf = term_frequency_vector(text)
    expected_keys = {token_to_index(t) for t in tokenize(text)}
    assert set(tf.keys()) == expected_keys
