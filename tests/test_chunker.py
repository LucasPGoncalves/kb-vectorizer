from kb_vectorizer.chunking.interfaces import Chunk
from kb_vectorizer.chunking.recursive_token_chunker import TiktokenRecursiveChunker


def test_chunker_initialization():
    """Verify that the chunker initializes with the correct parameters."""
    chunker = TiktokenRecursiveChunker(chunk_size=150, chunk_overlap=30, encoding="cl100k_base")
    assert chunker.chunk_size == 150
    assert chunker.chunk_overlap == 30
    assert chunker._enc.name == "cl100k_base"


def test_single_chunk_no_split():
    """Ensure that text smaller than the chunk size fits into a single chunk."""
    chunker = TiktokenRecursiveChunker(chunk_size=50, chunk_overlap=10)
    text = "This is a short text that easily fits in one chunk."
    chunks = chunker.chunk(text, metadata={"source": "test"}, doc_id="doc1")
    
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == text
    assert chunks[0].id == "doc1:000000"
    assert chunks[0].metadata["source"] == "test"
    assert chunks[0].metadata["doc_id"] == "doc1"
    assert "tok" in chunks[0].metadata


def test_hard_limit_and_overlap():
    """Test that chunks strictly respect the maximum size and generate sequential IDs."""
    chunker = TiktokenRecursiveChunker(chunk_size=15, chunk_overlap=5)
    text = "This is a longer text that needs to be split into multiple chunks so we can test it."
    chunks = chunker.chunk(text, metadata={}, doc_id="doc2")
    
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        # Assert hard limit is strictly respected
        assert chunker._toklen(c.text) <= 15
        
        # Test sequential IDs
        assert c.id == f"doc2:{i:06d}"


def test_image_placeholder_preservation():
    """Verify that image placeholders are kept intact and not split by the chunker."""
    # A chunk size of 15 tokens. The placeholder is ~11 tokens.
    chunker = TiktokenRecursiveChunker(chunk_size=15, chunk_overlap=5)
    text = "Here is an image doc-bc2bda3ee2eb.png and here is more text"
    chunks = chunker.chunk(text, metadata={}, doc_id="doc3")
    
    # We want to ensure that "doc-bc2bda3ee2eb.png" appears entirely in at least one chunk
    # and is not split in half by the tokenizer overlap logic.
    placeholder_found_intact = any("doc-bc2bda3ee2eb.png" in c.text for c in chunks)
    assert placeholder_found_intact, "The image placeholder was improperly split!"


def test_fallback_token_level_splitting():
    """Check that token-level splitting falls back gracefully for ultra-long strings."""
    # If a single word is longer than chunk_size, it has no choice but to split mid-word.
    chunker = TiktokenRecursiveChunker(chunk_size=10, chunk_overlap=2)
    long_string_no_spaces = "a" * 100  # Will be many tokens
    
    chunks = chunker.chunk(long_string_no_spaces, metadata={}, doc_id="doc4")
    assert len(chunks) > 1
    
    for c in chunks:
        assert chunker._toklen(c.text) <= 10