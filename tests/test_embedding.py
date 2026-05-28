import pytest

from kb_vectorizer.embedding.cloud_embedder import CloudEmbedder
from kb_vectorizer.embedding.sentence_transformers_embedder import SentenceTransformerEmbedder


def test_sentence_transformer_embedder_initialization():
    """Verify that the SentenceTransformerEmbedder initializes correctly with given parameters."""
    embedder = SentenceTransformerEmbedder(model_id="all-MiniLM-L6-v2", device="cpu", batch_size=64)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.device == "cpu"
    assert embedder.max_batch_size == 64
    assert embedder.dimension is not None
    assert isinstance(embedder.dimension, int)


def test_sentence_transformer_embed():
    """Ensure that the SentenceTransformerEmbedder generates embeddings correctly for input texts."""
    embedder = SentenceTransformerEmbedder(model_id="all-MiniLM-L6-v2", device="cpu", batch_size=2)
    texts = ["This is a test document.", "Another test text."]
    response = embedder.embed(texts)
    
    assert response.model == "all-MiniLM-L6-v2"
    assert response.dimension == embedder.dimension
    assert len(response.vectors) == 2
    assert len(response.vectors[0]) == embedder.dimension
    assert len(response.vectors[1]) == embedder.dimension


def test_cloud_embedder_initialization(monkeypatch):
    """Verify that the CloudEmbedder initializes correctly with the required environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    embedder = CloudEmbedder(model_id="text-embedding-3-small", batch_size=100)
    
    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.api_key == "fake-key"
    assert embedder.max_batch_size == 100
    assert embedder.api_url == "https://api.openai.com/v1/embeddings"
    assert embedder.dimension is None


def test_cloud_embedder_missing_key(monkeypatch):
    """Ensure that CloudEmbedder raises a ValueError if the OpenAI API key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CloudEmbedder requires an API key"):
        CloudEmbedder()