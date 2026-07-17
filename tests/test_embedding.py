import torch
import pytest
from sentence_transformers import SentenceTransformer

from kb_vectorizer.embedding import local_embedder
from kb_vectorizer.embedding.cloud_embedder import CloudEmbedder
from kb_vectorizer.embedding.local_embedder import LocalEmbedder
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


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------


def test_local_embedder_missing_path_raises(tmp_path):
    """Constructing LocalEmbedder with a non-existent path raises FileNotFoundError."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LocalEmbedder(missing)


def test_local_embedder_detects_sentence_transformers_format(tmp_path):
    """_is_sentence_transformers_format() detects the marker files without loading any model."""
    embedder = object.__new__(LocalEmbedder)
    embedder.model_path = tmp_path

    assert embedder._is_sentence_transformers_format() is False

    (tmp_path / "modules.json").write_text("[]")
    assert embedder._is_sentence_transformers_format() is True


def test_local_embedder_detects_sentence_transformers_config_marker(tmp_path):
    """config_sentence_transformers.json alone is also recognised as the ST-format marker."""
    embedder = object.__new__(LocalEmbedder)
    embedder.model_path = tmp_path
    (tmp_path / "config_sentence_transformers.json").write_text("{}")

    assert embedder._is_sentence_transformers_format() is True


def test_local_embedder_raises_without_st_format_or_transformers(monkeypatch, tmp_path):
    """With no ST-format markers and no transformers/sentence-transformers available, raises RuntimeError."""
    monkeypatch.setattr(local_embedder, "SentenceTransformer", None)
    monkeypatch.setattr(local_embedder, "AutoModel", None)
    monkeypatch.setattr(local_embedder, "AutoTokenizer", None)

    with pytest.raises(RuntimeError, match="transformers"):
        LocalEmbedder(tmp_path)


def test_local_embedder_loads_real_sentence_transformers_checkpoint(tmp_path):
    """A model saved in sentence-transformers format is auto-detected and loaded from disk."""
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    st_model.save(str(tmp_path))

    embedder = LocalEmbedder(tmp_path, device="cpu")

    assert embedder._mode == "sentence-transformers"
    assert embedder.dimension == st_model.get_embedding_dimension()

    response = embedder.embed(["hello world"])
    assert response.dimension == embedder.dimension
    assert len(response.vectors) == 1
    assert len(response.vectors[0]) == embedder.dimension


class _FakeHFOutput:
    """Minimal stand-in for a HuggingFace model's forward() output."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeHFModel:
    """Deterministic stand-in for AutoModel — token embedding equals its token id, broadcast."""

    hidden_size = 4

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        hidden = input_ids.unsqueeze(-1).float().expand(-1, -1, self.hidden_size)
        return _FakeHFOutput(last_hidden_state=hidden)

    @property
    def config(self):
        return type("Cfg", (), {"hidden_size": self.hidden_size})()


class _FakeHFTokenizerOutput(dict):
    """Dict subclass so `.to(device)` can be chained like a real BatchEncoding."""

    def to(self, device):
        return self


class _FakeHFTokenizer:
    """Deterministic stand-in for AutoTokenizer — fixed 2-text, 3-token batch."""

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        input_ids = torch.tensor([[1, 2, 0], [1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        return _FakeHFTokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)


def test_local_embedder_raw_hf_mean_pooling_masks_padding(monkeypatch, tmp_path):
    """The raw-HF fallback mean-pools only real tokens, excluding padded positions via attention_mask."""
    monkeypatch.setattr(local_embedder, "SentenceTransformer", None)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(path):
            return _FakeHFModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            return _FakeHFTokenizer()

    monkeypatch.setattr(local_embedder, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(local_embedder, "AutoTokenizer", _FakeAutoTokenizer)

    embedder = LocalEmbedder(tmp_path, device="cpu")

    assert embedder._mode == "raw-hf"
    assert embedder.dimension == 4

    response = embedder.embed(["short", "longer text"])

    # text0: tokens [1,2] valid (mask [1,1,0] excludes the padded 3rd position)
    #   -> mean of token embeddings [1,1,1,1] and [2,2,2,2] = [1.5, 1.5, 1.5, 1.5]
    # text1: tokens [1,2,3] all valid
    #   -> mean of [1,1,1,1], [2,2,2,2], [3,3,3,3] = [2.0, 2.0, 2.0, 2.0]
    assert response.vectors[0] == pytest.approx([1.5, 1.5, 1.5, 1.5])
    assert response.vectors[1] == pytest.approx([2.0, 2.0, 2.0, 2.0])