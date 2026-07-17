from __future__ import annotations

from pathlib import Path

import torch
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from kb_vectorizer.embedding.interfaces import BaseEmbedder, EmbedResponse

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


class LocalEmbedder(BaseEmbedder):
    """Embeds text with a model loaded entirely from local disk — no Hub download.

    Intended for fine-tuned checkpoints you don't want to upload anywhere
    (private or proprietary models). Unlike ``SentenceTransformerEmbedder``,
    which accepts a HuggingFace Hub model ID, this always resolves *model_path*
    as a local directory and never touches the network to fetch model files.

    The directory's format is auto-detected on load, in order:

    1. **sentence-transformers format** — the directory contains
       ``modules.json`` or ``config_sentence_transformers.json`` (the normal
       output of ``model.save(path)`` after fine-tuning with the
       sentence-transformers library). Loaded via ``SentenceTransformer``,
       which already knows the correct pooling strategy for the model.
    2. **Raw HuggingFace checkpoint** — plain ``AutoModel``/``AutoTokenizer``
       files with no sentence-transformers wrapper (e.g. saved via
       ``model.save_pretrained()`` from a custom training loop). Loaded
       directly and mean-pooled over the last hidden state, masked by
       attention, which is the standard way to turn per-token hidden states
       into a single sentence embedding for a model with no dedicated
       pooling head.

    Args:
        model_path: Local directory containing the model files.
        device: Device to run the model on ('cpu', 'cuda', 'mps', etc.).
        batch_size: Default batch size used when embedding texts.
        max_seq_length: Maximum token length, used only for the raw
            HuggingFace fallback — the sentence-transformers path uses the
            model's own configured sequence length instead.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
        RuntimeError: If neither sentence-transformers nor transformers is
            installed, or if loading otherwise fails.

    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 128,
        max_seq_length: int = 512,
    ):
        """Initialize the local embedder by loading a model from disk.

        Args:
            model_path: Local directory containing the model files.
            device: The device to run the model on ('cpu', 'cuda', 'mps', etc.).
            batch_size: The default batch size to use when embedding texts.
            max_seq_length: Maximum token length for the raw-HuggingFace
                fallback path only.

        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {self.model_path}")

        self.model_name = str(self.model_path)
        self.device = device
        self.max_batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.dimension = None

        self._load()

    def _is_sentence_transformers_format(self) -> bool:
        """Return whether *model_path* looks like a saved sentence-transformers model."""
        return (self.model_path / "modules.json").exists() or (
            self.model_path / "config_sentence_transformers.json"
        ).exists()

    def _load(self) -> None:
        """Load the model from disk, picking the loading strategy that matches its format.

        Raises:
            RuntimeError: If the required library for the detected (or
                fallback) format is not installed.

        """
        if SentenceTransformer is not None and self._is_sentence_transformers_format():
            self._st_model = SentenceTransformer(str(self.model_path), device=self.device)
            self.dimension = self._st_model.get_embedding_dimension()
            self._mode = "sentence-transformers"
            return

        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError(
                "No sentence-transformers-format files found at "
                f"'{self.model_path}', and 'transformers' is not installed "
                "to fall back to a raw checkpoint. Install 'transformers', "
                "or point model_path at a sentence-transformers-format directory."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._hf_model = AutoModel.from_pretrained(str(self.model_path)).to(self.device)
        self._hf_model.eval()
        self.dimension = self._hf_model.config.hidden_size
        self._mode = "raw-hf"

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(0.2, 2), reraise=True)
    def embed(self, texts: list[str]) -> EmbedResponse:
        """Embed a batch of texts into vector representations.

        Args:
            texts: A list of strings to embed.

        Returns:
            An EmbedResponse containing the computed vectors, model path
            (as ``model``), and dimension.

        """
        if self._mode == "sentence-transformers":
            tensors = self._st_model.encode(texts, convert_to_numpy=False, device=self.device)
            vectors = [t.tolist() for t in tensors]
        else:
            vectors = self._embed_raw_hf(texts)
        return EmbedResponse(
            vectors=vectors,
            model=self.model_name,
            dimension=self.dimension or len(vectors[0]),
        )

    def _embed_raw_hf(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* with the raw HuggingFace checkpoint via masked mean pooling.

        Args:
            texts: A list of strings to embed.

        Returns:
            One vector per input text, in order.

        """
        encoded = self._tokenizer(  # type: ignore[misc]
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self._hf_model(**encoded)

        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return [v.tolist() for v in pooled]
