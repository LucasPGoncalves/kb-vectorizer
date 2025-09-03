from __future__ import annotations

import requests
import torch
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from kb_vectorizer.embedding.interfaces import BaseEmbedder, EmbedResponse

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class SetenceTransformerEmbedder(BaseEmbedder):
    """Embed using a SentenceTransformer/HuggingFace model. Two modes:
    1) Local in-process via SentenceTransformer (if installed)
    2) TEI server via REST (fast).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        local: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        tei_url: str | None = None,
        batch_size: int = 128,
    ):
        self.model_name = model_id
        self.local = local
        self.device = device
        self.tei_url = tei_url
        self.max_batch_size = batch_size
        self.dimension = None

        if self.local:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed; install it or run a TEI server.")
            self.model = SentenceTransformer(model_id)
            self.dimension = self.model.get_sentence_embedding_dimension()

        elif not tei_url:
            raise ValueError("For remote mode, tei_url must be provided.")

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(0.2, 2), reraise=True)
    def embed(self, texts: list[str]) -> EmbedResponse:
        if self.local:
            tensors = self.model.encode(texts, convert_to_numpy=False, device=self.device)
            vecs = [t.tolist() for t in tensors]
            return EmbedResponse(vectors=vecs, model=self.model_name, dimension=self.dimension or len(vecs[0]))

        else:
            resp = requests.post(
                f"{self.tei_url.rstrip('/')}/embed",
                json={"inputs": texts},
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            vecs: list[list[float]] = data.get("embeddings") or data.get("results") or []
            if self.dimension is None and vecs:
                self.dimension = len(vecs[0])
            return EmbedResponse(vectors=vecs, model=self.model_name, dimension=self.dimension)

