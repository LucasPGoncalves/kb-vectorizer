from __future__ import annotations

import torch
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from kb_vectorizer.embedding.interfaces import BaseEmbedder, EmbedResponse

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class SentenceTransformerEmbedder(BaseEmbedder):
    """Embeds text locally using HuggingFace's sentence-transformers library.
    
    This embedder runs in-process and requires the 'sentence-transformers' package.
    It processes texts in batches and returns the resulting embeddings.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 128,
    ):
        """Initialize the local SentenceTransformer embedder.

        Args:
            model_id: The HuggingFace model ID to use (e.g., "all-MiniLM-L6-v2").
            device: The device to run the model on ('cpu', 'cuda', 'mps', etc.).
            batch_size: The default batch size to use when embedding texts.

        """
        self.model_name = model_id
        self.device = device
        self.max_batch_size = batch_size
        self.dimension = None

        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed; please install it to use the local embedder.")
        
        self.model = SentenceTransformer(model_id)
        self.dimension = self.model.get_embedding_dimension()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(0.2, 2), reraise=True)
    def embed(self, texts: list[str]) -> EmbedResponse:
        """Embeds a list of texts into vector representations.
        
        Args:
            texts: A list of strings to embed.
            
        Returns:
            An EmbedResponse containing the computed vectors, model name, and dimension.

        """
        tensors = self.model.encode(texts, convert_to_numpy=False, device=self.device)
        vecs = [t.tolist() for t in tensors]
        return EmbedResponse(
            vectors=vecs, 
            model=self.model_name, 
            dimension=self.dimension or len(vecs[0])
        )

