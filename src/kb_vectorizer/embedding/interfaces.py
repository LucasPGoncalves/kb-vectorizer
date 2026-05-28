from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class EmbedResponse:
    """Data structure containing the results of an embedding operation.
    
    Attributes:
        vectors: A list of vectors (each a list of floats) corresponding to the input texts.
        model: The name of the model that generated the embeddings.
        dimension: The dimensionality of the vectors.

    """

    vectors: list[list[float]]
    model: str
    dimension: int

class BaseEmbedder(ABC):
    """Abstract base class defining the interface for all text embedders.
    
    It enforces a batch-first architecture to allow processing pipelines to stream 
    and batch data efficiently before sending to the embedder (e.g. for GPU or network efficiency).
    """

    model_name: str
    max_batch_size: int
    dimension: int | None = None  # some APIs report dimension after first call

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbedResponse:
        """Embed a batch of texts. Must preserve order 1:1.
        
        Args:
            texts: A list of strings to embed.
            
        Returns:
            An EmbedResponse containing the generated vectors.

        """

    def embed_iter(self, texts: Iterable[str], batch_size: int | None = None) -> Iterable[list[float]]:
            """Stream an iterator of texts through the embedder in batches.

            Args:
                texts: An iterable of strings to embed.
                batch_size: Override the embedder's default max_batch_size.
                
            Yields:
                Individual vectors (list of floats) as they are computed.

            """
            bs = batch_size or self.max_batch_size or 64
            buf: list[str] = []
            for t in texts:
                buf.append(t)
                if len(buf) >= bs:
                    out = self.embed(buf).vectors
                    for v in out:
                        yield v
                    buf.clear()
            if buf:
                out = self.embed(buf).vectors
                for v in out:
                    yield v
