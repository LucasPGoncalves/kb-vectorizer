from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class EmbedResponse:
    vectors: list[list[float]]
    model: str
    dimension: int

class BaseEmbedder(ABC):
    """Simple, batch-first interface so we can stream -> batch -> embed."""

    model_name: str
    max_batch_size: int
    dimension: int | None = None  # some APIs report dimension after first call

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbedResponse:
        """Embed a batch of texts. Must preserve order 1:1."""

    def embed_iter(self, texts: Iterable[str], batch_size: int | None = None) -> Iterable[list[float]]:
        """Convenience: stream an iterator through the embedder in batches."""
        bs = batch_size or self.max_batch_size or 64
        buf: list[str] = []
        for t in texts:
            buf.append(t)
            if len(buf) >= bs:
                out = self.embed(texts).vectors
                for v in out:
                    yield v
                buf.clear()
        if buf:
            out = self.embed(texts).vectors
            for v in out:
                yield v
