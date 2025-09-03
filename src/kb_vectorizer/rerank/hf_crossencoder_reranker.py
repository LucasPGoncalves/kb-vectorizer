from __future__ import annotations

from typing import Any

from .interfaces import BaseReranker

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

class HFCrossEncoderReranker(BaseReranker):
    """Cross-Encoder reranker using Sentence-Transformers.

    Works with:
    - Qwen/Qwen3-Reranker-* (0.6B/4B/8B)
    - cross-encoder/ms-marco-... (MiniLM, Electra, TinyBERT, etc.).
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-Reranker-0.6B", device: str | None = None):
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers not installed. `uv add sentence-transformers`")
        self.model_id = model_id
        self.model = CrossEncoder(model_id, device=device)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int | None = None) -> list[int]:
        if not candidates:
            return []
        pairs = [(query, c.get("document") or c.get("text") or "") for c in candidates]
        scores = self.model.predict(pairs)  # higher = better
        # argsort descending
        ordered = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        if top_n is not None:
            ordered = ordered[:top_n]
        return ordered
