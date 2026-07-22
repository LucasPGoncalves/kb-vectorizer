from __future__ import annotations

from collections.abc import Sequence

from kb_vectorizer.storage.interfaces import StoredRecord

from .interfaces import BaseReranker

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


def _candidate_text(record: StoredRecord) -> str:
    """Return the best available text for *record*, bridging backend conventions.

    :class:`~kb_vectorizer.storage.chromadb_store.ChromaStore` populates
    ``StoredRecord.document`` directly. :class:`~kb_vectorizer.storage.qdrant_store.QdrantStore`
    never does — Qdrant has no first-class document slot, so any text
    stored via ``upsert(documents=...)`` ends up under the ``"document"``
    metadata key instead. Falls back to an empty string if neither is
    present, rather than raising, so one bad candidate degrades gracefully
    (scores low) instead of failing the whole batch.

    Args:
        record: A candidate to extract text from.

    Returns:
        The candidate's text, or ``""`` if none is available.

    """
    if record.document:
        return record.document
    return (record.metadata or {}).get("document", "")


class HFCrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers' ``CrossEncoder``.

    Loads any HuggingFace model packaged in ``CrossEncoder``-compatible
    format — a sequence-classification model that scores a
    ``(query, passage)`` pair directly, rather than embedding each side
    separately and comparing vectors. This is *not* "any reranker on
    HuggingFace": bi-encoder/embedding models (e.g.
    ``sentence-transformers/all-MiniLM-L6-v2``) won't load correctly here,
    and causal-LM-based rerankers only work if their repo ships the
    necessary ``CrossEncoder`` config.

    Known to work with:

    - ``cross-encoder/ms-marco-*`` (MiniLM, Electra, TinyBERT, …)
    - ``BAAI/bge-reranker-*``, ``mixedbread-ai/mxbai-rerank-*``,
      ``jinaai/jina-reranker-*``
    - ``Qwen/Qwen3-Reranker-*`` — as long as the checkpoint ships
      appropriate ``CrossEncoder`` config; if loading fails, check that
      model's card for its recommended loading method.

    Args:
        model_id: HuggingFace model ID or local path.
        device: Device to run the model on ('cpu', 'cuda', etc.). ``None``
            lets sentence-transformers pick automatically.
        batch_size: Batch size used when scoring ``(query, passage)`` pairs.

    Raises:
        RuntimeError: If sentence-transformers is not installed.

    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Load the cross-encoder model.

        Args:
            model_id: HuggingFace model ID or local path.
            device: Device to run the model on. ``None`` auto-selects.
            batch_size: Batch size used when scoring pairs.

        Raises:
            RuntimeError: If sentence-transformers is not installed.

        """
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers not installed. `uv add sentence-transformers`")
        self.model_id = model_id
        self.batch_size = batch_size
        self.model = CrossEncoder(model_id, device=device)

    def rerank(
        self,
        query: str,
        candidates: Sequence[StoredRecord],
        top_n: int | None = None,
    ) -> list[int]:
        """Return indices into *candidates*, reordered by cross-encoder relevance.

        Args:
            query: Query string, paired with each candidate's text.
            candidates: Candidates to rerank. Text is read from
                ``document`` (Chroma) or ``metadata["document"]`` (Qdrant)
                — see :func:`_candidate_text`.
            top_n: Maximum number of indices to return. ``None`` returns
                all of them, reordered.

        Returns:
            Indices into *candidates*, best match first.

        """
        if not candidates:
            return []

        pairs = [(query, _candidate_text(c)) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)  # type: ignore[arg-type]  # higher = better
        ordered = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        if top_n is not None:
            ordered = ordered[:top_n]
        return ordered
