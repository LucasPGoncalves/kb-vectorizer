from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kb_vectorizer.storage.interfaces import StoredRecord


def group_by_doc_id(records: Sequence[StoredRecord]) -> list[StoredRecord]:
    """Collapse chunk-level hits down to one hit per parent document.

    Many chunks retrieved by a query can belong to the same source
    document; this keeps only the first (best-ranked) occurrence per
    parent document.

    Args:
        records: Retrieval hits, typically one inner list from
            :meth:`~kb_vectorizer.storage.interfaces.BaseVectorStore.query`,
            already ordered best-first.

    Returns:
        One :class:`StoredRecord` per distinct parent document, in the
        same relative order as *records*. The parent document ID is read
        from ``metadata["doc_id"]`` if present, else falls back to the
        record's own ``id`` — i.e. treating each record as its own parent
        document.

    """
    seen: set[str] = set()
    unique: list[StoredRecord] = []
    for record in records:
        doc_id = (record.metadata or {}).get("doc_id") or record.id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(record)
    return unique


def resolve_parent_documents(records: Sequence[StoredRecord]) -> list[StoredRecord]:
    """Resolve each chunk-level hit to its full parent document's content.

    Reads a file path from ``metadata["source_path"]`` and, if it exists,
    replaces the chunk's ``document`` with the full parent document's text
    (typically Markdown, with image references already resolved by the
    preprocessing step). Usually called after :func:`group_by_doc_id`, so
    each input record represents a distinct source document.

    Args:
        records: Chunk-level hits to resolve.

    Returns:
        One :class:`StoredRecord` per input record: ``id`` set to the
        resolved parent document ID, ``document`` set to the full parent
        content (``None`` if ``source_path`` is missing or doesn't exist
        on disk), and ``metadata``/``score`` carried over unchanged.

    """
    resolved: list[StoredRecord] = []
    for record in records:
        meta = record.metadata or {}
        doc_id = meta.get("doc_id") or record.id
        source_path = meta.get("source_path")

        full_doc: str | None = None
        if source_path:
            path = Path(source_path)
            if path.exists():
                full_doc = path.read_text(encoding="utf-8")

        resolved.append(
            StoredRecord(id=doc_id, document=full_doc, metadata=meta, score=record.score)
        )
    return resolved
