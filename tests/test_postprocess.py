"""Unit tests for kb_vectorizer.postprocessing.postprocess."""

from __future__ import annotations

from kb_vectorizer.postprocessing.postprocess import group_by_doc_id, resolve_parent_documents
from kb_vectorizer.storage.interfaces import StoredRecord

# ---------------------------------------------------------------------------
# group_by_doc_id
# ---------------------------------------------------------------------------


def test_group_by_doc_id_collapses_chunks_from_same_document():
    """Multiple chunk hits sharing metadata['doc_id'] collapse to one record."""
    records = [
        StoredRecord(id="doc1:000", metadata={"doc_id": "doc1"}),
        StoredRecord(id="doc1:001", metadata={"doc_id": "doc1"}),
        StoredRecord(id="doc2:000", metadata={"doc_id": "doc2"}),
    ]

    unique = group_by_doc_id(records)

    assert len(unique) == 2
    assert unique[0].id == "doc1:000"  # first (best-ranked) occurrence kept
    assert unique[1].id == "doc2:000"


def test_group_by_doc_id_falls_back_to_record_id_without_doc_id_metadata():
    """A record with no metadata['doc_id'] is treated as its own parent document."""
    records = [
        StoredRecord(id="a", metadata=None),
        StoredRecord(id="b", metadata={"other_field": "x"}),
    ]

    unique = group_by_doc_id(records)

    assert [r.id for r in unique] == ["a", "b"]


def test_group_by_doc_id_empty_list():
    """Grouping an empty list returns an empty list."""
    assert group_by_doc_id([]) == []


def test_group_by_doc_id_preserves_relative_order():
    """Kept records preserve their original relative order (best-first)."""
    records = [
        StoredRecord(id="c1", metadata={"doc_id": "docC"}),
        StoredRecord(id="a1", metadata={"doc_id": "docA"}),
        StoredRecord(id="a2", metadata={"doc_id": "docA"}),
        StoredRecord(id="b1", metadata={"doc_id": "docB"}),
    ]

    unique = group_by_doc_id(records)

    assert [r.id for r in unique] == ["c1", "a1", "b1"]


# ---------------------------------------------------------------------------
# resolve_parent_documents
# ---------------------------------------------------------------------------


def test_resolve_parent_documents_reads_source_file(tmp_path):
    """A record with a valid source_path gets the full file content as its document."""
    source = tmp_path / "article.md"
    source.write_text("# Full Article\n\nFull content here.", encoding="utf-8")
    records = [
        StoredRecord(
            id="chunk1",
            document="just a chunk excerpt",
            metadata={"doc_id": "doc1", "source_path": str(source)},
            score=0.9,
        )
    ]

    resolved = resolve_parent_documents(records)

    assert len(resolved) == 1
    assert resolved[0].id == "doc1"
    assert resolved[0].document == "# Full Article\n\nFull content here."
    assert resolved[0].score == 0.9
    assert resolved[0].metadata == records[0].metadata


def test_resolve_parent_documents_missing_source_path_yields_none_document():
    """A record with no source_path in metadata gets document=None."""
    records = [StoredRecord(id="chunk1", metadata={"doc_id": "doc1"})]

    resolved = resolve_parent_documents(records)

    assert resolved[0].document is None


def test_resolve_parent_documents_nonexistent_file_yields_none_document():
    """A source_path pointing to a nonexistent file gets document=None, not an error."""
    records = [
        StoredRecord(id="chunk1", metadata={"doc_id": "doc1", "source_path": "/no/such/file.md"})
    ]

    resolved = resolve_parent_documents(records)

    assert resolved[0].document is None


def test_resolve_parent_documents_falls_back_to_record_id_without_doc_id():
    """A record with no metadata['doc_id'] uses the record's own id as the resolved id."""
    records = [StoredRecord(id="standalone", metadata=None)]

    resolved = resolve_parent_documents(records)

    assert resolved[0].id == "standalone"


def test_resolve_parent_documents_empty_list():
    """Resolving an empty list returns an empty list."""
    assert resolve_parent_documents([]) == []


# ---------------------------------------------------------------------------
# Combined pipeline usage
# ---------------------------------------------------------------------------


def test_group_then_resolve_pipeline(tmp_path):
    """group_by_doc_id followed by resolve_parent_documents mirrors real pipeline usage."""
    source = tmp_path / "doc1.md"
    source.write_text("Full doc1 content", encoding="utf-8")

    chunks = [
        StoredRecord(id="doc1:c0", metadata={"doc_id": "doc1", "source_path": str(source)}, score=0.95),
        StoredRecord(id="doc1:c1", metadata={"doc_id": "doc1", "source_path": str(source)}, score=0.80),
        StoredRecord(id="doc2:c0", metadata={"doc_id": "doc2"}, score=0.70),
    ]

    unique = group_by_doc_id(chunks)
    resolved = resolve_parent_documents(unique)

    assert len(resolved) == 2
    assert resolved[0].id == "doc1"
    assert resolved[0].document == "Full doc1 content"
    assert resolved[1].id == "doc2"
    assert resolved[1].document is None
