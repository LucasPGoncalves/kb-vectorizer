from __future__ import annotations

from typing import Any

from .interfaces import BasePreprocessor, DocumentRecord


class JSONDocumentPreprocessor(BasePreprocessor[DocumentRecord]):
    """Converts a raw JSON-sourced dict into a :class:`DocumentRecord`.

    This preprocessor is the entry point for documents ingested from JSON
    sources. It wraps the full document dictionary in a ``DocumentRecord``
    so downstream steps can access fields by attribute (``doc.title``)
    instead of string key (``doc["title"]``).

    Optionally validates that a set of required fields are present before
    returning, catching malformed source documents early in the pipeline.

    Args:
        required_fields: Field names that must be present in the document.
            If any are missing, :meth:`process` raises ``ValueError``.

    """

    def __init__(self, required_fields: list[str] | None = None) -> None:
        """Initialize the preprocessor.

        Args:
            required_fields: Field names that must be present in every
                document passed to :meth:`process`. Pass ``None`` (default)
                to skip validation.

        """
        self.required_fields = required_fields or []

    def process(self, document: dict[str, Any]) -> DocumentRecord:
        """Wrap *document* in a :class:`DocumentRecord`.

        Args:
            document: Raw document dictionary from the ingestor.

        Returns:
            A ``DocumentRecord`` exposing all document fields as attributes.

        Raises:
            ValueError: If any field listed in ``required_fields`` is absent
                from *document*.

        """
        missing = [f for f in self.required_fields if f not in document]
        if missing:
            raise ValueError(f"Document is missing required fields: {missing}")

        return DocumentRecord(data=document)
