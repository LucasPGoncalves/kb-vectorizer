from __future__ import annotations

from typing import Any

from .interfaces import BasePreprocessor


class JSONToHTMLPreprocessor(BasePreprocessor):
    """Transforms a dictionary from the ingestor into a structured HTML string.

    for the main content and a separate dictionary for metadata.
    """

    def __init__(self, content_fields: list[str], metadata_fields: list[str]):
        self.content_fields = content_fields
        self.metadata_fields = metadata_fields

    def process(self, document: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        html_parts = []
        for field in self.content_fields:
            content = document.get(field, "")
            if content:
                html_parts.append(f'<div class="content-field" id="{field}">{content}</div>')
        
        html_content = "\n".join(html_parts)

        metadata = {field: document.get(field) for field in self.metadata_fields if field in document}
        
        return html_content, metadata
