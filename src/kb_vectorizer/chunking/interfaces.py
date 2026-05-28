from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a chunk of text with associated metadata.
    
    Attributes:
        id: Unique identifier for the chunk.
        text: The text content of the chunk.
        metadata: Dictionary containing metadata for the chunk (e.g., doc_id, order, token count).

    """

    id: str
    text: str
    metadata: dict[str, str]

class BaseChunker(ABC):
    """Abstract base class for all chunkers."""

    @abstractmethod
    def chunk(self, text: str, *, metadata: dict[str, str], doc_id: str) -> list[Chunk]:
        """Split a text into multiple chunks.

        Args:
                    text: The text to chunk.
                    metadata: Base metadata to attach to each chunk.
                    doc_id: The document identifier.
                    
        Returns:
                    A list of Chunk objects.
                    
        """
        ...
