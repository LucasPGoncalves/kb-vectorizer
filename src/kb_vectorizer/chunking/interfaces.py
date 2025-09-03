from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, str]

class BaseChunker(ABC):

    @abstractmethod
    def chunk(self, text: str, *, metadata: dict[str, str], doc_id: str) -> list[Chunk]:
        ...
