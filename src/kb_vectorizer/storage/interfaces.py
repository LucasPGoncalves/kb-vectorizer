from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class StoredRecord:
    id: str
    vector: list[float] | None = None
    document: str | None = None
    metadata: dict[str, Any] | None = None

class BaseVectorStore(ABC):
    """Minimal, backend-agnostic vector store API."""

    @abstractmethod
    def create_collection(self, name: str) -> None: ...
    @abstractmethod
    def get_collection(self, name: str): ...
    @abstractmethod
    def delete_collection(self, name: str) -> None: ...

    @abstractmethod
    def upsert(
        self, *, collection: str, ids: Sequence[str],
        vectors: Sequence[Sequence[float]] | None = None,
        documents: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None: ...

    @abstractmethod
    def delete(
        self, *, collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    def get(
        self, *, collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[StoredRecord]: ...

    @abstractmethod
    def query(
        self, *, collection: str,
        query_texts: Sequence[str] | None = None,
        query_vectors: Sequence[Sequence[float]] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: Sequence[str] = ("metadatas", "documents", "distances", "embeddings"),
    ) -> dict[str, Any]: ...

    @abstractmethod
    def count(self, *, collection: str) -> int: ...

    @abstractmethod
    def persist(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
