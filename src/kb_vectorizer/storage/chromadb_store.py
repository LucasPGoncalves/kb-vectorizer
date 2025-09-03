# src/kb_vectorizer/storage/chromadb_store.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kb_vectorizer.storage.interfaces import BaseVectorStore, StoredRecord


class ChromaStore(BaseVectorStore):
    """Thin adapter over an injected Chroma client.

    Pass any chroma client:
      - chromadb.Client()              # in-memory
      - chromadb.PersistentClient(path=...)  # on-disk persistence
      - chromadb.HttpClient(host=..., port=..., ssl=...)  # server mode
      - chromadb.CloudClient()         # Chroma Cloud.
    """

    def __init__(self, client):
        self.client = client  # injected “engine”

    # ---- collections ----
    def create_collection(self, name: str) -> None:
        self.client.get_or_create_collection(name=name)  # recommended pattern.

    def get_collection(self, name: str):
        return self.client.get_or_create_collection(name=name)

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)

    # ---- data ops ----
    def upsert(
        self, *, collection: str, ids: Sequence[str],
        vectors: Sequence[Sequence[float]] | None = None,
        documents: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        col = self.get_collection(collection)
        # Upsert preserves idempotency vs add(). (see API)
        col.upsert(ids=list(ids), embeddings=vectors, documents=documents, metadatas=metadatas)

    def delete(
        self, *, collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> None:
        col = self.get_collection(collection)
        col.delete(ids=list(ids) if ids else None, where=where, where_document=where_document)

    def get(
        self, *, collection: str,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[StoredRecord]:
        col = self.get_collection(collection)
        res = col.get(ids=list(ids) if ids else None, where=where, limit=limit)  # get vs query.
        out: list[StoredRecord] = []
        for i, _id in enumerate(res.get("ids", [])):
            out.append(
                StoredRecord(
                    id=_id,
                    vector=(res.get("embeddings") or [None])[i] if res.get("embeddings") else None,
                    document=(res.get("documents") or [None])[i] if res.get("documents") else None,
                    metadata=(res.get("metadatas") or [None])[i] if res.get("metadatas") else None,
                )
            )
        return out

    def query(
        self, *, collection: str,
        query_texts: Sequence[str] | None = None,
        query_vectors: Sequence[Sequence[float]] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: Sequence[str] = ("metadatas", "documents", "distances", "embeddings"),
    ) -> dict[str, Any]:
        col = self.get_collection(collection)
        # where / where_document filters are supported as documented.
        return col.query(
            query_texts=list(query_texts) if query_texts else None,
            query_embeddings=list(query_vectors) if query_vectors else None,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=list(include),
        )

    def count(self, *, collection: str) -> int:
        col = self.get_collection(collection)
        return col.count()

    def persist(self) -> None:
        # PersistentClient exposes .persist(); HttpClient/Client may not.
        if hasattr(self.client, "persist"):
            self.client.persist()  # flush to disk for persistent mode.

    def close(self) -> None:
        # Best-effort flush for persistent engines; no-op otherwise.
        try:
            self.persist()
        finally:
            # Let GC clean up; Chroma doesn’t require explicit close.
            self.client = None
