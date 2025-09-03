from __future__ import annotations

from typing import Any

from kb_vectorizer.chunking.recursive_token_chunker import TiktokenRecursiveChunker
from kb_vectorizer.embedding.interfaces import BaseEmbedder
from kb_vectorizer.retrieval.keyword_index import KeywordIndex
from kb_vectorizer.storage.chromadb_store import ChromaStore


def index_document(
    *,
    doc_id: str,
    markdown: str,
    metadata: dict[str, Any],
    store: ChromaStore,
    collection: str,
    chunker: TiktokenRecursiveChunker,
    embedder: BaseEmbedder,
    keyword_index: KeywordIndex | None = None,
    batch_size: int = 256,
) -> None:
    # 1) chunk (token-aware, overlap)
    chunks = chunker.split(markdown, metadata=metadata, doc_id=doc_id)

    # 2) embed in micro-batches
    ids, vecs, docs, metas = [], [], [], []
    buf_texts, buf_ids = [], []
    for ch in chunks:
        buf_texts.append(ch.text)
        buf_ids.append(ch.id)
        if len(buf_texts) >= batch_size:
            res = embedder.embed(texts=buf_texts)
            ids.extend(buf_ids) 
            vecs.extend(res.vectors)
            # doc text and metadata per chunk
            docs.extend(buf_texts) 
            metas.extend([ch.metadata for ch in chunks[len(ids)-len(buf_ids): len(ids)]])
            buf_texts, buf_ids = [], []

    if buf_texts:
        res = embedder.embed(texts=buf_texts)
        ids.extend(buf_ids) 
        vecs.extend(res.vectors)
        docs.extend(buf_texts) 
        metas.extend([c.metadata for c in chunks[len(ids)-len(buf_ids): len(ids)]])

    # 3) upsert to vector store
    store.upsert(collection=collection, ids=ids, vectors=vecs, documents=docs, metadatas=metas)

    # 4) index full Markdown in keyword sidecar (doc-level)
    if keyword_index is not None:
        keyword_index.upsert([(doc_id, markdown)])
