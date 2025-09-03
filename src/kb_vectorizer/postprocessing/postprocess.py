# src/kb_vectorizer/retrieval/postprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Any


def flatten_chroma_result(raw: dict[str, Any]) -> list[dict[str, Any]]:
    ids = raw.get("ids", [[]])[0]
    dists = raw.get("distances", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    embs = raw.get("embeddings", [[]])[0] if raw.get("embeddings", None) is not None else []
    out = []
    for i, _id in enumerate(ids):
        out.append({
            "id": _id,
            "distance": dists[i] if i < len(dists) else None,
            "document": docs[i] if i < len(docs) else None,
            "metadata": metas[i] if i < len(metas) else {},
            "embedding": embs[i] if len(embs) > i else None,
        })
    return out

def group_by_doc_id(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for it in items:
        doc_id = (it.get("metadata") or {}).get("doc_id") or it["id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(it)
    return unique

def resolve_parent_documents(unique_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for it in unique_hits:
        meta = it.get("metadata") or {}
        md_path = meta.get("source_path")
        full_doc = Path(md_path).read_text(encoding="utf-8") if md_path and Path(md_path).exists() else None
        out.append({
            "doc_id": meta.get("doc_id") or it["id"],
            "distance": it.get("distance"),
            "metadata": meta,
            "markdown": full_doc,  # full article (Markdown with image refs)
        })
    return out
