from typing import Any

from kb_vectorizer.config import HybridConfig
from kb_vectorizer.fusion.rrf_fusor import rrf_fuse
from kb_vectorizer.postprocessing.postprocess import flatten_chroma_result, group_by_doc_id, resolve_parent_documents
from kb_vectorizer.rerank.interfaces import BaseReranker
from kb_vectorizer.rerank.mmr_reranker import MMRReranker
from kb_vectorizer.retrieval.keyword_index import KeywordIndex
from kb_vectorizer.storage.chromadb_store import ChromaStore


def retrieve_docs(
    *,
    query: str,
    store: ChromaStore,
    collection: str,
    config: HybridConfig,
    keyword_index: KeywordIndex | None = None,
    reranker: BaseReranker | None = None,
) -> list[dict[str, Any]]:
    include = ["metadatas", "documents", "distances"]
    # Only ask for embeddings if needed (MMR or a reranker that needs them)
    needs_embeddings = config.use_mmr or getattr(reranker, "needs_embeddings", False)
    if needs_embeddings:
        include.append("embeddings")

    raw = store.query(
        collection=collection,
        query_texts=[query],
        n_results=config.k_vec,
        include=include,
    )  # Chroma query API supports include fields.

    items = flatten_chroma_result(raw)

    # Rerank at the CHUNK level (preferred), else optional MMR
    if reranker is not None:
        order = reranker.rerank(query, items, top_n=min(config.mmr_top_n, len(items)))
        items = [items[i] for i in order]
    elif config.use_mmr and items and items[0].get("embedding") is not None:
        sims = [1 - (it["distance"] if it["distance"] is not None else 1.0) for it in items]
        cand_embs = [it["embedding"] for it in items]
        keep_idx = MMRReranker.rerank(
            query_similarity=sims,
            candidate_embeddings=cand_embs,
            lambda_mult=config.mmr_lambda,
            top_n=min(config.mmr_top_n, len(items)),
        )
        items = [items[i] for i in keep_idx]

    # Collapse chunk hits to unique docs (parent-doc retrieval)
    doc_hits = group_by_doc_id(items)
    vec_ranked_doc_ids = [(it["metadata"].get("doc_id") or it["id"]) for it in doc_hits]

    # Optional keyword channel (hybrid)
    kw_ranked_doc_ids: list[str] = []
    if config.use_hybrid and keyword_index is not None:
        kw_results = keyword_index.search(query, k=config.k_kw)
        kw_ranked_doc_ids = [doc_id for (doc_id, _score) in kw_results]

    # RRF fuse doc id rankings
    rankings = [vec_ranked_doc_ids] + ([kw_ranked_doc_ids] if kw_ranked_doc_ids else [])
    fused = rrf_fuse(rankings, k=config.rrf_k) if len(rankings) > 1 else [(d, 0.0) for d in vec_ranked_doc_ids]
    fused_doc_ids = [doc_id for (doc_id, _score) in fused]

    # Reorder and return full Markdown docs (default behavior)
    by_doc = { (it["metadata"].get("doc_id") or it["id"]) : it for it in doc_hits }
    ordered_hits = [by_doc[d] for d in fused_doc_ids if d in by_doc] or doc_hits
    return resolve_parent_documents(ordered_hits)
