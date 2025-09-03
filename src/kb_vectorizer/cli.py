import json
import os
from pathlib import Path
from typing import Any

import typer

from kb_vectorizer.embedding.sentence_tranformers_embedder import SetenceTransformerEmbedder
from kb_vectorizer.ingestion.mysql_ingestor import MySQLIngestor
from kb_vectorizer.pipeline.stream_ingest_pipeline import custom_streaming_pipeline
from kb_vectorizer.postprocessing.postprocess import flatten_chroma_result, group_by_doc_id, resolve_parent_documents
from kb_vectorizer.rerank.mmr_reranker import MMRReranker
from kb_vectorizer.storage.chroma_client_factory import make_chroma_client
from kb_vectorizer.storage.chromadb_store import ChromaStore
from kb_vectorizer.utils._json_default import _json_default

app = typer.Typer(help="Vectorize a knowledge base (ingest → chunk → embed → index)")

@app.command()
def ingest(path: str):
    """Ingest files from PATH into normalized documents."""
    typer.echo(f"Ingesting from {path}...")

@app.command()
def build_index(model: str = "all-MiniLM-L6-v2", kb_dir: str = "data/normalized", out: str = "index.faiss"):
    """Embed docs and write a FAISS index."""
    typer.echo(f"Embedding {kb_dir} with {model} → {out}")


@app.command()
def ingest_from_mysql(
    url: str = typer.Argument(..., help="SQLAlchemy MySQL URL"),
    query: str = typer.Argument(..., help="SQL query (use named params like :id)"),
    stream: bool = typer.Option(False, "--stream", help="Stream the response (JSONL to stdout)"),
    params: str | None = typer.Option(
        None,
        "--params",
        help='JSON string of parameters, e.g. \'{"id": 123}\'',
    ),
):
    try:
        parsed_params: dict[str, Any] | None = json.loads(params) if params else None
    except json.JSONDecodeError as e:
        typer.secho(f"Invalid --params JSON: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    ingestor = MySQLIngestor(url=url, stream=stream)

    try:
        results = ingestor.ingest(query=query, params=parsed_params)
    except Exception as e:
        typer.secho(f"Query failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Streaming mode: iterator of row dicts -> JSONL to stdout
    if stream and not isinstance(results, dict):
        for row in results:
            typer.echo(json.dumps(row, ensure_ascii=False, default=_json_default))
        return

    # Non-streaming: either list[dict] or {"rowcount": int}
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default))

@app.command()
def run_pipeline():
    
    DB_URL = "mysql+pymysql://root:root@localhost:3306/crmv3"
    DB_QUERY = """
        SELECT *, data_cadastro AS updated_at, codigo AS id
        FROM base_conhecimento
        WHERE data_cadastro > :last_updated_at
        ORDER BY data_cadastro ASC
    """
    COLLECTION_NAME = "my_knowledge_base"
    CHECKPOINT_KEY = "mysql_ingestion_stream"

    custom_streaming_pipeline(
        db_url=DB_URL,
        db_query=DB_QUERY,
        collection_name=COLLECTION_NAME,
        checkpoint_key=CHECKPOINT_KEY,
    )

@app.command("chroma-query")
def chroma_query_cmd(
    query: str = typer.Argument(...),
    engine: str = typer.Option("persistent"),
    chroma_path: Path = typer.Option(Path("var/chroma")),
    collection: str = typer.Option("kb"),
    k: int = typer.Option(12, help="retrieve up to k chunks before post-processing"),
    return_mode: str = typer.Option("docs", help="chunks|parent|docs"),
    unique_per_doc: bool = typer.Option(True, help="deduplicate by doc_id"),
    # MMR toggles:
    mmr: bool = typer.Option(False, help="Enable Maximal Marginal Relevance re-ranking"),
    mmr_lambda: float = typer.Option(0.5, help="MMR lambda (0=diversity, 1=relevance)"),
    mmr_top_n: int = typer.Option(8, help="How many items to keep after MMR (≤ k)"),
):
    client = make_chroma_client(engine=engine, path=str(chroma_path))
    store = ChromaStore(client)

    embedder = SetenceTransformerEmbedder(local=True)
    embeds = embedder.embed(texts=[query])

    include = ["metadatas", "documents", "distances"]
    if mmr:
        rerank = MMRReranker()
        include.append("embeddings")  # needed for intra-set similarity

    raw = store.query(collection=collection, query_texts=[query], query_vectors=embeds.vectors, n_results=k, include=include)
    items = flatten_chroma_result(raw)

    # Optional MMR re-ranking at the CHUNK level (improves diversity)
    if mmr and items and items[0].get("embedding") is not None:
        # Convert distances->similarities (assuming cosine distance)
        sims = [1 - (it["distance"] if it["distance"] is not None else 1.0) for it in items]
        cand_embs = [it["embedding"] for it in items]
        candidates = [{"similarity": s, "embedding": e} for s, e in zip(sims, cand_embs, strict=True)]
        keep_idx = rerank.rerank(candidates=candidates, top_n=mmr_top_n)
        items = [items[i] for i in keep_idx]

    # Unique per article?
    hits = group_by_doc_id(items) if unique_per_doc else items

    # Return mode
    if return_mode == "docs":
        result = resolve_parent_documents(hits)
        typer.echo(json.dumps({"mode": "docs", "results": result}, ensure_ascii=False, indent=2))
    elif return_mode == "parent":
        # (optional) implement a parent-window slice here if desired
        result = resolve_parent_documents(hits)
        typer.echo(json.dumps({"mode": "parent", "results": result}, ensure_ascii=False, indent=2))
    else:
        for i in hits:
            i.pop("embedding", None)
        typer.echo(json.dumps({"mode": "chunks", "results": hits}, ensure_ascii=False, indent=2, default=_json_default))

@app.command("run")
def run():
    os.system('tail -f /dev/null')
    
if __name__ == "__main__":
    app()
