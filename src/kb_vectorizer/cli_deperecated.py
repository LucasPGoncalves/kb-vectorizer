import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from kb_vectorizer.chunking.recursive_token_chunker import TiktokenRecursiveChunker
from kb_vectorizer.embedding.google_embedder import VertexAIEmbedder
from kb_vectorizer.embedding.sentence_tranformers_embedder import HuggingFaceEmbedder
from kb_vectorizer.embedding.interfaces import EmbedRequest
from kb_vectorizer.preprocessing.html_preprocessor import preprocess_html
from kb_vectorizer.rerank.mmr import mmr_rerank
from kb_vectorizer.postprocessing.postprocess import flatten_chroma_result, group_by_doc_id, resolve_parent_documents
from kb_vectorizer.storage.chroma_client_factory import make_chroma_client
from kb_vectorizer.storage.chromadb_store import ChromaStore
import typer

from .ingestion.sql_loader import ingest_table, ingest_sql
from .ingestion.async_sql_loader import a_ingest_table, a_ingest_sql

from kb_vectorizer.ingestion.mysql_ingestor import MySQLIngestor

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
def sql_ingest_table(
    url: str,
    table: str,
    schema: str = typer.Option(None),
    where: str = typer.Option(None),
    pk: str = typer.Option(None),
    updated_at: str = typer.Option(None),
    columns: str = typer.Option(None, help="Comma-separated list"),
    checkpoint_file: str = typer.Option(None, help="Path to JSON checkpoint file"),
    limit: int = typer.Option(5),
):
    cols = [c.strip() for c in columns.split(",")] if columns else None
    gen = ingest_table(
        url, table, schema=schema, where=where, pk_column=pk, updated_at_column=updated_at,
        columns=cols, checkpoint_path=checkpoint_file
    )
    for i, doc in zip(range(limit), gen):
        typer.echo(json.dumps(asdict(doc), default=str))

@app.command()
def sql_ingest(
    url: str,
    sql: str,
    pk_field: str = typer.Option(None),
    updated_at_field: str = typer.Option(None),
    checkpoint_file: str = typer.Option(None),
    since_param_name: str = typer.Option("since"),
    limit: int = typer.Option(5),
):
    gen = ingest_sql(
        url, sql, pk_field_in_result=pk_field, updated_at_field_in_result=updated_at_field,
        checkpoint_path=checkpoint_file, since_param_name=since_param_name
    )
    for i, doc in zip(range(limit), gen):
        typer.echo(json.dumps(asdict(doc), default=str))

@app.command("a-sql-ingest-2")
def sql_ingest_2(
    url: str,
    sql: str,
    pk_field: str = typer.Option(None),
    updated_at_field: str = typer.Option(None),
    checkpoint_file: str = typer.Option(None),
    since_param_name: str = typer.Option("since"),
    limit: int = typer.Option(5),
):
    async def _main():
        gen = await asyncio.to_thread(
            ingest_sql, url, sql,
            pk_field_in_result=pk_field,
            updated_at_field_in_result=updated_at_field,
            checkpoint_path=checkpoint_file,
            since_param_name=since_param_name,
        )
        for i, doc in zip(range(limit), gen):
            typer.echo(json.dumps(asdict(doc), default=str))

    asyncio.run(_main())

@app.command("ingest-db-table-async")
def ingest_db_table_async(
    url_async: str = typer.Argument(..., help="Async DB URL, e.g., mysql+pymysql://user:password@host:port/db_name"),
    table: str = typer.Option(..., help="Table name"),
    schema: str = typer.Option(None, help="Schema (optional)"),
    where: str = typer.Option(None, help='Optional WHERE, e.g. "updated_at >= :ts"'),
    pk: str = typer.Option(None, help="Primary key column for source_id"),
    updated_at: str = typer.Option(None, help="Updated-at column"),
    columns: str = typer.Option(None, help="Comma-separated columns to select"),
    checkpoint_file: str = typer.Option(None, help="Path to JSON checkpoint file"),
    limit: int = typer.Option(5, help="Print first N docs and stop"),
    chunk_size: int = typer.Option(1000, help="Rows per batch for streaming"),
):
    async def _main():
        cols = [c.strip() for c in columns.split(",")] if columns else None
        count = 0
        async for doc in a_ingest_table(
            url_async,
            table,
            schema=schema,
            where=where,
            pk_column=pk,
            updated_at_column=updated_at,
            columns=cols,
            checkpoint_path=checkpoint_file,
            chunk_size=chunk_size,
        ):
            typer.echo(json.dumps(asdict(doc), default=str))
            count += 1
            if count >= limit:
                break
    asyncio.run(_main())


@app.command("ingest-db-sql-async")
def ingest_db_sql_async(
    url_async: str = typer.Argument(..., help="Async DB URL, e.g., mysql+aiomysql://user:pw@host:3306/db"),
    sql: str = typer.Argument(..., help="Raw SELECT (JOINs allowed, can include :since parameter)"),
    pk_field: str = typer.Option(None, help="Field in result used to build source_id"),
    updated_at_field: str = typer.Option(None, help="Field in result used for incremental checkpoint"),
    checkpoint_file: str = typer.Option(None, help="Path to JSON checkpoint file"),
    since_param_name: str = typer.Option("since", help="Param name bound to last checkpoint (if any)"),
    limit: int = typer.Option(5, help="Print first N docs and stop"),
    chunk_size: int = typer.Option(1000, help="Rows per batch for streaming"),
):
    async def _main():
        count = 0
        async for doc in a_ingest_sql(
            url_async,
            sql,
            pk_field_in_result=pk_field,
            updated_at_field_in_result=updated_at_field,
            since_param_name=since_param_name,
            checkpoint_path=checkpoint_file,
            chunk_size=chunk_size,
        ):
            typer.echo(json.dumps(asdict(doc), default=str))
            count += 1
            if count >= limit:
                break
    asyncio.run(_main())  

@app.command("preprocess-html")
def preprocess_html_cmd(
    input_path: Path = typer.Argument(..., help="Path to an .html or .txt containing HTML"),
    out_dir: Path = typer.Option(Path("data/normalized"), help="Output directory for text + images"),
    image_subdir: str = typer.Option("images"),
    doc_stem: str = typer.Option("doc"),
    keep_remote_img: bool = typer.Option(True, help="Keep remote http(s) images as Markdown"),
    write_outputs: bool = typer.Option(True, help="Write markdown + text files"),
):
    html = input_path.read_text(encoding="utf-8")
    result = preprocess_html(
        html,
        out_dir=out_dir,
        image_subdir=image_subdir,
        doc_stem=doc_stem,
        keep_remote_img=keep_remote_img,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if write_outputs:
        (out_dir / f"{doc_stem}.md").write_text(result.markdown, encoding="utf-8")
        (out_dir / f"{doc_stem}.txt").write_text(result.text, encoding="utf-8")

    manifest = [
        {
            "path": str(img.path),
            "mime": img.mime,
            "sha256": img.sha256,
            "alt": img.alt,
            "title": img.title,
            "caption": img.caption,
        }
        for img in result.images
    ]
    typer.echo(json.dumps({"markdown": f"{doc_stem}.md", "text": f"{doc_stem}.txt", "images": manifest}, ensure_ascii=False))
    typer.echo(result)

@app.command("chunk-markdown")
def chunk_markdown_cmd(
    md_path: Path = typer.Argument(..., help="Path to a Markdown file to chunk"),
    doc_id: str = typer.Option("doc-1", help="Stable document id for chunk ids"),
    chunk_size: int = typer.Option(500, help="Target tokens per chunk"),
    chunk_overlap: int = typer.Option(60, help="Overlap tokens (~10-15% of size)"),
    encoding: str = typer.Option(None, help='tiktoken encoding, e.g., "cl100k_base"'),
    model: str = typer.Option(None, help='tiktoken model, e.g., "gpt-4o-mini"'),
    out_path: Path = typer.Option(Path("chunks.ndjson"), help="Output NDJSON with chunks"),
):
    text = md_path.read_text(encoding="utf-8")
    chunker = TiktokenRecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding=encoding,
        model=model,
    )
    chunks = chunker.split(text, metadata={"source_path": str(md_path)}, doc_id=doc_id)

    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps({"id": ch.id, "text": ch.text, "metadata": ch.metadata}, ensure_ascii=False) + "\n")

    typer.echo(f"Wrote {len(chunks)} chunks → {out_path}")

@app.command("embed-chunks")
def embed_chunks_cmd(
    chunks_path: Path = typer.Argument(..., help="NDJSON with {'id','text',...} per line"),
    out_path: Path = typer.Option(Path("embeddings.ndjson"), help="Output NDJSON with {'id','vector'}"),
    provider: str = typer.Option("hf"),
    model: str = typer.Option("text-embedding-3-small"),
    batch_size: int = typer.Option(256),
    local: bool = typer.Option(True),
    tei_url: str = typer.Option("http://localhost:8000"),
    # Provider-specific options
    openai_api_key: str = typer.Option(None),
    vertex_project: str = typer.Option(None),
    vertex_location: str = typer.Option("us-central1"),
    vertex_token: str = typer.Option(None),
    ollama_host: str = typer.Option("http://localhost:11434"),
    oai_compat_base: str = typer.Option("", help="Base URL for OpenAI-compatible embeddings"),
    oai_compat_key: str = typer.Option(None),
):
    # choose embedder
    if provider == "hf":
        embedder = HuggingFaceEmbedder(model_id=model, local=local, tei_url=tei_url, batch_size=batch_size)
    elif provider == "vertex":
        embedder = VertexAIEmbedder(model=model, project_id=vertex_project, location=vertex_location, access_token=vertex_token, max_batch_size=batch_size)
    else:
        raise typer.BadParameter("Unknown provider")

    # stream-read chunks and embed in batches
    ids, texts = [], []
    with chunks_path.open("r", encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as out:
        def flush():
            nonlocal ids, texts
            if not texts:
                return
            res = embedder.embed(EmbedRequest(texts=texts, ids=ids))
            for cid, vec in zip(ids, res.vectors):
                out.write(json.dumps({"id": cid, "vector": vec, "model": res.model, "dim": res.dimension}) + "\n")
            ids, texts = [], []

        for line in fh:
            obj = json.loads(line)
            ids.append(obj["id"])
            texts.append(obj["text"])
            if len(texts) >= embedder.max_batch_size:
                flush()
        flush()

    typer.echo(f"Wrote embeddings → {out_path}")

@app.command("chroma-load")
def chroma_load_cmd(
    embeddings_path: Path = typer.Argument(...),
    engine: str = typer.Option("persistent", help="persistent|http|memory|cloud"),
    chroma_path: Path = typer.Option(Path("var/chroma"), help="For persistent engine"),
    collection: str = typer.Option("kb-vec"),
    http_host: str = typer.Option("localhost"), http_port: int = typer.Option(8000), http_ssl: bool = typer.Option(False),
    cloud_api_key: str = typer.Option(None), cloud_tenant: str = typer.Option(None), cloud_database: str = typer.Option(None),
    batch_size: int = typer.Option(2048),
):
    client = make_chroma_client(
        engine=engine,
        path=str(chroma_path),
        host=http_host, port=http_port, ssl=http_ssl,
        api_key=cloud_api_key, tenant=cloud_tenant, database=cloud_database,
    )
    store = ChromaStore(client)
    store.create_collection(collection)

    ids, vecs, docs, metas = [], [], [], []
    with embeddings_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            ids.append(obj["id"])
            vecs.append(obj["vector"])
            # optional passthroughs if present in your chunks:
            docs.append(obj.get("document"))
            metas.append(obj.get("metadata") or {})

            if len(ids) >= batch_size:
                store.upsert(collection=collection, ids=ids, vectors=vecs, documents=docs, metadatas=metas)
                ids, vecs, docs, metas = [], [], [], []
        if ids:
            store.upsert(collection=collection, ids=ids, vectors=vecs, documents=docs, metadatas=metas)

    store.persist()
    typer.echo(f"Loaded embeddings into '{collection}' using engine={engine}")

@app.command("chroma-query")
def chroma_query_cmd(
    query: str = typer.Argument(...),
    engine: str = typer.Option("persistent"),
    chroma_path: Path = typer.Option(Path("var/chroma")),
    collection: str = typer.Option("kb"),
    k: int = typer.Option(12, help="retrieve up to k chunks before post-processing"),
    return_mode: str = typer.Option("docs", help="chunks|parent|docs"),  # default -> docs
    unique_per_doc: bool = typer.Option(True, help="deduplicate by doc_id"),
    # MMR toggles:
    mmr: bool = typer.Option(False, help="Enable Maximal Marginal Relevance re-ranking"),
    mmr_lambda: float = typer.Option(0.5, help="MMR lambda (0=diversity, 1=relevance)"),
    mmr_top_n: int = typer.Option(8, help="How many items to keep after MMR (≤ k)"),
    # HTTP / Cloud params omitted for brevity ...
):
    client = make_chroma_client(engine=engine, path=str(chroma_path))
    store = ChromaStore(client)

    include = ["metadatas", "documents", "distances"]
    if mmr:
        include.append("embeddings")  # needed for intra-set similarity

    raw = store.query(collection=collection, query_texts=[query], n_results=k, include=include)
    items = flatten_chroma_result(raw)

    # Optional MMR re-ranking at the CHUNK level (improves diversity)
    if mmr and items and items[0].get("embedding") is not None:
        # Convert distances->similarities (assuming cosine distance)
        sims = [1 - (it["distance"] if it["distance"] is not None else 1.0) for it in items]
        cand_embs = [it["embedding"] for it in items]
        keep_idx = mmr_rerank(query_similarity=sims, candidate_embeddings=cand_embs,
                              lambda_mult=mmr_lambda, top_n=mmr_top_n)
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
        typer.echo(json.dumps({"mode": "chunks", "results": hits}, ensure_ascii=False, indent=2))

@app.command()
def ingest_from_mysql(
    url: str = typer.Argument(..., help="SQLAlchemy MySQL URL"),
    query: str = typer.Argument(..., help="SQL query (use named params like :id)"),
    stream: bool = typer.Option(False, "--stream", help="Stream the response (JSONL to stdout)"),
    params: Optional[str] = typer.Option(
        None,
        "--params",
        help='JSON string of parameters, e.g. \'{"id": 123}\'',
    ),
):
    try:
        parsed_params: Optional[Dict[str, Any]] = json.loads(params) if params else None
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
            typer.echo(json.dumps(asdict(row), ensure_ascii=False, default=str))
        return

    # Non-streaming: either list[dict] or {"rowcount": int}
    typer.echo(json.dumps(asdict(results), ensure_ascii=False, indent=2, default=str))

@app.command("run")
def run():
    os.system('tail -f /dev/null')
    

# if __name__ == "__main__":
#     app()
