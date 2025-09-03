from collections.abc import Iterator
from pathlib import Path

import chromadb

from kb_vectorizer.chunking.recursive_token_chunker import TiktokenRecursiveChunker
from kb_vectorizer.embedding.sentence_tranformers_embedder import SetenceTransformerEmbedder
from kb_vectorizer.ingestion.mysql_ingestor import MySQLIngestor, Row
from kb_vectorizer.preprocessing.html_preprocessor import HTMLProcessor
from kb_vectorizer.preprocessing.json_to_html_processor import JSONToHTMLPreprocessor
from kb_vectorizer.storage.chromadb_store import ChromaStore
from kb_vectorizer.utils.checkpoint_v2 import Checkpoint


def custom_streaming_pipeline(
    db_url: str,
    db_query: str,
    collection_name: str,
    checkpoint_key: str,
    chroma_path: str = "/tmp/chromadb",
    batch_size: int = 100
):
    """
    A memory-efficient pipeline that streams data from MySQL,
    chunks, embeds, and stores it in ChromaDB.
    """
    # 1. Initialize Components
    print("--- Initializing components for STREAMING mode ---")
    # Set stream=True here. This is the key change.
    ingestor = MySQLIngestor(url=db_url, stream=True)
    client = chromadb.HttpClient(host="localhost", port=8006, ssl=False)
    store = ChromaStore(client=client)
    store.create_collection(name=collection_name)
    checkpoint = Checkpoint(path=Path("pipeline_checkpoint.json"))
    chunker = TiktokenRecursiveChunker()
    embedder = SetenceTransformerEmbedder()
    html_processor = HTMLProcessor()
    json_to_html = JSONToHTMLPreprocessor(['roteiro', 'problema', 'solucao'], ['tipo', 'sistema', 'consultor', 'versao'])

    # 2. Load Checkpoint
    print(f"--- Loading checkpoint for key: '{checkpoint_key}' ---")
    last_run_data = checkpoint.get(checkpoint_key)
    last_updated_at = "1970-01-01 00:00:00"
    if last_run_data:
        last_updated_at = last_run_data[0]
    print(f"Found last updated_at: {last_updated_at}")

    # 3. Ingest Data from MySQL as a Stream (Generator)
    print("--- Ingesting data from MySQL as a stream ---")
    params = {"last_updated_at": last_updated_at}
    # 'documents' is now an Iterator, not a list.
    documents_iterator: Iterator[Row] = ingestor.ingest(query=db_query, params=params)

    latest_updated_at = last_updated_at
    latest_id = -1
    processed_count = 0

    # 4. Process the Stream of Documents
    print("--- Processing and storing documents from the stream ---")
    # The for loop consumes the generator, fetching rows as needed.
    for doc in documents_iterator:
        processed_count += 1
        html, metadata = json_to_html.process(document=doc)
        processed_html = html_processor.process(html=html, out_dir='out')
        doc_id = str(doc.get("id", "unknown_id"))
        #markdown_content = doc.get("content", "")
        print(f"Processing document with ID: {doc_id}")

        chunks = chunker.chunk(processed_html.markdown, metadata=metadata, doc_id=doc_id)
        
        # Batch processing for embedding and upserting remains the same
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            buf_texts = [ch.text for ch in batch_chunks]
            buf_ids = [ch.id for ch in batch_chunks]
            
            res = embedder.embed(texts=buf_texts)
            
            store.upsert(
                collection=collection_name, 
                ids=buf_ids, 
                vectors=res.vectors, 
                documents=buf_texts, 
                metadatas=[ch.metadata for ch in batch_chunks]
            )
        
        if "updated_at" in doc:
            latest_updated_at = str(doc["updated_at"])
            latest_id = doc["id"]

    if processed_count == 0:
        print("No new documents to process. Pipeline finished.")
        return
        
    # 5. Save Checkpoint
    if latest_id != -1:
        print("--- Saving checkpoint ---")
        checkpoint.put(key=checkpoint_key, updated_at=latest_updated_at, last_id=latest_id)
        print(f"Processed {processed_count} documents. Checkpoint saved with updated_at: {latest_updated_at}")

    print("--- Pipeline finished successfully ---")
