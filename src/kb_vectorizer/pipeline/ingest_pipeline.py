from pathlib import Path

import chromadb

from kb_vectorizer.chunking.recursive_token_chunker import TiktokenRecursiveChunker
from kb_vectorizer.embedding.sentence_tranformers_embedder import SetenceTransformerEmbedder
from kb_vectorizer.ingestion.mysql_ingestor import MySQLIngestor
from kb_vectorizer.pipeline.indexer import index_document
from kb_vectorizer.storage.chromadb_store import ChromaStore
from kb_vectorizer.utils.checkpoint_v2 import Checkpoint


def main():
    # 1. Initialize the components
    ingestor = MySQLIngestor(url="mysql+pymysql://user:password@host/db")
    client = chromadb.HttpClient(host="host.docker.internal", port=8000, ssl=False)
    store = ChromaStore(client=client)
    checkpoint = Checkpoint(path=Path("checkpoint.json"))
    chunker = TiktokenRecursiveChunker(model="gpt-4")
    embedder = SetenceTransformerEmbedder(local=True) # Replace with your actual embedder

    # 2. Get the last updated_at from the checkpoint
    _, last_updated_at = checkpoint.get("last_run") or (None, "1970-01-01 00:00:00")

    # 3. Fetch the data from the MySQL database
    query = "SELECT * FROM your_table WHERE updated_at > :last_updated_at"
    params = {"last_updated_at": last_updated_at}
    documents = ingestor.ingest(query=query, params=params)

    # 4. Index the documents into the ChromaStore
    for doc in documents:
        index_document(
            doc_id=doc["id"],
            markdown=doc["content"],
            metadata={"source": "mysql"},
            store=store,
            collection="my_collection",
            chunker=chunker,
            embedder=embedder,
        )

    # 5. Save a checkpoint with the last updated_at timestamp
    if documents:
        last_updated_at = documents[-1]["updated_at"]
        checkpoint.put("last_run", str(last_updated_at), documents[-1]["id"])

if __name__ == "__main__":
    main()