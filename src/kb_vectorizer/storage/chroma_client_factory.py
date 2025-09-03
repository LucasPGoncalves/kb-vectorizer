from __future__ import annotations

import chromadb


def make_chroma_client(
        engine: str = "http",
        path: str = ".chroma",
        host: str = "localhost", port: int = 8006, ssl: bool = False,
        api_key: str | None = None, tenant: str | None = None, database: str | None = None
):
    
    engine = engine.lower()
    if engine == "persistent":
        return chromadb.PersistentClient(path=path)
    if engine == "http":
        return chromadb.HttpClient(host=host, port=port, ssl=ssl)
    if engine == "memory":
        return chromadb.Client()
    if engine == "cloud":
        return chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)
    raise ValueError(f"Unknown engine '{engine}'")