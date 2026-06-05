from __future__ import annotations

import chromadb


def make_chroma_client(
    engine: str = "http",
    path: str = ".chroma",
    host: str = "chroma",
    port: int = 8000,
    ssl: bool = False,
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
) -> chromadb.ClientAPI:
    """Build and return a Chroma client for the requested *engine*.

    Args:
        engine: Deployment mode.  One of:

            - ``"memory"``     — in-process ephemeral client (no persistence,
              ideal for tests and one-shot scripts).
            - ``"persistent"`` — on-disk ``PersistentClient`` stored at *path*.
            - ``"http"``       — (default) ``HttpClient`` connecting to a
              standalone Chroma server or Docker container.
            - ``"cloud"``      — ``CloudClient`` connecting to Chroma Cloud;
              requires *api_key*, *tenant*, and *database*.

        path: Local directory for ``"persistent"`` mode.
        host: Hostname for ``"http"`` mode.
        port: Port for ``"http"`` mode.
        ssl: Enable TLS for ``"http"`` mode.
        api_key: API key for ``"cloud"`` mode.
        tenant: Tenant name for ``"cloud"`` mode.
        database: Database name for ``"cloud"`` mode.

    Returns:
        A fully-constructed Chroma client object.

    Raises:
        ValueError: If *engine* is not one of the recognised values.

    """
    engine = engine.lower()
    if engine == "memory":
        return chromadb.EphemeralClient()
    if engine == "persistent":
        return chromadb.PersistentClient(path=path)
    if engine == "http":
        return chromadb.HttpClient(host=host, port=port, ssl=ssl)
    if engine == "cloud":
        return chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)
    raise ValueError(
        f"Unknown engine '{engine}'. Choose: 'memory', 'persistent', 'http', or 'cloud'."
    )
