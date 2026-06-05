from __future__ import annotations

from qdrant_client import QdrantClient


def make_qdrant_client(
    engine: str = "http",
    url: str = "http://localhost:6333",
    path: str = ".qdrant",
    api_key: str | None = None,
    prefer_grpc: bool = False,
) -> QdrantClient:
    """Build and return a Qdrant client for the requested *engine*.

    Args:
        engine: Deployment mode.  One of:

            - ``"memory"``     — in-process ephemeral client (no persistence,
              ideal for tests and one-shot scripts).
            - ``"persistent"`` — on-disk storage at *path*.
            - ``"http"``       — (default) connects to a Qdrant server at
              *url*, optionally authenticated with *api_key*.

        url: Full base URL for ``"http"`` mode, e.g.
            ``"http://localhost:6333"`` or ``"https://xyz.cloud.qdrant.io"``.
        path: Local directory for ``"persistent"`` mode.
        api_key: Optional API key for Qdrant Cloud or secured deployments.
        prefer_grpc: Use gRPC transport instead of HTTP for ``"http"`` mode.
            Offers lower latency for bulk operations but requires the gRPC
            port (default 6334) to be open.

    Returns:
        A fully-constructed :class:`qdrant_client.QdrantClient`.

    Raises:
        ValueError: If *engine* is not one of the recognised values.

    """
    engine = engine.lower()
    if engine == "memory":
        return QdrantClient(":memory:")
    if engine == "persistent":
        return QdrantClient(path=path)
    if engine == "http":
        return QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
    raise ValueError(
        f"Unknown engine '{engine}'. Choose: 'memory', 'persistent', or 'http'."
    )
