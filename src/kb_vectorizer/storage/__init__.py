"""Vector stores.

The backends are resolved lazily (PEP 562), for the same reason as
:mod:`kb_vectorizer.embedding`: importing anything from this package — even
just :mod:`.interfaces` — runs this ``__init__`` first, and eagerly importing
every backend made a Qdrant-only consumer load ``chromadb`` as well (which in
turn pulled the torch-backed embedders in through ``embedding/__init__``).
Now each backend's dependencies load only when that backend is asked for.

``QdrantStore``/``make_qdrant_client`` are listed in ``__all__`` only when
``qdrant-client`` (the ``qdrant`` extra) is installed, as before.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

from .interfaces import BaseVectorStore, StoredRecord

if TYPE_CHECKING:
    from .chroma_client_factory import make_chroma_client
    from .chromadb_store import ChromaStore
    from .qdrant_client_factory import make_qdrant_client
    from .qdrant_store import QdrantStore

_LAZY = {
    'ChromaStore': '.chromadb_store',
    'make_chroma_client': '.chroma_client_factory',
    'QdrantStore': '.qdrant_store',
    'make_qdrant_client': '.qdrant_client_factory',
}

__all__ = [
    'BaseVectorStore',
    'ChromaStore',
    'StoredRecord',
    'make_chroma_client',
]

if importlib.util.find_spec('qdrant_client') is not None:
    __all__ += ['QdrantStore', 'make_qdrant_client']


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # resolve once; later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
