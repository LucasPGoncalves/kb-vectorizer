"""Embedders.

The concrete embedders are resolved lazily (PEP 562). Importing anything from
this package — even just :mod:`.interfaces` — runs this ``__init__`` first, and
two of the embedders ``import torch`` (plus sentence-transformers/transformers)
at module level: roughly 430MB of RSS per process, paid by every consumer that
only needed :class:`BaseEmbedder` or talks to a remote embedding service.
Resolving on first attribute access keeps ``from kb_vectorizer.embedding import
SentenceTransformerEmbedder`` working unchanged while making it the only thing
that pays for torch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .interfaces import BaseEmbedder, EmbedResponse

if TYPE_CHECKING:
    from .cloud_embedder import CloudEmbedder
    from .local_embedder import LocalEmbedder
    from .sentence_transformers_embedder import SentenceTransformerEmbedder

_LAZY = {
    'SentenceTransformerEmbedder': '.sentence_transformers_embedder',
    'CloudEmbedder': '.cloud_embedder',
    'LocalEmbedder': '.local_embedder',
}

__all__ = [
    'BaseEmbedder',
    'EmbedResponse',
    'SentenceTransformerEmbedder',
    'CloudEmbedder',
    'LocalEmbedder',
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # resolve once; later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
