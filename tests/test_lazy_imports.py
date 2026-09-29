"""The package ``__init__`` files must not drag in torch or chromadb.

Importing any submodule runs every parent ``__init__`` first, so an eager
``from .sentence_transformers_embedder import ...`` in ``embedding/__init__.py``
made even ``kb_vectorizer.storage.interfaces`` cost ~430MB of torch +
sentence-transformers — for consumers that only use Qdrant with a remote
embedder and never touch a local model.

Each check runs in a fresh interpreter: other tests in this session import
torch directly, so ``sys.modules`` here would already contain it.
"""

import subprocess
import sys

import pytest

HEAVY = ('torch', 'sentence_transformers', 'transformers', 'chromadb')


def _heavy_modules_after(code: str) -> list[str]:
    script = f"import sys\n{code}\nprint(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
    return [m for m in out.stdout.strip().split(',') if m]


@pytest.mark.parametrize(
    'code',
    [
        'import kb_vectorizer.embedding.interfaces',
        'import kb_vectorizer.storage.interfaces',
        'import kb_vectorizer.rerank.interfaces',
        'from kb_vectorizer.embedding import BaseEmbedder, EmbedResponse',
        'from kb_vectorizer.storage import BaseVectorStore, StoredRecord',
        'from kb_vectorizer.storage.qdrant_store import QdrantStore',
        'from kb_vectorizer.storage import QdrantStore, make_qdrant_client',
        'from kb_vectorizer.embedding import CloudEmbedder',
        'from kb_vectorizer.fusion.rrf_fusor import RRFFusor',
    ],
)
def test_paths_without_a_local_model_import_nothing_heavy(code):
    """Interfaces, Qdrant and remote embedders load without torch or chromadb."""
    assert _heavy_modules_after(code) == []


def test_public_names_still_resolve():
    """Lazy, not removed: the old package-level imports keep working."""
    loaded = _heavy_modules_after(
        'from kb_vectorizer.embedding import SentenceTransformerEmbedder, LocalEmbedder\n'
        'from kb_vectorizer.storage import ChromaStore, make_chroma_client'
    )
    assert 'torch' in loaded
    assert 'chromadb' in loaded


def test_unknown_name_still_raises_attribute_error():
    """``__getattr__`` must not swallow typos into ``None``."""
    import kb_vectorizer.embedding as embedding
    import kb_vectorizer.storage as storage

    with pytest.raises(AttributeError):
        embedding.DoesNotExist  # noqa: B018
    with pytest.raises(AttributeError):
        storage.DoesNotExist  # noqa: B018


def test_all_and_dir_list_the_lazy_names():
    """Lazy names stay discoverable for star-imports and tab completion."""
    import kb_vectorizer.embedding as embedding
    import kb_vectorizer.storage as storage

    assert 'SentenceTransformerEmbedder' in dir(embedding)
    assert 'ChromaStore' in dir(storage)
    assert {'QdrantStore', 'make_qdrant_client'} <= set(storage.__all__)
