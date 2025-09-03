# kb-vectorizer

Pipeline to ingest, chunk, embed, and index knowledge bases.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
pre-commit install

kb-vectorizer --help
