.PHONY: setup lint format type test build pkg docker

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev] && pre-commit install
lint:
\truff check .
format:
\truff format .
type:
\tmypy src
test:
\tpytest
pkg:
\tpython -m build
docker:
\tdocker build -t kb-vectorizer:dev .
