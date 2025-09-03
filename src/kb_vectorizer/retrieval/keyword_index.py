from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path


class KeywordIndex:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
            USING fts5(doc_id UNINDEXED, content, tokenize='unicode61');
        """)

    def upsert(self, docs: Iterable[tuple[str, str]]) -> None:
        # docs: iterable of (doc_id, markdown_text)
        cur = self._conn.cursor()
        cur.execute("BEGIN")
        for doc_id, content in docs:
            # delete then insert to emulate upsert
            cur.execute("DELETE FROM docs_fts WHERE doc_id = ?", (doc_id,))
            cur.execute("INSERT INTO docs_fts(doc_id, content) VALUES (?, ?)", (doc_id, content))
        self._conn.commit()

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        # Try bm25(); if not available in this SQLite build, fallback to default rank
        try:
            rows = self._conn.execute(
                "SELECT doc_id, bm25(docs_fts) AS score FROM docs_fts WHERE docs_fts MATCH ? ORDER BY score LIMIT ?",
                (query, k),
            ).fetchall()
            # FTS5 bm25: lower score is better; convert to descending relevance by negating
            return [(doc_id, -score) for (doc_id, score) in rows]
        except sqlite3.OperationalError:
            rows = self._conn.execute(
                "SELECT doc_id, rank FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, k),
            ).fetchall()
            return [(doc_id, -rank) for (doc_id, rank) in rows]

    def close(self):
        self._conn.close()
