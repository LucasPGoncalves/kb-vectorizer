from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Result

from kb_vectorizer.utils._json_default import _json_default
from kb_vectorizer.utils.retry_v2 import retryable

from .interfaces import BaseIngestor

Row = dict[str, Any]
RowList = list[dict[str, Any]]
DmlResult = dict[str, int]
QueryResult = RowList | DmlResult

class MySQLIngestor(BaseIngestor):

    def __init__(self, url: str, stream: bool = False):
        self._stream = stream
        self._engine = self._make_engine(url=url)

    def _make_engine(self, url: str) -> Engine:
        return create_engine(url, pool_pre_ping=True, future=True)
    
    def _execute_stream(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        chunk_size: int = 1000,
    ) -> Iterator[Row] | DmlResult:
        conn = self._engine.connect()
        trans = conn.begin()
        try:
            exec_conn = conn.execution_options(stream_results=True)
            result: Result = exec_conn.execute(text(query), params or {})

            if not result.returns_rows:
                rc = result.rowcount
                trans.commit()
                result.close()
                conn.close()
                return {"rowcount": rc}
            
            def _row_generator() -> Iterator[Row]:
                try:
                    while True:
                        batch = result.mappings().fetchmany(chunk_size)
                        if not batch:
                            break
                        for r in batch:
                            yield dict(r)
                    trans.commit()
                finally:
                    result.close()
                    conn.close()

            return _row_generator()
        except Exception:
            try:
                trans.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            raise

    @retryable
    def _ingest(self, query: str, params: dict[str, Any] | None = None) -> QueryResult:
        stmt = text(query)

        with self._engine.begin() as conn:
            result: Result = conn.execute(stmt, params or {})

            if result.returns_rows:
                return [dict(r) for r in result.mappings().all()]
            return {"rowcount": result.rowcount}
        
    def write_result_to_file(
        self,
        result: dict | list[dict] | Iterable[dict],
        path: str,
    ) -> None:
        
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(result, dict):
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return

        if getattr(self, "_stream", True):
            with out_path.open("w", encoding="utf-8") as f:
                for row in result:
                    f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        else:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
    
    def ingest(self, query: str, params: dict[str, Any] | None = None):
        try:
            if getattr(self, "_stream", True):
                return self._execute_stream(query, params=params)
            return self._ingest(query=query, params=params)
        except Exception as e:
            print(e)

    async def a_ingest(self, query: str, params: dict[str, Any] | None = None) -> QueryResult:
        try:
            if getattr(self, "_stream", True):
                return await asyncio.to_thread(self._execute_stream, query, params=params)
            return await asyncio.to_thread(self._ingest, query=query, params=params)  
        except Exception as e:
            print(e)
        
    
