from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Result

from kb_vectorizer.utils._json_default import _json_default
from kb_vectorizer.utils.retry import retryable

from .interfaces import BaseIngestor

Row = dict[str, Any]
RowList = list[dict[str, Any]]
DmlResult = dict[str, int]
QueryResult = RowList | DmlResult
IngestResult = QueryResult | Iterator[Row]


class MySQLIngestor(BaseIngestor):
    """Runs parameterized SQL against a MySQL database via SQLAlchemy Core.

    Two execution modes, chosen at construction time:

    - ``stream=False`` (default): loads the full result set into memory as
      a list of dicts. Simple, but unsuitable for very large result sets.
    - ``stream=True``: returns a generator that fetches rows from the
      server in ``chunk_size``-row batches, keeping memory bounded
      regardless of result set size. The underlying connection and
      transaction stay open while the generator is being iterated, and are
      released once it's exhausted, errors, or is garbage-collected.

    Transient connection errors (``OperationalError``, ``DBAPIError``) are
    retried with exponential backoff — see
    :func:`~kb_vectorizer.utils.retry.retryable`. Retries only cover query
    setup: once a streaming generator has been handed back to the caller,
    a failure mid-iteration can't be retried transparently, since the
    caller already holds a reference to that specific (now-failed)
    generator.

    Always pass user-supplied values via *params* (bound parameters),
    never string-interpolated into *query* — SQLAlchemy's ``text()`` binds
    *params* safely, which is what prevents SQL injection here.

    Args:
        url: SQLAlchemy connection URL, e.g.
            ``"mysql+pymysql://user:pass@host/db"``.
        stream: Whether :meth:`ingest`/:meth:`a_ingest` stream results in
            batches instead of loading them all into memory.

    """

    def __init__(self, url: str, stream: bool = False) -> None:
        """Initialize the ingestor and its underlying SQLAlchemy engine.

        Args:
            url: SQLAlchemy connection URL.
            stream: Whether to stream results in batches (see class docstring).

        """
        self._stream = stream
        self._engine = self._make_engine(url=url)

    def _make_engine(self, url: str) -> Engine:
        """Create the SQLAlchemy engine for *url*.

        Args:
            url: SQLAlchemy connection URL.

        Returns:
            A configured ``Engine`` with connection health checks
            (``pool_pre_ping``) enabled, so dropped connections are
            detected and replaced before use rather than failing mid-query.

        """
        return create_engine(url, pool_pre_ping=True, future=True)

    @retryable
    def _execute_stream(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        chunk_size: int = 1000,
    ) -> Iterator[Row] | DmlResult:
        """Execute *query*, returning a row generator or a DML result.

        Args:
            query: SQL to execute.
            params: Bound parameters for *query*.
            chunk_size: Number of rows fetched from the server per batch.

        Returns:
            An iterator of row dicts if *query* returns rows, otherwise a
            ``{"rowcount": N}`` dict for INSERT/UPDATE/DELETE statements.

        """
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
            # Best-effort cleanup: the original exception (often a dead
            # connection) may make rollback/close themselves fail — don't
            # let a secondary cleanup error mask the real one.
            with suppress(Exception):
                trans.rollback()
            with suppress(Exception):
                conn.close()
            raise

    @retryable
    def _ingest(self, query: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute *query* and return the full result set (non-streaming).

        Args:
            query: SQL to execute.
            params: Bound parameters for *query*.

        Returns:
            A list of row dicts if *query* returns rows, otherwise a
            ``{"rowcount": N}`` dict.

        """
        stmt = text(query)

        with self._engine.begin() as conn:
            result: Result = conn.execute(stmt, params or {})

            if result.returns_rows:
                return [dict(r) for r in result.mappings().all()]
            return {"rowcount": result.rowcount}

    def write_result_to_file(
        self,
        result: IngestResult,
        path: str,
    ) -> None:
        """Write an :meth:`ingest`/:meth:`a_ingest` result to a JSON file.

        The output format is chosen from *result*'s actual runtime type,
        not this ingestor's ``stream`` setting, so it's always correct even
        if you pass in a result obtained some other way:

        - A DML ``{"rowcount": N}`` dict is written as a single JSON object.
        - A materialized row list is written as a single JSON array.
        - Anything else (assumed to be a row iterator) is written as
          newline-delimited JSON, one row object per line, so the file can
          be produced without ever holding the full result in memory.

        Args:
            result: A result from :meth:`ingest`/:meth:`a_ingest`.
            path: Destination file path; parent directories are created if
                missing.

        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(result, dict):
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return

        if isinstance(result, list):
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
            return

        with out_path.open("w", encoding="utf-8") as f:
            for row in result:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    def ingest(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        chunk_size: int = 1000,
    ) -> IngestResult:
        """Execute *query* synchronously, per this ingestor's configured mode.

        Args:
            query: SQL to execute.
            params: Bound parameters for *query*.
            chunk_size: Rows fetched per batch when streaming; ignored
                otherwise.

        Returns:
            A row iterator (streaming mode, rows returned), a list of row
            dicts (non-streaming mode, rows returned), or a
            ``{"rowcount": N}`` dict (INSERT/UPDATE/DELETE, either mode).

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the query fails after all
                retry attempts are exhausted.

        """
        if self._stream:
            return self._execute_stream(query, params=params, chunk_size=chunk_size)
        return self._ingest(query=query, params=params)

    async def a_ingest(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        chunk_size: int = 1000,
    ) -> IngestResult:
        """Execute *query* asynchronously by offloading the blocking call to a thread.

        Args:
            query: SQL to execute.
            params: Bound parameters for *query*.
            chunk_size: Rows fetched per batch when streaming; ignored
                otherwise.

        Returns:
            Same as :meth:`ingest`.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the query fails after all
                retry attempts are exhausted.

        """
        if self._stream:
            return await asyncio.to_thread(
                self._execute_stream, query, params=params, chunk_size=chunk_size
            )
        return await asyncio.to_thread(self._ingest, query=query, params=params)

    def close(self) -> None:
        """Dispose of the underlying SQLAlchemy engine's connection pool.

        Safe to call multiple times.
        """
        self._engine.dispose()
