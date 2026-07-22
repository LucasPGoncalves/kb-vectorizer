"""Unit tests for kb_vectorizer.ingestion.mysql_ingestor.

Runs against an in-memory SQLite database instead of a real MySQL server —
the ingestor only relies on generic SQLAlchemy Core behavior (text(),
execution_options, mappings()), all of which SQLite supports identically.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.exc import OperationalError

from kb_vectorizer.ingestion.mysql_ingestor import MySQLIngestor

CREATE_TABLE = "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)"


@pytest.fixture()
def ingestor(tmp_path):
    """Non-streaming ingestor backed by a fresh file-based SQLite database.

    A file (rather than ``:memory:``) is used because ``a_ingest()`` runs
    the query via ``asyncio.to_thread`` in a different thread, and SQLite's
    ``:memory:`` databases are private per-connection — a new thread would
    see a completely empty database. A real MySQL server has no such
    limitation; this is purely a test-environment concern.
    """
    db_path = tmp_path / "test.db"
    ing = MySQLIngestor(f"sqlite:///{db_path}", stream=False)
    ing.ingest(CREATE_TABLE)
    yield ing
    ing.close()


@pytest.fixture()
def streaming_ingestor(tmp_path):
    """Streaming ingestor backed by a fresh file-based SQLite database."""
    db_path = tmp_path / "test_stream.db"
    ing = MySQLIngestor(f"sqlite:///{db_path}", stream=True)
    ing.ingest(CREATE_TABLE)
    yield ing
    ing.close()


# ---------------------------------------------------------------------------
# Non-streaming mode
# ---------------------------------------------------------------------------


def test_ingest_insert_returns_rowcount(ingestor: MySQLIngestor):
    """An INSERT returns a {"rowcount": N} dict, not row data."""
    result = ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": "alice"})
    assert result == {"rowcount": 1}


def test_ingest_select_returns_row_list(ingestor: MySQLIngestor):
    """A SELECT returns a list of row dicts with the actual data."""
    ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": "alice"})

    rows = ingestor.ingest("SELECT * FROM t")

    assert rows == [{"id": 1, "name": "alice"}]


def test_ingest_select_empty_table_returns_empty_list(ingestor: MySQLIngestor):
    """A SELECT against an empty table returns an empty list, not None."""
    rows = ingestor.ingest("SELECT * FROM t")
    assert rows == []


def test_ingest_propagates_errors(ingestor: MySQLIngestor):
    """A failing query raises rather than being swallowed and returning None."""
    with pytest.raises(OperationalError):
        ingestor.ingest("SELECT * FROM this_table_does_not_exist")


def test_ingest_params_are_bound_not_interpolated(ingestor: MySQLIngestor):
    """A value containing SQL-special characters is treated as data, not executed."""
    malicious = "'; DROP TABLE t; --"
    ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": malicious})

    rows = ingestor.ingest("SELECT * FROM t")
    assert isinstance(rows, list)
    assert rows[0]["name"] == malicious  # table still exists, value stored verbatim


# ---------------------------------------------------------------------------
# Streaming mode
# ---------------------------------------------------------------------------


def test_ingest_stream_returns_generator(streaming_ingestor: MySQLIngestor):
    """In streaming mode, a SELECT returns an iterator, not a materialized list."""
    result = streaming_ingestor.ingest("SELECT * FROM t")
    assert hasattr(result, "__next__")


def test_ingest_stream_yields_all_rows(streaming_ingestor: MySQLIngestor):
    """Consuming the streamed generator yields every row, across multiple batches."""
    for i in range(5):
        streaming_ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": f"user{i}"})

    result = streaming_ingestor.ingest("SELECT * FROM t", chunk_size=2)
    assert not isinstance(result, (list, dict))
    rows = list(result)

    assert len(rows) == 5
    assert {r["name"] for r in rows} == {f"user{i}" for i in range(5)}


def test_ingest_stream_insert_returns_rowcount_not_generator(streaming_ingestor: MySQLIngestor):
    """In streaming mode, a non-SELECT statement still returns a rowcount dict."""
    result = streaming_ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": "bob"})
    assert result == {"rowcount": 1}


def test_ingest_stream_propagates_errors(streaming_ingestor: MySQLIngestor):
    """A failing streamed query raises rather than being swallowed."""
    with pytest.raises(OperationalError):
        streaming_ingestor.ingest("SELECT * FROM this_table_does_not_exist")


# ---------------------------------------------------------------------------
# write_result_to_file
# ---------------------------------------------------------------------------


def test_write_result_to_file_dict(ingestor: MySQLIngestor, tmp_path):
    """A DML result dict is written as a single JSON object."""
    result = ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": "alice"})
    out = tmp_path / "out.json"

    ingestor.write_result_to_file(result, str(out))

    assert '"rowcount": 1' in out.read_text(encoding="utf-8")


def test_write_result_to_file_list(ingestor: MySQLIngestor, tmp_path):
    """A materialized row list is written as a single JSON array."""
    ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": "alice"})
    rows = ingestor.ingest("SELECT * FROM t")
    out = tmp_path / "out.json"

    ingestor.write_result_to_file(rows, str(out))

    content = out.read_text(encoding="utf-8")
    assert content.strip().startswith("[")
    assert "alice" in content


def test_write_result_to_file_iterator_writes_ndjson(streaming_ingestor: MySQLIngestor, tmp_path):
    """A row iterator is written as newline-delimited JSON, one row per line."""
    for i in range(3):
        streaming_ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": f"user{i}"})
    gen = streaming_ingestor.ingest("SELECT * FROM t", chunk_size=2)
    out = tmp_path / "out.ndjson"

    streaming_ingestor.write_result_to_file(gen, str(out))

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_write_result_to_file_creates_parent_dirs(ingestor: MySQLIngestor, tmp_path):
    """Missing parent directories are created automatically."""
    out = tmp_path / "nested" / "dir" / "out.json"

    ingestor.write_result_to_file({"rowcount": 0}, str(out))

    assert out.exists()


def test_write_result_to_file_dispatches_by_runtime_type_not_stream_flag(
    ingestor: MySQLIngestor, tmp_path
):
    """Output format follows the actual argument type, not self._stream.

    A non-streaming ingestor (stream=False) handed a raw iterator should
    still write newline-delimited JSON, proving the branch isn't keyed off
    stale instance state.
    """
    out = tmp_path / "out.ndjson"
    row_iter = iter([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])

    ingestor.write_result_to_file(row_iter, str(out))

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# Lifecycle: close() / context manager
# ---------------------------------------------------------------------------


def test_close_disposes_engine():
    """close() disposes the underlying SQLAlchemy engine without raising."""
    ing = MySQLIngestor("sqlite:///:memory:")
    ing.close()  # must not raise


def test_close_is_idempotent():
    """close() can be called multiple times without error."""
    ing = MySQLIngestor("sqlite:///:memory:")
    ing.close()
    ing.close()  # must not raise


def test_context_manager_calls_close():
    """Using the ingestor as a context manager disposes the engine on exit."""
    ing = MySQLIngestor("sqlite:///:memory:")
    with ing as ctx:
        assert ctx is ing
    # No public "is closed" flag exists, but a second close() must still be safe
    ing.close()


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


def test_ingest_retries_transient_errors_then_succeeds(ingestor: MySQLIngestor, monkeypatch):
    """A transient OperationalError is retried and the call eventually succeeds."""
    real_begin = ingestor._engine.begin
    call_count = {"n": 0}

    def flaky_begin(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise OperationalError("stmt", {}, Exception("connection lost"))
        return real_begin(*args, **kwargs)

    monkeypatch.setattr(ingestor._engine, "begin", flaky_begin)
    # Avoid real sleep delays slowing down the test suite
    monkeypatch.setattr("kb_vectorizer.utils.retry.time.sleep", lambda _: None)

    result = ingestor.ingest("SELECT * FROM t")

    assert result == []
    assert call_count["n"] == 3


def test_ingest_gives_up_after_max_retry_attempts(ingestor: MySQLIngestor, monkeypatch):
    """After all retry attempts are exhausted, the original error still propagates."""

    def always_fails(*args, **kwargs):
        raise OperationalError("stmt", {}, Exception("connection lost"))

    monkeypatch.setattr(ingestor._engine, "begin", always_fails)
    monkeypatch.setattr("kb_vectorizer.utils.retry.time.sleep", lambda _: None)

    with pytest.raises(OperationalError):
        ingestor.ingest("SELECT * FROM t")


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


def test_a_ingest_returns_same_result_as_sync(ingestor: MySQLIngestor):
    """a_ingest() produces the same result as the synchronous ingest() call."""
    ingestor.ingest("INSERT INTO t (name) VALUES (:n)", params={"n": "alice"})

    result = asyncio.run(ingestor.a_ingest("SELECT * FROM t"))

    assert result == [{"id": 1, "name": "alice"}]


def test_a_ingest_propagates_errors(ingestor: MySQLIngestor):
    """a_ingest() raises on failure instead of swallowing the error."""
    with pytest.raises(OperationalError):
        asyncio.run(ingestor.a_ingest("SELECT * FROM this_table_does_not_exist"))
