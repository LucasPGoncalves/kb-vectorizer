from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

_MYSQL_RETRY_ERRCODES = {
    1213,   # Deadlock found… (SQLSTATE 40001)
    1205,   # Lock wait timeout… (SQLSTATE HY000)
    2006,   # MySQL server has gone away
    2013,   # Lost connection to MySQL server during query
}

_MYSQL_RETRY_SQLSTATES = {
    "40001",  # serialization failure / deadlock
    # HY000 is “general error” (used by 1205); numeric check catches it
}

def _is_transient(exc: BaseException) -> bool:
    # Generic SQLAlchemy “disconnect” flag (any DB)
    if isinstance(exc, DBAPIError) and getattr(exc, "connection_invalidated", False):
        return True  # e.g., server closed idle connection.

    # MySQL-specific checks
    if isinstance(exc, DBAPIError | OperationalError):
        err = getattr(exc, "orig", None)
        if err is not None:
            # numeric errcode (first arg)
            code = None
            try:
                code = getattr(err, "args", [None])[0]
            except Exception:
                code = None
            if isinstance(code, int) and code in _MYSQL_RETRY_ERRCODES:
                return True

            # SQLSTATE if exposed
            state = getattr(err, "sqlstate", None) or getattr(getattr(err, "orig", None), "sqlstate", None)
            if isinstance(state, str) and state in _MYSQL_RETRY_SQLSTATES:
                return True

    # Fallback: treat generic OperationalError as potentially transient (network/etc.)
    if isinstance(exc, OperationalError):
        return True

    return False

def retry_db(*, attempts: int = 6) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=0.25, max=8.0),
            retry=retry_if_exception(_is_transient),
        )
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            session: Session | None = kwargs.get("session")
            try:
                return fn(*args, **kwargs)
            except Exception:
                # After DBAPIError, SQLAlchemy requires rollback before reusing session.
                if session is not None:
                    try:
                        session.rollback()
                    except Exception:
                        pass
                raise
        return wrapped
    return decorator
