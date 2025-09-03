from __future__ import annotations

import time

from sqlalchemy.exc import DBAPIError, OperationalError


def retryable(fn, *, attempts: int = 6, base: float = 0.25, max_wait: float = 8.0):
    def wrapper(*args, **kwargs):
        delay = base
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except (OperationalError, DBAPIError):
                if i == attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2.0, max_wait)
    return wrapper