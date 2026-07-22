from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseIngestor(ABC):
    """Backend-agnostic interface for pulling data from a source system.

    Implementations run a query against their backend and return the
    results, either synchronously (:meth:`ingest`) or asynchronously
    (:meth:`a_ingest`).

    **Context manager:** every implementation automatically supports the
    ``with`` statement because :meth:`__enter__` and :meth:`__exit__` are
    provided here and delegate to :meth:`close`.
    """

    @abstractmethod
    def ingest(self, query: str) -> Any:
        """Run *query* synchronously and return its results.

        Args:
            query: Backend-specific query string.

        Returns:
            The query results; shape is backend-specific.

        """
        ...

    @abstractmethod
    async def a_ingest(self, query: str) -> Any:
        """Run *query* asynchronously and return its results.

        Args:
            query: Backend-specific query string.

        Returns:
            The query results; shape is backend-specific.

        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this ingestor (connections, pools, …).

        Called automatically by :meth:`__exit__` when using the ingestor as
        a context manager.
        """
        ...

    def __enter__(self) -> BaseIngestor:
        """Return self so the ingestor can be used as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Call :meth:`close` on exit, regardless of whether an exception occurred."""
        self.close()
