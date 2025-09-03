from __future__ import annotations

from abc import ABC, abstractmethod


class BaseIngestor(ABC):

    @abstractmethod
    def ingest(self, query):
        ...
    
    @abstractmethod
    async def a_ingest(self, query):
        pass