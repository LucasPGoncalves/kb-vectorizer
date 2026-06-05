from __future__ import annotations

import tiktoken

from kb_vectorizer.chunking.interfaces import BaseChunker, Chunk

DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]

class TiktokenRecursiveChunker(BaseChunker):
    """Split text recursively using a list of separators.

    Ensures that each chunk is within the specified token size limit. Overlaps 
    are computed at the separator level to prevent truncating words, URLs, 
    or image placeholders.
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        encoding: str | None = None,
        model: str | None = None,
        separators: list[str] | None = None,
    ):
        """Initialize the recursive token chunker.
        
        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Target number of overlapping tokens between chunks.
            encoding: The tiktoken encoding name (e.g., "cl100k_base").
            model: The tiktoken model name (used to infer encoding if encoding is not provided).
            separators: List of strings to use as separators.

        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = self._get_encoder(encoding, model)
        self._seps = separators or list(DEFAULT_SEPARATORS)

    @staticmethod
    def _get_encoder(encoding: str | None, model: str | None):
        if encoding:
            return tiktoken.get_encoding(encoding)
        if model:
            return tiktoken.encoding_for_model(model)
        return tiktoken.get_encoding("cl100k_base")

    def _toklen(self, s: str) -> int:
        return len(self._enc.encode(s))

    def _split_by_sep(self, text: str, sep: str) -> list[str]:
        if sep == "":
            return list(text)
        parts = text.split(sep)
        out: list[str] = []
        for i, p in enumerate(parts):
            if i > 0:
                out.append(sep)
            if p:
                out.append(p)
        return out

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge sub-splits into chunks up to chunk_size, applying chunk_overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for d in splits:
            _len = self._toklen(d)
            if total + _len > self.chunk_size:
                if total > 0:
                    docs.append("".join(current_doc))
                
                # Setup overlap for next chunk
                while total > self.chunk_overlap or (
                    total + _len > self.chunk_size and total > 0
                ):
                    total -= self._toklen(current_doc[0])
                    current_doc.pop(0)
            
            current_doc.append(d)
            total += _len
            
        if current_doc:
            docs.append("".join(current_doc))
            
        return docs

    def _split_recursive(self, text: str) -> list[str]:
        """Recursively split text by separators until pieces fit in chunk_size, then merges them with overlap."""
        if self._toklen(text) <= self.chunk_size:
            return [text]

        for sep in self._seps:
            pieces = self._split_by_sep(text, sep)
            if len(pieces) == 1:
                continue

            sub_splits: list[str] = []
            for piece in pieces:
                if self._toklen(piece) > self.chunk_size:
                    sub_splits.extend(self._split_recursive(piece))
                else:
                    sub_splits.append(piece)

            return self._merge_splits(sub_splits)

        # Fallback if no separators work: split by raw tokens
        toks = self._enc.encode(text)
        out = []
        for i in range(0, len(toks), self.chunk_size - self.chunk_overlap):
            end_idx = min(i + self.chunk_size, len(toks))
            out.append(self._enc.decode(toks[i:end_idx]))
            if end_idx == len(toks):
                break
        return out
    
    def chunk(self, text: str, *, metadata: dict[str, str], doc_id: str) -> list[Chunk]:
        """Chunks a single text string into multiple Chunk objects."""
        base_chunks = self._split_recursive(text)
        out: list[Chunk] = []
        for i, ch in enumerate(base_chunks):
            out.append(
                Chunk(
                    id=f"{doc_id}:{i:06d}",
                    text=ch,
                    metadata={**metadata, "order": str(i), "doc_id": doc_id, "tok": str(self._toklen(ch))}
                )
            )
        return out
