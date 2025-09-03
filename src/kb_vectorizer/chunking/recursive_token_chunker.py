from __future__ import annotations

import tiktoken

from kb_vectorizer.chunking.interfaces import BaseChunker, Chunk

DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]

class TiktokenRecursiveChunker(BaseChunker):
    chunk_size: int = 200          # tokens per chunk (start 400–600; tune later)
    chunk_overlap: int = 60        # ~10–15% overlap (Azure guidance)
    encoding: str | None = None # e.g., "cl100k_base"
    model: str | None = None    # alternative to encoding
    separators: list[str] = None

    @staticmethod
    def _get_encoder(encoding: str | None, model: str | None):
        if encoding:
            return tiktoken.get_encoding(encoding)
        if model:
            return tiktoken.encoding_for_model(model)
        return tiktoken.get_encoding("cl100k_base")

    def __init__(self):
        self._enc = self._get_encoder(self.encoding, self.model)
        self._seps = self.separators or list(DEFAULT_SEPARATORS)

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

    def _split_recursive(self, text: str, chunk_size: int) -> list[str]:
        if self._toklen(text) <= chunk_size:
            return [text]

        for sep in self._seps:
            pieces = self._split_by_sep(text, sep)
            if len(pieces) == 1:
                continue

            sub_splits: list[str] = []
            for piece in pieces:
                sub_splits.extend(self._split_recursive(piece, chunk_size))

            merged: list[str] = []
            cur: list[str] = []
            cur_len = 0
            for s in sub_splits:
                s_len = self._toklen(s)
                if cur and cur_len + s_len > chunk_size:
                    merged.append("".join(cur))
                    cur, cur_len = [], 0
                cur.append(s)
                cur_len += s_len
            if cur:
                merged.append("".join(cur))

            return merged

        toks = self._enc.encode(text)
        out = []
        for i in range(0, len(toks), chunk_size):
            out.append(self._enc.decode(toks[i:i + chunk_size]))
        return out

    def _apply_overlap(self, chunks: list[str], overlap: int) -> list[str]:
        if overlap <= 0 or not chunks:
            return chunks
        if overlap >= self.chunk_size:
            overlap = max(0, self.chunk_size // 5)

        out: list[str] = []
        prev_tail_tokens: list[int] = []
        for idx, ch in enumerate(chunks):
            if idx == 0:
                out.append(ch)
                prev_tail_tokens = self._enc.encode(ch)[-overlap:]
                continue
            head_tokens = self._enc.encode(ch)
            merged_tokens = prev_tail_tokens + head_tokens
            out.append(self._enc.decode(merged_tokens))
            prev_tail_tokens = self._enc.encode(ch)[-overlap:]
        return out

    def chunk(self, text: str, *, metadata: dict[str, str], doc_id: str) -> list[Chunk]:
        base_chunks = self._split_recursive(text, self.chunk_size)
        chunks_with_overlap = self._apply_overlap(base_chunks, self.chunk_overlap)
        out: list[Chunk] = []
        for i, ch in enumerate(chunks_with_overlap):
            out.append(
                Chunk(
                    id=f"{doc_id}:{i:06d}",
                    text=ch,
                    metadata={**metadata, "order": str(i), "doc_id": doc_id, "tok": str(self._toklen(ch))}
                )
            )
        return out
