from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

# Token character class covers:
#   a-z 0-9        - ASCII alphanumerics
#   U+00C0-U+017E  - Latin Extended A/B (all pt-BR accented chars + other
#                    Romance / European scripts that may appear in documents)
# Built from codepoints (rather than a literal character range) so this
# source file stays pure ASCII regardless of editor/terminal encoding.
_LATIN_EXTENDED = f"{chr(0x00C0)}-{chr(0x017E)}"
_TOKEN_RE = re.compile(
    rf"[a-z0-9{_LATIN_EXTENDED}]+(?:[-_./][a-z0-9{_LATIN_EXTENDED}]+)*",
    re.UNICODE,
)
_SEPARATOR_RE = re.compile(r"[-_./]")


def normalize_accents(text: str) -> str:
    """Strip diacritic marks from a Unicode string.

    Lets accented and unaccented forms match during keyword search.

    Args:
        text: Input string, possibly containing accented characters.

    Returns:
        The same string with combining diacritical marks removed.

    Examples:
        >>> normalize_accents("verificação")
        'verificacao'
        >>> normalize_accents("São Paulo")
        'Sao Paulo'

    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    """Fuzzy multi-lingual tokenizer optimized for Brazilian Portuguese.

    With graceful support for English and other Latin-script languages.

    Three expansion layers per raw token
    -------------------------------------
    Given the raw token ``'SN-X2047-QR'`` extracted from lowercased text:

    1. **Original token** → ``'sn-x2047-qr'``
       Preserves the canonical form so exact-match queries score highest.
    2. **Flattened (no separator)** → ``'snx2047qr'``
       Handles user typos / copy-paste that drops hyphens, dots, etc.
    3. **Accent-stripped variants** (only when the token has non-ASCII chars)
       e.g. ``'verificação'`` → also indexes ``'verificacao'``.

    Stop-word removal is intentionally omitted: short function words carry
    meaningful disambiguation weight in pt-BR technical text.

    Args:
        text: Raw text to tokenize.

    Returns:
        One or more tokens per occurrence in *text* — the original surface
        form plus its expansions. Repeated words intentionally produce
        repeated tokens (no cross-occurrence deduplication), so callers
        that count frequencies (e.g. BM25) see accurate term counts;
        duplicates are only suppressed *within* a single occurrence's own
        expansion set.

    """
    tokens: list[str] = []
    raw_text_lower = text.lower()

    for match in _TOKEN_RE.finditer(raw_text_lower):
        original = match.group()
        occurrence: list[str] = [original]

        has_separator = bool(_SEPARATOR_RE.search(original))
        if has_separator:
            flat = _SEPARATOR_RE.sub("", original)
            if flat not in occurrence:
                occurrence.append(flat)

        has_accents = any(ord(ch) > 127 for ch in original)
        if has_accents:
            stripped = normalize_accents(original)
            if stripped not in occurrence:
                occurrence.append(stripped)
            if has_separator:
                flat_stripped = _SEPARATOR_RE.sub("", stripped)
                if flat_stripped not in occurrence:
                    occurrence.append(flat_stripped)

        tokens.extend(occurrence)

    return tokens


def token_to_index(token: str, vocab_size: int = 2**31 - 1) -> int:
    """Map a token string to a stable integer index via feature hashing.

    Uses MD5 rather than Python's built-in ``hash()``, which is randomized
    per-process (``PYTHONHASHSEED``) and would map the same token to a
    different index every run — breaking consistency between the indices
    computed at upsert time and at query time. This is the same "hashing
    trick" used by fixed-vocabulary sparse encoders (e.g. Qdrant's own
    ``Qdrant/bm25`` FastEmbed model): no shared vocabulary table needs to be
    built or persisted, so each document's sparse vector can be computed
    independently, in isolation, without ever holding the whole corpus in
    memory.

    Args:
        token: A single token string.
        vocab_size: Modulus bounding the returned index range.

    Returns:
        A stable non-negative integer in ``[0, vocab_size)``.

    """
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % vocab_size


def term_frequency_vector(text: str) -> dict[int, float]:
    """Compute a sparse term-frequency vector for *text*.

    Tokenizes *text* via :func:`tokenize`, then counts occurrences of each
    token and maps token strings to stable indices via :func:`token_to_index`.
    Deliberately corpus-independent: computing this for one document never
    requires seeing any other document, which is what allows a server-side
    IDF modifier (e.g. Qdrant's ``Modifier.IDF``) to take raw term
    frequencies and apply corpus-wide statistics without the client ever
    holding the full corpus in memory.

    Args:
        text: Raw text to vectorize.

    Returns:
        A mapping of token index to raw term frequency (count).

    """
    counts = Counter(tokenize(text))
    return {token_to_index(token): float(count) for token, count in counts.items()}
