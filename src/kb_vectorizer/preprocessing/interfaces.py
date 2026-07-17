from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class ImageRecord:
    """An image extracted from a preprocessed document.

    Attributes:
        path: Absolute path to the saved image file on disk.
        mime: MIME type of the image (e.g. ``"image/png"``).
        sha256: Hex-encoded SHA-256 digest of the raw image bytes.
        alt: Alt text from the source ``<img>`` tag, if present.
        title: Title attribute from the source ``<img>`` tag, if present.
        caption: Inferred figure caption, if present.

    """

    path: Path
    mime: str
    sha256: str
    alt: str | None = None
    title: str | None = None
    caption: str | None = None


@dataclass
class PreprocessResult:
    """Output produced by :class:`HTMLProcessor`.

    Attributes:
        markdown: Document content converted to Markdown, with embedded
            images rewritten to local file references — for rendering/display.
        text: Pure prose extracted straight from the HTML, with all
            formatting syntax removed (no ``**bold**``, ``# heading``,
            ``[link](url)``, or ``![alt](path)`` markup). Images contribute
            only their caption/alt/title text, if any — never a file path
            or placeholder token. Intended for embedding.
        images: All images that were extracted and saved to disk.

    """

    markdown: str
    text: str
    images: list[ImageRecord]


@dataclass
class DocumentRecord:
    """A raw document dict exposed as a Python object with attribute access.

    Wraps any ``dict[str, Any]`` so downstream code can use ``doc.title``
    instead of ``doc["title"]``, while still supporting dict-style helpers
    via :meth:`get` and :meth:`keys`.

    Attributes:
        data: The underlying document dictionary.

    """

    data: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        """Return the document field *name* as an attribute."""
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(f"DocumentRecord has no field '{name}'") from None

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``data[key]`` or *default* if the key is absent."""
        return self.data.get(key, default)

    def keys(self):
        """Return the document field names."""
        return self.data.keys()

    def __repr__(self) -> str:
        """Return a concise string showing available field names."""
        return f"DocumentRecord(fields={list(self.data.keys())})"


class BasePreprocessor(ABC, Generic[T]):
    """Abstract base class for all preprocessing steps.

    Subclasses must implement :meth:`process`, which transforms a raw input
    into a typed output ``T``.
    """

    @abstractmethod
    def process(self, data: Any) -> T:
        """Transform *data* into the preprocessed output.

        Args:
            data: Raw input — its type depends on the concrete subclass.

        Returns:
            The preprocessed result, typed by the subclass's ``T``.

        """
        ...
