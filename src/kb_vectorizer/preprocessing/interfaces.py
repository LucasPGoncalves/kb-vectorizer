from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageRecord:
    path: Path
    mime: str
    sha256: str
    alt: str | None = None
    title: str | None = None
    caption: str | None = None

@dataclass
class PreprocessResult:
    markdown: str
    text: str
    images: list[ImageRecord]

class BasePreprocessor(ABC):

    @abstractmethod
    def process(data: Any, out_dir: Path):
        ...