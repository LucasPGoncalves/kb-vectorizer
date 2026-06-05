from .html_preprocessor import HTMLProcessor
from .interfaces import BasePreprocessor, DocumentRecord, ImageRecord, PreprocessResult
from .json_to_html_processor import JSONDocumentPreprocessor

__all__ = [
    "BasePreprocessor",
    "DocumentRecord",
    "HTMLProcessor",
    "ImageRecord",
    "JSONDocumentPreprocessor",
    "PreprocessResult",
]
