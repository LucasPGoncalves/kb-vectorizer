"""Unit tests for the kb_vectorizer.preprocessing module.

Covers:
  - DocumentRecord   (interfaces.py)
  - JSONDocumentPreprocessor  (json_to_html_processor.py)
  - HTMLProcessor    (html_preprocessor.py)
"""

import pytest

from kb_vectorizer.preprocessing import (
    DocumentRecord,
    HTMLProcessor,
    JSONDocumentPreprocessor,
    PreprocessResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal 1×1 transparent PNG encoded as a data URI — used to test image
# extraction without needing any image files on disk.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_TINY_PNG_DATA_URI = f"data:image/png;base64,{_TINY_PNG_B64}"


# ---------------------------------------------------------------------------
# DocumentRecord
# ---------------------------------------------------------------------------


def test_document_record_attribute_access():
    """Fields are accessible as attributes on DocumentRecord."""
    doc = DocumentRecord(data={"title": "Hello", "body": "World"})

    assert doc.title == "Hello"
    assert doc.body == "World"


def test_document_record_missing_attribute_raises():
    """Accessing a non-existent field raises AttributeError, not KeyError."""
    doc = DocumentRecord(data={"title": "Hello"})

    with pytest.raises(AttributeError, match="no field 'missing'"):
        _ = doc.missing


def test_document_record_get_returns_value():
    """get() returns the value when the key exists."""
    doc = DocumentRecord(data={"score": 0.95})

    assert doc.get("score") == 0.95


def test_document_record_get_returns_default():
    """get() returns the supplied default when the key is absent."""
    doc = DocumentRecord(data={})

    assert doc.get("score", -1) == -1
    assert doc.get("score") is None


def test_document_record_keys():
    """keys() mirrors the keys of the underlying dict."""
    data = {"a": 1, "b": 2, "c": 3}
    doc = DocumentRecord(data=data)

    assert set(doc.keys()) == {"a", "b", "c"}


def test_document_record_repr_contains_field_names():
    """repr() lists the field names so the object is easy to inspect."""
    doc = DocumentRecord(data={"x": 1, "y": 2})
    r = repr(doc)

    assert "x" in r
    assert "y" in r
    assert "DocumentRecord" in r


# ---------------------------------------------------------------------------
# JSONDocumentPreprocessor
# ---------------------------------------------------------------------------


def test_json_preprocessor_returns_document_record():
    """process() always returns a DocumentRecord instance."""
    preprocessor = JSONDocumentPreprocessor()
    result = preprocessor.process({"title": "T", "text": "Body"})

    assert isinstance(result, DocumentRecord)


def test_json_preprocessor_preserves_all_fields():
    """Every key in the source dict is accessible on the returned DocumentRecord."""
    doc = {"id": "doc-1", "title": "T", "text": "B", "score": 0.9}
    preprocessor = JSONDocumentPreprocessor()
    result = preprocessor.process(doc)

    assert result.id == "doc-1"
    assert result.title == "T"
    assert result.text == "B"
    assert result.score == 0.9


def test_json_preprocessor_empty_dict():
    """An empty dict is accepted when no required_fields are set."""
    preprocessor = JSONDocumentPreprocessor()
    result = preprocessor.process({})

    assert isinstance(result, DocumentRecord)
    assert list(result.keys()) == []


def test_json_preprocessor_required_fields_all_present():
    """No error is raised when all required fields are present."""
    preprocessor = JSONDocumentPreprocessor(required_fields=["title", "text"])
    result = preprocessor.process({"title": "T", "text": "B", "extra": "ok"})

    assert result.title == "T"


def test_json_preprocessor_required_fields_missing_raises():
    """ValueError is raised and lists the missing field names."""
    preprocessor = JSONDocumentPreprocessor(required_fields=["title", "text"])

    with pytest.raises(ValueError, match="text"):
        preprocessor.process({"title": "T"})


def test_json_preprocessor_multiple_missing_fields_reported():
    """All missing fields are included in the ValueError message."""
    preprocessor = JSONDocumentPreprocessor(required_fields=["a", "b", "c"])

    with pytest.raises(ValueError) as exc_info:
        preprocessor.process({})

    message = str(exc_info.value)
    assert "a" in message
    assert "b" in message
    assert "c" in message


def test_json_preprocessor_no_required_fields_by_default():
    """Default construction imposes no field requirements."""
    preprocessor = JSONDocumentPreprocessor()
    # Should not raise regardless of content
    preprocessor.process({})
    preprocessor.process({"anything": True})


# ---------------------------------------------------------------------------
# HTMLProcessor
# ---------------------------------------------------------------------------


def test_html_processor_returns_preprocess_result(tmp_path):
    """process() returns a PreprocessResult."""
    processor = HTMLProcessor()
    result = processor.process("<p>Hello</p>", out_dir=tmp_path)

    assert isinstance(result, PreprocessResult)


def test_html_processor_converts_headings_to_markdown(tmp_path):
    """HTML headings are converted to Markdown heading syntax."""
    processor = HTMLProcessor()
    result = processor.process("<h1>Title</h1><p>Body</p>", out_dir=tmp_path)

    assert "# Title" in result.markdown or "Title" in result.markdown
    assert "Body" in result.markdown


def test_html_processor_plain_text_has_no_html_tags(tmp_path):
    """The text field contains no HTML markup."""
    processor = HTMLProcessor()
    result = processor.process("<h1>Hello</h1><p>World</p>", out_dir=tmp_path)

    assert "<" not in result.text
    assert ">" not in result.text
    assert "Hello" in result.text
    assert "World" in result.text


def test_html_processor_no_images_returns_empty_list(tmp_path):
    """A document without <img> tags produces an empty images list."""
    processor = HTMLProcessor()
    result = processor.process("<p>No images here.</p>", out_dir=tmp_path)

    assert result.images == []


def test_html_processor_data_uri_image_saved_to_disk(tmp_path):
    """A base64 data URI image is extracted and saved as a file on disk."""
    processor = HTMLProcessor()
    html = f'<p>Intro</p><img src="{_TINY_PNG_DATA_URI}" alt="tiny"/>'
    result = processor.process(html, out_dir=tmp_path, doc_stem="test-doc")

    assert len(result.images) == 1
    record = result.images[0]
    assert record.path.exists(), "Image file was not written to disk"
    assert record.mime == "image/png"
    assert record.alt == "tiny"
    assert len(record.sha256) == 64


def test_html_processor_data_uri_image_referenced_in_markdown(tmp_path):
    """After extraction the Markdown contains a local image reference, not the data URI."""
    processor = HTMLProcessor()
    html = f'<img src="{_TINY_PNG_DATA_URI}" alt="chart"/>'
    result = processor.process(html, out_dir=tmp_path)

    assert "data:" not in result.markdown
    assert "chart" in result.markdown or ".png" in result.markdown


def test_html_processor_remote_image_kept_by_default(tmp_path):
    """Remote <img> src is rewritten to a Markdown image reference."""
    processor = HTMLProcessor()
    html = '<img src="https://example.com/photo.jpg" alt="photo"/>'
    result = processor.process(html, out_dir=tmp_path, keep_remote_img=True)

    assert "https://example.com/photo.jpg" in result.markdown
    assert result.images == []


def test_html_processor_remote_image_stripped_when_disabled(tmp_path):
    """Remote images are replaced by [image] placeholder when keep_remote_img=False."""
    processor = HTMLProcessor()
    html = '<img src="https://example.com/photo.jpg" alt="photo"/>'
    result = processor.process(html, out_dir=tmp_path, keep_remote_img=False)

    assert "https://example.com/photo.jpg" not in result.markdown
    assert result.images == []


def test_html_processor_image_subdir_created(tmp_path):
    """The image subdirectory is created automatically inside out_dir."""
    processor = HTMLProcessor()
    processor.process(
        f'<img src="{_TINY_PNG_DATA_URI}"/>',
        out_dir=tmp_path,
        image_subdir="extracted",
    )

    assert (tmp_path / "extracted").is_dir()


def test_html_processor_idempotent_on_plain_text(tmp_path):
    """Processing a fragment without special HTML produces consistent output."""
    processor = HTMLProcessor()
    html = "<p>Simple paragraph.</p>"
    r1 = processor.process(html, out_dir=tmp_path)
    r2 = processor.process(html, out_dir=tmp_path)

    assert r1.markdown == r2.markdown
    assert r1.text == r2.text
