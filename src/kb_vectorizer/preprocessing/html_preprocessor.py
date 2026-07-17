from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import re
from pathlib import Path

import bleach
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md  # or: from html_to_markdown import convert as md

from .interfaces import BasePreprocessor, ImageRecord, PreprocessResult

# --- config ---

BLEACH_TAGS = [
    "p", "br", "strong", "em", "b", "i", "ul", "ol", "li", "blockquote",
    "code", "pre", "hr", "a", "h1","h2","h3","h4","h5","h6", "img", "figure", "figcaption"
]
BLEACH_ATTRS = {"a": ["href", "title"], "img": ["src", "alt", "title"]}

DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[^;,\s]+)?(?:;charset=[^;,\s]+)?(?P<b64>;base64)?,(?P<data>.*)$",
    re.DOTALL,
)

class HTMLProcessor(BasePreprocessor[PreprocessResult]):
    """Converts an HTML string into Markdown, plain text, and extracted images.

    Markdown and plain text are built from two independent parses of the
    same HTML, since they serve different purposes:

    - **Markdown** (for display/rendering): base64-embedded images are
      extracted to disk exactly once and rewritten as Markdown image
      references (``![alt](path)``); the sanitized HTML is then converted
      to Markdown via markdownify.
    - **Plain text** (for embedding): extracted directly from the HTML tags
      themselves — never from the generated Markdown — so no formatting
      syntax (``**bold**``, ``# heading``, ``[link](url)``, ``![alt](path)``)
      leaks into it. Each image contributes only its caption/alt/title text,
      if any; images with no descriptive text simply disappear rather than
      leaving a placeholder or file path behind. This pass never touches
      disk — image bytes are only ever written once, while building the
      Markdown version.

    The result is a :class:`~interfaces.PreprocessResult` containing the
    Markdown string, plain text, and a list of
    :class:`~interfaces.ImageRecord` objects for every image saved to disk.
    """

    @staticmethod
    def _ext_for_mime(mime: str) -> str:
        """Return the file extension for *mime*, falling back to mimetypes."""
        common = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/svg+xml": ".svg",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
        }
        return common.get(mime) or (mimetypes.guess_extension(mime) or "")

    @staticmethod
    def _decode_data_uri(data_uri: str) -> tuple[bytes, str]:
        """RFC 2397: data:[<mediatype>][;base64],<data>
        Returns (bytes, mime).
        """
        m = DATA_URI_RE.match(data_uri)
        if not m:
            raise ValueError("Invalid data URI")
        mime = m.group("mime") or "text/plain"
        is_b64 = bool(m.group("b64"))
        payload = m.group("data") or ""
        if is_b64:
            payload = re.sub(r"\s+", "", payload)
            raw = base64.b64decode(payload, validate=True)
        else:
            from urllib.parse import unquote_to_bytes
            raw = unquote_to_bytes(payload)
        return raw, mime

    @staticmethod
    def _extract_caption(img_tag: Tag) -> str | None:
        """Return the caption for *img_tag* from a ``<figcaption>`` or adjacent inline element."""
        parent = img_tag.parent
        if isinstance(parent, Tag) and parent.name == "figure":
            cap = parent.find("figcaption")
            if cap:
                return cap.get_text(" ", strip=True)
        sib = img_tag.find_next_sibling()
        if isinstance(sib, Tag) and sib.name in {"em", "i", "small"}:
            return sib.get_text(" ", strip=True)
        return None

    @classmethod
    def _image_description(cls, img_tag: Tag) -> str | None:
        """Return the best available human-readable description of *img_tag*.

        Preference order: caption, then alt text, then title. Returns
        ``None`` if the image has no descriptive text at all — callers
        should drop such images from plain text entirely rather than
        inserting a placeholder.
        """
        caption = cls._extract_caption(img_tag)
        alt = img_tag.get("alt")
        title = img_tag.get("title")
        alt_str = str(alt) if alt is not None else None
        title_str = str(title) if title is not None else None
        return caption or alt_str or title_str or None

    def process(
        self,
        html: str,
        *,
        out_dir: str | Path,
        image_subdir: str = "images",
        doc_stem: str = "doc",
        keep_remote_img: bool = True,
    ) -> PreprocessResult:
        """Convert *html* to Markdown, extract images, and return a :class:`PreprocessResult`.

        Args:
            html: Raw HTML string to process.
            out_dir: Directory where extracted images will be saved.
            image_subdir: Sub-directory name inside *out_dir* for images.
            doc_stem: Prefix used when naming extracted image files.
            keep_remote_img: When ``True``, remote ``<img>`` tags are rewritten
                as Markdown references. When ``False``, they are replaced with
                the placeholder ``[image]``.

        Returns:
            A :class:`PreprocessResult` with ``markdown``, ``text``, and
            ``images`` populated.

        """
        out_dir = Path(out_dir)
        img_dir = out_dir / image_subdir
        img_dir.mkdir(parents=True, exist_ok=True)

        # Parse the Markdown and plain-text trees independently — each
        # <img> tag is handled differently for each output (see class
        # docstring), and mutating one tree must never affect the other.
        soup_md = BeautifulSoup(html, "html.parser")
        soup_text = BeautifulSoup(html, "html.parser")
        images: list[ImageRecord] = []

        # --- Markdown tree: extract images to disk, rewrite as Markdown refs ---
        for img in list(soup_md.find_all("img")):
            src = str(img.get("src") or "").strip()
            alt: str | None = str(img["alt"]) if img.get("alt") is not None else None
            title: str | None = str(img["title"]) if img.get("title") is not None else None
            caption = self._extract_caption(img)

            if src.lower().startswith("data:"):  # embedded base64
                try:
                    blob, mime = self._decode_data_uri(src)
                except Exception:
                    img.replace_with(soup_md.new_string("[image: invalid-data-uri]"))
                    continue

                sha = hashlib.sha256(blob).hexdigest()
                ext = self._ext_for_mime(mime)
                fname = f"{doc_stem}-{sha[:12]}{ext}"
                fpath = img_dir / fname

                tmp = fpath.with_suffix(fpath.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    f.write(blob)
                os.replace(tmp, fpath)

                images.append(ImageRecord(path=fpath, mime=mime, sha256=sha, alt=alt, title=title, caption=caption))

                alt_text = alt or title or (caption or "image")
                md_img = f"![{alt_text}]({image_subdir}/{fname})"
                if caption and caption != alt_text:
                    md_img += f" _{caption}_"
                img.replace_with(BeautifulSoup(md_img, "html.parser"))

            else:
                # Remote / file images
                if keep_remote_img and src:
                    alt_text = alt or title or "image"
                    md_img = f"![{alt_text}]({src})"
                    if caption and caption != alt_text:
                        md_img += f" _{caption}_"
                    img.replace_with(BeautifulSoup(md_img, "html.parser"))
                else:
                    img.replace_with(soup_md.new_string("[image]"))

        safe_md_html = bleach.clean(
            str(soup_md),
            tags=BLEACH_TAGS,
            attributes=BLEACH_ATTRS,
            protocols={"http", "https", "mailto", "tel", "data"},
            strip=True,
        )
        markdown = md(safe_md_html, strip=["img"])

        # --- Plain-text tree: no disk I/O, images collapse to description or nothing ---
        for img in list(soup_text.find_all("img")):
            description = self._image_description(img)
            if description:
                img.replace_with(soup_text.new_string(description))
            else:
                img.decompose()

        safe_text_html = bleach.clean(
            str(soup_text),
            tags=BLEACH_TAGS,
            attributes=BLEACH_ATTRS,
            protocols={"http", "https", "mailto", "tel", "data"},
            strip=True,
        )
        text = BeautifulSoup(safe_text_html, "html.parser").get_text(" ", strip=True)

        return PreprocessResult(markdown=markdown, text=text, images=images)
