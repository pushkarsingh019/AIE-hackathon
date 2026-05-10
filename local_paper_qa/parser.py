"""PDF parsing utilities using Docling with fallback to PyPDF.

The :func:`extract_pages` function returns a list where each element is the concatenated
text for a given PDF page.  It attempts to use Docling (which provides structured items
such as section headings, tables, figures, etc.).  When Docling is not available or
fails, the classic PyPDF extraction is used as a fallback.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Lazy import Docling – it may not be installed in all environments.
try:
    from docling.document_converter import DocumentConverter
except Exception:  # pragma: no cover
    DocumentConverter = None  # type: ignore

from pypdf import PdfReader


def _extract_with_docling(path: Path) -> List[str]:
    """Extract page‑wise text using Docling.

    The function iterates over ``doc.iterate_items()`` which yields ``(item, level)``
    tuples.  ``item.prov[0].page_no`` gives the 1‑based page number.  We concatenate
    the textual representation of each item (section headings, plain text, tables,
    figures) into a per‑page buffer.
    """
    if DocumentConverter is None:  # pragma: no cover
        raise RuntimeError("Docling is not installed")

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    # Map page number -> list of strings
    page_texts: dict[int, List[str]] = {}
    for item, _level in doc.iterate_items():
        # Most items expose ``prov`` with a ``page_no`` attribute.
        try:
            page_no = int(item.prov[0].page_no)  # type: ignore[attr-defined]
        except Exception:
            # If provenance is missing, skip the item.
            continue
        # Resolve textual representation based on type.
        txt: str | None = None
        # TextItem and subclasses provide a ``text`` attribute.
        if hasattr(item, "text"):
            txt = getattr(item, "text")
        # Section headers may have a ``title`` attribute.
        elif hasattr(item, "title"):
            txt = getattr(item, "title")
        # Tables and figures usually expose ``caption`` or ``description``.
        elif hasattr(item, "caption"):
            txt = getattr(item, "caption")
        elif hasattr(item, "description"):
            txt = getattr(item, "description")
        if txt:
            page_texts.setdefault(page_no, []).append(txt.strip())
    # Build ordered list; missing pages become empty strings.
    max_page = max(page_texts.keys(), default=0)
    return [" ".join(page_texts.get(i, [])) for i in range(1, max_page + 1)]


def _extract_with_pypdf(path: Path) -> List[str]:
    """Fallback extraction using PyPDF – returns raw page text strings."""
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def extract_pages(path: str | Path) -> List[str]:
    """Public helper returning a list of page texts.

    Parameters
    ----------
    path: str | Path
        Path to the PDF file.

    Returns
    -------
    List[str]
        A list where each element corresponds to the concatenated text of a page.
        The order matches the original PDF order (1‑based indexing).
    """
    pdf_path = Path(path).expanduser().resolve()
    # Try Docling first – it may provide richer structure.
    try:
        logger.info("Attempting Docling extraction for %s", path)
        return _extract_with_docling(pdf_path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Docling extraction failed (%s); falling back to PyPDF", exc)
        return _extract_with_pypdf(pdf_path)
