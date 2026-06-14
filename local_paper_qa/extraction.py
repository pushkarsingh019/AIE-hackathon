from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from local_paper_qa.domain import (
    EvidenceSpan,
    ExtractedPaper,
    ExtractionQuality,
    ExtractionStatus,
    Paper,
)

logger = logging.getLogger(__name__)


def extract_paper(path: str | Path) -> ExtractedPaper:
    pdf_path = Path(path).expanduser().resolve()
    paper_id = hashlib.sha1(str(pdf_path).encode()).hexdigest()[:12]

    try:
        pages, pdf_meta = extract_pdf_pages(pdf_path)
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", pdf_path, exc)
        paper = Paper(
            paper_id=paper_id,
            file_path=str(pdf_path),
            title=pdf_path.stem.replace("_", " "),
            authors="Unknown",
            year="n.d.",
            source_name=pdf_path.name,
        )
        status = ExtractionStatus(
            quality=ExtractionQuality.FAILED,
            message=f"PDF extraction failed: {exc}",
            page_count=0,
            span_count=0,
        )
        return ExtractedPaper(paper=paper, page_count=0, spans=[], status=status)

    first_title = pdf_meta.get("Title") or pdf_path.stem.replace("_", " ")
    meta = extract_metadata(pdf_path, pages, pdf_meta)
    paper = Paper(
        paper_id=paper_id,
        file_path=str(pdf_path),
        title=meta.get("title") or first_title,
        authors=meta.get("authors") or "Unknown",
        year=meta.get("year") or "n.d.",
        venue=meta.get("venue") or "",
        doi=meta.get("doi") or "",
        abstract=meta.get("abstract") or "",
        source_name=pdf_path.name,
    )
    spans = build_spans(paper_id, paper.title, pdf_path.name, pages)
    status = assess_extraction_status(pages, spans)
    return ExtractedPaper(paper=paper, page_count=len(pages), spans=spans, status=status)


def extract_pdf_pages(path: Path) -> tuple[list[str], dict[str, str]]:
    from local_paper_qa.parser import extract_pages
    from pypdf import PdfReader

    pages = extract_pages(path)
    reader = PdfReader(str(path))
    metadata = {str(k).lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    return pages, metadata


def extract_metadata(path: Path, pages: list[str], pdf_meta: dict[str, str]) -> dict[str, str]:
    first_page = pages[0] if pages else ""
    meta = {
        "title": pdf_meta.get("Title") or guess_title(path, first_page),
        "authors": pdf_meta.get("Author") or guess_authors(first_page),
        "year": pdf_meta.get("Published") or pdf_meta.get("Date") or guess_year(first_page),
        "venue": pdf_meta.get("Book") or pdf_meta.get("Subject") or guess_venue(first_page),
        "doi": pdf_meta.get("doi") or guess_doi("\n".join(pages[:2])),
        "abstract": pdf_meta.get("Description-Abstract") or guess_abstract(first_page),
    }

    sample_text = "\n".join(pages[:5])
    doi_missing = not meta.get("doi")
    authors_unknown = meta.get("authors", "").strip().lower() in {"unknown", "unknown author"}
    if not sample_text.strip() or not (doi_missing or authors_unknown):
        return meta

    try:
        from local_paper_qa.metadata.enhanced_extractor import EnhancedMetadataExtractor

        extractor = EnhancedMetadataExtractor()
        extracted = extractor.extract_enhanced_metadata(sample_text, path)

        if extracted.confidence_score >= 0.5:
            if doi_missing and extracted.doi:
                meta["doi"] = extracted.doi
            if (not meta.get("title") or meta["title"] == guess_title(path, first_page)) and extracted.title:
                meta["title"] = extracted.title
            if authors_unknown and extracted.authors:
                meta["authors"] = ", ".join(extracted.authors)
            if meta.get("year") in {"n.d.", ""} and extracted.year is not None:
                meta["year"] = str(extracted.year)
            if meta.get("venue") in {"", "Unknown"} and extracted.venue:
                meta["venue"] = extracted.venue
            if not meta.get("abstract") and extracted.abstract:
                meta["abstract"] = extracted.abstract
    except Exception as exc:
        logger.debug("Enhanced metadata extraction failed for %s: %s", path, exc)

    return meta


def build_spans(paper_id: str, title: str, source_name: str, pages: list[str]) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for page_number, page_text in enumerate(pages, start=1):
        section = "Unknown section"
        for para_index, paragraph in enumerate(split_paragraphs(page_text), start=1):
            heading = detect_section_heading(paragraph)
            if heading:
                section = heading
                continue
            if len(paragraph.split()) < 20:
                continue
            spans.append(
                EvidenceSpan(
                    span_id=f"{paper_id}-p{page_number}-{para_index}",
                    paper_id=paper_id,
                    paper_title=title,
                    page=page_number,
                    section=section,
                    quote=paragraph,
                    metadata={"source": source_name},
                )
            )
    return spans


def assess_extraction_status(pages: list[str], spans: list[EvidenceSpan]) -> ExtractionStatus:
    page_count = len(pages)
    span_count = len(spans)
    if not pages:
        return ExtractionStatus(
            quality=ExtractionQuality.POOR,
            message="No pages were found in the PDF.",
            page_count=page_count,
            span_count=span_count,
        )
    if not any(page.strip() for page in pages):
        return ExtractionStatus(
            quality=ExtractionQuality.POOR,
            message="No extractable text was found. The PDF may be scanned or image-only.",
            page_count=page_count,
            span_count=span_count,
        )
    if not spans:
        return ExtractionStatus(
            quality=ExtractionQuality.POOR,
            message="Extracted text did not contain usable evidence spans.",
            page_count=page_count,
            span_count=span_count,
        )
    return ExtractionStatus(
        quality=ExtractionQuality.GOOD,
        message="Extraction produced usable evidence spans.",
        page_count=page_count,
        span_count=span_count,
    )


def split_paragraphs(text: str) -> list[str]:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("-"):
            cleaned_lines.append(stripped[:-1])
        else:
            cleaned_lines.append(re.sub(r"\s+", " ", stripped).strip())

    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in cleaned_lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        if detect_section_heading(line):
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            paragraphs.append(line)
            continue
        buffer.append(line)
        if sum(len(part.split()) for part in buffer) >= 150:
            paragraphs.append(" ".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer))
    return paragraphs


def detect_section_heading(text: str) -> str:
    text = text.strip()
    if len(text) > 90 or len(text.split()) > 15:
        return ""
    if re.match(
        r"^(\d+(\.\d+)*\.?\s+)?(abstract|introduction|background|related work|methods?|methodology|experiments?|results?|discussion|limitations?|conclusion|references|acknowledgments|supplementary|appendix)\b",
        text,
        re.I,
    ):
        return text[:80]
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][a-zA-Z\s.,:;]+", text):
        return text[:80]
    if re.match(r"^[A-Z][A-Z\s.,:;]{10,}$", text):
        return text[:80]
    return ""


def guess_title(path: Path, text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), path.stem.replace("_", " "))[:160]


def guess_authors(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[1][:160] if len(lines) > 1 else "Unknown"


def guess_year(text: str) -> str:
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else "n.d."


def guess_venue(text: str) -> str:
    match = re.search(r"(proceedings|journal|conference|transactions|letters).*", text, re.I)
    return match.group(0)[:160] if match else ""


def guess_doi(text: str) -> str:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
    return match.group(0) if match else ""


def guess_abstract(text: str) -> str:
    match = re.search(r"abstract\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]{2,40}\n|\Z)", text, re.I | re.S)
    return re.sub(r"\s+", " ", match.group(1)).strip()[:1000] if match else ""
