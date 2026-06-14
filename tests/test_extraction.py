from __future__ import annotations

from pathlib import Path

from local_paper_qa import extraction
from local_paper_qa.domain import EvidenceRelation, EvidenceSpan, ExtractionQuality
from local_paper_qa.service import LocalPaperQA


def _minimal_text_pdf(text_lines: list[str]) -> bytes:
    text_ops = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(text_lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            text_ops.append("0 -20 Td")
        text_ops.append(f"({escaped}) Tj")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode()

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")

    startxref = sum(len(chunk) for chunk in chunks)
    xref_lines = [b"xref\n", b"0 6\n", b"0000000000 65535 f \n"]
    xref_lines.extend(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    chunks.extend(
        [
            *xref_lines,
            b"trailer\n<< /Root 1 0 R /Size 6 >>\n",
            b"startxref\n",
            str(startxref).encode(),
            b"\n%%EOF\n",
        ]
    )
    return b"".join(chunks)


def test_evidence_span_keeps_minimum_inspection_fields():
    span = EvidenceSpan(
        span_id="span-1",
        paper_id="paper-1",
        paper_title="Readable Systems",
        page=3,
        section="Methods",
        quote="This is the exact evidence text preserved for inspection.",
    )
    relation = EvidenceRelation(span_id=span.span_id, relation="supports")

    assert span.paper_id == "paper-1"
    assert span.page == 3
    assert span.section == "Methods"
    assert span.quote.startswith("This is the exact evidence")
    assert relation.span_id == span.span_id


def test_extract_paper_creates_page_and_section_spans(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF synthetic test placeholder")
    pages = [
        "\n".join(
            [
                "1. Introduction",
                "This study evaluates retrieval quality across a curated literature corpus using repeatable evidence spans, stable page references, and section labels that can be inspected during review.",
                "",
                "Methods",
                "The method separates PDF extraction from question answering so each paper can be parsed once and reused for later corpus questions without repeating extraction work.",
            ]
        )
    ]

    monkeypatch.setattr(
        extraction,
        "extract_pdf_pages",
        lambda _path: (
            pages,
            {
                "Title": "Sample Paper",
                "Author": "Doe, Jane",
                "Date": "2026",
                "doi": "10.1234/sample",
            },
        ),
    )

    extracted = extraction.extract_paper(pdf_path)

    assert extracted.paper.title == "Sample Paper"
    assert extracted.status.quality == ExtractionQuality.GOOD
    assert extracted.status.is_usable is True
    assert extracted.status.page_count == 1
    assert extracted.status.span_count == 2
    assert [span.page for span in extracted.spans] == [1, 1]
    assert extracted.spans[0].section == "1. Introduction"
    assert extracted.spans[1].section == "Methods"
    assert extracted.spans[0].metadata["source"] == "sample.pdf"


def test_extract_paper_smoke_reads_generated_pdf_with_real_pypdf_path(monkeypatch, tmp_path: Path):
    from local_paper_qa import parser

    pdf_path = tmp_path / "generated.pdf"
    pdf_path.write_bytes(
        _minimal_text_pdf(
            [
                "1. Introduction",
                "This generated PDF contains enough extractable text to test the real parser path and verify evidence spans keep page section and quote fields during extraction.",
                "Methods",
                "The method parses the PDF through PyPDF and builds an evidence span without mocking the extraction boundary in this smoke test.",
            ]
        )
    )
    monkeypatch.setattr(parser, "DocumentConverter", None)

    extracted = extraction.extract_paper(pdf_path)

    assert extracted.status.quality == ExtractionQuality.GOOD
    assert extracted.page_count == 1
    assert len(extracted.spans) == 2
    assert extracted.spans[0].page == 1
    assert extracted.spans[0].section == "1. Introduction"
    assert "real parser path" in extracted.spans[0].quote
    assert extracted.spans[1].section == "Methods"
    assert "without mocking the extraction boundary" in extracted.spans[1].quote


def test_extract_paper_marks_scanned_looking_pdf_as_poor(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF synthetic scanned placeholder")

    monkeypatch.setattr(
        extraction,
        "extract_pdf_pages",
        lambda _path: (["   ", ""], {"Title": "Scanned Paper", "Author": "Unknown"}),
    )

    extracted = extraction.extract_paper(pdf_path)

    assert extracted.status.quality == ExtractionQuality.POOR
    assert extracted.status.is_usable is False
    assert extracted.status.page_count == 2
    assert extracted.status.span_count == 0
    assert "No extractable text" in extracted.status.message
    assert extracted.spans == []


def test_service_adapts_extraction_status_to_existing_paper_document(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF synthetic test placeholder")

    monkeypatch.setattr(
        extraction,
        "extract_pdf_pages",
        lambda _path: (
            [
                "\n".join(
                    [
                        "Results",
                        "The extracted span remains available through the existing PaperDocument chunks API, preserving title, page, section, text, and source metadata for current callers.",
                    ]
                )
            ],
            {"Title": "Adapter Paper", "Author": "Doe, Jane", "Date": "2026", "doi": "10.1234/adapter"},
        ),
    )

    qa = LocalPaperQA(papers_dir=str(tmp_path), use_enhanced_lineage=False)
    paper = qa._load_paper(pdf_path)

    assert paper.title == "Adapter Paper"
    assert paper.extraction_quality == "good"
    assert paper.extraction_message == "Extraction produced usable evidence spans."
    assert len(paper.chunks) == 1
    assert paper.chunks[0].section == "Results"
    assert paper.chunks[0].metadata["extraction_quality"] == "good"
