from __future__ import annotations

from pathlib import Path

import pytest

from local_paper_qa.models import PaperDocument
from local_paper_qa.service import LocalPaperQA


def _make_paper(papers_dir: Path) -> PaperDocument:
    return PaperDocument(
        paper_id="p1",
        file_path=str(papers_dir / "p1.pdf"),
        title="Test Paper",
        authors="Doe, Jane; Smith, John",
        year="2020",
        venue="Test Journal",
        doi="10.1234/test",
        abstract="Abstract",
        page_count=1,
        chunks=[],
    )


def test_paper_lineage_enhanced_success(monkeypatch, tmp_path: Path):
    qa = LocalPaperQA(papers_dir=str(tmp_path), use_enhanced_lineage=True)
    paper = _make_paper(tmp_path)

    lineage_graph = {
        "node_type": "source",
        "paper": {
            "title": "Test Paper",
            "authors": ["Doe, Jane"],
            "year": 2020,
            "doi": "10.1234/test",
            "url": "https://example.com/source",
            "pdf_url": "https://example.com/source.pdf",
            "abstract": "Abstract",
            "citations_count": 0,
            "confidence_score": 1.0,
        },
        "children": [
            {
                "node_type": "citing",
                "relationship_strength": 0.9,
                "paper": {
                    "title": "Citing Paper",
                    "authors": ["Alice Author"],
                    "year": 2021,
                    "doi": "10.1234/citing",
                    "url": "https://example.com/citing",
                    "pdf_url": "https://example.com/citing.pdf",
                    "abstract": "Citing abstract",
                    "citations_count": 0,
                    "confidence_score": 0.9,
                },
                "children": [],
            },
            {
                "node_type": "cited",
                "relationship_strength": 0.8,
                "paper": {
                    "title": "Cited Paper",
                    "authors": ["Bob Author"],
                    "year": 2019,
                    "doi": "10.1234/cited",
                    "url": "https://example.com/cited",
                    "pdf_url": "https://example.com/cited.pdf",
                    "abstract": "Cited abstract",
                    "citations_count": 0,
                    "confidence_score": 0.8,
                },
                "children": [],
            },
            {
                "node_type": "related",
                "relationship_strength": 0.7,
                "paper": {
                    "title": "Related Paper",
                    "authors": ["Carol Author"],
                    "year": 2018,
                    "doi": "10.1234/related",
                    "url": "https://example.com/related",
                    "pdf_url": "https://example.com/related.pdf",
                    "abstract": "Related abstract",
                    "citations_count": 0,
                    "confidence_score": 0.7,
                },
                "children": [],
            },
        ],
    }

    def fake_get_enhanced_paper_lineage(_enhanced_paper, limit: int = 10):
        return {
            "success": True,
            "lineage_file": str(tmp_path / "lineage.json"),
            "lineage_report": {
                "source_paper": {
                    "title": "Test Paper",
                    "authors": ["Doe, Jane"],
                    "year": 2020,
                    "doi": "10.1234/test",
                    "pdf_url": "https://example.com/source.pdf",
                },
                "lineage_graph": lineage_graph,
            },
        }

    monkeypatch.setattr(
        qa.enhanced_lineage_service,
        "get_enhanced_paper_lineage",
        fake_get_enhanced_paper_lineage,
    )

    lineage = qa.paper_lineage(paper, limit=2)
    assert lineage["legacy_mode"] is False
    assert lineage["source_paper"]["title"] == "Test Paper"
    assert len(lineage["results"]["prior_work"]) == 1
    assert len(lineage["results"]["citing_work"]) == 1
    assert len(lineage["results"]["related_work"]) == 1


def test_paper_lineage_enhanced_failure_falls_back_to_legacy(monkeypatch, tmp_path: Path):
    qa = LocalPaperQA(papers_dir=str(tmp_path), use_enhanced_lineage=True)
    paper = _make_paper(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(qa.enhanced_lineage_service, "get_enhanced_paper_lineage", boom)

    legacy_payload = {
        "source_paper": {"title": paper.title},
        "results": {},
        "lineage_file": None,
        "legacy_mode": True,
    }
    monkeypatch.setattr(qa, "_legacy_paper_lineage", lambda *_a, **_k: legacy_payload)

    lineage = qa.paper_lineage(paper, limit=1)
    assert lineage["legacy_mode"] is True
    assert lineage["source_paper"]["title"] == paper.title


def test_paper_lineage_legacy_only_mode(monkeypatch, tmp_path: Path):
    qa = LocalPaperQA(papers_dir=str(tmp_path), use_enhanced_lineage=False)
    paper = _make_paper(tmp_path)

    legacy_payload = {
        "source_paper": {"title": paper.title},
        "results": {"prior_work": []},
        "lineage_file": None,
        "legacy_mode": True,
    }
    called = {"n": 0}

    def fake_legacy(*_a, **_k):
        called["n"] += 1
        return legacy_payload

    monkeypatch.setattr(qa, "_legacy_paper_lineage", fake_legacy)

    lineage = qa.paper_lineage(paper, limit=1)
    assert lineage["legacy_mode"] is True
    assert called["n"] == 1
