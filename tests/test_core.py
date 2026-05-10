"""Unit tests for core LocalPaperQA modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from local_paper_qa.citations import format_apa, _format_authors_for_apa
from local_paper_qa.models import PaperCitation, PaperChunk, PaperDocument


class TestCitations:
    def test_format_apa_basic(self):
        citation = PaperCitation(
            paper_id="abc",
            paper_title="Test Paper Title",
            authors="Smith, John",
            year="2023",
            venue="Test Journal",
            doi="10.1234/test",
        )
        result = format_apa(citation)
        assert "Smith, John (2023). Test Paper Title." in result
        assert "Test Journal" in result
        assert "https://doi.org/10.1234/test" in result

    def test_format_apa_no_venue(self):
        citation = PaperCitation(
            paper_id="abc",
            paper_title="Test Paper",
            authors="Doe, Jane",
            year="2022",
            doi="10.1234/test",
        )
        result = format_apa(citation)
        assert "Doe, Jane (2022). Test Paper." in result

    def test_format_authors_comma_separated(self):
        assert "Smith, John" == _format_authors_for_apa("Smith, John")
        assert "Unknown" == _format_authors_for_apa("Unknown")
        assert "Unknown" == _format_authors_for_apa("")

    def test_format_authors_space_separated(self):
        result = _format_authors_for_apa("John Smith")
        assert "Smith, J." in result


class TestModels:
    def test_paper_chunk_defaults(self):
        chunk = PaperChunk(
            chunk_id="c1",
            paper_id="p1",
            paper_title="Test",
            page=1,
            section="Intro",
            text="Hello world",
        )
        assert chunk.score == 0.0
        assert chunk.metadata == {}

    def test_paper_document_defaults(self):
        doc = PaperDocument(
            paper_id="p1",
            file_path="/tmp/test.pdf",
            title="Test Paper",
            authors="John",
            year="2023",
        )
        assert doc.venue == ""
        assert doc.doi == ""
        assert doc.abstract == ""
        assert doc.page_count == 0
        assert doc.chunks == []

    def test_paper_citation_defaults(self):
        cit = PaperCitation(
            paper_id="p1",
            paper_title="Test",
            authors="John",
            year="2023",
        )
        assert cit.venue == ""
        assert cit.doi == ""
        assert cit.page is None
        assert cit.section is None
        assert cit.quote == ""
        assert cit.claim == ""
        assert cit.score == 0.0


class TestSettings:
    def test_load_toml_basic(self):
        from local_paper_qa.settings import _load_toml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("# comment\nchat_url = \"http://example.com\"\n")
            f.write("chunk_size = 100\n")
            f.write("reranking_enabled = true\n")
            path = Path(f.name)

        config = _load_toml(path)
        assert config["chat_url"] == "http://example.com"
        assert config["chunk_size"] == 100
        assert config["reranking_enabled"] is True
        path.unlink()

    def test_get_defaults(self):
        from local_paper_qa.settings import DEFAULTS

        assert "chat_url" in DEFAULTS
        assert "embedding_url" in DEFAULTS
        assert DEFAULTS["max_citations"] == 8

    def test_config_override_by_env(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PAPER_QA_CHAT_URL", "http://override.com/v1")
        from local_paper_qa import settings
        settings._cache = None
        assert settings.get_chat_url() == "http://override.com/v1"


class TestParser:
    def test_pypdf_fallback(self):
        from local_paper_qa.parser import _extract_with_pypdf
        from pathlib import Path

        pdf_path = Path("papers/untitled.pdf")
        if pdf_path.exists():
            pages = _extract_with_pypdf(pdf_path)
            assert len(pages) > 0
            assert isinstance(pages[0], str)
