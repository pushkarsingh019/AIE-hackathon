"""Unit tests for core LocalPaperQA modules."""

from __future__ import annotations

import json
import sys
import tempfile
from types import SimpleNamespace
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

    def test_load_toml_nested_embedding_section(self):
        from local_paper_qa.settings import _load_toml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[embedding]\n")
            f.write("provider = \"openai\"\n")
            f.write("model = \"text-embedding-3-large\"\n")
            f.write("dimension = 3072\n")
            path = Path(f.name)

        config = _load_toml(path)
        assert config["embedding_provider"] == "openai"
        assert config["embedding_model"] == "text-embedding-3-large"
        assert config["embedding_dimension"] == 3072
        path.unlink()

    def test_load_toml_nested_chat_section(self):
        from local_paper_qa.settings import _load_toml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[chat]\n")
            f.write("provider = \"openai\"\n")
            f.write("model = \"gpt-5.5\"\n")
            path = Path(f.name)

        config = _load_toml(path)
        assert config["chat_provider"] == "openai"
        assert config["chat_model"] == "gpt-5.5"
        path.unlink()

    def test_load_toml_nested_openai_section(self):
        from local_paper_qa.settings import _load_toml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[openai]\n")
            f.write("api_key = \"secret-openai-key\"\n")
            f.write("chat_model = \"gpt-5.5\"\n")
            f.write("reasoning_effort = \"medium\"\n")
            f.write("embedding_model = \"text-embedding-3-large\"\n")
            f.write("vision_model = \"gpt-5.5\"\n")
            path = Path(f.name)

        config = _load_toml(path)
        assert config["openai_api_key"] == "secret-openai-key"
        assert config["openai_chat_model"] == "gpt-5.5"
        assert config["openai_reasoning_effort"] == "medium"
        assert config["openai_embedding_model"] == "text-embedding-3-large"
        assert config["openai_vision_model"] == "gpt-5.5"
        path.unlink()

    def test_load_toml_nested_indexing_figure_cap(self):
        from local_paper_qa.settings import _load_toml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[indexing]\n")
            f.write("profile = \"deep_figures\"\n")
            f.write("figure_max_candidates = 3\n")
            path = Path(f.name)

        config = _load_toml(path)
        assert config["indexing_profile"] == "deep_figures"
        assert config["figure_indexing_max_candidates"] == 3
        path.unlink()

    def test_load_dotenv_supported_keys(self):
        from local_paper_qa.settings import _load_dotenv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("GEMINI_API_KEY=secret-test-key\n")
            f.write("OPENAI_API_KEY=secret-openai-key\n")
            f.write("LOCAL_PAPER_QA_CHAT_PROVIDER=openai\n")
            f.write("LOCAL_PAPER_QA_CHAT_URL=http://local-chat/v1\n")
            f.write("UNRELATED=value\n")
            path = Path(f.name)

        config = _load_dotenv(path)
        assert config["gemini_api_key"] == "secret-test-key"
        assert config["openai_api_key"] == "secret-openai-key"
        assert config["chat_provider"] == "openai"
        assert config["chat_url"] == "http://local-chat/v1"
        assert "UNRELATED" not in config
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


class TestFigureDescriptions:
    def test_describe_figure_candidates_uses_selected_candidates(self, tmp_path):
        from local_paper_qa.visuals import FigureCandidate, describe_figure_candidates

        image_path = tmp_path / "page.png"
        image_path.write_bytes(b"png")

        class FakeVisionClient:
            def __init__(self):
                self.calls = []

            def describe_image(self, image_path_arg, prompt):
                self.calls.append((Path(image_path_arg), prompt))
                return "Visual description\nClaim from figure"

        candidate = FigureCandidate(
            visual_id="v1",
            paper_id="p1",
            paper_title="Test Paper",
            pdf_path=str(tmp_path / "paper.pdf"),
            page=1,
            figure_label="Figure 1",
            caption="Figure 1 shows the experiment.",
            nearby_span_ids=["span-1"],
        )
        client = FakeVisionClient()
        paper = PaperDocument(
            paper_id="p1",
            file_path=str(tmp_path / "paper.pdf"),
            title="Test Paper",
            authors="Doe",
            year="2026",
        )

        def fake_render(selected, artifact_root):
            assert selected == candidate
            return image_path

        import local_paper_qa.visuals as visuals

        original_render = visuals.render_page_artifact
        visuals.render_page_artifact = fake_render
        try:
            notes = describe_figure_candidates(
                paper,
                tmp_path,
                client,
                model="vision-test",
                candidates=[candidate],
            )
        finally:
            visuals.render_page_artifact = original_render

        assert len(notes) == 1
        assert len(client.calls) == 1
        assert notes[0].paper_claim_about_figure == "Claim from figure"


class TestRepresentationRetrieval:
    def test_select_evidence_can_return_figure_notes(self, tmp_path):
        from local_paper_qa.service import LocalPaperQA
        from local_paper_qa.visuals import FigureNote

        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"%PDF placeholder")
        paper = PaperDocument(
            paper_id="paper-1",
            file_path=str(pdf_path),
            title="Neural Figure Study",
            authors="Doe",
            year="2026",
            chunks=[
                PaperChunk(
                    chunk_id="span-1",
                    paper_id="paper-1",
                    paper_title="Neural Figure Study",
                    page=2,
                    section="Results",
                    text="The text chunk discusses unrelated baseline methods.",
                )
            ],
        )

        qa = LocalPaperQA(str(tmp_path), use_enhanced_lineage=False)
        qa.corpus_store.upsert_papers(
            [paper],
            {paper.file_path: {"size": pdf_path.stat().st_size, "mtime": pdf_path.stat().st_mtime}},
            profile="deep_figures",
        )
        qa.corpus_store.replace_figure_notes(
            paper.paper_id,
            [
                FigureNote(
                    visual_id="visual-1",
                    paper_id=paper.paper_id,
                    page=3,
                    figure_label="Figure 2",
                    caption="Figure 2 shows power-law correlations.",
                    artifact_path="artifacts/paper-1/pages/page-003.png",
                    nearby_span_ids=["span-1"],
                    visual_description="visual description: power-law correlation structure in recordings",
                    paper_claim_about_figure="The figure supports power-law correlation structure.",
                    retrieval_content="figure: Figure 2 | visual description: power-law correlation structure",
                    model="vision-test",
                )
            ],
        )
        figure_representation = next(
            item
            for item in qa.corpus_store.list_representations([paper.paper_id])
            if item.representation_type == "figure_note"
        )
        qa.corpus_store.upsert_embeddings(
            [figure_representation],
            [[1.0, 0.0]],
            provider="test",
            model="test-embedding",
            dimension=2,
            profile="test-profile",
        )
        qa._embedding_provider = SimpleNamespace(
            info=SimpleNamespace(provider="test", model="test-embedding", dimension=2, profile="test-profile"),
            embed_query=lambda _text: [1.0, 0.0],
        )

        citations = qa.select_evidence("Does any figure show power-law correlations?", [paper], paper.chunks)

        assert citations
        assert citations[0].claim == "figure_note"
        assert citations[0].page == 3
        assert citations[0].section == "Figure 2"
        assert "power-law correlation" in citations[0].quote


class TestParser:
    def test_pypdf_fallback(self):
        from local_paper_qa.parser import _extract_with_pypdf
        from pathlib import Path

        pdf_path = Path("papers/untitled.pdf")
        if pdf_path.exists():
            pages = _extract_with_pypdf(pdf_path)
            assert len(pages) > 0
            assert isinstance(pages[0], str)


class TestCorpusStore:
    def test_store_builds_representations_and_fts(self, tmp_path):
        from local_paper_qa.corpus_store import CorpusStore

        paper = PaperDocument(
            paper_id="paper-1",
            file_path=str(tmp_path / "paper.pdf"),
            title="Neural Activation Study",
            authors="Doe",
            year="2026",
            abstract="A study about activation patterns.",
            page_count=1,
            extraction_quality="good",
            chunks=[
                PaperChunk(
                    chunk_id="span-1",
                    paper_id="paper-1",
                    paper_title="Neural Activation Study",
                    page=1,
                    section="Results",
                    text="The intervention increased neural activation in the measured condition.",
                )
            ],
        )
        Path(paper.file_path).write_bytes(b"%PDF placeholder")
        store = CorpusStore(tmp_path / "corpus.db")

        store.upsert_papers(
            [paper],
            {paper.file_path: {"size": Path(paper.file_path).stat().st_size, "mtime": Path(paper.file_path).stat().st_mtime}},
        )

        representations = store.list_representations()
        assert {item.representation_type for item in representations} == {"quote", "contextual_span", "paper"}
        assert store.search_spans("activation") == ["span-1"]

    def test_fast_profile_only_builds_quote_representations(self):
        from local_paper_qa.corpus_store import build_retrieval_representations

        paper = PaperDocument(
            paper_id="paper-1",
            file_path="/tmp/paper.pdf",
            title="Neural Activation Study",
            authors="Doe",
            year="2026",
            chunks=[
                PaperChunk(
                    chunk_id="span-1",
                    paper_id="paper-1",
                    paper_title="Neural Activation Study",
                    page=1,
                    section="Results",
                    text="The intervention increased neural activation.",
                )
            ],
        )

        representations = build_retrieval_representations(paper, profile="fast")

        assert [item.representation_type for item in representations] == ["quote"]

    def test_existing_embeddings_are_not_missing(self, tmp_path):
        from local_paper_qa.corpus_store import CorpusStore

        paper = PaperDocument(
            paper_id="paper-1",
            file_path=str(tmp_path / "paper.pdf"),
            title="Neural Activation Study",
            authors="Doe",
            year="2026",
            chunks=[
                PaperChunk(
                    chunk_id="span-1",
                    paper_id="paper-1",
                    paper_title="Neural Activation Study",
                    page=1,
                    section="Results",
                    text="The intervention increased neural activation.",
                )
            ],
        )
        Path(paper.file_path).write_bytes(b"%PDF placeholder")
        store = CorpusStore(tmp_path / "corpus.db")
        store.upsert_papers([paper], {paper.file_path: {"size": 16, "mtime": 1}}, profile="fast")
        representations = store.list_representations()

        store.upsert_embeddings(
            representations,
            [[0.1, 0.2, 0.3]],
            provider="gemini",
            model="gemini-embedding-2",
            dimension=3,
            profile="fast",
        )

        assert store.missing_embeddings(
            representations,
            provider="gemini",
            model="gemini-embedding-2",
            dimension=3,
            profile="fast",
        ) == []
        loaded = store.load_embeddings(
            [representations[0].representation_id],
            provider="gemini",
            model="gemini-embedding-2",
            dimension=3,
            profile="fast",
        )
        assert loaded[representations[0].representation_id] == [0.1, 0.2, 0.3]

    def test_figure_notes_are_stored_as_retrieval_representations(self, tmp_path):
        from local_paper_qa.corpus_store import CorpusStore
        from local_paper_qa.visuals import FigureNote

        store = CorpusStore(tmp_path / "corpus.db")
        note = FigureNote(
            visual_id="visual-1",
            paper_id="paper-1",
            page=4,
            figure_label="Figure 2",
            caption="Figure 2 shows a dose-response curve.",
            artifact_path="artifacts/paper-1/pages/page-004.png",
            nearby_span_ids=["span-1"],
            visual_description="The image shows a rising dose-response curve.",
            paper_claim_about_figure="The result increases with dose.",
            retrieval_content="figure: Figure 2 | dose-response curve",
            model="gpt-5.5",
        )

        store.replace_figure_notes("paper-1", [note])

        representations = store.list_representations(["paper-1"])
        assert len(representations) == 1
        assert representations[0].representation_type == "figure_note"
        assert representations[0].source_type == "visual_evidence"
        assert representations[0].metadata["artifact_path"].endswith("page-004.png")


class TestVisuals:
    def test_detect_figure_candidates_from_extracted_text(self):
        from local_paper_qa.visuals import detect_figure_candidates

        paper = PaperDocument(
            paper_id="paper-1",
            file_path="/tmp/paper.pdf",
            title="Dose Response Study",
            authors="Doe",
            year="2026",
            chunks=[
                PaperChunk(
                    chunk_id="span-1",
                    paper_id="paper-1",
                    paper_title="Dose Response Study",
                    page=4,
                    section="Results",
                    text="Figure 2 shows that the response increases with dose.",
                )
            ],
        )

        candidates = detect_figure_candidates(paper)

        assert len(candidates) == 1
        assert candidates[0].figure_label == "Figure 2"
        assert candidates[0].page == 4
        assert candidates[0].nearby_span_ids == ["span-1"]


class TestModelClients:
    def test_openai_chat_client_uses_medium_reasoning(self, monkeypatch):
        from local_paper_qa import model_clients

        calls = []

        class FakeResponses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(output_text="The evidence supports this answer.")

        class FakeOpenAI:
            def __init__(self, api_key):
                self.responses = FakeResponses()

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

        client = model_clients.OpenAIChatClient(
            api_key="sk-test",
            model="gpt-5.5",
            reasoning_effort="medium",
            max_output_tokens=1200,
        )
        answer = client.complete("Answer from evidence.", max_output_tokens=300)

        assert answer.startswith("The evidence")
        assert client.info.model == "gpt-5.5"
        assert client.info.reasoning_effort == "medium"
        assert calls[0]["model"] == "gpt-5.5"
        assert calls[0]["reasoning"] == {"effort": "medium"}
        assert calls[0]["max_output_tokens"] == 300

    def test_create_openai_chat_client_uses_config(self, monkeypatch):
        from local_paper_qa import model_clients

        class FakeResponses:
            def create(self, **kwargs):
                return SimpleNamespace(output_text="ok")

        class FakeOpenAI:
            def __init__(self, api_key):
                self.responses = FakeResponses()

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
        monkeypatch.setattr(model_clients, "get_chat_model", lambda: "gpt-5.5")
        monkeypatch.setattr(model_clients, "get_openai_chat_model", lambda: "gpt-5.5")
        monkeypatch.setattr(model_clients, "get_openai_reasoning_effort", lambda: "medium")
        monkeypatch.setattr(model_clients, "get_openai_chat_max_output_tokens", lambda: 1200)
        monkeypatch.setattr(model_clients, "get_openai_api_key", lambda: "sk-test")

        client = model_clients.create_chat_client()

        assert client.info.provider == "openai"
        assert client.info.model == "gpt-5.5"
        assert client.info.reasoning_effort == "medium"

    def test_create_openai_embedding_provider(self, monkeypatch):
        from local_paper_qa import model_clients

        created = {}

        class FakeEmbeddings:
            def create(self, **kwargs):
                created.update(kwargs)
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

        class FakeOpenAI:
            def __init__(self, api_key):
                self.api_key = api_key
                self.embeddings = FakeEmbeddings()

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
        monkeypatch.setattr(model_clients, "get_embedding_provider", lambda: "openai")
        monkeypatch.setattr(model_clients, "get_embedding_model", lambda: "text-embedding-3-large")
        monkeypatch.setattr(model_clients, "get_embedding_dimension", lambda: 3)
        monkeypatch.setattr(model_clients, "get_embedding_batch_size", lambda: 2)
        monkeypatch.setattr(model_clients, "get_indexing_profile", lambda: "fast")
        monkeypatch.setattr(model_clients, "get_openai_api_key", lambda: "sk-test")

        provider = model_clients.create_embedding_provider()
        embedding = provider.embed_query("neuron firing")

        assert provider.info.provider == "openai"
        assert provider.info.model == "text-embedding-3-large"
        assert embedding == [0.1, 0.2, 0.3]
        assert created["dimensions"] == 3
        assert created["input"] == ["neuron firing"]

    def test_openai_embedding_provider_batches_documents(self, monkeypatch):
        from local_paper_qa import model_clients

        calls = []

        class FakeEmbeddings:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(embedding=[float(index), 1.0])
                        for index, _ in enumerate(kwargs["input"], start=1)
                    ]
                )

        class FakeOpenAI:
            def __init__(self, api_key):
                self.embeddings = FakeEmbeddings()

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

        provider = model_clients.OpenAIEmbeddingProvider(
            api_key="sk-test",
            model="text-embedding-3-large",
            dimension=2,
            batch_size=2,
            profile="fast",
        )

        embeddings = provider.embed_documents(["alpha", "beta", "gamma"])

        assert len(calls) == 2
        assert calls[0]["input"] == ["alpha", "beta"]
        assert calls[1]["input"] == ["gamma"]
        assert embeddings == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]

    def test_openai_vision_client_sends_base64_image(self, monkeypatch, tmp_path):
        from local_paper_qa import model_clients

        calls = []

        class FakeResponses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(output_text="Figure 2 shows a dose-response pattern.")

        class FakeOpenAI:
            def __init__(self, api_key):
                self.responses = FakeResponses()

        monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
        image_path = tmp_path / "figure.png"
        image_path.write_bytes(b"not-real-png")

        client = model_clients.OpenAIVisionClient(
            api_key="sk-test",
            model="gpt-5.5",
            detail="low",
            max_output_tokens=128,
        )
        description = client.describe_image(image_path, "Describe this figure.")

        image_item = calls[0]["input"][0]["content"][1]
        assert description.startswith("Figure 2")
        assert calls[0]["model"] == "gpt-5.5"
        assert calls[0]["max_output_tokens"] == 128
        assert image_item["type"] == "input_image"
        assert image_item["detail"] == "low"
        assert image_item["image_url"].startswith("data:image/png;base64,")

    def test_local_embedding_provider_is_rejected(self, monkeypatch):
        from local_paper_qa import model_clients

        monkeypatch.setattr(model_clients, "get_embedding_provider", lambda: "local_openai_compatible")

        with pytest.raises(model_clients.EmbeddingError, match="Local embedding providers are disabled"):
            model_clients.create_embedding_provider()
