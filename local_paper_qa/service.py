from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, fields
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from local_paper_qa.citations import format_apa
from local_paper_qa.config.manager import ConfigManager
from local_paper_qa.extraction import extract_paper
from local_paper_qa.settings import (
    get_chat_model,
    get_chat_url,
    get_embedding_model,
    get_embedding_url,
    get_papers_dir,
)
from local_paper_qa.models import AnswerSegment, PaperChunk, PaperCitation, PaperDocument, StructuredAnswer, SupportedClaim
from local_paper_qa.lineage.enhanced_service import EnhancedLineageService, EnhancedPaperDocument

logger = logging.getLogger(__name__)


class AskResult(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)
    apa_references: list[str] = Field(default_factory=list)
    papers: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)


class LocalPaperQA:
    def __init__(self, papers_dir: str | None = None, use_enhanced_lineage: bool | None = None):
        resolved_papers_dir = papers_dir or get_papers_dir()
        self.papers_dir = Path(resolved_papers_dir).expanduser().resolve()
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = self.papers_dir / ".research_index"
        self.index_file = self.index_dir / "index.json"
        self.vector_db_path = self.index_dir / "vectors.db"
        
        # Enhanced lineage service
        if use_enhanced_lineage is None:
            try:
                self.use_enhanced_lineage = ConfigManager().get_config().api.enable_enhanced_lineage
            except Exception as exc:
                logger.warning("Could not read enhanced lineage config; enabling enhanced lineage: %s", exc)
                self.use_enhanced_lineage = True
        else:
            self.use_enhanced_lineage = use_enhanced_lineage

        if self.use_enhanced_lineage:
            self.enhanced_lineage_service = EnhancedLineageService(resolved_papers_dir)
        
        # Vector store for efficient retrieval
        self._init_vector_store()
        
        # Legacy Exa integration (fallback)
        self.legacy_exa_available = bool(self._exa_api_key())

    def _current_index_fingerprint(self) -> dict[str, str | int | float | bool]:
        """Fingerprint for index validity.

        We include embedding config because embeddings are stored in the cached chunks
        and must match the embedding endpoint/model.
        """

        return {
            "embedding_url": get_embedding_url(),
            "embedding_model": get_embedding_model(),
        }

    def _init_vector_store(self) -> None:
        from local_paper_qa.vector_store import VectorStore
        try:
            self.vector_store = VectorStore(self.vector_db_path)
        except Exception as exc:
            logger.debug("Vector store unavailable at %s: %s", self.vector_db_path, exc)
            self.vector_store = None

    def list_papers(self) -> list[PaperDocument]:
        return [self._load_paper(path) for path in sorted(self.papers_dir.glob("*.pdf"))]

    def ensure_index(self, force: bool = False) -> list[PaperDocument]:
        if not force:
            indexed = self._load_index_if_fresh()
            if indexed is not None:
                return indexed

        cached_by_path = self._cached_papers_by_fresh_path() if force else {}
        papers: list[PaperDocument] = []
        for path in sorted(self.papers_dir.glob("*.pdf")):
            cached = cached_by_path.get(str(path))
            if cached is not None:
                papers.append(cached)
                continue
            paper = self._load_paper(path)
            for chunk in paper.chunks:
                chunk.metadata["embedding"] = self.embed_text(chunk.text[:2000])
            papers.append(paper)
        self._save_index(papers)
        self._save_vector_store(papers)
        return papers

    def ask(self, question: str) -> AskResult:
        papers = self.ensure_index()
        citations = self.retrieve(question, papers=papers)
        answer = self.answer_from_evidence(question, citations)
        return AskResult(
            answer=answer,
            citations=[asdict(citation) for citation in citations],
            apa_references=self._unique_references(citations),
            papers=[paper.title for paper in papers],
            unsupported=[] if citations else [question],
        )

    def paper_lineage(self, paper: PaperDocument, limit: int = 5) -> dict:
        """Get paper lineage using enhanced academic APIs or legacy Exa as fallback."""
        if self.use_enhanced_lineage and hasattr(self, 'enhanced_lineage_service'):
            try:
                # Convert PaperDocument to EnhancedPaperDocument
                enhanced_paper = EnhancedPaperDocument(
                    paper_id=paper.paper_id,
                    file_path=paper.file_path,
                    title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                    venue=paper.venue,
                    doi=paper.doi,
                    abstract=paper.abstract,
                    page_count=paper.page_count,
                    chunks=paper.chunks
                )
                
                # Get enhanced lineage
                result = self.enhanced_lineage_service.get_enhanced_paper_lineage(enhanced_paper, limit)
                
                if result['success']:
                    report = result["lineage_report"]
                    source = report.get("source_paper", {})
                    graph = report.get("lineage_graph", {})

                    def extract_nodes(root: dict) -> list[dict]:
                        nodes: list[dict] = []
                        stack = [root] if root else []
                        while stack:
                            node = stack.pop()
                            if node.get("node_type") and node.get("paper"):
                                nodes.append(node)
                            children = node.get("children", []) or []
                            stack.extend(reversed([child for child in children if isinstance(child, dict)]))
                        return nodes

                    nodes = extract_nodes(graph)
                    def make_item(n: dict) -> dict:
                        paper = n.get("paper", {})
                        authors = paper.get("authors") or []
                        author_str = str(authors[0]) if isinstance(authors, list) and authors else str(authors) if authors else ""
                        year = paper.get("year")
                        url = paper.get("pdf_url") or paper.get("url") or ""
                        abstract = paper.get("abstract") or ""
                        return {
                            "title": str(paper.get("title") or "Untitled"),
                            "url": str(url),
                            "published_date": str(year or ""),
                            "author": author_str,
                            "snippet": str(abstract)[:900],
                        }

                    citing_work = [make_item(n) for n in nodes if n.get("node_type") == "citing"]
                    prior_work = [make_item(n) for n in nodes if n.get("node_type") == "cited"]
                    related_work = [make_item(n) for n in nodes if n.get("node_type") == "related"]

                    lineage = {
                        "source_paper": {
                            "title": str(source.get("title") or paper.title),
                            "authors": str(source.get("authors") or paper.authors),
                            "year": str(source.get("year") or paper.year),
                            "doi": str(source.get("doi") or paper.doi),
                            "file_path": paper.file_path,
                        },
                        "results": {
                            "prior_work": prior_work[:limit],
                            "citing_work": citing_work[:limit],
                            "related_work": related_work[:limit],
                        },
                        "lineage_file": result.get("lineage_file"),
                        "legacy_mode": False,
                    }
                    return lineage
                else:
                    # Fall back to legacy method
                    return self._legacy_paper_lineage(paper, limit)
                    
            except Exception as e:
                logger.warning("Enhanced lineage failed; falling back to legacy method: %s", e)
                return self._legacy_paper_lineage(paper, limit)
        else:
            # Use legacy Exa-based lineage
            return self._legacy_paper_lineage(paper, limit)
    
    def _legacy_paper_lineage(self, paper: PaperDocument, limit: int = 5) -> dict:
        """Legacy paper lineage using Exa API (fallback method)."""
        api_key = self._exa_api_key()
        if not api_key:
            raise RuntimeError("Set EXA_API_KEY in your shell or local .env file to look up paper lineage with Exa.")

        searches = {
            "prior_work": f'foundational earlier papers cited by "{paper.title}" {paper.authors}',
            "citing_work": f'recent papers that cite "{paper.title}" {paper.authors}',
            "related_work": f'related research papers similar to "{paper.title}" {paper.authors}',
        }
        lineage = {
            "source_paper": {
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "doi": paper.doi,
                "file_path": paper.file_path,
            },
            "results": {},
            "legacy_mode": True  # Indicate this is using the legacy system
        }

        for group, query in searches.items():
            lineage["results"][group] = self._exa_search(query, api_key=api_key, limit=limit)

        output_path = self._lineage_path(paper)
        output_path.write_text(json.dumps(lineage, ensure_ascii=False, indent=2))
        lineage["lineage_file"] = str(output_path)
        return lineage

    def retrieve(self, question: str, papers: list[PaperDocument] | None = None) -> list[PaperCitation]:
        papers = papers or self.ensure_index()
        chunks = [chunk for paper in papers for chunk in paper.chunks]
        return self.select_evidence(question, papers, chunks)

    def _exa_search(self, query: str, api_key: str, limit: int) -> list[dict[str, str]]:
        response = httpx.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "type": "neural",
                "numResults": limit,
                "category": "research paper",
                "contents": {"text": {"maxCharacters": 900}},
            },
            timeout=45,
        )
        response.raise_for_status()
        results = []
        for item in response.json().get("results", []):
            results.append(
                {
                    "title": str(item.get("title") or "Untitled"),
                    "url": str(item.get("url") or ""),
                    "published_date": str(item.get("publishedDate") or ""),
                    "author": str(item.get("author") or ""),
                    "snippet": str(item.get("text") or ""),
                }
            )
        return results

    def _lineage_path(self, paper: PaperDocument) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", paper.title.lower()).strip("-")[:80] or paper.paper_id
        return self.papers_dir / f"lineage-{slug}.json"

    def download_lineage_paper(self, item: dict) -> Path:
        title = str(item.get("title") or "lineage paper")
        url = str(item.get("url") or "").strip()
        if not url:
            raise RuntimeError("Selected lineage result has no URL to download.")

        candidates = self._pdf_url_candidates(url)
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                response = httpx.get(candidate, follow_redirects=True, timeout=90)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                content = response.content
                if "pdf" not in content_type and not content.startswith(b"%PDF"):
                    raise RuntimeError(f"URL did not return a PDF: {candidate}")

                output_path = self._download_path(title, candidate)
                output_path.write_bytes(content)
                return output_path
            except Exception as e:
                last_error = e
        raise RuntimeError(f"Could not download a PDF for {title}: {last_error}")

    def download_first_available_lineage_paper(self, items: list[dict]) -> tuple[Path, str]:
        errors = []
        for item in items:
            try:
                return self.download_lineage_paper(item), str(item.get("title") or "lineage paper")
            except Exception as e:
                errors.append(str(e))

        detail = " | ".join(errors[:3])
        suffix = f" {detail}" if detail else ""
        raise RuntimeError(f"No downloadable lineage PDF found.{suffix}")
    
    def download_enhanced_lineage_paper(self, item: dict) -> Path | None:
        """Download a paper using enhanced lineage system."""
        if not self.use_enhanced_lineage or not hasattr(self, 'enhanced_lineage_service'):
            # Fall back to legacy method
            return self.download_lineage_paper(item)
        
        try:
            # Create AcademicPaper from item
            from local_paper_qa.academic.base import AcademicPaper
            
            paper = AcademicPaper(
                title=str(item.get("title") or "Untitled"),
                authors=[str(item.get("author") or "")] if item.get("author") else [],
                year=self._extract_year_from_date(str(item.get("published_date") or "")),
                doi=item.get("doi"),
                abstract=item.get("snippet"),
                url=item.get("url"),
                pdf_url=item.get("pdf_url") or item.get("url"),
                source="enhanced_lineage"
            )
            
            # Use enhanced lineage service to download
            return self.enhanced_lineage_service.download_enhanced_lineage_paper(paper)
            
        except Exception as e:
            logger.warning("Enhanced download failed; falling back to legacy method: %s", e)
            return self.download_lineage_paper(item)
    
    def download_first_enhanced_lineage_paper(self, items: list[dict]) -> tuple[Path | None, str]:
        """Download the first available paper using enhanced lineage system."""
        errors = []
        for item in items:
            try:
                path = self.download_enhanced_lineage_paper(item)
                if path:
                    return path, str(item.get("title") or "enhanced lineage paper")
            except Exception as e:
                errors.append(str(e))
        
        # Fall back to legacy method
        try:
            return self.download_first_available_lineage_paper(items)
        except Exception as e:
            errors.append(str(e))
        
        raise RuntimeError("No downloadable enhanced lineage PDF found. " + " | ".join(errors[:3]))
    
    def get_enhanced_lineage_summary(self, lineage_path: str) -> dict:
        """Get enhanced lineage summary if available."""
        if not self.use_enhanced_lineage or not hasattr(self, 'enhanced_lineage_service'):
            return {}
        
        try:
            return self.enhanced_lineage_service.get_lineage_summary(lineage_path)
        except Exception as e:
            logger.warning("Could not read enhanced lineage summary from %s: %s", lineage_path, e)
            return {}
    
    def find_lineage_papers_by_type(self, lineage_path: str, paper_type: str, limit: int = 5) -> list[dict]:
        """Find papers by type in enhanced lineage."""
        if not self.use_enhanced_lineage or not hasattr(self, 'enhanced_lineage_service'):
            return []
        
        try:
            return self.enhanced_lineage_service.find_lineage_papers_by_type(lineage_path, paper_type, limit)
        except Exception as e:
            logger.warning("Could not find lineage papers by type %s in %s: %s", paper_type, lineage_path, e)
            return []

    def _pdf_url_candidates(self, url: str) -> list[str]:
        candidates = [url]
        parsed = urlparse(url)
        if parsed.netloc.endswith("arxiv.org") and parsed.path.startswith("/abs/"):
            arxiv_id = parsed.path.removeprefix("/abs/")
            candidates.insert(0, f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        if parsed.netloc.endswith("arxiv.org") and parsed.path.startswith("/html/"):
            arxiv_id = parsed.path.removeprefix("/html/")
            candidates.insert(0, f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        if parsed.netloc.endswith("openreview.net") and parsed.path == "/forum" and parsed.query:
            candidates.insert(0, f"https://openreview.net/pdf?{parsed.query}")
        return list(dict.fromkeys(candidates))
    
    def _extract_year_from_date(self, date_str: str) -> int | None:
        """Extract year from various date formats."""
        if not date_str:
            return None
        
        import re
        year_match = re.search(r"(19|20)\d{2}", date_str)
        return int(year_match.group(1)) if year_match else None

    def _download_path(self, title: str, url: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:90] or "lineage-paper"
        path = self.papers_dir / f"{slug}.pdf"
        counter = 2
        while path.exists():
            path = self.papers_dir / f"{slug}-{counter}.pdf"
            counter += 1
        return path

    def _exa_api_key(self) -> str:
        api_key = os.environ.get("EXA_API_KEY", "").strip()
        if api_key:
            return api_key

        env_paths = [Path.cwd() / ".env", self.papers_dir.parent / ".env"]
        for env_path in env_paths:
            if not env_path.exists():
                continue
            for line in env_path.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep and key.strip() == "EXA_API_KEY":
                    return value.strip().strip('"').strip("'")
        return ""

    def answer_from_evidence(self, question: str, citations: list[PaperCitation]) -> str:
        return self._answer_with_chat(question, citations) or self._fallback_answer(question, citations)

    def answer_with_claims(self, question: str, citations: list[PaperCitation]) -> StructuredAnswer:
        raw_answer = self._answer_with_claims_chat(question, citations)
        structured = self._parse_structured_answer(raw_answer, citations)
        if structured is not None:
            return structured
        answer = self.answer_from_evidence(question, citations)
        return self._fallback_structured_answer(answer, citations)

    def select_evidence(
        self, question: str, papers: list[PaperDocument], chunks: list[PaperChunk]
    ) -> list[PaperCitation]:
        query_embedding = self.embed_text(question)
        for chunk in chunks:
            embedding = chunk.metadata.get("embedding") or []
            chunk.score = self._cosine(query_embedding, embedding) if query_embedding else 0.0

        # Rerank top results using a hybrid of embedding similarity and lexical overlap
        top_n = min(50, len(chunks))
        top_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]
        for chunk in top_chunks:
            lexical = self._lexical_score(question, chunk)
            chunk.score = chunk.score * 0.7 + lexical * 0.3

        citations: list[PaperCitation] = []
        seen: set[tuple[str, int]] = set()
        for chunk in sorted(chunks, key=lambda c: c.score, reverse=True)[:12]:
            key = (chunk.paper_id, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            paper = next(paper for paper in papers if paper.paper_id == chunk.paper_id)
            citations.append(
                PaperCitation(
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    authors=paper.authors,
                    year=paper.year,
                    venue=paper.venue,
                    doi=paper.doi,
                    page=chunk.page,
                    section=chunk.section,
                    # Use the full chunk text so the UI can show exact sentences used
                    # for answer grounding.
                    quote=chunk.text,
                    claim=question,
                    score=chunk.score,
                )
            )
        return citations[:8]

    def embed_text(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            return []
        base_url = get_embedding_url()
        model = get_embedding_model()
        try:
            response = httpx.post(
                f"{base_url.rstrip('/')}/embeddings",
                json={"model": model, "input": text},
                timeout=60,
            )
            response.raise_for_status()
            return [float(v) for v in response.json().get("data", [{}])[0].get("embedding", [])]
        except Exception as exc:
            logger.debug("Embedding request failed: %s", exc)
            return []

    def _answer_with_chat(self, question: str, citations: list[PaperCitation]) -> str:
        if not citations:
            return ""
        evidence = "\n\n".join(
            f"Paper: {c.paper_title}\nAuthors: {c.authors}\nYear: {c.year}\nPage: {c.page}\nSection: {c.section}\nQuote: {c.quote}"
            for c in citations
        )
        prompt = (
            "Answer using only the supplied paper evidence. Every factual claim must cite paper, page, and quote. "
            "If evidence is insufficient, say so.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        base_url = get_chat_url()
        model = get_chat_model()
        try:
            response = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 900,
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as exc:
            logger.debug("Chat completion request failed: %s", exc)
            return ""

    def stream_answer_with_claims(self, question: str, citations: list[PaperCitation]):
        """Yield (token, full_so_far) tuples as the LLM streams its response."""
        if not citations:
            yield "", ""
            return
        evidence = "\n\n".join(
            f"[{index}] Paper: {citation.paper_title}\n"
            f"Authors: {citation.authors}\n"
            f"Year: {citation.year}\n"
            f"Page: {citation.page}\n"
            f"Section: {citation.section}\n"
            f"Quote: {citation.quote}"
            for index, citation in enumerate(citations, start=1)
        )
        prompt = (
            "Answer using only the supplied paper evidence. Return only valid JSON, with no markdown fences. "
            "Split the answer into short answer_parts. Each answer part must be supported by exactly one claim_id. "
            "Each claim must cite one or more evidence ids from the supplied evidence. If evidence is insufficient, "
            "say that in one answer part and return an empty claims array.\n\n"
            "JSON schema:\n"
            "{\n"
            "  \"answer_parts\": [{\"text\": \"answer sentence or clause\", \"claim_id\": 1}],\n"
            "  \"claims\": [{\"id\": 1, \"claim\": \"short supported claim\", \"citation_ids\": [1]}]\n"
            "}\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        base_url = get_chat_url()
        model = get_chat_model()
        accumulated = ""
        try:
            with httpx.stream(
                "POST",
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1100,
                    "stream": True,
                },
                timeout=180,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            accumulated += token
                            yield token, accumulated
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except Exception as exc:
            logger.debug("Streaming chat completion failed: %s", exc)
            pass

    def _answer_with_claims_chat(self, question: str, citations: list[PaperCitation]) -> str:
        if not citations:
            return ""
        evidence = "\n\n".join(
            f"[{index}] Paper: {citation.paper_title}\n"
            f"Authors: {citation.authors}\n"
            f"Year: {citation.year}\n"
            f"Page: {citation.page}\n"
            f"Section: {citation.section}\n"
            f"Quote: {citation.quote}"
            for index, citation in enumerate(citations, start=1)
        )
        prompt = (
            "Answer using only the supplied paper evidence. Return only valid JSON, with no markdown fences. "
            "Split the answer into short answer_parts. Each answer part must be supported by exactly one claim_id. "
            "Each claim must cite one or more evidence ids from the supplied evidence. If evidence is insufficient, "
            "say that in one answer part and return an empty claims array.\n\n"
            "JSON schema:\n"
            "{\n"
            "  \"answer_parts\": [{\"text\": \"answer sentence or clause\", \"claim_id\": 1}],\n"
            "  \"claims\": [{\"id\": 1, \"claim\": \"short supported claim\", \"citation_ids\": [1]}]\n"
            "}\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        base_url = get_chat_url()
        model = get_chat_model()
        try:
            response = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1100,
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as exc:
            logger.debug("Structured chat completion request failed: %s", exc)
            return ""

    def _parse_structured_answer(self, raw_answer: str, citations: list[PaperCitation]) -> StructuredAnswer | None:
        if not raw_answer.strip():
            return None
        payload_text = self._extract_json_object(raw_answer)
        if not payload_text:
            return None
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return None

        claims: list[SupportedClaim] = []
        for item in payload.get("claims", []):
            claim_id = self._safe_int(item.get("id") or item.get("claim_id"))
            text = str(item.get("claim") or item.get("text") or "").strip()
            citation_ids = [
                citation_id
                for citation_id in (self._safe_int(value) for value in item.get("citation_ids", []))
                if 1 <= citation_id <= len(citations)
            ]
            if claim_id and text:
                claims.append(SupportedClaim(claim_id=claim_id, text=text, citation_ids=citation_ids))

        known_claims = {claim.claim_id for claim in claims}
        segments: list[AnswerSegment] = []
        for item in payload.get("answer_parts", []):
            text = str(item.get("text") or "").strip()
            claim_id = self._safe_int(item.get("claim_id"))
            if text:
                segments.append(AnswerSegment(text=text, claim_id=claim_id if claim_id in known_claims else 0))

        if not segments:
            return None
        answer = " ".join(segment.text for segment in segments).strip()
        return StructuredAnswer(answer=answer, segments=segments, claims=claims)

    def _fallback_structured_answer(self, answer: str, citations: list[PaperCitation]) -> StructuredAnswer:
        claims = [
            SupportedClaim(
                claim_id=index,
                text=f"Evidence from {citation.paper_title}, page {citation.page}",
                citation_ids=[index],
            )
            for index, citation in enumerate(citations[:8], start=1)
        ]
        return StructuredAnswer(answer=answer, segments=[AnswerSegment(answer, claims[0].claim_id if claims else 0)], claims=claims)

    def _extract_json_object(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        return text[start : end + 1] if start != -1 and end != -1 and end > start else ""

    def _safe_int(self, value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _fallback_answer(self, question: str, citations: list[PaperCitation]) -> str:
        if not citations:
            return "No indexed paper evidence was found."
        lines = [f"Question: {question}", "", "Evidence:"]
        for citation in citations:
            lines.append(f"- {citation.paper_title}, p. {citation.page}, {citation.section}")
            lines.append(f"  Quote: {citation.quote}")
        lines.append("\nAPA References:")
        for reference in self._unique_references(citations):
            lines.append(f"- {reference}")
        return "\n".join(lines)

    def _load_paper(self, path: Path) -> PaperDocument:
        extracted = extract_paper(path)
        paper = extracted.paper
        chunks = [
            PaperChunk(
                chunk_id=span.span_id,
                paper_id=span.paper_id,
                paper_title=span.paper_title,
                page=span.page,
                section=span.section,
                text=span.quote,
                metadata={
                    **span.metadata,
                    "extraction_quality": extracted.status.quality.value,
                    "extraction_message": extracted.status.message,
                },
            )
            for span in extracted.spans
        ]
        return PaperDocument(
            paper_id=paper.paper_id,
            file_path=paper.file_path,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            venue=paper.venue,
            doi=paper.doi,
            abstract=paper.abstract,
            page_count=extracted.page_count,
            extraction_quality=extracted.status.quality.value,
            extraction_message=extracted.status.message,
            chunks=chunks,
        )

    def _load_index_if_fresh(self) -> list[PaperDocument] | None:
        if not self.index_file.exists():
            return None
        payload = json.loads(self.index_file.read_text())
        if payload.get("fingerprint") != self._current_index_fingerprint():
            return None
        current_files = self._current_file_state()
        if not self._same_file_map(payload.get("files", {}), current_files):
            return None
        return [self._paper_from_dict(item) for item in payload.get("papers", [])]

    def _cached_papers_by_fresh_path(self) -> dict[str, PaperDocument]:
        if not self.index_file.exists():
            return {}
        payload = json.loads(self.index_file.read_text())
        if payload.get("fingerprint") != self._current_index_fingerprint():
            return {}
        indexed_files = payload.get("files", {})
        papers = [self._paper_from_dict(item) for item in payload.get("papers", [])]
        current_files = self._current_file_state()
        return {
            paper.file_path: paper
            for paper in papers
            if paper.file_path in current_files
            and self._same_file_state(indexed_files.get(paper.file_path), current_files[paper.file_path])
        }

    def _save_index(self, papers: list[PaperDocument]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        files = self._current_file_state()
        self.index_file.write_text(
            json.dumps(
                {
                    "fingerprint": self._current_index_fingerprint(),
                    "files": files,
                    "papers": [asdict(p) for p in papers],
                },
                ensure_ascii=False,
            )
        )

    def _save_vector_store(self, papers: list[PaperDocument]) -> None:
        if self.vector_store is None:
            return
        items = []
        for paper in papers:
            for chunk in paper.chunks:
                embedding = chunk.metadata.get("embedding")
                if embedding:
                    items.append((chunk.chunk_id, embedding, {
                        "paper_id": chunk.paper_id,
                        "paper_title": chunk.paper_title,
                        "page": chunk.page,
                        "section": chunk.section,
                    }))
        self.vector_store.insert_many(items)

    def _query_vector_store(self, query_embedding: list[float], limit: int = 10) -> list[str]:
        """Return chunk IDs from the vector store for the top-K results."""
        if self.vector_store is None:
            return []
        results = self.vector_store.query(query_embedding, limit=limit)
        return [r["id"] for r in results]

    def _current_file_state(self) -> dict[str, dict[str, float | int]]:
        return {
            str(path): {"mtime": path.stat().st_mtime, "size": path.stat().st_size}
            for path in sorted(self.papers_dir.glob("*.pdf"))
        }

    def _same_file_state(self, indexed: object, current: dict[str, float | int]) -> bool:
        if isinstance(indexed, dict):
            return indexed.get("mtime") == current.get("mtime") and indexed.get("size") == current.get("size")
        return indexed == current.get("mtime")

    def _same_file_map(self, indexed: dict, current: dict[str, dict[str, float | int]]) -> bool:
        if set(indexed) != set(current):
            return False
        return all(self._same_file_state(indexed[path], current[path]) for path in current)

    def _paper_from_dict(self, data: dict) -> PaperDocument:
        chunk_fields = {field.name for field in fields(PaperChunk)}
        chunks = [PaperChunk(**{k: v for k, v in chunk.items() if k in chunk_fields}) for chunk in data.get("chunks", [])]
        paper_fields = {field.name for field in fields(PaperDocument)}
        return PaperDocument(**{k: v for k, v in data.items() if k in paper_fields and k != "chunks"}, chunks=chunks)

    def _unique_references(self, citations: list[PaperCitation]) -> list[str]:
        seen: set[str] = set()
        refs: list[str] = []
        for citation in citations:
            if citation.paper_id in seen:
                continue
            seen.add(citation.paper_id)
            refs.append(format_apa(citation))
        return refs

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def _lexical_score(self, question: str, chunk: PaperChunk) -> float:
        q = set(re.findall(r"[a-z0-9]+", question.lower()))
        t = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
        return len(q & t) / len(q | t) if q and t else 0.0
