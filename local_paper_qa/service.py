from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, fields
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from local_paper_qa.citations import format_apa
from local_paper_qa.config import get_chat_model, get_chat_url, get_embedding_model, get_embedding_url
from local_paper_qa.models import AnswerSegment, PaperChunk, PaperCitation, PaperDocument, StructuredAnswer, SupportedClaim
from local_paper_qa.lineage.enhanced_service import EnhancedLineageService, EnhancedPaperDocument


class AskResult(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)
    apa_references: list[str] = Field(default_factory=list)
    papers: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)


class LocalPaperQA:
    def __init__(self, papers_dir: str = "papers", use_enhanced_lineage: bool = True):
        self.papers_dir = Path(papers_dir).expanduser().resolve()
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = self.papers_dir / ".research_index"
        self.index_file = self.index_dir / "index.json"
        
        # Enhanced lineage service
        self.use_enhanced_lineage = use_enhanced_lineage
        if use_enhanced_lineage:
            self.enhanced_lineage_service = EnhancedLineageService(papers_dir)
        
        # Legacy Exa integration (fallback)
        self.legacy_exa_available = bool(self._exa_api_key())

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
                    return result['lineage_report']
                else:
                    # Fall back to legacy method
                    return self._legacy_paper_lineage(paper, limit)
                    
            except Exception as e:
                print(f"Enhanced lineage error: {e}, falling back to legacy method")
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

        demo_item = {
            "title": "Playing Atari with Deep Reinforcement Learning",
            "url": "https://arxiv.org/pdf/1312.5602.pdf",
        }
        try:
            return self.download_lineage_paper(demo_item), "Demo fallback: Playing Atari with Deep Reinforcement Learning"
        except Exception as e:
            errors.append(str(e))
        raise RuntimeError("No downloadable lineage PDF found. " + " | ".join(errors[:3]))
    
    def download_enhanced_lineage_paper(self, item: dict) -> Optional[Path]:
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
            print(f"Enhanced download error: {e}, falling back to legacy method")
            return self.download_lineage_paper(item)
    
    def download_first_enhanced_lineage_paper(self, items: list[dict]) -> tuple[Optional[Path], str]:
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
            print(f"Error getting enhanced lineage summary: {e}")
            return {}
    
    def find_lineage_papers_by_type(self, lineage_path: str, paper_type: str, limit: int = 5) -> list[dict]:
        """Find papers by type in enhanced lineage."""
        if not self.use_enhanced_lineage or not hasattr(self, 'enhanced_lineage_service'):
            return []
        
        try:
            return self.enhanced_lineage_service.find_lineage_papers_by_type(lineage_path, paper_type, limit)
        except Exception as e:
            print(f"Error finding lineage papers by type: {e}")
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
    
    def _extract_year_from_date(self, date_str: str) -> Optional[int]:
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
        except Exception:
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
        except Exception:
            return ""

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
        except Exception:
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
        paper_id = hashlib.sha1(str(path).encode()).hexdigest()[:12]
        pages, pdf_meta = self._extract_pdf_pages(path)
        title = pdf_meta.get("Title") or path.stem.replace("_", " ")
        meta = self._extract_metadata(path, pages, pdf_meta)
        chunks = self._build_chunks(paper_id, title, path.name, pages)
        return PaperDocument(
            paper_id=paper_id,
            file_path=str(path),
            title=meta.get("title") or title,
            authors=meta.get("authors") or "Unknown",
            year=meta.get("year") or "n.d.",
            venue=meta.get("venue") or "",
            doi=meta.get("doi") or "",
            abstract=meta.get("abstract") or "",
            page_count=len(pages),
            chunks=chunks,
        )

    def _extract_pdf_pages(self, path: Path) -> tuple[list[str], dict[str, str]]:
        # Use the unified parser that prefers Docling and falls back to PyPDF.
        from .parser import extract_pages
        # Extract page‑wise text using the helper.
        pages = extract_pages(path)
        # For metadata we still rely on PyPDF as Docling does not expose it directly.
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        metadata = {str(k).lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
        return pages, metadata

    def _extract_metadata(self, path: Path, pages: list[str], pdf_meta: dict[str, str]) -> dict[str, str]:
        first_page = pages[0] if pages else ""
        return {
            "title": pdf_meta.get("Title") or self._guess_title(path, first_page),
            "authors": pdf_meta.get("Author") or self._guess_authors(first_page),
            "year": pdf_meta.get("Published") or pdf_meta.get("Date") or self._guess_year(first_page),
            "venue": pdf_meta.get("Book") or pdf_meta.get("Subject") or self._guess_venue(first_page),
            "doi": pdf_meta.get("doi") or self._guess_doi("\n".join(pages[:2])),
            "abstract": pdf_meta.get("Description-Abstract") or self._guess_abstract(first_page),
        }

    def _build_chunks(self, paper_id: str, title: str, source_name: str, pages: list[str]) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        for page_number, page_text in enumerate(pages, start=1):
            section = "Unknown section"
            for para_index, paragraph in enumerate(self._paragraphs(page_text), start=1):
                heading = self._section_heading(paragraph)
                if heading:
                    section = heading
                    continue
                if len(paragraph.split()) < 20:
                    continue
                chunks.append(PaperChunk(f"{paper_id}-p{page_number}-{para_index}", paper_id, title, page_number, section, paragraph, metadata={"source": source_name}))
        return chunks

    def _paragraphs(self, text: str) -> list[str]:
        # Clean up hyphenation at line ends and collapse whitespace
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.rstrip()
            # Remove hyphenation caused by PDF extraction
            if stripped.endswith("-"):
                cleaned_lines.append(stripped[:-1])
            else:
                cleaned_lines.append(re.sub(r"\s+", " ", stripped).strip())
        # Rejoin into paragraphs separated by blank lines
        paragraphs: list[str] = []
        buffer: list[str] = []
        for line in cleaned_lines:
            if not line:
                if buffer:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
                continue
            buffer.append(line)
            if sum(len(part.split()) for part in buffer) >= 150:
                paragraphs.append(" ".join(buffer))
                buffer = []
        if buffer:
            paragraphs.append(" ".join(buffer))
        return paragraphs

    def _section_heading(self, text: str) -> str:
        text = text.strip()
        if len(text) > 90 or len(text.split()) > 15:
            return ""
        if re.match(r"^(\d+(\.\d+)*\s+)?(abstract|introduction|background|related work|methods?|methodology|experiments?|results?|discussion|limitations?|conclusion|references|acknowledgments|supplementary|appendix)\b", text, re.I):
            return text[:80]
        # Detect numbered section titles like "1. Introduction", "2.1 Methods"
        if re.match(r"^\d+(\.\d+)*\s+[A-Z][a-zA-Z\s.,:;]+", text):
            return text[:80]
        # All-caps headings (common in papers)
        if re.match(r"^[A-Z][A-Z\s.,:;]{10,}$", text):
            return text[:80]
        return ""

    def _load_index_if_fresh(self) -> list[PaperDocument] | None:
        if not self.index_file.exists():
            return None
        payload = json.loads(self.index_file.read_text())
        current_files = self._current_file_state()
        if not self._same_file_map(payload.get("files", {}), current_files):
            return None
        return [self._paper_from_dict(item) for item in payload.get("papers", [])]

    def _cached_papers_by_fresh_path(self) -> dict[str, PaperDocument]:
        if not self.index_file.exists():
            return {}
        payload = json.loads(self.index_file.read_text())
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
        self.index_file.write_text(json.dumps({"files": files, "papers": [asdict(p) for p in papers]}, ensure_ascii=False))

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

    def _guess_title(self, path: Path, text: str) -> str:
        return next((line.strip() for line in text.splitlines() if line.strip()), path.stem.replace("_", " "))[:160]

    def _guess_authors(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[1][:160] if len(lines) > 1 else "Unknown"

    def _guess_year(self, text: str) -> str:
        match = re.search(r"(19|20)\d{2}", text)
        return match.group(0) if match else "n.d."

    def _guess_venue(self, text: str) -> str:
        match = re.search(r"(proceedings|journal|conference|transactions|letters).*", text, re.I)
        return match.group(0)[:160] if match else ""

    def _guess_doi(self, text: str) -> str:
        match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
        return match.group(0) if match else ""

    def _guess_abstract(self, text: str) -> str:
        match = re.search(r"abstract\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]{2,40}\n|\Z)", text, re.I | re.S)
        return re.sub(r"\s+", " ", match.group(1)).strip()[:1000] if match else ""
