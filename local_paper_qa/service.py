from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, fields
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from local_paper_qa.citations import format_apa
from local_paper_qa.models import AnswerSegment, PaperChunk, PaperCitation, PaperDocument, StructuredAnswer, SupportedClaim


class AskResult(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)
    apa_references: list[str] = Field(default_factory=list)
    papers: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)


class LocalPaperQA:
    CHAT_BASE_URL = "http://100.67.104.58:8001/v1"
    EMBEDDING_BASE_URL = "http://100.67.104.58:8003/v1"
    MODEL = "unsloth/Qwen3.6"

    def __init__(self, papers_dir: str = "papers"):
        self.papers_dir = Path(papers_dir).expanduser().resolve()
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = self.papers_dir / ".research_index"
        self.index_file = self.index_dir / "index.json"

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

    def retrieve(self, question: str, papers: list[PaperDocument] | None = None) -> list[PaperCitation]:
        papers = papers or self.ensure_index()
        chunks = [chunk for paper in papers for chunk in paper.chunks]
        return self.select_evidence(question, papers, chunks)

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
            chunk.score = self._cosine(query_embedding, embedding) if query_embedding else self._lexical_score(question, chunk)

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
        base_url = os.environ.get("LOCAL_PAPER_QA_EMBEDDING_URL", self.EMBEDDING_BASE_URL)
        model = os.environ.get("LOCAL_PAPER_QA_EMBEDDING_MODEL", self.MODEL)
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
        base_url = os.environ.get("LOCAL_PAPER_QA_CHAT_URL", self.CHAT_BASE_URL)
        model = os.environ.get("LOCAL_PAPER_QA_CHAT_MODEL", self.MODEL)
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
        base_url = os.environ.get("LOCAL_PAPER_QA_CHAT_URL", self.CHAT_BASE_URL)
        model = os.environ.get("LOCAL_PAPER_QA_CHAT_MODEL", self.MODEL)
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
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        metadata = {str(k).lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
        return [page.extract_text() or "" for page in reader.pages], metadata

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
                if len(paragraph.split()) < 25:
                    continue
                chunks.append(PaperChunk(f"{paper_id}-p{page_number}-{para_index}", paper_id, title, page_number, section, paragraph, metadata={"source": source_name}))
        return chunks

    def _paragraphs(self, text: str) -> list[str]:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        paragraphs: list[str] = []
        buffer: list[str] = []
        for line in lines:
            if not line:
                if buffer:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
                continue
            buffer.append(line)
            if sum(len(part.split()) for part in buffer) >= 110:
                paragraphs.append(" ".join(buffer))
                buffer = []
        if buffer:
            paragraphs.append(" ".join(buffer))
        return paragraphs

    def _section_heading(self, text: str) -> str:
        text = text.strip()
        if len(text) > 90 or len(text.split()) > 10:
            return ""
        if re.match(r"^(\d+(\.\d+)*\s+)?(abstract|introduction|background|related work|methods?|methodology|experiments?|results?|discussion|limitations?|conclusion|references)\b", text, re.I):
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
