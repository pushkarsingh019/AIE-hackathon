from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PaperCitation:
    paper_id: str
    paper_title: str
    authors: str
    year: str
    venue: str = ""
    doi: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    quote: str = ""
    claim: str = ""
    score: float = 0.0


@dataclass
class PaperChunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    page: int
    section: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PaperDocument:
    paper_id: str
    file_path: str
    title: str
    authors: str
    year: str
    venue: str = ""
    doi: str = ""
    abstract: str = ""
    page_count: int = 0
    chunks: list[PaperChunk] = field(default_factory=list)


@dataclass
class SupportedClaim:
    claim_id: int
    text: str
    citation_ids: list[int] = field(default_factory=list)


@dataclass
class AnswerSegment:
    text: str
    claim_id: int = 0


@dataclass
class StructuredAnswer:
    answer: str
    segments: list[AnswerSegment] = field(default_factory=list)
    claims: list[SupportedClaim] = field(default_factory=list)
