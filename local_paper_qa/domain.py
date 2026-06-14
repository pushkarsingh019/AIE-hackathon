from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionScope(str, Enum):
    PAPER = "paper"
    IDEA = "idea"
    FINDING = "finding"
    CORPUS = "corpus"


class ExtractionQuality(str, Enum):
    GOOD = "good"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class ResearchProject:
    name: str
    corpus_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Paper:
    paper_id: str
    file_path: str
    title: str
    authors: str
    year: str
    venue: str = ""
    doi: str = ""
    abstract: str = ""
    source_name: str = ""


@dataclass
class ExtractionStatus:
    quality: ExtractionQuality
    message: str = ""
    page_count: int = 0
    span_count: int = 0

    @property
    def is_usable(self) -> bool:
        return self.quality == ExtractionQuality.GOOD and self.span_count > 0


@dataclass
class EvidenceSpan:
    span_id: str
    paper_id: str
    paper_title: str
    page: int
    section: str
    quote: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedPaper:
    paper: Paper
    page_count: int
    spans: list[EvidenceSpan] = field(default_factory=list)
    status: ExtractionStatus = field(
        default_factory=lambda: ExtractionStatus(ExtractionQuality.POOR, "Not extracted")
    )


@dataclass
class EvidenceRelation:
    span_id: str
    relation: str
    explanation: str = ""


@dataclass
class EvidenceSet:
    spans: list[EvidenceSpan] = field(default_factory=list)
    relations: list[EvidenceRelation] = field(default_factory=list)


@dataclass
class AnswerSegment:
    text: str
    relations: list[EvidenceRelation] = field(default_factory=list)


@dataclass
class WorkbenchAnswer:
    question: str
    answer: str
    scope: QuestionScope = QuestionScope.CORPUS
    segments: list[AnswerSegment] = field(default_factory=list)
    evidence_set: EvidenceSet = field(default_factory=EvidenceSet)
