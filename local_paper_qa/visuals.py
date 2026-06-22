from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from local_paper_qa.models import PaperDocument


class VisionClient(Protocol):
    def describe_image(self, image_path: str | Path, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class FigureCandidate:
    visual_id: str
    paper_id: str
    paper_title: str
    pdf_path: str
    page: int
    figure_label: str
    caption: str
    nearby_span_ids: list[str]


@dataclass(frozen=True)
class FigureNote:
    visual_id: str
    paper_id: str
    page: int
    figure_label: str
    caption: str
    artifact_path: str
    nearby_span_ids: list[str]
    visual_description: str
    paper_claim_about_figure: str
    retrieval_content: str
    model: str


_FIGURE_RE = re.compile(r"\b(?:fig(?:ure)?\.?)\s*([0-9]+[a-z]?)", re.IGNORECASE)


def detect_figure_candidates(paper: PaperDocument) -> list[FigureCandidate]:
    candidates: dict[tuple[int, str], FigureCandidate] = {}
    for chunk in paper.chunks:
        match = _FIGURE_RE.search(chunk.text)
        if match is None:
            continue
        label = f"Figure {match.group(1).upper()}"
        key = (chunk.page, label)
        previous = candidates.get(key)
        caption = _clip(chunk.text, 1600)
        span_ids = [chunk.chunk_id]
        if previous is not None:
            caption = _clip(f"{previous.caption} {caption}", 1600)
            span_ids = [*previous.nearby_span_ids, chunk.chunk_id]
        candidates[key] = FigureCandidate(
            visual_id=_visual_id(paper.paper_id, chunk.page, label),
            paper_id=paper.paper_id,
            paper_title=paper.title,
            pdf_path=paper.file_path,
            page=chunk.page,
            figure_label=label,
            caption=caption,
            nearby_span_ids=list(dict.fromkeys(span_ids)),
        )
    return list(candidates.values())


def describe_figure_candidates(
    paper: PaperDocument,
    artifact_root: Path,
    vision_client: VisionClient,
    *,
    model: str,
    candidates: list[FigureCandidate] | None = None,
) -> list[FigureNote]:
    notes: list[FigureNote] = []
    selected_candidates = candidates if candidates is not None else detect_figure_candidates(paper)
    for candidate in selected_candidates:
        artifact_path = render_page_artifact(candidate, artifact_root)
        description = vision_client.describe_image(artifact_path, _figure_prompt(candidate))
        claim = _extract_claim(description)
        notes.append(
            FigureNote(
                visual_id=candidate.visual_id,
                paper_id=candidate.paper_id,
                page=candidate.page,
                figure_label=candidate.figure_label,
                caption=candidate.caption,
                artifact_path=str(artifact_path.relative_to(artifact_root.parent)),
                nearby_span_ids=candidate.nearby_span_ids,
                visual_description=description,
                paper_claim_about_figure=claim,
                retrieval_content=_retrieval_content(candidate, description, claim),
                model=model,
            )
        )
    return notes


def render_page_artifact(candidate: FigureCandidate, artifact_root: Path) -> Path:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on optional environment state
        raise RuntimeError("Install pymupdf to render PDF pages for figure indexing.") from exc

    paper_dir = artifact_root / candidate.paper_id / "pages"
    paper_dir.mkdir(parents=True, exist_ok=True)
    output_path = paper_dir / f"page-{candidate.page:03d}.png"
    if output_path.exists():
        return output_path

    document = fitz.open(candidate.pdf_path)
    try:
        page = document.load_page(max(0, candidate.page - 1))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pixmap.save(output_path)
    finally:
        document.close()
    return output_path


def _figure_prompt(candidate: FigureCandidate) -> str:
    return (
        "You are indexing a scientific paper figure for retrieval. "
        "Use only the image, figure label, caption, and paper title. "
        "Return a concise description of what the figure visually shows and what the paper appears to claim from it. "
        "Do not add broad background knowledge.\n\n"
        f"Paper title: {candidate.paper_title}\n"
        f"Figure: {candidate.figure_label}\n"
        f"Page: {candidate.page}\n"
        f"Caption or nearby extracted text: {candidate.caption}"
    )


def _retrieval_content(candidate: FigureCandidate, description: str, claim: str) -> str:
    parts = [
        f"title: {candidate.paper_title}",
        f"figure: {candidate.figure_label}",
        f"page: {candidate.page}",
        f"caption: {candidate.caption}",
        f"visual description: {description}",
        f"paper claim about figure: {claim or 'none'}",
    ]
    return " | ".join(parts)


def _extract_claim(description: str) -> str:
    lines = [line.strip(" -") for line in description.splitlines() if line.strip()]
    return _clip(lines[-1] if lines else description, 600)


def _visual_id(paper_id: str, page: int, label: str) -> str:
    digest = hashlib.sha256(f"{paper_id}|{page}|{label}".encode("utf-8")).hexdigest()
    return digest[:24]


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
