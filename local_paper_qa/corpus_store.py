from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from local_paper_qa.models import PaperChunk, PaperDocument
from local_paper_qa.visuals import FigureNote


@dataclass(frozen=True)
class RetrievalRepresentation:
    representation_id: str
    paper_id: str
    source_type: str
    source_id: str
    representation_type: str
    content: str
    content_hash: str
    metadata: dict


class CorpusStore:
    """SQLite-backed Extracted Corpus store."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def upsert_papers(
        self,
        papers: list[PaperDocument],
        file_state: dict[str, dict[str, float | int]],
        *,
        profile: str = "deep",
    ) -> None:
        with self._connect() as conn:
            for paper in papers:
                self._upsert_paper(conn, paper)
                self._upsert_source_file(conn, paper, file_state.get(paper.file_path, {}))
                self._replace_spans(conn, paper)
                self._replace_representations(conn, paper.paper_id, build_retrieval_representations(paper, profile))

    def upsert_embeddings(
        self,
        representations: list[RetrievalRepresentation],
        embeddings: list[list[float]],
        *,
        provider: str,
        model: str,
        dimension: int,
        profile: str,
    ) -> None:
        if len(representations) != len(embeddings):
            raise ValueError("representations and embeddings must have the same length")
        with self._connect() as conn:
            for representation, embedding in zip(representations, embeddings):
                if not embedding:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings (
                        representation_id, provider, model, dimension, profile,
                        input_hash, embedding, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        representation.representation_id,
                        provider,
                        model,
                        dimension,
                        profile,
                        representation.content_hash,
                        json.dumps(embedding),
                    ),
                )

    def replace_figure_notes(self, paper_id: str, notes: list[FigureNote]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM visual_evidence WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM figure_notes WHERE paper_id = ?", (paper_id,))
            conn.execute(
                "DELETE FROM retrieval_representations WHERE paper_id = ? AND representation_type = ?",
                (paper_id, "figure_note"),
            )
            conn.execute(
                "DELETE FROM representation_fts WHERE paper_id = ? AND representation_type = ?",
                (paper_id, "figure_note"),
            )
            for note in notes:
                conn.execute(
                    """
                    INSERT INTO visual_evidence (
                        visual_id, paper_id, page, figure_label, caption,
                        artifact_path, nearby_span_ids_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        note.visual_id,
                        note.paper_id,
                        note.page,
                        note.figure_label,
                        note.caption,
                        note.artifact_path,
                        json.dumps(note.nearby_span_ids, ensure_ascii=False),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO figure_notes (
                        visual_id, paper_id, model, visual_description,
                        paper_claim_about_figure, retrieval_content, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        note.visual_id,
                        note.paper_id,
                        note.model,
                        note.visual_description,
                        note.paper_claim_about_figure,
                        note.retrieval_content,
                    ),
                )
                self._insert_representation(
                    conn,
                    _representation(
                        paper_id=note.paper_id,
                        source_type="visual_evidence",
                        source_id=note.visual_id,
                        representation_type="figure_note",
                        content=note.retrieval_content,
                        metadata={
                            "page": note.page,
                            "figure_label": note.figure_label,
                            "artifact_path": note.artifact_path,
                            "nearby_span_ids": note.nearby_span_ids,
                        },
                    ),
                )

    def list_representations(self, paper_ids: list[str] | None = None) -> list[RetrievalRepresentation]:
        where = ""
        params: list[str] = []
        if paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            where = f"WHERE paper_id IN ({placeholders})"
            params = paper_ids
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT representation_id, paper_id, source_type, source_id,
                       representation_type, content, content_hash, metadata_json
                FROM retrieval_representations
                {where}
                ORDER BY paper_id, representation_type, representation_id
                """,
                params,
            ).fetchall()
        return [
            RetrievalRepresentation(
                representation_id=row["representation_id"],
                paper_id=row["paper_id"],
                source_type=row["source_type"],
                source_id=row["source_id"],
                representation_type=row["representation_type"],
                content=row["content"],
                content_hash=row["content_hash"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]

    def missing_embeddings(
        self,
        representations: list[RetrievalRepresentation],
        *,
        provider: str,
        model: str,
        dimension: int,
        profile: str,
    ) -> list[RetrievalRepresentation]:
        if not representations:
            return []
        representation_ids = [representation.representation_id for representation in representations]
        placeholders = ",".join("?" for _ in representation_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT representation_id, input_hash
                FROM embeddings
                WHERE provider = ?
                  AND model = ?
                  AND dimension = ?
                  AND profile = ?
                  AND representation_id IN ({placeholders})
                """,
                (provider, model, dimension, profile, *representation_ids),
            ).fetchall()
        existing = {row["representation_id"]: row["input_hash"] for row in rows}
        return [
            representation
            for representation in representations
            if existing.get(representation.representation_id) != representation.content_hash
        ]

    def load_embeddings(
        self,
        representation_ids: list[str],
        *,
        provider: str,
        model: str,
        dimension: int,
        profile: str,
    ) -> dict[str, list[float]]:
        if not representation_ids:
            return {}
        placeholders = ",".join("?" for _ in representation_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT representation_id, embedding
                FROM embeddings
                WHERE provider = ?
                  AND model = ?
                  AND dimension = ?
                  AND profile = ?
                  AND representation_id IN ({placeholders})
                """,
                (provider, model, dimension, profile, *representation_ids),
            ).fetchall()
        return {
            row["representation_id"]: [float(value) for value in json.loads(row["embedding"])]
            for row in rows
        }

    def search_spans(self, query: str, limit: int = 10) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT span_id
                FROM span_fts
                WHERE span_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [row["span_id"] for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    year TEXT NOT NULL,
                    venue TEXT NOT NULL DEFAULT '',
                    doi TEXT NOT NULL DEFAULT '',
                    abstract TEXT NOT NULL DEFAULT '',
                    file_path TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    extraction_quality TEXT NOT NULL DEFAULT 'unknown',
                    extraction_message TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS source_files (
                    paper_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    size INTEGER NOT NULL DEFAULT 0,
                    mtime REAL NOT NULL DEFAULT 0,
                    sha256 TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS evidence_spans (
                    span_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    quote TEXT NOT NULL,
                    span_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS retrieval_representations (
                    representation_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    representation_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    representation_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    profile TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (representation_id, provider, model, dimension, profile)
                );

                CREATE TABLE IF NOT EXISTS visual_evidence (
                    visual_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    figure_label TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    nearby_span_ids_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS figure_notes (
                    visual_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    visual_description TEXT NOT NULL,
                    paper_claim_about_figure TEXT NOT NULL,
                    retrieval_content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS span_fts USING fts5(
                    span_id UNINDEXED,
                    paper_id UNINDEXED,
                    title,
                    abstract,
                    section,
                    quote
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS representation_fts USING fts5(
                    representation_id UNINDEXED,
                    paper_id UNINDEXED,
                    representation_type UNINDEXED,
                    content
                );
                """
            )

    def _upsert_paper(self, conn: sqlite3.Connection, paper: PaperDocument) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO papers (
                paper_id, title, authors, year, venue, doi, abstract, file_path,
                page_count, extraction_quality, extraction_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper.paper_id,
                paper.title,
                paper.authors,
                paper.year,
                paper.venue,
                paper.doi,
                paper.abstract,
                paper.file_path,
                paper.page_count,
                paper.extraction_quality,
                paper.extraction_message,
            ),
        )

    def _upsert_source_file(
        self,
        conn: sqlite3.Connection,
        paper: PaperDocument,
        state: dict[str, float | int],
    ) -> None:
        path = Path(paper.file_path)
        sha256 = _file_hash(path) if path.exists() else ""
        conn.execute(
            """
            INSERT OR REPLACE INTO source_files (paper_id, path, size, mtime, sha256)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                paper.paper_id,
                paper.file_path,
                int(state.get("size", 0)),
                float(state.get("mtime", 0)),
                sha256,
            ),
        )

    def _replace_spans(self, conn: sqlite3.Connection, paper: PaperDocument) -> None:
        conn.execute("DELETE FROM evidence_spans WHERE paper_id = ?", (paper.paper_id,))
        conn.execute("DELETE FROM span_fts WHERE paper_id = ?", (paper.paper_id,))
        for chunk in paper.chunks:
            span_hash = _hash_text(chunk.text)
            conn.execute(
                """
                INSERT INTO evidence_spans (
                    span_id, paper_id, page, section, quote, span_hash, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.paper_id,
                    chunk.page,
                    chunk.section,
                    chunk.text,
                    span_hash,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                ),
            )
            conn.execute(
                """
                INSERT INTO span_fts (span_id, paper_id, title, abstract, section, quote)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk.chunk_id, paper.paper_id, paper.title, paper.abstract, chunk.section, chunk.text),
            )

    def _replace_representations(
        self,
        conn: sqlite3.Connection,
        paper_id: str,
        representations: Iterable[RetrievalRepresentation],
    ) -> None:
        conn.execute("DELETE FROM retrieval_representations WHERE paper_id = ?", (paper_id,))
        conn.execute("DELETE FROM representation_fts WHERE paper_id = ?", (paper_id,))
        for representation in representations:
            self._insert_representation(conn, representation)

    def _insert_representation(self, conn: sqlite3.Connection, representation: RetrievalRepresentation) -> None:
        conn.execute(
            """
            INSERT INTO retrieval_representations (
                representation_id, paper_id, source_type, source_id,
                representation_type, content, content_hash, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                representation.representation_id,
                representation.paper_id,
                representation.source_type,
                representation.source_id,
                representation.representation_type,
                representation.content,
                representation.content_hash,
                json.dumps(representation.metadata, ensure_ascii=False),
            ),
        )
        conn.execute(
            """
            INSERT INTO representation_fts (
                representation_id, paper_id, representation_type, content
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                representation.representation_id,
                representation.paper_id,
                representation.representation_type,
                representation.content,
            ),
        )


def build_retrieval_representations(paper: PaperDocument, profile: str = "deep") -> list[RetrievalRepresentation]:
    representations: list[RetrievalRepresentation] = []
    chunks_by_page = _chunks_by_page(paper.chunks)
    for chunk in paper.chunks:
        representations.append(
            _representation(
                paper_id=paper.paper_id,
                source_type="evidence_span",
                source_id=chunk.chunk_id,
                representation_type="quote",
                content=chunk.text,
                metadata={"page": chunk.page, "section": chunk.section},
            )
        )
        if profile == "fast":
            continue
        representations.append(
            _representation(
                paper_id=paper.paper_id,
                source_type="evidence_span",
                source_id=chunk.chunk_id,
                representation_type="contextual_span",
                content=_contextual_span_content(paper, chunk, chunks_by_page.get(chunk.page, [])),
                metadata={"page": chunk.page, "section": chunk.section},
            )
        )
    if profile == "fast":
        return representations

    paper_content = _paper_content(paper)
    if paper_content:
        representations.append(
            _representation(
                paper_id=paper.paper_id,
                source_type="paper",
                source_id=paper.paper_id,
                representation_type="paper",
                content=paper_content,
                metadata={},
            )
        )
    return representations


def _representation(
    *,
    paper_id: str,
    source_type: str,
    source_id: str,
    representation_type: str,
    content: str,
    metadata: dict,
) -> RetrievalRepresentation:
    content = _clean_text(content)
    content_hash = _hash_text(content)
    representation_id = _hash_text("|".join([paper_id, source_type, source_id, representation_type, content_hash]))[:24]
    return RetrievalRepresentation(
        representation_id=representation_id,
        paper_id=paper_id,
        source_type=source_type,
        source_id=source_id,
        representation_type=representation_type,
        content=content,
        content_hash=content_hash,
        metadata=metadata,
    )


def _contextual_span_content(paper: PaperDocument, chunk: PaperChunk, page_chunks: list[PaperChunk]) -> str:
    siblings = [item.text for item in page_chunks if item.chunk_id != chunk.chunk_id]
    nearby = " ".join(siblings[:2])
    parts = [
        f"title: {paper.title}",
        f"abstract: {paper.abstract or 'none'}",
        f"section: {chunk.section}",
        f"nearby: {nearby or 'none'}",
        f"text: {chunk.text}",
    ]
    return " | ".join(parts)


def _paper_content(paper: PaperDocument) -> str:
    parts = [paper.title, paper.abstract]
    if paper.chunks:
        parts.append(" ".join(chunk.text for chunk in paper.chunks[:3]))
        parts.append(" ".join(chunk.text for chunk in paper.chunks[-2:]))
    return _clean_text(" ".join(part for part in parts if part))


def _chunks_by_page(chunks: list[PaperChunk]) -> dict[int, list[PaperChunk]]:
    grouped: dict[int, list[PaperChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.page, []).append(chunk)
    return grouped


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
