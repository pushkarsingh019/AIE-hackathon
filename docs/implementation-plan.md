# Literature Review Workbench Implementation Plan

This plan turns the current hackathon prototype into a smaller, more readable v1 architecture for project-scoped scientific literature review. It follows the language in `CONTEXT.md` and the architectural decisions in `docs/adr/`.

## V1 Target

Build a TUI-first Literature Review Workbench where a Research Project contains PDFs, those PDFs are converted once into a persisted Extracted Corpus, and the user can ask Paper Questions, Idea Questions, and prompt-based Finding Searches. Answers should be clean prose by default. Each evidence-bearing Answer Segment must be inspectable on demand to show the exact Evidence Spans and Evidence Relations behind it.

## Non-Goals For V1

- Do not build Saved Findings or a linked finding graph.
- Do not build a full evidence-grading system.
- Do not fork a coding-agent TUI unless the existing Textual TUI becomes unworkable.
- Do not model supplements, appendices, or source files as special domain objects yet.
- Do not require local-only models; keep model configuration flexible.
- Do not solve OCR/scanned PDF quality deeply. Detect poor extraction and exclude those Papers from citation-bearing answers until fixed.

## Current Friction

- `local_paper_qa/service.py` is too broad. It currently handles project setup, PDF extraction, metadata guessing, chunking, embedding calls, retrieval, answer synthesis, lineage lookup, downloads, cache validation, and vector-store writes.
- `tui.py` is too broad. It currently handles UI state, answer streaming, paper listing, citation rendering, evidence inspection, lineage views, theme management, and file-opening behavior in one file.
- Retrieval is not truly backed by a persisted Extracted Corpus. The vector store is written, but normal retrieval scores chunks loaded from JSON in memory.
- Current answer objects use `Claim` and `Citation` language. The new domain model needs Answer Segments, Evidence Spans, Evidence Relations, and Findings.
- There are two config systems: `local_paper_qa/settings.py` and `local_paper_qa/config/manager.py`.

## Architecture Shape

Keep the architecture boring. Prefer a few deep modules with small interfaces over many shallow wrappers.

### Proposed Modules

`local_paper_qa/domain.py`

Owns project vocabulary as plain dataclasses or Pydantic models:

- `ResearchProject`
- `Paper`
- `ExtractedPaper`
- `EvidenceSpan`
- `EvidenceRelation`
- `EvidenceSet`
- `AnswerSegment`
- `WorkbenchAnswer`
- `QuestionScope`

Keep this file small and dependency-light. It should not call models, databases, or parsers.

`local_paper_qa/extraction.py`

Owns conversion from PDFs to extracted paper content:

- Docling first, PyPDF fallback.
- Preserve page number, section, and exact text.
- Produce extraction status per Paper.
- No retrieval or LLM calls.

This should absorb useful pieces from `parser.py`, `_extract_pdf_pages`, `_extract_metadata`, `_build_chunks`, `_paragraphs`, and `_section_heading`.

`local_paper_qa/corpus_store.py`

Owns the persisted Extracted Corpus:

- Store Papers, extracted text spans, source file state, embeddings, and lexical index data.
- Use SQLite as the simple durable store.
- Use SQLite FTS for exact-term search.
- Keep embedding storage simple and swappable.
- Expose a small interface:
  - `refresh() -> CorpusStatus`
  - `list_papers() -> list[Paper]`
  - `get_span(span_id) -> EvidenceSpan`
  - `search(request) -> list[EvidenceSpan]`

This replaces the JSON-first `index.json` flow over time.

`local_paper_qa/model_clients.py`

Owns configurable model calls:

- chat/synthesis model
- embedding model
- optional reranker later

The rest of the code should not know whether a model is local, hosted, or OpenAI-compatible. It should ask through this module.

`local_paper_qa/retrieval.py`

Owns hybrid retrieval and evidence selection:

- semantic retrieval
- exact-term retrieval
- deduplication
- useful coverage across Papers
- disagreement-aware searches when useful
- Candidate Evidence only for inspection or broadened search

Keep the first implementation simple: combine normalized semantic and lexical scores, then diversify by Paper.

`local_paper_qa/answering.py`

Owns answer synthesis:

- classify scope lightly: Paper Question, Idea Question, Finding Search
- create clean prose answers
- create Answer Segments
- attach Evidence Spans and Evidence Relations to segments
- keep Background Context separate and only when asked

It should return structured `WorkbenchAnswer`, not just a string.

`local_paper_qa/workbench.py`

Small facade used by CLI, API, and TUI:

- `refresh_corpus()`
- `list_papers()`
- `ask(question, reading_context=None)`
- `inspect_segment(answer_id, segment_id)`
- `open_paper_location(span_id)`

This replaces `LocalPaperQA` as the main interface. `LocalPaperQA` can remain temporarily as a compatibility adapter.

`local_paper_qa/lineage/`

Leave lineage mostly alone during v1. It is useful but not part of the core evidence loop. Do not let lineage complexity block the Extracted Corpus and Answer Segment work.

## Implementation Slices

### Slice 1: Stabilize The Domain Objects

Add `local_paper_qa/domain.py`.

Map old names to new names conservatively:

- `PaperDocument` -> `Paper` / `ExtractedPaper`
- `PaperChunk` -> internal extracted text span
- `PaperCitation` -> `EvidenceSpan`
- `SupportedClaim` -> likely temporary compatibility type
- `StructuredAnswer` -> `WorkbenchAnswer`

Do not rename the whole codebase in one pass. Add new types, then migrate callers gradually.

Tests:

- domain object serialization
- Evidence Span minimum fields: Paper, page, section, quote
- Answer Segment can reference one or more Evidence Spans with relations

### Slice 2: Extract The Extraction Module

Move PDF parsing, metadata extraction, section detection, paragraph splitting, and span creation out of `service.py`.

Expected result:

- `service.py` no longer contains parsing or chunking details.
- extraction returns structured spans with page and section.
- poor extraction is detected before QA.

Keep the code flat:

- no recursive parsers
- no deep class hierarchy
- small pure helpers for text cleanup
- one public `extract_paper(path) -> ExtractedPaper`

Tests:

- text PDF extracts pages and spans
- empty/scanned-looking PDF marks poor Extraction Quality
- section detection stays stable
- spans always carry page and section

### Slice 3: Build The Persisted Extracted Corpus

Create `corpus_store.py`.

Use SQLite tables for:

- `papers`
- `spans`
- `source_files`
- `embeddings`
- `span_fts`

Keep migrations simple in v1: create missing tables on startup. Avoid a migration framework until schema churn proves we need one.

Expected result:

- repeated questions do not re-parse PDFs
- changed PDFs are re-extracted
- deleted PDFs are removed from the Extracted Corpus
- TUI can list Papers and extraction status like `ls`

Tests:

- first refresh extracts all PDFs
- second refresh reuses existing extracted content
- changed file re-extracts only that Paper
- failed extraction excludes Paper from citation-bearing search

### Slice 4: Make Retrieval Actually Hybrid

Create `retrieval.py`.

First version:

- embed the user question
- search semantic embeddings
- search FTS exact terms
- merge scores
- diversify by Paper
- return an Evidence Set

Avoid cleverness:

- no agentic recursive retrieval
- no multi-hop search until simple retrieval is reliable
- no forced support/opposition sections

Tests:

- exact acronym queries find exact mentions
- semantic paraphrase queries find related spans
- repeated spans from one Paper do not crowd out all other Papers
- Finding Search can retrieve support and disagreement when present

### Slice 5: Add Structured Answers With Inspectable Segments

Create `answering.py`.

The answer prompt should request structured JSON with:

- answer text
- Answer Segments
- segment-to-evidence mappings
- Evidence Relations
- optional Background Context only when asked

The visible answer remains clean prose. Evidence is shown only when a segment is inspected.

Tests:

- every evidence-bearing segment maps to at least one Evidence Span
- unsupported corpus claims are rejected or marked as evidence gaps
- Background Context is separated from corpus-grounded answer content
- malformed model JSON falls back to a conservative answer

### Slice 6: Introduce The Workbench Facade

Create `workbench.py`.

CLI, API, and TUI should call this facade instead of reaching into indexing, retrieval, and answer synthesis directly.

Temporary compatibility:

- keep `LocalPaperQA` but make it delegate to the Workbench where possible
- remove old `LocalPaperQA` internals only after CLI/TUI/API are migrated

Tests:

- CLI can refresh and ask through Workbench
- API can list Papers and ask through Workbench
- TUI can ask through Workbench

### Slice 7: Reshape The Existing TUI

Keep Textual. Do not fork another TUI for v1.

TUI changes:

- Papers view behaves like `ls` for the Literature Corpus.
- Main answer pane shows clean prose.
- Answer Segments are selectable.
- Inspector pane shows Evidence Spans and Evidence Relations for the selected segment.
- Paper Questions use Reading Context by default.
- Idea Questions search the Literature Corpus.
- Prompt-based Finding Search works from text in the current question.

Reduce `tui.py` by extracting view helpers only when they hide real complexity:

- `tui_state.py` for small state dataclasses if needed
- `tui_rendering.py` for pure Rich/Textual rendering helpers if needed

Do not create many tiny files just to move lines around.

Manual checks:

- launch TUI
- list Papers
- ask a Paper Question
- inspect an Answer Segment
- ask an Idea Question
- confirm evidence does not appear inline by default

### Slice 8: Keep Evaluation Small But Real

Replace the current keyword benchmark with a small evidence-focused harness.

Gold case shape:

```json
{
  "question": "...",
  "expected_papers": ["..."],
  "expected_terms": ["..."],
  "must_have_evidence": true,
  "question_type": "idea"
}
```

Measure:

- right Paper retrieved
- answer has inspectable segments
- segment evidence includes page, section, quote
- unsupported answers are avoided

Do not chase automated judgment quality in v1. The harness should catch regressions, not certify scientific truth.

## Code Readability Rules

- Keep public module interfaces small.
- Prefer dataclasses and plain functions before abstract base classes.
- Add a seam only when two real adapters exist or the ADR requires configurability.
- Avoid recursive control flow unless the data is naturally recursive.
- Keep methods under roughly 40 lines unless splitting would create shallow pass-through helpers.
- Prefer explicit intermediate names over dense comprehensions for scoring and evidence mapping.
- Do not hide failures with bare `except Exception: return ""`; return status objects or log actionable errors.
- Keep prompts in one module, near their parsers and fallback behavior.
- Keep TUI rendering separate from Corpus Question execution.
- Delete compatibility code once callers have moved; do not keep both old and new paths indefinitely.

## Suggested File Migration Order

1. Add `domain.py`.
2. Add `extraction.py`; move parsing/chunking out of `service.py`.
3. Add `corpus_store.py`; persist Extracted Corpus in SQLite.
4. Add `model_clients.py`; consolidate model config and calls.
5. Add `retrieval.py`; implement hybrid retrieval over the persisted corpus.
6. Add `answering.py`; return structured Answer Segments.
7. Add `workbench.py`; migrate CLI/API/TUI to it.
8. Trim `service.py` into a compatibility adapter or delete it.
9. Split only the TUI pieces that improve locality.
10. Update benchmarks into an evidence-focused harness.

## First PR

The first PR should be intentionally small:

- add `domain.py`
- add `extraction.py`
- move PDF extraction and span creation out of `service.py`
- keep existing CLI/TUI/API behavior working
- add tests for extracted spans and extraction status

This creates the first deep module without forcing the whole architecture rewrite at once.
