# Embedding And Figure Indexing Implementation Plan

This plan defines the retrieval-indexing work for the Literature Review Workbench. The intent is quality-first indexing: spend time and model cost during corpus refresh so Corpus Questions have rich, inspectable material to search later.

## Target Behavior

The Extracted Corpus should contain multiple Retrieval Representations for each Paper while preserving Evidence Spans and Visual Evidence as the only citation-bearing units.

At query time, retrieval should be able to use:

- exact text search over Evidence Spans
- exact-span embeddings
- contextual-span embeddings
- paper-level embeddings
- Retrieval Note embeddings
- Figure Note embeddings

Answers should cite:

- Evidence Spans for textual support
- Visual Evidence for figure-derived support

Retrieval Notes and Figure Notes help find evidence. They are not evidence by themselves.

## Non-Goals For The First Serious Pass

- Do not embed whole PDFs as the primary retrieval source.
- Do not use the old local embedding endpoint as a fallback.
- Do not solve full figure/panel segmentation immediately.
- Do not treat model-generated notes as cited facts unless they resolve to Evidence Spans or Visual Evidence.
- Do not build a complex evidence-ranker UI.

## Proposed Modules

`local_paper_qa/corpus_store.py`

Owns the Extracted Corpus SQLite database and local Visual Artifact paths.

Responsibilities:

- store Papers, source files, Evidence Spans, Visual Evidence, Retrieval Notes, Figure Notes, Retrieval Representations, embeddings, and FTS content
- track content hashes and indexing status
- expose small read/write methods for extraction, indexing, retrieval, and TUI status

`local_paper_qa/indexing.py`

Owns the refresh pipeline.

Responsibilities:

- detect changed Papers
- run extraction
- build Retrieval Representations
- call embedding providers in batches
- run figure indexing when configured
- record indexing status and failures

`local_paper_qa/model_clients.py`

Owns provider adapters.

Responsibilities:

- chat model calls
- text embedding calls
- multimodal figure-description calls
- hosted embedding configuration
- retries, timeouts, dimensions, and batch sizing

`local_paper_qa/retrieval.py`

Owns search and Evidence Set construction.

Responsibilities:

- FTS exact search
- semantic search across representation types
- merge and normalize scores
- resolve Retrieval Notes and Figure Notes back to Evidence Spans or Visual Evidence
- diversify results across Papers and evidence types

`local_paper_qa/visuals.py`

Owns figure-oriented PDF visual processing.

Responsibilities:

- render PDF pages into local Visual Artifacts
- detect figure labels and captions from extracted text
- create figure/page artifact records
- provide crop support later without changing retrieval contracts

## Data Model

Use SQLite for v1. Keep schemas explicit and boring.

### `papers`

- `paper_id`
- `title`
- `authors`
- `year`
- `venue`
- `doi`
- `abstract`
- `file_path`
- `page_count`
- `extraction_quality`
- `extraction_message`

### `source_files`

- `paper_id`
- `path`
- `size`
- `mtime`
- `sha256`

### `evidence_spans`

- `span_id`
- `paper_id`
- `page`
- `section`
- `quote`
- `span_hash`
- `metadata_json`

### `visual_evidence`

- `visual_id`
- `paper_id`
- `page`
- `figure_label`
- `caption`
- `artifact_path`
- `nearby_span_ids_json`
- `metadata_json`

`artifact_path` should be relative to `.research_index/` and easy to inspect from the terminal.

### `retrieval_notes`

- `note_id`
- `paper_id`
- `note_type`
- `content`
- `source_span_ids_json`
- `note_hash`
- `metadata_json`

Expected `note_type` values:

- `paper_claims`
- `methods`
- `measured_outcomes`
- `result_patterns`
- `limitations`

### `figure_notes`

- `note_id`
- `visual_id`
- `paper_id`
- `figure_label`
- `page`
- `caption`
- `visual_description`
- `paper_claim_about_figure`
- `measured_variables`
- `direction_of_result`
- `limitations_or_uncertainty`
- `linked_span_ids_json`
- `note_hash`

### `retrieval_representations`

- `representation_id`
- `paper_id`
- `source_type`
- `source_id`
- `representation_type`
- `content`
- `content_hash`
- `metadata_json`

Expected `representation_type` values:

- `quote`
- `contextual_span`
- `paper`
- `retrieval_note`
- `figure_note`

### `embeddings`

- `representation_id`
- `provider`
- `model`
- `dimension`
- `profile`
- `input_hash`
- `embedding`
- `created_at`

Store embeddings in a format that can be loaded quickly. JSON is acceptable for the first migration if vector volume is small, but the interface should allow a later sqlite-vec, LanceDB, or Qdrant backend.

### `span_fts`

SQLite FTS table over:

- quote
- section
- paper title
- abstract

### `representation_fts`

SQLite FTS table over Retrieval Representation content, including Retrieval Notes and Figure Notes.

## Local Artifact Layout

Use predictable paths:

```text
papers/.research_index/
  corpus.db
  artifacts/
    <paper_id>/
      pages/
        page-001.png
        page-002.png
      figures/
        figure-001-page-003.png
        figure-002-page-005.png
```

If exact figure cropping is unavailable, store the rendered page artifact and attach the figure label/caption to that page.

## Configuration

Add one coherent config shape and phase out duplicate config systems.

```toml
[indexing]
quality = "best_available"  # best_available | local | balanced | cheap
profile = "deep"            # fast | standard | deep | deep_figures

[embedding]
provider = "openai"
model = "text-embedding-3-large"
dimension = 3072
batch_size = 64

[multimodal]
provider = "openai"         # auto | openai
model = "gpt-5.5"
figure_indexing = "auto"    # auto | off
```

Default behavior:

- prefer the strongest available encoder profile
- use hosted embeddings by default; keep local models for chat only
- automatically index detected figures in `deep_figures`
- keep final citations tied to Evidence Spans or Visual Evidence

## Pipeline

### Phase 1: Persist Text Corpus First

Build `corpus_store.py` and move away from JSON-first indexing.

Steps:

1. Create SQLite tables for Papers, source files, Evidence Spans, FTS, Retrieval Representations, and embeddings.
2. Store extracted spans from the existing `extraction.py`.
3. Track file hashes so unchanged Papers are not re-extracted.
4. Keep current `LocalPaperQA` behavior by reading from the new store through an adapter.

Acceptance checks:

- second refresh skips unchanged PDFs
- changed PDF re-extracts only that Paper
- TUI can list Papers with extraction quality
- no question path reparses PDFs unnecessarily

### Phase 2: Add Embedding Provider Abstraction

Build `model_clients.py`.

Steps:

1. Add an `EmbeddingProvider` interface with `embed_texts(inputs)`.
2. Implement hosted embedding providers.
3. Add provider-level batching, timeout, retry, and dimension validation.
4. Record failed embeddings explicitly instead of silently storing empty vectors.
5. Include provider, model, dimension, and input hash in embedding identity.

Acceptance checks:

- embeddings are batched
- changing model or dimension invalidates only affected embeddings
- failed embedding calls show indexing status
- retrieval does not pretend semantic search succeeded when embeddings are missing

### Phase 3: Build Text Retrieval Representations

Generate multiple text Retrieval Representations.

Steps:

1. `quote`: exact Evidence Span text.
2. `contextual_span`: title, abstract, section, neighboring spans, exact quote.
3. `paper`: title, abstract, introduction/conclusion snippets when available.
4. `retrieval_note`: model-derived notes for claims, methods, outcomes, patterns, and limitations.
5. Embed all representations according to indexing profile.

Acceptance checks:

- each representation resolves to a Paper and, where applicable, source Evidence Spans
- Retrieval Notes never become cited evidence directly
- embedding refresh reuses unchanged representation hashes
- exact FTS remains available even if embeddings fail

### Phase 4: Implement Hybrid Retrieval Over Representations

Replace in-memory chunk scanning with retrieval over the persisted corpus.

Steps:

1. Embed the Corpus Question using the configured query format.
2. Search semantic representations.
3. Search `span_fts` and `representation_fts`.
4. Merge scores by normalized rank rather than raw model score only.
5. Resolve candidate representations to Evidence Spans or Visual Evidence.
6. Diversify by Paper and evidence type.

Acceptance checks:

- exact acronym queries find exact text
- paraphrase queries find semantically related spans
- paper-level hits expand into relevant spans
- Retrieval Note hits resolve to source spans before answering
- no single Paper crowds out all results unless the corpus really only has one relevant Paper

### Phase 5: Add Automatic Figure Indexing

Build the first figure path without full segmentation.

Steps:

1. Detect figure captions and figure labels from extracted page text.
2. Render pages containing detected figures into Visual Artifacts.
3. Store Visual Evidence records with paper, page, figure label, caption, artifact path, and nearby span ids.
4. Send the rendered page or crop to the multimodal provider.
5. Create caption-guided Figure Notes with structured fields.
6. Add `figure_note` Retrieval Representations and embeddings.

Figure Note schema:

```json
{
  "figure_label": "Figure 3",
  "page": 7,
  "caption": "...",
  "visual_description": "...",
  "paper_claim_about_figure": "...",
  "measured_variables": ["..."],
  "direction_of_result": "...",
  "limitations_or_uncertainty": "...",
  "linked_span_ids": ["..."]
}
```

Acceptance checks:

- every detected figure has a local Visual Artifact or explicit failure status
- Figure Notes are searchable
- Figure Notes cite Visual Evidence, not themselves
- artifact paths are easy to inspect with `ls`
- if exact cropping fails, page-level Visual Evidence is still stored

### Phase 6: Update Answering And Inspection

Make answers use the richer retrieval output without exposing all evidence inline.

Steps:

1. Answer from an Evidence Set containing textual Evidence Spans and Visual Evidence.
2. Segment the answer into inspectable Answer Segments.
3. Attach Evidence Relations explaining why each evidence item is present.
4. Keep Figure Notes available in the inspector as retrieval rationale.
5. Show the exact quote, caption, and artifact path when inspecting visual evidence.

Acceptance checks:

- answer text stays clean by default
- clicking a segment shows text spans and visual evidence
- visual evidence inspection shows figure label, page, caption, artifact path, and model note
- unsupported claims are marked as evidence gaps rather than smoothed over

### Phase 7: TUI Status And Local Inspection

Expose indexing state without adding mental overhead.

Steps:

1. Papers view shows extraction, embedding, and figure-indexing status.
2. Add a way to list Visual Artifacts for a Paper.
3. Add a way to open or reveal an artifact path from evidence inspection.
4. Keep folder paths predictable so terminal inspection works naturally.

Acceptance checks:

- user can see which Papers are fully indexed
- user can find generated page and figure artifacts with `ls`
- figure-indexing failures are visible before QA

## Recommended First Implementation Slice

Do not start with figure images. Start by making the indexing substrate correct while using hosted embeddings.

First slice:

1. Add `corpus_store.py`.
2. Persist Papers and Evidence Spans in SQLite.
3. Add FTS over Evidence Spans.
4. Move embeddings out of chunk metadata and into an `embeddings` table.
5. Add hosted embedding providers.
6. Keep existing retrieval behavior working through an adapter.

This gives a stable base for richer representations and automatic figure indexing.

Second slice:

1. Add Retrieval Representations.
2. Add contextual span and paper representations.
3. Add Retrieval Notes.
4. Update retrieval to search across representation types.

Third slice:

1. Add Visual Artifacts.
2. Add figure detection from captions.
3. Add rendered page artifacts.
4. Add Figure Notes through a multimodal provider.
5. Add figure-note retrieval and inspection.

## Testing Strategy

Unit tests:

- representation hash stability
- embedding identity invalidation
- provider batching and failure recording
- Figure Note parsing
- Visual Artifact path generation

Integration tests:

- generated text PDF indexes spans and FTS
- generated PDF with figure caption creates Visual Evidence
- fake embedding provider supports semantic retrieval deterministically
- fake multimodal provider creates Figure Notes deterministically
- changed PDF refreshes only affected Papers

Manual checks:

- run corpus refresh on a small literature folder
- inspect `.research_index/artifacts/` with `ls`
- ask an Idea Question requiring text evidence
- ask a question about a figure
- inspect answer segments and verify citations point to exact text or local artifacts

## Implementation Order

1. Persist Extracted Corpus in SQLite.
2. Add provider abstraction and batched embeddings.
3. Add Retrieval Representations.
4. Add Retrieval Notes.
5. Replace retrieval with persisted hybrid retrieval.
6. Add Visual Artifact storage.
7. Add automatic figure detection and Figure Notes.
8. Update answer inspection for Visual Evidence.
9. Add TUI indexing status.
10. Remove legacy JSON chunk embedding path.
