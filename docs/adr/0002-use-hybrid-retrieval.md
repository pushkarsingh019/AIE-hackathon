# Use Hybrid Retrieval

Corpus Questions use hybrid retrieval that combines semantic matching with exact-term matching. Scientific literature review depends on both meaning and precise terms such as methods, acronyms, datasets, genes, metrics, and result phrases, so vector-only retrieval is not sufficient for trustworthy evidence discovery.

Retrieval can use multiple Retrieval Representations of the same Paper content, including exact Evidence Span representations, contextual span representations, and paper-level representations. These representations help discover candidate material, but answer grounding and user-visible citations resolve back to Evidence Spans.

Indexing may also create model-derived Retrieval Notes, such as paper claims, methods, measured outcomes, limitations, and result patterns. Retrieval Notes are discovery aids only: they can help find Candidate Evidence, but they are not cited evidence unless resolved to exact Evidence Spans.

Because scientific findings can be carried by figures, tables, diagrams, and page layout, the retrieval architecture must allow Visual Evidence as a first-class citation source. Text extraction can be the first implementation path, but the corpus model should not assume all useful evidence is plain paragraph text.

The first multimodal retrieval path should focus on figures: render pages, identify figure references, create Figure Notes describing what each figure appears to show, and attach those notes to inspectable Visual Evidence such as the page, figure label, caption, or crop. Figure Notes are retrieval aids, not proof by themselves.

Figure Notes should use caption-guided interpretation: they may combine the figure image, figure label, caption, and nearby Paper text to describe what the Paper claims the figure shows. They should not use broad free-form scientific inference as evidence unless that background context is separately requested by the user.

Figure indexing should run automatically for every detected figure during indexing. When exact figure cropping is not reliable, the system should fall back to page-level Visual Evidence tied to the figure label, caption, and page location rather than skipping the figure.

Visual Artifacts should be stored locally in predictable project cache paths so the user can inspect them directly from the terminal or TUI. These artifacts are part of the Extracted Corpus rather than opaque temporary files.
