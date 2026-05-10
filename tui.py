from __future__ import annotations

import argparse
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from rich.console import Group
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.tree import Tree
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, ListItem, ListView, Static

from local_paper_qa.models import AnswerSegment, PaperCitation, PaperDocument, StructuredAnswer, SupportedClaim
from local_paper_qa.service import LocalPaperQA


@dataclass
class ChatEntry:
    role: str
    text: str = ""
    segments: list[AnswerSegment] = field(default_factory=list)


class PaperQATui(App):
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "reindex", "Reindex"),
        ("o", "open_evidence", "Open PDF"),
        ("l", "paper_lineage", "Lineage"),
        ("1", "paper_lineage", "Lineage"),
        ("d", "download_lineage_paper", "Download Lineage Paper"),
        ("escape", "clear_detail", "Clear Detail"),
        ("n", "next_citation", "Next Citation"),
        ("p", "prev_citation", "Prev Citation"),
        ("c", "show_citation_chain", "Citation Chain"),
    ]

    CLAIM_COLORS = [
        "#2563eb",
        "#7c3aed",
        "#059669",
        "#d97706",
        "#dc2626",
        "#0891b2",
        "#4f46e5",
        "#be185d",
        "#65a30d",
        "#0d9488",
        "#9333ea",
        "#ea580c",
    ]

    WAITING_LINES = [
        "RAG-time story: checking the footnotes...",
        "Looking for receipts in the PDF aisle...",
        "Embedding your curiosity locally...",
        "Cross-examining the citations...",
        "Page-turner mode: literally...",
        "Asking the papers to show their work...",
        "Scanning for the sharpest supporting sentences...",
        "Summoning evidence from the stack...",
    ]

    WELCOME_ART = r"""
   ____  ___    ____  __________     ____  ___       ____  ___
  / __ \/   |  / __ \/ ____/ __ \   / __ \/   |     / __ \/   |
 / /_/ / /| | / /_/ / __/ / /_/ /  / / / / /| |    / / / / /| |
/ ____/ ___ |/ ____/ /___/ _, _/  / /_/ / ___ |   / /_/ / ___ |
/_/   /_/  |_/_/   /_____/_/ |_|   \___\/_/  |_|   \___\/_/  |_|

       local-first paper chaos machine  //  citations or it didn't happen
       press l/1 for lineage  //  press d after lineage to ingest a paper
""".strip("\n")

    CSS = """
    Screen {
        background: #fafafa;
        color: #1a1a2e;
    }
    #root {
        height: 1fr;
    }
    #chat-pane {
        width: 1.6fr;
        height: 1fr;
        border: solid #e0e0e0;
    }
    #side-pane {
        width: 1.4fr;
        min-width: 52;
        height: 1fr;
    }
    #chat-log {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    #question-input {
        dock: bottom;
        margin: 0 1 1 1;
    }
    #indexing-banner {
        height: 5;
        margin: 1;
        padding: 1 2;
        border: heavy #3344cc;
        background: #f0f4ff;
        color: #1a1a2e;
        text-style: bold;
        display: none;
    }
    #indexing-banner.visible {
        display: block;
    }
    .panel-title {
        text-style: bold;
        color: #3344cc;
        padding: 0 1;
    }
    #references-panel {
        height: 1.2fr;
        border: solid #e0e0e0;
    }
    #paper-panel {
        height: 1.8fr;
        border: solid #e0e0e0;
    }
    #references-list, #paper-list {
        height: 1fr;
    }
    #claim-list {
        height: 30%;
    }
    #chunk-detail {
        height: 70%;
        border-top: solid #e0e0e0;
    }
    #chunk-detail-content {
        padding: 1;
        color: #1a1a2e;
    }
    ListItem {
        padding: 0 1;
    }
    #detail {
        height: 10;
        border-top: solid #e0e0e0;
        padding: 1;
        color: #1a1a2e;
        overflow-y: auto;
    }
    """

    def __init__(self, papers_dir: str = "papers"):
        super().__init__()
        self.qa = LocalPaperQA(papers_dir=papers_dir)
        self.papers_dir = self.qa.papers_dir

        self.papers: list[PaperDocument] = []
        self.citations: list[PaperCitation] = []
        self.structured_answer: StructuredAnswer | None = None

        self.active_claim_id: int | None = None
        self.active_reference_paper_id: str | None = None
        self.active_paper: PaperDocument | None = None
        self.active_lineage_items: list[dict] = []
        self.active_evidence: PaperCitation | None = None

        self.citation_chain: list[PaperCitation] = []
        self.active_citation_index: int = 0
        self.claim_color_map: dict[int, str] = {}
        self.chat_entries: list[ChatEntry] = []
        self.references_items: dict[str, str] = {}  # list item id -> paper_id
        self.claim_items: dict[str, int] = {}  # list item id -> claim_id
        self.paper_items: dict[str, int] = {}  # list item id -> index into self.papers

        self.waiting = False
        self.wait_started_at = 0.0
        self.waiting_message = "Thinking"
        self.show_welcome_art = True
        self.status_message = "Indexing status will appear here. Ask a question below."
        self.indexing = False
        self.watcher_ready = False
        self.paper_signature: tuple[tuple[str, float, int], ...] = ()

        self.papers_refresh_nonce = 0
        self.references_refresh_nonce = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="root"):
            with Vertical(id="chat-pane"):
                yield Static("Local Paper QA", classes="panel-title")
                yield Static(
                    "Indexing papers...\nNew or changed PDFs are being embedded. Unchanged PDFs are reused.",
                    id="indexing-banner",
                )
                yield Static(
                    f"Watching: {self.papers_dir}\nIndexing status will appear here. Ask a question below.",
                    id="chat-log",
                )
                yield Input(placeholder="Ask a question about your papers...", id="question-input")

            with Vertical(id="side-pane"):
                with Vertical(id="references-panel"):
                    yield Static("References", classes="panel-title")
                    yield ListView(id="references-list")
                with Vertical(id="paper-panel"):
                    yield Static("Papers In Project", classes="panel-title")
                    yield ListView(id="paper-list")
                    yield Static("Inspector (click a reference claim)", classes="panel-title", id="detail-header")
                    yield ListView(id="claim-list")
                    with VerticalScroll(id="chunk-detail"):
                        yield Static("Select a paper or a reference claim to inspect sentences.", id="chunk-detail-content")

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#question-input", Input).focus()
        self.set_interval(0.4, self._tick_waiting)
        self.load_index()
        self.set_interval(5, self.check_for_paper_changes)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Trigger chat generation when the user presses Enter in the question box.
        if self.waiting or self.indexing:
            return

        question = (getattr(event, "value", None) or "").strip()
        if not question:
            return

        input_widget = self.query_one("#question-input", Input)
        input_widget.value = ""

        self._add_user_message(question)
        self.set_waiting(True, "Generating")
        self.answer_question(question)

    def _tick_waiting(self) -> None:
        if not self.waiting:
            return
        self.render_chat()

    @work(thread=True)
    def load_index(self, force: bool = False) -> None:
        if self.indexing:
            return
        self.indexing = True
        self.call_from_thread(self._show_indexing_banner, True)
        self.call_from_thread(self._set_status, "Loading paper index...")

        try:
            papers = self.qa.ensure_index(force=force)
            self.call_from_thread(self.set_papers, papers)

            if force:
                self.call_from_thread(self.set_references, [])
                self.call_from_thread(self.clear_detail)
                self.structured_answer = None
                self.claim_color_map = {}
                self.citations = []

            signature = self.current_paper_signature()
            self.call_from_thread(self.set_paper_signature, signature)
            self.call_from_thread(self._set_status, f"Ready. {len(papers)} papers indexed.")
        finally:
            self.call_from_thread(self._show_indexing_banner, False)
            self.indexing = False

    @work(thread=True)
    def answer_question(self, question: str) -> None:
        # UI transitions are done via call_from_thread.
        try:
            papers = self.qa.ensure_index()
            citations = self.qa.retrieve(question, papers=papers)

            self.call_from_thread(self.set_papers, papers)
            self.call_from_thread(self.set_references, citations)
            self.call_from_thread(self._set_status, "Generating answer...")

            structured = self.qa.answer_with_claims(question, citations)
            self.call_from_thread(self._apply_structured_answer, structured)
            self.call_from_thread(self._add_assistant_message, structured)
            self.call_from_thread(self._set_status, "Ready.")
        except Exception as e:
            # If the background work fails, ensure the input is re-enabled.
            self.call_from_thread(self._set_status, f"Error: {e}")
        finally:
            self.call_from_thread(self.set_waiting, False)

    def action_next_citation(self) -> None:
        if not self.citation_chain:
            return
        self.active_citation_index = (self.active_citation_index + 1) % len(self.citation_chain)
        self._show_active_citation()

    def action_prev_citation(self) -> None:
        if not self.citation_chain:
            return
        self.active_citation_index = (self.active_citation_index - 1) % len(self.citation_chain)
        self._show_active_citation()

    def action_show_citation_chain(self) -> None:
        if not self.citation_chain:
            self._set_status("No citation chain available. Ask a question first.")
            return
        self._set_status(f"Citation chain: {len(self.citation_chain)} citations. Use n/p to navigate.")

    def _show_active_citation(self) -> None:
        if not self.citation_chain or self.active_citation_index >= len(self.citation_chain):
            return
        citation = self.citation_chain[self.active_citation_index]
        self.show_reference_detail(citation.paper_id)
        self._set_status(f"Citation {self.active_citation_index + 1}/{len(self.citation_chain)}: {citation.paper_title[:50]}")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id in self.claim_items:
            self.show_claim_detail(self.claim_items[item_id])
            return
        if item_id in self.references_items:
            self.show_reference_detail(self.references_items[item_id])
            return
        if item_id in self.paper_items:
            idx = self.paper_items[item_id]
            if 0 <= idx < len(self.papers):
                self.show_paper_detail(self.papers[idx])

    def action_reindex(self) -> None:
        self._set_status("Reindex requested.")
        self.load_index(force=True)

    def action_clear_detail(self) -> None:
        self.clear_detail()

    def action_paper_lineage(self) -> None:
        if self.waiting or self.indexing:
            return
        if not self.active_paper:
            self._set_status("Select a paper in Papers In Project first, then press l for lineage.")
            return
        paper = self.active_paper
        self.show_lineage_loading(paper)
        self.set_waiting(True, f"Searching paper lineage for {paper.title}")
        self.lookup_paper_lineage(paper)

    def action_download_lineage_paper(self) -> None:
        if self.waiting or self.indexing:
            return
        if not self.active_lineage_items:
            self._set_status("Run lineage first, then press d to download a lineage paper.")
            return
        title = str(self.active_lineage_items[0].get("title") or "lineage paper")
        self.show_download_loading(title)
        self.set_waiting(True, f"Downloading lineage paper {title}")
        self.download_lineage_paper(list(self.active_lineage_items))

    @work(thread=True)
    def download_lineage_paper(self, items: list[dict]) -> None:
        try:
            path, source_title = self.qa.download_first_available_lineage_paper(items)
            papers = self.qa.ensure_index(force=True)
            self.call_from_thread(self.set_papers, papers)
            self.call_from_thread(self.set_paper_signature, self.current_paper_signature())
            self.call_from_thread(self._set_status, f"Downloaded and indexed new paper: {path.name}")
            self.call_from_thread(self.show_download_complete, path, source_title)
        except Exception as e:
            self.call_from_thread(self._set_status, f"Download failed: {e}")
        finally:
            self.call_from_thread(self.set_waiting, False)

    @work(thread=True)
    def lookup_paper_lineage(self, paper: PaperDocument) -> None:
        try:
            lineage = self.qa.paper_lineage(paper)
            self.call_from_thread(self.show_lineage_detail, lineage)
            self.call_from_thread(self._set_status, f"Lineage saved to {lineage.get('lineage_file', 'papers/')}")
        except Exception as e:
            self.call_from_thread(self._set_status, f"Lineage lookup failed: {e}")
        finally:
            self.call_from_thread(self.set_waiting, False)

    def action_open_evidence(self) -> None:
        if not self.active_reference_paper_id:
            self._set_status("Select a reference claim first.")
            return
        if not self.active_evidence or self.active_evidence.page is None:
            self._set_status("No active evidence to open.")
            return

        paper = next((p for p in self.papers if p.paper_id == self.active_reference_paper_id), None)
        if not paper:
            self._set_status("Could not resolve paper file path.")
            return

        page = int(self.active_evidence.page)
        pdf_path = paper.file_path
        if not Path(pdf_path).exists():
            self._set_status("Evidence PDF missing on disk. Reindexing...")
            self.load_index(force=True)
            return

        # 1) Open + jump page using AppleScript (Preview reliably supports go-to-page)
        # 2) Copy the evidence sentence so you can instantly search/spot it
        # Copy a larger snippet so you can find the exact portion in Preview search.
        evidence_text = (self.active_evidence.quote or "").strip()
        if not evidence_text:
            snippet = ""
        else:
            # Prefer a sentence if we can split; else fall back to a chunk prefix.
            sents = self.split_sentences(evidence_text)
            snippet = sents[0] if sents else self.trim(evidence_text, 900)

        if snippet:
            try:
                proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                proc.communicate(input=snippet.encode("utf-8"), timeout=2)
            except Exception:
                pass

        # Preview page scripting is often 0-based vs our 1-based PDF page numbers.
        preview_page = max(page - 1, 1)

        try:
            # Escape for AppleScript (POSIX file expects a quoted string).
            apath = pdf_path.replace("\\", "\\\\").replace('"', '\\"')

            # Preview scripting:
            # - open via AppleScript (one statement)
            # - jump via document property: `set page to <n>`
            open_script = f'tell application "Preview" to open POSIX file "{apath}"'
            set_page_script = f'tell application "Preview" to tell document 1 to set page to {preview_page}'

            subprocess.run(["osascript", "-e", open_script], check=False, timeout=10)
            # Small delay so the opened document becomes the front document.
            time.sleep(0.2)
            subprocess.run(["osascript", "-e", set_page_script], check=False, timeout=10)

            sec = self.active_evidence.section or "Unknown section"
            sc = float(self.active_evidence.score or 0.0)
            self._set_status(
                f"Opened {paper.title} (requested page {page}, set Preview page {preview_page}) (section: {sec}, score: {sc:.3f}). Evidence snippet copied to clipboard."
            )
        except Exception as e:
            # Fallback: open as a URL.
            try:
                encoded_path = quote(pdf_path)
                target_url = f"file://{encoded_path}#page={page}"
                subprocess.Popen(["open", target_url])
                self._set_status(f"Opened {paper.title} (page {page} may not be set). Evidence snippet copied.")
            except Exception:
                self._set_status(f"Failed to open PDF: {e}")

    def clear_detail(self) -> None:
        self.active_claim_id = None
        self.active_reference_paper_id = None
        self.active_paper = None
        self.active_lineage_items = []
        self.active_evidence = None
        self.active_citation_index = 0
        self.claim_color_map = self.claim_color_map if self.claim_color_map else {}
        self.query_one("#chunk-detail-content", Static).update("Select a paper or a reference claim to inspect sentences.")
        self.query_one("#claim-list", ListView).clear()
        self.references_items = getattr(self, "references_items", {})
        self.render_chat()

    def check_for_paper_changes(self) -> None:
        signature = self.current_paper_signature()
        if self.watcher_ready and signature != self.paper_signature and not self.indexing and not self.waiting:
            self._set_status("PDF folder changed. Reindexing automatically...")
            self.load_index(force=True)

    def current_paper_signature(self) -> tuple[tuple[str, float, int], ...]:
        return tuple(
            sorted(
                (str(path), path.stat().st_mtime, path.stat().st_size)
                for path in self.papers_dir.glob("*.pdf")
                if path.is_file()
            )
        )

    def set_paper_signature(self, signature: tuple[tuple[str, float, int], ...]) -> None:
        self.paper_signature = signature
        self.watcher_ready = True

    def _show_indexing_banner(self, visible: bool) -> None:
        banner = self.query_one("#indexing-banner", Static)
        banner.set_class(visible, "visible")

    def _set_status(self, message: str) -> None:
        self.status_message = message
        self.render_chat()

    def set_waiting(self, waiting: bool, message: str = "Thinking") -> None:
        self.waiting = waiting
        if waiting:
            self.wait_started_at = time.time()
            self.waiting_message = message
            self.query_one("#question-input", Input).disabled = True
            self._set_status(f"{message}...")
        else:
            self.query_one("#question-input", Input).disabled = False
            self.wait_started_at = 0.0
            self.waiting_message = "Thinking"
            self._set_status(self.status_message or "Ready.")
        self.render_chat()

    def _add_user_message(self, question: str) -> None:
        self.chat_entries.append(ChatEntry(role="You", text=question))
        self.render_chat()

    def _add_assistant_message(self, structured: StructuredAnswer) -> None:
        self.chat_entries.append(ChatEntry(role="Assistant", text=structured.answer, segments=structured.segments))
        self.render_chat()

    def _render_chat_entry(self, entry: ChatEntry) -> Panel:
        if entry.role == "You":
            text = Text(entry.text, style=Style(color="#059669"))
            return Panel(text, border_style="#059669")
        else:
            return Panel(self._render_answer_segments(entry.segments, entry.text), border_style="#e0e0e0")

    def _apply_structured_answer(self, structured: StructuredAnswer) -> None:
        self.structured_answer = structured
        self.claim_color_map = {claim.claim_id: self.CLAIM_COLORS[i % len(self.CLAIM_COLORS)] for i, claim in enumerate(structured.claims)}

    def render_chat(self) -> None:
        renderables: list = []

        # Status line / waiting panel
        if self.waiting:
            elapsed = time.time() - self.wait_started_at
            pun = self.WAITING_LINES[int(elapsed) % len(self.WAITING_LINES)]
            waiting_text = Text(f"{self.waiting_message} for {elapsed:.1f}s...\n{pun}", style=Style(color="#3344cc", bold=True))
            renderables.append(Panel(waiting_text, border_style="#3344cc"))
        else:
            renderables.append(Text(self.status_message, style=Style(color="#1a1a2e")))

        if self.show_welcome_art and not self.chat_entries:
            art = Text(self.WELCOME_ART, style=Style(color="#3344cc", bold=True))
            renderables.append(Panel(art, title="PAPER PUNK", border_style="#3344cc"))

        # Chat history
        for entry in self.chat_entries:
            renderables.append(self._render_chat_entry(entry))

        self.query_one("#chat-log", Static).update(Group(*renderables))

    def _render_answer_segments(self, segments: list[AnswerSegment], fallback_answer: str = "") -> Text:
        if segments:
            out = Text()
            for i, seg in enumerate(segments):
                claim_id = seg.claim_id
                underline = self.active_claim_id is not None and claim_id == self.active_claim_id
                if underline and claim_id in self.claim_color_map:
                    style = Style(color=self.claim_color_map[claim_id], underline=True, bold=True)
                else:
                    style = Style(color="#4a5568")
                sep = " " if i > 0 else ""
                out.append(sep + seg.text, style=style)
            return out
        if fallback_answer:
            return Text(fallback_answer, style=Style(color="#4a5568"))
        return Text("(no answer)", style=Style(color="#666"))

    def set_papers(self, papers: list[PaperDocument]) -> None:
        self.papers = papers
        self.papers_refresh_nonce += 1
        self.paper_items = {}

        paper_list = self.query_one("#paper-list", ListView)
        paper_list.clear()

        if not papers:
            paper_list.append(ListItem(Static("No PDFs found in papers/"), id=f"paper-empty-{self.papers_refresh_nonce}"))
            return

        for idx, paper in enumerate(papers):
            item_id = f"paper-{self.papers_refresh_nonce}-{idx}"
            self.paper_items[item_id] = idx
            title = self.trim(paper.title, 46)
            meta = f"{paper.year} · {paper.page_count} pages"
            paper_list.append(ListItem(Static(f"{idx + 1}. {title}\n   {meta}"), id=item_id))

    def set_references(self, citations: list[PaperCitation]) -> None:
        self.citations = citations
        self.references_refresh_nonce += 1
        self.references_items = {}

        # Build citation chain for n/p navigation
        self.citation_chain = list(citations)
        self.active_citation_index = 0

        references_list = self.query_one("#references-list", ListView)
        references_list.clear()

        if not citations:
            references_list.append(ListItem(Static("No matching references."), id=f"ref-empty-{self.references_refresh_nonce}"))
            return

        by_paper: dict[str, list[tuple[int, PaperCitation]]] = defaultdict(list)
        for idx, c in enumerate(citations):
            by_paper[c.paper_id].append((idx, c))

        for paper_id, items in by_paper.items():
            # Choose top-scoring citation for preview
            items_sorted = sorted(items, key=lambda t: t[1].score, reverse=True)
            top_c = items_sorted[0][1]
            sentences = self.split_sentences(top_c.quote)
            preview = sentences[0] if sentences else self.trim(top_c.quote, 90)

            label = f"{self.trim(top_c.paper_title, 36)}\n   {top_c.year} · p. {top_c.page} · {top_c.section}\n   \"{self.trim(preview, 62)}\""
            item_id = f"ref-{self.references_refresh_nonce}-{paper_id}"
            self.references_items[item_id] = paper_id
            references_list.append(ListItem(Static(label), id=item_id))

    def show_reference_detail(self, paper_id: str) -> None:
        self.active_reference_paper_id = paper_id
        self.active_paper = None
        self.active_claim_id = None
        self.render_chat()

        citations_for_paper = [c for c in self.citations if c.paper_id == paper_id]
        if not citations_for_paper:
            self.query_one("#chunk-detail-content", Static).update("No references for this paper.")
            self.query_one("#claim-list", ListView).clear()
            self.claim_items = {}
            return

        # Populate claim list for this paper (click a claim to underline its answer parts).
        self.claim_items = {}
        claim_list = self.query_one("#claim-list", ListView)
        claim_list.clear()

        if not self.structured_answer or not self.structured_answer.claims:
            self.query_one("#chunk-detail-content", Static).update("No structured claim mapping available.")
            return

        claims_for_paper: list[SupportedClaim] = []
        for claim in self.structured_answer.claims:
            if any(
                1 <= cid <= len(self.citations) and self.citations[cid - 1].paper_id == paper_id
                for cid in claim.citation_ids
            ):
                claims_for_paper.append(claim)

        if not claims_for_paper:
            self.query_one("#chunk-detail-content", Static).update("No claims mapped to this paper.")
        else:
            for claim in sorted(claims_for_paper, key=lambda c: c.claim_id):
                color = self.claim_color_map.get(claim.claim_id, "#4a5568")
                preview = self.trim(claim.text, 90)

                # Best citation score for this claim within the selected paper.
                claim_scores: list[float] = []
                for cid in claim.citation_ids:
                    if 1 <= cid <= len(self.citations):
                        pc = self.citations[cid - 1]
                        if pc.paper_id == paper_id and pc.score is not None:
                            claim_scores.append(float(pc.score or 0.0))
                best_score = max(claim_scores) if claim_scores else None

                # Include refresh nonce so Textual never sees duplicates during refresh.
                item_id = f"claim-item-{self.references_refresh_nonce}-{paper_id}-{claim.claim_id}"
                self.claim_items[item_id] = claim.claim_id
                claim_list.append(
                    ListItem(
                        Static(
                            f"Claim {claim.claim_id}: {preview}"
                            + (f"  · score {best_score:.3f}" if best_score is not None else "")
                        ),
                        id=item_id,
                    )
                )

        # Show full evidence chunk(s) sentences for this paper.
        title = citations_for_paper[0].paper_title
        authors = citations_for_paper[0].authors
        year = citations_for_paper[0].year
        paper_citations_sorted = sorted(citations_for_paper, key=lambda c: (-(c.score or 0.0)))

        evidence_panels: list = []
        for c in paper_citations_sorted[:6]:
            sentences = self.split_sentences(c.quote)
            sent_list = Text("\n".join([f"- {s}" for s in sentences[:25]]) or "- (no sentences found)")
            evidence_panels.append(
                Panel(
                    Group(
                        Text(f"p. {c.page} · {c.section} · score {c.score:.3f}", style=Style(color="#4a5568", bold=True)),
                        Text("\nSentences preview (from evidence text):", style=Style(bold=True)),
                        sent_list,
                        Text("\nEvidence chunk (truncated):", style=Style(bold=True)),
                        Text(self.trim(c.quote, 1600)),
                    ),
                    border_style="#e0e0e0",
                )
            )

        content = Group(
            Text(title, style=Style(bold=True, color="#3344cc")),
            Text(f"{authors} · {year}"),
            Text(f"DOI: {citations_for_paper[0].doi or 'None'}"),
            Text(f"Citation chain position: {self.active_citation_index + 1}/{len(self.citation_chain)}" if self.citation_chain else ""),
            Text("\nSupporting evidence (click a Claim below to underline answer parts):", style=Style(bold=True)),
            *evidence_panels,
        )
        self.query_one("#chunk-detail-content", Static).update(Panel(content, border_style="#e0e0e0"))

    def show_claim_detail(self, claim_id: int) -> None:
        if not self.structured_answer:
            return

        if self.active_reference_paper_id is None:
            return

        self.active_claim_id = claim_id
        self.render_chat()

        paper_id = self.active_reference_paper_id
        citation_ids = []
        claim = next((c for c in self.structured_answer.claims if c.claim_id == claim_id), None)
        if claim:
            citation_ids = claim.citation_ids

        claim_style = Style(color=self.claim_color_map.get(claim_id, "#4a5568"), bold=True)

        citations_for_claim: list[PaperCitation] = []
        for cid in citation_ids:
            if 1 <= cid <= len(self.citations):
                pc = self.citations[cid - 1]
                if pc.paper_id == paper_id:
                    citations_for_claim.append(pc)

        # Store the evidence chunk we consider "active" for this claim.
        # We use it for the Open PDF at page action.
        self.active_evidence = None
        if citations_for_claim:
            self.active_evidence = max(citations_for_claim, key=lambda x: float(x.score or 0.0))

        title = citations_for_claim[0].paper_title if citations_for_claim else ""
        authors = citations_for_claim[0].authors if citations_for_claim else ""
        year = citations_for_claim[0].year if citations_for_claim else ""

        panels: list = []
        for c in sorted(citations_for_claim, key=lambda x: x.page):
            sentences = self.split_sentences(c.quote)
            sent_bullets = "\n".join([f"- {s}" for s in sentences[:25]]) or "- (no sentences found)"
            panels.append(
                Panel(
                    Group(
                        Text(f"Claim {claim_id}", style=claim_style),
                        Text(claim.text if claim else ""),
                        Text(f"\np. {c.page} · {c.section} · score {c.score:.3f}", style=Style(bold=True)),
                        Text("\nSentences (from the exact evidence chunk):", style=Style(bold=True)),
                        Text(sent_bullets),
                        Text("\nEvidence chunk (truncated):", style=Style(bold=True)),
                        Text(self.trim(c.quote, 1600)),
                    ),
                    border_style=self.claim_color_map.get(claim_id, "#e0e0e0"),
                )
            )

        if not panels:
            self.query_one("#chunk-detail-content", Static).update(f"No evidence chunks found for Claim {claim_id} in this paper.")
            return

        content = Group(
            Text(title, style=Style(bold=True, color="#3344cc")),
            Text(f"{authors} · {year}"),
            *panels,
        )
        self.query_one("#chunk-detail-content", Static).update(Panel(content, border_style="#e0e0e0"))

    def show_paper_detail(self, paper: PaperDocument) -> None:
        # Bottom-right list: project introspection
        sections = []
        seen = set()
        for chunk in paper.chunks:
            if chunk.section not in seen:
                seen.add(chunk.section)
                sections.append(chunk.section)
            if len(sections) >= 10:
                break

        detail = Group(
            Text(paper.title, style=Style(bold=True, color="#3344cc")),
            Text(f"{paper.authors} · {paper.year}"),
            Text(f"Pages: {paper.page_count}"),
            Text(f"Path: {paper.file_path}"),
            Text("Press l to look up this paper's lineage and save it in papers/."),
            Text("\nSections found:"),
            Text("\n".join([f"- {s}" for s in sections]) or "- (none)"),
        )
        self.active_reference_paper_id = None
        self.active_paper = paper
        self.active_claim_id = None
        self.query_one("#claim-list", ListView).clear()
        self.claim_items = {}
        self.query_one("#chunk-detail-content", Static).update(Panel(detail, border_style="#e0e0e0"))

    def show_lineage_loading(self, paper: PaperDocument) -> None:
        self.active_lineage_items = []
        detail = Group(
            Text("Searching Paper Lineage", style=Style(bold=True, color="#3344cc")),
            Text(paper.title, style=Style(bold=True, color="#3344cc")),
            Text(f"{paper.authors} · {paper.year}"),
            Text("\nLineage lookup is running. This can take a little while because it searches prior, citing, and related papers."),
            Text("\nWhen it finishes, the lineage report will appear here and be saved under papers/lineage-*.json."),
        )
        self.query_one("#claim-list", ListView).clear()
        self.claim_items = {}
        self.query_one("#chunk-detail-content", Static).update(Panel(detail, title="Loading", border_style="#3344cc"))

    def show_lineage_detail(self, lineage: dict) -> None:
        source = lineage.get("source_paper", {})
        results = lineage.get("results", {})
        self.active_lineage_items = self.flatten_lineage_items(results)
        source_title = str(source.get("title") or "Selected paper")
        source_meta = f"{source.get('authors', 'Unknown')} · {source.get('year', 'n.d.')}"
        chart = self.build_lineage_flowchart(source_title, results)

        flow = Tree(Text("Paper Lineage", style=Style(color="#3344cc", bold=True)), guide_style="#3344cc")
        prior = flow.add(Text("Prior work feeding into this paper", style=Style(color="#7c3aed", bold=True)))
        self.add_lineage_nodes(prior, results.get("prior_work", []), "No prior-work results.")

        source_node = flow.add(Text(f"CURRENT PAPER: {self.trim(source_title, 90)}", style=Style(color="#3344cc", bold=True)))
        source_node.add(Text(source_meta, style=Style(color="#4a5568")))
        if source.get("doi"):
            source_node.add(Text(f"DOI: {source.get('doi')}", style=Style(color="#4a5568")))

        citing = source_node.add(Text("Cited by / descendants", style=Style(color="#059669", bold=True)))
        self.add_lineage_nodes(citing, results.get("citing_work", []), "No citing-work results.")

        related = source_node.add(Text("Related sibling papers", style=Style(color="#0891b2", bold=True)))
        self.add_lineage_nodes(related, results.get("related_work", []), "No related-work results.")

        content = Group(
            Panel(Text(chart, style=Style(color="#1a1a2e")), title="Lineage Flowchart", border_style="#3344cc"),
            flow,
            Text("Press d to download the top lineage paper into papers/ and reindex it.", style=Style(color="#3344cc", bold=True)),
            Text(f"Saved: {lineage.get('lineage_file', '')}"),
        )
        self.query_one("#claim-list", ListView).clear()
        self.claim_items = {}
        self.query_one("#chunk-detail-content", Static).update(Panel(content, border_style="#3344cc"))

    def show_download_loading(self, title: str) -> None:
        detail = Group(
            Text("Downloading Lineage Paper", style=Style(bold=True, color="#3344cc")),
            Text(self.trim(title, 90), style=Style(bold=True, color="#3344cc")),
            Text("\nSaving PDF into papers/ and forcing a reindex so it appears in Papers In Project."),
        )
        self.query_one("#chunk-detail-content", Static).update(Panel(detail, title="Downloading", border_style="#3344cc"))

    def show_download_complete(self, path: Path, source_title: str) -> None:
        detail = Group(
            Text("New Paper Added", style=Style(bold=True, color="#059669")),
            Text(path.name, style=Style(bold=True, color="#3344cc")),
            Text(f"Downloaded from: {source_title}"),
            Text(f"Path: {path}"),
            Text("\nThe paper list has been reindexed. Ask a new question to search across the expanded paper set."),
        )
        self.query_one("#chunk-detail-content", Static).update(Panel(detail, border_style="#059669"))

    def flatten_lineage_items(self, results: dict) -> list[dict]:
        items = []
        seen = set()
        for group in ["citing_work", "related_work", "prior_work"]:
            for item in results.get(group, []):
                url = item.get("url")
                if url and url not in seen:
                    seen.add(url)
                    items.append(item)
        return items

    def build_lineage_flowchart(self, source_title: str, results: dict) -> str:
        prior = self.lineage_titles(results.get("prior_work", []), 3)
        citing = self.lineage_titles(results.get("citing_work", []), 3)
        related = self.lineage_titles(results.get("related_work", []), 3)
        current = self.trim(source_title, 54)

        lines = [
            "PRIOR WORK",
            *[f"  [{idx}] {title}" for idx, title in enumerate(prior, start=1)],
            "       \\",
            "        \\",
            f"         ==> [ CURRENT PAPER ] {current}",
            "        /                 \\",
            "       /                   \\",
            "CITED BY / DESCENDANTS     RELATED SIBLINGS",
        ]
        max_rows = max(len(citing), len(related), 1)
        for idx in range(max_rows):
            left = f"[{idx + 1}] {citing[idx]}" if idx < len(citing) else ""
            right = f"[{idx + 1}] {related[idx]}" if idx < len(related) else ""
            lines.append(f"  {left:<34} {right}")
        return "\n".join(lines)

    def lineage_titles(self, items: list, limit: int) -> list[str]:
        titles = [self.trim(str(item.get("title") or "Untitled"), 32) for item in items[:limit]]
        return titles or ["No results"]

    def add_lineage_nodes(self, tree: Tree, items: list, empty_message: str) -> None:
        if not items:
            tree.add(Text(empty_message, style=Style(color="#4a5568")))
            return
        for idx, item in enumerate(items, start=1):
            item_title = self.trim(str(item.get("title") or "Untitled"), 90)
            date = str(item.get("published_date") or "n.d.")
            url = str(item.get("url") or "")
            snippet = self.trim(str(item.get("snippet") or ""), 220)
            node = tree.add(Text(f"{idx}. {item_title}", style=Style(bold=True)))
            node.add(Text(date, style=Style(color="#4a5568")))
            if url:
                node.add(Text(url, style=Style(color="blue", underline=True)))
            if snippet:
                node.add(Text(snippet, style=Style(color="#4a5568")))

    def split_sentences(self, text: str) -> list[str]:
        text = (text or "").strip()
        # Preserve newlines for fallback splitting while normalizing repeated spaces.
        text = re.sub(r"[ \t]+", " ", text)
        if not text:
            return []
        out: list[str] = []

        # Prefer punctuation-based splits.
        parts = re.split(r"(?<=[.!?])\s+", text)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p.split()) < 3:
                continue
            out.append(p)

        # Fallback: if PDF text doesn't contain usable punctuation, split on sentence-like newlines.
        if not out or len(out) < 3:
            # Use original (non-condensed) separators as much as possible.
            lines = re.split(r"[\n\r]+", text)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if len(line.split()) < 3:
                    continue
                out.append(line)

        return out

    def trim(self, text: str, length: int) -> str:
        if not text:
            return ""
        return text if len(text) <= length else text[: length - 1].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal UI for Local Paper QA")
    parser.add_argument("--papers-dir", default="papers", help="Directory containing PDF papers")
    args = parser.parse_args()
    PaperQATui(papers_dir=args.papers_dir).run()


if __name__ == "__main__":
    main()
