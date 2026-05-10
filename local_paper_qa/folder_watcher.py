"""Automatic folder watcher that triggers reindexing when PDFs change.

Usage as a module::

    from local_paper_qa.folder_watcher import FolderWatcher

    watcher = FolderWatcher("papers")
    watcher.start()
    # ... your app runs ...
    watcher.stop()

Or from the CLI::

    python -m local_paper_qa.folder_watcher --papers-dir papers
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:  # pragma: no cover
    Observer = None  # type: ignore
    FileSystemEventHandler = object  # type: ignore

logger = logging.getLogger(__name__)


class PDFChangeHandler(FileSystemEventHandler):
    """Handles file system events in the papers directory."""

    def __init__(self, on_change: Callable[[], None], debounce_seconds: float = 2.0):
        self.on_change = on_change
        self.debounce_seconds = debounce_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return
        self._schedule_reindex()

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return
        self._schedule_reindex()

    def on_deleted(self, event):
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return
        self._schedule_reindex()

    def _schedule_reindex(self):
        """Debounce reindexing to avoid triggering multiple times."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self.on_change)
            self._timer.daemon = True
            self._timer.start()


class FolderWatcher:
    """Watches a directory for PDF file changes and triggers a callback."""

    def __init__(
        self,
        papers_dir: str | Path,
        on_change: Callable[[], None],
        debounce_seconds: float = 2.0,
    ):
        if Observer is None:  # pragma: no cover
            raise RuntimeError("watchdog is not installed. Run: pip install watchdog")
        self.papers_dir = Path(papers_dir).expanduser().resolve()
        self.on_change = on_change
        self.debounce_seconds = debounce_seconds
        self.observer = Observer()
        self.handler = PDFChangeHandler(self.on_change, self.debounce_seconds)

    def start(self):
        """Start watching the directory."""
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.observer.schedule(self.handler, str(self.papers_dir), recursive=False)
        self.observer.start()
        logger.info("Watching %s for PDF changes", self.papers_dir)

    def stop(self):
        """Stop watching the directory."""
        self.observer.stop()
        self.observer.join(timeout=5)
        logger.info("Stopped watching %s", self.papers_dir)


def main():
    parser = argparse.ArgumentParser(description="Watch papers folder for changes")
    parser.add_argument("--papers-dir", default="papers", help="Directory containing PDFs")
    parser.add_argument("--debounce", type=float, default=2.0, help="Debounce seconds")
    args = parser.parse_args()

    from local_paper_qa.service import LocalPaperQA

    qa = LocalPaperQA(args.papers_dir)

    def on_change():
        print("\n[FolderWatcher] PDF changed. Reindexing...", file=sys.stderr)
        try:
            papers = qa.ensure_index(force=True)
            print(f"[FolderWatcher] Indexed {len(papers)} papers.", file=sys.stderr)
        except Exception as e:
            print(f"[FolderWatcher] Reindex failed: {e}", file=sys.stderr)

    watcher = FolderWatcher(args.papers_dir, on_change, debounce_seconds=args.debounce)

    def signal_handler(sig, frame):
        print("\n[FolderWatcher] Stopping...", file=sys.stderr)
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    watcher.start()
    print(f"[FolderWatcher] Watching {args.papers_dir} (debounce={args.debounce_seconds}s). Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()


if __name__ == "__main__":
    main()
