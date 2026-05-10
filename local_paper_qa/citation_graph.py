"""Citation graph builder that maps relationships between indexed papers.

Builds a directed graph where nodes are papers and edges represent shared
chunk similarity (i.e., two papers are cited together for the same query).

Usage::

    from local_paper_qa.citation_graph import CitationGraph

    graph = CitationGraph(papers_dir="papers")
    graph.build()
    graph.save("citation_graph.json")
    graph.visualize()
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from local_paper_qa.models import PaperDocument, PaperCitation


class CitationGraph:
    """Builds and stores a citation graph from indexed papers."""

    def __init__(self, papers_dir: str = "papers"):
        self.papers_dir = Path(papers_dir).expanduser().resolve()
        self.papers: list[PaperDocument] = []
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)

    def build(self) -> None:
        """Build the citation graph from the index."""
        from local_paper_qa.service import LocalPaperQA

        qa = LocalPaperQA(str(self.papers_dir))
        self.papers = qa.ensure_index()

        # Build co-citation matrix
        co_cite: Dict[Tuple[str, str], int] = defaultdict(int)
        citations = qa.retrieve("general query", papers=self.papers)
        
        # Group citations by paper
        by_paper: Dict[str, list[PaperCitation]] = defaultdict(list)
        for c in citations:
            by_paper[c.paper_id].append(c)

        # Co-citation: papers cited together get an edge
        paper_ids = list(by_paper.keys())
        for i in range(len(paper_ids)):
            for j in range(i + 1, len(paper_ids)):
                pi, pj = paper_ids[i], paper_ids[j]
                co_cite[(pi, pj)] += 1

        # Build nodes
        self.nodes = []
        for paper in self.papers:
            self.nodes.append({
                "id": paper.paper_id,
                "title": paper.title[:80],
                "authors": paper.authors[:60],
                "year": str(paper.year),
                "chunk_count": len(paper.chunks),
            })

        # Build edges (only co-citations with count >= 1)
        self.edges = []
        for (pi, pj), count in co_cite.items():
            edge = {
                "source": pi,
                "target": pj,
                "weight": count,
            }
            self.edges.append(edge)
            self.adjacency[pi].append(pj)
            self.adjacency[pj].append(pi)

    def get_neighbors(self, paper_id: str) -> List[str]:
        """Get papers that are co-cited with the given paper."""
        return list(self.adjacency.get(paper_id, []))

    def get_most_cited(self, limit: int = 5) -> List[dict]:
        """Get the most connected papers in the graph."""
        degrees = [(pid, len(neighbors)) for pid, neighbors in self.adjacency.items()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "paper_id": pid,
                "title": next((n["title"] for n in self.nodes if n["id"] == pid), ""),
                "degree": deg,
            }
            for pid, deg in degrees[:limit]
        ]

    def get_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find a simple path between two papers using BFS."""
        if source == target:
            return [source]
        visited: set[str] = set()
        queue: List[Tuple[str, List[str]]] = [(source, [source])]
        while queue:
            current, path = queue.pop(0)
            if current == target:
                return path
            visited.add(current)
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        return None

    def save(self, path: str | Path) -> None:
        """Save the graph to a JSON file."""
        data = {
            "nodes": self.nodes,
            "edges": self.edges,
            "adjacency": dict(self.adjacency),
            "stats": {
                "total_papers": len(self.nodes),
                "total_edges": len(self.edges),
                "avg_connections": sum(len(v) for v in self.adjacency.values()) / max(len(self.nodes), 1),
            },
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def visualize_text(self) -> str:
        """Generate a text-based visualization of the graph."""
        lines = [f"Citation Graph: {len(self.nodes)} papers, {len(self.edges)} connections"]
        lines.append("=" * 60)
        
        # Show most connected papers
        most_cited = self.get_most_cited(5)
        if most_cited:
            lines.append("\nMost Connected Papers:")
            for c in most_cited:
                lines.append(f"  - {c['title']} ({c['degree']} connections)")
        
        # Show adjacency
        lines.append("\nAdjacency List:")
        for node in self.nodes[:10]:
            neighbors = self.get_neighbors(node["id"])
            if neighbors:
                neighbor_titles = [
                    next((n["title"][:30] for n in self.nodes if n["id"] == pid), pid)
                    for pid in neighbors[:3]
                ]
                lines.append(f"  {node['title'][:40]} -> {', '.join(neighbor_titles)}")
        
        return "\n".join(lines)
