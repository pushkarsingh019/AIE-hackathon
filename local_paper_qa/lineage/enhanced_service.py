import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from ..academic.base import AcademicPaper, LineageResult
from ..academic.manager import AcademicAPIManager, APIClientType


@dataclass
class EnhancedPaperDocument:
    """Enhanced paper document with academic API metadata."""
    paper_id: str
    file_path: str
    title: str
    authors: str
    year: str
    venue: str = ""
    doi: str = ""
    abstract: str = ""
    page_count: int = 0
    chunks: list = None
    # Enhanced fields
    semantic_scholar_data: Optional[Dict] = None
    crossref_data: Optional[Dict] = None
    arxiv_data: Optional[Dict] = None
    citation_count: int = 0
    reference_count: int = 0
    last_updated: str = ""
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class LineageNode:
    """Node in the lineage graph."""
    paper: AcademicPaper
    node_type: str  # "source", "citing", "cited", "related", "methodological", "temporal"
    relationship_strength: float = 0.0
    children: List['LineageNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class EnhancedLineageService:
    """Enhanced paper lineage service with multiple API integration."""
    
    def __init__(self, papers_dir: str = "papers"):
        self.papers_dir = Path(papers_dir).expanduser().resolve()
        self.lineage_dir = self.papers_dir / ".enhanced_lineage"
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        
        # Academic API manager
        self.api_manager = AcademicAPIManager()
        
        # Initialize available APIs (will be configured later)
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize available academic APIs."""
        # Check for API keys in environment or config files
        import os
        
        # Semantic Scholar
        ss_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_KEY")
        # Semantic Scholar supports unauthenticated requests (rate-limited), so always enable it.
        self.api_manager.add_client(APIClientType.SEMANTIC_SCHOLAR, ss_key)
        
        # Crossref (doesn't require API key, but can use one for higher limits)
        crossref_key = os.environ.get("CROSSREF_API_KEY") or os.environ.get("CROSSREF_KEY")
        self.api_manager.add_client(APIClientType.CROSSREF, crossref_key)
        
        # arXiv (doesn't require API key)
        self.api_manager.add_client(APIClientType.ARXIV)
    
    def get_enhanced_paper_lineage(self, paper: EnhancedPaperDocument, limit: int = 10) -> Dict[str, Any]:
        """Get enhanced lineage information for a paper."""
        try:
            # Use existing DOI if available, otherwise search by title
            if paper.doi:
                source_paper = self.api_manager.search_paper_by_doi(paper.doi)
            else:
                source_paper = self.api_manager.search_paper_by_title(paper.title, 1)
                if source_paper:
                    source_paper = source_paper[0]
            
            if not source_paper:
                # Create minimal source paper from document info
                source_paper = AcademicPaper(
                    title=paper.title,
                    authors=paper.authors.split(", ") if paper.authors else [],
                    year=int(paper.year) if paper.year.isdigit() else None,
                    doi=paper.doi,
                    confidence_score=0.5,
                    source="document"
                )
            
            # Get enhanced lineage from all APIs
            lineage_result = self.api_manager.get_enhanced_lineage(
                paper.title, 
                paper.authors, 
                limit
            )
            
            # Build lineage graph
            lineage_graph = self._build_lineage_graph(source_paper, lineage_result)
            
            # Generate lineage visualization data
            visualization_data = self._generate_visualization_data(lineage_graph)
            
            # Save lineage report
            lineage_report = {
                "source_paper": asdict(source_paper),
                "lineage_graph": asdict(lineage_graph),
                "visualization_data": visualization_data,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "api_sources": lineage_result.search_metadata.get("api_sources_used", []),
                    "confidence_score": lineage_result.confidence_score,
                    "total_papers_found": lineage_result.search_metadata.get("source_papers_found", 0)
                }
            }
            
            # Save to file
            lineage_path = self._get_lineage_path(paper)
            lineage_path.write_text(json.dumps(lineage_report, ensure_ascii=False, indent=2))
            
            return {
                "lineage_report": lineage_report,
                "lineage_file": str(lineage_path),
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "lineage_file": None
            }
    
    def _build_lineage_graph(self, source_paper: AcademicPaper, lineage_result: LineageResult) -> LineageNode:
        """Build a lineage graph from the results."""
        # Create source node
        source_node = LineageNode(
            paper=source_paper,
            node_type="source",
            relationship_strength=1.0
        )
        
        # Add citing papers
        for citing_paper in lineage_result.citing_papers:
            citing_node = LineageNode(
                paper=citing_paper,
                node_type="citing",
                relationship_strength=citing_paper.confidence_score
            )
            source_node.children.append(citing_node)
        
        # Add cited papers
        for cited_paper in lineage_result.cited_by_papers:
            cited_node = LineageNode(
                paper=cited_paper,
                node_type="cited",
                relationship_strength=cited_paper.confidence_score
            )
            source_node.children.append(cited_node)
        
        # Add related papers
        for related_paper in lineage_result.related_papers:
            related_node = LineageNode(
                paper=related_paper,
                node_type="related",
                relationship_strength=related_paper.confidence_score * 0.8
            )
            source_node.children.append(related_node)
        
        # Add methodological papers
        for method_paper in lineage_result.methodological_papers:
            method_node = LineageNode(
                paper=method_paper,
                node_type="methodological",
                relationship_strength=method_paper.confidence_score * 0.7
            )
            source_node.children.append(method_node)
        
        # Add temporal papers
        for temporal_paper in lineage_result.temporal_papers:
            temporal_node = LineageNode(
                paper=temporal_paper,
                node_type="temporal",
                relationship_strength=temporal_paper.confidence_score * 0.6
            )
            source_node.children.append(temporal_node)
        
        return source_node
    
    def _generate_visualization_data(self, lineage_graph: LineageNode) -> Dict[str, Any]:
        """Generate data for lineage visualization."""
        nodes = []
        edges = []
        
        def add_node(node: LineageNode, parent_id: Optional[str] = None):
            node_id = f"node_{len(nodes)}"
            
            # Add node
            nodes.append({
                "id": node_id,
                "title": node.paper.title,
                "authors": ", ".join(node.paper.authors[:3]) + ("..." if len(node.paper.authors) > 3 else ""),
                "year": node.paper.year,
                "doi": node.paper.doi,
                "node_type": node.node_type,
                "confidence": node.paper.confidence_score,
                "citation_count": node.paper.citations_count or 0,
                "url": node.paper.url
            })
            
            # Add edge if there's a parent
            if parent_id:
                edges.append({
                    "from": parent_id,
                    "to": node_id,
                    "relationship": node.node_type,
                    "strength": node.relationship_strength
                })
            
            # Add children
            for child in node.children:
                add_node(child, node_id)
        
        add_node(lineage_graph)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": {
                    node_type: len([n for n in nodes if n["node_type"] == node_type])
                    for node_type in set(n["node_type"] for n in nodes)
                }
            }
        }
    
    def _get_lineage_path(self, paper: EnhancedPaperDocument) -> Path:
        """Get the path for storing lineage data."""
        slug = self._slugify(paper.title)
        return self.lineage_dir / f"enhanced-lineage-{slug}-{int(time.time())}.json"
    
    def _slugify(self, text: str) -> str:
        """Create a URL-friendly slug from text."""
        import re
        return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:80]
    
    def find_lineage_papers_by_type(self, lineage_path: str, paper_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find papers of a specific type in a lineage report."""
        try:
            lineage_file = Path(lineage_path)
            if not lineage_file.exists():
                return []
            
            with open(lineage_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            viz_data = report.get('visualization_data', {})
            nodes = viz_data.get('nodes', [])
            
            # Filter by node type
            typed_nodes = [node for node in nodes if node.get('node_type') == paper_type]
            
            # Sort by confidence score and return top results
            typed_nodes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return typed_nodes[:limit]
            
        except Exception as e:
            print(f"Error finding lineage papers: {e}")
            return []
    
    def get_lineage_summary(self, lineage_path: str) -> Dict[str, Any]:
        """Get a summary of a lineage report."""
        try:
            lineage_file = Path(lineage_path)
            if not lineage_file.exists():
                return {}
            
            with open(lineage_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            metadata = report.get('metadata', {})
            viz_data = report.get('visualization_data', {})
            stats = viz_data.get('stats', {})
            
            return {
                "generated_at": metadata.get('generated_at'),
                "confidence_score": metadata.get('confidence_score', 0),
                "total_papers": stats.get('total_nodes', 0),
                "api_sources": metadata.get('api_sources', []),
                "node_types": stats.get('node_types', {}),
                "source_paper_title": report.get('source_paper', {}).get('title', '')
            }
            
        except Exception as e:
            print(f"Error getting lineage summary: {e}")
            return {}
    
    def download_enhanced_lineage_paper(self, paper: AcademicPaper) -> Optional[Path]:
        """Download a paper from lineage results."""
        try:
            if not paper.pdf_url:
                return None
            
            import httpx
            
            # Try to download the paper
            response = httpx.get(paper.pdf_url, timeout=60, follow_redirects=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not response.content.startswith(b'%PDF'):
                return None
            
            # Save the paper
            slug = self._slugify(paper.title)
            download_path = self.papers_dir / f"{slug}.pdf"
            
            # Handle filename conflicts
            counter = 2
            while download_path.exists():
                download_path = self.papers_dir / f"{slug}-{counter}.pdf"
                counter += 1
            
            download_path.write_bytes(response.content)
            return download_path
            
        except Exception as e:
            print(f"Error downloading lineage paper: {e}")
            return None
