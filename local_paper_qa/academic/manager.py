import logging
import time
from typing import List, Optional, Dict, Any
from enum import Enum
from .base import AcademicAPIClient, AcademicPaper, LineageResult
from .semantic_scholar import SemanticScholarClient
from .crossref import CrossrefClient
from .arxiv import ArxivClient

logger = logging.getLogger(__name__)


class APIClientType(Enum):
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    ARXIV = "arxiv"
    EXA = "exa"  # Keep existing Exa integration


class AcademicAPIManager:
    """Manages multiple academic API clients with fallback strategy and confidence scoring."""
    
    def __init__(self):
        self.clients: Dict[APIClientType, AcademicAPIClient] = {}
        self.api_keys: Dict[APIClientType, str] = {}
        self.priority_order = [
            APIClientType.SEMANTIC_SCHOLAR,
            APIClientType.CROSSREF, 
            APIClientType.ARXIV
        ]
        self.last_request_time = {}
        self.rate_limits = {
            APIClientType.SEMANTIC_SCHOLAR: 100,  # per hour
            APIClientType.CROSSREF: 1000,        # per hour
            APIClientType.ARXIV: 3000,           # per hour
        }
    
    def add_client(self, client_type: APIClientType, api_key: Optional[str] = None):
        """Add an API client."""
        self.api_keys[client_type] = api_key or ""
        
        if client_type == APIClientType.SEMANTIC_SCHOLAR:
            self.clients[client_type] = SemanticScholarClient(api_key)
        elif client_type == APIClientType.CROSSREF:
            self.clients[client_type] = CrossrefClient(api_key)
        elif client_type == APIClientType.ARXIV:
            self.clients[client_type] = ArxivClient(api_key)
    
    def search_paper_by_title(self, title: str, limit: int = 5) -> List[AcademicPaper]:
        """Search for papers by title across all available APIs with fallback."""
        results = []
        seen_dois = set()
        
        for client_type in self.priority_order:
            if client_type not in self.clients:
                continue
                
            client = self.clients[client_type]
            
            try:
                # Check rate limit
                if self._check_rate_limit(client_type):
                    continue
                
                papers = client.search_paper_by_title(title, limit)
                
                # Filter out duplicates and add confidence scores
                for paper in papers:
                    if paper.doi and paper.doi in seen_dois:
                        continue
                    
                    # Adjust confidence based on API source
                    if client_type == APIClientType.SEMANTIC_SCHOLAR:
                        paper.confidence_score *= 1.0
                    elif client_type == APIClientType.CROSSREF:
                        paper.confidence_score *= 0.9
                    elif client_type == APIClientType.ARXIV:
                        paper.confidence_score *= 0.85
                    
                    if paper.doi:
                        seen_dois.add(paper.doi)
                    results.append(paper)
                
                # If we found enough results, stop
                if len(results) >= limit:
                    break
                    
            except Exception as e:
                logger.warning("Search by title failed with %s: %s", client_type.value, e)
                continue
        
        # Sort by confidence score and return top results
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        return results[:limit]
    
    def search_paper_by_doi(self, doi: str) -> Optional[AcademicPaper]:
        """Search for paper by DOI across all available APIs with fallback."""
        # Try each API in priority order
        for client_type in self.priority_order:
            if client_type not in self.clients:
                continue
                
            client = self.clients[client_type]
            
            try:
                # Check rate limit
                if self._check_rate_limit(client_type):
                    continue
                
                paper = client.search_paper_by_doi(doi)
                
                if paper:
                    # Adjust confidence based on API source
                    if client_type == APIClientType.SEMANTIC_SCHOLAR:
                        paper.confidence_score *= 1.0
                    elif client_type == APIClientType.CROSSREF:
                        paper.confidence_score *= 0.95  # DOI lookup is Crossref's strength
                    elif client_type == APIClientType.ARXIV:
                        paper.confidence_score *= 0.8
                    
                    return paper
                    
            except Exception as e:
                logger.warning("Search by DOI failed with %s: %s", client_type.value, e)
                continue
        
        return None
    
    def get_enhanced_lineage(self, paper_title: str, authors: str = "", limit: int = 10) -> LineageResult:
        """Get enhanced lineage information from multiple APIs."""
        # First, find the source paper
        source_paper = self._find_source_paper(paper_title, authors)
        if not source_paper:
            # Fallback to basic paper info
            source_paper = AcademicPaper(
                title=paper_title,
                authors=[authors] if authors else [],
                confidence_score=0.5,
                source="unknown"
            )
        
        # Collect results from all APIs
        citing_papers = []
        cited_by_papers = []
        related_papers = []
        methodological_papers = []
        temporal_papers = []
        
        for client_type in self.priority_order:
            if client_type not in self.clients:
                continue
                
            client = self.clients[client_type]
            
            try:
                # Check rate limit
                if self._check_rate_limit(client_type):
                    continue
                
                # Get citations (papers that cite this paper)
                if source_paper.doi or source_paper.title:
                    citations_limit = max(1, limit // 2)
                    citations = client.get_citations(source_paper.doi or source_paper.title, citations_limit)
                    citing_papers.extend(citations)
                
                # Get references (papers cited by this paper)
                if source_paper.doi or source_paper.title:
                    references_limit = max(1, limit // 2)
                    references = client.get_references(source_paper.doi or source_paper.title, references_limit)
                    cited_by_papers.extend(references)
                
                # Get similar papers
                if source_paper.doi or source_paper.title:
                    similar_limit = max(1, limit // 2)
                    similar = client.get_similar_papers(source_paper.doi or source_paper.title, similar_limit)
                    related_papers.extend(similar)
                
                # For methodological and temporal papers, we'll use simple heuristics
                if source_paper.title:
                    methodological_limit = max(1, limit // 3)
                    methodological = self._find_methodological_papers(client, source_paper.title, methodological_limit)
                    methodological_papers.extend(methodological)
                    
                    temporal_limit = max(1, limit // 3)
                    temporal = self._find_temporal_papers(client, source_paper.year, temporal_limit)
                    temporal_papers.extend(temporal)
                
            except Exception as e:
                logger.warning("Enhanced lineage lookup failed with %s: %s", client_type.value, e)
                continue
        
        # Remove duplicates and combine results
        all_papers = citing_papers + cited_by_papers + related_papers + methodological_papers + temporal_papers
        unique_papers = self._remove_duplicate_papers(all_papers)
        
        # Distribute papers to different lineage types
        final_citing = unique_papers[:limit]
        final_cited_by = unique_papers[limit:limit*2]
        final_related = unique_papers[limit*2:limit*3]
        final_methodological = unique_papers[limit*3:limit*4]
        final_temporal = unique_papers[limit*4:limit*5]
        
        # Calculate overall confidence score
        confidence = self._calculate_lineage_confidence([
            final_citing, final_cited_by, final_related, 
            final_methodological, final_temporal
        ])
        
        return LineageResult(
            source_paper=source_paper,
            citing_papers=final_citing,
            cited_by_papers=final_cited_by,
            related_papers=final_related,
            methodological_papers=final_methodological,
            temporal_papers=final_temporal,
            confidence_score=confidence,
            search_metadata={
                "source_papers_found": len(all_papers),
                "api_sources_used": [client_type.value for client_type in self.priority_order if client_type in self.clients],
                "search_timestamp": time.time()
            }
        )
    
    def _find_source_paper(self, title: str, authors: str = "") -> Optional[AcademicPaper]:
        """Find the source paper using multiple APIs."""
        # Try DOI-based search first if we can extract a DOI
        doi = self._extract_doi(title + " " + authors)
        if doi:
            paper = self.search_paper_by_doi(doi)
            if paper:
                return paper
        
        # Try title-based search
        papers = self.search_paper_by_title(title, 1)
        return papers[0] if papers else None
    
    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text."""
        import re
        doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
        match = re.search(doi_pattern, text, re.I)
        return match.group(0) if match else None
    
    def _remove_duplicate_papers(self, papers: List[AcademicPaper]) -> List[AcademicPaper]:
        """Remove duplicate papers based on DOI or title."""
        unique_papers = {}
        seen_titles = set()
        seen_dois = set()
        
        for paper in papers:
            # Use DOI as primary key if available
            if paper.doi and paper.doi not in seen_dois:
                unique_papers[paper.doi] = paper
                seen_dois.add(paper.doi)
            # Use title as secondary key
            elif paper.title.lower() not in seen_titles:
                unique_papers[paper.title.lower()] = paper
                seen_titles.add(paper.title.lower())
        
        return list(unique_papers.values())
    
    def _find_methodological_papers(self, client: AcademicAPIClient, title: str, limit: int) -> List[AcademicPaper]:
        """Find methodologically similar papers."""
        # Simple heuristic: search for papers with similar keywords
        method_keywords = ["method", "approach", "algorithm", "technique", "model", "framework"]
        query = " OR ".join([f'ti:"{kw}"' for kw in method_keywords[:3]])
        
        try:
            papers = client.search_papers_by_query(query, limit)
            return papers
        except Exception as exc:
            logger.debug("Methodological paper search failed: %s", exc)
            return []
    
    def _find_temporal_papers(self, client: AcademicAPIClient, year: Optional[int], limit: int) -> List[AcademicPaper]:
        """Find papers from the same time period."""
        if not year:
            return []
        
        # Search for papers from same year ± 2 years
        year_range = f"{year-2}..{year+2}"
        query = f'date:[{year_range}]'
        
        try:
            papers = client.search_papers_by_query(query, limit)
            return papers
        except Exception as exc:
            logger.debug("Temporal paper search failed: %s", exc)
            return []
    
    def _calculate_lineage_confidence(self, paper_lists: List[List[AcademicPaper]]) -> float:
        """Calculate overall confidence score for lineage results."""
        total_papers = sum(len(papers) for papers in paper_lists)
        avg_confidence = sum(
            sum(paper.confidence_score for paper in papers) 
            for papers in paper_lists
        ) / max(total_papers, 1)
        
        # Boost confidence if we have results from multiple APIs
        api_count = len([client_type for client_type in self.priority_order if client_type in self.clients])
        api_bonus = min(api_count * 0.1, 0.3)
        
        return min(avg_confidence + api_bonus, 1.0)
    
    def _check_rate_limit(self, client_type: APIClientType) -> bool:
        """Check if rate limit would be exceeded.

        Instead of skipping work, wait until we're allowed to call again.
        """
        current_time = time.time()
        last_time = self.last_request_time.get(client_type, 0)
        rate_limit = self.rate_limits.get(client_type, 1000)

        min_interval_seconds = 3600 / rate_limit
        elapsed = current_time - last_time
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)

        self.last_request_time[client_type] = time.time()
        return False
