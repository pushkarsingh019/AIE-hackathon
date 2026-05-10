from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class AcademicPaper:
    """Standardized paper information from academic APIs."""
    title: str
    authors: List[str]
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citations_count: Optional[int] = None
    references_count: Optional[int] = None
    citation_ids: List[str] = None
    reference_ids: List[str] = None
    confidence_score: float = 0.0
    source: str = ""  # API source name
    published_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.citation_ids is None:
            self.citation_ids = []
        if self.reference_ids is None:
            self.reference_ids = []


@dataclass
class LineageResult:
    """Enhanced lineage result with multiple types of relationships."""
    source_paper: AcademicPaper
    citing_papers: List[AcademicPaper]  # Papers that cite the source
    cited_by_papers: List[AcademicPaper]  # Papers cited by the source  
    related_papers: List[AcademicPaper]  # Papers on similar topics
    methodological_papers: List[AcademicPaper]  # Papers with similar methods
    temporal_papers: List[AcademicPaper]  # Papers from same time period
    confidence_score: float = 0.0
    search_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.search_metadata is None:
            self.search_metadata = {}


class AcademicAPIClient(ABC):
    """Abstract base class for academic API clients."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = ""
        self.rate_limit = 1000  # requests per hour
        self.requests_made = 0
        self.last_request_time = None
    
    @abstractmethod
    def search_paper_by_title(self, title: str, limit: int = 5) -> List[AcademicPaper]:
        """Search for papers by title."""
        pass
    
    @abstractmethod
    def search_paper_by_doi(self, doi: str) -> Optional[AcademicPaper]:
        """Search for paper by DOI."""
        pass
    
    @abstractmethod
    def get_citations(self, paper_id: str, limit: int = 50) -> List[AcademicPaper]:
        """Get papers that cite the given paper."""
        pass
    
    @abstractmethod
    def get_references(self, paper_id: str, limit: int = 50) -> List[AcademicPaper]:
        """Get papers referenced by the given paper."""
        pass
    
    @abstractmethod
    def get_similar_papers(self, paper_id: str, limit: int = 10) -> List[AcademicPaper]:
        """Get papers similar to the given paper."""
        pass
    
    @abstractmethod
    def search_papers_by_query(self, query: str, limit: int = 10) -> List[AcademicPaper]:
        """Search papers by general query."""
        pass
    
    def _make_request(self, url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and error handling."""
        import httpx
        import time
        
        # Simple rate limiting
        if self.requests_made >= self.rate_limit:
            time.sleep(3600 / self.rate_limit)  # Wait based on rate limit
        
        self.requests_made += 1
        self.last_request_time = time.time()
        
        default_headers = {"Accept": "application/json"}
        if self.api_key:
            default_headers["Authorization"] = f"Bearer {self.api_key}"
        
        if headers:
            default_headers.update(headers)
        
        response = httpx.get(url, params=params, headers=default_headers, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def _extract_year_from_date(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from various date formats."""
        if not date_str:
            return None
        
        import re
        year_match = re.search(r"(19|20)\d{2}", date_str)
        return int(year_match.group(1)) if year_match else None
    
    def _normalize_authors(self, authors_data: Any) -> List[str]:
        """Normalize author data from different API formats."""
        authors = []
        
        if isinstance(authors_data, str):
            authors.append(authors_data)
        elif isinstance(authors_data, list):
            for author in authors_data:
                if isinstance(author, str):
                    authors.append(author)
                elif isinstance(author, dict):
                    name = author.get("name", "")
                    if name:
                        authors.append(name)
        
        return authors[:10]  # Limit to first 10 authors