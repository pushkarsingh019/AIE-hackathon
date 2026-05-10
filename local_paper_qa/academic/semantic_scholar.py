import json
import re
from typing import List, Optional, Dict, Any
from .base import AcademicAPIClient, AcademicPaper


class SemanticScholarClient(AcademicAPIClient):
    """Semantic Scholar API client for academic paper discovery."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper"
        self.rate_limit = 100 if api_key else 1000  # Lower limit for unauthenticated users
    
    def search_paper_by_title(self, title: str, limit: int = 5) -> List[AcademicPaper]:
        """Search for papers by title using Semantic Scholar."""
        try:
            # Use the search endpoint
            search_url = f"{self.base_url}/search"
            params = {
                "query": title,
                "limit": limit,
                "fields": "title,authors,year,doi,abstract,venue,url,citationsCount,referencesCount,publicationDate"
            }
            
            data = self._make_request(search_url, params=params)
            papers = []
            
            for item in data.get("data", []):
                paper = self._parse_paper_from_search(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []
    
    def search_paper_by_doi(self, doi: str) -> Optional[AcademicPaper]:
        """Search for paper by DOI using Semantic Scholar."""
        try:
            paper_id = self._normalize_paper_id(doi)
            url = f"{self.base_url}/{paper_id}"
            params = {
                "fields": "paperId,externalIds,url,title,venue,year,abstract,citationCount,referenceCount,publicationDate,authors,openAccessPdf"
            }
            data = self._make_request(url, params=params)
            return self._parse_paper_from_graph(data)
        except Exception as e:
            print(f"Semantic Scholar DOI search error: {e}")
            return None
    
    def get_citations(self, paper_id: str, limit: int = 50) -> List[AcademicPaper]:
        """Get papers that cite the given paper."""
        try:
            # Use Semantic Scholar's citation search
            paper_id = self._normalize_paper_id(paper_id)
            url = f"{self.base_url}/{paper_id}/citations"
            params = {
                "limit": limit,
                "fields": "paperId,externalIds,url,title,venue,year,abstract,citationCount,referenceCount,publicationDate,authors"
            }
            
            data = self._make_request(url, params=params)
            papers = []
            
            for item in data.get("data", []):
                citing_paper = item.get("citingPaper") or {}
                paper = self._parse_paper_from_graph(citing_paper)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar citations error: {e}")
            return []
    
    def get_references(self, paper_id: str, limit: int = 50) -> List[AcademicPaper]:
        """Get papers referenced by the given paper."""
        try:
            paper_id = self._normalize_paper_id(paper_id)
            url = f"{self.base_url}/{paper_id}/references"
            params = {
                "limit": limit,
                "fields": "paperId,externalIds,url,title,venue,year,abstract,citationCount,referenceCount,publicationDate,authors"
            }
            
            data = self._make_request(url, params=params)
            papers = []
            
            for item in data.get("data", []):
                cited_paper = item.get("citedPaper") or {}
                paper = self._parse_paper_from_graph(cited_paper)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar references error: {e}")
            return []
    
    def get_similar_papers(self, paper_id: str, limit: int = 10) -> List[AcademicPaper]:
        """Get papers similar to the given paper."""
        try:
            # Semantic Scholar doesn't have a direct similar papers endpoint
            # Use citations and references to find related papers
            half = max(1, limit // 2)
            citing_papers = self.get_citations(paper_id, half)
            reference_papers = self.get_references(paper_id, half)
            
            # Combine and remove duplicates
            all_papers = citing_papers + reference_papers
            unique_papers = {}
            
            for paper in all_papers:
                if paper.doi:
                    unique_papers[paper.doi] = paper
                else:
                    # Use title as fallback key for papers without DOI
                    unique_papers[paper.title] = paper
            
            return list(unique_papers.values())[:limit]
            
        except Exception as e:
            print(f"Semantic Scholar similar papers error: {e}")
            return []
    
    def search_papers_by_query(self, query: str, limit: int = 10) -> List[AcademicPaper]:
        """Search papers by general query."""
        try:
            search_url = f"{self.base_url}/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,authors,year,doi,abstract,venue,url,citationsCount,referencesCount,publicationDate"
            }
            
            data = self._make_request(search_url, params=params)
            papers = []
            
            for item in data.get("data", []):
                paper = self._parse_paper_from_search(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar query search error: {e}")
            return []
    
    def _parse_paper_from_search(self, data: Dict[str, Any]) -> Optional[AcademicPaper]:
        """Parse paper data from search results."""
        try:
            title = data.get("title", "")
            if not title:
                return None
            
            authors = self._normalize_authors(data.get("authors", []))
            year = self._extract_year_from_date(data.get("publicationDate"))
            doi = data.get("doi")
            abstract = data.get("abstract")
            venue = data.get("venue")
            url = data.get("url")
            citations_count = data.get("citationsCount")
            references_count = data.get("referencesCount")
            
            # Try to find PDF URL
            pdf_url = None
            if url and "semanticscholar.org" in url:
                pdf_url = url.replace("/paper/", "/pdf/")
            
            return AcademicPaper(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                abstract=abstract,
                venue=venue,
                url=url,
                pdf_url=pdf_url,
                citations_count=citations_count,
                references_count=references_count,
                confidence_score=0.9,  # High confidence for Semantic Scholar
                source="Semantic Scholar"
            )
            
        except Exception as e:
            print(f"Error parsing Semantic Scholar paper: {e}")
            return None

    def _normalize_paper_id(self, paper_id: str) -> str:
        """Normalize DOI into the Semantic Scholar Graph paperId format."""
        if not paper_id:
            return paper_id
        if paper_id.startswith("DOI:"):
            return paper_id
        if re.match(r"^10\.\d{4,9}/", paper_id, re.I):
            return f"DOI:{paper_id}"
        return paper_id

    def _parse_paper_from_graph(self, data: Dict[str, Any]) -> Optional[AcademicPaper]:
        """Parse paper fields from Semantic Scholar Graph responses."""
        try:
            title = data.get("title") or ""
            if not title:
                return None

            authors = self._normalize_authors(data.get("authors", []))

            year = data.get("year")
            if year is None:
                year = self._extract_year_from_date(data.get("publicationDate"))

            external_ids = data.get("externalIds") or {}
            doi = external_ids.get("DOI")

            abstract = data.get("abstract")
            venue = data.get("venue")
            if isinstance(venue, dict):
                venue = venue.get("name") or venue.get("text") or ""

            url = data.get("url")

            pdf_url = None
            open_access_pdf = data.get("openAccessPdf") or {}
            if isinstance(open_access_pdf, dict):
                pdf_url = open_access_pdf.get("url") or open_access_pdf.get("pdfUrl")

            citations_count = data.get("citationCount")
            references_count = data.get("referenceCount")

            return AcademicPaper(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                abstract=abstract,
                venue=str(venue) if venue else "",
                url=url,
                pdf_url=pdf_url,
                citations_count=citations_count,
                references_count=references_count,
                confidence_score=0.9,
                source="Semantic Scholar",
            )
        except Exception as e:
            print(f"Error parsing Semantic Scholar paper: {e}")
            return None
    
    def _parse_citation_from_search(self, data: Dict[str, Any]) -> Optional[AcademicPaper]:
        """Parse citation data from search results."""
        return self._parse_paper_from_search(data.get("paper", {}))
    
    def _parse_reference_from_search(self, data: Dict[str, Any]) -> Optional[AcademicPaper]:
        """Parse reference data from search results."""
        return self._parse_paper_from_search(data.get("paper", {}))
