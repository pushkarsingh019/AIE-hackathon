import json
import re
from typing import List, Optional, Dict, Any
from .base import AcademicAPIClient, AcademicPaper


class CrossrefClient(AcademicAPIClient):
    """Crossref API client for academic paper discovery."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.base_url = "https://api.crossref.org/works"
        self.rate_limit = 1000  # Crossref has generous rate limits
    
    def search_paper_by_title(self, title: str, limit: int = 5) -> List[AcademicPaper]:
        """Search for papers by title using Crossref."""
        try:
            # Use exact match for title search
            params = {
                "query": title,
                "rows": limit,
                "filter": "type:journal-article",
                "cursor": "*"
            }
            
            data = self._make_request(self.base_url, params=params)
            return self._parse_papers_from_response(data)
            
        except Exception as e:
            print(f"Crossref search error: {e}")
            return []
    
    def search_paper_by_doi(self, doi: str) -> Optional[AcademicPaper]:
        """Search for paper by DOI using Crossref."""
        try:
            # Construct DOI URL
            if not doi.startswith("10."):
                doi = f"10.{doi}"
            
            url = f"{self.base_url}/{doi}"
            data = self._make_request(url)

            # Crossref DOI lookup returns a single "message" object, not message.items.
            # For robustness, try items parsing first, then fall back to single-message parsing.
            papers = self._parse_papers_from_response(data)
            if papers:
                return papers[0]

            message = data.get("message") or {}
            paper = self._parse_paper_from_item(message)
            return paper
            
        except Exception as e:
            print(f"Crossref DOI search error: {e}")
            return None
    
    def get_citations(self, paper_id: str, limit: int = 50) -> List[AcademicPaper]:
        """Get papers that cite the given paper."""
        try:
            # Crossref doesn't have a direct citation search endpoint
            # Use "cited-by" parameter if available, otherwise search by title
            # For now, we'll search for papers with similar titles
            
            # Try to get the paper first to get its title
            paper = self.search_paper_by_doi(paper_id)
            if not paper:
                return []
            
            # Search for papers with similar titles
            params = {
                "query": paper.title[:100],  # Limit query length
                "rows": limit,
                "filter": "type:journal-article"
            }
            
            data = self._make_request(self.base_url, params=params)
            return self._parse_papers_from_response(data)
            
        except Exception as e:
            print(f"Crossref citations error: {e}")
            return []
    
    def get_references(self, paper_id: str, limit: int = 50) -> List[AcademicPaper]:
        """Get papers referenced by the given paper."""
        try:
            # Crossref may reference other papers in its references field
            # For now, we'll return empty list as Crossref doesn't easily support this
            return []
            
        except Exception as e:
            print(f"Crossref references error: {e}")
            return []
    
    def get_similar_papers(self, paper_id: str, limit: int = 10) -> List[AcademicPaper]:
        """Get papers similar to the given paper."""
        try:
            # Get the paper first
            paper = self.search_paper_by_doi(paper_id)
            if not paper:
                return []
            
            # Search for papers by similar title keywords
            title_words = re.findall(r'\b\w+\b', paper.title.lower())
            query = " ".join(title_words[:5])  # Use first 5 words
            
            params = {
                "query": query,
                "rows": limit,
                "filter": "type:journal-article"
            }
            
            data = self._make_request(self.base_url, params=params)
            return self._parse_papers_from_response(data)
            
        except Exception as e:
            print(f"Crossref similar papers error: {e}")
            return []
    
    def search_papers_by_query(self, query: str, limit: int = 10) -> List[AcademicPaper]:
        """Search papers by general query."""
        try:
            params = {
                "query": query,
                "rows": limit,
                "filter": "type:journal-article"
            }
            
            data = self._make_request(self.base_url, params=params)
            return self._parse_papers_from_response(data)
            
        except Exception as e:
            print(f"Crossref query search error: {e}")
            return []
    
    def _parse_papers_from_response(self, data: Dict[str, Any]) -> List[AcademicPaper]:
        """Parse papers from Crossref API response."""
        papers = []
        
        try:
            items = data.get("message", {}).get("items", [])
            
            for item in items:
                paper = self._parse_paper_from_item(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error parsing Crossref papers: {e}")
            return []
    
    def _parse_paper_from_item(self, item: Dict[str, Any]) -> Optional[AcademicPaper]:
        """Parse paper data from Crossref item."""
        try:
            # Extract title
            title = ""
            if isinstance(item.get("title"), list) and item["title"]:
                title = item["title"][0]
            
            if not title:
                return None
            
            # Extract authors
            authors = []
            if "author" in item:
                for author in item["author"]:
                    if isinstance(author, dict):
                        name = author.get("given", "") + " " + author.get("family", "")
                        name = name.strip()
                        if name:
                            authors.append(name)
            
            # Extract year
            year = None
            if "published-print" in item and "date-parts" in item["published-print"]:
                date_parts = item["published-print"]["date-parts"]
                if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                    year = date_parts[0][0]
            
            # Extract DOI
            doi = item.get("DOI")
            
            # Extract abstract
            abstract = item.get("abstract")
            
            # Extract venue
            venue = ""
            if "container-title" in item:
                venue = item["container-title"][0] if isinstance(item["container-title"], list) else item["container-title"]
            
            # Extract URL
            url = item.get("URL")
            
            # Extract PDF URL
            pdf_url = None
            if "link" in item:
                for link in item["link"]:
                    if link.get("content-type") == "application/pdf":
                        pdf_url = link.get("URL")
                        break
            
            # Extract citation count (if available)
            citations_count = None
            if "is-referenced-by-count" in item:
                citations_count = item["is-referenced-by-count"]
            
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
                confidence_score=0.8,  # Good confidence for Crossref
                source="Crossref"
            )
            
        except Exception as e:
            print(f"Error parsing Crossref paper: {e}")
            return None
