import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any
from urllib.parse import quote, urljoin
from .base import AcademicAPIClient, AcademicPaper

logger = logging.getLogger(__name__)


class ArxivClient(AcademicAPIClient):
    """arXiv API client for preprint paper discovery."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit = 3000  # arXiv has generous rate limits
    
    def search_paper_by_title(self, title: str, limit: int = 5) -> List[AcademicPaper]:
        """Search for papers by title using arXiv."""
        try:
            # Use exact match for title search
            query = f'all:"{title}"'
            params = {
                "search_query": query,
                "start": 0,
                "max_results": limit
            }
            
            data = self._make_request(self.base_url, params=params)
            return self._parse_papers_from_xml(data)
            
        except Exception as e:
            logger.warning("arXiv title search failed: %s", e)
            return []
    
    def search_paper_by_doi(self, doi: str) -> Optional[AcademicPaper]:
        """Search for paper by DOI using arXiv."""
        try:
            # arXiv papers may have DOIs in their metadata
            query = f'doi:"{doi}"'
            params = {
                "search_query": query,
                "start": 0,
                "max_results": 1
            }
            
            data = self._make_request(self.base_url, params=params)
            papers = self._parse_papers_from_xml(data)
            
            return papers[0] if papers else None
            
        except Exception as e:
            logger.warning("arXiv DOI search failed: %s", e)
            return None
    
    def get_citations(self, paper_id: str, limit: int = 50) -> List[AcademicPaper]:
        """Get papers that cite the given paper."""
        try:
            # arXiv doesn't have a direct citation search endpoint
            # For now, we'll search for papers with similar titles
            # Extract arXiv ID from paper_id
            arxiv_id = self._extract_arxiv_id(paper_id)
            if not arxiv_id:
                return []
            
            # Get the paper details first
            paper = self.get_paper_by_id(arxiv_id)
            if not paper:
                return []
            
            # Search for papers with similar titles
            title_words = re.findall(r'\b\w+\b', paper.title.lower())
            query = " OR ".join([f'ti:"{word}"' for word in title_words[:3]])
            
            params = {
                "search_query": query,
                "start": 0,
                "max_results": limit
            }
            
            data = self._make_request(self.base_url, params=params)
            return self._parse_papers_from_xml(data)
            
        except Exception as e:
            logger.warning("arXiv citations lookup failed: %s", e)
            return []
    
    def get_references(self, paper_id: str, limit: int = 50) -> List[AcademicPaper]:
        """Get papers referenced by the given paper."""
        try:
            # arXiv doesn't easily support reference tracking
            # Return empty list for now
            return []
            
        except Exception as e:
            logger.warning("arXiv references lookup failed: %s", e)
            return []
    
    def get_similar_papers(self, paper_id: str, limit: int = 10) -> List[AcademicPaper]:
        """Get papers similar to the given paper."""
        try:
            arxiv_id = self._extract_arxiv_id(paper_id)
            if not arxiv_id:
                return []
            
            # Get the paper details first
            paper = self.get_paper_by_id(arxiv_id)
            if not paper:
                return []
            
            # Search for papers in the same category with similar titles
            query = f'cat:{paper.category} AND (ti:"{paper.title.split()[0]}")'
            
            params = {
                "search_query": query,
                "start": 0,
                "max_results": limit
            }
            
            data = self._make_request(self.base_url, params=params)
            return self._parse_papers_from_xml(data)
            
        except Exception as e:
            logger.warning("arXiv similar papers lookup failed: %s", e)
            return []
    
    def search_papers_by_query(self, query: str, limit: int = 10) -> List[AcademicPaper]:
        """Search papers by general query."""
        try:
            params = {
                "search_query": query,
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            data = self._make_request(self.base_url, params=params)
            return self._parse_papers_from_xml(data)
            
        except Exception as e:
            logger.warning("arXiv query search failed: %s", e)
            return []
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[AcademicPaper]:
        """Get paper by arXiv ID directly."""
        try:
            # Handle both old and new arXiv ID formats
            if not arxiv_id.startswith("arXiv:"):
                arxiv_id = f"arXiv:{arxiv_id}"
            
            # For PDF URLs, we need the raw ID
            raw_id = arxiv_id.replace("arXiv:", "")
            
            query = f'id:"{raw_id}"'
            params = {
                "search_query": query,
                "start": 0,
                "max_results": 1
            }
            
            data = self._make_request(self.base_url, params=params)
            papers = self._parse_papers_from_xml(data)
            
            return papers[0] if papers else None
            
        except Exception as e:
            logger.warning("arXiv ID search failed: %s", e)
            return None
    
    def _parse_papers_from_xml(self, xml_data: str) -> List[AcademicPaper]:
        """Parse papers from arXiv XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_data)
            
            # Define namespace
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            for entry in root.findall('atom:entry', namespaces):
                paper = self._parse_paper_from_entry(entry, namespaces)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.warning("Error parsing arXiv papers: %s", e)
            return []
    
    def _parse_paper_from_entry(self, entry, namespaces: Dict[str, str]) -> Optional[AcademicPaper]:
        """Parse paper data from arXiv entry."""
        try:
            # Extract title
            title_elem = entry.find('atom:title', namespaces)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            if not title:
                return None
            
            # Extract authors
            authors = []
            author_elems = entry.findall('atom:author', namespaces)
            for author_elem in author_elems:
                name_elem = author_elem.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract year from published date
            year = None
            published_elem = entry.find('atom:published', namespaces)
            if published_elem is not None:
                date_str = published_elem.text
                year_match = re.search(r"(19|20)\d{2}", date_str)
                if year_match:
                    year = int(year_match.group(1))
            
            # Extract DOI if available
            doi = None
            doi_elem = entry.find('arxiv:doi', namespaces)
            if doi_elem is not None:
                doi = doi_elem.text.strip()
            
            # Extract abstract
            summary_elem = entry.find('atom:summary', namespaces)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Extract category
            category = ""
            category_elem = entry.find('atom:category', namespaces)
            if category_elem is not None:
                category = category_elem.get('term', '')
            
            # Extract URL
            url_elem = entry.find('atom:id', namespaces)
            url = url_elem.text.strip() if url_elem is not None else ""
            
            # Generate PDF URL
            pdf_url = self._generate_pdf_url(url)
            
            # Extract arXiv ID
            arxiv_id = self._extract_arxiv_id(url)
            
            return AcademicPaper(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                abstract=abstract,
                venue=f"arXiv:{category}",
                url=url,
                pdf_url=pdf_url,
                confidence_score=0.9,  # High confidence for arXiv
                source="arXiv"
            )
            
        except Exception as e:
            logger.warning("Error parsing arXiv paper: %s", e)
            return None
    
    def _extract_arxiv_id(self, url_or_id: str) -> Optional[str]:
        """Extract arXiv ID from URL or ID string."""
        # Handle various arXiv ID formats
        patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)',
            r'arxiv\.org/pdf/(\d+\.\d+)',
            r'arXiv:(\d+\.\d+)',
            r'(\d+\.\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return None
    
    def _generate_pdf_url(self, arxiv_url: str) -> str:
        """Generate PDF URL from arXiv abstract URL."""
        if not arxiv_url:
            return ""
        
        # Convert abs URL to PDF URL
        if "arxiv.org/abs/" in arxiv_url:
            pdf_url = arxiv_url.replace("/abs/", "/pdf/")
        elif "arxiv.org/abs" in arxiv_url:
            # Handle base case
            pdf_url = arxiv_url.replace("arxiv.org/abs", "arxiv.org/pdf")
        else:
            # For other URLs, try to construct PDF URL
            arxiv_id = self._extract_arxiv_id(arxiv_url)
            if arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            else:
                pdf_url = ""
        
        return pdf_url
