import re
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnhancedMetadata:
    """Enhanced metadata extraction result."""
    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    venue: str
    abstract: str
    keywords: List[str]
    references: List[str]
    citation_count: Optional[int]
    is_review: bool
    is_preprint: bool
    confidence_score: float


class EnhancedMetadataExtractor:
    """Enhanced metadata extraction with DOI detection and validation."""
    
    def __init__(self):
        self.doi_patterns = [
            r'10\.\d{4,9}/[-._;()/:A-Z0-9]+',
            r'10\.\d{4}/[-._;()/:A-Z0-9]+',
            r'10\.\d{3,4}/[-._;()/:A-Z0-9]+'
        ]
        
        self.author_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:,?\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)',
        ]
        
        self.year_patterns = [
            r'\b(19|20)\d{2}\b',
            r'\b\d{4}\b'
        ]
        
        self.venue_patterns = [
            r'\b([A-Z][A-Za-z\s]+(?:Journal|Proceedings|Conference|Transactions|Letters|Review))\b',
            r'\b([A-Z][A-Za-z\s]+(?:and\s+)?(?:Comp|Comput|Sci|Tech|Eng|Math))\b'
        ]
    
    def extract_enhanced_metadata(self, pdf_text: str, file_path: Path) -> EnhancedMetadata:
        """Extract enhanced metadata from PDF text."""
        # Clean the text
        clean_text = self._clean_text(pdf_text)
        
        # Extract individual components
        title = self._extract_title(clean_text, file_path)
        authors = self._extract_authors(clean_text)
        year = self._extract_year(clean_text)
        doi = self._extract_doi(clean_text)
        venue = self._extract_venue(clean_text)
        abstract = self._extract_abstract(clean_text)
        keywords = self._extract_keywords(clean_text)
        references = self._extract_references(clean_text)
        citation_count = self._extract_citation_count(clean_text)
        is_review = self._detect_review_paper(clean_text)
        is_preprint = self._detect_preprint(clean_text, doi)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract
        )
        
        return EnhancedMetadata(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            venue=venue,
            abstract=abstract,
            keywords=keywords,
            references=references,
            citation_count=citation_count,
            is_review=is_review,
            is_preprint=is_preprint,
            confidence_score=confidence
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize PDF text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common PDF artifacts
        text = re.sub(r'\f', '', text)  # Form feeds
        text = re.sub(r'\r', '', text)  # Carriage returns
        # Normalize quotes
        # Map various quote characters to a plain double quote.
        text = re.sub(r"[\"'`]", '"', text)
        return text.strip()
    
    def _extract_title(self, text: str, file_path: Path) -> str:
        """Extract paper title with multiple strategies."""
        # Strategy 1: Look for title patterns in first few lines
        lines = text.split('\n')[:20]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # Check if it looks like a title (capitalized, reasonable length)
                if (line[0].isupper() and 
                    line.count('.') < 2 and 
                    not line.startswith('http') and
                    not re.match(r'\d{4}', line)):
                    return line
        
        # Strategy 2: Use filename as fallback
        filename = file_path.stem
        filename = filename.replace('_', ' ').replace('-', ' ')
        if len(filename) > 3:
            return filename
        
        # Strategy 3: Find the longest capitalized line
        longest_line = ""
        for line in lines:
            line = line.strip()
            if (len(line) > len(longest_line) and 
                line[0].isupper() and 
                line.count('.') < 3):
                longest_line = line
        
        return longest_line or "Untitled"
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names with multiple patterns."""
        authors = []
        
        # Look for author patterns
        for pattern in self.author_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean and format author names
                author = re.sub(r'[,\s]+', ' ', match).strip()
                if author and len(author) > 3:
                    authors.append(author)
        
        # Remove duplicates and limit to reasonable number
        authors = list(dict.fromkeys(authors))[:10]
        
        return authors if authors else ["Unknown"]
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year."""
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Try to validate the year (1900-2030)
                year = int(match)
                if 1900 <= year <= 2030:
                    return year
        return None
    
    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract and validate DOI."""
        for pattern in self.doi_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Basic DOI validation
                if self._validate_doi(match):
                    return match
        return None
    
    def _validate_doi(self, doi: str) -> bool:
        """Basic DOI validation."""
        if not doi.startswith('10.'):
            return False
        
        # Check length
        if len(doi) < 8 or len(doi) > 100:
            return False
        
        # Check for valid characters
        valid_chars = r'10.[/_;():A-Z0-9a-z.-]+'
        if not re.fullmatch(valid_chars, doi, re.IGNORECASE):
            return False
        
        return True
    
    def _extract_venue(self, text: str) -> str:
        """Extract publication venue/journal/conference."""
        venue = ""
        
        # Look for venue patterns
        for pattern in self.venue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                venue = matches[0]
                break
        
        # Fallback: look for common venue indicators
        if not venue:
            venue_indicators = ['journal', 'conference', 'proceedings', 'transactions', 'letters']
            for indicator in venue_indicators:
                if indicator in text.lower():
                    venue = indicator.title()
                    break
        
        return venue or "Unknown"
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section."""
        # Look for abstract section
        abstract_patterns = [
            r'abstract\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]{2,40}\n|\Z)',
            r'ABSTRACT\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]{2,40}\n|\Z)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = re.sub(r'\s+', ' ', match.group(1)).strip()
                if len(abstract) > 50:  # Reasonable abstract length
                    return abstract[:2000]  # Limit length
        
        # Fallback: extract first substantial paragraph
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs[:5]:  # Check first 5 paragraphs
            paragraph = paragraph.strip()
            if len(paragraph) > 100 and len(paragraph) < 2000:
                # Check if it looks like an abstract (not too many technical details)
                sentence_count = len(re.findall(r'[.!?]+', paragraph))
                if 2 <= sentence_count <= 10:
                    return paragraph[:2000]
        
        return ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        keywords = []
        
        # Look for keywords section
        keyword_patterns = [
            r'keywords?\s*[:\-]?\s*(.*?)(?:\n|$)',
            r'KEYWORDS?\s*[:\-]?\s*(.*?)(?:\n|$)',
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords_text = match.group(1)
                # Split keywords by comma, semicolon, or dash
                keywords = re.split(r'[,;\-]\s*', keywords_text)
                break
        
        # Clean and validate keywords
        cleaned_keywords = []
        for keyword in keywords[:10]:  # Limit to 10 keywords
            keyword = keyword.strip()
            if (len(keyword) > 2 and 
                len(keyword) < 50 and 
                not keyword.startswith('http')):
                cleaned_keywords.append(keyword)
        
        return cleaned_keywords
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract reference section information."""
        references = []
        
        # Look for references section
        ref_patterns = [
            r'references?\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]{2,40}\n|\Z)',
            r'bibliography\s*(.*?)(?:\n\s*[A-Z][A-Za-z ]{2,40}\n|\Z)',
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                references_text = match.group(1)
                # Count references (rough estimate)
                ref_count = len(re.findall(r'\[\d+\]', references_text))
                if ref_count > 0:
                    references = [f"Estimated {ref_count} references"]
                    break
        
        return references
    
    def _extract_citation_count(self, text: str) -> Optional[int]:
        """Extract citation count if mentioned in text."""
        # Look for citation count patterns
        citation_patterns = [
            r'cited\s+by\s*(\d+)',
            r'citations?\s*[:\-]?\s*(\d+)',
            r'has\s*(\d+)\s*citations?',
        ]
        
        for pattern in citation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _detect_review_paper(self, text: str) -> bool:
        """Detect if this is a review paper."""
        review_indicators = [
            'review', 'survey', 'perspective', 'commentary', 'editorial',
            'introduction to', 'tutorial', 'overview'
        ]
        
        text_lower = text.lower()
        for indicator in review_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def _detect_preprint(self, text: str, doi: Optional[str]) -> bool:
        """Detect if this is a preprint."""
        if doi and 'arxiv' in doi.lower():
            return True
        
        arxiv_indicators = ['arxiv:', 'arxiv.org', 'preprint', 'working paper']
        text_lower = text.lower()
        
        for indicator in arxiv_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def _calculate_confidence(self, title: str, authors: List[str], year: Optional[int], 
                            doi: Optional[str], abstract: str) -> float:
        """Calculate confidence score for metadata extraction."""
        confidence = 0.0
        
        # Title quality
        if len(title) > 10 and len(title) < 200:
            confidence += 0.2
        
        # Authors
        if len(authors) >= 1 and len(authors) <= 10:
            confidence += 0.2
        elif len(authors) > 0:
            confidence += 0.1
        
        # Year
        if year is not None:
            confidence += 0.2
        
        # DOI (strong indicator of quality metadata)
        if doi is not None:
            confidence += 0.3
        
        # Abstract
        if len(abstract) > 100:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def validate_metadata(self, metadata: EnhancedMetadata) -> bool:
        """Validate extracted metadata quality."""
        # Basic validation
        if not metadata.title or len(metadata.title) < 3:
            return False
        
        if not metadata.authors or all(len(author) < 2 for author in metadata.authors):
            return False
        
        # High confidence if DOI is present
        if metadata.doi:
            return metadata.confidence_score >= 0.5
        
        # Lower confidence requirements for papers without DOI
        return metadata.confidence_score >= 0.3
