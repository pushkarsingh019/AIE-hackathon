from __future__ import annotations

import re

from local_paper_qa.models import PaperCitation


def _format_authors_for_apa(authors_str: str) -> str:
    """Format authors string for APA style (Last, A. A., Last, B. B.)."""
    authors_str = authors_str.strip()
    if not authors_str or authors_str == "Unknown":
        return "Unknown"
    
    # Already in good format if it has commas
    if "," in authors_str:
        # Clean up: ensure each author is "Last, First"
        parts = [a.strip() for a in authors_str.split(",")]
        if len(parts) >= 2:
            # Likely already in "Last, First" format
            return authors_str[:200]
    
    # Try to parse "First Last" -> "Last, F."
    authors_list = []
    for name in re.split(r"[;,&]+", authors_str):
        name = name.strip()
        if not name:
            continue
        parts = name.split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = "".join(p[0].upper() + "." for p in parts[:-1] if p)
            authors_list.append(f"{last}, {initials}")
        elif len(parts) == 1:
            authors_list.append(parts[0])
    
    if not authors_list:
        return authors_str[:200]
    
    # APA: max 20 authors, use "..." before last if more than 20
    if len(authors_list) > 20:
        authors_list = authors_list[:19] + ["..."]
    return ", ".join(authors_list)


def format_apa(citation: PaperCitation) -> str:
    """Format a citation in APA style.

    Example output:
        Bucci, M. A., Onofrio, A. (2023). Curriculum learning for data-driven modeling of dynamical systems. The European Physical Journal E. https://doi.org/10.1140/epje/s10189-023-00269-8
    """
    authors = _format_authors_for_apa(citation.authors)
    year = str(citation.year).strip() or "n.d."
    title = citation.paper_title.strip() or "Untitled"
    venue = citation.venue.strip()
    doi = citation.doi.strip()

    parts = [f"{authors} ({year}). {title}."]
    if venue:
        parts.append(venue + ".")
    if doi:
        parts.append(f"https://doi.org/{doi}")
    return " ".join(parts)
