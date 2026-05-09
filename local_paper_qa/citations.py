from __future__ import annotations

from local_paper_qa.models import PaperCitation


def format_apa(citation: PaperCitation) -> str:
    authors = citation.authors.strip() or "Unknown author"
    year = citation.year.strip() or "n.d."
    title = citation.paper_title.strip() or "Untitled"
    venue = citation.venue.strip()
    doi = citation.doi.strip()

    reference = f"{authors} ({year}). {title}."
    if venue:
        reference += f" {venue}."
    if doi:
        reference += f" https://doi.org/{doi}"
    return reference
