import unittest
from pathlib import Path


class TestEnhancedMetadataExtractor(unittest.TestCase):
    def test_extracts_doi(self):
        from local_paper_qa.metadata.enhanced_extractor import EnhancedMetadataExtractor

        extractor = EnhancedMetadataExtractor()
        sample_text = """
        A Sample Paper Title

        John Doe, Jane Smith

        ABSTRACT
        This paper studies something important.

        DOI: 10.1000/182
        """.strip()

        meta = extractor.extract_enhanced_metadata(sample_text, Path("sample.pdf"))
        self.assertEqual(meta.doi, "10.1000/182")


if __name__ == "__main__":
    unittest.main()
