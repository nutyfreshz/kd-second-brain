import unittest
from pathlib import Path

from core.md_parser import parse_markdown
from core.retrieval import HybridRetriever


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        text = Path("tests/fixtures/policy.md").read_text(encoding="utf-8")
        self.doc = parse_markdown("policy.md", text)

    def test_lexical_keeps_codes_and_numbers(self):
        retriever = HybridRetriever(
            self.doc.chunks, semantic_enabled=False, model_name="unused"
        )
        results = retriever.search("P2 3 completed jobs", evidence_k=3)
        self.assertTrue(results)
        self.assertIn("P2", results[0].text)

    def test_source_filter(self):
        retriever = HybridRetriever(
            self.doc.chunks, semantic_enabled=False, model_name="unused"
        )
        results = retriever.search(
            "KPI", allowed_source_ids={"other.md"}, evidence_k=3
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
