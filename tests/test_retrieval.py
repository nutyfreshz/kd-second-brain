import unittest
from pathlib import Path

from core.md_parser import parse_markdown
from core.retrieval import HybridRetriever, is_overview_query
from core.schemas import EvidenceChunk


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

    def test_overview_query_is_source_diverse(self):
        chunks = [
            EvidenceChunk("a:0", "a", "Install Overview", "Overview", "Install incentive overview", 0),
            EvidenceChunk("a:1", "a", "Install Overview", "Rule", "Install team incentive details", 1),
            EvidenceChunk("b:0", "b", "MA Overview", "Overview", "MA incentive overview", 0),
            EvidenceChunk("b:1", "b", "MA Overview", "Rule", "MA zone incentive details", 1),
            EvidenceChunk("c:0", "c", "Other", "Other", "unrelated material", 0),
        ]
        retriever = HybridRetriever(chunks, semantic_enabled=False, model_name="unused")
        results = retriever.search("Incentive คิดยังไง", evidence_k=4)
        source_ids = {item.source_id for item in results}
        self.assertIn("a", source_ids)
        self.assertIn("b", source_ids)
        self.assertTrue(is_overview_query("Incentive คิดยังไง"))

    def test_retrieval_expands_adjacent_context(self):
        chunks = [
            EvidenceChunk("a:0", "a", "Policy", "Intro", "general introduction", 0),
            EvidenceChunk("a:1", "a", "Policy", "Rule", "special code P9 threshold", 1),
            EvidenceChunk("a:2", "a", "Policy", "Exception", "P9 exception applies on HOLD", 2),
        ]
        retriever = HybridRetriever(chunks, semantic_enabled=False, model_name="unused")
        results = retriever.search("P9 threshold", evidence_k=3)
        ids = {item.chunk_id for item in results}
        self.assertIn("a:1", ids)
        self.assertTrue("a:0" in ids or "a:2" in ids)

    def test_global_overview_can_return_representatives_without_term_match(self):
        chunks = [
            EvidenceChunk("a:0", "a", "Alpha", "Intro", "alpha content", 0),
            EvidenceChunk("b:0", "b", "Beta", "Intro", "beta content", 0),
            EvidenceChunk("c:0", "c", "Gamma", "Intro", "gamma content", 0),
        ]
        retriever = HybridRetriever(chunks, semantic_enabled=False, model_name="unused")
        results = retriever.search("สรุปทุกเอกสารทั้งหมด", evidence_k=3)
        self.assertEqual({item.source_id for item in results}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()