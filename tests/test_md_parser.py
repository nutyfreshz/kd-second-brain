import unittest
from pathlib import Path

from core.md_parser import is_active_status, parse_markdown


class MarkdownParserTests(unittest.TestCase):
    def test_parses_metadata_and_chunks(self):
        text = Path("tests/fixtures/policy.md").read_text(encoding="utf-8")
        doc = parse_markdown("policy.md", text, updated_at="2026-10-02")
        self.assertEqual(doc.meta.doc_id, "demo_policy")
        self.assertEqual(doc.meta.title, "Demo Incentive Policy")
        self.assertTrue(doc.meta.external_llm_allowed)
        self.assertGreaterEqual(len(doc.chunks), 2)
        self.assertEqual(doc.chunks[0].source_id, "policy.md")

    def test_canonical_status_is_active(self):
        self.assertTrue(is_active_status("canonical"))
        self.assertTrue(is_active_status("published"))
        self.assertFalse(is_active_status("draft"))


if __name__ == "__main__":
    unittest.main()