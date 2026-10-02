import tempfile
import unittest
from datetime import date
from pathlib import Path

from core.kb_sync import (
    KnowledgeManager,
    LocalKnowledgeSource,
    resolve_authoritative_sources,
)


class KBSyncTests(unittest.TestCase):
    def test_detects_unresolved_duplicate_doc_authority(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            p.joinpath("a.md").write_text(
                "---\ndoc_id: same\nstatus: published\nversion: v1\n---\n# A\none",
                encoding="utf-8",
            )
            p.joinpath("b.md").write_text(
                "---\ndoc_id: same\nstatus: published\nversion: v2\n---\n# B\ntwo",
                encoding="utf-8",
            )
            manager = KnowledgeManager(
                LocalKnowledgeSource(td),
                semantic_enabled=False,
                embedding_model="unused",
            )
            snap = manager.sync()
            allowed, conflicts = resolve_authoritative_sources(
                snap.sources, date(2026, 10, 2)
            )
            self.assertEqual(allowed, set())
            self.assertEqual(conflicts, ("same",))

    def test_supersedes_resolves_authority(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            p.joinpath("a.md").write_text(
                "---\ndoc_id: same\nstatus: published\nversion: v1\n---\n# A\none",
                encoding="utf-8",
            )
            p.joinpath("b.md").write_text(
                "---\ndoc_id: same\nstatus: published\nversion: v2\nsupersedes: [v1]\n---\n# B\ntwo",
                encoding="utf-8",
            )
            manager = KnowledgeManager(
                LocalKnowledgeSource(td),
                semantic_enabled=False,
                embedding_model="unused",
            )
            snap = manager.sync()
            allowed, conflicts = resolve_authoritative_sources(
                snap.sources, date(2026, 10, 2)
            )
            self.assertEqual(conflicts, ())
            self.assertEqual(len(allowed), 1)

    def test_canonical_document_is_included_and_authoritative(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            p.joinpath("canonical.md").write_text(
                "---\\ndoc_id: rules\\nstatus: canonical\\nversion: v1\\n---\\n# Rules\\ncanonical content",
                encoding="utf-8",
            )
            manager = KnowledgeManager(
                LocalKnowledgeSource(td),
                semantic_enabled=False,
                embedding_model="unused",
            )
            snap = manager.sync()
            self.assertEqual(len(snap.sources), 1)
            self.assertEqual(snap.sources[0].status, "canonical")
            allowed, conflicts = resolve_authoritative_sources(
                snap.sources, date(2026, 10, 2)
            )
            self.assertEqual(conflicts, ())
            self.assertEqual(allowed, {"canonical.md"})


if __name__ == "__main__":
    unittest.main()