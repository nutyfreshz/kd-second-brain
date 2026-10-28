import tempfile
import unittest
from datetime import date
from pathlib import Path

from core.kb_sync import (
    KnowledgeManager,
    LocalKnowledgeSource,
    RawKnowledgeFile,
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
                "---\ndoc_id: rules\nstatus: canonical\nversion: v1\n---\n# Rules\ncanonical content",
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

    def test_manifest_check_skips_reindex_until_source_changes(self):
        class FakeManifestSource:
            def __init__(self):
                self.version = "v1"
                self.reads = 0
                self.last_manifest_fingerprint = None

            def manifest_fingerprint(self):
                return self.version

            def read_all(self):
                self.reads += 1
                self.last_manifest_fingerprint = self.version
                return [
                    RawKnowledgeFile(
                        source_id="x.txt",
                        text=f"rule {self.version}",
                        updated_at=self.version,
                        title_hint="x.txt",
                        use_frontmatter=False,
                    )
                ]

        source = FakeManifestSource()
        manager = KnowledgeManager(
            source,
            semantic_enabled=False,
            embedding_model="unused",
        )
        first = manager.sync()
        self.assertEqual(source.reads, 1)

        manager._last_manifest_check = 0.0
        same = manager.maybe_sync(1)
        self.assertEqual(source.reads, 1)
        self.assertEqual(first.snapshot_id, same.snapshot_id)

        source.version = "v2"
        manager._last_manifest_check = 0.0
        changed = manager.maybe_sync(1)
        self.assertEqual(source.reads, 2)
        self.assertNotEqual(first.snapshot_id, changed.snapshot_id)


if __name__ == "__main__":
    unittest.main()