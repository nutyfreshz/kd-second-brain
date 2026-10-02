import unittest
from datetime import date

from core.conversation import resolve_effective_date


class ConversationContextTests(unittest.TestCase):
    def test_iso_month(self):
        self.assertEqual(resolve_effective_date("กฎ 2026-07"), date(2026, 7, 1))

    def test_thai_buddhist_month(self):
        self.assertEqual(
            resolve_effective_date("กฎเดือนสิงหาคม 2569"), date(2026, 8, 1)
        )


if __name__ == "__main__":
    unittest.main()
