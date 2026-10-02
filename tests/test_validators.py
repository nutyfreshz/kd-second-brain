import unittest

from core.schemas import Citation, Claim, EvidenceChunk
from core.validators import validate_answer


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.chunk = EvidenceChunk(
            chunk_id="x:1",
            source_id="x",
            title="X",
            heading="Rule",
            text="Threshold is 80% and payout is 100 points.",
            ordinal=1,
        )

    def test_valid_citation_and_number(self):
        result = validate_answer(
            claims=[Claim("Threshold is 80%.", ("x:1",))],
            citations=[
                Citation("x:1", "Threshold is 80%", "x", "X", "Rule")
            ],
            evidence=[self.chunk],
            user_text="",
        )
        self.assertTrue(result.valid)

    def test_rejects_unsupported_number(self):
        result = validate_answer(
            claims=[Claim("Threshold is 95%.", ("x:1",))],
            citations=[
                Citation("x:1", "Threshold is 80%", "x", "X", "Rule")
            ],
            evidence=[self.chunk],
            user_text="",
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("unsupported_number:95%" in e for e in result.errors))

    def test_rejects_non_verbatim_quote(self):
        result = validate_answer(
            claims=[Claim("Threshold is 80%.", ("x:1",))],
            citations=[
                Citation("x:1", "Threshold equals 80%", "x", "X", "Rule")
            ],
            evidence=[self.chunk],
            user_text="",
        )
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
