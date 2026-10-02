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

    def test_valid_grounded_claim(self):
        result = validate_answer(
            claims=[Claim("Threshold is 80%.", ("x:1",))],
            citations=[
                Citation("x:1", self.chunk.text, "x", "X", "Rule")
            ],
            evidence=[self.chunk],
            user_text="",
        )
        self.assertTrue(result.valid)

    def test_allows_derived_or_reformatted_number_when_grounded(self):
        result = validate_answer(
            claims=[Claim("There are 2 numeric rules in this excerpt.", ("x:1",))],
            citations=[
                Citation("x:1", self.chunk.text, "x", "X", "Rule")
            ],
            evidence=[self.chunk],
            user_text="",
        )
        self.assertTrue(result.valid)

    def test_rejects_unknown_citation(self):
        result = validate_answer(
            claims=[Claim("Threshold is 80%.", ("unknown:1",))],
            citations=[
                Citation("unknown:1", "text", "x", "X", "Rule")
            ],
            evidence=[self.chunk],
            user_text="",
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("unknown_citation" in e for e in result.errors))

    def test_rejects_claim_without_citation(self):
        result = validate_answer(
            claims=[Claim("Threshold is 80%.", ())],
            citations=[],
            evidence=[self.chunk],
            user_text="",
        )
        self.assertFalse(result.valid)
        self.assertIn("claim_without_citation", result.errors)


if __name__ == "__main__":
    unittest.main()
