import unittest

from core.answer_policy import parse_model_json
from core.schemas import EvidenceChunk


class AnswerPolicyTests(unittest.TestCase):
    def test_derives_citations_from_claim_ids(self):
        evidence = [
            EvidenceChunk(
                chunk_id="doc:1",
                source_id="doc",
                title="Rules",
                heading="Incentive",
                text="Team Incentive = Install Cases x 30.",
                ordinal=1,
            )
        ]
        parsed = parse_model_json(
            '{"status":"answer","answer_th":"คิดจากจำนวนงาน","claims":'
            '[{"text":"Team incentive คิดจากจำนวนงาน","citation_ids":["doc:1"]}],'
            '"clarification_questions":[]}',
            evidence,
        )
        self.assertEqual(len(parsed["citations"]), 1)
        self.assertEqual(parsed["citations"][0].chunk_id, "doc:1")
        self.assertEqual(
            parsed["citations"][0].quote,
            "Team Incentive = Install Cases x 30.",
        )


if __name__ == "__main__":
    unittest.main()
