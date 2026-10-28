import unittest
from unittest.mock import patch

from adapters.providers.openrouter import OpenRouterProvider


class OpenRouterProviderTests(unittest.TestCase):
    @patch("adapters.providers.openrouter.post_json")
    def test_current_paid_uat_model_uses_single_route_and_schema(self, post_json):
        post_json.return_value = {
            "model": "openai/gpt-6-luna",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"status":"not_found","answer_th":"ไม่พบข้อมูล",'
                            '"claims":[],"clarification_questions":[]}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        provider = OpenRouterProvider(
            "test-key",
            "openai/gpt-6-luna",
            paid_allowed=True,
        )

        result = provider.generate(system_prompt="system", user_prompt="question")

        self.assertEqual(result.model, "openai/gpt-6-luna")
        payload = post_json.call_args.args[1]
        self.assertEqual(payload["model"], "openai/gpt-6-luna")
        self.assertNotIn("models", payload)
        self.assertNotIn("provider", payload)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])

    def test_paid_model_requires_explicit_paid_gate(self):
        disabled = OpenRouterProvider(
            "test-key",
            "openai/gpt-6-luna",
            paid_allowed=False,
        )
        enabled = OpenRouterProvider(
            "test-key",
            "openai/gpt-6-luna",
            paid_allowed=True,
        )
        self.assertFalse(disabled.enabled)
        self.assertTrue(enabled.enabled)

    def test_obsolete_or_unapproved_models_are_disabled(self):
        provider = OpenRouterProvider(
            "test-key",
            "google/gemini-2.5-flash-lite",
            paid_allowed=True,
        )
        self.assertFalse(provider.enabled)


if __name__ == "__main__":
    unittest.main()
