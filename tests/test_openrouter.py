import unittest
from unittest.mock import patch

from adapters.providers.openrouter import OpenRouterProvider


class OpenRouterProviderTests(unittest.TestCase):
    @patch("adapters.providers.openrouter.post_json")
    def test_uses_strict_json_schema_and_free_model(self, post_json):
        post_json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"status":"not_found","answer_th":"ไม่พบข้อมูล",'
                            '"claims":[],"citations":[],"clarification_questions":[]}'
                        )
                    }
                }
            ],
            "usage": {},
        }
        provider = OpenRouterProvider(
            "test-key",
            "openai/gpt-oss-120b:free",
            free_verified=True,
        )

        result = provider.generate(
            system_prompt="system",
            user_prompt="question",
        )

        self.assertEqual(result.model, "openai/gpt-oss-120b:free")
        payload = post_json.call_args.args[1]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertTrue(payload["provider"]["require_parameters"])
        self.assertEqual(payload["model"], "openai/gpt-oss-120b:free")

    def test_paid_qwen_requires_explicit_paid_gate(self):
        disabled = OpenRouterProvider(
            "test-key",
            "qwen/qwen3.5-9b",
            paid_allowed=False,
        )
        enabled = OpenRouterProvider(
            "test-key",
            "qwen/qwen3.5-9b",
            paid_allowed=True,
        )
        self.assertFalse(disabled.enabled)
        self.assertTrue(enabled.enabled)


if __name__ == "__main__":
    unittest.main()