from __future__ import annotations

from adapters.providers.base import GenerationResult, ProviderError
from adapters.providers.http_json import post_json


class OpenRouterProvider:
    ALLOWED_MODELS = {
        "qwen/qwen3.5-9b",
        "google/gemini-2.5-flash-lite",
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b:free",
    }

    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        free_verified: bool = False,
        paid_allowed: bool = False,
        fallback_models: tuple[str, ...] = (),
    ):
        self.api_key = api_key
        self.model = model
        self.free_verified = free_verified
        self.paid_allowed = paid_allowed
        self.fallback_models = tuple(
            m for m in fallback_models if m in self.ALLOWED_MODELS and m != model
        )

    @property
    def enabled(self) -> bool:
        if not self.api_key or self.model not in self.ALLOWED_MODELS:
            return False
        if self.model.endswith(":free"):
            return self.free_verified
        return self.paid_allowed

    @property
    def model_id(self) -> str:
        return self.model

    def generate(self, *, system_prompt: str, user_prompt: str) -> GenerationResult:
        if not self.enabled:
            raise ProviderError(
                "OpenRouter is not configured.", code="not_configured", retryable=False
            )
        payload = {
            "temperature": 0.1,
            "max_tokens": 1000,
            "provider": {"require_parameters": True},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ais_knowledge_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["answer", "clarify", "not_found", "conflict"],
                            },
                            "answer_th": {"type": "string"},
                            "claims": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "citation_ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["text", "citation_ids"],
                                    "additionalProperties": False,
                                },
                            },
                            "citations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id": {"type": "string"},
                                    },
                                    "required": ["chunk_id"],
                                    "additionalProperties": False,
                                },
                            },
                            "clarification_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "status",
                            "answer_th",
                            "claims",
                            "citations",
                            "clarification_questions",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        }
        if self.fallback_models:
            payload["models"] = [self.model, *self.fallback_models]
        else:
            payload["model"] = self.model
        data = post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except Exception as exc:
            raise ProviderError(
                "OpenRouter response shape was not recognized.",
                code="bad_response_shape",
                retryable=False,
            ) from exc
        return GenerationResult(
            text=text,
            model=str(data.get("model") or self.model),
            provider="openrouter",
            usage=data.get("usage"),
        )