from __future__ import annotations

from urllib.parse import quote

from adapters.providers.base import GenerationResult, ProviderError
from adapters.providers.http_json import post_json


class GeminiProvider:
    ALLOWED_MODELS = {"gemini-3.7-flash"}

    def __init__(self, api_key: str | None, model: str, *, free_verified: bool = False):
        self.api_key = api_key
        self.model = model
        self.free_verified = free_verified

    @property
    def enabled(self) -> bool:
        return bool(
            self.api_key
            and self.free_verified
            and self.model in self.ALLOWED_MODELS
        )

    @property
    def model_id(self) -> str:
        return self.model

    def generate(self, *, system_prompt: str, user_prompt: str) -> GenerationResult:
        if not self.enabled:
            raise ProviderError(
                "Gemini is not configured.", code="not_configured", retryable=False
            )
        model = quote(self.model, safe="-_.")
        key = quote(self.api_key or "", safe="")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:"
            f"generateContent?key={key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        data = post_json(url, payload)
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = str(data.get("promptFeedback") or "")
            if "block" in feedback.lower():
                raise ProviderError(
                    "Gemini blocked the request.",
                    code="safety_block",
                    safety_block=True,
                )
            raise ProviderError(
                "Gemini returned no candidate.", code="empty_response", retryable=True
            )
        try:
            text = candidates[0]["content"]["parts"][0]["text"]
        except Exception as exc:
            raise ProviderError(
                "Gemini response shape was not recognized.",
                code="bad_response_shape",
                retryable=False,
            ) from exc
        return GenerationResult(
            text=text,
            model=self.model,
            provider="gemini",
            usage=data.get("usageMetadata"),
        )
