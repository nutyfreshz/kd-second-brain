from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    provider: str
    usage: dict | None = None


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool = False,
        quota_exhausted: bool = False,
        safety_block: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.quota_exhausted = quota_exhausted
        self.safety_block = safety_block


class LLMProvider(Protocol):
    @property
    def enabled(self) -> bool: ...

    @property
    def model_id(self) -> str: ...

    def generate(self, *, system_prompt: str, user_prompt: str) -> GenerationResult: ...
