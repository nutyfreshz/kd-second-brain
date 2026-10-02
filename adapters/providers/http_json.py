from __future__ import annotations

import json
from urllib import error, request

from adapters.providers.base import ProviderError


def post_json(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 45,
) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = request.Request(url, data=body, headers=req_headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:2000]
        status = exc.code
        if status in {401, 403}:
            raise ProviderError(
                "Provider authentication/authorization failed.",
                code=f"http_{status}",
                retryable=False,
            ) from exc
        if status == 429:
            raise ProviderError(
                "Provider quota or rate limit reached.",
                code="http_429",
                retryable=False,
                quota_exhausted=True,
            ) from exc
        if 500 <= status < 600:
            raise ProviderError(
                "Provider temporarily unavailable.",
                code=f"http_{status}",
                retryable=True,
            ) from exc
        if "safety" in raw.lower() or "blocked" in raw.lower():
            raise ProviderError(
                "Provider blocked the request.",
                code=f"http_{status}_safety",
                safety_block=True,
            ) from exc
        raise ProviderError(
            "Provider request failed.",
            code=f"http_{status}",
            retryable=False,
        ) from exc
    except (error.URLError, TimeoutError) as exc:
        raise ProviderError(
            "Provider connection failed.",
            code="network_error",
            retryable=True,
        ) from exc
