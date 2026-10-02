from __future__ import annotations

from core.schemas import Identity
from settings import Settings


def identity_from_headers(headers, settings: Settings) -> Identity:
    def get(name: str) -> str | None:
        try:
            value = headers.get(name)
        except Exception:
            value = None
        return value.strip() if isinstance(value, str) and value.strip() else None

    user_id = get("x-forwarded-user")
    username = get("x-forwarded-preferred-username") or get("x-forwarded-email") or user_id
    email = get("x-forwarded-email")

    if not user_id:
        if not settings.allow_anonymous_local:
            raise PermissionError(
                "No trusted Databricks identity header. "
                "Anonymous mode is disabled."
            )
        user_id = "local-dev"
        username = "local-dev"

    username = username or user_id
    admin_candidates = {user_id, username}
    if email:
        admin_candidates.add(email)
    return Identity(
        user_id=user_id,
        username=username,
        email=email,
        is_admin=bool(admin_candidates & settings.admin_users)
        or (settings.allow_anonymous_local and user_id == "local-dev"),
    )
