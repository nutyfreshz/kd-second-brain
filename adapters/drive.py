from __future__ import annotations

import hashlib
import json
from typing import Any

from core.kb_sync import RawKnowledgeFile
from core.text_extraction import extract_text_from_bytes, is_markdown_name


_FOLDER_MIME = "application/vnd.google-apps.folder"
_GOOGLE_DOC = "application/vnd.google-apps.document"
_GOOGLE_SHEET = "application/vnd.google-apps.spreadsheet"
_GOOGLE_SLIDE = "application/vnd.google-apps.presentation"

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class GoogleDriveKnowledgeSource:
    """Read-only recursive adapter for a Drive knowledge tree.

    Every supported text-bearing file is normalized to text before indexing.
    Unsupported binary files are skipped rather than indexed as garbage.
    """

    def __init__(
        self,
        folder_id: str,
        *,
        service_account_json: str | None = None,
        service_account_file: str | None = None,
    ):
        if not folder_id:
            raise ValueError("GOOGLE_DRIVE_PUBLISHED_FOLDER_ID is required")
        self.folder_id = folder_id
        self.service_account_json = service_account_json
        self.service_account_file = service_account_file
        self.last_skipped: tuple[str, ...] = ()
        self.last_errors: tuple[str, ...] = ()
        self.last_manifest_fingerprint: str | None = None

    def _service(self):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        if self.service_account_json:
            info: dict[str, Any] = json.loads(self.service_account_json)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            )
        elif self.service_account_file:
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_file, scopes=scopes
            )
        else:
            raise RuntimeError(
                "Google Drive identity is not configured. "
                "Set GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE."
            )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def _list_children(self, service, folder_id: str) -> list[dict[str, Any]]:
        page_token = None
        items: list[dict[str, Any]] = []
        query = f"'{folder_id}' in parents and trashed = false"
        while True:
            result = (
                service.files()
                .list(
                    q=query,
                    fields=(
                        "nextPageToken,"
                        "files(id,name,mimeType,modifiedTime,webViewLink)"
                    ),
                    pageToken=page_token,
                    pageSize=1000,
                )
                .execute()
            )
            items.extend(result.get("files", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                return items

    def _walk_tree(self, service):
        pending = [self.folder_id]
        visited: set[str] = set()
        while pending:
            folder_id = pending.pop()
            if folder_id in visited:
                continue
            visited.add(folder_id)
            for item in self._list_children(service, folder_id):
                yield item
                if item.get("mimeType") == _FOLDER_MIME:
                    pending.append(item["id"])

    @staticmethod
    def _fingerprint_items(items: list[dict[str, Any]]) -> str:
        payload = [
            (
                item.get("id", ""),
                item.get("name", ""),
                item.get("mimeType", ""),
                item.get("modifiedTime", ""),
            )
            for item in items
        ]
        payload.sort()
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def manifest_fingerprint(self) -> str:
        service = self._service()
        items = list(self._walk_tree(service))
        return self._fingerprint_items(items)

    @staticmethod
    def _bytes(result: Any) -> bytes:
        if isinstance(result, bytes):
            return result
        if isinstance(result, str):
            return result.encode("utf-8")
        return bytes(result)

    def _extract_item(self, service, item: dict[str, Any]) -> str | None:
        file_id = item["id"]
        name = item.get("name", file_id)
        mime = item.get("mimeType", "")

        if mime == _GOOGLE_DOC:
            data = self._bytes(
                service.files().export(
                    fileId=file_id, mimeType=_DOCX_MIME
                ).execute()
            )
            return extract_text_from_bytes(
                name=name + ".docx", mime_type=_DOCX_MIME, data=data
            )

        if mime == _GOOGLE_SHEET:
            data = self._bytes(
                service.files().export(
                    fileId=file_id, mimeType=_XLSX_MIME
                ).execute()
            )
            return extract_text_from_bytes(
                name=name + ".xlsx", mime_type=_XLSX_MIME, data=data
            )

        if mime == _GOOGLE_SLIDE:
            data = self._bytes(
                service.files().export(
                    fileId=file_id, mimeType=_PPTX_MIME
                ).execute()
            )
            return extract_text_from_bytes(
                name=name + ".pptx", mime_type=_PPTX_MIME, data=data
            )

        # Other Google Workspace-native binary types (Forms, Drawings, etc.) do
        # not expose a stable text export through this adapter yet.
        if mime.startswith("application/vnd.google-apps."):
            return None

        data = self._bytes(service.files().get_media(fileId=file_id).execute())
        return extract_text_from_bytes(name=name, mime_type=mime, data=data)

    def read_all(self) -> list[RawKnowledgeFile]:
        service = self._service()
        out: list[RawKnowledgeFile] = []
        skipped: list[str] = []
        errors: list[str] = []
        manifest_items: list[dict[str, Any]] = []

        for item in self._walk_tree(service):
            manifest_items.append(item)
            if item.get("mimeType") == _FOLDER_MIME:
                continue

            name = item.get("name", item.get("id", "source"))
            try:
                text = self._extract_item(service, item)
            except Exception as exc:
                errors.append(f"{name}:{type(exc).__name__}")
                continue

            if not text or not text.strip():
                skipped.append(name)
                continue

            out.append(
                RawKnowledgeFile(
                    source_id=item["id"],
                    text=text,
                    updated_at=item.get("modifiedTime", ""),
                    origin_url=item.get("webViewLink"),
                    title_hint=name,
                    use_frontmatter=is_markdown_name(name),
                )
            )

        self.last_skipped = tuple(skipped)
        self.last_errors = tuple(errors)
        self.last_manifest_fingerprint = self._fingerprint_items(manifest_items)
        return out
