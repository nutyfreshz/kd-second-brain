from __future__ import annotations

import json
from typing import Any

from core.kb_sync import RawKnowledgeFile


_FOLDER_MIME = "application/vnd.google-apps.folder"


class GoogleDriveKnowledgeSource:
    """Read-only adapter for one explicit published Drive tree."""

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

    def read_all(self) -> list[RawKnowledgeFile]:
        service = self._service()
        out: list[RawKnowledgeFile] = []
        pending = [self.folder_id]
        visited: set[str] = set()

        while pending:
            folder_id = pending.pop()
            if folder_id in visited:
                continue
            visited.add(folder_id)

            for item in self._list_children(service, folder_id):
                if item.get("mimeType") == _FOLDER_MIME:
                    pending.append(item["id"])
                    continue

                name = item.get("name", "")
                mime = item.get("mimeType", "")
                if not name.lower().endswith(".md") and mime != "text/markdown":
                    continue

                data = service.files().get_media(fileId=item["id"]).execute()
                text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                out.append(
                    RawKnowledgeFile(
                        source_id=item["id"],
                        text=text,
                        updated_at=item.get("modifiedTime", ""),
                        origin_url=item.get("webViewLink"),
                    )
                )

        return out
