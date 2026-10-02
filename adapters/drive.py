from __future__ import annotations

import json
from typing import Any

from core.kb_sync import RawKnowledgeFile


class GoogleDriveKnowledgeSource:
    """Read-only adapter for one explicit published folder."""

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

    def read_all(self) -> list[RawKnowledgeFile]:
        service = self._service()
        page_token = None
        out: list[RawKnowledgeFile] = []
        query = (
            f"'{self.folder_id}' in parents and trashed = false "
            "and mimeType != 'application/vnd.google-apps.folder'"
        )
        while True:
            result = (
                service.files()
                .list(
                    q=query,
                    fields="nextPageToken,files(id,name,mimeType,modifiedTime,webViewLink)",
                    pageToken=page_token,
                    pageSize=1000,
                )
                .execute()
            )
            for item in result.get("files", []):
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
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        return out
