from __future__ import annotations

import threading
import uuid

from core.schemas import Conversation, Identity, Message, utc_now_iso


class ConversationNotFound(KeyError):
    pass


class ConversationForbidden(PermissionError):
    pass


class InMemoryConversationStore:
    def __init__(self):
        self._items: dict[str, Conversation] = {}
        self._lock = threading.RLock()

    def create(self, identity: Identity) -> Conversation:
        with self._lock:
            conversation = Conversation(
                conversation_id=str(uuid.uuid4()),
                owner_user_id=identity.user_id,
                created_at=utc_now_iso(),
            )
            self._items[conversation.conversation_id] = conversation
            return conversation

    def get(self, conversation_id: str, identity: Identity) -> Conversation:
        with self._lock:
            item = self._items.get(conversation_id)
            if item is None:
                raise ConversationNotFound(conversation_id)
            if item.owner_user_id != identity.user_id:
                raise ConversationForbidden(conversation_id)
            return item

    def add_message(
        self, conversation_id: str, identity: Identity, message: Message
    ) -> Message:
        with self._lock:
            conversation = self.get(conversation_id, identity)
            conversation.messages.append(message)
            if message.client_message_id:
                conversation.processed_client_ids[message.client_message_id] = (
                    message.message_id
                )
            return message

    def response_for_client_id(
        self, conversation_id: str, identity: Identity, client_message_id: str
    ) -> Message | None:
        with self._lock:
            conversation = self.get(conversation_id, identity)
            message_id = conversation.processed_client_ids.get(client_message_id)
            if not message_id:
                return None
            return next(
                (m for m in conversation.messages if m.message_id == message_id), None
            )
