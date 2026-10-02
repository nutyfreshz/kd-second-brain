import unittest

from adapters.conversation_store import (
    ConversationForbidden,
    InMemoryConversationStore,
)
from core.schemas import Identity, Message, utc_now_iso


class ConversationStoreTests(unittest.TestCase):
    def test_owner_enforced_and_client_id_indexed(self):
        store = InMemoryConversationStore()
        a = Identity("a", "a")
        b = Identity("b", "b")
        c = store.create(a)
        with self.assertRaises(ConversationForbidden):
            store.get(c.conversation_id, b)

        message = Message(
            message_id="m1",
            role="assistant",
            text="ok",
            created_at=utc_now_iso(),
            client_message_id="client-1",
            status="answer",
        )
        store.add_message(c.conversation_id, a, message)
        self.assertEqual(
            store.response_for_client_id(c.conversation_id, a, "client-1").message_id,
            "m1",
        )


if __name__ == "__main__":
    unittest.main()
