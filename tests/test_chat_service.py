import tempfile
import unittest
from pathlib import Path

from adapters.conversation_store import InMemoryConversationStore
from core.chat_service import ChatService
from core.kb_sync import KnowledgeManager, LocalKnowledgeSource
from core.schemas import AnswerStatus, ChatRequest, Identity


class ChatServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        source_dir = Path(self.tmp.name)
        source_dir.joinpath("policy.md").write_text(
            Path("tests/fixtures/policy.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        knowledge = KnowledgeManager(
            LocalKnowledgeSource(str(source_dir)),
            semantic_enabled=False,
            embedding_model="unused",
        )
        knowledge.sync()
        self.service = ChatService(
            knowledge=knowledge,
            store=InMemoryConversationStore(),
            primary=None,
            fallback=None,
        )
        self.identity = Identity("u1", "u1")
        self.cid = self.service.create_conversation(self.identity)

    def tearDown(self):
        self.tmp.cleanup()

    def test_degrades_to_search_only_without_provider(self):
        response = self.service.send_message(
            self.identity,
            ChatRequest(
                conversation_id=self.cid,
                client_message_id="c1",
                text="P2 requires how many completed jobs?",
            ),
        )
        self.assertEqual(response.status, AnswerStatus.SEARCH_ONLY)
        self.assertTrue(response.citations)

    def test_duplicate_client_id_does_not_append_second_user_turn(self):
        request = ChatRequest(
            conversation_id=self.cid,
            client_message_id="same",
            text="P2 completed jobs",
        )
        first = self.service.send_message(self.identity, request)
        before = len(self.service.get_conversation(self.identity, self.cid).messages)
        second = self.service.send_message(self.identity, request)
        after = len(self.service.get_conversation(self.identity, self.cid).messages)
        self.assertEqual(first.message_id, second.message_id)
        self.assertEqual(before, after)
        self.assertTrue(second.cache_hit)


if __name__ == "__main__":
    unittest.main()
