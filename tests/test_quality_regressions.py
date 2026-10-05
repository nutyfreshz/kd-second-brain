"""Synthetic public fixtures only; never add production knowledge to this suite."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from adapters.conversation_store import InMemoryConversationStore
from adapters.providers.base import GenerationResult
from adapters.providers.openrouter import OpenRouterProvider
from core.answer_policy import parse_model_json
from core.chat_service import ChatService
from core.conversation import retrieval_query
from core.kb_sync import KnowledgeManager, LocalKnowledgeSource
from core.md_parser import parse_markdown
from core.retrieval import BM25Index, HybridRetriever, SemanticIndex, tokenize
from core.schemas import AnswerStatus, ChatRequest, Citation, Claim, Conversation, EvidenceChunk, Identity, Message
from core.validators import validate_answer


def chunk(sid, text, ordinal=0):
    return EvidenceChunk(f'{sid}:{ordinal}', sid, sid, 'Rule', text, ordinal)


class RetrievalQualityTests(unittest.TestCase):
    def test_codes_remain_atomic(self):
        self.assertEqual(tokenize('P1 P2 P5 BBAM_24 3BB'), ['p1', 'p2', 'p5', 'bbam_24', '3bb'])

    def test_scope_before_candidate_cutoff(self):
        chunks = [chunk(f'd{i}', 'repair SLA') for i in range(40)]
        chunks.append(chunk('selected', 'repair SLA ' + 'other ' * 100))
        r = HybridRetriever(chunks, semantic_enabled=False, model_name='unused')
        self.assertEqual(r.search('repair SLA', allowed_source_ids={'selected'})[0].source_id, 'selected')

    def test_empty_scope_is_not_all_sources(self):
        r = HybridRetriever([chunk('a', 'SLA')], semantic_enabled=False, model_name='unused')
        self.assertEqual(r.search('SLA', allowed_source_ids=set()), [])

    def test_thai_query_and_unknown_query(self):
        r = HybridRetriever([chunk('a', 'งานซ่อมต้องเสร็จภายใน 24 ชั่วโมง'), chunk('b', 'งานติดตั้งนัดหมายล่วงหน้า')], semantic_enabled=False, model_name='unused')
        self.assertEqual(r.search('ซ่อมเสร็จภายในกี่ชั่วโมง')[0].source_id, 'a')
        self.assertEqual(r.search('zzzzunknown'), [])

    def test_semantic_tail_and_scope_with_fake_encoder(self):
        import numpy as np
        class Tokenizer:
            def encode(self, text, **kwargs): return list(text)
            def decode(self, tokens, **kwargs): return ''.join(tokens)
        class Encoder:
            tokenizer = Tokenizer()
            max_seq_length = 512
            def encode(self, texts, **kwargs):
                return np.array([[1., 0.] if 'TAIL' in t else [0., 1.] for t in texts])
        with patch('core.retrieval._load_embedding_model', return_value=Encoder()):
            index = SemanticIndex([chunk('a', 'x' * 1800 + 'TAIL'), chunk('b', 'TAIL')], 'fake')
        self.assertTrue(index.available)
        self.assertGreater(len(index._owners), 2)
        self.assertEqual(index.search('TAIL', allowed_source_ids={'a'})[0][0].source_id, 'a')
        self.assertEqual(index.search('TAIL', allowed_source_ids={'a'})[0][1], 1.)


class IngestionQualityTests(unittest.TestCase):
    def test_long_thai_line_bounded_and_tail_preserved(self):
        body = 'เงื่อนไขการทำงาน' * 700 + 'ข้อยกเว้นสำคัญ'
        doc = parse_markdown('a', '# Root\n## Rule\n' + body)
        self.assertTrue(all(len(c.text) <= 1600 for c in doc.chunks))
        self.assertIn('ข้อยกเว้นสำคัญ', doc.chunks[-1].text)
        self.assertEqual(doc.chunks[0].heading, 'Root > Rule')

    def test_long_table_repeats_column_headers(self):
        body = '| Code | Hours |\n| --- | --- |\n' + '\n'.join(f'| TASK{i} | 24 |' for i in range(300))
        doc = parse_markdown('table', body)
        self.assertGreater(len(doc.chunks), 1)
        self.assertTrue(all(c.text.startswith('| Code | Hours |') and len(c.text) <= 1600 for c in doc.chunks))
        self.assertIn('TASK299', doc.chunks[-1].text)

    def test_fenced_heading_does_not_split_section(self):
        doc = parse_markdown('a', '# Root\n```python\n# comment\nx=1\n```\nend')
        self.assertEqual(len(doc.chunks), 1)
        self.assertIn('# comment', doc.chunks[0].text)

    def test_crlf_and_external_consent_fail_closed(self):
        doc = parse_markdown('a', '\ufeff---\r\nexternal_llm_allowed: "false"\r\n---\r\n# A\r\ntext')
        self.assertFalse(doc.meta.external_llm_allowed)

    def test_content_change_changes_citation_id(self):
        self.assertNotEqual(parse_markdown('a', 'old').chunks[0].chunk_id, parse_markdown('a', 'new').chunks[0].chunk_id)

    def test_noop_sync_reuses_index_and_rename_changes_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'a.md'; file.write_text('# A\nrule')
            km = KnowledgeManager(LocalKnowledgeSource(tmp), semantic_enabled=False, embedding_model='unused')
            first = km.sync()
            self.assertIs(km.sync(), first)
            file.rename(Path(tmp) / 'b.md')
            self.assertNotEqual(km.sync().snapshot_id, first.snapshot_id)


class GroundingQualityTests(unittest.TestCase):
    def setUp(self):
        self.c = chunk('a', 'Threshold is 80% and payout is 1,000 บาท within 24 ชั่วโมง.')
        self.cite = Citation(self.c.chunk_id, self.c.text, 'a', 'a', 'Rule')

    def check(self, text):
        return validate_answer(claims=[Claim(text, (self.c.chunk_id,))], citations=[self.cite], evidence=[self.c], user_text='')

    def test_reject_empty_answer_contract(self):
        with self.assertRaises(ValueError):
            parse_model_json('{"status":"answer","answer_th":"Invented","claims":[]}', [self.c])

    def test_reject_wrong_numeric_units(self):
        self.assertFalse(self.check('Threshold is 90%.').valid)
        self.assertFalse(self.check('Deadline is 24 วัน.').valid)
        self.assertFalse(self.check('Payout is 900 บาท.').valid)

    def test_accept_normalized_thai_numbers(self):
        self.assertTrue(self.check('เกณฑ์ ๘๐ เปอร์เซ็นต์ จ่าย 1000 บาท ภายใน 24 hours').valid)

    def test_reject_forged_quote(self):
        result = validate_answer(claims=[Claim('Threshold is 80%.', (self.c.chunk_id,))], citations=[replace(self.cite, quote='invented')], evidence=[self.c], user_text='')
        self.assertFalse(result.valid)

    def test_full_evidence_not_truncated_at_900(self):
        c = replace(self.c, text='a' * 1000 + 'IMPORTANT EXCEPTION')
        parsed = parse_model_json(json.dumps({'status':'answer', 'answer_th':'draft', 'claims':[{'text':'Exception', 'citation_ids':[c.chunk_id]}]}), [c])
        self.assertIn('IMPORTANT EXCEPTION', parsed['citations'][0].quote)

    def test_malformed_claim_ids_rejected(self):
        with self.assertRaises(ValueError):
            parse_model_json('{"status":"answer","answer_th":"a","claims":[{"text":"a","citation_ids":"a:0"}]}', [self.c])


class ConversationQualityTests(unittest.TestCase):
    def test_new_short_topic_not_polluted(self):
        conv = Conversation('c', 'u', '', messages=[Message('m', 'user', 'Safety accident rules', '')])
        self.assertEqual(retrieval_query(conv, 'Install SLA'), ('Install SLA', ''))
        query, _ = retrieval_query(conv, 'แล้ว P2 ล่ะ')
        self.assertIn('Safety accident rules', query)

    def test_clarification_reply_keeps_original_question(self):
        conv = Conversation('c', 'u', '', messages=[Message('1','user','Deadline',''), Message('2','assistant','Clarify','', clarification_questions=['Install or repair?'])])
        query, context = retrieval_query(conv, 'repair')
        self.assertIn('Deadline', query)
        self.assertIn('Install or repair?', context)


class ServiceQualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        Path(self.tmp.name, 'a.md').write_text('---\nexternal_llm_allowed: true\n---\n# Policy\nSLA is 24 hours.\n')
        self.km = KnowledgeManager(LocalKnowledgeSource(self.tmp.name), semantic_enabled=False, embedding_model='unused')
        self.km.sync()
        self.identity = Identity('u','u')
        self.service = ChatService(knowledge=self.km, store=InMemoryConversationStore())
        self.cid = self.service.create_conversation(self.identity)

    def tearDown(self): self.tmp.cleanup()

    def request(self, key='1', text='SLA', selected=None):
        return ChatRequest(self.cid, key, text, selected)

    def provider(self, status='answer', text='SLA is 24 hours.', delay=0):
        cid = self.km.snapshot.chunks[0].chunk_id
        class Provider:
            enabled=True
            model_id='fake'
            calls=0
            def generate(self, **kwargs):
                self.calls += 1
                time.sleep(delay)
                return GenerationResult(json.dumps({'status':status, 'answer_th':'UNVALIDATED INVENTION', 'claims':[{'text':text, 'citation_ids':[cid]}], 'clarification_questions':[]}), 'fake', 'fake')
        p=Provider(); self.service.primary=p; return p

    def test_render_only_checked_claims(self):
        self.provider()
        response = self.service.send_message(self.identity, self.request())
        self.assertEqual(response.status, AnswerStatus.ANSWER)
        self.assertEqual(response.answer_th, 'SLA is 24 hours. [1]')

    def test_unsupported_number_fails_closed(self):
        self.provider(text='SLA is 48 hours.')
        response=self.service.send_message(self.identity,self.request())
        self.assertEqual(response.status, AnswerStatus.VALIDATION_FAILED)

    def test_concurrent_duplicate_calls_provider_once(self):
        p=self.provider(delay=.03)
        with ThreadPoolExecutor(max_workers=5) as pool:
            responses=list(pool.map(lambda _:self.service.send_message(self.identity,self.request()),range(5)))
        self.assertEqual(p.calls, 1)
        self.assertEqual(len({r.message_id for r in responses}), 1)
        self.assertTrue(all(r.snapshot_id == self.km.snapshot.snapshot_id for r in responses))
        self.assertEqual(len(self.service.get_conversation(self.identity,self.cid).messages),2)

    def test_empty_selection_returns_no_evidence(self):
        self.assertEqual(self.service.send_message(self.identity,self.request(selected=())).status, AnswerStatus.NOT_FOUND)

    def test_unrelated_conflict_does_not_block_selected_document(self):
        for name in ['x','y']:
            Path(self.tmp.name,name+'.md').write_text('---\ndoc_id: conflict\n---\n# Other\nSLA is 48 hours.')
        self.km.sync()
        r=self.service.send_message(self.identity,self.request(selected=('a.md',)))
        self.assertEqual(r.status, AnswerStatus.PROVIDER_UNAVAILABLE)

    def test_citation_survives_sync_and_other_conversation_cannot_resolve(self):
        self.provider()
        r=self.service.send_message(self.identity,self.request())
        citation=r.citations[0]
        Path(self.tmp.name,'a.md').write_text('# Policy\nSLA is 72 hours.')
        self.km.sync()
        self.assertEqual(self.service.get_citation(self.identity,self.cid,citation.chunk_id).quote, 'SLA is 24 hours.')
        other=self.service.create_conversation(self.identity)
        with self.assertRaises(KeyError): self.service.get_citation(self.identity,other,citation.chunk_id)

    def test_invalid_calendar_date_clarifies(self):
        self.assertEqual(self.service.send_message(self.identity,self.request(text='SLA 2026-02-31')).status, AnswerStatus.CLARIFY)

    def test_availability_failure_not_cached(self):
        self.service.send_message(self.identity,self.request())
        self.assertEqual(len(self.service._cache),0)
        p=self.provider()
        self.assertEqual(self.service.send_message(self.identity,self.request('2')).status,AnswerStatus.ANSWER)
        self.assertEqual(p.calls,1)

    def test_cache_date_and_context_isolation(self):
        args=dict(conversation_id='c',question='q',selected={'a'},snapshot_id='s')
        self.assertNotEqual(self.service._cache_key(**args,effective_date='2026-01-01'), self.service._cache_key(**args,effective_date='2026-02-01'))
        self.assertNotEqual(self.service._cache_key(**args,context='Install'), self.service._cache_key(**args,context='Repair'))

    def test_not_found_prose_cannot_smuggle_answer(self):
        self.provider(status='not_found')
        r=self.service.send_message(self.identity,self.request())
        self.assertNotIn('INVENTION',r.answer_th)
        self.assertFalse(r.claims)


class CostGuardTests(unittest.TestCase):
    def test_free_route_cannot_fallback_to_paid(self):
        p=OpenRouterProvider('key','openai/gpt-oss-120b:free',free_verified=True, fallback_models=('google/gemini-2.5-flash-lite','openai/gpt-oss-20b:free'))
        self.assertEqual(p.fallback_models, ('openai/gpt-oss-20b:free',))

    def test_paid_route_cannot_fallback_to_unverified_free(self):
        p=OpenRouterProvider('key','google/gemini-2.5-flash-lite',paid_allowed=True, fallback_models=('openai/gpt-oss-20b:free',))
        self.assertEqual(p.fallback_models, ())
