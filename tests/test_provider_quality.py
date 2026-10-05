import unittest
from unittest.mock import patch
from adapters.providers.gemini import GeminiProvider
from adapters.providers.openrouter import OpenRouterProvider
from adapters.providers.base import ProviderError


class ProviderQualityTests(unittest.TestCase):
    @patch('adapters.providers.gemini.post_json')
    def test_gemini_joins_output_skips_thought_and_uses_header(self, post):
        post.return_value={'candidates':[{'content':{'parts':[{'text':'private','thought':True},{'text':'{"a":'},{'text':'1}'}]},'finishReason':'STOP'}]}
        p=GeminiProvider('test-key','gemini-3.7-flash',free_verified=True)
        self.assertEqual(p.generate(system_prompt='s',user_prompt='u').text,'{"a":1}')
        self.assertNotIn('test-key',post.call_args.args[0])
        self.assertEqual(post.call_args.kwargs['headers']['x-goog-api-key'],'test-key')

    @patch('adapters.providers.gemini.post_json')
    def test_gemini_truncated_and_safety_responses_rejected(self, post):
        p=GeminiProvider('test-key','gemini-3.7-flash',free_verified=True)
        for reason in ['MAX_TOKENS','SAFETY']:
            post.return_value={'candidates':[{'finishReason':reason}]}
            with self.assertRaises(ProviderError) as cm: p.generate(system_prompt='s',user_prompt='u')
            self.assertEqual(cm.exception.safety_block,reason=='SAFETY')

    @patch('adapters.providers.openrouter.post_json')
    def test_openrouter_truncated_response_rejected(self, post):
        post.return_value={'choices':[{'message':{'content':'{"partial":'},'finish_reason':'length'}]}
        p=OpenRouterProvider('key','openai/gpt-oss-120b:free',free_verified=True)
        with self.assertRaises(ProviderError): p.generate(system_prompt='s',user_prompt='u')
