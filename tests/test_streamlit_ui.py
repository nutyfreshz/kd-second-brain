import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, 'Install Streamlit for UI integration tests')
class StreamlitUITests(unittest.TestCase):
    def test_source_selection_chat_citations_and_new_chat(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'ALLOW_ANONYMOUS_LOCAL':'true', 'ENABLE_SEMANTIC':'false',
            'KNOWLEDGE_SOURCE':'local', 'LOCAL_KNOWLEDGE_DIR':tmp,
            'GEMINI_API_KEY':'', 'OPENROUTER_API_KEY':'',
        }):
            Path(tmp,'demo.md').write_text('# Demo\nSLA is 24 hours.')
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'ui/streamlit_app.py')).run()
            self.assertFalse(app.exception)
            app.sidebar.checkbox[0].uncheck().run()
            self.assertTrue(app.chat_input[0].disabled)
            app.sidebar.multiselect[0].select('demo.md').run()
            self.assertFalse(app.chat_input[0].disabled)
            app.chat_input[0].set_value('SLA').run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message),2)
            self.assertTrue(any('24 hours' in item.value for item in app.code))
            app.sidebar.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message),0)
