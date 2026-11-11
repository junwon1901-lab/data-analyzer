from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import pytest
ROOT=Path(__file__).resolve().parents[1]
class Client:
    def plan(self,text,*args):
        action={'선형회귀 해줘':'linear','비선형회귀 해줘':'polynomial','신경망 학습해줘':'neural','보고서 만들어줘':'report'}[text]
        return {'action':action,'reply':'','degree':2,'values':{}},{'model':'test double','seconds':0,'raw_json':'{}'}
@pytest.mark.parametrize('prompt',['선형회귀 해줘','비선형회귀 해줘','신경망 학습해줘'])
def test_chat_analysis(prompt):
    with patch('llm_lab.cloud_client.LlamaCppClient',return_value=Client()):
        a=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
        assert not a.exception and len(a.button)==0
        a.chat_input[0].set_value(prompt).run()
        assert not a.exception and not a.error
        assert len(a.get('plotly_chart'))>0
        a.chat_input[0].set_value('보고서 만들어줘').run()
        assert not a.exception
        a.chat_input[0].set_value('새 대화 시작').run()
        assert not a.exception and not a.session_state['hull_llm_session']['runs']
