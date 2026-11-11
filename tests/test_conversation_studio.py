from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from llm_lab.input_data import parse_xy, command
from llm_lab.agent import handle
from llm_lab import session as memory
from llm_lab import small_sample

EXAMPLE = 'x: 1,2,3,4,5,6,7,8 y: 2.1,3.9,8.2,15.8,24.6,37.1,49.5,64.9 가장 잘 맞는 모델 찾아줘'
class NoLLM:
    def plan(self,*args):
        raise AssertionError('Explicit numeric requests must not need an LLM')

@pytest.mark.parametrize('text',['x: 1,2 y: 1','x: 1,2,3 y: 1,2,3','x: 1,2,3,4,5,1e400 y: 1,2,3,4,5,6'])
def test_invalid_short_data(text):
    with pytest.raises(ValueError): parse_xy(text)

def test_linear_prediction_and_equation_from_original_units():
    dataset = parse_xy('x: 1,2,3,4,5,6 y: 5,7,9,11,13,15')
    result, model = small_sample.fit('linear', 2, dataset)
    assert result['equation'] == 'y = 2·x +3'
    assert next(m for m in result['metrics'] if m['split']=='교차검증')['RMSE'] < 1e-10
    run = small_sample.run('linear', 2, dataset)
    assert small_sample.predict({'x':10},run,dataset)['value'] == pytest.approx(23)

def test_short_data_and_followup(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_SESSION_DIR',str(tmp_path))
    s=memory.new_session()
    output=handle(EXAMPLE,s,NoLLM())
    assert output['result']['evaluation_label']=='교차검증'
    assert len(output['result']['models'])==3
    assert all(r['split']!='테스트' for m in output['result']['models'] for r in m['metrics'])
    assert len({m['run_id'] for m in s['messages'] if 'run_id' in m})==1
    prediction=handle('x가 10일 때 예측해줘',s,NoLLM())
    assert np.isfinite(prediction['evidence']['prediction']['value'])
    handle('3차 다항식으로 해줘',s,NoLLM())
    assert s['runs'][-1]['degree']==3
    handle('신경망 학습해줘',s,NoLLM())
    assert s['runs'][-1]['models'][0]['history']
    handle('보고서 만들어줘',s,NoLLM())
    assert '교차검증' in s['messages'][-1]['report']
    assert '사용자 입력' in s['messages'][-1]['report']

def test_inline_results_and_fixed_guide(tmp_path,monkeypatch):
    monkeypatch.setenv('LLM_SESSION_DIR',str(tmp_path))
    with patch('llm_lab.cloud_client.LlamaCppClient', return_value=NoLLM()):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=90).run()
        assert not app.exception and not app.button
        app.chat_input[0].set_value(EXAMPLE).run()
        assert not app.exception and not app.error
        chart_count=len(app.get('plotly_chart'))
        app.chat_input[0].set_value('x가 10일 때 예측해줘').run()
        assert not app.exception and len(app.get('plotly_chart'))==chart_count
        assert any('항상 표시' in m.value for m in app.markdown)
        assert not app.button
