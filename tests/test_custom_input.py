import pytest
import numpy as np
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from llm_lab.input_data import parse_dataset
from llm_lab import analysis, session
from llm_lab.agent import handle
from pathlib import Path
CSV='x,y\n'+'\n'.join(f'{i},{2*i+3}' for i in range(40))
class NoLLM:
    def plan(self,*args):
        raise AssertionError('Explicit numerical commands must not use the LLM')
@pytest.mark.parametrize('kind',['linear','polynomial','neural'])
def test_custom_models(kind):
    data=parse_dataset(CSV)
    run=analysis.run_analysis(kind,dataset=data)
    prediction=analysis.predict_custom({'x':12},run,data)
    assert np.isfinite(prediction['value']) and prediction['target']=='y'
    if kind=='linear': assert prediction['value']==pytest.approx(27)
    assert sum(m['rows'] for m in run['models'][0]['metrics'])==40
@pytest.mark.parametrize('csv',['x,y\n1,2','x,x\n'+ '\n'.join('1,2' for _ in range(30)), CSV.replace('0,3','0,nan'),CSV.replace('0,3','0,3,5')])
def test_invalid(csv):
    with pytest.raises(ValueError): parse_dataset(csv)
def test_chat_workflow():
    with patch('llm_lab.cloud_client.LlamaCppClient',return_value=NoLLM()):
        a=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=60).run()
        for prompt in ['데이터 입력\n'+CSV,'선형회귀 해줘','예측: x=12','비선형회귀 해줘','신경망 학습해줘']:
            a.chat_input[0].set_value(prompt).run()
            assert not a.exception and not a.error and len(a.button)==0
            assert any('항상 표시' in m.value for m in a.markdown)
        assert len(a.session_state['hull_llm_session']['runs'])==3
        a.chat_input[0].set_value('기본 데이터 사용').run()
        assert not a.session_state['hull_llm_session']['runs']
        assert 'dataset' not in a.session_state['hull_llm_session']
def test_isolation_and_missing_values():
    a,b=session.new_session(),session.new_session()
    handle('데이터 입력\n'+CSV,a,NoLLM())
    assert 'dataset' not in b
    handle('선형회귀 해줘',a,NoLLM())
    with pytest.raises(ValueError): handle('예측: z=1',a,NoLLM())
    with pytest.raises(ValueError): handle('예측: x=nan',a,NoLLM())
