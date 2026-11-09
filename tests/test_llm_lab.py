import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from llm_lab.client import validate_plan, OllamaError
from llm_lab import analysis, session as memory
from llm_lab.agent import handle


def test_plan_rejects_code_and_nonfinite_values():
    good={'action':'linear','reply':'실행','degree':2,'values':{}}
    validate_plan(good)
    for bad in ({**good,'action':'exec'},{**good,'code':'print(1)'},
                {**good,'degree':99},{**good,'values':{'Fn':float('nan')}},
                {**good,'values':{'unknown':5}},{**good,'values':{'Fn':True}}):
        with pytest.raises(ValueError): validate_plan(bad)


def test_real_data_and_no_group_leakage():
    df,data=analysis.load_data()
    split=analysis.split_indices(data,analysis.config())
    assert df.shape==(1018,21)
    groups=[set(data.groups[split[k]]) for k in ['학습','검증','테스트']]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    assert sum(len(v) for v in split.values())==1018


def test_metrics_match_saved_predictions():
    result,_=analysis.fit('linear')
    points=result['test_points']
    rmse=np.sqrt(np.mean([(p['actual']-p['predicted'])**2 for p in points]))
    metrics=next(m for m in result['metrics'] if m['split']=='테스트')
    assert metrics['RMSE']==pytest.approx(rmse)
    assert len(points)==metrics['rows']


def test_session_persists_but_does_not_share_ids(tmp_path,monkeypatch):
    monkeypatch.setenv('LLM_SESSION_DIR',str(tmp_path))
    one,two=memory.new_session(),memory.new_session()
    one['messages']=[{'role':'user','content':'내 이름은 준원'}]
    memory.save(one)
    assert memory.load(one['id'])['messages']==one['messages']
    assert memory.load(two['id'])['messages']==[]
    with pytest.raises(ValueError): memory.load('../../etc/passwd')


def test_failed_llm_does_not_fake_success(tmp_path,monkeypatch):
    monkeypatch.setenv('LLM_SESSION_DIR',str(tmp_path))
    class Failure:
        def plan(self,*args): raise OllamaError('연결 실패')
    session=memory.new_session()
    before=deepcopy(session)
    with pytest.raises(OllamaError): handle('분석해줘',session,Failure())
    assert session==before


def test_prediction_requires_explicit_inputs():
    with pytest.raises(ValueError,match='빠졌습니다'):
        analysis.predict({'Fn':0.4},{'selected':'neural','degree':2})


def test_empty_report_does_not_claim_results():
    report=memory.report(memory.new_session())
    assert '분석 실행 기록이 없다' in report


def test_ui_initial_load(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('LLM_SESSION_DIR',str(tmp_path))
    at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'llm_chat.py'),default_timeout=30).run()
    assert not at.exception
    assert any('선체 저항' in t.value for t in at.title)
