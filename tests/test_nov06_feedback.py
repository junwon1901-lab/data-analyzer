from copy import deepcopy
import numpy as np
import pytest
from llm_lab import analysis, session as memory


def test_epoch_selection_withholds_test_until_choice(monkeypatch):
    real_train, real_evaluate = analysis.train_network, analysis.evaluate_test_set
    candidates, evaluated = [], []
    def train(*args, **kwargs):
        assert kwargs['evaluate_test'] is False
        model = real_train(*args, **kwargs)
        assert '테스트' not in set(model.metrics['데이터'])
        candidates.append(model)
        return model
    def evaluate(model):
        assert len(candidates) == 2
        evaluated.append(model)
        return real_evaluate(model)
    monkeypatch.setattr(analysis, 'train_network', train)
    monkeypatch.setattr(analysis, 'evaluate_test_set', evaluate)
    run = analysis.compare_epochs()
    rows = run['epoch_candidates']
    selected = min(rows, key=lambda r:r['validation_RMSE'])
    assert run['selected_max_epochs'] == selected['max_epochs']
    assert len(evaluated) == 1
    assert all(r['best_epoch'] <= r['executed_epochs'] <= r['max_epochs'] for r in rows)
    assert np.array_equal(candidates[0].splits['검증'], candidates[1].splits['검증'])
    points = run['models'][0]['test_points']
    test_metric = next(m for m in run['models'][0]['metrics'] if m['split']=='테스트')
    assert test_metric['RMSE'] == pytest.approx(np.sqrt(np.mean([p['residual']**2 for p in points])))
    state = memory.new_session()
    state['runs'] = [run]
    assert 'epoch 상한 비교' in memory.report(state)


def test_error_audit_sorts_without_mutating_or_dividing_by_zero():
    run = {'selected':'neural','models':[{'kind':'neural','test_points':[
        {'row':1,'actual':100.,'predicted':50.},
        {'row':2,'actual':0.,'predicted':2.},
        {'row':3,'actual':80.,'predicted':140.}]}]}
    before = deepcopy(run)
    rows = analysis.worst_errors(run)
    assert [r['row'] for r in rows] == [3,1,2]
    assert rows[1]['predicted_actual_ratio'] == .5
    assert rows[2]['predicted_actual_ratio'] is None
    assert run == before


def test_new_prediction_keeps_selected_epoch_cap(monkeypatch):
    calls = []
    class Model:
        def predict(self, frame): return np.array([1.0])
    def fit(kind, degree=2, **kwargs):
        calls.append(kwargs['max_epochs'])
        return {'name':'test'}, Model()
    monkeypatch.setattr(analysis, 'fit', fit)
    analysis.predict(dict(zip(analysis.ALIASES,[.2,1e7,.5,.4,.7,4.,3.,0.])),
                     {'selected':'neural','degree':2,'selected_max_epochs':50})
    assert calls == [50]
