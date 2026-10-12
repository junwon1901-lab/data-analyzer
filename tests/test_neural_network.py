from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.regression import linear_regression, NONLINEAR_MODELS
from models.neural_network import (
    ANNConfig, DEFAULT_FEATURES, DEFAULT_TARGET, prepare_data,
    split_indices, train_network, training_fingerprint, compare_architectures,
    evaluate_test_set,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    x = rng.uniform(-2, 2, 150)
    z = rng.normal(size=150)
    return pd.DataFrame({"x": x, "z": z, "y": x**2 + 0.3*z,
                         "hull": np.repeat(np.arange(15), 10)})


@pytest.fixture
def config():
    return ANNConfig(hidden_layers=(16, 8), max_epochs=70, patience=15)


def test_group_sets_are_disjoint(frame, config):
    data = prepare_data(frame, ["x", "z"], "y", "hull")
    splits = split_indices(data, replace(config, split_mode="group"))
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(frame)))
    sets = [set(data.groups[idx]) for idx in splits.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))


def test_training_only_scalers_and_reproducibility(frame, config):
    data = prepare_data(frame, ["x", "z"], "y")
    result = train_network(data, config)
    indices = result.splits["학습"]
    np.testing.assert_allclose(result.x_scaler.mean_, data.X.iloc[indices].mean())
    np.testing.assert_allclose(result.y_scaler.mean_, [data.y.iloc[indices].mean()])
    repeated = train_network(data, config)
    np.testing.assert_allclose(result.predict(data.X), repeated.predict(data.X))
    assert result.best_epoch == int(result.history.loc[result.history["검증 MSE(표준화)"].idxmin(), "epoch"])
    assert len(result.prediction_table()) == len(frame)


def test_test_targets_do_not_influence_fitting(frame, config):
    data = prepare_data(frame, ["x", "z"], "y")
    original = train_network(data, config)
    changed = frame.copy()
    changed.loc[original.splits["테스트"], "y"] += 1e6
    altered = train_network(prepare_data(changed, ["x", "z"], "y"), config)
    assert original.best_epoch == altered.best_epoch
    np.testing.assert_allclose(original.predict(data.X), altered.predict(data.X))
    pd.testing.assert_frame_equal(original.history, altered.history)


def test_cleaning_counts_and_positions(frame):
    frame.loc[0, "x"] = np.nan
    frame.loc[1, "y"] = np.inf
    frame.loc[2, "hull"] = np.nan
    data = prepare_data(frame, ["x", "z"], "y", "hull")
    assert data.dropped_rows == 3
    assert data.positions[0] == 3
    assert np.isfinite(data.X.to_numpy()).all()


@pytest.mark.parametrize("features,target,group", [
    ([], "y", None), (["x", "x"], "y", None), (["x", "y"], "y", None),
    (["missing"], "y", None), (["x"], "y", "x"), (["x"], "y", "y"),
])
def test_invalid_selection_is_rejected(frame, features, target, group):
    with pytest.raises(ValueError):
        prepare_data(frame, features, target, group)


def test_insufficient_and_constant_data(frame, config):
    for invalid in (frame.head(10), frame.assign(y=1), frame.assign(x=1, z=1)):
        with pytest.raises(ValueError):
            prepare_data(invalid, ["x", "z"], "y")
    few_groups = prepare_data(frame.assign(hull=np.arange(len(frame)) % 3), ["x"], "y", "hull")
    with pytest.raises(ValueError):
        train_network(few_groups, replace(config, split_mode="group"))


def test_ordered_split_and_fingerprint(frame, config):
    data = prepare_data(frame, ["x", "z"], "y")
    splits = split_indices(data, replace(config, split_mode="ordered"))
    assert max(splits["학습"]) < min(splits["검증"]) < min(splits["테스트"])
    signature = training_fingerprint(frame, ["x", "z"], "y", None, config)
    assert signature != training_fingerprint(frame.assign(x=frame.x+1), ["x", "z"], "y", None, config)
    assert signature != training_fingerprint(frame, ["x", "z"], "y", None, replace(config, max_epochs=80))
    assert signature != training_fingerprint(frame, ["z", "x"], "y", None, config)


def test_prediction_schema_and_single_input(frame, config):
    result = train_network(prepare_data(frame, ["x"], "y"), config)
    pred = result.predict(frame[["z", "x"]].head(2))
    assert pred.shape == (2,) and np.isfinite(pred).all()
    for invalid in (pd.DataFrame({"z": [1]}), pd.DataFrame({"x": [np.nan]}), pd.DataFrame({"x": []})):
        with pytest.raises(ValueError):
            result.predict(invalid)


def test_shipped_hull_dataset():
    frame = pd.read_csv(ROOT / "data/example_hull_resistance_full.csv")
    assert frame.shape == (1018, 21)
    data = prepare_data(frame, DEFAULT_FEATURES, DEFAULT_TARGET, "선형번호")
    result = train_network(data, ANNConfig(split_mode="group"))
    assert result.metrics[["R2", "RMSE", "MAE"]].notna().all().all()
    assert result.metrics.set_index("데이터").loc["테스트", "R2"] > 0.5
    assert result.predict(data.X.head(5)).shape == (5,)


def test_original_regression_functions_remain_usable():
    x = np.linspace(0.1, 1, 30)
    y = 2*x + 3
    result = linear_regression(x, y)
    assert result["R2"] == pytest.approx(1)
    for function in NONLINEAR_MODELS.values():
        result = function(x, y)
        assert np.isfinite(result["predict"](x)).all()


def test_architecture_selection_does_not_evaluate_or_depend_on_test(frame, config, monkeypatch):
    from sklearn.neural_network import MLPRegressor
    data = prepare_data(frame, ["x", "z"], "y")
    config = replace(config, max_epochs=12)
    test = split_indices(data, config)["테스트"]
    original_predict = MLPRegressor.predict

    def guarded_predict(network, X):
        # 이 분할에서는 테스트만 30행이다. 후보 탐색 중 테스트 예측을 금지한다.
        assert len(X) != len(test), "test set was predicted during architecture selection"
        return original_predict(network, X)

    with monkeypatch.context() as context:
        context.setattr(MLPRegressor, "predict", guarded_predict)
        compared = compare_architectures(data, config)
        changed = frame.copy()
        changed.loc[test, "y"] += 1e6
        altered = compare_architectures(prepare_data(changed, ["x", "z"], "y"), config)
    pd.testing.assert_frame_equal(compared.table, altered.table)
    result = compared.selected_result
    assert result.baseline_metrics is None
    assert set(result.metrics["데이터"]) == {"학습", "검증"}
    assert "테스트" not in set(result.prediction_table()["데이터 구분"])
    chosen = compared.table.loc[compared.table["검증 기준 선택"]].iloc[0]
    assert chosen["검증 RMSE"] == compared.table["검증 RMSE"].min()
    evaluate_test_set(result)
    evaluate_test_set(result)
    assert len(result.metrics) == 3
    assert result.baseline_metrics is not None


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_displayed_weights_reconstruct_the_actual_prediction(frame, config, activation):
    result = train_network(prepare_data(frame, ["x", "z"], "y"), replace(config, activation=activation))
    row = result.data.X.iloc[[result.splits["학습"][0]]]
    state = result.x_scaler.transform(row.to_numpy(float))
    for i, (weights, bias) in enumerate(zip(result.network.coefs_, result.network.intercepts_)):
        state = state @ weights + bias
        if i < len(result.network.coefs_) - 1:
            state = np.maximum(0, state) if activation == "relu" else np.tanh(state)
        if i == 0:
            _, trace = result.first_node_trace()
            assert trace["h = 활성화(z)"] == pytest.approx(state[0, 0])
    np.testing.assert_allclose(result.y_scaler.inverse_transform(state).ravel(), result.predict(row))
    assert int(result.layer_summary()["파라미터 수"].sum()) == (2 + 1)*16 + (16 + 1)*8 + (8 + 1)


def test_log_target_positive_and_original_scale_metrics(frame, config):
    frame = frame.copy()
    frame['y'] = np.exp(frame['x'])
    data = prepare_data(frame, ['x', 'z'], 'y')
    result = train_network(data, replace(config, target_transform='log'))
    np.testing.assert_allclose(result.y_scaler.mean_, np.log(data.y.iloc[result.splits['학습']]).mean())
    predictions = result.predict(data.X)
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    idx = result.splits['테스트']
    expected = np.sqrt(np.mean((data.y.iloc[idx].to_numpy() - predictions[idx])**2))
    assert result.metrics.set_index('데이터').loc['테스트', 'RMSE'] == pytest.approx(expected)


def test_log_target_rejects_zero_and_negative(frame, config):
    for value in (0, -1):
        positive = frame.copy()
        positive['y'] = np.exp(positive['x'])
        positive.loc[0, 'y'] = value
        with pytest.raises(ValueError, match='0보다'):
            train_network(prepare_data(positive, ['x', 'z'], 'y'), replace(config, target_transform='log'))
