"""Short user-supplied x/y series, with out-of-fold evaluation, never fake test scores."""
import warnings
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits
from core.regression import calc_metrics

NAMES = {'linear': '선형회귀', 'polynomial': '다항회귀', 'neural': '인공신경망'}


def estimator(kind, degree):
    if kind == 'linear':
        network = make_pipeline(StandardScaler(), LinearRegression())
    elif kind == 'polynomial':
        network = make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False), LinearRegression())
    else:
        network = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(16, 8), activation='tanh', learning_rate_init=.01,
            max_iter=500, n_iter_no_change=50, tol=1e-5, alpha=.001, random_state=42))
    return TransformedTargetRegressor(regressor=network, transformer=StandardScaler())


def equation(model, kind, x_name, y_name):
    if kind == 'neural':
        return f'{y_name} = MLP({x_name}) · 은닉층 16 → 8 · tanh'
    pipe = model.regressor_
    scale = pipe.steps[0][1]
    linear = pipe.steps[-1][1]
    normalized_x = Polynomial([-scale.mean_[0] / scale.scale_[0], 1 / scale.scale_[0]])
    poly = Polynomial([float(linear.intercept_), *linear.coef_])(normalized_x)
    poly = poly * model.transformer_.scale_[0] + model.transformer_.mean_[0]
    terms = []
    for power in range(len(poly.coef) - 1, -1, -1):
        coefficient = float(poly.coef[power])
        variable = '' if power == 0 else f'·{x_name}' + (f'^{power}' if power > 1 else '')
        terms.append(f'{coefficient:+.6g}{variable}')
    return f'{y_name} = ' + ' '.join(terms).lstrip('+')


def fit(kind, degree, dataset, *, evaluate=True):
    if kind not in NAMES or degree not in (2, 3):
        raise ValueError('지원하지 않는 모델 또는 차수입니다.')
    frame = pd.DataFrame(dataset['rows'], columns=dataset['columns'])
    X, y = frame.iloc[:, :-1], frame.iloc[:, -1]
    model = estimator(kind, degree)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        model.fit(X, y)
        fitted = model.predict(X)
        cv = KFold(n_splits=min(5, len(frame)), shuffle=True, random_state=42)
        predicted = cross_val_predict(clone(model), X, y, cv=cv) if evaluate else fitted
    if not np.isfinite(predicted).all():
        raise ValueError('예측값이 유한하지 않습니다. 숫자의 크기와 데이터 범위를 확인하세요.')
    name = NAMES[kind] + (f' ({degree}차)' if kind == 'polynomial' else '')
    metrics = [{'model': name, 'split': label, 'rows': len(frame),
                **calc_metrics(y, p), 'negative_predictions': int((p < 0).sum())}
               for label, p in [('학습', fitted), ('교차검증', predicted)]]
    points = [{'row': i + 1, 'actual': float(a), 'predicted': float(p),
               'residual': float(a-p), 'absolute_error': float(abs(a-p)),
               'predicted_actual_ratio': float(p/a) if abs(a) > 1e-12 else None,
               'hull': '사용자 입력', 'Fn': float(X.iloc[i, 0])}
              for i, (a, p) in enumerate(zip(y, predicted))]
    grid = pd.DataFrame({X.columns[0]: np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 180)})
    curve = [{'x': float(x), 'y': float(p)} for x, p in zip(grid.iloc[:, 0], model.predict(grid))]
    history = []
    if kind == 'neural':
        history = [{'epoch': i+1, '학습 손실(표준화)': float(loss)}
                   for i, loss in enumerate(model.regressor_.steps[-1][1].loss_curve_)]
    return {'kind': kind, 'name': name, 'degree': degree, 'metrics': metrics,
            'test_points': points, 'history': history, 'best_epoch': None,
            'equation': equation(model, kind, X.columns[0], y.name), 'curve': curve,
            'input_points': [{'x': float(x), 'y': float(a)} for x, a in zip(X.iloc[:, 0], y)],
            'config': {'standardize_train_only': True, 'cross_validation_folds': min(5, len(frame)),
                       'hidden_layers': [16, 8] if kind == 'neural' else [],
                       'activation': 'tanh' if kind == 'neural' else None,
                       'degree': degree if kind == 'polynomial' else 1},
            'data': {'source': '사용자 입력', 'features': list(X.columns), 'target': y.name, 'rows': len(frame)}}, model


def run(action, degree, dataset):
    if action == 'epoch_compare':
        raise ValueError('짧은 숫자 목록에서는 모델 3가지 비교해줘를 사용하세요.')
    kinds = list(NAMES) if action == 'compare' else [action]
    models = [fit(kind, degree, dataset)[0] for kind in kinds]
    selected = min(models, key=lambda m: next(r['RMSE'] for r in m['metrics'] if r['split'] == '교차검증'))
    return {'action': action, 'degree': degree, 'models': models, 'selected': selected['kind'],
            'custom_data': True, 'small_sample': True, 'evaluation_label': '교차검증',
            'selection_basis': '동일한 최대 5-fold 교차검증 RMSE (각 행을 학습에서 제외해 예측)',
            'note': '짧은 목록의 연습용 분석입니다. 곡선·식은 전체 입력 데이터에 맞춘 결과이며, '
                    '표의 교차검증 점수는 각 행을 제외한 모델의 예측으로 계산합니다. '
                    '모델 선택에도 같은 교차검증을 사용했으므로 독립 테스트 성능은 아닙니다. '
                    '표본이 적으면 점수와 외삽 예측이 불안정하며, 단위와 출처는 입력 자료를 확인해야 합니다.'}


def predict(values, previous, dataset):
    if not previous:
        raise ValueError('먼저 선형회귀 해줘, 비선형회귀 해줘 또는 신경망 학습해줘를 입력하세요.')
    name = dataset['columns'][0]
    if set(values) != {name}:
        raise ValueError(f'예측: {name}=10 형식으로 입력하세요.')
    result, model = fit(previous['selected'], previous.get('degree', 2), dataset, evaluate=False)
    value = float(model.predict(pd.DataFrame([values]))[0])
    if not np.isfinite(value):
        raise ValueError('유한하지 않은 예측값입니다. 입력값의 크기를 확인하세요.')
    x = [row[0] for row in dataset['rows']]
    return {'value': value, 'target': dataset['columns'][-1], 'model': result['name'],
            'inputs': values, 'outside_training_range': [name] if not min(x) <= values[name] <= max(x) else [],
            'warning': '전체 입력 목록으로 학습한 모델의 예측입니다. 적은 데이터의 외삽은 특히 불확실합니다.'}
