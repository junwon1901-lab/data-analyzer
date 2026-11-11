"""기존 데이터·MLP를 재사용. 비교는 같은 입력, 같은 선박 그룹 분리로 수행."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits
from core.regression import calc_metrics
from models.neural_network import DEFAULT_FEATURES, DEFAULT_TARGET, ANNConfig, prepare_data, split_indices, train_network, evaluate_test_set
from llm_lab.client import ALIASES

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data/example_hull_resistance_full.csv'
NAMES = {'linear': '다변량 선형회귀', 'polynomial': '다항회귀', 'neural': '인공신경망'}
FEATURE_MAP = dict(zip(ALIASES, DEFAULT_FEATURES))


def config():
    return ANNConfig(hidden_layers=(64, 32, 16), split_mode='group',
                     target_transform='log', max_epochs=500, patience=40, random_state=42)


def load_data():
    if not DATA_PATH.exists():
        raise ValueError('기존 저장소의 data/example_hull_resistance_full.csv가 필요합니다. '
                         '추가 파일을 기존 data-analyzer 폴더에 넣으세요.')
    df = pd.read_csv(DATA_PATH)
    data = prepare_data(df, DEFAULT_FEATURES, DEFAULT_TARGET, '선형번호')
    return df, data


def describe():
    df, data = load_data()
    splits = split_indices(data, config())
    return {'rows': len(df), 'columns': len(df.columns), 'hulls': int(df['선형번호'].nunique()),
            'dropped_rows': data.dropped_rows, 'features': list(DEFAULT_FEATURES), 'target': DEFAULT_TARGET,
            'sha256': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            'split_rows': {k: len(v) for k,v in splits.items()},
            'split_hulls': {k: len(np.unique(data.groups[v])) for k,v in splits.items()},
            'source': 'https://data.mendeley.com/datasets/gw23dgzn6h/4', 'license': 'CC BY 4.0'}


def fit(kind, degree=2, *, max_epochs=500, trained_model=None, dataset=None):
    if kind not in NAMES or degree not in (2, 3):
        raise ValueError('지원하지 않는 모델 또는 차수입니다.')
    df, data = load_data() if dataset is None else custom_data(dataset)
    cfg = replace(config(), max_epochs=max_epochs) if dataset is None else replace(config(), max_epochs=max_epochs, split_mode="random", target_transform="standard")
    splits = split_indices(data, cfg)
    train = splits['학습']
    model = None
    history, best_epoch = [], None
    with threadpool_limits(limits=1):
        if kind == 'neural':
            model = trained_model or train_network(data, cfg, evaluate_test=False)
            evaluate_test_set(model)
            history = model.history.to_dict('records')
            best_epoch = model.best_epoch
        else:
            model = (make_pipeline(StandardScaler(), LinearRegression()) if kind == 'linear' else
                     make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False),
                                   StandardScaler(), Ridge(alpha=1.0)))
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                model.fit(data.X.iloc[train], data.y.iloc[train])
        metrics, points = [], []
        for label, idx in splits.items():
            pred = np.asarray(model.predict(data.X.iloc[idx]), float)
            actual = data.y.iloc[idx].to_numpy(float)
            if not np.isfinite(pred).all():
                raise ValueError('유한하지 않은 예측값입니다.')
            metrics.append({'model': NAMES[kind] + (f' ({degree}차)' if kind=='polynomial' else ''),
                            'split': label, 'rows': len(idx), **calc_metrics(actual, pred),
                            'negative_predictions': int((pred<0).sum())})
            if label == '테스트':
                points = [{'row': int(data.positions[i])+1, 'actual': float(y), 'predicted': float(p),
                           'residual': float(y-p), 'absolute_error': float(abs(y-p)),
                           'predicted_actual_ratio': float(p/y) if abs(y)>1e-12 else None,
                           'hull': str(df.iloc[data.positions[i]].get('선형번호', '사용자 데이터')),
                           'Fn': float(data.X.iloc[i,0])}
                          for i,y,p in zip(idx,actual,pred)]
    result = {'kind': kind, 'name': NAMES[kind], 'degree': degree, 'metrics': metrics,
              'test_points': points, 'history': history, 'best_epoch': best_epoch,
              'config': asdict(cfg) if kind=='neural' else
                        {'standardize_train_only': True, 'degree': 1 if kind=='linear' else degree,
                         'ridge_alpha': 1.0 if kind=='polynomial' else None},
              'data': describe() if dataset is None else {'source': '사용자 입력', 'features': list(data.X.columns), 'target': data.y.name, 'rows': len(df)}}
    return result, model


def compare_epochs():
    """11/6 피드백: 동일 조건으로 epoch 상한 50/500 비교, 검증으로만 선택."""
    _, data = load_data()
    rows, trained = [], []
    for cap in (50, 500):
        model = train_network(data, replace(config(), max_epochs=cap), evaluate_test=False)
        metrics = model.metrics.set_index('데이터')
        history = model.history
        rows.append({'max_epochs': cap, 'executed_epochs': len(history),
                     'best_epoch': model.best_epoch,
                     'train_RMSE': float(metrics.loc['학습', 'RMSE']),
                     'validation_RMSE': float(metrics.loc['검증', 'RMSE']),
                     'history': history.to_dict('records'),
                     'minimum_validation_MSE': float(history['검증 MSE(표준화)'].min()),
                     'last_validation_MSE': float(history.iloc[-1]['검증 MSE(표준화)'])})
        trained.append(model)
    index = min(range(len(rows)), key=lambda i: rows[i]['validation_RMSE'])
    cap = rows[index]['max_epochs']
    result, _ = fit('neural', max_epochs=cap, trained_model=trained[index])
    return {'action': 'epoch_compare', 'models': [result], 'selected': 'neural', 'degree': 2,
            'selected_max_epochs': cap, 'epoch_candidates': rows,
            'selection_basis': '검증 RMSE로 50/500 epoch 상한 선택 → 선택 모델만 테스트 평가',
            'note': '50은 교수님의 비교 제안이며 최적값이라는 뜻은 아닙니다. '
                    '두 후보 모두 patience=40과 최소 검증 손실 checkpoint 복원을 유지합니다. '
                    '따라서 상한·실제 실행·채택 epoch는 다를 수 있습니다. '
                    '이 테스트 분할은 이전 과제에서도 확인했으므로 새로운 독립 외부 검증은 아닙니다.'}


def worst_errors(run, limit=10):
    """선택 모델의 이미 계산한 테스트 결과를 절대오차 순으로 확인. 재학습 없음."""
    model = next(r for r in run['models'] if r['kind']==run['selected'])
    rows = []
    for point in model['test_points']:
        row = dict(point)
        row['absolute_error'] = abs(row['actual']-row['predicted'])
        row['predicted_actual_ratio'] = row['predicted']/row['actual'] if abs(row['actual'])>1e-12 else None
        rows.append(row)
    return sorted(rows, key=lambda p: p['absolute_error'], reverse=True)[:limit]


def run_analysis(action, degree=2, dataset=None):
    if dataset is not None and dataset.get('small_sample'):
        from llm_lab.small_sample import run
        return run(action, degree, dataset)
    if action == 'epoch_compare':
        if dataset is not None:
            raise ValueError('사용자 데이터에서는 선형회귀·비선형회귀·신경망·모델 3가지 비교를 사용하세요.')
        return compare_epochs()
    kinds = list(NAMES) if action=='compare' else [action]
    results = [fit(kind, degree, dataset=dataset)[0] for kind in kinds]
    # 검증 세트로 선택한다. 테스트 점수로 최적 모델을 선택하지 않는다.
    best = min(results, key=lambda r: next(m['RMSE'] for m in r['metrics'] if m['split']=='검증'))
    return {'action': action, 'models': results, 'selected': best['kind'], 'degree': degree,
            'selection_basis': '검증 RMSE (같은 선박 그룹 분리)' if dataset is None else '검증 RMSE (사용자 데이터 동일 무작위 분할, seed=42)',
            'custom_data': dataset is not None,
            'note': ('사용자 데이터는 독립 표본을 가정합니다. 시계열·반복 측정 자료에는 이 분할이 적절하지 않을 수 있습니다. ' if dataset is not None else '') + '비교를 반복하며 설정을 바꾸면 테스트 세트에 간접적으로 맞출 수 있습니다. '
                    '이 결과는 고정 설정의 수업용 비교이며 실제 설계용 검증이 아닙니다.'}


def predict(values, previous):
    if set(values) - set(ALIASES) or not all(np.isfinite(v) for v in values.values()):
        raise ValueError('입력 변수 이름과 유한한 숫자인지 확인하세요: ' + ', '.join(ALIASES))
    missing = [k for k in ALIASES if k not in values]
    if missing:
        raise ValueError('예측에 필요한 8개 입력 중 다음 값이 빠졌습니다: ' + ', '.join(missing) +
                         '. 입력하지 않은 값은 임의로 채우지 않습니다.')
    if not previous:
        raise ValueError('먼저 선형회귀, 다항회귀, 신경망 중 하나를 실행하세요.')
    _, data = load_data()
    row = pd.DataFrame([{FEATURE_MAP[k]: values[k] for k in ALIASES}], columns=DEFAULT_FEATURES)
    result, model = fit(previous['selected'], previous.get('degree', 2),
                        max_epochs=previous.get('selected_max_epochs', 500))
    value = float(model.predict(row)[0])
    train = split_indices(data, config())['학습']
    reference = data.X.iloc[train]
    outside = [alias for alias,col in FEATURE_MAP.items()
               if not reference[col].min() <= values[alias] <= reference[col].max()]
    if not np.isfinite(value):
        raise ValueError('입력 범위를 확인하세요. 예측값이 유한하지 않습니다.')
    return {'value': value, 'target': DEFAULT_TARGET, 'model': result['name'],
            'inputs': values, 'outside_training_range': outside,
            'warning': '변수별 범위 안이어도 실제 선박 조합의 타당성과 정확도가 보장되지는 않습니다.'}



def custom_data(dataset):
    df = pd.DataFrame(dataset['rows'], columns=dataset['columns'])
    return df, prepare_data(df, list(df.columns[:-1]), df.columns[-1])


def predict_custom(values, previous, dataset):
    if dataset.get('small_sample'):
        from llm_lab.small_sample import predict
        return predict(values, previous, dataset)
    if not previous:
        raise ValueError('먼저 선형회귀, 비선형회귀 또는 신경망 학습을 실행하세요.')
    _, data = custom_data(dataset)
    if set(values) != set(data.X.columns):
        raise ValueError('입력 변수 이름을 모두 정확하게 지정하세요: ' + ', '.join(data.X.columns))
    result, model = fit(previous['selected'], previous.get('degree', 2), dataset=dataset)
    row = pd.DataFrame([values], columns=data.X.columns)
    value = float(model.predict(row)[0])
    if not np.isfinite(value):
        raise ValueError('예측값이 유한하지 않습니다. 입력 범위를 확인하세요.')
    cfg = replace(config(), split_mode='random', target_transform='standard')
    reference = data.X.iloc[split_indices(data, cfg)['학습']]
    return {'value': value, 'target': data.y.name, 'model': result['name'], 'inputs': values,
            'outside_training_range': [k for k in values if not reference[k].min() <= values[k] <= reference[k].max()],
            'warning': '사용자 데이터로 학습한 예측값입니다. 단위와 적용 가능 범위는 원자료를 확인하세요.'}
