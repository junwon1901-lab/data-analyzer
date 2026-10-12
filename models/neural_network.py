"""회귀용 다층 퍼셉트론: 분리 → 학습 데이터만 표준화 → 검증 조기 종료 → 테스트.

Streamlit에 의존하지 않아 단독 실행 및 자동 테스트가 가능하다.
테스트 세트는 가중치 갱신, 표준화 통계, epoch 선택에 사용하지 않는다.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from core.regression import calc_metrics

DEFAULT_FEATURES = (
    "프루드수_Fn", "레이놀즈수_Rn", "주형계수_Cp", "방형계수_Cb",
    "수선면계수_Cw", "길이폭비_L/B", "폭흘수비_B/T", "부심위치_LCB/L",
)
DEFAULT_TARGET = "전저항_Rt/Δ×10³"
ARCHITECTURES = ((32,), (32, 16), (64, 32), (64, 32, 16))


@dataclass(frozen=True)
class ANNConfig:
    hidden_layers: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    learning_rate: float = 0.001
    max_epochs: int = 500
    patience: int = 40
    min_delta: float = 0.00001
    alpha: float = 0.0001
    test_fraction: float = 0.2
    validation_fraction: float = 0.2  # 테스트 제외 나머지 중 검증 비율
    random_state: int = 42
    split_mode: str = "random"  # random, group, ordered

    def __post_init__(self):
        if not 1 <= len(self.hidden_layers) <= 3 or any(
            not isinstance(n, int) or not 1 <= n <= 128 for n in self.hidden_layers
        ):
            raise ValueError("은닉층은 1~3개, 각 층의 뉴런 수는 1~128개로 지정하세요.")
        if self.activation not in ("relu", "tanh"):
            raise ValueError("활성화 함수는 relu 또는 tanh를 선택하세요.")
        if not 1 <= self.max_epochs <= 2000 or not 1 <= self.patience <= 200:
            raise ValueError("최대 epoch는 1~2000, 조기 종료 대기는 1~200입니다.")
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.1:
            raise ValueError("학습률은 0보다 크고 0.1 이하여야 합니다.")
        if not 0.1 <= self.test_fraction <= 0.4 or not 0.1 <= self.validation_fraction <= 0.4:
            raise ValueError("테스트·검증 비율은 각각 0.1~0.4로 지정하세요.")
        if self.split_mode not in ("random", "group", "ordered"):
            raise ValueError("데이터 분리 방식을 확인하세요.")
        if not 0 <= self.random_state < 2**32:
            raise ValueError("시드는 0 이상 2의 32제곱 미만 정수여야 합니다.")
        if not np.isfinite([self.alpha, self.min_delta]).all() or min(self.alpha, self.min_delta) < 0:
            raise ValueError("규제와 개선 기준은 0 이상의 유한한 값이어야 합니다.")


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    groups: np.ndarray | None
    positions: np.ndarray
    dropped_rows: int
    group_column: str | None


def prepare_data(df, features, target, group_column=None):
    features = tuple(features)
    if not df.columns.is_unique:
        raise ValueError("중복된 열 이름을 서로 다르게 수정하세요.")
    if not features:
        raise ValueError("입력변수를 1개 이상 선택하세요.")
    if len(features) != len(set(features)):
        raise ValueError("입력변수를 중복 선택할 수 없습니다.")
    if target in features:
        raise ValueError("정답인 목표변수를 입력변수에 넣을 수 없습니다.")
    if group_column in (*features, target):
        raise ValueError("그룹 구분용 열은 입력변수·목표변수와 분리하세요.")
    needed = [*features, target] + ([group_column] if group_column else [])
    if any(c not in df.columns for c in needed):
        raise ValueError("선택한 열이 데이터에 없습니다.")
    if len(features) > 50 or len(df) > 50000:
        raise ValueError("실습용 한도는 50개 입력변수·50,000행입니다. 데이터를 줄여 주세요.")
    numeric = df[[*features, target]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    if group_column:
        valid &= df[group_column].notna().to_numpy()
    positions = np.flatnonzero(valid)
    cleaned = numeric.iloc[positions].reset_index(drop=True).astype(float)
    if len(cleaned) < 30:
        raise ValueError("결측값·무한대 제거 후 데이터가 30행 이상 필요합니다.")
    if cleaned[target].nunique() < 2:
        raise ValueError("목표값이 모두 같습니다. 서로 다른 목표값이 필요합니다.")
    if all(cleaned[c].nunique() < 2 for c in features):
        raise ValueError("입력값이 모두 일정합니다. 변하는 입력변수가 필요합니다.")
    groups = (
        pd.factorize(df.iloc[positions][group_column], sort=False)[0]
        if group_column else None
    )
    return PreparedData(
        X=cleaned[list(features)], y=cleaned[target], groups=groups,
        positions=positions, dropped_rows=len(df) - len(cleaned), group_column=group_column,
    )


def split_indices(data: PreparedData, config: ANNConfig):
    indices = np.arange(len(data.y))
    if config.split_mode == "group":
        if data.groups is None or len(np.unique(data.groups)) < 5:
            raise ValueError("그룹 분리는 서로 다른 그룹이 5개 이상 필요합니다.")
        outer = GroupShuffleSplit(
            n_splits=1, test_size=config.test_fraction, random_state=config.random_state,
        )
        pool, test = next(outer.split(indices, groups=data.groups))
        inner = GroupShuffleSplit(
            n_splits=1, test_size=config.validation_fraction, random_state=(config.random_state + 1) % 2**32,
        )
        train_rel, val_rel = next(inner.split(pool, groups=data.groups[pool]))
        train, val = pool[train_rel], pool[val_rel]
    elif config.split_mode == "ordered":
        n_test = int(np.ceil(len(indices) * config.test_fraction))
        n_val = int(np.ceil((len(indices) - n_test) * config.validation_fraction))
        train, val, test = indices[:-n_test - n_val], indices[-n_test - n_val:-n_test], indices[-n_test:]
    else:
        pool, test = train_test_split(
            indices, test_size=config.test_fraction, random_state=config.random_state,
        )
        train, val = train_test_split(
            pool, test_size=config.validation_fraction, random_state=(config.random_state + 1) % 2**32,
        )
    if len(train) < 10 or min(len(val), len(test)) < 2:
        raise ValueError("분리 후 학습 10행, 검증·테스트 각각 2행 이상 필요합니다.")
    if data.y.iloc[train].nunique() < 2:
        raise ValueError("학습 세트의 목표값이 모두 같습니다. 분리 설정을 바꿔 주세요.")
    if all(data.X.iloc[train][c].nunique() < 2 for c in data.X):
        raise ValueError("학습 세트 입력값이 모두 일정합니다. 분리 설정을 바꿔 주세요.")
    return {"학습": train, "검증": val, "테스트": test}


def training_fingerprint(df, features, target, group_column, config):
    """데이터/열/설정 변경 시 이전 학습 결과를 재사용하지 않도록 식별."""
    columns = list(dict.fromkeys([*features, target] + ([group_column] if group_column else [])))
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df[columns], index=True).values.tobytes())
    h.update(json.dumps(
        {"columns": columns, "features": list(features), "target": target,
         "group": group_column, "config": asdict(config)},
        ensure_ascii=False, sort_keys=True,
    ).encode())
    return h.hexdigest()


@dataclass
class ANNResult:
    data: PreparedData
    config: ANNConfig
    splits: dict[str, np.ndarray]
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    network: MLPRegressor
    baseline: LinearRegression | None
    history: pd.DataFrame
    metrics: pd.DataFrame
    baseline_metrics: dict | None
    best_epoch: int
    stopped_early: bool

    def predict(self, frame):
        columns = self.data.X.columns.tolist()
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("예측 입력은 중복 열이 없는 표 형태여야 합니다.")
        if any(c not in frame.columns for c in columns):
            raise ValueError("학습에 사용한 입력변수가 모두 필요합니다.")
        X = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("예측 입력에 빈칸·문자·무한대가 있습니다.")
        z = self.network.predict(self.x_scaler.transform(X))
        values = self.y_scaler.inverse_transform(z.reshape(-1, 1)).ravel()
        if not np.isfinite(values).all():
            raise ValueError("예측값이 유한하지 않습니다. 입력값과 학습 설정을 확인하세요.")
        return values

    def prediction_table(self):
        frames = []
        for label, indices in self.splits.items():
            # 구조 선택만 한 결과에서는 보류 중인 테스트 세트를 예측하지 않는다.
            if label == "테스트" and self.baseline_metrics is None:
                continue
            X = self.data.X.iloc[indices]
            pred = self.predict(X)
            frames.append(pd.DataFrame({
                "정리 전 행 위치(1부터)": self.data.positions[indices] + 1,
                "데이터 구분": label, "실제값": self.data.y.iloc[indices].to_numpy(),
                "예측값": pred, "잔차(실제-예측)": self.data.y.iloc[indices].to_numpy() - pred,
            }))
        return pd.concat(frames, ignore_index=True)

    def layer_summary(self):
        """그림의 연결선·노드와 실제 학습 파라미터 배열을 대응한다."""
        rows = []
        for i, (weights, biases) in enumerate(zip(self.network.coefs_, self.network.intercepts_)):
            output = i == len(self.network.coefs_) - 1
            rows.append({
                "연결": f"{'입력' if i == 0 else f'은닉 {i}'} → {'출력' if output else f'은닉 {i+1}'}",
                "가중치 배열 (앞층, 다음층)": f"{weights.shape[0]} × {weights.shape[1]}",
                "가중치 수": weights.size, "편향 수": biases.size,
                "파라미터 수": weights.size + biases.size,
                "다음층 활성화": "identity (항등)" if output else self.config.activation,
            })
        return pd.DataFrame(rows)

    def first_node_trace(self):
        """첫 학습 행의 첫 은닉 노드: 표준화 입력 × 학습된 가중치 + 편향."""
        index = int(self.splits["학습"][0])
        raw = self.data.X.iloc[index].to_numpy(float)
        scaled = self.x_scaler.transform(raw.reshape(1, -1))[0]
        weights = self.network.coefs_[0][:, 0]
        bias = float(self.network.intercepts_[0][0])
        terms = pd.DataFrame({
            "입력변수": self.data.X.columns, "원래 입력 x": raw,
            "표준화 입력": scaled, "학습된 가중치 w": weights,
            "입력 × 가중치": scaled * weights,
        })
        z = float(scaled @ weights + bias)
        h = max(0.0, z) if self.config.activation == "relu" else float(np.tanh(z))
        return terms, {
            "정리 전 행 위치(1부터)": int(self.data.positions[index]) + 1,
            "가중합 (편향 전)": float(scaled @ weights), "편향 b": bias,
            "z = 가중합 + b": z, "h = 활성화(z)": h,
        }


def train_network(data: PreparedData, config: ANNConfig, progress: Callable | None = None,
                  *, evaluate_test: bool = True):
    splits = split_indices(data, config)
    train, val, test = (splits[name] for name in ("학습", "검증", "테스트"))
    X = data.X.to_numpy(float)
    y = data.y.to_numpy(float)
    x_scaler = StandardScaler().fit(X[train])
    y_scaler = StandardScaler().fit(y[train, None])
    X_train = x_scaler.transform(X[train])
    X_val = x_scaler.transform(X[val])
    y_train = y_scaler.transform(y[train, None]).ravel()
    y_val = y_scaler.transform(y[val, None]).ravel()
    network = MLPRegressor(
        hidden_layer_sizes=config.hidden_layers, activation=config.activation,
        solver="adam", alpha=config.alpha, learning_rate_init=config.learning_rate,
        batch_size=min(64, len(train)), max_iter=1, early_stopping=False,
        random_state=np.random.RandomState(config.random_state), shuffle=True,
    )
    history = []
    best_loss, patience_reference = float("inf"), float("inf")
    best_network, best_epoch, waiting = None, 0, 0
    # 작은 표 데이터에서는 스레드 과다 사용이 더 느리고 공유 CPU를 소모한다.
    with threadpool_limits(limits=1):
        for epoch in range(1, config.max_epochs + 1):
            network.partial_fit(X_train, y_train)
            train_loss = float(np.mean((y_train - network.predict(X_train)) ** 2))
            val_loss = float(np.mean((y_val - network.predict(X_val)) ** 2))
            if not np.isfinite([train_loss, val_loss]).all():
                raise ValueError("학습이 발산했습니다. 학습률을 낮추고 극단적인 데이터를 확인하세요.")
            history.append({"epoch": epoch, "학습 MSE(표준화)": train_loss, "검증 MSE(표준화)": val_loss})
            # 최소 검증 오차 모델은 항상 보존한다. 대기 카운터는 min_delta 이상 개선 시 초기화.
            if val_loss < best_loss:
                best_loss, best_epoch, best_network = val_loss, epoch, deepcopy(network)
            if val_loss < patience_reference - config.min_delta:
                patience_reference, waiting = val_loss, 0
            else:
                waiting += 1
            if progress is not None and (epoch == 1 or epoch % 10 == 0):
                progress(epoch, config.max_epochs)
            if waiting >= config.patience:
                break
        metric_rows = []
        for name, indices in splits.items():
            if name == "테스트":
                continue
            pred_z = best_network.predict(x_scaler.transform(X[indices]))
            pred = y_scaler.inverse_transform(pred_z[:, None]).ravel()
            metric_rows.append({"데이터": name, "행 수": len(indices), **calc_metrics(y[indices], pred)})
    result = ANNResult(
        data=data, config=config, splits=splits, x_scaler=x_scaler, y_scaler=y_scaler,
        network=best_network, baseline=None, history=pd.DataFrame(history),
        metrics=pd.DataFrame(metric_rows), baseline_metrics=None,
        best_epoch=best_epoch, stopped_early=len(history) < config.max_epochs,
    )
    if evaluate_test:
        evaluate_test_set(result)
    if progress is not None:
        progress(len(history), len(history))
    return result


def evaluate_test_set(result: ANNResult):
    """구조·epoch를 선택한 다음 테스트 평가. 같은 결과에 중복 행을 만들지 않는다."""
    if result.baseline_metrics is not None:
        return result
    train, test = result.splits["학습"], result.splits["테스트"]
    X = result.data.X.to_numpy(float)
    y = result.data.y.to_numpy(float)
    with threadpool_limits(limits=1):
        baseline = LinearRegression().fit(result.x_scaler.transform(X[train]), y[train])
        prediction = result.predict(result.data.X.iloc[test])
        baseline_prediction = baseline.predict(result.x_scaler.transform(X[test]))
    result.metrics = pd.concat([result.metrics, pd.DataFrame([{
        "데이터": "테스트", "행 수": len(test), **calc_metrics(y[test], prediction),
    }])], ignore_index=True)
    result.baseline = baseline
    result.baseline_metrics = calc_metrics(y[test], baseline_prediction)
    return result


@dataclass
class ArchitectureComparison:
    table: pd.DataFrame
    selected_result: ANNResult


def compare_architectures(data: PreparedData, config: ANNConfig,
                          progress: Callable | None = None):
    """같은 분할·설정에서 네 구조를 실험. 선택에는 검증 RMSE만 사용한다."""
    rows, selected_result, best_rmse = [], None, float("inf")
    for i, architecture in enumerate(ARCHITECTURES):
        candidate = replace(config, hidden_layers=architecture)
        result = train_network(data, candidate, evaluate_test=False)
        metrics = result.metrics.set_index("데이터")
        rmse = float(metrics.loc["검증", "RMSE"])
        rows.append({
            "은닉층 구조": ",".join(map(str, architecture)), "은닉층 수": len(architecture),
            "파라미터 수": int(result.layer_summary()["파라미터 수"].sum()),
            "학습 RMSE": float(metrics.loc["학습", "RMSE"]), "검증 RMSE": rmse,
            "검증 R2": float(metrics.loc["검증", "R2"]),
            "실행 epoch": len(result.history), "채택 epoch": result.best_epoch,
        })
        if rmse < best_rmse:
            best_rmse, selected_result = rmse, result
        if progress:
            progress(i + 1, len(ARCHITECTURES))
    table = pd.DataFrame(rows)
    table["검증 기준 선택"] = table.index == table["검증 RMSE"].idxmin()
    return ArchitectureComparison(table=table, selected_result=selected_result)
