"""
회귀분석 모듈
- 선형회귀:   y = a*x + b                (최소제곱법, np.polyfit)
- 다항회귀:   y = c_n*x^n + ... + c_0     (최소제곱법, np.polyfit)
- 지수회귀:   y = a * exp(b*x)            (비선형 최소제곱, scipy curve_fit)
- 거듭제곱:   y = a * x^b                 (비선형 최소제곱, scipy curve_fit)
- 로그회귀:   y = a * ln(x) + b           (ln(x)에 대한 선형회귀)

모든 함수는 같은 형태의 dict를 반환하므로, 나중에 인공신경망 모델도
같은 형식으로 반환하게 만들면 UI 코드를 거의 바꾸지 않고 추가할 수 있다.
"""
import numpy as np
from scipy.optimize import curve_fit


def calc_metrics(y, y_pred):
    """R², RMSE, MAE 계산"""
    y, y_pred = np.asarray(y, float), np.asarray(y_pred, float)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "RMSE": float(np.sqrt(np.mean((y - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y - y_pred))),
    }


def _result(name, equation, predict, x, y, params):
    return {
        "name": name,
        "equation": equation,
        "predict": predict,          # 새 x값을 넣으면 예측값을 돌려주는 함수
        "params": params,
        **calc_metrics(y, predict(x)),
    }


def linear_regression(x, y):
    a, b = np.polyfit(x, y, 1)
    return _result("선형", f"y = {a:.4g}·x + {b:.4g}",
                   lambda t: a * np.asarray(t) + b, x, y, {"a": a, "b": b})


def polynomial_regression(x, y, degree=2):
    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)
    terms = []
    for i, c in enumerate(coefs):
        p = degree - i
        terms.append(f"{c:.4g}" + ("" if p == 0 else "·x" if p == 1 else f"·x^{p}"))
    return _result(f"다항({degree}차)", "y = " + " + ".join(terms),
                   poly, x, y, {f"c{degree - i}": c for i, c in enumerate(coefs)})


def exponential_regression(x, y):
    if np.any(y <= 0):
        raise ValueError("지수회귀는 y가 모두 0보다 커야 합니다.")
    b0, ln_a0 = np.polyfit(x, np.log(y), 1)          # 초기값: ln(y) 선형회귀
    f = lambda t, a, b: a * np.exp(b * t)
    (a, b), _ = curve_fit(f, x, y, p0=[np.exp(ln_a0), b0], maxfev=10000)
    return _result("지수", f"y = {a:.4g}·e^({b:.4g}·x)",
                   lambda t: f(np.asarray(t), a, b), x, y, {"a": a, "b": b})


def power_regression(x, y):
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("거듭제곱회귀는 x, y가 모두 0보다 커야 합니다.")
    b0, ln_a0 = np.polyfit(np.log(x), np.log(y), 1)  # 초기값: log-log 선형회귀
    f = lambda t, a, b: a * np.power(t, b)
    (a, b), _ = curve_fit(f, x, y, p0=[np.exp(ln_a0), b0], maxfev=10000)
    return _result("거듭제곱", f"y = {a:.4g}·x^{b:.4g}",
                   lambda t: f(np.asarray(t, float), a, b), x, y, {"a": a, "b": b})


def logarithmic_regression(x, y):
    if np.any(x <= 0):
        raise ValueError("로그회귀는 x가 모두 0보다 커야 합니다.")
    a, b = np.polyfit(np.log(x), y, 1)
    return _result("로그", f"y = {a:.4g}·ln(x) + {b:.4g}",
                   lambda t: a * np.log(np.asarray(t, float)) + b, x, y, {"a": a, "b": b})


NONLINEAR_MODELS = {
    "다항": polynomial_regression,
    "지수": exponential_regression,
    "거듭제곱": power_regression,
    "로그": logarithmic_regression,
}
