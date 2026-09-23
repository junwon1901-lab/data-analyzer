"""
데이터 분석 프로그램 (Data Analyzer)
어떤 CSV / Excel 파일이든 불러와서 X/Y축을 고르고 선형·비선형 회귀분석을 수행한다.
실행: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import load_file, excel_sheets, numeric_columns, clean_xy
from core.regression import linear_regression, NONLINEAR_MODELS

# ===== 프로그램 이름은 여기서만 바꾸면 됩니다 =====
APP_NAME = "데이터 분석 프로그램"
APP_ICON = "📊"
EXAMPLES = {
    "선체 저항 (Mendeley 범선 저항시험 데이터)": "data/example_hull_resistance.csv",
}

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")
st.title(f"{APP_ICON} {APP_NAME}")
st.caption("CSV·Excel 파일을 불러와 X/Y축을 고르면 선형·비선형 회귀분석 결과를 보여줍니다.")

# ---------------- 1. 데이터 불러오기 ----------------
with st.sidebar:
    st.header("1. 데이터 불러오기")
    uploaded = st.file_uploader("CSV / Excel / 텍스트 파일",
                                type=["csv", "xlsx", "xlsm", "xls", "txt", "data"])
    example = None
    if uploaded is None:
        example = st.selectbox("또는 예제 데이터 사용", ["사용 안 함", *EXAMPLES])

    sheet, header_row = 0, 0
    if uploaded is not None and uploaded.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        try:
            sheets = excel_sheets(uploaded)
            if len(sheets) > 1:
                sheet = st.selectbox("시트 선택", sheets)
        except Exception:
            pass
        header_row = st.number_input("열 이름이 있는 행 번호", min_value=1, value=1,
                                     help="표 위에 제목 등이 있으면 열 이름이 있는 행 번호를 입력하세요.") - 1

try:
    if uploaded is not None:
        df = load_file(uploaded, sheet=sheet, header_row=header_row)
    elif example and example != "사용 안 함":
        df = load_file(EXAMPLES[example])
    else:
        st.info("👈 왼쪽에서 CSV 또는 Excel 파일을 올리거나 예제 데이터를 선택하세요.")
        st.stop()
except Exception as e:
    st.error(f"파일을 읽지 못했습니다: {e}")
    st.stop()

num_cols = numeric_columns(df)
if len(num_cols) < 2:
    st.error("숫자로 된 열이 2개 이상 있어야 회귀분석을 할 수 있습니다. 파일이 어떻게 읽혔는지 확인하세요.")
    st.write(f"읽힌 크기: {df.shape[0]}행 × {df.shape[1]}열")
    st.dataframe(df.head(10), use_container_width=True)
    st.stop()

# ---------------- 2. 축 선택 ----------------
with st.sidebar:
    st.header("2. 축 선택")
    x_col = st.selectbox("X축 (독립변수)", num_cols, index=0)
    y_col = st.selectbox("Y축 (종속변수)", num_cols, index=min(1, len(num_cols) - 1))

# ---------------- 3. 데이터 거르기 ----------------
with st.sidebar:
    text_cols = [c for c in df.columns if c not in num_cols and 1 < df[c].nunique() <= 200]
    st.header("3. 데이터 거르기 (선택)")
    for col in text_cols:
        picked = st.multiselect(col, sorted(df[col].dropna().astype(str).unique()),
                                placeholder="전체")
        if picked:
            df = df[df[col].astype(str).isin(picked)]
    lo, hi = float(df[x_col].min()), float(df[x_col].max())
    if lo < hi:
        x_range = st.slider(f"{x_col} 범위", lo, hi, (lo, hi))
        df = df[df[x_col].between(*x_range)]
    st.caption(f"사용 중인 데이터: {len(df)}행")

if x_col == y_col:
    st.warning("X축과 Y축에 서로 다른 열을 선택하세요.")
    st.stop()
x, y = clean_xy(df, x_col, y_col)
if len(x) < 3 or np.ptp(x) == 0:
    st.warning("회귀분석을 하려면 서로 다른 X값을 가진 데이터가 3개 이상 필요합니다.")
    st.stop()
x_line = np.linspace(x.min(), x.max(), 300)


# ---------------- 그래프·결과 출력 함수 ----------------
def scatter_with_fits(fits, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="데이터",
                             marker=dict(size=7, opacity=0.7)))
    for r in fits:
        fig.add_trace(go.Scatter(x=x_line, y=r["predict"](x_line), mode="lines",
                                 name=f'{r["name"]} (R²={r["R2"]:.4f})'))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col,
                      height=480, legend=dict(orientation="h", y=-0.2))
    return fig


def residual_plot(r):
    fig = go.Figure(go.Scatter(x=x, y=y - r["predict"](x), mode="markers"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(title="잔차 그래프 (실제값 − 예측값)", xaxis_title=x_col,
                      yaxis_title="잔차", height=320)
    return fig


def show_result(r):
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (결정계수)", f'{r["R2"]:.4f}')
    c2.metric("RMSE", f'{r["RMSE"]:.4g}')
    c3.metric("MAE", f'{r["MAE"]:.4g}')
    st.code(r["equation"], language=None)


def predictor(r, key):
    """회귀식으로 새 X값의 Y를 예측"""
    with st.expander("🔮 이 회귀식으로 예측해보기"):
        new_x = st.number_input(f"{x_col} 값 입력", value=float(np.median(x)), key=key, format="%.6g")
        try:
            st.write(f"예측된 {y_col}: **{float(r['predict'](np.array([new_x]))[0]):.6g}**")
        except Exception:
            st.write("이 값에서는 예측할 수 없습니다.")


tab_data, tab_lin, tab_nonlin, tab_cmp = st.tabs(
    ["📋 데이터", "📈 선형회귀", "〰️ 비선형회귀", "🏆 모델 비교"])

with tab_data:
    st.subheader("데이터 미리보기")
    st.dataframe(df, use_container_width=True, height=300)
    st.subheader("기본 통계")
    st.dataframe(df[num_cols].describe().T, use_container_width=True)
    st.plotly_chart(scatter_with_fits([], f"{x_col} vs {y_col}"), use_container_width=True)

with tab_lin:
    lin = linear_regression(x, y)
    show_result(lin)
    st.plotly_chart(scatter_with_fits([lin], "선형회귀"), use_container_width=True)
    st.plotly_chart(residual_plot(lin), use_container_width=True)
    predictor(lin, "pred_lin")

with tab_nonlin:
    model_name = st.radio("모델 선택", list(NONLINEAR_MODELS), horizontal=True)
    kwargs = {"degree": st.slider("다항식 차수", 2, 6, 2)} if model_name == "다항" else {}
    try:
        nl = NONLINEAR_MODELS[model_name](x, y, **kwargs)
        show_result(nl)
        st.plotly_chart(scatter_with_fits([nl], f"{nl['name']} 회귀"), use_container_width=True)
        st.plotly_chart(residual_plot(nl), use_container_width=True)
        predictor(nl, "pred_nl")
    except Exception as e:
        st.warning(f"{model_name} 회귀를 이 데이터에 적용할 수 없습니다: {e}")

with tab_cmp:
    results, skipped = [lin], []
    for name, func in NONLINEAR_MODELS.items():
        try:
            results.append(func(x, y, degree=2) if name == "다항" else func(x, y))
        except Exception as e:
            skipped.append(f"{name}: {e}")
    table = pd.DataFrame([{k: r[k] for k in ("name", "equation", "R2", "RMSE", "MAE")}
                          for r in results]).sort_values("R2", ascending=False)
    table.columns = ["모델", "회귀식", "R²", "RMSE", "MAE"]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.success(f"R² 기준 가장 잘 맞는 모델: **{table.iloc[0]['모델']}**")
    for msg in skipped:
        st.caption(f"제외된 모델 — {msg}")
    st.plotly_chart(scatter_with_fits(results, "전체 모델 비교"), use_container_width=True)
    st.download_button("결과표 CSV 다운로드", table.to_csv(index=False).encode("utf-8-sig"),
                       file_name="regression_results.csv", mime="text/csv")
