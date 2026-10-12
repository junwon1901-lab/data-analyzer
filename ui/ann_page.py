"""2차 과제 화면. 1차 과제의 회귀 화면 및 학습 결과와 독립적으로 동작."""
from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import numeric_columns
from models.neural_network import (
    ANNConfig, DEFAULT_FEATURES, DEFAULT_TARGET, prepare_data,
    split_indices, train_network, training_fingerprint, compare_architectures,
)


def render_ann(df, source_name="사용자 데이터"):
    st.subheader("인공신경망으로 새로운 데이터 예측하기")
    st.caption("연속적인 숫자를 예측하는 회귀용 MLP입니다. 분류·자동 시계열 예측 기능은 아닙니다.")
    # 예제 → 업로드 파일처럼 표 구조가 바뀌면 이전 그룹/변수 선택을 가져오지 않는다.
    context = (str(source_name), tuple(df.columns))
    if st.session_state.get("_ann_source_context") != context:
        for key in list(st.session_state):
            if key in {"ann_target", "ann_features", "ann_split", "ann_group", "ann_run", "ann_experiment"} or key.startswith("ann_filter_"):
                st.session_state.pop(key, None)
        st.session_state["_ann_source_context"] = context
    if not df.columns.is_unique:
        st.error("중복된 열 이름을 서로 다르게 수정하세요.")
        return
    with st.sidebar:
        st.header("2. 신경망 데이터 필터")
        categories = [c for c in df if not pd.api.types.is_numeric_dtype(df[c])
                      and 1 < df[c].nunique() <= 200]
        for column in categories:
            values = sorted(df[column].dropna().astype(str).unique())
            selected = st.multiselect(column, values, key=f"ann_filter_{column}",
                                      placeholder="전체 사용")
            if selected:
                df = df[df[column].astype(str).isin(selected)]
        st.caption(f"필터 적용 후 {len(df):,}행")
    if df.empty:
        st.session_state.pop("ann_run", None)
        st.warning("필터 결과가 없습니다. 선택을 바꿔 주세요.")
        return

    cols = numeric_columns(df)
    target = st.selectbox(
        "예측할 목표변수 (Y)", cols,
        index=cols.index(DEFAULT_TARGET) if DEFAULT_TARGET in cols else len(cols) - 1,
        key="ann_target",
    )
    available = [c for c in cols if c != target]
    defaults = [c for c in DEFAULT_FEATURES if c in available]
    features = st.multiselect(
        "학습에 사용할 입력변수 (X, 여러 개 선택 가능)", available,
        default=defaults or available[:min(3, len(available))], key="ann_features",
    )
    st.caption("정답으로부터 계산한 값, 측정 이후에만 알 수 있는 값, 단순 일련번호는 입력에서 제외하세요.")
    if target == DEFAULT_TARGET:
        st.info("목표값은 데이터에 기록된 정규화 저항 Rt/Δ × 10³입니다. 힘 단위 N의 저항값이 아닙니다.")

    left, right = st.columns(2)
    group_candidates = [c for c in df if c != target and c not in features
                        and 1 < df[c].nunique() < len(df)]
    default_group = "선형번호" if "선형번호" in group_candidates else None
    modes = ["그룹별 분리", "무작위 행 분리", "행 순서 분리"]
    with left:
        split_name = st.selectbox(
            "학습·검증·테스트 분리 방식", modes,
            index=0 if default_group and df[default_group].nunique() >= 5 else 1,
            key="ann_split",
        )
    with right:
        group_column = None
        if split_name == "그룹별 분리":
            if not group_candidates:
                st.warning("그룹으로 구분할 열이 없습니다. 다른 분리 방식을 선택하세요.")
                st.session_state.pop("ann_run", None)
                return
            group_column = st.selectbox(
                "같은 대상인지 구분하는 열", group_candidates,
                index=group_candidates.index(default_group) if default_group else 0,
                key="ann_group",
            )
            st.caption("같은 그룹의 모든 행은 학습·검증·테스트 중 한 곳에만 들어갑니다.")
        elif split_name == "무작위 행 분리":
            st.warning("같은 선박·사람의 반복 측정은 여러 세트에 섞일 수 있습니다. "
                       "새로운 대상 예측 능력을 평가하려면 그룹별 분리를 사용하세요.")
        else:
            st.warning("현재 행 순서를 유지해 앞부분은 학습, 중간은 검증, 뒷부분은 테스트로 씁니다. "
                       "시간 열로 자동 정렬하지 않으므로 먼저 시간순으로 정렬한 파일을 올리세요.")

    with st.expander("학습 설정", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            architecture = st.selectbox(
                "은닉층 구조", ["32", "32,16", "64,32", "64,32,16"], index=2, key="ann_arch",
                help="64,32는 은닉층 2개에 뉴런이 각각 64개, 32개라는 뜻입니다.",
            )
            activation = st.selectbox("활성화 함수", ["relu", "tanh"], key="ann_activation")
        with c2:
            rate = st.selectbox("학습률", [0.0001, 0.0005, 0.001, 0.005, 0.01],
                                index=2, key="ann_rate")
            epochs = st.slider("최대 학습 횟수 (epoch)", 50, 2000, 500, 50, key="ann_epochs")
        with c3:
            test_fraction = st.slider("테스트 비율", 0.1, 0.4, 0.2, 0.05, key="ann_test")
            seed = st.number_input("재현용 시드", 0, 1000000, 42, key="ann_seed")
        st.caption("나머지 데이터의 20%는 검증용으로 분리합니다. 검증 오차가 충분히 개선되지 "
                   "않는 상태가 40회 지속되면 멈추고, 검증 오차가 가장 낮았던 가중치를 복원합니다. "
                   "그룹별 분리에서는 비율이 행 수가 아닌 그룹 수 기준입니다.")
    config = ANNConfig(
        hidden_layers=tuple(int(n) for n in architecture.split(",")), activation=activation,
        learning_rate=rate, max_epochs=epochs, test_fraction=test_fraction,
        random_state=int(seed),
        split_mode=dict(zip(modes, ("group", "random", "ordered")))[split_name],
    )
    try:
        data = prepare_data(df, features, target, group_column)
        splits = split_indices(data, config)
        signature = training_fingerprint(df, features, target, group_column, config)
    except ValueError as error:
        st.session_state.pop("ann_run", None)
        st.warning(str(error))
        return
    if data.dropped_rows:
        st.warning(f"선택한 열의 결측값·무한대 또는 빈 그룹값이 있는 {data.dropped_rows}행을 제외했습니다.")
    st.write(f"입력 {len(features)}개 → 은닉층 {architecture} → 출력 1개")
    st.write(" · ".join(f"{name} {len(idx)}행" for name, idx in splits.items()))
    with st.expander("입력·출력 형태 이해하기", expanded=True):
        st.write(f"X: {len(data.X):,}행 × {len(features)}열, y: {len(data.y):,}개의 정답. "
                 f"한 행은 관측 1건이고, 입력층 노드 수는 입력변수의 수인 {len(features)}개입니다.")
        st.caption("선체 예제의 한 행은 선박 1척의 속도 조건 1건입니다. "
                   "1,018행이라고 입력 노드가 1,018개가 되는 것은 아닙니다. "
                   "이 앱은 여러 숫자 입력에서 숫자 목표 1개를 예측합니다.")
        st.dataframe(pd.DataFrame([
            {"세트": name, "X 형태 (행, 열)": f"({len(idx)}, {len(features)})",
             "y 형태": f"({len(idx)},)"} for name, idx in splits.items()
        ]), use_container_width=True, hide_index=True)
    with st.expander("선택 데이터와 출처"):
        st.write(f"파일/예제: {source_name}")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("아래 결과 CSV의 행 위치는 필터 적용 후, 결측값 제거 전 표를 기준으로 합니다.")
        if str(source_name).startswith("선체 저항"):
            st.markdown("기존 저장소에 포함된 Mendeley Data의 "
                        "[Sailboat Hull Resistance Dataset](https://data.mendeley.com/datasets/gw23dgzn6h/4)"
                        "를 사용합니다. Caprace & Faller Fahrnholz, CC BY 4.0. "
                        "원자료의 한글 열 이름 정리본이며, 원 저자의 예측 모델을 재현한 것은 아닙니다.")

    # 단일 모델 구조 선택은 비교 실험의 공통 조건이 아니므로 식별값에서 고정한다.
    experiment_signature = training_fingerprint(
        df, features, target, group_column, replace(config, hidden_layers=(1,)),
    )
    _render_architecture_experiment(data, config, source_name, experiment_signature)

    cached = st.session_state.get("ann_run")
    if cached and cached["signature"] != signature:
        st.session_state.pop("ann_run", None)
        st.info("데이터·변수·설정이 바뀌어 이전 결과를 숨겼습니다. 다시 학습해 주세요.")
    if st.button("인공신경망 학습 시작", type="primary", key="ann_train"):
        st.session_state.pop("ann_run", None)
        bar = st.progress(0.0, text="학습 준비 중")
        try:
            def progress(current, maximum):
                bar.progress(current / maximum, text=f"학습 중: {current} epoch")
            with st.spinner("학습·검증 데이터로 신경망을 학습하고 있습니다."):
                result = train_network(data, config, progress=progress)
            st.session_state["ann_run"] = {"signature": signature, "result": result}
        except (ValueError, FloatingPointError, OverflowError) as error:
            st.error(f"학습을 완료하지 못했습니다: {error}")
        finally:
            bar.empty()
    cached = st.session_state.get("ann_run")
    if not cached or cached["signature"] != signature:
        st.info("설정을 확인한 뒤 ‘인공신경망 학습 시작’을 누르세요.")
        return
    _render_results(cached["result"], source_name, signature)


def _apply_compared_architecture():
    experiment = st.session_state.get("ann_experiment")
    if experiment:
        selected = experiment["comparison"].selected_result.config.hidden_layers
        st.session_state["ann_arch"] = ",".join(map(str, selected))


def _render_architecture_experiment(data, config, source_name, signature):
    cached = st.session_state.get("ann_experiment")
    if cached and cached["signature"] != signature:
        st.session_state.pop("ann_experiment", None)
    with st.expander("은닉층 구조 비교 실험", expanded=True):
        st.write("32 / 32,16 / 64,32 / 64,32,16의 네 구조를 같은 데이터 분리, "
                 "활성화 함수, 학습률, 최대 epoch, 시드로 실험합니다.")
        st.caption("구조 선택은 검증 RMSE로 합니다. 이 비교 단계에서는 테스트를 평가하지 않습니다. "
                   "노드 수가 커지면 파라미터 수가 늘지만 예측이 항상 좋아지는 것은 아닙니다.")
        if st.button("구조 비교 실험 실행", key="ann_compare"):
            st.session_state.pop("ann_experiment", None)
            bar = st.progress(0.0, text="구조 비교 준비 중")
            try:
                def progress(current, maximum):
                    bar.progress(current / maximum, text=f"구조 {current}/{maximum} 완료")
                comparison = compare_architectures(data, config, progress=progress)
                st.session_state["ann_experiment"] = {
                    "signature": signature, "comparison": comparison,
                }
            except (ValueError, FloatingPointError, OverflowError) as error:
                st.error(f"비교를 완료하지 못했습니다: {error}")
            finally:
                bar.empty()
        cached = st.session_state.get("ann_experiment")
        if not cached or cached["signature"] != signature:
            return
        experiment = cached["comparison"]
        st.dataframe(experiment.table, use_container_width=True, hide_index=True)
        selected = experiment.selected_result
        label = ",".join(map(str, selected.config.hidden_layers))
        st.info(f"이 실험의 검증 RMSE 기준 선택: {label}. 구조를 적용한 뒤 "
                "‘인공신경망 학습 시작’을 눌러 테스트 성능을 확인하세요.")
        st.button("비교에서 선택한 구조 적용", key="ann_apply_arch",
                  on_click=_apply_compared_architecture)
        common = asdict(config)
        common.pop("hidden_layers")
        record = {
            "source": source_name, "features": data.X.columns.tolist(), "target": data.y.name,
            "group_column": data.group_column, "common_config": common,
            "selection_rule": "minimum validation RMSE; first candidate wins exact tie",
            "test_evaluated_during_comparison": False,
            "selected_hidden_layers": list(selected.config.hidden_layers),
            "results": json.loads(experiment.table.to_json(orient="records", force_ascii=False)),
            "split_positions_before_cleaning_1based": {
                name: (data.positions[idx] + 1).tolist() for name, idx in selected.splits.items()
            },
        }
        c1, c2 = st.columns(2)
        c1.download_button("구조 비교 CSV",
                           experiment.table.to_csv(index=False).encode("utf-8-sig"),
                           "ann_architecture_comparison.csv", "text/csv")
        c2.download_button("구조 비교 기록 JSON",
                           json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
                           "ann_architecture_comparison.json", "application/json")


def _render_learning_explanation(result):
    with st.expander("학습된 가중치·편향과 노드 계산 확인", expanded=True):
        summary = result.layer_summary()
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.write(f"학습된 가중치와 편향: 총 {summary['파라미터 수'].sum():,}개")
        st.caption("사용자는 입력·출력, 은닉층 구조, 활성화 함수와 학습률을 정합니다. "
                   "학습은 오차를 줄이도록 가중치와 편향의 값을 찾는 과정입니다.")
        st.latex(r"z_j=\sum_i \tilde{x}_i w_{ij}+b_j,\qquad h_j=\phi(z_j)")
        terms, trace = result.first_node_trace()
        st.write(f"실제 계산 예: 첫 학습 행(정리 전 위치 {trace['정리 전 행 위치(1부터)']})의 첫 은닉 노드")
        st.dataframe(terms, use_container_width=True, hide_index=True)
        st.write(f"가중합 {trace['가중합 (편향 전)']:.6g} + 편향 {trace['편향 b']:.6g} "
                 f"= z {trace['z = 가중합 + b']:.6g}; "
                 f"{result.config.activation}(z) = {trace['h = 활성화(z)']:.6g}")
        st.caption("표는 표시 자리에서 반올림됩니다. 이 값은 은닉 노드 하나의 출력이며, "
                   "최종 예측은 나머지 층과 목표값 역표준화를 거칩니다. "
                   "ReLU는 max(0,z), tanh는 쌍곡탄젠트입니다. "
                   "이 회귀 모델의 출력층은 항등함수입니다.")


def _render_results(result, source_name, signature):
    st.success(f"학습 완료 · 실행 {len(result.history)} epoch · 채택 {result.best_epoch} epoch")
    if result.stopped_early:
        st.caption("검증 오차의 개선이 멈춰 조기 종료했습니다.")
    else:
        st.caption("설정한 최대 학습 횟수에 도달했습니다. 검증 오차가 가장 낮았던 모델을 사용합니다.")
    _render_learning_explanation(result)
    test_metrics = result.metrics.set_index("데이터").loc["테스트"]
    c1, c2, c3 = st.columns(3)
    c1.metric("테스트 R²", "정의 불가" if pd.isna(test_metrics["R2"]) else f"{test_metrics['R2']:.4f}")
    c2.metric("테스트 RMSE", f"{test_metrics['RMSE']:.4g}")
    c3.metric("테스트 MAE", f"{test_metrics['MAE']:.4g}")
    st.caption("테스트는 가중치 학습·표준화·epoch 선택에 사용하지 않은 데이터입니다. "
               "R²는 정답률이 아니며 음수도 가능합니다. 실제 목표값이 일정한 세트에서는 정의하지 않습니다.")
    st.dataframe(result.metrics, use_container_width=True, hide_index=True)
    st.warning("테스트 결과를 보면서 설정을 반복 조정하면 테스트 세트에도 간접적으로 맞추게 됩니다. "
               "설정 선택은 검증 성능으로 하고, 최종 성능은 별도로 확보한 데이터에서 다시 확인하세요.")

    comparison = pd.DataFrame([
        {"모델": "인공신경망", **test_metrics[["R2", "RMSE", "MAE"]].to_dict()},
        {"모델": "선형회귀 기준 모델", **result.baseline_metrics},
    ])
    with st.expander("같은 입력·같은 분리로 선형회귀와 비교", expanded=True):
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.caption("기준 모델도 동일한 학습 행·입력변수로만 학습합니다. "
                   "여러 입력을 선택하면 다중선형회귀이며, 1차 과제 탭의 전체 데이터 적합도와는 다른 평가입니다.")
    history = result.history
    fig = go.Figure()
    for name in ("학습 MSE(표준화)", "검증 MSE(표준화)"):
        fig.add_trace(go.Scatter(x=history["epoch"], y=history[name], name=name))
    fig.add_vline(x=result.best_epoch, line_dash="dash")
    fig.update_layout(title="학습 곡선 · 점선은 채택한 epoch", xaxis_title="epoch",
                      yaxis_title="표준화한 Y의 평균제곱오차 (규제항 제외)", height=340)
    st.plotly_chart(fig, use_container_width=True)

    predictions = result.prediction_table()
    test = predictions[predictions["데이터 구분"] == "테스트"]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=test["실제값"], y=test["예측값"], mode="markers",
                                  name="테스트 데이터"))
        low, high = test[["실제값", "예측값"]].min().min(), test[["실제값", "예측값"]].max().max()
        fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="완전 일치",
                                 line=dict(dash="dash")))
        fig.update_layout(title="실제값과 예측값", xaxis_title="실제값", yaxis_title="예측값", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Scatter(x=test["예측값"], y=test["잔차(실제-예측)"], mode="markers"))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(title="테스트 잔차", xaxis_title="예측값", yaxis_title="실제값 − 예측값", height=350)
        st.plotly_chart(fig, use_container_width=True)
    if (result.data.y >= 0).all() and (test["예측값"] < 0).any():
        st.warning("일부 예측값이 음수입니다. 이 신경망에는 비음수 제약이 없으며 값을 임의로 0으로 보정하지 않습니다.")

    st.subheader("새로운 입력값으로 예측")
    train_X = result.data.X.iloc[result.splits["학습"]]
    with st.form(f"ann_predict_{signature[:12]}"):
        cells = st.columns(min(3, len(train_X.columns)))
        values = {}
        for i, name in enumerate(train_X.columns):
            values[name] = cells[i % len(cells)].number_input(
                name, value=float(train_X[name].median()), format="%.6g",
                help=f"학습 범위: {train_X[name].min():.6g} ~ {train_X[name].max():.6g}",
                key=f"ann_input_{signature[:12]}_{i}",
            )
        predict_clicked = st.form_submit_button("입력값 예측")
    if predict_clicked:
        try:
            new_frame = pd.DataFrame([values])
            estimate = float(result.predict(new_frame)[0])
            st.metric(f"예측 {result.data.y.name}", f"{estimate:.6g}")
            outside = [c for c in values if not train_X[c].min() <= values[c] <= train_X[c].max()]
            if outside:
                st.warning("학습 범위를 벗어난 입력(외삽): " + ", ".join(outside))
            if estimate < 0 and (result.data.y >= 0).all():
                st.warning("음수 예측입니다. 물리적으로 가능한 값인지 확인하세요.")
        except (ValueError, FloatingPointError, OverflowError) as error:
            st.error(str(error))
    st.caption("각 입력이 학습 범위 안이어도 그 조합까지 학습된 것은 아닙니다. "
               "새 선박 설계나 실무 의사결정에 바로 적용할 수 있는 검증 모델은 아닙니다.")

    c1, c2, c3 = st.columns(3)
    c1.download_button("예측 결과 CSV", predictions.to_csv(index=False).encode("utf-8-sig"),
                       "ann_predictions.csv", "text/csv")
    c2.download_button("학습 곡선 CSV", history.to_csv(index=False).encode("utf-8-sig"),
                       "ann_learning_curve.csv", "text/csv")
    report = {
        "source": source_name, "features": train_X.columns.tolist(), "target": result.data.y.name,
        "group_column": result.data.group_column, "config": asdict(result.config),
        "fingerprint": signature, "best_epoch": result.best_epoch,
        "input_shape": list(result.data.X.shape),
        "layer_parameters": result.layer_summary().to_dict(orient="records"),
        "epochs_run": len(history), "dropped_rows": result.data.dropped_rows,
        "metrics": json.loads(result.metrics.to_json(orient="records", force_ascii=False)),
        "comparison": json.loads(comparison.to_json(orient="records", force_ascii=False)),
        "split_positions_before_cleaning_1based": {
            name: (result.data.positions[idx] + 1).tolist() for name, idx in result.splits.items()
        },
    }
    c3.download_button("설정·평가 JSON", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                       "ann_report.json", "application/json")
