from pathlib import Path
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def ann_app():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio(key="analysis_mode").set_value("인공신경망 (2차 과제)").run()
    assert not app.exception
    return app


def test_original_mode_and_sample():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.selectbox(key="example_regression").set_value("선체 저항 (Mendeley 범선 저항시험 데이터)").run()
    assert not app.exception
    assert len(app.tabs) == 4
    assert any(m.label == "R² (결정계수)" for m in app.metric)


def test_ann_train_predict_and_invalidate():
    app = ann_app()
    app.button(key="ann_train").click().run()
    assert not app.exception
    assert any(m.label == "테스트 R²" for m in app.metric)
    trained = app.session_state["ann_run"]["result"]
    # 입력값·다운로드 등의 화면 재실행으로 학습이 다시 발생해서는 안 된다.
    with patch("ui.ann_page.train_network", side_effect=AssertionError("unexpected retraining")):
        inputs = [n for n in app.number_input if (n.key or "").startswith("ann_input")]
        inputs[0].set_value(float(trained.data.X.iloc[trained.splits["학습"]].iloc[0, 0]))
        next(b for b in app.button if b.label == "입력값 예측").click().run()
        assert not app.exception
        assert any(m.label.startswith("예측 ") for m in app.metric)
    app.slider(key="ann_epochs").set_value(550).run()
    assert not app.exception
    assert not any(m.label == "테스트 R²" for m in app.metric)


def test_empty_feature_selection_and_small_filtered_group():
    app = ann_app()
    app.multiselect(key="ann_features").set_value([]).run()
    assert not app.exception
    assert any("1개 이상" in warning.value for warning in app.warning)
    app = ann_app()
    app.multiselect(key="ann_filter_선형번호").set_value(["Delft-1"]).run()
    assert not app.exception
    assert len(app.warning) > 0


@pytest.mark.parametrize("example", [
    "선체 저항 (Mendeley 범선 저항시험 데이터)", "선체 저항 전체 변수 (인공신경망용)"
])
def test_example_switches_without_crash(example):
    app = ann_app()
    app.selectbox(key="example_ann").set_value(example).run()
    assert not app.exception


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_user_upload_and_rename_columns(extension):
    app = ann_app()
    if not hasattr(app, "file_uploader"):
        pytest.skip("이 Streamlit 버전은 파일 업로드 UI 테스트를 제공하지 않습니다.")
    rng = np.random.default_rng(17)
    x = rng.normal(size=120)
    frame = pd.DataFrame({"길이": x, "힘": 2*x + 3})
    if extension == "csv":
        content = frame.to_csv(index=False).encode("utf-8")
        mime = "text/csv"
    else:
        buffer = BytesIO()
        frame.to_excel(buffer, index=False)
        content = buffer.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    app.file_uploader[0].upload(f"user_data.{extension}", content, mime).run()
    assert not app.exception
    assert app.selectbox(key="ann_target").value == "힘"
    app.button(key="ann_train").click().run()
    assert not app.exception
    assert any(m.label == "테스트 R²" for m in app.metric)


def test_compare_apply_and_invalidate_experiment():
    app = ann_app()
    app.slider(key="ann_epochs").set_value(50).run()
    app.button(key="ann_compare").click().run()
    assert not app.exception
    experiment = app.session_state["ann_experiment"]["comparison"]
    assert len(experiment.table) == 4
    assert experiment.selected_result.baseline_metrics is None
    assert not any(m.label == "테스트 R²" for m in app.metric)
    selected = ",".join(map(str, experiment.selected_result.config.hidden_layers))
    app.button(key="ann_apply_arch").click().run()
    assert not app.exception
    assert app.selectbox(key="ann_arch").value == selected
    app.button(key="ann_train").click().run()
    assert not app.exception
    assert any(m.label == "테스트 R²" for m in app.metric)
    app.selectbox(key="ann_rate").set_value(0.005).run()
    assert not app.exception
    assert "ann_experiment" not in app.session_state
    assert not any(m.label == "테스트 R²" for m in app.metric)
