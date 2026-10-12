"""10/2 강의 보강 실험: 검증 RMSE로 네 구조 비교 → 선택 모델 테스트 평가.

저장소 루트에서 python scripts/compare_ann.py 실행. 외부 네트워크를 사용하지 않는다.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn

from models.neural_network import (
    ANNConfig, DEFAULT_FEATURES, DEFAULT_TARGET, prepare_data,
    compare_architectures, evaluate_test_set,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/ann_experiment")
    args = parser.parse_args()
    path = ROOT / "data/example_hull_resistance_full.csv"
    df = pd.read_csv(path)
    data = prepare_data(df, DEFAULT_FEATURES, DEFAULT_TARGET, "선형번호")
    config = ANNConfig(split_mode="group")
    comparison = compare_architectures(data, config)
    # 여기까지는 어느 후보도 테스트 예측/평가를 수행하지 않는다.
    result = evaluate_test_set(comparison.selected_result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.table.to_csv(args.output_dir / "architecture_comparison.csv", index=False, encoding="utf-8-sig")
    result.history.to_csv(args.output_dir / "selected_learning_curve.csv", index=False, encoding="utf-8-sig")
    result.prediction_table().to_csv(args.output_dir / "selected_predictions.csv", index=False, encoding="utf-8-sig")
    result.layer_summary().to_csv(args.output_dir / "layer_parameters.csv", index=False, encoding="utf-8-sig")
    terms, trace = result.first_node_trace()
    terms.to_csv(args.output_dir / "first_node_terms.csv", index=False, encoding="utf-8-sig")
    report = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str(path.relative_to(ROOT)),
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "features": list(DEFAULT_FEATURES), "target": DEFAULT_TARGET,
        "group_column": "선형번호", "rows": len(data.X), "dropped_rows": data.dropped_rows,
        "selected_config": asdict(result.config),
        "selection_rule": "minimum validation RMSE; first candidate wins exact tie",
        "test_evaluated_during_comparison": False,
        "prior_test_exposure": "10/2 기본 모델 평가에서 같은 테스트 세트를 이미 관찰함. 새 독립 검증으로 주장하지 않음.",
        "architecture_comparison": json.loads(comparison.table.to_json(orient="records", force_ascii=False)),
        "metrics": json.loads(result.metrics.to_json(orient="records", force_ascii=False)),
        "baseline_test": result.baseline_metrics,
        "epochs_run": len(result.history), "best_epoch": result.best_epoch,
        "groups_per_split": {k: len(set(data.groups[i])) for k, i in result.splits.items()},
        "split_positions_before_cleaning_1based": {
            k: (data.positions[i] + 1).tolist() for k, i in result.splits.items()
        },
        "first_node_trace": trace,
    }
    (args.output_dir / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(comparison.table.to_string(index=False))
    print(result.metrics.to_string(index=False))
    print(json.dumps({"selected": list(result.config.hidden_layers),
                      "first_node_trace": trace, "saved_to": str(args.output_dir)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
