"""기본 예제 검증 결과를 stdout으로 출력한다. 외부 네트워크를 사용하지 않는다."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from models.neural_network import ANNConfig, DEFAULT_FEATURES, DEFAULT_TARGET, prepare_data, train_network


def main():
    df = pd.read_csv(ROOT / "data/example_hull_resistance_full.csv")
    data = prepare_data(df, DEFAULT_FEATURES, DEFAULT_TARGET, "선형번호")
    result = train_network(data, ANNConfig(split_mode="group"))
    print(result.metrics.to_string(index=False))
    print(json.dumps({
        "rows": len(df), "hulls": df["선형번호"].nunique(),
        "epochs_run": len(result.history), "best_epoch": result.best_epoch,
        "groups_per_split": {k: len(set(data.groups[i])) for k, i in result.splits.items()},
        "linear_baseline_test": result.baseline_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
