from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow

from src.data_loader import encode_target, load_raw_data, split_train_test
from src.metrics import assert_metric_thresholds, compute_metrics
from src.model import load_model
from src.preprocessing import preprocess_inference

DATA_PATH = Path("southAfrican_credit_data.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_extra_trees_model.pkl"
EVAL_METRICS_PATH = ARTIFACT_DIR / "eval_metrics.json"

ENCODER_COLS = ["Sex", "Housing", "Saving accounts", "Checking account"]


def _load_encoders():
    return {
        col: joblib.load(ARTIFACT_DIR / f"{col}_label_encoder.pkl")
        for col in ENCODER_COLS
    }


def main() -> None:
    df = load_raw_data(DATA_PATH)
    _, X_test, _, y_test = split_train_test(df, target_col="Risk")
    y_test = encode_target(y_test)

    model = load_model(MODEL_PATH)
    encoders = _load_encoders()
    X_test_proc = preprocess_inference(X_test, encoders)

    y_pred = model.predict(X_test_proc)
    y_proba = (
        model.predict_proba(X_test_proc)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = compute_metrics(y_test, y_pred, y_proba)
    assert_metric_thresholds(metrics, min_f1=0.50, min_auc=0.60)

    EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk")
    with mlflow.start_run(run_name="extra_trees_eval"):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("quality_gate", "passed")
        mlflow.log_artifact(str(EVAL_METRICS_PATH))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
