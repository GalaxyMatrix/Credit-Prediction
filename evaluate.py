from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import yaml

from src.data_loader import encode_target, load_raw_data, split_train_test
from src.metrics import assert_metric_thresholds, compute_metrics
from src.model import load_model
from src.preprocessing import preprocess_inference

DATA_PATH = Path("southAfrican_credit_data.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_extra_trees_model.pkl"
EVAL_METRICS_PATH = ARTIFACT_DIR / "eval_metrics.json"
PARAMS_PATH = Path("params.yaml")
ENCODER_COLS = ["Sex", "Housing", "Saving accounts", "Checking account"]


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def _load_encoders():
    return {
        col: joblib.load(ARTIFACT_DIR / f"{col}_label_encoder.pkl")
        for col in ENCODER_COLS
    }


def main() -> None: 
    params = load_params() 
    train_cfg = params["train"]
    eval_cfg = params["evaluate"]
    mlflow_cfg = params["mlflow"]


    df = load_raw_data(DATA_PATH)
    _, X_test, _, y_test = split_train_test(
        df,
        target_col="Risk",
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )


    y_test = encode_target(y_test)

    model = load_model(MODEL_PATH)
    encoders = _load_encoders() 
    X_test_proc = preprocess_inference(X_test, encoders)

    y_pred = model.predict(X_test_proc)
    y_prob = (
        model.predict_proba(X_test_proc)[:, 1]
        if hasattr(model, "predict_proba")
        else None 
    )

    metrics = compute_metrics(y_test, y_pred, y_prob)

    EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    with mlflow.start_run(run_name="extra_trees_eval"):
        mlflow.set_tag("stage", "evaluate")
        mlflow.log_params(
            {
                "min_f1": eval_cfg["min_f1"],
                "min_auc": eval_cfg["min_auc"],
                "test_size": train_cfg["test_size"],
                "random_state": train_cfg["random_state"],
            }
            
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(EVAL_METRICS_PATH))


        try:
            assert_metric_thresholds(metrics,
                min_f1=eval_cfg["min_f1"],
                min_auc=eval_cfg["min_auc"],
            )
            mlflow.set_tag("quality_gate", "passed")
        except ValueError:
            mlflow.set_tag("quality_gate", "failed")
            raise
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
        