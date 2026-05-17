import json
from pathlib import Path
import mlflow
import pandas as pd
from src.data_loader import encode_target, load_raw_data, split_train_test
from src.metrics import compute_metrics
from src.model import build_model, save_model, train_model
from src.preprocessing import preprocess_train, preprocess_inference, save_encoders
DATA_PATH = Path("southAfrican_credit_data.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_extra_trees_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "train_metrics.json"
def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_train_test(df, target_col="Risk")
    y_train = encode_target(y_train)
    y_test = encode_target(y_test)
    X_train_proc, encoders = preprocess_train(X_train)
    X_test_proc = preprocess_inference(X_test, encoders)
    model = build_model(random_state=42, n_estimators=300)
    model = train_model(model, X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = compute_metrics(y_test, y_pred, y_proba)
    save_model(model, MODEL_PATH)
    save_encoders(encoders, ARTIFACT_DIR)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit-risk")
    with mlflow.start_run(run_name="extra_trees_train"):
        mlflow.log_params({"model_type": "ExtraTreesClassifier", "n_estimators": 300, "random_state": 42})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
if __name__ == "__main__":
    main()
