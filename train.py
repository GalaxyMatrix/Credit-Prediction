import json
from pathlib import Path

import mlflow
import yaml

from src.data_loader import encode_target, load_raw_data, split_train_test 
from src.metrics import compute_metrics
from src.model import build_model, save_model, train_model
from src.preprocessing import preprocess_train, preprocess_inference, save_encoders 



DATA_PATH = Path("southAfrican_credit_data.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_extra_trees_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "train_metrics.json"
PARAMS_PATH = Path("params.yaml")


def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()
    train_cfg = params["train"] 
    mlflow_cfg = params["mlflow"]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_train_test(
        df, 
        target_col = "Risk",
        test_size = train_cfg["test_size"],
        random_state = train_cfg["random_state"],
    )

    y_train = encode_target(y_train)
    y_test = encode_target(y_test) 

    X_train_proc, encoders = preprocess_train(X_train)
    X_test_proc = preprocess_inference(X_test, encoders)


    model = build_model(
        random_state = train_cfg["random_state"],
        n_estimators = train_cfg["n_estimators"],
    )
    model = train_model(model, X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)
    y_proba  = (
        model.predict_proba(X_test_proc)[:, 1]
        if hasattr(model, "predict_proba") else None
    )
    metrics = compute_metrics(y_test, y_pred, y_proba)

    save_model(model, MODEL_PATH)
    save_encoders(encoders, ARTIFACT_DIR) 

    with METRICS_PATH.open("w") as f:
        json.dump(metrics, f, indent=2)
    

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    with mlflow.start_run(run_name="extra_trees_train"):
        mlflow.set_tag("stage", "train")
        mlflow.set_tag("model_type", train_cfg["model_type"])


        mlflow.log_params(
            {
                "model_type": train_cfg["model_type"],
                "n_estimators": train_cfg["n_estimators"],
                "random_state": train_cfg["random_state"],
                "test_size": train_cfg["test_size"],
            }
        )


        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
        for encoder_path in ARTIFACT_DIR.glob("*_label_encoder.pkl"):
            mlflow.log_artifact(str(encoder_path))
        
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

    

