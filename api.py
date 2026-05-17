from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# -----------------------------
# Prometheus metrics
# -----------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total predictions by outcome",
    ["outcome"],
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total prediction errors",
)

# -----------------------------
# Model + encoders
# -----------------------------
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_extra_trees_model.pkl"
ENCODER_COLS = ["Sex", "Housing", "Saving accounts", "Checking account"]

model = joblib.load(MODEL_PATH)
encoders = {
    col: joblib.load(ARTIFACT_DIR / f"{col}_label_encoder.pkl") for col in ENCODER_COLS
}

app = FastAPI(title="Credit Risk API", version="1.0.0")


class CreditRequest(BaseModel):
    Age: int = Field(ge=18, le=100)
    Sex: Literal["male", "female"]
    Job: int = Field(ge=0, le=3)
    Housing: Literal["own", "rent", "free"]
    Saving_accounts: Literal["little", "moderate", "rich", "quite rich"] = Field(
        alias="Saving accounts"
    )
    Checking_account: Literal["little", "moderate"] = Field(alias="Checking account")
    Credit_amount: int = Field(ge=0, alias="Credit amount")
    Duration: int = Field(ge=1, le=120)

    model_config = {"populate_by_name": True}


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start = time.perf_counter()
    status_code = 500
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint, status=str(status_code)
        ).inc()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(req: CreditRequest):
    try:
        payload = req.model_dump(by_alias=True)
        X = pd.DataFrame([payload])

        # encode categorical fields using training encoders
        for col in ENCODER_COLS:
            X[col] = encoders[col].transform(X[col].astype(str))

        pred = int(model.predict(X)[0])
        proba = (
            float(model.predict_proba(X)[0][1])
            if hasattr(model, "predict_proba")
            else None
        )

        outcome = "bad_risk" if pred == 0 else "good_risk"
        PREDICTION_COUNT.labels(outcome=outcome).inc()

        return {
            "prediction": pred,
            "prediction_label": "bad_risk" if pred == 0 else "good_risk",
            "probability_bad_risk": proba,
        }

    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")