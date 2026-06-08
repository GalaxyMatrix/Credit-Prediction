# 💳 Credit Risk Prediction — End-to-End MLOps

> A production-style machine learning system that predicts loan default risk — served as a **live web app backed by a containerized REST API**, with experiment tracking, automated quality gates, monitoring, and CI/CD.

<p align="center">
  <a href="https://matrix-credit-prediction.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://credit-prediction-em5m.onrender.com/docs">
    <img src="https://img.shields.io/badge/🔌_Live_API-Swagger_Docs-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="API Docs">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/MLflow-3.12-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://github.com/GalaxyMatrix/Credit-Prediction/actions/workflows/ci-cd.yml/badge.svg" alt="CI">
</p>

---

## 🎯 Try it now

| | Link |
|---|---|
| **Web app (UI)** | **https://matrix-credit-prediction.streamlit.app/** |
| **REST API (Swagger)** | **https://credit-prediction-em5m.onrender.com/docs** |

> ⏳ **Heads up:** the API runs on a free tier that sleeps after ~15 min of inactivity. The **first** request may take 30–60s to wake the service — after that it's instant. If a prediction seems slow, just retry once.

---

## ✨ Why this project stands out

This is **not a notebook** — it's a full ML lifecycle wired together the way real teams ship models:

- 🧱 **Modular pipeline** — research notebook refactored into testable `src/` modules (`data_loader`, `preprocessing`, `model`, `metrics`).
- 🔀 **Client/server architecture** — a thin **Streamlit** UI calls a **FastAPI** model service; each deploys, scales, and is monitored independently.
- 🧪 **Experiment tracking** — every train/eval run logged to **MLflow** (params, metrics, artifacts).
- 🚦 **Automated quality gate** — `evaluate.py` **fails the build** if F1 / ROC-AUC drop below thresholds, so a regression can never ship.
- 📊 **Monitoring built in** — FastAPI exposes **Prometheus** metrics (request counts, latency, prediction outcomes, errors).
- ♻️ **Reproducibility** — config-driven via `params.yaml`, a **DVC** pipeline, and exactly pinned dependencies.
- 🐳 **Containerized** — production `Dockerfile` (non-root user + healthcheck) and `docker-compose` for API + Prometheus.
- ✅ **CI/CD** — GitHub Actions runs **train → evaluate → test** on every push and uploads model artifacts.
- 🎯 **No training/serving skew** — a single fitted `ColumnTransformer` (`preprocessor.pkl`) is the one source of truth used in both training and inference.

---

## 🏗️ Architecture

```text
                 ┌──────────────────────────┐
                 │   Streamlit UI (client)   │   matrix-credit-prediction.streamlit.app
                 │        app.py             │
                 └────────────┬─────────────┘
                              │  POST /predict  (JSON over HTTPS)
                              ▼
                 ┌──────────────────────────┐
                 │   FastAPI service (API)   │   credit-prediction-em5m.onrender.com
                 │        api.py             │   /predict  /health  /metrics  /docs
                 │  ┌────────────────────┐   │
                 │  │ preprocessor.pkl   │   │   ColumnTransformer (ordinal + one-hot)
                 │  │ model.pkl          │   │   Extra Trees Classifier
                 │  └────────────────────┘   │
                 │   Prometheus /metrics     │
                 └──────────────────────────┘
```

**Training & ops flow:**

```text
southAfrican_credit_data.csv
        │
        ▼  src/data_loader.py      load + split + encode target (good→1, bad→0)
        ▼  src/preprocessing.py    ColumnTransformer: ordinal accounts, one-hot nominals
        ▼  train.py                fit Extra Trees → model.pkl + preprocessor.pkl  (logged to MLflow)
        ▼  evaluate.py             quality gates (F1, ROC-AUC) → eval_metrics.json
        └─ app.py / api.py         serve predictions
```

---

## 🧠 The problem

Lenders need to estimate whether an applicant will repay a loan (**good risk**) or default (**bad risk**). This project builds a binary classifier from applicant demographics and financial features to support that decision. In production, such models work alongside policy rules, human review, and compliance checks — never as the sole decision maker.

## 📦 Dataset

- **Source:** `southAfrican_credit_data.csv` (~1,000 loan applications)
- **Target:** `Risk` (`good` / `bad`)
- **Class balance:** imbalanced ~70% good / ~30% bad — handled with `class_weight="balanced"`

| Feature | Type | Description |
|---------|------|-------------|
| Age | numeric | Applicant age |
| Sex | nominal | male / female |
| Job | numeric | Job category (0–3) |
| Housing | nominal | own / rent / free |
| Saving accounts | **ordinal** | little → moderate → rich → quite rich |
| Checking account | **ordinal** | little → moderate → rich |
| Credit amount | numeric | Loan amount (DM) |
| Duration | numeric | Loan term (months) |
| Purpose | nominal | car, radio/TV, education, business, … |

### Why the encoding matters

A key modeling decision: **`Saving accounts` and `Checking account` are ordinal** (their levels have a natural order), so they use an `OrdinalEncoder` that preserves that order. `Sex`, `Housing`, and `Purpose` are **nominal** (no order), so they use a `OneHotEncoder`. This is implemented as a single `ColumnTransformer` saved as `preprocessor.pkl` — eliminating training/serving skew and bumping hold-out accuracy from **0.648 → 0.695**.

## 📈 Model & metrics

**Model:** Extra Trees Classifier (`class_weight="balanced"`, 300 estimators)

Hold-out evaluation (from `artifacts/eval_metrics.json`):

| Metric | Value |
|--------|------:|
| Accuracy | 0.695 |
| Precision | 0.776 |
| Recall | 0.793 |
| F1 | 0.784 |
| ROC-AUC | 0.716 |

> For credit risk, **recall** and **precision** matter more than raw accuracy — missing a default (false negative) is usually costlier than flagging a good applicant for review. A naive "always good" baseline would hit ~70% accuracy while catching **zero** defaults; this model achieves comparable accuracy with strong recall on the minority (bad-risk) class.

### Model selection (from `Credit Risk modeling.ipynb`)

GridSearchCV (5-fold) compared several models before Extra Trees was chosen for its balance of performance and simplicity:

| Model | Test accuracy |
|-------|---------------:|
| Decision Tree | 0.581 |
| Random Forest | 0.619 |
| Extra Trees | 0.648 |
| XGBoost | 0.676 |

---

## 🛠️ Tech stack

| Layer | Tools |
|-------|-------|
| ML | scikit-learn (Extra Trees, ColumnTransformer), pandas, NumPy |
| Serving | FastAPI, Uvicorn, Pydantic, Streamlit |
| Experiment tracking | MLflow |
| Monitoring | Prometheus |
| Reproducibility | DVC, `params.yaml`, pinned `requirements.txt` |
| Packaging | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

## 🗂️ Project structure

```text
├── app.py                 Streamlit UI (calls the API)
├── api.py                 FastAPI inference service + Prometheus metrics
├── train.py               Training entrypoint (logs to MLflow)
├── evaluate.py            Evaluation + quality gates
├── params.yaml            Central config (hyperparams, thresholds, MLflow)
├── src/
│   ├── data_loader.py     load, split, target encoding
│   ├── preprocessing.py   ColumnTransformer (ordinal + one-hot)
│   ├── model.py           build / train / save / load
│   └── metrics.py         metric computation + quality-gate thresholds
├── artifacts/             model.pkl, preprocessor.pkl, metrics JSON
├── tests/                 pytest suite (data, model, evaluate, api)
├── docker/                Dockerfile + docker-compose + Prometheus config
├── dvc.yaml               DVC pipeline stages
└── .github/workflows/     CI: train → evaluate → pytest
```



## ⚠️ Limitations

- Trained on a historical, region-specific dataset; performance may not generalize to other markets or time periods.
- No fairness audit by default; deployers should evaluate metrics across protected groups before any real use.
- Built for **educational / portfolio** purposes — not for production lending decisions.

## 📄 License

See repository license (if applicable).
