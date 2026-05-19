# Credit Risk Prediction



End-to-end machine learning system for predicting credit default risk on South African credit application data. The project includes modular training and evaluation pipelines, experiment tracking with MLflow, a FastAPI inference service with Prometheus metrics, a Streamlit demo UI, Docker deployment, and GitHub Actions CI.

## Problem

Banks and lenders need to estimate whether an applicant is likely to repay a loan (**good risk**) or default (**bad risk**). This project builds a binary classifier from applicant demographics and financial features to support that decision. In production, models like this are used alongside policy rules, human review, and compliance checks—not as the sole decision maker.

## Dataset

- **Source file:** `southAfrican_credit_data.csv`
- **Size:** ~1,000 loan applications
- **Target:** `Risk` (`good` / `bad`)
- **Features used in the model:**

| Feature | Description |
|---------|-------------|
| Age | Applicant age |
| Sex | male / female |
| Job | Job category (0–3) |
| Housing | own / rent / free |
| Saving accounts | Savings level |
| Checking account | Checking account level |
| Credit amount | Loan amount (DM) |
| Duration | Loan term (months) |

The `Purpose` column exists in the raw CSV but is excluded from modeling to match the deployed feature set.

## Insights from exploratory analysis

From `Credit Risk modeling.ipynb`:

### Class imbalance

The target is imbalanced: **~700 good** vs **~300 bad** (~70% / 30%). A naive model that always predicts “good” would reach ~70% accuracy while failing to catch defaults.

All tree-based models in the notebook address this explicitly:

- **Random Forest & Extra Trees:** `class_weight="balanced"`
- **XGBoost:** `scale_pos_weight` set from the train-set class ratio

The production pipeline (`train.py`) uses `class_weight="balanced"` on Extra Trees and reports **F1 (0.80)** and **recall (0.81)** on hold-out data—better indicators than accuracy alone for imbalanced credit data.

### Model comparison (notebook, test-set accuracy)

GridSearchCV with 5-fold CV, scoring=`accuracy`:

| Model | Test accuracy |
|-------|---------------:|
| Decision Tree | 0.581 |
| Random Forest | 0.619 |
| Extra Trees | 0.648 |
| XGBoost | 0.676 |

**Extra Trees** was chosen for deployment (saved as `best_extra_trees_model.pkl`) as a strong balance of performance and simplicity, even though XGBoost had slightly higher accuracy in the notebook. The refactored pipeline improves on raw accuracy with stronger **precision/recall/F1** (see metrics below), using a fixed feature set and reproducible `src/` preprocessing.

> Research and training live in the notebook; production training runs via `train.py` / `evaluate.py`.

## Model & metrics

**Model:** Extra Trees Classifier (`class_weight=balanced`, 300 estimators)

Hold-out evaluation metrics (from `artifacts/eval_metrics.json`):

| Metric | Value |
|--------|------:|
| Accuracy | 0.715 |
| Precision | 0.786 |
| Recall | 0.814 |
| F1 | 0.800 |
| ROC-AUC | 0.767 |

For credit risk, **recall on bad-risk cases** and **precision** often matter more than accuracy alone: missing a default (false negative) can be costlier than flagging a good applicant for review (false positive).

## Pipeline

```text
southAfrican_credit_data.csv
        │
        ▼
  src/data_loader.py      load + split + encode target (good→1, bad→0)
        │
        ▼
  src/preprocessing.py    label-encode categoricals, handle missing values
        │
        ▼
  train.py                fit ExtraTrees → artifacts/*.pkl + train_metrics.json
        │
        ▼
  evaluate.py             quality gates (F1, ROC-AUC) → eval_metrics.json
        │
        ├── app.py          Streamlit UI
        └── api.py          FastAPI + Prometheus /metrics
```

Tracked experiments are stored under `./mlruns` (MLflow).

## Project structure

```text
├── app.py                 Streamlit demo
├── api.py                 FastAPI inference service
├── train.py               Training entrypoint
├── evaluate.py            Evaluation + quality gates
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── metrics.py
├── artifacts/             Model, encoders, metrics JSON
├── tests/
├── docker/                Dockerfile + docker-compose + Prometheus config
├── dvc.yaml               DVC pipeline stages
└── .github/workflows/     CI: train → evaluate → pytest
```

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
uv venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

uv pip install -r requirements.txt
```

### Train & evaluate

```bash
uv run python train.py
uv run python evaluate.py
```

Artifacts are written to `artifacts/`:

- `best_extra_trees_model.pkl`
- `*_label_encoder.pkl`
- `train_metrics.json`, `eval_metrics.json`

### Run tests

```bash
uv run pytest -q --disable-warnings
```

API tests require artifacts; run `train.py` first or rely on CI to train before pytest.

### Streamlit UI

```bash
uv run streamlit run app.py
```

### FastAPI

```bash
uv run uvicorn api:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /docs` | Swagger UI |
| `POST /predict` | Credit risk prediction |
| `GET /metrics` | Prometheus metrics |

### Docker

```bash
python train.py   # ensure artifacts/ exists
cd docker
docker compose up --build
```

- API: http://localhost:8000  
- Prometheus: http://localhost:9090  

### MLflow UI

```bash
uv run mlflow ui
```

Open http://localhost:5000 to browse training and evaluation runs.

## CI/CD

On push/PR to `main`, GitHub Actions:

1. Installs dependencies  
2. Runs `train.py` and `evaluate.py`  
3. Runs `pytest`  
4. Uploads `artifacts/` as a workflow artifact  

## Limitations

- Trained on a historical, region-specific dataset; performance may not generalize to other markets or time periods.
- No fairness audit is included by default; deployers should evaluate metrics across protected groups before production use.
- This repository is for educational and portfolio purposes, not production lending decisions.

## License

See repository license (if applicable).
