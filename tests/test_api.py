from pathlib import Path

import pytest

ARTIFACT_MODEL = Path("artifacts/best_extra_trees_model.pkl")

pytestmark = pytest.mark.skipif(
    not ARTIFACT_MODEL.exists(),
    reason="Model artifacts missing; run: python train.py",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_contract(client):
    payload = {
        "Age": 30,
        "Sex": "male",
        "Job": 1,
        "Housing": "own",
        "Saving accounts": "little",
        "Checking account": "moderate",
        "Credit amount": 2000,
        "Duration": 12,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert body["prediction"] in [0, 1]