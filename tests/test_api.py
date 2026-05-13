from fastapi.testclient import TestClient 
from api import app 


client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_predict_contract():
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