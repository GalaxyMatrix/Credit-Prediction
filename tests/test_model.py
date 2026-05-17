import pandas as pd 


from src.data_loader import encode_target
from src.preprocessing import preprocess_train, preprocess_inference
from src.model import build_model, train_model



def _sample_df():
       return pd.DataFrame(
        {
            "Age": [25, 40, 35, 30],
            "Sex": ["male", "female", "male", "female"],
            "Job": [1, 2, 0, 3],
            "Housing": ["own", "rent", "free", "own"],
            "Saving accounts": ["little", "moderate", "rich", "quite rich"],
            "Checking account": ["little", "moderate", "little", "moderate"],
            "Credit amount": [1000, 3000, 1500, 5000],
            "Duration": [12, 24, 18, 36],
            "Risk": ["good", "bad", "good", "bad"],
        }
    )


def test_preprocess_roundtrip():
    df = _sample_df()
    X  = df.drop(columns=["Risk"])
    X_train_proc, enc = preprocess_train(X)
    X_inf_proc = preprocess_inference(X, enc)
    assert list(X_train_proc.columns) == list(X_inf_proc.columns)
    assert X_inf_proc.shape == X.shape



def test_model_trains_and_predicts():
    df = _sample_df()
    X = df.drop(columns=["Risk"])
    y = encode_target(df["Risk"])
    X_proc, _ = preprocess_train(X)
    model = build_model(random_state=42, n_estimators=20)
    model = train_model(model, X_proc, y)
    preds = model.predict(X_proc)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


