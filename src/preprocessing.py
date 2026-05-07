from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


import joblib 
import pandas as pd 

from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLUMNS = ["Sex", "Housing", "Saving accounts", "Checking account"]

def fill_missing_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() 
    for col in  CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("unknown")
    return out 



def fit_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders : Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def transform_with_encoders(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    out = df.copy()
    for col, le in encoders.items():
        values = out[col].astype(str)
        unknown = sorted(set(values.unique()) - set(le.classes_))
        if unknown:
            raise ValueError(f"Unknown values in {col}: {unknown}")
        out[col] = le.transform(values)
    return out 



def preprocess_train(
    X_train: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    X_train = fill_missing_categories(X_train)
    encoders = fit_label_encoders(X_train)
    X_train = transform_with_encoders(X_train, encoders)
    return X_train, encoders



def preprocess_inference(
    X: pd.DataFrame, encoders: Dict[str, LabelEncoder]
) -> pd.DataFrame:
    X = fill_missing_categories(X)
    return transform_with_encoders(X, encoders)


def save_encoders(encoders: Dict[str, LabelEncoder], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for col, enc in encoders.items():
        joblib.dump(enc, out_dir / f"{col}_label_encoder.pkl")

        