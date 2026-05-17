from __future__ import annotations 

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

RISK_MAPPING = {"good": 1, "bad": 0}



FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
]

REQUIRED_COLUMNS = set(FEATURE_COLUMNS) | {"Risk"}

def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def encode_target(y: pd.Series) -> pd.Series:
    """Map Risk labels good/bad to 1/0 for sklearn."""
    y_norm = y.astype(str).str.strip().str.lower()
    unknown = sorted(set(y_norm.unique()) - set(RISK_MAPPING.keys()))
    if unknown:
        raise ValueError(f"Unknown Risk labels: {unknown}")
    return y_norm.map(RISK_MAPPING).astype(int)


def split_train_test(
    df: pd.DataFrame,
    target_col: str = "Risk",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)