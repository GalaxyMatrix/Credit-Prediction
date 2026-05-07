from __future__ import annotations 

from pathlib import Path 
from typing import Tuple 


import pandas as pd 
from sklearn.model_selection import train_test_split



REQUIRED_COLUMNS = {
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Risk",
}

def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
         raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df



def split_train_test(
    df: pd.DataFrame,
    target_col: str = "Risk",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)