from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Ordinal features: order matters (low -> high). "unknown" is the lowest bucket.
ORDINAL_COLUMNS = {
    "Saving accounts": ["unknown", "little", "moderate", "rich", "quite rich"],
    "Checking account": ["unknown", "little", "moderate", "rich"],
}

# Nominal features: no inherent order -> one-hot
NOMINAL_COLUMNS = ["Sex", "Housing", "Purpose"]

# Numeric features: passthrough
NUMERIC_COLUMNS = ["Age", "Job", "Credit amount", "Duration"]

CATEGORICAL_COLUMNS = list(ORDINAL_COLUMNS.keys()) + NOMINAL_COLUMNS

PREPROCESSOR_FILENAME = "preprocessor.pkl"


def fill_missing_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype(str).replace({"nan": "unknown", "NA": "unknown"})
        out[col] = out[col].where(out[col] != "", "unknown")
    return out


def build_preprocessor() -> ColumnTransformer:
    ordinal = OrdinalEncoder(
        categories=[
            ORDINAL_COLUMNS["Saving accounts"],
            ORDINAL_COLUMNS["Checking account"],
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    nominal = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("ord", ordinal, list(ORDINAL_COLUMNS.keys())),
            ("nom", nominal, NOMINAL_COLUMNS),
            ("num", "passthrough", NUMERIC_COLUMNS),
        ]
    )


def _to_frame(arr, preprocessor: ColumnTransformer) -> pd.DataFrame:
    return pd.DataFrame(arr, columns=preprocessor.get_feature_names_out())


def preprocess_train(
    X_train: pd.DataFrame,
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    X_train = fill_missing_categories(X_train)
    preprocessor = build_preprocessor()
    arr = preprocessor.fit_transform(X_train)
    return _to_frame(arr, preprocessor), preprocessor


def preprocess_inference(
    X: pd.DataFrame, preprocessor: ColumnTransformer
) -> pd.DataFrame:
    X = fill_missing_categories(X)
    arr = preprocessor.transform(X)
    return _to_frame(arr, preprocessor)


def save_preprocessor(preprocessor: ColumnTransformer, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_dir / PREPROCESSOR_FILENAME)


def load_preprocessor(out_dir: str | Path) -> ColumnTransformer:
    return joblib.load(Path(out_dir) / PREPROCESSOR_FILENAME)
