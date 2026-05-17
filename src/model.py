from __future__ import annotations

from pathlib import Path 
from typing import Any 


import joblib 

import pandas as pd  


from sklearn.ensemble import ExtraTreesClassifier 


def build_model(random_state: int = 42, n_estimators: int = 300) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_estimators=n_estimators,
        n_jobs=-1,
    )


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    model.fit(X_train, y_train)
    return model 

def save_model(model: Any, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)
