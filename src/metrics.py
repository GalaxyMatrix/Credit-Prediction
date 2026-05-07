from __future__ import annotations
from typing import Dict


import numpy as np 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics
def assert_metric_thresholds(metrics: Dict[str, float], min_f1: float = 0.50, min_auc: float = 0.60) -> None:
    if metrics.get("f1", 0.0) < min_f1:
        raise ValueError(f"F1 below threshold: {metrics['f1']:.4f} < {min_f1}")
    auc = metrics.get("roc_auc", np.nan)
    if not np.isnan(auc) and auc < min_auc:
        raise ValueError(f"ROC AUC below threshold: {auc:.4f} < {min_auc}")
