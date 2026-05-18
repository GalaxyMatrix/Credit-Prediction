import pytest

from src.metrics import assert_metric_thresholds, compute_metrics


def test_assert_metric_thresholds_passes():
  metrics = {"f1": 0.75, "roc_auc": 0.70}
  assert_metric_thresholds(metrics, min_f1=0.50, min_auc=0.60)


def test_assert_metric_thresholds_fails_low_f1():
  metrics = {"f1": 0.30, "roc_auc": 0.80}
  with pytest.raises(ValueError, match="F1 below threshold"):
    assert_metric_thresholds(metrics, min_f1=0.50, min_auc=0.60)


def test_assert_metric_thresholds_fails_low_auc():
  metrics = {"f1": 0.80, "roc_auc": 0.40}
  with pytest.raises(ValueError, match="ROC AUC below threshold"):
    assert_metric_thresholds(metrics, min_f1=0.50, min_auc=0.60)


def test_compute_metrics_basic():
  y_true = [1, 0, 1, 0]
  y_pred = [1, 0, 0, 0]
  metrics = compute_metrics(y_true, y_pred)
  assert "accuracy" in metrics
  assert "f1" in metrics
  assert 0.0 <= metrics["f1"] <= 1.0