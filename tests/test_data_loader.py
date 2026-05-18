from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import encode_target, load_raw_data


def test_load_raw_data_missing_columns(tmp_path: Path):
  df = pd.DataFrame({"Age": [25], "Sex": ["male"]})
  csv_path = tmp_path / "bad.csv"
  df.to_csv(csv_path, index=False)

  with pytest.raises(ValueError, match="Missing required columns"):
    load_raw_data(csv_path)


def test_load_raw_data_ok(tmp_path: Path):
  df = pd.DataFrame(
    {
      "Age": [25],
      "Sex": ["male"],
      "Job": [1],
      "Housing": ["own"],
      "Saving accounts": ["little"],
      "Checking account": ["moderate"],
      "Credit amount": [1000],
      "Duration": [12],
      "Risk": ["good"],
    }
  )
  csv_path = tmp_path / "ok.csv"
  df.to_csv(csv_path, index=False)

  out = load_raw_data(csv_path)
  assert len(out) == 1
  assert "Risk" in out.columns


def test_encode_target_good_bad():
  y = pd.Series(["good", "bad", "good"])
  encoded = encode_target(y)
  assert list(encoded) == [1, 0, 1]


def test_encode_target_unknown_label():
  y = pd.Series(["good", "unknown"])
  with pytest.raises(ValueError, match="Unknown Risk labels"):
    encode_target(y)