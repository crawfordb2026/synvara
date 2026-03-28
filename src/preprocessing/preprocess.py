"""
Data loading, profiling, and preprocessing for the ULB Credit Card Fraud dataset.

Steps:
  1. Load raw CSV from local path or S3.
  2. Profile the dataset (shape, nulls, class balance, distributions).
  3. Split into train / val / test (fit transforms on train only).
  4. Standardise numeric columns (excluding target).
  5. Save processed splits to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download creditcard.csv from Kaggle and place it in data/raw/."
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def profile(df: pd.DataFrame, target_col: str = "Class") -> dict:
    """Return a data card dict and print a summary."""
    n_rows, n_cols = df.shape
    null_counts = df.isnull().sum().to_dict()
    class_counts = df[target_col].value_counts().to_dict()
    fraud_rate = class_counts.get(1, 0) / n_rows

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    stats_summary = {}
    for col in numeric_cols:
        stats_summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "skew": float(df[col].skew()),
        }

    card = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "null_counts": null_counts,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "fraud_rate_pct": round(fraud_rate * 100, 4),
        "numeric_columns": numeric_cols,
        "feature_stats": stats_summary,
    }

    print("=" * 60)
    print("DATA CARD")
    print("=" * 60)
    print(f"  Shape          : {n_rows:,} rows × {n_cols} columns")
    print(f"  Nulls          : {sum(null_counts.values())}")
    print(f"  Class balance  : {class_counts}")
    print(f"  Fraud rate     : {fraud_rate * 100:.4f}%")
    print("=" * 60)

    return card


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
    target_col: str = "Class",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split."""
    test_frac = 1.0 - train_frac - val_frac

    train, temp = train_test_split(
        df,
        test_size=(val_frac + test_frac),
        random_state=random_state,
        stratify=df[target_col],
    )
    val_relative = val_frac / (val_frac + test_frac)
    val, test = train_test_split(
        temp,
        test_size=(1.0 - val_relative),
        random_state=random_state,
        stratify=temp[target_col],
    )

    print(f"Split sizes  →  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        rate = split_df[target_col].mean() * 100
        print(f"  {split_name} fraud rate: {rate:.4f}%")

    return train, val, test


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _is_binary_column(series: pd.Series) -> bool:
    """Returns True if a numeric column contains only 0 and 1 values."""
    unique_vals = set(series.dropna().unique())
    return unique_vals <= {0, 1}


class DataPreprocessor:
    """
    Fits a StandardScaler on continuous numeric columns only (excluding target
    and binary 0/1 columns, which should not be scaled).
    Applies the same transform to val/test.
    """

    def __init__(self, target_col: str = "Class"):
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.continuous_cols: list[str] = []
        self.binary_cols: list[str] = []
        self.feature_cols: list[str] = []  # all non-target numeric cols

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != self.target_col
        ]
        self.feature_cols = numeric_cols
        self.binary_cols = [c for c in numeric_cols if _is_binary_column(df[c])]
        self.continuous_cols = [c for c in numeric_cols if c not in self.binary_cols]

        out = df.copy()
        if self.continuous_cols:
            out[self.continuous_cols] = self.scaler.fit_transform(df[self.continuous_cols])
        print(f"  Scaling {len(self.continuous_cols)} continuous cols, "
              f"leaving {len(self.binary_cols)} binary cols unchanged.")
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.continuous_cols:
            out[self.continuous_cols] = self.scaler.transform(df[self.continuous_cols])
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.continuous_cols:
            out[self.continuous_cols] = self.scaler.inverse_transform(df[self.continuous_cols])
        return out

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "feature_cols": self.feature_cols,
            "continuous_cols": self.continuous_cols,
            "binary_cols": self.binary_cols,
            "target_col": self.target_col,
            "scaler_mean": self.scaler.mean_.tolist() if self.continuous_cols else [],
            "scaler_scale": self.scaler.scale_.tolist() if self.continuous_cols else [],
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved preprocessor metadata to {path}")


# ---------------------------------------------------------------------------
# Full pipeline helper
# ---------------------------------------------------------------------------


def run_preprocessing(
    flat_path: str | Path,
    output_dir: str | Path,
    target_col: str = "DECEASED",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DataPreprocessor, dict]:
    """
    End-to-end preprocessing from a pre-flattened CSV.
    Returns (train, val, test, preprocessor, data_card).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(flat_path)
    card = profile(df, target_col=target_col)

    train_raw, val_raw, test_raw = split_dataset(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        random_state=random_state,
        target_col=target_col,
    )

    preprocessor = DataPreprocessor(target_col=target_col)
    train = preprocessor.fit_transform(train_raw)
    val = preprocessor.transform(val_raw)
    test = preprocessor.transform(test_raw)

    # Save splits and metadata
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    preprocessor.save_metadata(output_dir / "preprocessor_meta.json")

    with open(output_dir / "data_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print(f"Saved processed splits to {output_dir}")
    return train, val, test, preprocessor, card
