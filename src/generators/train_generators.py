"""
Train three synthetic tabular data generators:
  - Gaussian Copula (classical baseline)
  - CTGAN           (GAN-based)
  - TVAE            (VAE-based)

All three are from the SDV (Synthetic Data Vault) library.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer


GENERATOR_NAMES = ["copula", "ctgan", "tvae"]


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------


def build_metadata(df: pd.DataFrame, target_col: str = "Class") -> SingleTableMetadata:
    """Auto-detect column types, mark target and all binary columns as categorical.

    SDV auto-detects 0/1 integer columns as 'numerical', which causes TVAE to
    treat them as continuous and collapse low-frequency columns to a constant.
    Marking them as 'categorical' forces SDV to use discrete sampling, preserving
    the original class distributions.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Mark target as categorical
    metadata.update_column(column_name=target_col, sdtype="categorical")

    # Mark all binary (0/1) columns as categorical to prevent generator collapse
    binary_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col and set(df[c].dropna().unique()) <= {0, 1}
    ]
    for col in binary_cols:
        metadata.update_column(column_name=col, sdtype="categorical")

    return metadata


# ---------------------------------------------------------------------------
# Train individual generators
# ---------------------------------------------------------------------------


def train_copula(
    train_df: pd.DataFrame,
    metadata: SingleTableMetadata,
    default_distribution: str = "norm",
) -> GaussianCopulaSynthesizer:
    print("\n[Copula] Training Gaussian Copula baseline …")
    t0 = time.time()
    model = GaussianCopulaSynthesizer(
        metadata,
        default_distribution=default_distribution,
    )
    model.fit(train_df)
    elapsed = time.time() - t0
    print(f"[Copula] Training done in {elapsed:.1f}s")
    return model


def train_ctgan(
    train_df: pd.DataFrame,
    metadata: SingleTableMetadata,
    epochs: int = 300,
    batch_size: int = 500,
) -> CTGANSynthesizer:
    print(f"\n[CTGAN] Training for {epochs} epochs, batch_size={batch_size} …")
    t0 = time.time()
    model = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )
    model.fit(train_df)
    elapsed = time.time() - t0
    print(f"[CTGAN] Training done in {elapsed:.1f}s")
    return model


def train_tvae(
    train_df: pd.DataFrame,
    metadata: SingleTableMetadata,
    epochs: int = 300,
    batch_size: int = 500,
) -> TVAESynthesizer:
    print(f"\n[TVAE] Training for {epochs} epochs, batch_size={batch_size} …")
    t0 = time.time()
    model = TVAESynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
    )
    model.fit(train_df)
    elapsed = time.time() - t0
    print(f"[TVAE] Training done in {elapsed:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_model(model, name: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {name} model to {path}")
    return path


def load_model(name: str, model_dir: str | Path):
    path = Path(model_dir) / f"{name}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded {name} from {path}")
    return model


# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------


def generate_synthetic(
    model,
    name: str,
    n_rows: int,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Generating {n_rows:,} synthetic rows …")
    synth_df = model.sample(num_rows=n_rows)
    out_path = output_dir / f"{name}_synthetic.csv"
    synth_df.to_csv(out_path, index=False)
    print(f"[{name}] Saved synthetic data to {out_path}")
    return synth_df


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------


def run_training(
    train_df: pd.DataFrame,
    model_dir: str | Path,
    synthetic_dir: str | Path,
    target_col: str = "Class",
    ctgan_epochs: int = 300,
    tvae_epochs: int = 300,
    batch_size: int = 500,
) -> dict[str, pd.DataFrame]:
    """
    Train all three generators, save models, and return dict of synthetic DataFrames.
    """
    model_dir = Path(model_dir)
    synthetic_dir = Path(synthetic_dir)
    n_rows = len(train_df)

    metadata = build_metadata(train_df, target_col=target_col)

    # Train
    copula = train_copula(train_df, metadata)
    ctgan = train_ctgan(train_df, metadata, epochs=ctgan_epochs, batch_size=batch_size)
    tvae = train_tvae(train_df, metadata, epochs=tvae_epochs, batch_size=batch_size)

    # Save models
    save_model(copula, "copula", model_dir)
    save_model(ctgan, "ctgan", model_dir)
    save_model(tvae, "tvae", model_dir)

    # Generate synthetic datasets matching train size
    synthetic_datasets = {
        "copula": generate_synthetic(copula, "copula", n_rows, synthetic_dir),
        "ctgan": generate_synthetic(ctgan, "ctgan", n_rows, synthetic_dir),
        "tvae": generate_synthetic(tvae, "tvae", n_rows, synthetic_dir),
    }

    return synthetic_datasets
