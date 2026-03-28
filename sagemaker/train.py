"""
SageMaker training entry point.

Reads:  /opt/ml/input/data/train/patient_features.csv
Writes: /opt/ml/model/
  - {generator}_synthetic.csv
  - {generator}_model.pkl
  - training_manifest.json

Invoked with hyperparameter --generator [copula|ctgan|tvae]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# SageMaker paths
INPUT_DIR  = Path(os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
OUTPUT_DIR = Path(os.environ.get('SM_MODEL_DIR',     '/opt/ml/model'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split(target_col: str = 'DECEASED', random_state: int = 42):
    csv_path = INPUT_DIR / 'patient_features.csv'
    print(f'Loading {csv_path} ...')
    df = pd.read_csv(csv_path)
    print(f'  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols')

    train_df, _ = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df[target_col]
    )
    print(f'  Train split: {len(train_df):,} rows')
    return train_df


LOG_TRANSFORM_COLS = [
    'encounter_inpatient',
    'encounter_emergency',
    'encounter_total',
    'condition_count',
    'healthcare_expenses',
    'healthcare_coverage',
]


def scale_continuous(df: pd.DataFrame, target_col: str = 'DECEASED') -> pd.DataFrame:
    """Log-transform count/monetary cols then StandardScale all continuous cols."""
    import numpy as np
    numeric = [c for c in df.select_dtypes(include='number').columns if c != target_col]
    binary  = [c for c in numeric if set(df[c].dropna().unique()) <= {0, 1}]
    continuous = [c for c in numeric if c not in binary]

    out = df.copy()
    log_cols = [c for c in LOG_TRANSFORM_COLS if c in continuous]
    for col in log_cols:
        out[col] = np.log1p(out[col])

    scaler = StandardScaler()
    out[continuous] = scaler.fit_transform(out[continuous])
    print(f'  Log-transformed {len(log_cols)} count cols, scaled {len(continuous)} continuous cols')
    return out


def build_metadata(train_df: pd.DataFrame, target_col: str) -> object:
    """Auto-detect metadata and mark target + all binary columns as categorical."""
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    metadata.update_column(column_name=target_col, sdtype='categorical')

    # Mark all binary (0/1) columns as categorical to prevent TVAE/CTGAN collapse
    binary_cols = [
        c for c in train_df.select_dtypes(include='number').columns
        if c != target_col and set(train_df[c].dropna().unique()) <= {0, 1}
    ]
    for col in binary_cols:
        metadata.update_column(column_name=col, sdtype='categorical')

    return metadata


def train_copula(train_df: pd.DataFrame, target_col: str) -> object:
    from sdv.single_table import GaussianCopulaSynthesizer

    metadata = build_metadata(train_df, target_col)
    model = GaussianCopulaSynthesizer(metadata, default_distribution='norm')
    model.fit(train_df)
    return model


def train_ctgan(train_df: pd.DataFrame, target_col: str, epochs: int, batch_size: int) -> object:
    from sdv.single_table import CTGANSynthesizer

    metadata = build_metadata(train_df, target_col)
    model = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size, verbose=True)
    model.fit(train_df)
    return model


def train_tvae(train_df: pd.DataFrame, target_col: str, epochs: int, batch_size: int) -> object:
    from sdv.single_table import TVAESynthesizer

    metadata = build_metadata(train_df, target_col)
    model = TVAESynthesizer(metadata, epochs=epochs, batch_size=batch_size)
    model.fit(train_df)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator',    type=str, default='ctgan',
                        choices=['copula', 'ctgan', 'tvae'])
    parser.add_argument('--target-col',  type=str, default='DECEASED')
    parser.add_argument('--epochs',       type=int, default=300)
    parser.add_argument('--batch-size',   type=int, default=500)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    print('=' * 60)
    print(f'Generator : {args.generator}')
    print(f'Epochs    : {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print('=' * 60)

    train_df = load_and_split(target_col=args.target_col, random_state=args.random_state)
    train_df = scale_continuous(train_df, target_col=args.target_col)

    t0 = time.time()
    if args.generator == 'copula':
        model = train_copula(train_df, args.target_col)
    elif args.generator == 'ctgan':
        model = train_ctgan(train_df, args.target_col, args.epochs, args.batch_size)
    else:
        model = train_tvae(train_df, args.target_col, args.epochs, args.batch_size)
    elapsed = time.time() - t0
    print(f'Training done in {elapsed:.1f}s')

    # Generate synthetic data matching train size
    print(f'Generating {len(train_df):,} synthetic rows ...')
    synth_df = model.sample(num_rows=len(train_df))

    # Save outputs
    synth_path = OUTPUT_DIR / f'{args.generator}_synthetic.csv'
    model_path = OUTPUT_DIR / f'{args.generator}_model.pkl'
    synth_df.to_csv(synth_path, index=False)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    manifest = {
        'generator': args.generator,
        'train_rows': len(train_df),
        'synth_rows': len(synth_df),
        'elapsed_seconds': round(elapsed, 1),
        'target_col': args.target_col,
        'synth_positive_rate': float(synth_df[args.target_col].astype(float).mean()),
        'real_positive_rate': float(train_df[args.target_col].mean()),
    }
    with open(OUTPUT_DIR / 'training_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Outputs written to {OUTPUT_DIR}')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
