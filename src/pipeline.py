"""
End-to-end synthetic data pipeline.

Usage:
  python -m src.pipeline --config configs/config.yaml

Stages:
  1. Load and preprocess data
  2. Train generators (Copula, CTGAN, TVAE)
  3. Generate synthetic datasets
  4. Realism evaluation
  5. Downstream ML utility evaluation
  6. Privacy leakage analysis
  7. Save reports
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI windows needed

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config_path: str = "configs/config.yaml") -> dict:
    cfg = load_config(config_path)
    t_start = time.time()

    flat_path = cfg["data"]["flat_path"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    synthetic_dir = Path(cfg["data"]["synthetic_dir"])
    target_col = cfg["data"]["target_column"]
    reports_dir = Path("reports")
    model_dir = Path("outputs/models")

    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Preprocessing
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: PREPROCESSING")
    print("=" * 70)

    from src.preprocessing.preprocess import run_preprocessing

    train, val, test, preprocessor, data_card = run_preprocessing(
        flat_path=flat_path,
        output_dir=processed_dir,
        target_col=target_col,
        train_frac=cfg["splits"]["train"],
        val_frac=cfg["splits"]["val"],
        random_state=cfg["splits"]["random_state"],
    )

    # ------------------------------------------------------------------
    # Phase 2 & 3: Load synthetic datasets (pre-trained on SageMaker)
    #              or train locally if CSVs don't exist yet
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2-3: LOAD SYNTHETIC DATASETS")
    print("=" * 70)

    generator_names = ["copula", "ctgan", "tvae"]
    pre_built = {
        name: synthetic_dir / f"{name}_synthetic.csv"
        for name in generator_names
    }
    all_exist = all(p.exists() for p in pre_built.values())

    if all_exist:
        synth_datasets = {}
        # Identify binary columns from the real train set (excluding target)
        binary_cols = [
            c for c in train.select_dtypes(include="number").columns
            if c != target_col and set(train[c].dropna().unique()) <= {0, 1}
        ]
        for name, path in pre_built.items():
            df = pd.read_csv(path)
            # Generators output fractional values for binary columns — snap back to 0/1
            for col in binary_cols:
                if col in df.columns:
                    df[col] = df[col].round().clip(0, 1).astype(int)
            synth_datasets[name] = df
            print(f"  Loaded {name}: {len(df):,} rows  (snapped {len(binary_cols)} binary cols)")
    else:
        from src.generators.train_generators import run_training
        synth_datasets = run_training(
            train_df=train,
            model_dir=model_dir,
            synthetic_dir=synthetic_dir,
            target_col=target_col,
            ctgan_epochs=cfg["generators"]["ctgan"]["epochs"],
            tvae_epochs=cfg["generators"]["tvae"]["epochs"],
            batch_size=cfg["generators"]["ctgan"]["batch_size"],
        )

    # ------------------------------------------------------------------
    # Phase 4: Realism evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: REALISM EVALUATION")
    print("=" * 70)

    from src.evaluation.realism import (
        plot_correlation_heatmaps,
        plot_distributions,
        plot_pca_overlap,
        realism_scorecard,
    )

    real_scorecard = realism_scorecard(train, synth_datasets, target_col=target_col)
    real_scorecard.to_csv(reports_dir / "realism_scorecard.csv", index=False)

    plot_distributions(train, synth_datasets, target_col=target_col, output_dir=reports_dir)
    plot_correlation_heatmaps(train, synth_datasets, target_col=target_col, output_dir=reports_dir)
    plot_pca_overlap(train, synth_datasets, target_col=target_col, output_dir=reports_dir)

    # ------------------------------------------------------------------
    # Phase 5: Utility evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: DOWNSTREAM ML UTILITY EVALUATION")
    print("=" * 70)

    from src.evaluation.utility import (
        plot_metric_heatmap,
        plot_utility_comparison,
        run_utility_benchmark,
    )

    utility_results = run_utility_benchmark(
        train_real=train,
        test_real=test,
        synth_datasets=synth_datasets,
        target_col=target_col,
        aug_ratios=cfg["evaluation"]["aug_ratios"],
        random_state=cfg["splits"]["random_state"],
    )
    utility_results.to_csv(reports_dir / "utility_results.csv", index=False)

    plot_utility_comparison(utility_results, metric="auroc", output_dir=reports_dir)
    plot_utility_comparison(utility_results, metric="auprc", output_dir=reports_dir)
    plot_metric_heatmap(utility_results, output_dir=reports_dir)

    # ------------------------------------------------------------------
    # Phase 6: Privacy analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 6: PRIVACY LEAKAGE ANALYSIS")
    print("=" * 70)

    from src.privacy.privacy_checks import (
        plot_nn_distance_distributions,
        privacy_scorecard,
    )

    priv_scorecard = privacy_scorecard(
        train,
        synth_datasets,
        target_col=target_col,
        n_neighbors=cfg["privacy"]["n_neighbors"],
    )
    priv_scorecard.to_csv(reports_dir / "privacy_scorecard.csv", index=False)
    plot_nn_distance_distributions(train, synth_datasets, target_col=target_col, output_dir=reports_dir)

    # ------------------------------------------------------------------
    # Phase 7: Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("=" * 70)

    summary = {
        "data_card": data_card,
        "realism": real_scorecard.to_dict(orient="records"),
        "utility": utility_results.to_dict(orient="records"),
        "privacy": priv_scorecard.to_dict(orient="records"),
        "elapsed_seconds": elapsed,
    }

    with open(reports_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nReports saved to: {reports_dir.resolve()}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic data benchmark pipeline")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run(config_path=args.config)
