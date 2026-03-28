"""
Flatten the Synthea COVID-19 relational EHR into one patient-level feature matrix.

Sources:
  patients.csv    → demographics, costs, mortality target
  conditions.csv  → COVID diagnosis + key comorbidity flags + condition count
  observations.csv → mean vitals per patient (BP, HR, RR, O2 sat, temp, weight)
  encounters.csv  → total / inpatient / emergency encounter counts

Output:
  data/processed/patient_features.csv  — one row per patient, ready for generators

Target column:
  DECEASED (1 = patient has a death date, 0 = alive)
  ~19.5% positive rate — clean binary classification
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Condition name → column name mappings
# ---------------------------------------------------------------------------

COMORBIDITY_KEYWORDS = {
    "covid_confirmed":  "COVID-19",
    "covid_suspected":  "Suspected COVID-19",
    "obesity":          "Body mass index 30+ - obesity",
    "prediabetes":      "Prediabetes",
    "hypertension":     "Hypertension",
    "anemia":           "Anemia (disorder)",
    "pneumonia":        "Pneumonia (disorder)",
    "hypoxemia":        "Hypoxemia (disorder)",
    "diabetes":         "Diabetes",
    "loss_of_taste":    "Loss of taste (finding)",
    "fever":            "Fever (finding)",
    "fatigue":          "Fatigue (finding)",
}

VITAL_DESCRIPTIONS = {
    "systolic_bp":    "Systolic Blood Pressure",
    "diastolic_bp":   "Diastolic Blood Pressure",
    "heart_rate":     "Heart rate",
    "respiratory_rate": "Respiratory rate",
    "o2_saturation":  "Oxygen saturation in Arterial blood",
    "body_temp":      "Body temperature",
    "body_weight":    "Body Weight",
    "body_height":    "Body Height",
}


# ---------------------------------------------------------------------------
# Individual feature builders
# ---------------------------------------------------------------------------


def build_patient_base(patients: pd.DataFrame) -> pd.DataFrame:
    """Demographics, costs, and target from patients.csv."""
    df = patients.copy()

    # Age: use 2021 as the reference year (dataset is COVID-era)
    df["birth_year"] = df["BIRTHDATE"].str[:4].astype(int)
    df["age"] = 2021 - df["birth_year"]

    # Binary target
    df["DECEASED"] = df["DEATHDATE"].notna().astype(int)

    # Gender
    df["gender_M"] = (df["GENDER"] == "M").astype(int)

    # Race — one-hot (white is reference, dropped)
    for race in ["black", "asian", "native", "other"]:
        df[f"race_{race}"] = (df["RACE"] == race).astype(int)

    # Ethnicity
    df["hispanic"] = (df["ETHNICITY"] == "hispanic").astype(int)

    # Marital status
    df["married"] = (df["MARITAL"] == "M").astype(int)

    keep_cols = [
        "Id", "age", "gender_M",
        "race_black", "race_asian", "race_native", "race_other",
        "hispanic", "married",
        "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE",
        "DECEASED",
    ]
    return df[keep_cols].rename(columns={
        "Id": "patient_id",
        "HEALTHCARE_EXPENSES": "healthcare_expenses",
        "HEALTHCARE_COVERAGE": "healthcare_coverage",
    })


def build_condition_features(conditions: pd.DataFrame) -> pd.DataFrame:
    """
    Per-patient binary comorbidity flags + total distinct condition count.
    Each flag = 1 if the patient ever had that condition.
    """
    print("  Building condition features …")

    # Total distinct conditions per patient
    condition_counts = (
        conditions.groupby("PATIENT")["DESCRIPTION"]
        .nunique()
        .rename("condition_count")
        .reset_index()
        .rename(columns={"PATIENT": "patient_id"})
    )

    # Binary flags for each comorbidity
    flag_frames = [condition_counts]
    for col_name, keyword in COMORBIDITY_KEYWORDS.items():
        subset = conditions[conditions["DESCRIPTION"].str.contains(keyword, case=False, na=False, regex=False)]
        flag = (
            subset.groupby("PATIENT")
            .size()
            .gt(0)
            .astype(int)
            .rename(f"cond_{col_name}")
            .reset_index()
            .rename(columns={"PATIENT": "patient_id"})
        )
        flag_frames.append(flag)

    result = flag_frames[0]
    for frame in flag_frames[1:]:
        result = result.merge(frame, on="patient_id", how="left")

    # Fill zeros for patients who never had those conditions
    flag_cols = [c for c in result.columns if c.startswith("cond_")]
    result[flag_cols] = result[flag_cols].fillna(0).astype(int)
    return result


def build_vital_features(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Mean value of key vitals per patient from observations.csv.
    Only uses TYPE=numeric rows.
    16M rows — use groupby rather than pivot for memory efficiency.
    """
    print("  Building vital features (16M rows — takes ~30s) …")
    numeric_obs = observations[observations["TYPE"] == "numeric"].copy()
    numeric_obs["VALUE"] = pd.to_numeric(numeric_obs["VALUE"], errors="coerce")

    rows = []
    for col_name, description in VITAL_DESCRIPTIONS.items():
        subset = numeric_obs[numeric_obs["DESCRIPTION"] == description]
        means = (
            subset.groupby("PATIENT")["VALUE"]
            .mean()
            .rename(f"vital_{col_name}")
            .reset_index()
            .rename(columns={"PATIENT": "patient_id"})
        )
        rows.append(means)

    result = rows[0]
    for frame in rows[1:]:
        result = result.merge(frame, on="patient_id", how="left")

    return result


def build_encounter_features(encounters: pd.DataFrame) -> pd.DataFrame:
    """Total, inpatient, and emergency encounter counts per patient."""
    print("  Building encounter features …")

    total = (
        encounters.groupby("PATIENT")
        .size()
        .rename("encounter_total")
        .reset_index()
        .rename(columns={"PATIENT": "patient_id"})
    )

    inpatient = (
        encounters[encounters["ENCOUNTERCLASS"] == "inpatient"]
        .groupby("PATIENT")
        .size()
        .rename("encounter_inpatient")
        .reset_index()
        .rename(columns={"PATIENT": "patient_id"})
    )

    emergency = (
        encounters[encounters["ENCOUNTERCLASS"] == "emergency"]
        .groupby("PATIENT")
        .size()
        .rename("encounter_emergency")
        .reset_index()
        .rename(columns={"PATIENT": "patient_id"})
    )

    result = total.merge(inpatient, on="patient_id", how="left")
    result = result.merge(emergency, on="patient_id", how="left")
    result[["encounter_inpatient", "encounter_emergency"]] = (
        result[["encounter_inpatient", "encounter_emergency"]].fillna(0).astype(int)
    )
    return result


# ---------------------------------------------------------------------------
# Main flatten function
# ---------------------------------------------------------------------------


def flatten_ehr(
    raw_dir: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Load all source tables, build feature matrix, save to output_path.
    Returns the final DataFrame.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading source tables …")
    patients    = pd.read_csv(raw_dir / "patients.csv")
    conditions  = pd.read_csv(raw_dir / "conditions.csv")
    encounters  = pd.read_csv(raw_dir / "encounters.csv", low_memory=False)
    observations= pd.read_csv(raw_dir / "observations.csv", low_memory=False)

    print(f"  patients    : {len(patients):,} rows")
    print(f"  conditions  : {len(conditions):,} rows")
    print(f"  encounters  : {len(encounters):,} rows")
    print(f"  observations: {len(observations):,} rows")

    print("\nBuilding feature blocks …")
    base       = build_patient_base(patients)
    cond_feats = build_condition_features(conditions)
    vital_feats= build_vital_features(observations)
    enc_feats  = build_encounter_features(encounters)

    print("\nJoining feature blocks …")
    df = base
    df = df.merge(cond_feats,  on="patient_id", how="left")
    df = df.merge(vital_feats, on="patient_id", how="left")
    df = df.merge(enc_feats,   on="patient_id", how="left")

    # Drop patient_id — not a feature
    df = df.drop(columns=["patient_id"])

    # Fill remaining nulls
    # Condition flags → 0 (patient had no conditions table rows)
    cond_cols = [c for c in df.columns if c.startswith("cond_") or c == "condition_count"]
    df[cond_cols] = df[cond_cols].fillna(0)

    # Encounter counts → 0
    enc_cols = [c for c in df.columns if c.startswith("encounter_")]
    df[enc_cols] = df[enc_cols].fillna(0)

    # Vitals → median impute (missing means patient had no recorded vitals)
    vital_cols = [c for c in df.columns if c.startswith("vital_")]
    for col in vital_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Convert integer columns
    int_cols = cond_cols + enc_cols + [
        "gender_M", "race_black", "race_asian", "race_native", "race_other",
        "hispanic", "married", "DECEASED",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Drop zero-variance columns — all-zero flags carry no signal and confuse generators
    zero_var_cols = [c for c in df.columns if df[c].std() == 0]
    if zero_var_cols:
        print(f"\nDropping {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("FLATTENED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target      : DECEASED")
    print(f"  Positive    : {df['DECEASED'].sum():,} ({df['DECEASED'].mean()*100:.1f}%)")
    print(f"  Nulls       : {df.isnull().sum().sum()}")
    print(f"  Saved to    : {output_path}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")

    # Save a column manifest
    manifest = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "target": "DECEASED",
        "positive_rate_pct": round(df["DECEASED"].mean() * 100, 2),
        "columns": {
            col: str(df[col].dtype) for col in df.columns
        },
        "null_counts": df.isnull().sum().to_dict(),
    }
    manifest_path = output_path.parent / "flatten_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    return df


if __name__ == "__main__":
    flatten_ehr(
        raw_dir="data/raw/100k_synthea_covid19_csv",
        output_path="data/processed/patient_features.csv",
    )
