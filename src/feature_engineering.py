"""
feature_engineering.py — Phase 3: Feature Engineering.

Course modules:
  Module III: Feature Engineering (ICD grouping, medication aggregation,
              categorical encoding, derived features)
"""

import pandas as pd
from .config import (
    MEDICATION_COLUMNS, AGE_MAP, CATEGORICAL_COLUMNS,
    ID_TO_STRING_COLS, TOP_SPECIALTIES_N,
)


# ── ICD-9 Diagnosis Code Grouping ────────────────────────────────

def group_icd9(code) -> str:
    """Map an ICD-9 diagnosis code to one of 18 clinical categories."""
    if pd.isna(code) or code == "Unknown":
        return "Unknown"
    code = str(code)
    if code.startswith("E"):
        return "External"
    if code.startswith("V"):
        return "Supplementary"
    try:
        num = float(code)
    except ValueError:
        return "Other"

    ranges = [
        (1, 139, "Infectious"), (140, 239, "Neoplasms"),
        (240, 279, "Endocrine"), (280, 289, "Blood"),
        (290, 319, "Mental"), (320, 389, "Nervous"),
        (390, 459, "Circulatory"), (460, 519, "Respiratory"),
        (520, 579, "Digestive"), (580, 629, "Genitourinary"),
        (630, 679, "Pregnancy"), (680, 709, "Skin"),
        (710, 739, "Musculoskeletal"), (740, 759, "Congenital"),
        (760, 779, "Perinatal"), (780, 799, "Symptoms"),
        (800, 999, "Injury"),
    ]
    for lo, hi, label in ranges:
        if lo <= num <= hi:
            return label
    return "Other"


def engineer_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw ICD-9 codes with clinical group categories."""
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(group_icd9)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])
    print("  ICD-9 codes → 18 clinical group features")
    return df


# ── Medication Aggregation ────────────────────────────────────────

def engineer_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 23 individual medication columns into summary features."""
    med_cols = [c for c in MEDICATION_COLUMNS if c in df.columns]

    df["n_med_changed"] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v in ["Up", "Down"]), axis=1
    )
    df["n_med_on"] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v != "No"), axis=1
    )
    df["insulin_change"] = df["insulin"].map(
        {"No": 0, "Steady": 1, "Up": 2, "Down": 2}
    )
    df = df.drop(columns=med_cols)
    print("  Medications → insulin_change + n_med_changed + n_med_on")
    return df


# ── Encode & Derive ──────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary flags, ordinal age, lab results, and rare specialties."""
    df["change"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

    # Age → ordinal
    df["age_ordinal"] = df["age"].map(AGE_MAP)
    df = df.drop(columns=["age"])

    # Lab / A1C → ordinal
    df["max_glu_serum"] = (
        df["max_glu_serum"]
        .map({"None": 0, "Norm": 1, ">200": 2, ">300": 3})
        .fillna(0).astype(int)
    )
    df["A1Cresult"] = (
        df["A1Cresult"]
        .map({"None": 0, "Norm": 1, ">7": 2, ">8": 3})
        .fillna(0).astype(int)
    )

    # Group rare medical specialties
    top_specs = df["medical_specialty"].value_counts().head(TOP_SPECIALTIES_N).index.tolist()
    df["medical_specialty"] = df["medical_specialty"].apply(
        lambda x: x if x in top_specs else "Other"
    )
    print("  Encoded: age, change, diabetesMed, A1C, glucose, specialty")
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven interaction / threshold features."""
    df["total_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["high_utilizer"] = (df["number_inpatient"] >= 2).astype(int)
    df["many_meds"] = (df["num_medications"] >= 15).astype(int)
    df["long_stay"] = (df["time_in_hospital"] >= 7).astype(int)
    print("  Derived features: total_visits, high_utilizer, many_meds, long_stay")
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = list(CATEGORICAL_COLUMNS)
    for col in ID_TO_STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            cat_cols.append(col)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"  One-hot encoded → {df.shape[1]} total columns")
    return df


# ── Public API ────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full feature-engineering pipeline (Phase 3)."""
    print("=" * 60)
    print("  PHASE 3: FEATURE ENGINEERING")
    print("=" * 60)
    df = engineer_diagnosis_groups(df)
    df = engineer_medication_features(df)
    df = encode_features(df)
    df = create_derived_features(df)
    df = one_hot_encode(df)
    return df
