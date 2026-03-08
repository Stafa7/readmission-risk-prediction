"""
data_cleaning.py — Phase 1 & 2: Data Loading, Cleaning, and Leakage Prevention.

Course modules:
  Module II: Enterprise Data Science Lifecycle, Best Practices
"""

import pandas as pd
from .config import (
    DATA_FILE, LEAKAGE_DISCHARGE_CODES, DROP_HIGH_MISSING,
)


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """Load the raw UCI diabetes dataset, treating '?' as NaN."""
    df = pd.read_csv(path, na_values="?")
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def remove_leakage_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove encounters where readmission is impossible (death / hospice)."""
    mask = df["discharge_disposition_id"].isin(LEAKAGE_DISCHARGE_CODES)
    print(f"Removing {mask.sum():,} death/hospice encounters (leakage prevention)")
    return df[~mask].copy()


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 3-class readmission column into a binary 0/1 target."""
    df["target"] = (df["readmitted"] == "<30").astype(int)
    print(f"Target rate: {df['target'].mean()*100:.1f}% positive (readmitted <30 days)")
    return df


def handle_missing_and_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-missing columns, fill remaining NaNs, remove invalid rows."""
    # Drop columns with excessive missingness
    for col in DROP_HIGH_MISSING:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"  Dropped '{col}' (excessive missingness)")

    # Fill categorical missing values
    df["medical_specialty"] = df["medical_specialty"].fillna("Missing")
    df["race"] = df["race"].fillna("Unknown")
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].fillna("Unknown")

    # Remove rows with invalid gender
    invalid = df["gender"] == "Unknown/Invalid"
    if invalid.sum() > 0:
        print(f"  Removing {invalid.sum()} rows with invalid gender")
        df = df[~invalid].copy()

    # Drop identifiers that are no longer needed
    df = df.drop(columns=["encounter_id", "readmitted"])

    print(f"  Clean dataset: {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Remaining missing values: {df.isnull().sum().sum()}")
    return df


def run_cleaning(path: str = DATA_FILE) -> pd.DataFrame:
    """Execute the full cleaning pipeline (Phases 1–2)."""
    print("=" * 60)
    print("  PHASE 1–2: DATA LOADING & CLEANING")
    print("=" * 60)
    df = load_data(path)
    df = remove_leakage_encounters(df)
    df = create_binary_target(df)
    df = handle_missing_and_invalid(df)
    return df
