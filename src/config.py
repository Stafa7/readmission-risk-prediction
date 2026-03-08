"""
config.py — Central configuration for the Readmission Risk Pipeline.

All hyperparameters, file paths, and constants live here so that
every module draws from a single source of truth.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_FILE = os.path.join(DATA_DIR, "diabetic_data.csv")

# ── Random state ──────────────────────────────────────────────────
SEED = 42

# ── Data cleaning ─────────────────────────────────────────────────
# Discharge dispositions that indicate death/hospice → leakage
LEAKAGE_DISCHARGE_CODES = [11, 13, 14, 19, 20, 21]

# Columns to drop due to excessive missingness
DROP_HIGH_MISSING = ["weight", "payer_code"]

# ── Feature engineering ───────────────────────────────────────────
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

AGE_MAP = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
    "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
    "[80-90)": 8, "[90-100)": 9,
}

CATEGORICAL_COLUMNS = [
    "race", "gender", "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group",
]

ID_TO_STRING_COLS = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
]

TOP_SPECIALTIES_N = 15

# ── Splitting ─────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Modeling ──────────────────────────────────────────────────────
N_FOLDS = 5
TOP_K_PCT = 0.10  # Top 10 % for operational metrics

LR_PARAMS = dict(max_iter=1000, random_state=SEED, class_weight="balanced", solver="lbfgs", C=1.0)

RF_PARAMS = dict(
    n_estimators=300, max_depth=12, min_samples_leaf=20,
    class_weight="balanced", random_state=SEED, n_jobs=-1,
)

XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="aucpr", random_state=SEED, n_jobs=-1,
    reg_alpha=0.1, reg_lambda=1.0,
)

LGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1, verbose=-1,
    reg_alpha=0.1, reg_lambda=1.0,
)

# Stacking base estimators (lighter versions)
STACK_LR = dict(max_iter=1000, class_weight="balanced", random_state=SEED)
STACK_RF = dict(n_estimators=200, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1)
STACK_XGB = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    eval_metric="aucpr", random_state=SEED, n_jobs=-1, verbosity=0,
)

# ── Explainability ────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 2000

# ── Fairness ──────────────────────────────────────────────────────
FAIRNESS_MIN_N = 50
FAIRNESS_MIN_POS = 5
FAIRNESS_GAP_THRESHOLD = 0.05  # 5 pp

# ── Operational ───────────────────────────────────────────────────
TIER_LOW_PERCENTILE = 15
TIER_HIGH_PERCENTILE = 85
OUTREACH_EXPLANATION_N = 500
OUTREACH_TOP_REASONS = 3
