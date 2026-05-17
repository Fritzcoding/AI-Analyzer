"""
config.py — Central configuration for all scripts.
All file paths and shared constants live here.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
FIGURE_DIR  = OUTPUT_DIR / "figures"
REPORT_DIR  = ROOT_DIR / "report"

TRAIN_FILE  = DATA_DIR / "hypothyroid_cjlin2025_training.arff"
TEST_FILE   = DATA_DIR / "hypothyroid_cjlin2025_test.arff"
RESULTS_FILE = OUTPUT_DIR / "results.json"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"

# ── Create dirs on import ────────────────────────────────────────────────────
for _dir in [OUTPUT_DIR, MODEL_DIR, FIGURE_DIR, REPORT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Shared constants ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET_COL   = "Class"
CV_FOLDS     = 5
TEST_SIZE    = 0.2   # only used if splitting train set for validation

# ── TF Training defaults ─────────────────────────────────────────────────────
DNN_EPOCHS        = 100
DNN_PATIENCE      = 10
DNN_BATCH_SIZE    = 64
