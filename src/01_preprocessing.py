"""
01_preprocessing.py — Load, clean, and preprocess ARFF data.

Outputs:
  - outputs/models/preprocessor.pkl  (fitted ColumnTransformer pipeline)
  - X_train.npy, y_train.npy, X_test.npy, y_test.npy  in outputs/models/

Usage:
  python src/01_preprocessing.py
"""
import logging
import numpy as np
import pandas as pd
import joblib
from scipy.io import arff
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from config import (
    TRAIN_FILE, TEST_FILE, MODEL_DIR, PREPROCESSOR_FILE,
    TARGET_COL, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── 1. Load ARFF ─────────────────────────────────────────────────────────────

def load_arff(filepath) -> pd.DataFrame:
    """Load an ARFF file and return a clean pandas DataFrame.

    Byte strings from arff loader are decoded to str.
    Missing values '?' are replaced with np.nan.

    Args:
        filepath: Path to the .arff file.

    Returns:
        DataFrame with decoded strings and NaN for missing values.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # Decode byte strings to str
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    
    # Replace '?' with np.nan
    df = df.replace("?", np.nan)
    
    return df


# ── 2. Split features / target ────────────────────────────────────────────────

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Separate feature columns from the target column.

    Args:
        df: Full DataFrame including target column.

    Returns:
        Tuple of (X DataFrame, y numpy array as str labels).
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    return X, y


# ── 3. Build preprocessing pipeline ──────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that handles numeric and categorical columns.

    Numeric columns:  SimpleImputer(median) → StandardScaler
    Categorical cols: SimpleImputer(most_frequent) → OrdinalEncoder

    Args:
        X: Feature DataFrame (used only to detect column types).

    Returns:
        Unfitted ColumnTransformer pipeline.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    
    return preprocessor


# ── 4. Encode labels ──────────────────────────────────────────────────────────

def encode_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Fit LabelEncoder on train labels and transform both sets.

    Args:
        y_train: Raw string labels for training.
        y_test:  Raw string labels for test.

    Returns:
        Tuple of (y_train_enc, y_test_enc, fitted LabelEncoder).
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


# ── 5. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Loading training data from %s", TRAIN_FILE)
    df_train = load_arff(TRAIN_FILE)

    logger.info("Loading test data from %s", TEST_FILE)
    df_test = load_arff(TEST_FILE)

    logger.info("Class distribution (train):\n%s", df_train[TARGET_COL].value_counts())

    X_train, y_train = split_features_target(df_train)
    X_test,  y_test  = split_features_target(df_test)

    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    # Save
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    np.save(MODEL_DIR / "X_train.npy", X_train_proc)
    np.save(MODEL_DIR / "y_train.npy", y_train_enc)
    np.save(MODEL_DIR / "X_test.npy",  X_test_proc)
    np.save(MODEL_DIR / "y_test.npy",  y_test_enc)

    logger.info("Preprocessing complete. Files saved to %s", MODEL_DIR)
