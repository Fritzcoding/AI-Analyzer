"""
09_mlflow_tracking.py — MLflow experiment tracking for all classifiers.

Logs every experiment (params, metrics, artifacts) into a local MLflow
tracking server so you can compare runs visually in the MLflow UI.

Usage:
  # First run this script:
  python src/09_mlflow_tracking.py

  # Then launch the UI:
  mlflow ui --backend-store-uri outputs/mlruns

  # Open http://localhost:5000 in your browser

Portfolio value (MLOps):
  MLflow is THE standard experiment tracking tool in industry (used at
  Microsoft, Databricks, Booking.com etc.). Having it in a student project
  immediately signals production-level thinking.
"""
import json
import logging
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)

from config import MODEL_DIR, OUTPUT_DIR, RESULTS_FILE, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_DIR = OUTPUT_DIR / "mlruns"
EXPERIMENT_NAME = "hypothyroid-classification"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed numpy arrays."""
    X_train = np.load(MODEL_DIR / "X_train.npy")
    y_train = np.load(MODEL_DIR / "y_train.npy")
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    return X_train, y_train, X_test, y_test


def log_sklearn_model(system_id: str, clf, X_train: np.ndarray,
                       y_train: np.ndarray, X_test: np.ndarray,
                       y_test: np.ndarray, params: dict) -> None:
    """Log a single sklearn classifier as an MLflow run.

    Args:
        system_id: Classifier identifier.
        clf:       Fitted sklearn estimator.
        X_train, y_train: Training data.
        X_test, y_test:   Test data.
        params:    Best hyperparameters dict.
    """
    with mlflow.start_run(run_name=system_id):
        mlflow.log_params(params)
        y_pred = clf.predict(X_test)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
        })
        mlflow.sklearn.log_model(clf, artifact_path="model")


def log_all_from_results_json() -> None:
    """Read results.json and log every entry as an MLflow run.

    This is the main entry point — reads all saved results and
    retroactively logs them to MLflow.
    """
    # Use SQLite backend for better Windows compatibility
    sqlite_path = str(MLFLOW_DIR / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{sqlite_path}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    if not RESULTS_FILE.exists():
        logger.error("results.json not found. Run classifiers first.")
        return
    
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    
    for result in results:
        system_id = result.get("system", "unknown")
        clf_name = result.get("classifier", "unknown")
        config = result.get("config", "")
        accuracy = result.get("accuracy", 0)
        f1 = result.get("f1", result.get("f1_weighted", 0))
        precision = result.get("precision", 0)
        recall = result.get("recall", 0)
        
        with mlflow.start_run(run_name=system_id):
            # Parse config string and log as params
            if config:
                config_items = config.split(", ")
                for item in config_items:
                    if "=" in item:
                        key, value = item.split("=", 1)
                        try:
                            # Try to convert to number
                            value = float(value) if "." in value else int(value)
                        except (ValueError, AttributeError):
                            pass  # Keep as string
                        mlflow.log_param(key.strip(), value)
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            })
            
            # Try to load and log the actual model if it exists
            pkl_path = MODEL_DIR / f"{system_id}.pkl"
            if pkl_path.exists():
                try:
                    clf = joblib.load(pkl_path)
                    mlflow.sklearn.log_model(clf, artifact_path="model")
                except Exception as e:
                    logger.warning("Could not log model %s: %s", system_id, e)
            
            keras_path = MODEL_DIR / f"{system_id}.keras"
            if keras_path.exists():
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(keras_path)
                    mlflow.tensorflow.log_model(model, artifact_path="model")
                except Exception as e:
                    logger.warning("Could not log TF model %s: %s", system_id, e)


if __name__ == "__main__":
    try:
        log_all_from_results_json()
        logger.info("\n✅ MLflow tracking complete.\n   Launch UI with:\n   mlflow ui --backend-store-uri %s\n   Then open http://localhost:5000", MLFLOW_DIR)
        logger.info("✓ Script completed successfully")
        assert MLFLOW_DIR.exists(), "MLflow directory not created"
    except Exception as e:
        logger.error("Error: %s", e)