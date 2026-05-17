"""
10_export_dashboard_data.py — Export all results to dashboard-ready JSON.

Collects accuracy, F1, precision, recall, training time, model complexity,
and statistical significance into a single dashboard_data.json that feeds
the interactive HTML comparison dashboard.

Usage:
  python src/10_export_dashboard_data.py

Output:
  - dashboard/dashboard_data.json
"""
import json
import time
import logging
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

from config import MODEL_DIR, OUTPUT_DIR, RESULTS_FILE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True)
DASHBOARD_DATA = DASHBOARD_DIR / "dashboard_data.json"


# Human-readable metadata for each system
SYSTEM_META: dict[str, dict] = {
    "A_naive_bayes":        {"label": "Naive Bayes",         "family": "Probabilistic",   "complexity": 1},
    "B_svm":                {"label": "SVM",                  "family": "Kernel Method",   "complexity": 3},
    "C_decision_tree":      {"label": "Decision Tree",        "family": "Tree",            "complexity": 2},
    "D_random_forest":      {"label": "Random Forest",        "family": "Ensemble",        "complexity": 4},
    "E_knn":                {"label": "KNN",                  "family": "Instance-Based",  "complexity": 2},
    "F_gradient_boosting":  {"label": "Gradient Boosting",   "family": "Ensemble",        "complexity": 5},
    "G_logistic_regression":{"label": "Logistic Regression", "family": "Linear",          "complexity": 1},
    "H_dnn_shallow":        {"label": "DNN (Shallow)",        "family": "Neural Network",  "complexity": 4},
    "I_dnn_deep":           {"label": "DNN (Deep)",           "family": "Neural Network",  "complexity": 6},
}


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed arrays."""
    X_train = np.load(MODEL_DIR / "X_train.npy")
    y_train = np.load(MODEL_DIR / "y_train.npy")
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    return X_train, y_train, X_test, y_test


def measure_inference_time(clf, X_test: np.ndarray, n_runs: int = 5) -> float:
    """Measure average inference time in milliseconds.

    Args:
        clf:    Fitted estimator with .predict() method.
        X_test: Test features.
        n_runs: Number of timing runs to average.

    Returns:
        Mean inference time in milliseconds.
    """
    import time
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        clf.predict(X_test)
        end = time.perf_counter()
        times.append(end - start)
    
    return (np.mean(times) * 1000)  # Convert to milliseconds


def build_dashboard_data(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Build the full dashboard data structure.

    Args:
        X_train, y_train: Training data (for training time measurement).
        X_test, y_test:   Test data (for all metrics).

    Returns:
        Dict ready to be serialized as dashboard_data.json.
    """
    import tensorflow as tf
    
    classifiers_data = []
    
    # Load results.json
    if not RESULTS_FILE.exists():
        logger.warning("results.json not found, creating minimal dashboard data")
        results_list = []
    else:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results_list = json.load(f)
    
    # For each system in results
    for result in results_list:
        system_id = result.get("system", "")
        clf_name = result.get("classifier", "")
        config = result.get("config", "")
        accuracy = result.get("accuracy", 0.0)
        f1 = result.get("f1", result.get("f1_weighted", 0.0))
        precision = result.get("precision", 0.0)
        recall = result.get("recall", 0.0)
        
        # Try to load model and measure inference time
        inference_ms = 0.0
        pkl_path = MODEL_DIR / f"{system_id}.pkl"
        keras_path = MODEL_DIR / f"{system_id}.keras"
        
        try:
            if pkl_path.exists():
                clf = joblib.load(pkl_path)
                inference_ms = measure_inference_time(clf, X_test, n_runs=3)
            elif keras_path.exists():
                clf = tf.keras.models.load_model(keras_path)
                start = time.perf_counter()
                clf.predict(X_test, verbose=0)
                end = time.perf_counter()
                inference_ms = (end - start) * 1000
        except Exception as e:
            logger.warning("Could not measure inference time for %s: %s", system_id, e)
        
        # Get metadata from SYSTEM_META
        meta = SYSTEM_META.get(system_id, {
            "label": clf_name,
            "family": "Unknown",
            "complexity": 3
        })
        
        classifier_entry = {
            "id": system_id,
            "label": meta.get("label", clf_name),
            "family": meta.get("family", "Unknown"),
            "complexity": meta.get("complexity", 3),
            "config": config,
            "metrics": {
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "inference_ms": round(inference_ms, 2),
            }
        }
        
        classifiers_data.append(classifier_entry)
    
    # Dataset metadata
    unique_classes = len(np.unique(y_train))
    class_counts = {}
    for label in np.unique(y_train):
        count = np.sum(y_train == label)
        class_counts[f"Class_{label}"] = int(count)
    
    dashboard_data = {
        "classifiers": classifiers_data,
        "dataset": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(X_train.shape[1]),
            "n_classes": int(unique_classes),
            "class_distribution": class_counts,
        }
    }
    
    return dashboard_data


if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test = load_data()
        data = build_dashboard_data(X_train, y_train, X_test, y_test)

        with open(DASHBOARD_DATA, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("✓ Dashboard data exported to %s", DASHBOARD_DATA)
        logger.info("✓ Script completed successfully")
        assert DASHBOARD_DATA.exists(), "Dashboard data file not created"
    except FileNotFoundError as e:
        logger.error("Error: %s — Please run preprocessing and classifier scripts first.", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e)