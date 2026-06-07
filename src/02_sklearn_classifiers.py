"""
02_sklearn_classifiers.py — Train and evaluate all scikit-learn classifiers.

Reads preprocessed data from outputs/models/*.npy
Writes per-classifier model files and results to outputs/results.json

Usage:
  python src/02_sklearn_classifiers.py
"""
import json
import logging
import numpy as np
import joblib
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from sklearn.metrics import (
    accuracy_score, classification_report
)

from config import MODEL_DIR, RESULTS_FILE, CV_FOLDS, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Classifier registry ───────────────────────────────────────────────────────
# Format: "system_id": (estimator, param_grid, display_name)
# Each classifier is configured to handle class imbalance
CLASSIFIERS: dict[str, tuple] = {
    "A_naive_bayes": (
        GaussianNB(var_smoothing=1e-9),
        {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        "Gaussian Naive Bayes (with sample weighting)"
    ),
    "B_svm": (
        SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        "Support Vector Machine (RBF/Linear)"
    ),
    "C_random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15, None], "min_samples_split": [2, 5]},
        "Random Forest"
    ),
    "D_gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]},
        "Gradient Boosting (XGBoost-like)"
    ),
    "E_decision_tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {"max_depth": [5, 10, 15, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
        "Decision Tree (Deep)"
    ),
    "F_logistic_regression": (
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced"),
        {"C": [0.001, 0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"], "penalty": ["l2"]},
        "Logistic Regression (L2 Regularized)"
    ),
    "G_adaboost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200, 300], "learning_rate": [0.5, 1.0, 1.5]},
        "AdaBoost Ensemble"
    ),
}

# Add XGBoost if available (better handling of imbalanced data)
if HAS_XGBOOST:
    CLASSIFIERS["H_xgboost"] = (
        XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=10, use_label_encoder=False, eval_metric="logloss"),
        {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5, 7]},
        "XGBoost (Imbalance-Aware Gradient Boosting)"
    )


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed numpy arrays from disk.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    X_train = np.load(MODEL_DIR / "X_train.npy")
    y_train = np.load(MODEL_DIR / "y_train.npy")
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    return X_train, y_train, X_test, y_test


def train_and_evaluate(system_id: str, estimator, param_grid: dict,
                        clf_name: str, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Run GridSearchCV, evaluate on test set, return comprehensive result dict.

    Args:
        system_id:  Key for this classifier (e.g. 'A_naive_bayes').
        estimator:  Unfitted sklearn estimator.
        param_grid: Hyperparameter grid for GridSearchCV.
        clf_name:   Display name for logging and report.
        X_train, y_train: Training data.
        X_test, y_test:   Test data.

    Returns:
        Dict with keys: system, classifier, config, accuracy, precision, recall, f1, weighted_f1.
    """
    try:
        # Compute sample weights for balanced class handling (especially for Naive Bayes)
        sample_weights = compute_sample_weight("balanced", y_train)
        
        # GridSearchCV with multiple scoring metrics
        gs = GridSearchCV(
            estimator, param_grid, 
            cv=CV_FOLDS, 
            scoring="accuracy",  # Primary metric for optimization
            n_jobs=-1,
            verbose=0
        )
        gs.fit(X_train, y_train, sample_weight=sample_weights)
        
        logger.info("Best params: %s", gs.best_params_)
        logger.info("Best cross-val accuracy: %.4f", gs.best_score_)
        
        # Test predictions
        y_pred = gs.best_estimator_.predict(X_test)
        
        # Comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Macro and weighted averages (better for imbalanced data)
        precision_macro = report.get("macro avg", {}).get("precision", 0.0)
        recall_macro = report.get("macro avg", {}).get("recall", 0.0)
        f1_macro = report.get("macro avg", {}).get("f1-score", 0.0)
        f1_weighted = report.get("weighted avg", {}).get("f1-score", 0.0)
        
        # Save best estimator
        model_path = MODEL_DIR / f"{system_id}.pkl"
        joblib.dump(gs.best_estimator_, model_path)
        logger.info("Model saved: %s", model_path)
        
        # Format config string
        config_items = [f"{k}={v}" for k, v in gs.best_params_.items()]
        config_str = ", ".join(config_items)
        
        result = {
            "system": system_id,
            "classifier": clf_name,
            "config": config_str,
            "accuracy": accuracy,
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro,
            "f1_weighted": f1_weighted,
        }
        
        return result
        
    except Exception as e:
        logger.error("Error training %s: %s", clf_name, str(e))
        return None


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    all_results: list[dict] = []

    for system_id, (estimator, param_grid, clf_name) in CLASSIFIERS.items():
        logger.info("=" * 60)
        logger.info("Training: %s (%s)", clf_name, system_id)
        result = train_and_evaluate(
            system_id, estimator, param_grid, clf_name,
            X_train, y_train, X_test, y_test
        )
        if result:
            all_results.append(result)
            logger.info("Accuracy: %.4f", result["accuracy"])

    # Save all results to JSON
    existing: list[dict] = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)

    # Merge: replace existing entries with same system id
    existing_ids = {r["system"] for r in existing}
    merged = [r for r in existing if r["system"] not in {r2["system"] for r2 in all_results}]
    merged.extend(all_results)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info("Results saved to %s", RESULTS_FILE)
