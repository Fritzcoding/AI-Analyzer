"""
07_optuna_tuning.py — Bayesian hyperparameter optimization with Optuna.

Replaces brute-force GridSearchCV with intelligent search for the top
classifiers. Typically finds better hyperparameters in fewer trials.

Outputs:
  - outputs/models/optuna_{system_id}.pkl    (best model)
  - outputs/figures/optuna_{system_id}_history.png
  - outputs/optuna_results.json

Usage:
  python src/07_optuna_tuning.py

Portfolio value:
  Optuna is used at Preferred Networks, Uber, and across industry.
  It demonstrates you know how to move beyond grid search.
  The optimization history plot makes a great portfolio visual.
"""
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from config import MODEL_DIR, FIGURE_DIR, OUTPUT_DIR, RANDOM_STATE, CV_FOLDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50  # Increase for better results if time allows


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed numpy arrays."""
    X_train = np.load(MODEL_DIR / "X_train.npy")
    y_train = np.load(MODEL_DIR / "y_train.npy")
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    return X_train, y_train, X_test, y_test


# ── Objective functions per classifier ───────────────────────────────────────

def rf_objective(trial: optuna.Trial, X_train: np.ndarray,
                  y_train: np.ndarray) -> float:
    """Optuna objective for Random Forest.

    Args:
        trial:   Optuna Trial object (used to suggest hyperparams).
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Mean cross-validation accuracy (to maximize).
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    clf = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
    from sklearn.model_selection import cross_val_score
    return cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, scoring="accuracy").mean()


def svm_objective(trial: optuna.Trial, X_train: np.ndarray,
                   y_train: np.ndarray) -> float:
    """Optuna objective for SVM."""
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
    clf = SVC(**params, probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    from sklearn.model_selection import cross_val_score
    return cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, scoring="accuracy").mean()


def gb_objective(trial: optuna.Trial, X_train: np.ndarray,
                  y_train: np.ndarray) -> float:
    """Optuna objective for Gradient Boosting."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    clf = GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
    from sklearn.model_selection import cross_val_score
    return cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, scoring="accuracy").mean()


OBJECTIVES = {
    "C_rf_optuna": (rf_objective, RandomForestClassifier),
    "B_svm_optuna": (svm_objective, SVC),
    "D_gb_optuna": (gb_objective, GradientBoostingClassifier),
}


def run_study(system_id: str, objective_fn, X_train: np.ndarray,
              y_train: np.ndarray, X_test: np.ndarray,
              y_test: np.ndarray) -> dict:
    """Run an Optuna study for one classifier and return results.

    Args:
        system_id:    Key for this tuned model.
        objective_fn: Objective function to maximize.
        X_train, y_train: Training data.
        X_test, y_test:   Test data.

    Returns:
        Result dict matching the format in results.json.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    # Create study with TPE sampler
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: objective_fn(t, X_train, y_train), n_trials=N_TRIALS)
    
    logger.info("Best trial value: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)
    
    # Plot optimization history
    fig, ax = plt.subplots(figsize=(10, 6))
    values = [t.value for t in study.trials if t.value is not None]
    ax.plot(range(len(values)), values, marker="o", linestyle="-", markersize=4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV Accuracy")
    ax.set_title(f"Optuna Optimization History: {system_id}")
    ax.grid(True, alpha=0.3)
    history_path = FIGURE_DIR / f"optuna_{system_id}_history.png"
    plt.tight_layout()
    plt.savefig(history_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Optimization history saved: %s", history_path)
    
    # Retrain model with best params
    # Determine which estimator class to use
    if "RandomForest" in str(objective_fn):
        best_model = RandomForestClassifier(
            **study.best_params, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
        )
    elif "SVM" in str(objective_fn) or "svm" in system_id:
        best_model = SVC(
            **study.best_params, probability=True, random_state=RANDOM_STATE, class_weight="balanced"
        )
    elif "Gradient" in str(objective_fn) or "gb" in system_id:
        best_model = GradientBoostingClassifier(**study.best_params, random_state=RANDOM_STATE)
    else:
        logger.warning("Cannot determine model type for %s", system_id)
        return None
    
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Save model
    model_path = MODEL_DIR / f"{system_id}.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Model saved: %s", model_path)
    
    # Format config string
    config_items = [f"{k}={v}" for k, v in study.best_params.items()]
    config_str = ", ".join(config_items)
    
    result = {
        "system": system_id,
        "classifier": study.best_params.get("kernel", "Optuna"),
        "config": config_str,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    return result


if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test = load_data()
        all_results = []

        for system_id, (obj_fn, _) in OBJECTIVES.items():
            logger.info("Running Optuna for %s (%d trials)...", system_id, N_TRIALS)
            result = run_study(system_id, obj_fn, X_train, y_train, X_test, y_test)
            if result:
                all_results.append(result)
                logger.info("Optuna test accuracy: %.4f for %s", result["accuracy"], system_id)

        out_path = OUTPUT_DIR / "optuna_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("✓ Optuna results saved to %s", out_path)
        logger.info("✓ Script completed successfully")
        assert out_path.exists(), "Optuna results file not created"
    except FileNotFoundError as e:
        logger.error("Error: %s — Please run 01_preprocessing.py and 02_sklearn_classifiers.py first.", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e)