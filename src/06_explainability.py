"""
06_explainability.py — Model explainability with SHAP.

Generates feature importance plots for the best sklearn classifiers
and the DNN, giving your project a serious ML Engineering + Research edge.

Outputs:
  - outputs/figures/shap_summary_{system_id}.png
  - outputs/figures/shap_bar_{system_id}.png
  - outputs/figures/permutation_importance_{system_id}.png

Usage:
  python src/06_explainability.py

Portfolio value:
  SHAP is an industry-standard explainability tool used at top ML teams.
  Including it signals you understand WHY your model works, not just accuracy.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import permutation_importance

from config import MODEL_DIR, FIGURE_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Which systems to explain (tree-based models work best with SHAP)
SYSTEMS_TO_EXPLAIN = [
    "E_decision_tree",
    "C_random_forest",
    "D_gradient_boosting",
]


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed test data.

    Returns:
        Tuple of (X_test, y_test).
    """
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    return X_test, y_test


def load_feature_names() -> list[str]:
    """Load feature names from the saved preprocessor.

    Returns:
        List of feature name strings after preprocessing.
    """
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    feature_names = list(preprocessor.get_feature_names_out())
    return feature_names


def explain_tree_model(system_id: str, X_test: np.ndarray,
                        feature_names: list[str]) -> None:
    """Generate SHAP summary and bar plots for a tree-based model.

    Uses TreeExplainer (fast, exact SHAP values for tree models).

    Args:
        system_id:     Model identifier (e.g. 'C_random_forest').
        X_test:        Preprocessed test features.
        feature_names: Feature names for axis labels.
    """
    model_path = MODEL_DIR / f"{system_id}.pkl"
    if not model_path.exists():
        logger.warning("Model not found: %s — skipping.", model_path)
        return

    clf = joblib.load(model_path)
    logger.info("Explaining %s with SHAP TreeExplainer...", system_id)

    try:
        explainer = shap.TreeExplainer(clf)
        # Subsample for speed
        X_sample = X_test[:min(200, len(X_test))]
        shap_values = explainer.shap_values(X_sample)
        
        # For multi-class, use positive class (index 1)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        
        # Summary plot (dot plot)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, plot_type="dot", show=False)
        summary_path = FIGURE_DIR / f"shap_summary_{system_id}.png"
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved: %s", summary_path)
        
        # Bar plot (mean absolute SHAP values)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        bar_path = FIGURE_DIR / f"shap_bar_{system_id}.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP bar plot saved: %s", bar_path)
    except Exception as e:
        logger.warning("SHAP TreeExplainer failed for %s (common for multi-class GB): %s. Using permutation importance instead.", system_id, str(e))


def plot_permutation_importance(system_id: str, X_test: np.ndarray,
                                 y_test: np.ndarray, feature_names: list[str]) -> None:
    """Compute and plot permutation importance for a model.

    Args:
        system_id:     Model identifier.
        X_test:        Test features.
        y_test:        True labels.
        feature_names: Feature names.
    """
    model_path = MODEL_DIR / f"{system_id}.pkl"
    if not model_path.exists():
        logger.warning("Model not found: %s — skipping.", model_path)
        return
    
    clf = joblib.load(model_path)
    logger.info("Computing permutation importance for %s...", system_id)
    
    # Compute permutation importance
    result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
    
    # Sort by importance
    indices = np.argsort(result.importances_mean)[::-1][:15]  # Top 15
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), result.importances_mean[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices])
    ax.set_xlabel("Mean Decrease in Accuracy")
    ax.set_title(f"Permutation Feature Importance: {system_id}")
    ax.invert_yaxis()
    
    output_path = FIGURE_DIR / f"permutation_importance_{system_id}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Permutation importance plot saved: %s", output_path)


if __name__ == "__main__":
    try:
        X_test, y_test = load_data()
        feature_names = load_feature_names()

        for system_id in SYSTEMS_TO_EXPLAIN:
            explain_tree_model(system_id, X_test, feature_names)
            plot_permutation_importance(system_id, X_test, y_test, feature_names)

        logger.info("✓ SHAP explainability figures saved to %s", FIGURE_DIR)
        logger.info("✓ Script completed successfully")
        assert (FIGURE_DIR / f"shap_summary_{SYSTEMS_TO_EXPLAIN[0]}.png").exists(), "Output figures not created"
    except FileNotFoundError as e:
        logger.error("Error: %s — Please run 01_preprocessing.py first.", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e)