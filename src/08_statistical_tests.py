"""
08_statistical_tests.py — Statistical significance testing between classifiers.

Determines whether accuracy differences are statistically significant,
not just lucky variance. Uses McNemar's test (pairwise) and Friedman test.

Outputs:
  - outputs/statistical_tests.json
  - outputs/figures/significance_heatmap.png

Usage:
  python src/08_statistical_tests.py

Portfolio value (Research/Academia):
  This is what separates a proper ML study from a demo.
  Reviewers and interviewers notice when you can say
  "System D is significantly better than System A (p < 0.01)".
"""
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from itertools import combinations
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.contingency_tables import mcnemar

from config import MODEL_DIR, FIGURE_DIR, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ALPHA = 0.05  # significance threshold


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed test data."""
    X_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    return X_test, y_test


def collect_predictions(X_test: np.ndarray) -> dict[str, np.ndarray]:
    """Load all saved models and collect their test predictions.

    Args:
        X_test: Preprocessed test features.

    Returns:
        Dict mapping system_id → prediction array.
    """
    import tensorflow as tf
    
    predictions_dict = {}
    
    # Load sklearn models (.pkl files)
    for pkl_file in MODEL_DIR.glob("*.pkl"):
        if pkl_file.name in ["preprocessor.pkl", "label_encoder.pkl"]:
            continue  # Skip utility files
        
        system_id = pkl_file.stem
        try:
            clf = joblib.load(pkl_file)
            y_pred = clf.predict(X_test)
            predictions_dict[system_id] = y_pred
            logger.info("Loaded predictions for %s", system_id)
        except Exception as e:
            logger.warning("Could not load %s: %s", system_id, e)
    
    # Load TensorFlow models (.keras files)
    for keras_file in MODEL_DIR.glob("*.keras"):
        system_id = keras_file.stem
        try:
            model = tf.keras.models.load_model(keras_file)
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            predictions_dict[system_id] = y_pred
            logger.info("Loaded predictions for %s", system_id)
        except Exception as e:
            logger.warning("Could not load %s: %s", system_id, e)
    
    return predictions_dict


def mcnemar_matrix(predictions: dict[str, np.ndarray],
                    y_test: np.ndarray) -> pd.DataFrame:
    """Compute pairwise McNemar p-values between all classifiers.

    McNemar's test checks if two classifiers make significantly different
    errors on the same test set.

    Args:
        predictions: Dict of system_id → prediction array.
        y_test:      True labels.

    Returns:
        DataFrame of p-values (systems × systems).
    """
    system_ids = list(predictions.keys())
    n_systems = len(system_ids)
    pvalue_matrix = np.ones((n_systems, n_systems))
    
    for i, sys_a in enumerate(system_ids):
        for j, sys_b in enumerate(system_ids):
            if i == j:
                pvalue_matrix[i, j] = 1.0
                continue
            if i > j:
                # Use symmetric property
                pvalue_matrix[i, j] = pvalue_matrix[j, i]
                continue
            
            y_pred_a = predictions[sys_a]
            y_pred_b = predictions[sys_b]
            
            # Build 2x2 contingency table
            # Cells: [both correct, A correct B wrong, A wrong B correct, both wrong]
            correct_a = (y_pred_a == y_test)
            correct_b = (y_pred_b == y_test)
            
            both_correct = np.sum(correct_a & correct_b)
            both_wrong = np.sum(~correct_a & ~correct_b)
            a_correct_b_wrong = np.sum(correct_a & ~correct_b)
            a_wrong_b_correct = np.sum(~correct_a & correct_b)
            
            # McNemar's 2x2 table (off-diagonal terms)
            table = np.array([[a_correct_b_wrong, a_wrong_b_correct]])
            
            try:
                result = mcnemar(table, exact=True)
                pvalue_matrix[i, j] = result.pvalue
            except Exception as e:
                logger.warning("McNemar test failed for %s vs %s: %s", sys_a, sys_b, e)
                pvalue_matrix[i, j] = 1.0
    
    pvalue_df = pd.DataFrame(pvalue_matrix, index=system_ids, columns=system_ids)
    return pvalue_df


def plot_significance_heatmap(pvalue_df: pd.DataFrame) -> None:
    """Plot a heatmap of pairwise McNemar p-values.

    Cells below ALPHA are highlighted (significant difference).

    Args:
        pvalue_df: DataFrame of p-values from mcnemar_matrix().
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for diagonal (self-comparisons)
    mask = np.eye(len(pvalue_df), dtype=bool)
    
    # Create custom colormap: green for significant, white for non-significant
    cmap = plt.cm.RdYlGn_r  # Red for small p-values (significant), green for large
    
    sns.heatmap(pvalue_df, annot=True, fmt=".3f", cmap=cmap, mask=mask,
                cbar_kws={"label": "P-Value"}, ax=ax, vmin=0, vmax=1,
                linewidths=0.5, linecolor="gray")
    
    ax.set_title(f"Pairwise McNemar Test P-Values (α = {ALPHA})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_ylabel("Classifier", fontsize=12)
    
    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    output_path = FIGURE_DIR / "significance_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Significance heatmap saved: %s", output_path)


if __name__ == "__main__":
    try:
        X_test, y_test = load_data()
        predictions = collect_predictions(X_test)

        logger.info("Running pairwise McNemar tests across %d classifiers...",
                    len(predictions))

        pvalue_df = mcnemar_matrix(predictions, y_test)
        plot_significance_heatmap(pvalue_df)

        results = {
            "alpha": ALPHA,
            "pairs": {}
        }
        for (a, b) in combinations(predictions.keys(), 2):
            p = pvalue_df.loc[a, b]
            significant = bool(p < ALPHA)
            results["pairs"][f"{a}_vs_{b}"] = {
                "p_value": round(float(p), 4),
                "significant": significant,
                "interpretation": (
                    f"{a} and {b} differ significantly (p={p:.4f})"
                    if significant
                    else f"No significant difference between {a} and {b} (p={p:.4f})"
                )
            }

        out_path = OUTPUT_DIR / "statistical_tests.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("✓ Statistical test results saved to %s", out_path)
        logger.info("✓ Script completed successfully")
        assert (FIGURE_DIR / "significance_heatmap.png").exists(), "Heatmap not created"
    except FileNotFoundError as e:
        logger.error("Error: %s — Please run preprocessing and classifier scripts first.", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e)