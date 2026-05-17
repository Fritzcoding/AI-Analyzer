"""
04_evaluate.py — Unified evaluation: load results.json, generate comparison figures.

Outputs:
  - outputs/figures/accuracy_comparison.png  (bar chart of all systems)
  - outputs/figures/roc_curves.png           (ROC curves, OvR)

Usage:
  python src/04_evaluate.py
"""
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from config import RESULTS_FILE, FIGURE_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Try to use a CJK-compatible font (for Traditional Chinese labels in figures)
# Falls back gracefully if not installed
font_families = [
    "Noto Sans CJK TC",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans"
]

for font in font_families:
    try:
        matplotlib.rcParams["font.family"] = font
        break
    except Exception:
        continue


def load_results() -> list[dict]:
    """Load all classifier results from results.json.

    Returns:
        List of result dicts.

    Raises:
        FileNotFoundError: If results.json has not been generated yet.
    """
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"results.json not found at {RESULTS_FILE}. "
            "Run 02_sklearn_classifiers.py and 03_tensorflow_dnn.py first."
        )
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy_comparison(results: list[dict]) -> None:
    """Generate a horizontal bar chart comparing all systems' test accuracy, F1, and Precision.

    Args:
        results: List of result dicts from results.json.
    """
    # Sort by accuracy descending
    results_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    
    systems = [r["system"] for r in results_sorted]
    accuracies = [r["accuracy"] * 100 for r in results_sorted]
    f1_scores = [r.get("f1", r.get("f1_weighted", 0)) * 100 for r in results_sorted]  # DNN has "f1" key
    precisions = [r.get("precision", 0) * 100 for r in results_sorted]
    
    x = np.arange(len(systems))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x, f1_scores, width, label="F1 (Macro)", color="darkorange")
    bars3 = ax.bar(x + width, precisions, width, label="Precision (Macro)", color="seagreen")
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("System (Classifier)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("各分類器性能對比：正確率 vs F1分數 vs 精確率", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=45, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    
    output_path = FIGURE_DIR / "accuracy_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Accuracy comparison chart saved: %s", output_path)


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table to stdout with multiple metrics.

    Args:
        results: List of result dicts.
    """
    # Sort by accuracy descending
    results_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    
    print("\n" + "=" * 150)
    print(f"{'系統':<12} | {'分類器':<20} | {'正確率':<10} | {'F1分數':<10} | {'精確率':<10} | {'召回率':<10}")
    print("-" * 150)
    
    for r in results_sorted:
        system_id = r["system"]
        classifier = r["classifier"][:20]
        accuracy = r["accuracy"] * 100
        f1 = r.get("f1", r.get("f1_weighted", 0)) * 100
        precision = r.get("precision", 0) * 100
        recall = r.get("recall", 0) * 100
        
        print(f"{system_id:<12} | {classifier:<20} | {accuracy:>7.2f}% | {f1:>7.2f}% | {precision:>7.2f}% | {recall:>7.2f}%")
    
    print("=" * 150)
    print("\nKey Performance Metrics:")
    best_result = results_sorted[0]
    print(f"  ✓ Best Accuracy: {best_result['system']} ({best_result['classifier']}) - {best_result['accuracy']*100:.2f}%")
    
    # Find best F1
    best_f1_result = max(results_sorted, key=lambda r: r.get("f1", r.get("f1_weighted", 0)))
    if best_f1_result["f1"] > 0 if "f1" in best_f1_result else best_f1_result.get("f1_weighted", 0) > 0:
        print(f"  ✓ Best F1-Score: {best_f1_result['system']} - {best_f1_result.get('f1', best_f1_result.get('f1_weighted', 0))*100:.2f}%")
    
    print()


if __name__ == "__main__":
    results = load_results()
    logger.info("Loaded %d classifier results.", len(results))
    print_summary_table(results)
    plot_accuracy_comparison(results)
    logger.info("Evaluation figures saved to %s", FIGURE_DIR)
