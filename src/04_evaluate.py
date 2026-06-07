"""
04_evaluate.py - Advanced evaluation and visualization pipeline for Hypothyroid ML.

Outputs:
  - outputs/figures/*.png (research-style plots)
  - outputs/results.json (updated with advanced metrics)
  - outputs/analysis_summary.json (dataset/model diagnostics)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.io import arff
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import label_binarize

from config import FIGURE_DIR, MODEL_DIR, RESULTS_FILE, TRAIN_FILE, TEST_FILE, RANDOM_STATE, CV_FOLDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

ANALYSIS_SUMMARY_FILE = RESULTS_FILE.parent / "analysis_summary.json"
TOP_N_FEATURES = 15
PLOT_DPI = 220
MODEL_FAMILY_COLORS = {
    "Tree ensemble": "#1b9e77",
    "Single tree": "#66a61e",
    "Linear/kernel": "#386cb0",
    "Neural network": "#984ea3",
    "Probabilistic": "#e6ab02",
}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.facecolor"] = "#fbfcfe"
plt.rcParams["axes.facecolor"] = "#ffffff"
plt.rcParams["savefig.facecolor"] = "#fbfcfe"


def _load_arff_dataframe(file_path: Path) -> pd.DataFrame:
    data, _ = arff.loadarff(file_path)
    frame = pd.DataFrame(data)
    for col in frame.columns:
        if frame[col].dtype == "object":
            frame[col] = frame[col].apply(lambda value: value.decode("utf-8") if isinstance(value, bytes) else value)
    return frame.replace("?", np.nan)


def load_context() -> dict[str, Any]:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing {RESULTS_FILE}. Run training scripts first.")

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    x_train = np.load(MODEL_DIR / "X_train.npy")
    y_train = np.load(MODEL_DIR / "y_train.npy")
    x_test = np.load(MODEL_DIR / "X_test.npy")
    y_test = np.load(MODEL_DIR / "y_test.npy")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")

    train_df = _load_arff_dataframe(TRAIN_FILE)
    test_df = _load_arff_dataframe(TEST_FILE)

    return {
        "results": results,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "preprocessor": preprocessor,
        "train_df": train_df,
        "test_df": test_df,
    }


def _predict_scores(system_id: str, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "_dnn_" in system_id:
        model = tf.keras.models.load_model(MODEL_DIR / f"{system_id}.keras")
        y_proba = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred, y_proba

    model = joblib.load(MODEL_DIR / f"{system_id}.pkl")
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(x_test)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        y_proba = softmax(decision, axis=1)
    else:
        # Fallback for rare estimators without score outputs.
        n_classes = len(np.unique(y_pred))
        y_proba = np.zeros((len(y_pred), n_classes), dtype=float)
        y_proba[np.arange(len(y_pred)), y_pred] = 1.0
    return y_pred, y_proba


def build_predictions(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    x_test = context["x_test"]

    for row in context["results"]:
        system_id = row["system"]
        y_pred, y_proba = _predict_scores(system_id, x_test)
        predictions[system_id] = {"y_pred": y_pred, "y_proba": y_proba}

    return predictions


def add_advanced_metrics(
    results: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    y_test: np.ndarray,
    n_classes: int,
) -> list[dict[str, Any]]:
    y_true_bin = label_binarize(y_test, classes=np.arange(n_classes))

    updated_results: list[dict[str, Any]] = []
    for row in results:
        system_id = row["system"]
        y_pred = predictions[system_id]["y_pred"]
        y_proba = predictions[system_id]["y_proba"]

        row["precision_macro"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
        row["recall_macro"] = recall_score(y_test, y_pred, average="macro", zero_division=0)
        row["f1_macro"] = f1_score(y_test, y_pred, average="macro", zero_division=0)
        row["f1_weighted"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        row["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)

        if y_proba.shape[1] == n_classes:
            row["roc_auc_ovr_weighted"] = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="weighted")
            row["pr_auc_weighted"] = average_precision_score(y_true_bin, y_proba, average="weighted")
        else:
            row["roc_auc_ovr_weighted"] = None
            row["pr_auc_weighted"] = None

        updated_results.append(row)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(updated_results, file, ensure_ascii=False, indent=2)

    return updated_results


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved figure: %s", path)


def _clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def _model_family(row: dict[str, Any]) -> str:
    classifier = row["classifier"].lower()
    system = row["system"].lower()
    if "dnn" in system:
        return "Neural network"
    if "random forest" in classifier or "gradient boosting" in classifier or "adaboost" in classifier:
        return "Tree ensemble"
    if "decision tree" in classifier:
        return "Single tree"
    if "svm" in classifier or "support vector" in classifier or "logistic" in classifier:
        return "Linear/kernel"
    return "Probabilistic"


def _compact_model_label(system_id: str) -> str:
    replacements = {
        "A_naive_bayes": "A NB",
        "B_svm": "B SVM",
        "C_random_forest": "C RF",
        "D_gradient_boosting": "D GB",
        "E_decision_tree": "E DT",
        "F_logistic_regression": "F LR",
        "G_adaboost": "G Ada",
        "H_dnn_shallow": "H DNN-S",
        "I_dnn_medium": "I DNN-M",
        "J_dnn_deep": "J DNN-D",
        "K_dnn_wide": "K DNN-W",
    }
    return replacements.get(system_id, system_id)


def _compact_class_label(class_name: str) -> str:
    replacements = {
        "compensated_hypothyroid": "Comp.",
        "negative": "Negative",
        "primary_hypothyroid": "Primary",
        "secondary_hypothyroid": "Secondary",
    }
    return replacements.get(class_name, class_name)


def _estimate_complexity(system_id: str, x_train: np.ndarray) -> dict[str, Any]:
    if "_dnn_" in system_id:
        model = tf.keras.models.load_model(MODEL_DIR / f"{system_id}.keras")
        return {"complexity": int(model.count_params()), "complexity_label": "trainable params"}

    model = joblib.load(MODEL_DIR / f"{system_id}.pkl")
    if hasattr(model, "estimators_"):
        estimators = np.asarray(model.estimators_).ravel()
        node_count = int(sum(estimator.tree_.node_count for estimator in estimators if hasattr(estimator, "tree_")))
        return {"complexity": max(node_count, len(estimators)), "complexity_label": "ensemble tree nodes"}
    if hasattr(model, "tree_"):
        return {"complexity": int(model.tree_.node_count), "complexity_label": "tree nodes"}
    if hasattr(model, "support_vectors_"):
        return {"complexity": int(model.support_vectors_.shape[0] * x_train.shape[1]), "complexity_label": "support vectors x features"}
    if hasattr(model, "coef_"):
        return {"complexity": int(np.asarray(model.coef_).size), "complexity_label": "coefficients"}
    if hasattr(model, "theta_"):
        return {"complexity": int(np.asarray(model.theta_).size), "complexity_label": "distribution parameters"}
    return {"complexity": int(x_train.shape[1]), "complexity_label": "feature count proxy"}


def plot_dataset_imbalance(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    train_counts = train_df["Class"].value_counts().sort_values(ascending=True)
    test_counts = test_df["Class"].value_counts().reindex(train_counts.index).fillna(0).astype(int)
    all_classes = train_counts.index.tolist()

    negative_count = int(train_counts.get("negative", 0))
    positive_count = int(train_counts.sum() - negative_count)
    multiclass_ratio = float(train_counts.max() / max(1, train_counts.min()))
    screening_ratio = float(negative_count / max(1, positive_count))

    y = np.arange(len(all_classes))
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.barh(y + 0.18, train_counts.values, height=0.34, color="#264653", label="Train")
    ax.barh(y - 0.18, test_counts.values, height=0.34, color="#2a9d8f", label="Test")

    for idx, (train_value, test_value) in enumerate(zip(train_counts.values, test_counts.values)):
        ax.text(train_value * 1.05 + 0.5, idx + 0.18, f"{int(train_value):,}", va="center", fontsize=10, color="#263238")
        ax.text(max(test_value * 1.08, 1.0), idx - 0.18, f"{int(test_value):,}", va="center", fontsize=10, color="#263238")

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(all_classes)
    ax.set_xlabel("Sample count (log scale)")
    ax.set_ylabel("Class")
    ax.set_title("Dataset Class Imbalance: Rare Disease Subtypes Drive Evaluation Risk", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.text(
        0.02,
        0.95,
        f"Multiclass max:min = {multiclass_ratio:.0f}:1\nScreening negative:positive = {screening_ratio:.2f}:1",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#ffffff", "edgecolor": "#cfd8dc"},
    )
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    _save_plot(FIGURE_DIR / "dataset_imbalance.png")

    return {
        "train_class_distribution": {str(key): int(value) for key, value in train_counts.sort_values(ascending=False).items()},
        "test_class_distribution": {str(key): int(value) for key, value in test_counts.reindex(train_counts.sort_values(ascending=False).index).items()},
        "multiclass_imbalance_ratio": multiclass_ratio,
        "screening_imbalance_ratio": screening_ratio,
        "positive_train_samples": positive_count,
        "negative_train_samples": negative_count,
    }


def plot_model_ranking(results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda row: row["f1_weighted"], reverse=True)
    models = [row["system"] for row in ordered]
    values = [row["f1_weighted"] * 100 for row in ordered]

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(models))
    ax.plot(x, values, color="#1b4d89", linewidth=3, marker="o", markersize=7)
    ax.fill_between(x, values, min(values) - 0.5, alpha=0.15, color="#6ab0de")

    best_index = int(np.argmax(values))
    ax.scatter([best_index], [values[best_index]], s=240, color="#e63946", edgecolors="white", zorder=5)
    ax.annotate(
        f"Best: {models[best_index]}\nF1w={values[best_index]:.2f}%",
        xy=(best_index, values[best_index]),
        xytext=(best_index + 0.3, values[best_index] + 0.8),
        arrowprops={"arrowstyle": "->", "color": "#e63946", "lw": 2},
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title("Model Performance Ranking (Weighted F1)", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Weighted F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    _save_plot(FIGURE_DIR / "model_ranking_curve.png")


def plot_metric_fingerprint(results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda row: row["f1_weighted"], reverse=True)
    model_labels = [_compact_model_label(row["system"]) for row in ordered]
    metrics = ["accuracy", "f1_weighted", "f1_macro", "recall_macro", "balanced_accuracy", "pr_auc_weighted"]
    metric_labels = ["Accuracy", "Weighted F1", "Macro F1", "Macro Recall", "Balanced Acc", "PR-AUC"]
    frame = pd.DataFrame(
        [[row.get(metric, np.nan) for metric in metrics] for row in ordered],
        index=model_labels,
        columns=metric_labels,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 10),
        gridspec_kw={"height_ratios": [1.15, 0.85], "width_ratios": [1, 1], "hspace": 0.42, "wspace": 0.18},
    )
    heat_ax = axes[0, 0]
    note_ax = axes[0, 1]
    gap_ax = axes[1, 0]
    rank_ax = axes[1, 1]

    sns.heatmap(
        frame,
        annot=True,
        fmt=".3f",
        cmap="mako",
        vmin=0.55,
        vmax=1.0,
        linewidths=0.8,
        linecolor="#edf2f4",
        cbar=False,
        annot_kws={"fontsize": 9},
        ax=heat_ax,
    )
    heat_ax.set_title("Metric Fingerprint Across Models", fontweight="bold", pad=12)
    heat_ax.set_xlabel("")
    heat_ax.set_ylabel("")
    heat_ax.tick_params(axis="x", labelrotation=28, labelsize=9)
    heat_ax.tick_params(axis="y", labelsize=9)

    divergence = pd.DataFrame(
        {
            "system": model_labels,
            "Weighted F1 - Macro F1": [(row.get("f1_weighted", 0.0) - row.get("f1_macro", 0.0)) * 100 for row in ordered],
            "Accuracy - Balanced Acc": [(row.get("accuracy", 0.0) - row.get("balanced_accuracy", 0.0)) * 100 for row in ordered],
        }
    )
    y = np.arange(len(divergence))
    gap_ax.barh(y + 0.18, divergence["Weighted F1 - Macro F1"], height=0.34, color="#ef476f", label="Weighted F1 - Macro F1")
    gap_ax.barh(y - 0.18, divergence["Accuracy - Balanced Acc"], height=0.34, color="#118ab2", label="Accuracy - Balanced Acc")
    gap_ax.axvline(0, color="#455a64", linewidth=1)
    gap_ax.set_yticks(y)
    gap_ax.set_yticklabels(divergence["system"], fontsize=8.5)
    gap_ax.invert_yaxis()
    gap_ax.set_xlabel("Gap in percentage points")
    gap_ax.set_title("Metric Divergence Caused by Imbalance", fontweight="bold", pad=12)
    gap_ax.legend(fontsize=8.5, loc="lower right", frameon=True)
    gap_ax.grid(axis="x", alpha=0.25)
    gap_ax.grid(axis="y", visible=False)

    ranked = pd.DataFrame(
        {
            "system": model_labels,
            "Weighted F1": [row.get("f1_weighted", 0.0) * 100 for row in ordered],
            "Balanced Accuracy": [row.get("balanced_accuracy", 0.0) * 100 for row in ordered],
        }
    ).head(8)
    rank_y = np.arange(len(ranked))
    rank_ax.plot(ranked["Weighted F1"], rank_y, marker="o", color="#ef476f", linewidth=2.2, label="Weighted F1")
    rank_ax.plot(ranked["Balanced Accuracy"], rank_y, marker="o", color="#118ab2", linewidth=2.2, label="Balanced Accuracy")
    for idx, item in ranked.iterrows():
        rank_ax.plot([item["Balanced Accuracy"], item["Weighted F1"]], [idx, idx], color="#b0bec5", linewidth=1.5, zorder=0)
    rank_ax.set_yticks(rank_y)
    rank_ax.set_yticklabels(ranked["system"], fontsize=8.5)
    rank_ax.invert_yaxis()
    rank_ax.set_xlabel("Score (%)")
    rank_ax.set_title("Same Model, Different Clinical Story", fontweight="bold", pad=12)
    rank_ax.legend(fontsize=8.5, loc="lower left", frameon=True)
    rank_ax.grid(axis="x", alpha=0.25)
    rank_ax.grid(axis="y", visible=False)

    note_ax.axis("off")
    note_ax.text(
        0.03,
        0.88,
        "Interpretation",
        fontsize=15,
        fontweight="bold",
        color="#1b4d59",
        transform=note_ax.transAxes,
    )
    note_ax.text(
        0.03,
        0.68,
        "Weighted metrics answer: how well does the model perform on the observed population mix?",
        fontsize=11,
        wrap=True,
        transform=note_ax.transAxes,
    )
    note_ax.text(
        0.03,
        0.48,
        "Macro and balanced metrics answer: how much performance remains when rare disease classes receive equal weight?",
        fontsize=11,
        wrap=True,
        transform=note_ax.transAxes,
    )
    note_ax.text(
        0.03,
        0.22,
        "Large positive gaps indicate majority-class success may be masking minority-class weakness.",
        fontsize=11,
        fontweight="bold",
        color="#d62828",
        wrap=True,
        transform=note_ax.transAxes,
    )
    _save_plot(FIGURE_DIR / "metric_fingerprint.png")


def plot_complexity_performance(results: list[dict[str, Any]], x_train: np.ndarray) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in results:
        complexity_info = _estimate_complexity(row["system"], x_train)
        records.append(
            {
                "system": row["system"],
                "family": _model_family(row),
                "complexity": max(1, int(complexity_info["complexity"])),
                "complexity_label": complexity_info["complexity_label"],
                "balanced_accuracy": row.get("balanced_accuracy", 0.0) * 100,
                "f1_weighted": row.get("f1_weighted", 0.0) * 100,
                "f1_macro": row.get("f1_macro", 0.0) * 100,
            }
        )

    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    for family, group in frame.groupby("family"):
        ax.scatter(
            group["complexity"],
            group["balanced_accuracy"],
            s=np.clip((group["f1_weighted"] - 80) * 18, 90, 420),
            color=MODEL_FAMILY_COLORS.get(family, "#607d8b"),
            alpha=0.82,
            edgecolor="white",
            linewidth=1.2,
            label=family,
        )
        for _, item in group.iterrows():
            ax.annotate(
                _compact_model_label(item["system"]),
                (item["complexity"], item["balanced_accuracy"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9.5,
            )

    best = frame.sort_values("balanced_accuracy", ascending=False).iloc[0]
    ax.annotate(
        f"Best recall balance\n{_compact_model_label(str(best['system']))}: {best['balanced_accuracy']:.1f}%",
        xy=(best["complexity"], best["balanced_accuracy"]),
        xytext=(0.55, 0.92),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#d62828", "lw": 1.8},
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": "#d62828"},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Approximate model complexity (log scale)")
    ax.set_ylabel("Balanced Accuracy on Test Set (%)")
    ax.set_title("Model Complexity vs Minority-Class-Aware Performance", fontweight="bold", pad=12)
    ax.set_xlim(max(1, frame["complexity"].min() * 0.55), frame["complexity"].max() * 1.75)
    ax.legend(loc="lower right", frameon=True, fontsize=10, markerscale=0.8)
    ax.grid(alpha=0.25)
    _save_plot(FIGURE_DIR / "complexity_performance.png")

    return {"records": records}


def plot_roc_pr_curves(
    results: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    y_test: np.ndarray,
    n_classes: int,
) -> None:
    y_true_bin = label_binarize(y_test, classes=np.arange(n_classes))
    top_models = sorted(results, key=lambda row: row["f1_weighted"], reverse=True)[:6]

    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

    for row in top_models:
        model_id = row["system"]
        y_proba = predictions[model_id]["y_proba"]
        if y_proba.shape[1] != n_classes:
            continue

        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_proba.ravel())
        pr_auc = average_precision_score(y_true_bin, y_proba, average="weighted")

        ax_pr.plot(recall, precision, linewidth=2.3, label=f"{model_id} (AP={pr_auc:.4f})")

    baseline = np.mean(y_true_bin.ravel())
    ax_pr.axhline(baseline, linestyle="--", color="#607d8b", alpha=0.7, label=f"Micro prevalence baseline={baseline:.3f}")
    ax_pr.set_title("Precision-Recall Behavior Under Class Imbalance", fontweight="bold", pad=12)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=9, loc="lower left")
    ax_pr.grid(alpha=0.25)
    fig_pr.tight_layout()
    fig_pr.savefig(FIGURE_DIR / "pr_multi_model.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig_pr)


def plot_class_recall_heatmap(
    results: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    y_test: np.ndarray,
    class_names: list[str],
) -> None:
    recall_rows: list[list[float]] = []
    model_names: list[str] = []

    for row in sorted(results, key=lambda item: item["f1_weighted"], reverse=True):
        system_id = row["system"]
        report = classification_report(y_test, predictions[system_id]["y_pred"], output_dict=True, zero_division=0)
        recalls = [report.get(str(i), {}).get("recall", 0.0) for i in range(len(class_names))]
        recall_rows.append(recalls)
        model_names.append(system_id)

    recall_df = pd.DataFrame(recall_rows, index=model_names, columns=class_names)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(recall_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Recall"}, ax=ax)
    ax.set_title("Class-wise Recall by Model", fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Model")
    _save_plot(FIGURE_DIR / "class_recall_heatmap.png")


def plot_feature_importance(results: list[dict[str, Any]], preprocessor: Any) -> dict[str, Any]:
    feature_names = preprocessor.get_feature_names_out().tolist()
    tree_candidates = [
        row for row in results
        if row["classifier"].lower().startswith("random forest")
        or row["classifier"].lower().startswith("gradient boosting")
        or row["classifier"].lower().startswith("decision tree")
    ]
    if not tree_candidates:
        return {"feature_importance_model": None, "top_features": []}

    best_tree = sorted(tree_candidates, key=lambda row: row["f1_weighted"], reverse=True)[0]
    model = joblib.load(MODEL_DIR / f"{best_tree['system']}.pkl")
    if not hasattr(model, "feature_importances_"):
        return {"feature_importance_model": None, "top_features": []}

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:TOP_N_FEATURES]
    top_names = [_clean_feature_name(feature_names[i]) for i in idx]
    top_scores = [float(importances[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(11, 7.2))
    sns.barplot(
        x=top_scores[::-1],
        y=top_names[::-1],
        hue=top_names[::-1],
        palette=sns.color_palette("crest", n_colors=len(top_names)),
        legend=False,
        ax=ax,
    )
    cumulative_top3 = sum(top_scores[:3]) * 100
    ax.set_title(f"Dominant Clinical Signal Features - {best_tree['system']}", fontweight="bold", pad=12)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.text(
        0.98,
        0.08,
        f"Top 3 features explain {cumulative_top3:.1f}%\nof tree split importance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": "#cfd8dc"},
    )
    _save_plot(FIGURE_DIR / "feature_importance.png")

    return {
        "feature_importance_model": best_tree["system"],
        "top_features": [{"feature": name, "importance": score} for name, score in zip(top_names, top_scores)],
    }


def plot_learning_curve(best_model_id: str, x_train: np.ndarray, y_train: np.ndarray) -> None:
    if "_dnn_" in best_model_id:
        return

    model = joblib.load(MODEL_DIR / f"{best_model_id}.pkl")
    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        x_train,
        y_train,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_weighted",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
    )

    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    final_gap = float((train_mean[-1] - valid_mean[-1]) * 100)

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.plot(train_sizes, train_mean, marker="o", linewidth=2.6, color="#073b4c", label="Train")
    ax.plot(train_sizes, valid_mean, marker="o", linewidth=2.6, color="#06d6a0", label="Cross-validation")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.12, color="#073b4c")
    ax.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.18, color="#06d6a0")
    ax.fill_between(train_sizes, valid_mean, train_mean, alpha=0.08, color="#ef476f")
    ax.set_title(f"Learning Curve and Generalization Gap ({best_model_id})", fontweight="bold", pad=12)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Weighted F1")
    ax.legend(loc="lower right")
    ax.text(
        0.03,
        0.08,
        f"Final generalization gap: {final_gap:.2f} percentage points\nNarrow gap implies stable fit rather than memorization.",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": "#cfd8dc"},
    )
    ax.grid(alpha=0.25)
    _save_plot(FIGURE_DIR / "learning_curve_best_model.png")


def plot_error_distribution(best_model_id: str, y_test: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    false_positive_counts: list[int] = []
    false_negative_counts: list[int] = []
    supports: list[int] = []
    recall_values: list[float] = []

    for class_idx in range(len(class_names)):
        positives = y_test == class_idx
        predicted = y_pred == class_idx
        fp = int(np.sum(~positives & predicted))
        fn = int(np.sum(positives & ~predicted))
        support = int(np.sum(positives))
        recall = float(np.sum(positives & predicted) / max(1, support))
        false_positive_counts.append(fp)
        false_negative_counts.append(fn)
        supports.append(support)
        recall_values.append(recall)

    x = np.arange(len(class_names))
    compact_classes = [_compact_class_label(name) for name in class_names]
    width = 0.36
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.8, 6.8),
        gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.42},
    )
    axes[0].bar(x - width / 2, false_positive_counts, width, label="False Positive", color="#118ab2")
    axes[0].bar(x + width / 2, false_negative_counts, width, label="False Negative", color="#ef476f")
    axes[0].set_title(f"Clinical Error Profile ({_compact_model_label(best_model_id)})", fontweight="bold", pad=12)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Test-set error count")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(compact_classes, rotation=0, ha="center")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.64, 1.0), ncols=2, fontsize=10, frameon=True)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].grid(axis="x", visible=False)

    matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    sns.heatmap(
        normalized,
        annot=matrix,
        fmt="d",
        cmap="rocket_r",
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="#ffffff",
        cbar_kws={"label": "Row-normalized share"},
        xticklabels=compact_classes,
        yticklabels=compact_classes,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Flow: Counts Annotated, Rows Normalized", fontweight="bold", pad=12)
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].tick_params(axis="y", rotation=0)

    weakest_idx = int(np.argmin(recall_values))
    axes[0].text(
        0.02,
        0.95,
        f"Lowest recall: {_compact_class_label(class_names[weakest_idx])}\n"
        f"{recall_values[weakest_idx] * 100:.1f}% recall, support={supports[weakest_idx]}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ffffff", "edgecolor": "#ef476f"},
    )
    _save_plot(FIGURE_DIR / "error_distribution.png")

def summarize_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    feature_columns = [col for col in train_df.columns if col != "Class"]
    train_class_counts = train_df["Class"].value_counts().to_dict()
    test_class_counts = test_df["Class"].value_counts().to_dict()
    missing_ratio = train_df[feature_columns].isna().mean().sort_values(ascending=False)
    negative_count = int(train_class_counts.get("negative", 0))
    positive_count = int(sum(train_class_counts.values()) - negative_count)

    return {
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "train_class_distribution": train_class_counts,
        "test_class_distribution": test_class_counts,
        "class_imbalance_ratio": float(max(train_class_counts.values()) / max(1, min(train_class_counts.values()))),
        "multiclass_imbalance_ratio": float(max(train_class_counts.values()) / max(1, min(train_class_counts.values()))),
        "screening_imbalance_ratio": float(negative_count / max(1, positive_count)),
        "positive_train_samples": positive_count,
        "negative_train_samples": negative_count,
        "top_missing_features": [
            {"feature": feature, "missing_ratio": float(ratio)}
            for feature, ratio in missing_ratio.head(10).items()
        ],
    }


def run() -> None:
    context = load_context()
    predictions = build_predictions(context)

    y_test = context["y_test"]
    n_classes = len(context["label_encoder"].classes_)
    class_names = [str(name) for name in context["label_encoder"].classes_]

    results = add_advanced_metrics(context["results"], predictions, y_test, n_classes)
    best_model = sorted(results, key=lambda row: row["f1_weighted"], reverse=True)[0]
    best_model_id = best_model["system"]

    imbalance_summary = plot_dataset_imbalance(context["train_df"], context["test_df"])
    plot_metric_fingerprint(results)
    complexity_summary = plot_complexity_performance(results, context["x_train"])
    plot_roc_pr_curves(results, predictions, y_test, n_classes)
    plot_class_recall_heatmap(results, predictions, y_test, class_names)
    feature_summary = plot_feature_importance(results, context["preprocessor"])
    plot_learning_curve(best_model_id, context["x_train"], context["y_train"])

    y_pred_best = predictions[best_model_id]["y_pred"]
    plot_error_distribution(best_model_id, y_test, y_pred_best, class_names)

    dataset_summary = summarize_dataset(context["train_df"], context["test_df"])
    dataset_summary.update(imbalance_summary)
    summary = {
        "best_model": best_model,
        "dataset": dataset_summary,
        "feature_importance": feature_summary,
        "model_complexity": complexity_summary,
    }

    with ANALYSIS_SUMMARY_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    LOGGER.info("Advanced evaluation complete. Results updated at %s", RESULTS_FILE)
    LOGGER.info("Analysis summary saved at %s", ANALYSIS_SUMMARY_FILE)


if __name__ == "__main__":
    run()
