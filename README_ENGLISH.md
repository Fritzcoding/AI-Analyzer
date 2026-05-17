# Hypothyroid ML: Production-Grade Multi-Classifier Diagnostic System

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ML Classifiers](https://img.shields.io/badge/Classifiers-13%20Systems-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> A **production-grade machine learning pipeline** demonstrating end-to-end ML development: from raw medical data to interactive dashboards, with **99.41% accuracy** on hypothyroid disease detection.

---

## 🎯 Project Overview

### What This Project Is

This is a complete, reproducible machine learning system that detects hypothyroidism (thyroid disease) from patient medical records. The project implements and compares **13 different classifiers** across multiple complexity levels to find the optimal solution for a medical diagnosis task with severe class imbalance.

**Core Achievement**: 
- ✅ **99.41% test accuracy** using Gradient Boosting
- ✅ **95.97% recall rate** (critical for preventing missed diagnoses in medical applications)
- ✅ **Full end-to-end automated pipeline** from raw data to professional reports
- ✅ **Production-ready infrastructure** with experiment tracking, dashboards, and explainability

### Why This Matters

Hypothyroid disease detection is **fundamentally different from traditional ML classification** because:
1. **Missing one diagnosis can harm patients** → Recall (sensitivity) is more important than accuracy
2. **Real medical data is messy** → 92% class imbalance, missing values, mixed data types
3. **Clinicians need explanations** → Which features matter? Why did it make this decision?
4. **Different models have different tradeoffs** → A simpler model that doctors understand may be preferable to a perfect black-box

This project teaches all of these lessons through systematic experimentation.

---

## 💡 Motivation & Problem Statement

### The Medical Context

Hypothyroidism (甲狀腺功能低下症) is a common thyroid disorder where the thyroid gland doesn't produce enough thyroid hormone. Early diagnosis is critical because:
- Untreated hypothyroidism leads to serious complications
- Diagnosis requires analyzing multiple clinical indicators (TSH, T3, T4, etc.)
- Misdiagnosis (false negatives) are more dangerous than false alarms

### The Data Challenge

| Issue | Impact | Solution |
|-------|--------|----------|
| **Severe class imbalance** (92% healthy, 8% diseased) | Model defaults to guessing "healthy" | Class weighting, balanced sampling, custom metrics |
| **Missing values** in medical records | Can't just drop records | Imputation (median for numeric, mode for categorical) |
| **Mixed data types** (continuous + categorical) | Different preprocessing needed | Separate pipelines with ColumnTransformer |
| **Medical requirements** (prioritize recall) | Accuracy alone is misleading | Use F1, recall, and custom thresholds |

### The Solution: Systematic Comparison

Rather than assuming one algorithm is "best," we:
1. **Train 7 traditional ML classifiers** (Naive Bayes, SVM, Trees, Boosting, etc.)
2. **Build 4 deep learning architectures** (shallow vs deep vs wide networks)
3. **Compare comprehensively** using multiple evaluation metrics
4. **Document tradeoffs explicitly** to guide clinical deployment decisions

---

## 🏗️ How It Is Implemented: The 10-Stage Pipeline

The project is organized as a **10-stage production pipeline**, where each stage is a standalone Python script that can be run independently or as part of the full workflow.

```
┌─────────────────────────────────────────────────────────────────┐
│                  HYPOTHYROID ML 10-STAGE PIPELINE               │
└─────────────────────────────────────────────────────────────────┘

FOUNDATION LAYER (Stages 1-5: Core ML Pipeline)
├─ STAGE 1: Data Preprocessing
├─ STAGE 2: Traditional ML Training (sklearn)
├─ STAGE 3: Deep Learning Training (TensorFlow)
├─ STAGE 4: Model Evaluation & Comparison
└─ STAGE 5: Automated Report Generation

PORTFOLIO LAYER (Stages 6-10: Advanced Features)
├─ STAGE 6: Model Explainability (SHAP)
├─ STAGE 7: Hyperparameter Optimization (Optuna)
├─ STAGE 8: Statistical Testing & Significance
├─ STAGE 9: Experiment Tracking (MLflow)
└─ STAGE 10: Dashboard Data Export
```

### Stage 1: Data Preprocessing (`src/01_preprocessing.py`)

**Purpose**: Transform raw ARFF medical records into clean, normalized datasets ready for ML.

**Process**:
```
ARFF File Loading
  ↓
Convert to pandas DataFrame
  ↓
Decode byte strings (b'?') from ARFF format
  ↓
Identify & separate numeric vs categorical columns
  ↓
Handle missing values:
  • Numeric: median imputation
  • Categorical: mode imputation
  ↓
Build sklearn ColumnTransformer Pipeline:
  • Numeric: StandardScaler (mean=0, std=1)
  • Categorical: OrdinalEncoder (discrete values)
  ↓
Save pipeline + training/test arrays
```

**Outputs**:
- `outputs/models/preprocessor.pkl` — Fitted preprocessing pipeline
- `outputs/models/X_train.npy, y_train.npy` — Training arrays (3,057 samples)
- `outputs/models/X_test.npy, y_test.npy` — Test arrays (972 samples)
- `outputs/models/label_encoder.pkl` — Target variable encoder

**Key Insight**: Proper preprocessing is 50% of the ML battle. Missing values improperly handled can destroy model performance.

---

### Stage 2: Traditional ML Classifiers (`src/02_sklearn_classifiers.py`)

**Purpose**: Train and optimize 7 scikit-learn classifiers using GridSearchCV.

**Classifiers Trained**:

| ID | Model | Hyperparameter Grid | Best Accuracy | Use Case |
|----|----|-----------------|----------|----------|
| **A** | **Naive Bayes** | var_smoothing ∈ [1e-9, 1e-7, 1e-5] | ~95% | Fast baseline, probabilistic |
| **B** | **SVM (RBF)** | C ∈ [0.1, 1, 10], gamma ∈ [auto, scale] | ~98% | Non-linear boundaries |
| **C** | **Decision Tree** | max_depth ∈ [3, 5, 7, 10], min_samples_split ∈ [2, 5, 10] | ~99.1% | Interpretable decision rules |
| **D** | **Random Forest** | n_estimators ∈ [50, 100, 200], max_depth ∈ [5, 10, 15, None] | ~98.8% | Ensemble robustness |
| **E** | **K-Nearest Neighbors** | n_neighbors ∈ [3, 5, 7, 11], metric ∈ [euclidean, manhattan] | ~97% | Non-parametric baseline |
| **F** | **Gradient Boosting** | n_estimators ∈ [100, 200], lr ∈ [0.01, 0.1], max_depth ∈ [3, 5, 7] | **99.41%** ⭐ | **BEST PERFORMER** |
| **G** | **Logistic Regression** | C ∈ [0.001, 0.1, 1, 10], solver ∈ [lbfgs, liblinear] | ~94% | Linear baseline, explainable |

**Training Protocol**:
- **Hyperparameter Search**: GridSearchCV with 5-fold cross-validation
- **Imbalance Handling**: `class_weight="balanced"` in all models
- **Metric**: CV score = Accuracy (also track F1, precision, recall on test set)

**Outputs**:
- `outputs/models/A_naive_bayes.pkl` through `outputs/models/G_logistic_regression.pkl` — Trained models
- `outputs/figures/cm_A_*.png` through `outputs/figures/cm_G_*.png` — Confusion matrices
- `outputs/figures/roc_A_*.png` through `outputs/figures/roc_G_*.png` — ROC curves
- Console logs with best hyperparameters and CV scores

---

### Stage 3: Deep Learning (TensorFlow) (`src/03_tensorflow_dnn.py`)

**Purpose**: Train and compare 4 DenseNet architectures to explore depth vs width tradeoffs.

**Architectures Tested**:

| ID | Name | Architecture | Dropout | Learning Rate | Test Accuracy | Insight |
|----|----|------------|---------|---|----------|----------|
| **H** | **Shallow** | [64] (1 hidden) | 0.0 | 0.001 | 97.36% | DNN baseline |
| **I** | **Medium** | [128, 64] (2 hidden) | 0.2 | 0.001 | 97.07% | Moderate complexity |
| **J** | **Deep** | [256, 128, 64] (3 hidden) | 0.3 | 0.0005 | 95.89% | **Too deep!** |
| **K** | **Wide** | [512, 256] (2 hidden) | 0.2 | 0.001 | 96.48% | Wide can work better than deep |

**Training Details**:
- **Framework**: TensorFlow 2.11+ with Keras Functional API
- **Loss Function**: `sparse_categorical_crossentropy` (multi-class)
- **Optimizer**: Adam (adaptive learning rates)
- **Regularization**:
  - Dropout layers to prevent overfitting
  - BatchNormalization for stable training
  - `class_weight` to handle imbalance
- **Callbacks**:
  - EarlyStopping (patience=10): Stop if validation loss doesn't improve
  - ModelCheckpoint: Save best model automatically
- **Batch Size**: 64
- **Max Epochs**: 100 (usually stops earlier due to EarlyStopping)
- **Validation**: 10% of training data held out

**Key Findings**:
> ⚠️ **"Deeper doesn't mean better"**
> 
> On this ~3K-sample dataset, the shallow network (H) outperforms the deep network (J) by **1.47 percentage points**.
> This is a crucial lesson: deep architectures need large datasets (100K+ samples). For moderate-sized problems,
> prefer **wider and shallower** networks.

**Outputs**:
- `outputs/models/H_dnn_shallow.keras` through `outputs/models/K_dnn_wide.keras` — Trained models
- `outputs/figures/dnn_H_history.png` through `outputs/figures/dnn_K_history.png` — Training curves (loss & accuracy)

---

### Stage 4: Unified Evaluation (`src/04_evaluate.py`)

**Purpose**: Load all 11 trained models (7 sklearn + 4 TensorFlow) and evaluate on the held-out test set.

**Process**:
```
Load preprocessed X_test / y_test
  ↓
For each of 11 trained models:
  • Generate predictions on test set
  • Calculate: Accuracy, Precision (macro), Recall (macro), F1-Score (weighted)
  • Build confusion matrix
  ↓
Create visualizations:
  • Accuracy comparison bar chart (all 11 systems)
  • ROC-AUC curves (multi-class One-vs-Rest)
  • Calibration curves (predicted probability vs actual frequency)
  ↓
Consolidate results → outputs/results.json
```

**Metrics Calculated**:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) — overall correctness
- **Precision**: TP / (TP + FP) — of predicted diseased, how many are correct?
- **Recall (Sensitivity)**: TP / (TP + FN) — of actual diseased, how many did we catch? ⭐ [Most important for medical]
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean

**Outputs**:
- `outputs/results.json` — Master results file with all metrics
- `outputs/figures/accuracy_comparison.png` — Bar chart ranking all 11 systems
- `outputs/figures/confusion_matrix_heatmaps.png` — Grid of all 11 confusion matrices
- `outputs/figures/roc_comparison.png` — Overlaid ROC curves

**Example Output** (top performers):
```json
{
  "F_gradient_boosting": {
    "accuracy": 0.9941,
    "precision": 0.9086,
    "recall": 0.9597,
    "f1_score": 0.9327,
    "notes": "BEST PERFORMER - Highest accuracy + strong recall"
  },
  "C_decision_tree": {
    "accuracy": 0.9912,
    "precision": 0.9010,
    "recall": 0.9744,
    "f1_score": 0.9356,
    "notes": "Comparable to Gradient Boosting, more interpretable"
  }
}
```

---

### Stage 5: Automated Report Generation (`src/05_generate_report.py`)

**Purpose**: Generate a professional 4-5 page Traditional Chinese report suitable for academic submission.

**Report Structure**:

```
┌─────────────────────────────────────┐
│   海大資工 AI 機器學習作業報告      │  (Header)
├─────────────────────────────────────┤
│ (一) 實驗結果                        │  Section 1: Results
│  • Results table (System | Accuracy)│
│  • Confusion matrices & charts      │
├─────────────────────────────────────┤
│ (二) 系統比較                        │  Section 2: Analysis
│  • Why A outperforms B?            │
│  • Tradeoff discussions            │
│  • Deep vs Wide findings           │
├─────────────────────────────────────┤
│ (三) 結論                           │  Section 3: Conclusions
│  • Best system recommendation      │
│  • Future improvements             │
│  • Medical deployment considerations│
└─────────────────────────────────────┘
```

**Implementation Details**:
- **Format**: Microsoft Word (.docx) for easy editing
- **Generation**: python-docx library programmatically builds document
- **Content**: Pulls data from `outputs/results.json`
- **Styling**: Consistent fonts, colors, formatting
- **Figures**: All confusion matrices and charts embedded inline

**Outputs**:
- `report/報告.docx` — Final submission-ready report

---

### Stage 6: Model Explainability (`src/06_explainability.py`)

**Purpose**: Understand **why** models make predictions using SHAP (SHapley Additive exPlanations).

**Techniques Applied**:
1. **TreeExplainer** for sklearn tree-based models (Random Forest, Gradient Boosting, Decision Tree)
   - SHAP values quantify each feature's contribution to each prediction
   
2. **Permutation Importance** for non-tree models (SVM, Logistic Regression, Naive Bayes)
   - Shuffle each feature; measure drop in accuracy
   - High drop = feature is important

**Visualizations**:
- **Bar plots**: Which features matter most globally?
- **Waterfall plots**: For individual patient, which features pushed toward diagnosis?
- **Dependence plots**: As feature X increases, does prediction increase/decrease?

**Outputs**:
- `outputs/figures/shap_summary_top5_features.png` — Most important features across all patients
- `outputs/figures/shap_waterfall_example.png` — Instance-level explanation example

**Medical Value**:
> With SHAP, doctors can see: "This patient was classified as hypothyroid because TSH was high (contributes +0.45 to model output) and T3 was low (contributes +0.38)."
> This builds **trust** in the system.

---

### Stage 7: Hyperparameter Optimization (`src/07_optuna_tuning.py`)

**Purpose**: Use Bayesian optimization to find even better hyperparameters for top 3 classifiers.

**Approach**:
- **Algorithm**: Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Objective**: Maximize cross-validation accuracy
- **Trials**: 50 parameter combinations tested intelligently (not grid search)
- **Scope**: Applied to F (Gradient Boosting), C (Decision Tree), D (Random Forest)

**Example Search Space** (Gradient Boosting):
```python
{
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7, 9, 11],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
```

**Outputs**:
- `outputs/optuna_results.json` — Best parameters found
- `outputs/figures/optuna_optimization_history.png` — Trial score improvement over time

---

### Stage 8: Statistical Testing (`src/08_statistical_tests.py`)

**Purpose**: Determine if performance differences between classifiers are **statistically significant** or just random noise.

**Tests Applied**:

1. **McNemar's Test** (pairwise comparison)
   - Q: Is classifier A significantly better than B?
   - Test on test set predictions
   - If p < 0.05, difference is statistically significant

2. **Friedman Test** (all classifiers together)
   - Q: Do the 11 classifiers have significantly different performance?
   - Null hypothesis: All classifiers are equally good
   - If p < 0.05, reject null; classifiers differ significantly

3. **Post-hoc Pairwise Comparisons**
   - Which specific pairs differ significantly?

**Outputs**:
- `outputs/statistical_tests.json` — p-values and significance flags
- `outputs/figures/significance_heatmap.png` — Visual matrix of pairwise significance

**Interpretation Example**:
```
McNemar(Gradient Boosting vs Decision Tree):
  p-value = 0.312
  Interpretation: Not statistically significant (p > 0.05)
  → Both models perform comparably; choose Decision Tree if interpretability is needed.
```

---

### Stage 9: MLflow Experiment Tracking (`src/09_mlflow_tracking.py`)

**Purpose**: Create a persistent log of all experiments (models, parameters, metrics, artifacts).

**What Gets Tracked**:
- **Parameters**: All hyperparameters used for each model
- **Metrics**: Accuracy, precision, recall, F1 on test set
- **Artifacts**: Model files, confusion matrices, SHAP figures
- **Metadata**: Training time, data version, framework versions

**Usage**:
```bash
# Launch MLflow UI
mlflow ui --backend-store-uri outputs/mlruns

# Visit http://localhost:5000 to see all runs
```

**Dashboard Features**:
- Compare metrics across runs
- Plot parameter vs metric relationships
- Download artifacts
- Track experiment evolution over time

**Outputs**:
- `outputs/mlruns/` — MLflow backend storage (SQLite)

---

### Stage 10: Dashboard Data Export (`src/10_export_dashboard_data.py`)

**Purpose**: Package results into JSON format for interactive web dashboard.

**Data Exported**:
- Accuracy, precision, recall, F1 for all 11 models
- Model complexity metrics (parameters count, inference time)
- SHAP importance data
- Optuna trial history
- Statistical test results

**Outputs**:
- `dashboard/dashboard_data.json` — Master JSON serving the dashboard

---

## 🎯 Quick Start

### Installation

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. [OPTIONAL] For dashboard & explainability
pip install -r requirements-prod.txt
```

### Data Preparation

Place ARFF files in `data/`:
```
data/
├── hypothyroid_cjlin2025_training.arff   (3,057 samples)
└── hypothyroid_cjlin2025_test.arff       (972 samples)
```

### Run the Pipeline

```bash
# CORE PIPELINE (Stages 1-5: ~10-15 minutes)
python src/01_preprocessing.py          # 2 sec
python src/02_sklearn_classifiers.py    # 3-5 min
python src/03_tensorflow_dnn.py         # 2-3 min
python src/04_evaluate.py               # 10 sec
python src/05_generate_report.py        # 5 sec

# Check outputs
report/報告.docx                         # Open in Word
outputs/figures/accuracy_comparison.png  # View rankings

# PORTFOLIO PIPELINE (Stages 6-10: ~5-10 minutes)
python src/06_explainability.py         # 30 sec
python src/07_optuna_tuning.py          # 2 min (50 trials)
python src/08_statistical_tests.py      # 5 sec
python src/09_mlflow_tracking.py        # 10 sec
python src/10_export_dashboard_data.py  # 5 sec

# INTERACTIVE DASHBOARD
streamlit run app/streamlit_app.py      # Opens at http://localhost:8501
# MLflow UI
mlflow ui --backend-store-uri outputs/mlruns  # Opens at http://localhost:5000
```

### Expected Output

```
outputs/
├── models/
│   ├── A_naive_bayes.pkl ... G_logistic_regression.pkl  (7 sklearn models)
│   ├── H_dnn_shallow.keras ... K_dnn_wide.keras         (4 TensorFlow models)
│   ├── preprocessor.pkl                                  (preprocessing pipeline)
│   └── *.npy                                             (arrays)
├── figures/
│   ├── accuracy_comparison.png                           (rankings chart)
│   ├── cm_A_*.png ... cm_G_*.png                         (confusion matrices)
│   ├── dnn_*_history.png                                (training curves)
│   ├── shap_*.png                                        (explainability)
│   └── ... (many more)
├── results.json                                           (master results)
├── optuna_results.json
├── statistical_tests.json
└── mlruns/                                                (MLflow data)

report/
└── 報告.docx                                              (final submission)

dashboard/
└── dashboard_data.json                                    (web dashboard data)
```

---

## 📊 Key Results & Insights

### Performance Leaderboard

| Rank | System | Accuracy | F1-Score | Recall | Precision | Time (ms) |
|------|--------|----------|----------|--------|-----------|-----------|
| 🥇 | **F: Gradient Boosting** | **99.41%** | **93.27%** | 95.97% | 90.86% | 12 |
| 🥈 | C: Decision Tree | 99.12% | 93.56% | 97.44% | 90.10% | 2 |
| 🥉 | D: Random Forest | 98.83% | 98.70% | 98.72% | 98.68% | 8 |
| 4 | H: DenseNN (Shallow) | 97.36% | 67.68% | 58.97% | 80.99% | 15 |
| 5 | I: DenseNN (Medium) | 97.07% | 67.12% | 55.89% | 83.45% | 18 |

### Critical Insights

1. **Gradient Boosting is the best overall**
   - Highest accuracy (99.41%)
   - Strong recall (95.97%) → catches 96% of actual diseased patients
   - Only misses ~3% of cases

2. **Decision Tree is surprisingly competitive**
   - 99.12% accuracy (only 0.29% behind Gradient Boosting)
   - **MORE INTERPRETABLE** → Doctors can follow decision logic
   - Trade-off: Slightly lower precision (90.1% vs 90.9%)

3. **Deep networks underperform on this dataset**
   - Shallow DNN (H) scores 97.36%
   - Deep DNN (J) scores 95.89%
   - This is NOT a dataset size problem; it's an architecture mismatch
   - **Lesson**: Don't default to deep = better

4. **Ensemble methods (RF, GB) capture feature interactions well**
   - Top 3 performers are all ensemble-based
   - Suggests non-linear interactions between medical features

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Data** | pandas, numpy, scipy | 1.5+, 1.23+, 1.10+ | DataFrame ops, ARFF loading |
| **ML** | scikit-learn | 1.3+ | 7 classifiers, GridSearchCV |
| **DL** | TensorFlow/Keras | 2.11+ | DenseNN, EarlyStopping, class_weight |
| **Hyperopt** | Optuna | 3.5+ | Bayesian optimization (TPE) |
| **Explainability** | SHAP | 0.44+ | Feature importance (TreeExplainer) |
| **Statistics** | statsmodels, scipy | 0.14+ | McNemar test, Friedman test |
| **Tracking** | MLflow | 2.10+ | Experiment logging, model registry |
| **Reporting** | python-docx | 0.8.10+ | Dynamic Chinese report generation |
| **Visualization** | matplotlib, seaborn, plotly | 3.7+, 0.12+, 5.17+ | Charts, confusion matrices, interactive plots |
| **Dashboard** | Streamlit | 1.28+ | Interactive web UI |
| **Python** | - | 3.10+ | Type hints, pathlib, async |

---

## 📚 Project Structure

```
hypothyroid-ml/
├── src/
│   ├── config.py                    ← Central constants & paths
│   ├── 01_preprocessing.py          ← [Stage 1] Data loading & preprocessing
│   ├── 02_sklearn_classifiers.py    ← [Stage 2] Traditional ML training
│   ├── 03_tensorflow_dnn.py         ← [Stage 3] Deep learning training
│   ├── 04_evaluate.py               ← [Stage 4] Model evaluation
│   ├── 05_generate_report.py        ← [Stage 5] Report generation
│   ├── 06_explainability.py         ← [Stage 6] SHAP explanations
│   ├── 07_optuna_tuning.py          ← [Stage 7] Hyperparameter optimization
│   ├── 08_statistical_tests.py      ← [Stage 8] Significance testing
│   ├── 09_mlflow_tracking.py        ← [Stage 9] Experiment tracking
│   ├── 10_export_dashboard_data.py  ← [Stage 10] Dashboard export
│   └── utils/
│       ├── metrics.py               ← MetricsCalculator class
│       ├── visualization.py         ← Chart helpers
│       └── model_io.py              ← Model loading/saving
├── app/
│   ├── streamlit_app.py             ← Interactive dashboard
│   └── pages/                       ← Dashboard sub-pages
├── data/
│   ├── hypothyroid_cjlin2025_training.arff
│   └── hypothyroid_cjlin2025_test.arff
├── outputs/
│   ├── models/                      ← Saved models & arrays
│   ├── figures/                     ← Charts & visualizations
│   ├── mlruns/                      ← MLflow data
│   ├── results.json                 ← Master results
│   ├── optuna_results.json
│   └── statistical_tests.json
├── report/
│   └── 報告.docx                    ← Final submission
├── tests/                           ← Unit tests
├── requirements.txt                 ← Core dependencies
├── requirements-prod.txt            ← Dashboard dependencies
├── config.py                        ← Global config
├── README_ENGLISH.md                ← This file (English)
├── README.md                        ← Chinese version
├── DESCRIPTION.md                   ← Project description
├── RESULTS.md                       ← Detailed results analysis
├── ARCHITECTURE.md                  ← System design
├── AGENTS.md                        ← Pipeline stage descriptions
└── .github/copilot-instructions.md  ← Copilot rules
```

---

## 🎓 Learning Objectives

By studying this project, you'll learn:

1. **Full ML Pipeline**: From data loading to production deployment
2. **Hyperparameter Tuning**: GridSearchCV vs Bayesian optimization tradeoffs
3. **Imbalanced Classification**: Practical techniques for real medical data
4. **Deep Learning Design**: When deep ≠ better; architecture choices matter
5. **Statistical Rigor**: McNemar tests, cross-validation, significance testing
6. **Experiment Tracking**: MLflow for reproducible research
7. **Model Explainability**: SHAP for interpretable AI
8. **Professional Reporting**: Automated document generation with embedded charts
9. **Interactive Dashboards**: Streamlit for stakeholder communication

---

## 🚀 Deployment & Production Use

### Option 1: As a One-Time Analysis
```bash
# Run pipeline once, view report
python src/01_preprocessing.py && python src/02_sklearn_classifiers.py && ...
open report/報告.docx
```

### Option 2: With Interactive Dashboard
```bash
# Start Streamlit app for stakeholders
streamlit run app/streamlit_app.py

# Share URL: http://localhost:8501
# Features:
#  - Model comparison & filtering
#  - SHAP explainability viewer
#  - Patient prediction interface
#  - Hyperparameter analysis
```

### Option 3: MLflow Server (For Teams)
```bash
# Central experiment tracking server
mlflow server --backend-store-uri outputs/mlruns --host 0.0.0.0 --port 5000

# Access at: http://your-server:5000
# All team members can log & compare runs
```

---

## 🔍 Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| **README_ENGLISH.md** | This file — comprehensive English guide | Students, portfolio viewers |
| [README.md](README.md) | Traditional Chinese version | 海大資工 assignment |
| [DESCRIPTION.md](DESCRIPTION.md) | Concise technical summary | Quick reference |
| [RESULTS.md](RESULTS.md) | Detailed performance analysis & charts | Data scientists |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design & code organization | Developers |
| [AGENTS.md](AGENTS.md) | Pipeline stage breakdown | GitHub Copilot workflow |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Copilot coding rules | Development consistency |

---

## ⚡ Performance Benchmarks

| Stage | Time | Output Size | Notes |
|-------|------|-------------|-------|
| 01: Preprocessing | ~2 sec | 200 KB | Fast; just I/O |
| 02: sklearn Training | ~3-5 min | 5-8 MB | GridSearchCV is thorough |
| 03: DNN Training | ~2-3 min | 3-5 MB | TensorFlow optimization |
| 04: Evaluation | ~10 sec | 500 KB | Batch predictions |
| 05: Report Gen | ~5 sec | 2-3 MB | python-docx overhead |
| **Core Total** | **~10-15 min** | **~15 MB** | Reproducible baseline |
| 06: Explainability | ~30 sec | 1 MB | SHAP computation |
| 07: Optuna Tuning | ~2 min | 500 KB | 50 trials × 5-fold CV |
| 08: Stat Tests | ~5 sec | 50 KB | Fast pairwise comparisons |
| 09: MLflow Logging | ~10 sec | 1 MB | Metadata storage |
| 10: Dashboard Export | ~5 sec | 200 KB | JSON serialization |
| **Portfolio Total** | **~5-10 min** | **~5 MB** | Advanced features |
| **Grand Total** | **~20-25 min** | **~25 MB** | Full pipeline run |

**System Requirements**:
- Python 3.10+
- 2GB+ RAM (sufficient for all runs concurrently)
- 500MB+ disk space

---

## 🤝 Contributing & Extending

### Add a New Classifier

1. Edit `src/config.py`:
```python
CLASSIFIERS = {
    ...
    "H_your_classifier": {
        "model": YourClassifier(random_state=42),
        "params": {"param1": [v1, v2], "param2": [v3, v4]},
        "description": "Your model description"
    }
}
```

2. Re-run stages 2-5:
```bash
python src/02_sklearn_classifiers.py
python src/04_evaluate.py
python src/05_generate_report.py
```

### Modify DNN Architectures

Edit `src/03_tensorflow_dnn.py`:
```python
DNN_CONFIGS = [
    ("your_id", [hidden_units_list], dropout_rate, learning_rate, batch_size),
    ...
]
```

### Tune Hyperparameters

Edit `src/config.py` grid parameters or use Optuna (Stage 7) for Bayesian search.

---

## 📋 Best Practices Checklist

✅ **Reproducibility**
- Fixed RANDOM_STATE=42 throughout
- All hyperparameters in config.py
- Complete dependency versions in requirements.txt

✅ **Code Quality**
- Type hints on all functions
- Google-style docstrings
- logging module (not print statements)

✅ **ML Rigor**
- Strict train/test split (no data leakage)
- Cross-validation for hyperparameter selection
- Test metrics reported separately from training

✅ **Medical Context**
- Class weighting for imbalance
- Recall prioritized over accuracy
- Explainability (SHAP) for trust

✅ **Production Ready**
- Automated end-to-end pipeline
- Error handling & logging
- Model persistence (joblib + .keras)
- Dashboard for stakeholder communication

---

## 🐛 Troubleshooting

### TensorFlow GPU Warning (Windows)
```
⚠️ "TensorFlow GPU support is not available on native Windows"
→ Use WSL2 or TensorFlow with DirectML plugin (acceptable for this project)
```

### Out of Memory
```python
# In src/03_tensorflow_dnn.py, reduce batch size:
BATCH_SIZE = 32  # default 64

# Or reduce layer sizes:
DNN_CONFIGS = [("H", [32], 0.0, 0.001, 64)]  # [64] → [32]
```

### Module Import Errors
```bash
pip install -r requirements.txt --upgrade
python -m pip cache purge
pip install -e .
```

---

## 📖 References & Resources

- **scikit-learn**: https://scikit-learn.org/stable
- **TensorFlow/Keras**: https://tensorflow.org/guide
- **Class Imbalance**: https://imbalanced-learn.org
- **SHAP**: https://github.com/slundberg/shap
- **MLflow**: https://mlflow.org
- **Optuna**: https://optuna.org

---

## 📄 Citation

If you use this project in your research or coursework:

```bibtex
@project{hypothyroid_ml_2025,
  title={Hypothyroid ML: Production-Grade Multi-Classifier Diagnostic System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourname/hypothyroid-ml}
}
```

---

## 📝 License

Academic use only. See [LICENSE](LICENSE) for details.

---

**Ready to get started?** → [Installation & Setup](#-quick-start) above  
**Want all the details?** → See [DESCRIPTION.md](DESCRIPTION.md) and [AGENTS.md](AGENTS.md)  
**Results?** → Check [RESULTS.md](RESULTS.md)  

---

**Last Updated**: May 2025  
**Python Version**: 3.10+  
**Status**: ✅ Production Ready
