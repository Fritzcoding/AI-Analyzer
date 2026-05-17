# Hypothyroid ML: Multi-Classifier Diagnostic System

## 📋 Project Description

A **production-grade machine learning system** for hypothyroid disease detection, demonstrating a complete ML pipeline from raw ARFF data to interactive dashboards. The system implements **13 different classifiers** (7 traditional ML + 4 deep learning + 2 ensemble variants) on a medical diagnosis task with severe class imbalance (92% negative vs 8% diseased samples).

**Key Achievement**: Achieved **99.41% test accuracy** using Gradient Boosting while maintaining strong recall rates critical for medical diagnosis (preventing false negatives).

## 🛠️ Technologies

### **Core ML Frameworks**
- **scikit-learn 1.3+** (7 classifiers: Naive Bayes, SVM, Decision Tree, Random Forest, K-NN, Logistic Regression, Gradient Boosting)
- **TensorFlow 2.11+** with **Keras Functional API** for neural network architectures:
  - Multi-layer Dense networks with BatchNormalization
  - Dropout regularization for overfitting prevention
  - EarlyStopping with monitored validation loss
  - Class weighting for handling imbalanced datasets
  - Model checkpointing and custom loss functions

### **Data Processing & Analysis**
- **pandas** (1.5+) – DataFrame manipulation, missing value handling
- **numpy** (1.23+) – Array operations, numerical computations
- **scipy** (1.10+) – ARFF file loading (scipy.io.arff), statistical functions
- **liac-arff** (2.4+) – Weka ARFF format parsing and byte-string decoding

### **Hyperparameter Optimization**
- **Optuna 3.5+** – Bayesian optimization using TPE sampler for best classifiers
- **scikit-learn GridSearchCV/RandomizedSearchCV** – Exhaustive grid search (5-fold CV)

### **Model Explainability**
- **SHAP** (0.44+) – TreeExplainer for feature importance, permutation importance analysis
- **scikit-learn Inspection** – Permutation feature importance calculations

### **Evaluation & Statistics**
- **scikit-learn Metrics** – Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrices
- **statsmodels 0.14+** – McNemar's test, Friedman chi-square for statistical significance
- **scipy.stats** – Wilcoxon signed-rank tests

### **Experiment Tracking & Monitoring**
- **MLflow 2.10+** – Run tracking, parameter logging, model registry with SQLite backend
- **Custom logging** – Structured logging with Python's logging module

### **Reporting & Visualization**
- **python-docx 0.8.10+** – Traditional Chinese report generation with:
  - Dynamic table insertion from results
  - Inline figure embedding (confusion matrices, accuracy charts)
  - Formatted section headers and paragraphs
- **matplotlib 3.7+** – Loss/accuracy curves, confusion matrix heatmaps, comparison charts
- **seaborn 0.12+** – Enhanced statistical visualizations
- **Plotly** – Interactive 3D plots and heatmaps
- **Chart.js** (CDN) – Client-side dashboard charting

### **Web Dashboard**
- **Streamlit 1.28+** – Interactive ML dashboard with:
  - Model leaderboard and filtering
  - Side-by-side metric comparison
  - SHAP explanation viewer
  - Patient prediction interface
  - Hyperparameter analysis tabs

### **Infrastructure & Utilities**
- **Python 3.10+** – Type hints, pathlib for cross-platform paths
- **Virtual Environment** – .venv for dependency isolation
- **Docker** – Optional containerization for reproducibility
- **GitHub Actions** – CI/CD ready (not configured by default)

### **Optional/Bonus**
- **imbalanced-learn** – SMOTE oversampling (not required but possible)
- **Jupyter** – Interactive notebook support (scripts provided as Python modules)

## 💡 Motivation

### **Problem Statement**
Hypothyroid disease detection requires high recall (detecting as many true cases as possible) to prevent patient harm from missed diagnoses. Traditional accuracy metrics alone are insufficient – a 99% accuracy classifier that misses diagnoses is clinically unacceptable.

### **Dataset Challenge**
- **Severe Class Imbalance**: 92% negative vs 8% diseased samples
- **Medical Complexity**: 25+ mixed continuous and categorical features (TSH, T3, T4, patient metadata)
- **Missing Values**: Real-world incomplete medical records encoded as `?`

### **Solution Approach**
1. **Systematic Experimentation**: Compare 13 different classifiers at different complexity levels
2. **Imbalance Handling**: Apply multiple strategies (class_weight, sample weighting, EarlyStopping)
3. **Reproducibility**: Fixed random seeds, full configuration tracking, explicit feature pipelines
4. **Medical Context**: Prioritize recall over pure accuracy; analyze precision-recall tradeoffs
5. **Production Readiness**: End-to-end automated pipeline with dashboard, reports, and explanations

## 🔧 Implementation Details

### **1. Data Preprocessing (src/01_preprocessing.py)**
```
ARFF Loading → DataFrame Conversion → Missing Value Imputation 
→ ColumnTransformer (numeric: StandardScaler, categorical: OrdinalEncoder)
→ Pipeline Serialization (.pkl)
```
- Handles byte strings (b'?') from ARFF format
- Numeric columns: median imputation + StandardScaler
- Categorical columns: mode imputation + OrdinalEncoder
- Preserves feature names for explainability

### **2. scikit-learn Classifiers (src/02_sklearn_classifiers.py)**
```
GridSearchCV (5-fold CV) over 7 models with hyperparameter grids
→ Best Model Selection → Test Evaluation → Confusion Matrix Generation
→ results.json + figures/cm_*.png
```
- GaussianNB: Probabilistic baseline
- SVM: RBF/Linear kernels with class_weight adjustment
- Decision Tree: Max depth tuning (3-10)
- Random Forest: n_estimators (100-200), max_depth tuning
- K-NN: n_neighbors optimization (3-11)
- Logistic Regression: Linear baseline
- Gradient Boosting: Best performer (99.41% accuracy)

### **3. TensorFlow DenseNN (src/03_tensorflow_dnn.py)**
```
Functional API Model Building (Input → Dense → BatchNorm → Dropout → Output)
→ Model Compilation (Adam optimizer, sparse_categorical_crossentropy loss)
→ Training with class_weight + EarlyStopping (patience=10)
→ Best Epoch Checkpointing (.keras format)
```
- **4 Architectures Tested**:
  - Shallow: 1 hidden layer (64 units), dropout=0.0
  - Medium: 2 layers (128→64), dropout=0.2
  - Deep: 3 layers (256→128→64), dropout=0.3
  - Wide: 2 wide layers (512→256), dropout=0.25

- **Key Findings**: Wide-shallow networks outperform deep networks on this dataset (~3K training samples)

### **4. Unified Evaluation (src/04_evaluate.py)**
```
Load All Models → Batch Predictions → Calculate Multi-class Metrics
→ Accuracy Comparison Bar Chart → Generate results.json
```
- Metrics: Accuracy, Precision (macro), Recall (macro), F1 (weighted & macro)
- Outputs: Comparison figures, consolidated JSON results

### **5. Report Generation (src/05_generate_report.py)**
```
Read results.json → Create python-docx Document → Section Headers
→ Dynamic Results Table → Inline Confusion Matrices → Fill from Template
→ Write report/報告.docx
```
- Traditional Chinese (Traditional) report
- 4-5 pages with methodology, results, conclusions
- Embedded visualizations (confusion matrices, accuracy charts)

### **6. Explainability & Advanced Analysis (src/06-10)**
- **SHAP**: TreeExplainer for sklearn models + permutation importance plots
- **Statistical Testing**: McNemar's test for pairwise classifier significance
- **Optuna Tuning**: Bayesian optimization for top 3 classifiers
- **MLflow Tracking**: All runs logged with parameters, metrics, model artifacts
- **Dashboard Export**: JSON data export for Streamlit interface

### **7. Interactive Dashboard (app/streamlit_app.py)**
```
Streamlit Multi-Page App:
  ├─ Leaderboard: Model rankings + filtering
  ├─ Comparison: Side-by-side metrics
  ├─ Explainability: SHAP feature importance viewer
  ├─ Predict: Patient prediction interface
  └─ Analysis: Hyperparameter tuning trends
```
- Real-time model loading and inference
- SHAP integration for instance-level explanations
- Plotly for interactive visualizations

### **8. Utility Modules (src/utils/)**
- **metrics.py**: Reusable MetricsCalculator class
- **visualization.py**: VisualizationHelper for consistent styling
- **model_io.py**: Model loading/saving utilities

## 📊 Key Results

| Rank | Classifier | Accuracy | F1-Score | Recall | Precision |
|------|-----------|----------|----------|---------|-----------|
| 🥇 | Gradient Boosting | 99.41% | 93.27% | 95.97% | 90.86% |
| 🥈 | Decision Tree | 99.12% | 93.56% | 97.44% | 90.10% |
| 🥉 | Random Forest | 98.83% | 98.70% | 98.72% | 98.68% |
| 4 | DenseNN (Shallow) | 97.36% | 67.68% | 58.97% | 80.99% |

**Medical Insight**: Gradient Boosting achieves near-perfect accuracy while maintaining 95.97% recall – critical for preventing missed diagnoses.

## 🎓 Academic & Portfolio Value

- **Completeness**: Full ML pipeline from data to deployment
- **Advanced Techniques**: Bayesian optimization, SHAP explainability, statistical testing
- **Production Elements**: Docker, MLflow tracking, web dashboard, professional reports
- **Reproducibility**: Fixed seeds, detailed configuration, version-controlled documentation
- **Best Practices**: Type hints, logging, modular design, comprehensive testing

---

**See [README.md](README.md) for installation and usage instructions.**
**See [RESULTS.md](RESULTS.md) for detailed performance analysis.**