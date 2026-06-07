# AGENTS.md - Hypothyroid ML Classification Project

## Project Goal

Build classifiers for `hypothyroid_cjlin2025_training.arff` and `hypothyroid_cjlin2025_test.arff`, then generate a Traditional Chinese report with exactly these sections:

1. `(一) 實驗結果`
2. `(二) 系統比較`
3. `(三) 結論`

## Core Files

- `src/config.py`: central paths and constants
- `src/01_preprocessing.py`: ARFF loading, missing-value handling, encoding, scaling
- `src/02_sklearn_classifiers.py`: sklearn classifiers and GridSearchCV
- `src/03_tensorflow_dnn.py`: TensorFlow DenseNN variants
- `src/04_evaluate.py`: summary chart from `outputs/results.json`
- `src/05_generate_report.py`: final `report/報告.docx`

## Requirements

- Python 3.10+
- Use `pathlib.Path`; do not hardcode absolute paths.
- Use `logging`, not `print`, for script progress.
- Type hints are required on function signatures.
- Keep constants in ALL_CAPS near the top of each file.
- Use `RANDOM_STATE = 42`.
- Final reported metrics must be evaluated on the test set only.
- Do not use pickle for TensorFlow models; save them with `.keras`.

## Model Requirements

Train at least four scikit-learn classifiers. The current implementation includes:

- Gaussian Naive Bayes
- SVM
- Random Forest
- Gradient Boosting
- Decision Tree
- Logistic Regression
- AdaBoost

Train at least one TensorFlow neural network. The current implementation includes four DenseNN variants.

## Output Requirements

- Save models in `outputs/models/`.
- Save figures in `outputs/figures/`.
- Save unified metrics in `outputs/results.json`.
- Save the final report as `report/報告.docx`.
