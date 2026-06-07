# Hypothyroid ML: Multi-Classifier Diagnostic System

使用 `hypothyroid_cjlin2025_training.arff` 與 `hypothyroid_cjlin2025_test.arff` 建立多分類診斷系統，並輸出專業醫療 AI 風格報告。

## 報告固定主結構

1. `(一)實驗結果`
2. `(二)系統比較`
3. `(三)結論`

## Environment

- Python 3.10+
- `data/hypothyroid_cjlin2025_training.arff`
- `data/hypothyroid_cjlin2025_test.arff`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline

```bash
python src/01_preprocessing.py
python src/02_sklearn_classifiers.py
python src/03_tensorflow_dnn.py
python src/04_evaluate.py
python src/05_generate_report.py
```

## Models

scikit-learn classifiers (>=4, current implementation 7~8):

- Gaussian Naive Bayes
- SVM
- Decision Tree
- Random Forest
- Gradient Boosting
- Logistic Regression
- AdaBoost
- XGBoost (optional, if installed)

TensorFlow classifiers:

- DenseNN shallow / medium / deep / wide

## Updated Outputs

- `outputs/models/`: fitted preprocessors, sklearn `.pkl`, TensorFlow `.keras`, processed arrays
- `outputs/results.json`: unified test-set metrics (accuracy, weighted F1, macro F1, recall, balanced accuracy, ROC-AUC, PR-AUC)
- `outputs/analysis_summary.json`: dataset and model diagnostics summary
- `outputs/figures/`:
  - `model_ranking_curve.png`
  - `roc_multi_model.png`
  - `pr_multi_model.png`
  - `class_recall_heatmap.png`
  - `feature_importance.png`
  - `learning_curve_best_model.png`
  - `stability_boxplot.png`
  - `error_distribution.png`
  - `calibration_plot.png`
  - `prediction_confidence_distribution.png`
- `report/報告.docx`: final polished Traditional Chinese report
- `report/報告.md`: markdown export
- `report/報告.pdf`: optional (auto-export when `docx2pdf` is available)

## Notes

- Final reported metrics are computed on **test set only**.
- Confusion matrices are removed from report visuals and replaced by clearer research-style plots.
- `RANDOM_STATE = 42` is used throughout the project.
