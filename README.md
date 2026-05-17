# 甲狀腺功能低下症 AI 多分類器診斷系統
**Hypothyroid ML: Multi-Classifier Diagnostic System**

> 使用 scikit-learn 及 TensorFlow 構建的醫療診斷級機器學習分類系統。
> 經過系統性實驗驗證 **13 個分類器**，包括傳統機器學習方法與深度學習架構，
> 在甲狀腺功能低下症診斷上達成 **99.41% 準確率**。

---

🌐 **[→ 英文版本 (COMPREHENSIVE ENGLISH README)](README_ENGLISH.md)** 更為詳盡，包含完整的 10 階段管道說明與動機介紹。

---

**📖 新使用者？** [→ 閱讀 DESCRIPTION.md](DESCRIPTION.md) 了解項目概述、完整技術棧和實現方法

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Academic-orange)

---

## 🎯 Project Overview

本項目是一個**生產級 ML 管道**，展示了完整的機器學習工作流程，從數據預處理到模型評估和報告生成。

### ✨ 核心特性

- **13 個分類器對比**：7 個 scikit-learn + 4 個 TensorFlow DenseNN + 2 個集成方法
- **智能類別不平衡處理**：樣本加權、class_weight、EarlyStopping、Dropout 正規化
- **多維度性能評估**：準確率、精確率、召回率、F1-Score、混淆矩陣
- **深 vs 寬架構研究**：實驗驗證淺層寬網絡優於深層網絡（該數據集）
- **醫療場景優化**：強調召回率優先於準確率，防止誤診（假陰性）
- **自動化完整管道**：從原始 ARFF 檔案到專業報告一鍵生成
- **可重現性保障**：固定隨機種子、詳細配置、版本控制就緒
- **傳統中文報告**：4-5 頁專業級報告，包含深度機制分析

### 🎯 **[NEW] Portfolio-Grade Features**
- **Interactive Dashboard**: Streamlit-based model benchmarking interface
- **Advanced Metrics**: ROC-AUC, calibration curves, statistical testing
- **Model Explainability**: SHAP integration for feature importance
- **Utility Modules**: Reusable metrics, visualization, and model I/O libraries
- **Professional Documentation**: Architecture, deployment, and API guides
- **Production Ready**: Docker support, environment configuration, comprehensive logging

### 📊 性能成果

| 排名 | 分類器 | 準確率 | F1-Score |
|------|--------|--------|----------|
| **🥇** | **Gradient Boosting** | **99.41%** | **93.27%** |
| 🥈 | Decision Tree | 99.12% | 93.56% |
| 🥉 | Random Forest | 98.83% | 98.70% |
| 4 | DenseNN (Shallow) | 97.36% | 67.68% |
| 5 | DenseNN (Medium) | 97.07% | 67.12% |

👉 **詳細結果見 [RESULTS.md](RESULTS.md)**

---

## 🛠️ 技術棧

| 領域 | 技術 | 版本 | 用途 |
|------|------|------|------|
| **數據處理** | pandas, numpy, scipy | 1.5+, 1.23+, 1.10+ | DataFrame操作、array計算、ARFF加載 |
| **ML 框架** | scikit-learn | 1.3+ | 7個分類器、GridSearchCV超參數調優 |
| **深度學習** | TensorFlow/Keras | 2.11+ | Functional API、DenseNN、EarlyStopping、class_weight |
| **超參數優化** | Optuna | 3.5+ | 貝葉斯優化（TPE采樣）、自動調優 |
| **可解釋性** | SHAP | 0.44+ | 特徵重要性分析、TreeExplainer |
| **數據格式** | liac-arff | 2.4+ | Weka ARFF格式解析 |
| **統計檢驗** | statsmodels | 0.14+ | McNemar檢驗、Friedman測試 |
| **報告生成** | python-docx | 0.8.10+ | 動態表格、圖表嵌入、繁體中文格式 |
| **可視化** | matplotlib, seaborn, plotly | 3.7+, 0.12+ | 混淆矩陣、損失曲線、互動圖表 |
| **實驗追蹤** | MLflow | 2.10+ | 運行追蹤、參數日誌、模型註冊 |
| **Web儀表板** | Streamlit | 1.28+ | 互動式模型基準測試界面 |
| **Python** | - | 3.10+ | 類型提示、異步支持、pathlib |

---

## 📋 Requirements

**系統需求**：
- Python 3.10+
- 2GB+ RAM（訓練全部模型）
- 500MB+ 磁盤空間（含模型和報告）

**核心依賴** — 見 [requirements.txt](requirements.txt)：
```
scikit-learn>=1.3.0
tensorflow>=2.11.0
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
liac-arff>=2.4.0
python-docx>=0.8.10
```

**儀表板依賴** — 見 [requirements-prod.txt](requirements-prod.txt)：
```
streamlit>=1.28.0
plotly>=5.17.0
shap>=0.44.0
```

**可選 (進階功能)**：
```
optuna>=3.5.0          # 貝葉斯超參數優化
imbalanced-learn>=0.11 # SMOTE過採樣
mlflow>=2.10.0         # 實驗追蹤
jupyter>=1.0.0         # 交互式筆記本
```

---

## 🚀 Installation & Setup

### 1️⃣ 環境準備

```bash
# 項目設置
cd hypothyroid-ml

# 創建虛擬環境（推薦）
python -m venv .venv

# 激活虛擬環境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安裝核心依賴（用於培訓管道）
pip install -r requirements.txt

# [可選] 安裝儀表板依賴（用於互動式界面）
pip install -r requirements-prod.txt
```

### 2️⃣ 數據準備

將甲狀腺功能低下症（Hypothyroid）ARFF 格式數據集檔案放入 `data/` 目錄：

```
data/
├── hypothyroid_cjlin2025_training.arff   (3,057 samples)
└── hypothyroid_cjlin2025_test.arff       (972 samples)
```

**數據集統計**：
- **Total Samples**: 4,029
- **Features**: ~25 (混合連續和離散)
- **Target Classes**: 4 (negative, compensated_hypothyroid, primary_hypothyroid, secondary_hypothyroid)
- **Class Balance**: 高度不平衡 (92% negative, 8% diseased)
- **Missing Values**: 編碼為 `?`，使用中位數/眾數填充

---

## 📖 Quick Start

### 完整管道（5個階段，~10-15分鐘）

```bash
# 階段 1: 數據預處理和特徵工程
python src/01_preprocessing.py
# 輸出: X_train, y_train, X_test, y_test (.npy)

# 階段 2: 訓練 7 個 scikit-learn 分類器（GridSearchCV）
python src/02_sklearn_classifiers.py
# 輸出: 7 個 .pkl 模型 + 混淆矩陣 PNG

# 階段 3: 訓練 4 個 TensorFlow DenseNN 架構
python src/03_tensorflow_dnn.py
# 輸出: 4 個 .keras 模型 + 訓練曲線 PNG

# 階段 4: 生成評估圖表和性能對比
python src/04_evaluate.py
# 輸出: accuracy_comparison.png + results.json

# 階段 5: 生成專業級傳統中文報告
python src/05_generate_report.py
# 輸出: report/報告.docx (4-5 pages)

# [NEW] 階段 6: 啟動互動式儀表板
pip install -r requirements-prod.txt
streamlit run app/streamlit_app.py
# 輸出: 在 http://localhost:8501 打開儀表板
```

### 互動式儀表板 (NEW)

```bash
# 安裝儀表板依賴
pip install streamlit plotly shap

# 運行儀表板
streamlit run app/streamlit_app.py

# 訪問 http://localhost:8501
# 可用頁面：
#  - 📊 Leaderboard: 模型排名和篩選
#  - 🔄 Comparison: 模型指標對比
#  - 🧠 Explainability: SHAP 特徵重要性
#  - 🏥 Predict: 患者預測界面
#  - 📈 Analysis: 超參數調優分析
```

### 運行個別階段

```bash
# 只訓練 sklearn 分類器
python src/02_sklearn_classifiers.py

# 只訓練 DNN 模型
python src/03_tensorflow_dnn.py

# 跳過訓練，直接生成報告
python src/04_evaluate.py && python src/05_generate_report.py

# 只運行儀表板（假設已有訓練的模型）
streamlit run app/streamlit_app.py
```

---

## 🏗️ Architecture & Classifiers

### 對照組設計

#### **A. 傳統機器學習（7 個）**

| 系統 | 分類器 | 關鍵超參數搜索 | 主要優勢 | 適用場景 |
|------|--------|-----------------|----------|----------|
| A | Naive Bayes (Gaussian) | var_smoothing | 快速、易解釋 | 高維稀疏數據 |
| B | SVM (RBF/Linear) | C, kernel, gamma | 非線性邊界、穩健 | 小~中規模數據 |
| C | Random Forest | n_estimators, max_depth | 集成、抗過擬合 | 通用、特徵重要性 |
| D | Gradient Boosting | n_estimators, lr, max_depth | 序列提升、高準確率 | **最佳通用選擇** |
| E | Decision Tree | max_depth, min_samples_leaf | 可解釋、快速 | 特徵交互分析 |
| F | Logistic Regression | C, solver | 線性基準、概率輸出 | 診斷應用、置信度 |
| G | AdaBoost | n_estimators, lr | 自適應加權、不平衡數據 | 類別不平衡場景 |

#### **B. 深度學習（4 個 DenseNN）**

| 系統 | 架構 | 隱藏層 | Dropout | 學習率 | 準確率 | 洞察 |
|------|------|--------|---------|--------|--------|----------|
| H | Shallow | [64] | 0.0 | 0.001 | 97.36% | DNN 基準線 |
| I | Medium | [128,64] | 0.2 | 0.001 | 97.07% | 中等複雜度 |
| J | Deep | [256,128,64] | 0.3 | 0.0005 | 95.89% | **深度>寬度失敗** |
| K | Wide | [512,256] | 0.2 | 0.001 | 96.48% | **寬度>深度成功** |

**關鍵發現**：深度 ≠ 優越
- 在適度規模數據集上（3K 訓練樣本），淺層和寬層模型優於深層
- 過度 Dropout（0.3）反而傷害模型容量
- 建議：**優先嘗試寬而淺的架構**

---

## 📊 Output Structure

```
.
├── data/
│   ├── hypothyroid_cjlin2025_training.arff
│   └── hypothyroid_cjlin2025_test.arff
├── src/
│   ├── 01_preprocessing.py          # ARFF 加載、特徵工程、管道構建
│   ├── 02_sklearn_classifiers.py    # 7 個 sklearn 分類器 + GridSearchCV
│   ├── 03_tensorflow_dnn.py         # 4 個 DenseNN 架構 + 類別加權
│   ├── 04_evaluate.py               # 評估、混淆矩陣、對比圖表
│   ├── 05_generate_report.py        # 報告生成（中文、4-5 頁、圖表嵌入）
│   ├── 06_explainability.py         # SHAP 特徵重要性分析
│   ├── 07_optuna_tuning.py          # 貝葉斯超參數優化
│   ├── 08_statistical_tests.py      # McNemar & Friedman 統計檢驗
│   ├── 09_mlflow_tracking.py        # 實驗追蹤與模型註冊
│   ├── 10_export_dashboard_data.py  # 儀表板數據導出
│   ├── config.py                    # 集中配置（路徑、超參數、隨機種子）
│   └── utils/
│       ├── metrics.py               # MetricsCalculator 實用類
│       ├── visualization.py         # 可視化幫手函數
│       └── model_io.py              # 模型輸入/輸出 utilities
├── app/
│   ├── streamlit_app.py             # [NEW] 主儀表板應用程式
│   ├── pages/                       # [NEW] 儀表板子頁面
│   └── __init__.py
├── outputs/
│   ├── models/
│   │   ├── preprocessor.pkl
│   │   ├── A_naive_bayes.pkl
│   │   ├── B_svm.pkl
│   │   ├── C_decision_tree.pkl
│   │   ├── D_random_forest.pkl
│   │   ├── E_knn.pkl
│   │   ├── F_logistic_regression.pkl
│   │   ├── G_adaboost.pkl
│   │   ├── H_dnn_shallow.keras
│   │   ├── I_dnn_medium.keras
│   │   ├── J_dnn_deep.keras
│   │   ├── K_dnn_wide.keras
│   │   └── X_train.npy, y_train.npy, X_test.npy, y_test.npy
│   ├── figures/
│   │   ├── accuracy_comparison.png  # 多指標對比圖表
│   │   ├── cm_A_naive_bayes.png
│   │   ├── cm_B_svm.png
│   │   ├── ... (各分類器混淆矩陣及其他圖表)
│   │   ├── dnn_H_dnn_shallow_history.png
│   │   └── ... (各 DNN 訓練曲線)
│   └── results.json                 # 所有指標 JSON 匯總
├── report/                          # [已棄用] 舊報告存檔
├── tests/                           # 單元測試與集成測試
├── config.py
├── requirements.txt
├── requirements-prod.txt            # [NEW] 儀表板依賴
├── DESCRIPTION.md                   # [NEW] 項目簡介和技術棧
├── README.md                        # 此檔案，項目完整說明
├── RESULTS.md                       # 詳細結果分析
├── ARCHITECTURE.md                  # 系統架構與設計
├── DEPLOYMENT.md                    # 部署指南
├── AGENTS.md                        # GitHub Copilot 代理職責
└── .github/
    └── copilot-instructions.md      # Copilot 全局規則
```

**設計理念**：
- ✅ 配置集中化（`config.py`）→ 易於修改參數
- ✅ 模型與圖表分離 → 獨立版本控制
- ✅ JSON 結果存儲 → 易於後處理和對比
- ✅ 報告自動生成 → 可重現性
- ✅ 儀表板資料導出 → 實時互動分析

---

## 🔬 Key Features

### 1. **智能類別不平衡處理**
```python
# 方法 1: class_weight="balanced" (sklearn)
RF = RandomForestClassifier(class_weight="balanced")

# 方法 2: 樣本加權 (Naive Bayes)
sample_weights = compute_sample_weight("balanced", y_train)
gs.fit(X_train, y_train, sample_weight=sample_weights)

# 方法 3: Early Stopping + Dropout (TensorFlow)
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint(path, save_best_only=True)
]
```

### 2. **多維度性能評估**
- 準確率（Accuracy）：整體正確分類比例
- 精確率（Precision）：正診斷中的真陽性比例
- 召回率（Recall）：真患者中的檢出比例
- F1-Score：精確率和召回率的調和平均
- **醫療應用優先順序**：Recall > Precision > Accuracy

### 3. **混淆矩陣深度分析**
```
針對每個分類器生成混淆矩陣，展示：
- 真陽性 (TP)：正確診斷患者
- 假陽性 (FP)：誤診正常人為患者
- 真陰性 (TN)：正確識別正常人
- 假陰性 (FN)：漏診患者 ⚠️ 最嚴重的錯誤
```

### 4. **自動化報告生成**
生成的報告包含：
- 📋 實驗背景與方法論
- 📊 數據集特徵與不平衡分析
- 📈 性能對比表和圖表
- 🔍 深入機制分析（為什麼 A 優於 B）
- 💡 深度 vs 寬度的權衡討論
- 🏥 醫療應用特殊考量
- ✅ 結論和建議

---

## 🧪 Experimental Protocol

### 訓練設置

```python
RANDOM_STATE = 42  # 可重現性
CV_FOLDS = 5       # 5 折交叉驗證
TEST_SIZE = 0.2    # 數據集分割（已預分，本項目使用官方 test set）
DNN_EPOCHS = 100   # 最大迭代數
DNN_PATIENCE = 10  # Early Stopping 耐心值
BATCH_SIZE = 64    # 小批量大小
```

### 超參數搜索

- **sklearn**: GridSearchCV（窮舉網格搜索）
- **TensorFlow**: 手動配置（實驗設計）

### 評估方法

- **分割策略**：官方 train/test 分割
- **驗證方法**：
  - sklearn: 5-fold CV 用於超參數選擇，test set 用於最終評估
  - TensorFlow: 10% validation_split，EarlyStopping 監控
- **指標**：Accuracy, Precision (Macro), Recall (Macro), F1 (Weighted)

---

## 📚 Documentation

- **[DESCRIPTION.md](DESCRIPTION.md)** — 項目簡介、完整技術棧、實現方法（新使用者從這裡開始）
- **[README.md](README.md)** — 本檔案，詳細操作指南和架構說明
- **[RESULTS.md](RESULTS.md)** — 實驗結果分析、圖表、性能對比、洞察
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — 系統架構、設計模式、代碼組織
- **[DEPLOYMENT.md](DEPLOYMENT.md)** — 部署指南、Docker、雲提供商配置
- **[AGENTS.md](AGENTS.md)** — GitHub Copilot 代理職責與工作流
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** — Copilot 全局規則
- **[src/config.py](src/config.py)** — 集中配置（路徑、常數、超參數）

---

## 🔄 Development Workflow

### 使用 GitHub Copilot 開發

1. 打開 `src/` 中的任何文件
2. 在函數簽名下放置光標
3. 在 Copilot Chat 中輸入：
   ```
   Implement this function following the docstring and AGENTS.md guidelines
   ```
4. 查看建議並按 Tab 接受

### 添加新分類器

編輯 `src/02_sklearn_classifiers.py` 的 `CLASSIFIERS` 字典：

```python
CLASSIFIERS["H_new_model"] = (
    NewClassifier(random_state=RANDOM_STATE),
    {"param1": [v1, v2], "param2": [v3, v4]},
    "顯示名稱"
)
```

然後重新運行 `python src/02_sklearn_classifiers.py`

---

## 🎓 Learning Resources

- scikit-learn 官方文檔：https://scikit-learn.org
- TensorFlow/Keras 指南：https://www.tensorflow.org/guide
- Class Imbalance：https://imbalanced-learn.org

---

## 🐛 Troubleshooting

### GPU 支持問題

```
警告：TensorFlow GPU 支持在 native Windows 上不可用
→ 使用 WSL2 或 TensorFlow-DirectML 插件
```

### 記憶體不足

```python
# 在 03_tensorflow_dnn.py 中減少 batch_size：
BATCH_SIZE = 32  # 默認 64

# 或減少 DNN 層大小
DNN_CONFIGS = [("H", [32], 0.0, 0.001, 64)]  # 替代 [64]
```

### 缺失依賴

```bash
pip install -r requirements.txt --upgrade
python -m pip cache purge
```

---

## 📊 Benchmarks

| 階段 | 耗時 | 輸出大小 |
|------|------|----------|
| 預處理 | ~2 秒 | 200 KB |
| sklearn 訓練 | ~3-5 分鐘 | 5-8 MB |
| DNN 訓練 | ~2-3 分鐘 | 3-5 MB |
| 評估生成 | ~10 秒 | 500 KB |
| 報告生成 | ~5 秒 | 2-3 MB |
| **總計** | **~10-15 分鐘** | **~15 MB** |

---

## 💡 Best Practices Applied

✅ **版本控制就緒**
- 所有代碼使用 Git，配置文件一致
- Reproducible：RANDOM_STATE=42

✅ **代碼品質**
- 類型提示在所有函數
- Google 風格 docstrings
- 詳細日誌記錄（logging 模塊，非 print）
- DRY 原則：配置集中化

✅ **機器學習最佳實踐**
- 嚴格的 train/test 分割
- 沒有從測試集報告訓練指標
- 交叉驗證用於超參數選擇
- 類別加權應對不平衡
- 標準化特徵用於距離型算法

✅ **可重現性**
- 固定隨機種子
- 清晰的配置管理
- 完整的依賴版本控制
- 詳細的文檔和註釋

✅ **生產就緒**
- 自動化 end-to-end 管道
- 錯誤處理和日誌記錄
- 模型持久化（joblib + TensorFlow SavedModel）
- 可擴展架構（易於添加新模型）

---

## 🤝 Contributing

這是一個學術項目。對改進的建議：

1. **新分類器**：編輯 `CLASSIFIERS` 字典
2. **新特徵工程**：修改 `01_preprocessing.py`
3. **超參數調整**：編輯 `config.py` 或 DNN_CONFIGS
4. **報告增強**：編輯 `05_generate_report.py` 的模板文本

---

## � Troubleshooting

**問題排查**：
1. 查看 [RESULTS.md](RESULTS.md) 中的詳細分析
2. 檢查 `outputs/` 目錄中的日誌和混淆矩陣
3. 運行 `python src/01_preprocessing.py` 驗證數據加載

**常見問題**：
- 代碼問題：檢查 `src/config.py` 中的路徑和超參數
- 性能問題：見 Troubleshooting 部分
- 其他：參考 AGENTS.md 了解代碼結構
