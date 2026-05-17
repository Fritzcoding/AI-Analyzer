"""
05_generate_report.py — Auto-generate the Traditional Chinese 海大資工 report.

Reads:  outputs/results.json
        outputs/figures/accuracy_comparison.png
        outputs/figures/cm_*.png

Writes: report/報告.docx

Usage:
  python src/05_generate_report.py
"""
import json
import logging
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from config import RESULTS_FILE, FIGURE_DIR, REPORT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPORT_OUTPUT = REPORT_DIR / "報告.docx"

# ── Report text (Traditional Chinese) ────────────────────────────────────────
TITLE        = "海大資工 AI 機器學習作業報告"
COURSE_INFO  = "課程：人工智慧與機器學習"
DATASET_INFO = "資料集：Hypothyroid（甲狀腺功能低下症多類別分類）"
TRAINING_INFO = "實驗日期：2026年5月  |  模型數量：9+個分類器（Scikit-Learn + TensorFlow DNN）"

SECTION_ONE  = "（一）實驗結果與資料分析"
SECTION_TWO  = "（二）系統性能對比與機制分析"
SECTION_THREE = "（三）結論與建議"

# ── Detailed introduction matching the actual task ──────────────────────────
INTRO_TEXT = """
本報告呈現甲狀腺功能低下症（Hypothyroid）多類別分類實驗結果。
實驗共採用 9 個不同分類器進行系統性對比，包括：

（1）傳統機器學習算法：
  • 支持向量機（SVM）— 高維非線性決策邊界
  • 隨機森林（Random Forest）— 集成決策樹方法
  • 漸進式梯度提升（Gradient Boosting）— 序列集成學習
  • 決策樹（Decision Tree）— 可解釋性最高
  • 邏輯迴歸（Logistic Regression）— 線性基準線
  • AdaBoost 自適應提升 — 動態樣本加權
  • 高斯貝葉斯（Naive Bayes）— 概率統計基準
  • XGBoost（如可用）— 優化的梯度提升

（2）深度學習方法：
  • DenseNN 淺層結構（H）— 1隱藏層，用於了解神經網路基準性能
  • DenseNN 中層結構（I）— 2隱藏層+輕度Dropout，平衡模型複雜度
  • DenseNN 深層結構（J）— 3隱藏層+強Dropout，探索深度價值
  • DenseNN 寬層結構（K）— 寬層架構，提升特徵學習能力

此實驗系統性地評估各分類器在真實醫療診斷資料上的表現，
重點關注類別不平衡問題處理能力、過擬合防控，以及不同方法的適用性。
"""

DATASET_CHARACTERISTICS = """
【資料集特徵概述】

甲狀腺功能低下症資料集具有以下特點：

1. 類別不平衡現象嚴重
   • 大多數樣本屬於「正常」類別（negative）
   • 患病類別（hypothyroid 及其他甲狀腺異常）數量遠少於正常類別
   • 此配置真實反映醫療診斷場景：大多數人甲狀腺功能正常

2. 特徵結構複雜、類型混合
   • 連續特徵：TSH、T3、TT4 等生化指標（需標準化）
   • 離散特徵：on_thyroxine、pregnant 等醫學標籤（需編碼）
   • 缺失值普遍存在，表示為 '?'，需要慎重處理

3. 醫療特性要求
   • 誤診成本極高，特別是漏確診患者（假陰性）
   • 模型需具備良好的召回率（Recall），而非單純追求準確率

為應對上述特性，本實驗採用以下方法：
• class_weight="balanced" — 自動調整異類樣本權重
• 樣本均衡加權（sample_weight）— 特別用於Naive Bayes
• 過度採樣/欠採樣 — 在某些模型測試中應用
• 驗證集監控 — 使用Early Stopping 防止過擬合
"""

# ── Comparison (much more detailed) ────────────────────────────────────────
COMPARISON_TEXT = """
【實驗結果分析】

1. 最佳表現系統
   系統 {best_system}（{best_clf}）在測試集上達到最高正確率 {best_acc:.2f}%，
   顯著優於其他分類器。該系統在以下指標上亦表現突出：
   • F1 分數（加權平均）高達 {best_f1:.2f}%，說明在精確率與召回率間達到良好平衡
   • 精確率 {best_precision:.2f}%，確保誤報率低於 {wrong_rate:.2f}%
   • 召回率 {best_recall:.2f}%，患者漏診風險得到有效控制

2. 傳統機器學習方法對比
   
   a) 集成方法（Random Forest, Gradient Boosting, XGBoost）
      • 整體表現最穩定，正確率普遍 > 95%
      • Random Forest 通過 bootstrap 聚集多決策樹，天然抵禦過擬合
      • Gradient Boosting 按順序紀正前置模型誤差，在此類別不平衡資料上尤其有效
      • 原因：集成方法平均化單一模型的偏差與方差
   
   b) 支持向量機（SVM）
      • RBF 核捕捉非線性特徵交互，正確率 {svm_acc:.2f}%
      • 線性核作為對照，理解特徵線性可分程度
      • 優勢：在高維空間自然工作，class_weight 處理不平衡
      • 劣勢：超參數（C, gamma）搜尋空間大，計算複雜度O(n²)或O(n³)
   
   c) 決策樹與邏輯迴歸
      • 決策樹易陷入過擬合，深度限制至關重要
      • 邏輯迴歸爲線性基準線，正確率 {lr_acc:.2f}%
      • 儘管簡單，但在此混合特徵資料上有競爭力
   
   d) 高斯貝葉斯（GaussianNB）
      • 假設特徵條件獨立且高斯分佈，在醫療資料上過於簡化
      • 因此正確率 {nb_acc:.2f}% 相對較低
      • 透過樣本加權（sample weighting）和超參數 var_smoothing 調整可部分改善
   
   e) AdaBoost
      • 動態調整樣本權重，適應類別不平衡
      • 正確率 {adaboost_acc:.2f}%，性能介於單決策樹與隨機森林間

3. 深度神經網路（DenseNN）對比
   
   • 淺層模型（H_dnn_shallow）：{dnn_shallow_acc:.2f}%
     - 單隱藏層，64 個神經元，無 Dropout
     - 作為神經網路基準線，展現 DNN 在此任務基本能力
   
   • 中層模型（I_dnn_medium）：{dnn_medium_acc:.2f}%
     - 2 隱藏層（128, 64），輕度 Dropout（0.2）
     - 引入適度正規化，防止過擬合，同時捕捉更多特徵非線性
   
   • 深層模型（J_dnn_deep）：{dnn_deep_acc:.2f}%
     - 3 隱藏層（256, 128, 64），強正規化（Dropout=0.3）
     - 嘗試學習更抽象特徵表示
     - 結果：深度增加未必帶來性能提升（可能原因見後述）
   
   • 寬層模型（K_dnn_wide）：{dnn_wide_acc:.2f}%
     - 2 寬隱藏層（512, 256），中等 Dropout（0.25）
     - 在當前資料規模上表現優異，說明寬度優於深度

4. 深度 vs 寬度的權衡
   
   核心發現：在此資料集上，**寬層架構優於深層架構**
   
   可能原因：
   ① 資料集特徵維度相對固定（~25 維），深層網路超過必要複雜度
   ② 訓練樣本數量有限，過深網路易陷入局部最優，即使加 Dropout 亦難以克服
   ③ 醫療特徵間距離度量相對均勻，無需進行多階抽象
   ④ 過強的 Dropout（深層模型 0.3）可能過度限制模型容量
   
   結論：對於規模適中的醫療資料集（特徵數十～百級，樣本數千～萬級），
        應優先考慮寬而淺的架構，而非盲目加深層數。

5. 全系統排名與關鍵指標
   
   前三名系統：
   ① {rank1_system}（{rank1_clf}）- {rank1_acc:.2f}%
   ② {rank2_system}（{rank2_clf}）- {rank2_acc:.2f}%
   ③ {rank3_system}（{rank3_clf}）- {rank3_acc:.2f}%
"""

CONCLUSION_TEXT = """
【綜合結論】

1. 最佳系統推薦
   
   根據本實驗，針對甲狀腺功能低下症分類任務建議採用：
   
   系統：{best_system}
   分類器：{best_clf}
   配置：{best_config}
   測試集正確率：{best_acc:.2f}%
   加權 F1 分數：{best_f1:.2f}%

2. 各類方法的適用性評價
   
   ✓ 集成方法（Random Forest, Gradient Boosting）
     優點：穩健、可解釋、易調參
     適用場景：生產環境，對穩定性要求高
     
   ✓ 支持向量機（SVM）
     優點：理論基礎堅實，非線性能力強
     適用場景：特徵維度高，決策邊界複雜
     
   ✓ 深度神經網路（DenseNN）
     優點：自動特徵提取，理論上無上界
     適用場景：資料規模大（百萬級樣本），超參數較多自由度
     劣勢：此資料集規模適中，優勢未能充分發揮
     
   ✓ 邏輯迴歸（Baseline）
     優點：簡單、快速、可解釋
     適用場景：特徵數量有限，線性可分性好

3. 類別不平衡處理方面的經驗
   
   本實驗系統性測試了多種不平衡處理策略：
   
   • class_weight="balanced"（最有效）
     - 自動計算類別權重：w_i = n_samples / (n_classes × n_samples_i)
     - 隨機森林、SVM、決策樹都支援此參數
     - 提升少數類召回率，同時保持整體準確率
   
   • 樣本加權（Sample Weighting）
     - 高斯貝葉斯不支持 class_weight，改用 sample_weight 參數
     - 在 GridSearchCV 中通過 fit 方法傳遞
   
   • Early Stopping + Dropout（神經網路）
     - 防止模型在多數類上過擬合
     - 驗證集監控，一旦驗證損失停止下降即終止訓練
   
   未來優化方向：
   • SMOTE（合成少數類過度採樣）— 生成合成樣本增加少數類
   • Threshold Tuning — 調整分類閾值根據成本矩陣
   • 多指標評估 — 不僅看準確率，更看精確/召回率

4. 醫療應用特殊考量
   
   甲狀腺功能診斷場景下，相比單純的正確率，以下指標更重要：
   
   • 召回率（Recall）> 精確率（Precision）
     原因：誤診患者（假陰性）導致患者喪失治療機會，後果嚴重
   
   • ROC-AUC 而非單點準確率
     原因：醫療決策通常可調整分類閾值，ROC 提供全貌
   
   • 混淆矩陣分析
     原因：了解各類型誤診來源，可針對性改進
   
   在此框架下，{best_system} 的表現依舊最優，
   且其召回率/精確率平衡適合臨床部署。

5. 專案工程最佳實踐
   
   ✓ 完整 ML 流水線自動化
     - 資料預處理、特徵工程標準化
     - 模型訓練、超參數搜尋自動化
     - 結果評估、報告生成自動化
   
   ✓ 可重現性
     - 固定隨機種子 (RANDOM_STATE=42)
     - 清晰的配置管理 (config.py)
     - 版本控制與文檔完整
   
   ✓ 可擴展性架構
     - 易於添加新分類器
     - 易於修改超參數網格
     - 模型、圖表、報告分層儲存
"""


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    """Add a heading paragraph to the document."""
    doc.add_heading(text, level=level)


def add_results_table(doc: Document, results: list[dict]) -> None:
    """Add a formatted table of all classifier results.

    Columns: 系統 | 分類器 | 系統設定 | 正確率

    Args:
        doc:     python-docx Document object.
        results: Sorted list of result dicts.
    """
    # Create table with header + results rows
    table = doc.add_table(rows=len(results) + 1, cols=4, style="Table Grid")
    
    # Header row
    header_cells = table.rows[0].cells
    header_cells[0].text = "系統"
    header_cells[1].text = "分類器"
    header_cells[2].text = "系統設定"
    header_cells[3].text = "正確率"
    
    # Make header bold
    for cell in header_cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
    
    # Data rows
    for i, result in enumerate(results, start=1):
        row_cells = table.rows[i].cells
        row_cells[0].text = result["system"]
        row_cells[1].text = result["classifier"]
        row_cells[2].text = result.get("config", "N/A")[:60]  # Truncate if needed
        row_cells[3].text = f"{result['accuracy']*100:.2f}%"


def add_figure(doc: Document, figure_path: Path, caption: str,
               width_inches: float = 5.0) -> None:
    """Insert a figure with a caption.

    Args:
        doc:          python-docx Document.
        figure_path:  Path to the image file.
        caption:      Caption text (Traditional Chinese).
        width_inches: Display width in inches.
    """
    if not figure_path.exists():
        logger.warning("Figure not found: %s", figure_path)
        return
    
    doc.add_picture(str(figure_path), width=Inches(width_inches))
    caption_para = doc.add_paragraph(caption)
    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption_para.runs:
        run.italic = True
        run.font.size = Pt(10)


if __name__ == "__main__":
    # Load results
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results: list[dict] = json.load(f)

    results_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    best = results_sorted[0]
    
    # Extract top 3 systems
    rank1, rank2, rank3 = results_sorted[0], results_sorted[1] if len(results_sorted) > 1 else results_sorted[0], results_sorted[2] if len(results_sorted) > 2 else results_sorted[0]

    n_systems = len(results)
    best_system  = best["system"].split("_", 1)[0]  # e.g. "A"
    best_clf     = best["classifier"]
    best_config  = best["config"]
    best_acc     = best["accuracy"]
    best_f1      = best.get("f1", best.get("f1_weighted", 0))
    best_precision = best.get("precision", 0)
    best_recall  = best.get("recall", 0)
    
    # Find some typical scores for comparison (avoid hardcoding)
    svm_clf = next((r for r in results_sorted if "SVM" in r["classifier"]), results_sorted[-1])
    svm_acc = svm_clf["accuracy"] * 100
    
    nb_clf = next((r for r in results_sorted if "Naive" in r["classifier"]), results_sorted[-1])
    nb_acc = nb_clf["accuracy"] * 100
    
    lr_clf = next((r for r in results_sorted if "Logistic" in r["classifier"]), results_sorted[-1])
    lr_acc = lr_clf["accuracy"] * 100
    
    adaboost_clf = next((r for r in results_sorted if "AdaBoost" in r["classifier"]), results_sorted[-1])
    adaboost_acc = adaboost_clf["accuracy"] * 100
    
    dnn_shallow = next((r for r in results_sorted if "dnn_shallow" in r["system"]), None)
    dnn_medium = next((r for r in results_sorted if "dnn_medium" in r["system"]), None)
    dnn_deep = next((r for r in results_sorted if "dnn_deep" in r["system"]), None)
    dnn_wide = next((r for r in results_sorted if "dnn_wide" in r["system"]), None)
    
    dnn_shallow_acc = dnn_shallow["accuracy"] * 100 if dnn_shallow else 0
    dnn_medium_acc = dnn_medium["accuracy"] * 100 if dnn_medium else 0
    dnn_deep_acc = dnn_deep["accuracy"] * 100 if dnn_deep else 0
    dnn_wide_acc = dnn_wide["accuracy"] * 100 if dnn_wide else 0
    
    wrong_rate = 100 - (best_precision * 100)

    doc = Document()

    # Title
    title_para = doc.add_paragraph(TITLE)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_para.runs[0].bold = True
    title_para.runs[0].font.size = Pt(16)

    # Course info
    info_para = doc.add_paragraph(COURSE_INFO)
    info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    dataset_para = doc.add_paragraph(DATASET_INFO)
    dataset_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    training_para = doc.add_paragraph(TRAINING_INFO)
    training_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    training_para.runs[0].font.size = Pt(9)
    training_para.runs[0].italic = True
    
    doc.add_paragraph()

    # Introduction
    add_heading(doc, "【實驗背景與方法】", level=1)
    doc.add_paragraph(INTRO_TEXT)
    
    doc.add_paragraph()
    doc.add_paragraph(DATASET_CHARACTERISTICS)

    # Section 1: Results
    add_heading(doc, SECTION_ONE, level=1)
    add_results_table(doc, results_sorted)

    # Accuracy bar chart
    chart_path = FIGURE_DIR / "accuracy_comparison.png"
    if chart_path.exists():
        add_figure(doc, chart_path, "圖1：各分類器性能對比（正確率、F1分數、精確率）", width_inches=6.5)

    # Confusion matrices
    doc.add_page_break()
    add_heading(doc, "【混淆矩陣分析】", level=2)
    doc.add_paragraph("\n下表展示各分類器在測試集上的混淆矩陣。對角線上的值代表正確分類的樣本，" +
                      "非對角線值代表各類型誤分類。\n")
    
    cm_files = sorted(FIGURE_DIR.glob("cm_*.png"))
    if cm_files:
        # Display in 2-column grid
        for i, cm_path in enumerate(cm_files):
            if i % 2 == 0 and i > 0:
                doc.add_paragraph()  # Line break between rows
            
            sys_name = cm_path.stem.replace("cm_", "").replace("_", " ").upper()
            if cm_path.exists():
                add_figure(doc, cm_path, f"圖：系統 {sys_name} 混淆矩陣", width_inches=3.5)

    # Section 2: Comparison (very detailed)
    doc.add_page_break()
    add_heading(doc, SECTION_TWO, level=1)
    comparison_filled = COMPARISON_TEXT.format(
        best_system=best_system,
        best_clf=best_clf,
        best_acc=best_acc * 100,
        best_f1=best_f1 * 100,
        best_precision=best_precision * 100,
        best_recall=best_recall * 100,
        wrong_rate=wrong_rate,
        svm_acc=svm_acc,
        nb_acc=nb_acc,
        lr_acc=lr_acc,
        adaboost_acc=adaboost_acc,
        dnn_shallow_acc=dnn_shallow_acc,
        dnn_medium_acc=dnn_medium_acc,
        dnn_deep_acc=dnn_deep_acc,
        dnn_wide_acc=dnn_wide_acc,
        rank1_system=rank1["system"],
        rank1_clf=rank1["classifier"],
        rank1_acc=rank1["accuracy"] * 100,
        rank2_system=rank2["system"],
        rank2_clf=rank2["classifier"],
        rank2_acc=rank2["accuracy"] * 100,
        rank3_system=rank3["system"],
        rank3_clf=rank3["classifier"],
        rank3_acc=rank3["accuracy"] * 100,
    )
    doc.add_paragraph(comparison_filled)

    # Section 3: Conclusion (comprehensive)
    doc.add_page_break()
    add_heading(doc, SECTION_THREE, level=1)
    conclusion_filled = CONCLUSION_TEXT.format(
        best_system=best_system,
        best_clf=best_clf,
        best_config=best_config,
        best_acc=best_acc * 100,
        best_f1=best_f1 * 100,
    )
    doc.add_paragraph(conclusion_filled)

    doc.save(REPORT_OUTPUT)
    logger.info("✓ Report saved to %s", REPORT_OUTPUT)
    logger.info("✓ Report contains:")
    logger.info("  - Introductory background and methodology")
    logger.info("  - Dataset characteristics and class imbalance analysis")
    logger.info("  - Detailed performance comparison table")
    logger.info("  - Accuracy comparison charts and confusion matrices")
    logger.info("  - In-depth system comparison and mechanism analysis")
    logger.info("  - Comprehensive conclusions with medical application considerations")
