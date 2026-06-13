# MathBERT — MAP: Charting Student Math Misunderstandings

以 **MathBERT** 為基底的數學迷思概念（Misconception）分類模型，針對 Kaggle 競賽 [MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) 設計。採用 **5-Fold StratifiedKFold 訓練 + 機率平均 Ensemble 推論**，輸出每筆資料 Top-3 的 `Category:Misconception` 預測。

> **執行環境：Kaggle Notebook（GPU）。** 所有相依套件 Kaggle 環境皆已內建，無需額外安裝，開啟 GPU 加速器即可直接 Run All。

---

## 專案結構

```
mathbert-map-v2.ipynb     # 主程式（4 個 cell）
requirements.txt          # 環境紀錄（Kaggle 內建版本）
```

| Cell | 功能 |
|------|------|
| 1 | 匯入套件、設定隨機種子（seed=42）、偵測 GPU |
| 2 | 讀取資料、缺值處理、標籤合併與編碼、初始化 5-Fold 分層抽樣 |
| 3 | K-Fold 訓練核心：FP16 混合精度 + Label Smoothing + Cosine Scheduler，每個 Fold 儲存最佳權重 |
| 4 | 載入 5 個 Fold 權重推論、機率平均 Ensemble、產出 `submission.csv` |

---

## 方法說明

### 輸入格式

將題目、學生選答與解釋拼接為單一序列：

```
Question: {QuestionText} [SEP] Answer: {MC_Answer} [SEP] Explanation: {StudentExplanation}
```

### 標籤設計

`Category` 與 `Misconception` 以 `:` 合併為單一分類目標（`Misconception` 缺值補為 `NA`），再以 `LabelEncoder` 轉為整數標籤。

### 訓練設定

| 項目 | 設定值 |
|------|--------|
| 基底模型 | MathBERT（本地權重，掛載自 Kaggle Dataset） |
| 交叉驗證 | StratifiedKFold，n_splits=5，shuffle=True，seed=42 |
| 最大序列長度 | 256 |
| Batch Size | 16 |
| Epochs | 4（每 Fold） |
| Loss | CrossEntropyLoss + Label Smoothing 0.05 |
| 學習率排程 | Cosine schedule with warmup |
| 混合精度 | FP16（`autocast` + `GradScaler`） |
| 梯度裁剪 | max_norm=1.0 |

每個 Fold 以驗證集表現保存最佳權重為 `best_mathbert_fold{1..5}.pt`。

### 推論與 Ensemble

1. 依序載入 5 個 Fold 的最佳權重對測試集推論（FP16）。
2. 將 5 組 softmax 機率矩陣 **取平均**（soft voting）。
3. 取平均機率最高的 **Top-3** 類別，反轉編碼後以空格串接。
4. 輸出官方格式 `submission.csv`（欄位：`row_id`、`Category:Misconception`）。

---

## 在 Kaggle 上執行

### 1. 掛載資料來源（Add Input）

| 來源 | 路徑 |
|------|------|
| 競賽資料 | `/kaggle/input/competitions/map-charting-student-math-misunderstandings/` |
| MathBERT 本地權重（Dataset） | `/kaggle/input/datasets/huangtzuchen/my-mathbert-weights/mathbert-local` |

> 若你的帳號掛載後路徑不同（例如沒有 `competitions/`、`datasets/` 中間層），請對應修改 Cell 2 的 `DATA_DIR` 與 Cell 3 的 `MODEL_NAME`。

### 2. 設定 Notebook

- **Accelerator**：GPU（T4 x2 或 P100 皆可）
- **Internet**：可關閉（模型權重走本地 Dataset，不需連網）

### 3. 執行

Run All 即可。流程為：訓練 5 個 Fold（各自存檔）→ 5 模型 Ensemble 推論 → 產出 `submission.csv` 於 `/kaggle/working/`，可直接 Submit。

---

## 環境與版本

本專案僅使用 Kaggle 內建套件：`torch`、`transformers`、`huggingface-hub`、`accelerate`、`pandas`、`numpy`、`scikit-learn`、`tqdm`。完整版本紀錄見 `requirements.txt`。

若需鎖定執行當下的精確版本，在 notebook 第一個 cell 執行：

```python
!pip list --format=freeze 2>/dev/null | grep -iE "^(torch|transformers|tokenizers|huggingface|accelerate|pandas|numpy|scikit-learn|tqdm)==" > requirements.lock.txt
```

---

## 注意事項與常見問題

- **`torch.cuda.amp` 棄用警告**：新版 PyTorch（2.4+）會對 `from torch.cuda.amp import autocast, GradScaler` 顯示 FutureWarning，目前不影響執行；若要消除警告，改用 `torch.amp.autocast('cuda')` 與 `torch.amp.GradScaler('cuda')`。
- **執行時間**：5 Folds × 4 Epochs，在 P100 上約需數小時，請留意 Kaggle GPU 每週配額（30 hr）與單次 session 上限（12 hr）。
- **記憶體**：MAX_LEN=256、BATCH_SIZE=16 在 16GB GPU 上可正常執行；若遇 OOM 可將 BATCH_SIZE 降為 8 並搭配梯度累積。
- **再現性**：已固定 seed=42（numpy / torch / CUDA），但 FP16 與 cuDNN 非確定性演算法仍可能造成極小差異，屬正常現象。
- **`row_id` 欄位**：Cell 4 會優先使用 `test.csv` 的 `row_id`；若無此欄位則以 `QuestionId_MC_Answer` 組合代替，提交前請確認與官方 `sample_submission.csv` 格式一致。

---

## 團隊

Chang Gung University — AI 學程畢業專題團隊（指導教授：Prof. Hojjat Baghban）
