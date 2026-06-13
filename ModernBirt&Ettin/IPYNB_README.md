# MAP — Charting Student Math Misunderstandings
## ModernBERT & Ettin Encoder 系列比較

> Kaggle 競賽「MAP - Charting Student Math Misunderstandings」實作
> 負責部分：**ModernBERT 與 Ettin encoder 模型的訓練、評估與比較**

---

## 專案簡介

給定學生對數學選擇題的**開放式文字解釋**，預測最多 3 個最可能的 `Category:Misconception` 標籤（評估指標 **MAP@3**）。

本專案以**最精簡充分設計**部署三個 encoder 模型，形成兩個正交的受控比較，同時隔離 **scale（規模）** 與 **architecture（架構）** 兩個效應：

```
  MB-base (150M) ──── Scale Effect (+0.214) ────→ MB-large (395M)
                                                         │
                                                 Architecture Effect
                                                      (+0.017)
                                                         │
                                                         ↓
                                                  Ettin-400m (400M)
```

- **縱向（Scale 效應）**：ModernBERT-base → ModernBERT-large，固定架構，2.6× 參數量放大
- **橫向（Architecture 效應）**：ModernBERT-large vs Ettin-400m，參數量相近（395M vs 400M）、預訓練資料量相同（均 2T tokens）

---

## 實驗模型

| 模型 | Hugging Face ID | 參數量 | 架構類型 |
|------|----------------|--------|---------|
| ModernBERT-base | `answerdotai/ModernBERT-base` | ~150M | ModernBERT（RoPE + alternating attention） |
| ModernBERT-large | `answerdotai/ModernBERT-large` | ~395M | ModernBERT（同上，更大） |
| Ettin-encoder-400m | `jhu-clsp/ettin-encoder-400m` | ~400M | Bidirectional Ettin |

---

## 資料說明

| 檔案 | 筆數 | 欄位 |
|------|------|------|
| `train.csv` | 36,696 | `row_id, QuestionId, QuestionText, MC_Answer, StudentExplanation, Category, Misconception` |
| `test.csv` | 少量（本地測試；正式 re-run 約 16,000） | 同上去除標籤 |
| `sample_submission.csv` | — | 提交格式範例 |

**Label 結構**
- `Category`（6 種）× `Misconception`（35 種 + NA）= **65 個 label**
- Target = `Category + ":" + Misconception`（例如 `False_Misconception:Incomplete`）
- 嚴重類別不平衡：最多 14,802 筆 vs 最少 1 筆

**輸入文字長度**：平均約 37 字，95th percentile ≈ 63 字 → `max_length=64` token 已足夠。

---

## 方法與 Pipeline

完整流程：**資料探索 (EDA) → 資料切分 → 模型載入 → 訓練 → 結果分析 → Ensemble → Submission 生成**。

### 1. 輸入建構

三欄位拼接為單一輸入字串：

```
Question: {QuestionText}
Answer: {MC_Answer}
Explanation: {StudentExplanation}
```

### 2. 資料切分（90/10 stratified split）

- 依 label 做 stratified split
- 5 個只有 1 筆樣本的 label（singleton）強制併入 train、不參與 stratify
- 確保所有 65 個 label 都出現在訓練集中
- 結果：Train 33,026 筆 / Val 3,670 筆

### 3. Tokenization（動態 padding）

- `max_length=64`、`truncation=True`，不預先 padding
- `DataCollatorWithPadding(padding="longest")` 於 batch 內動態 padding，節省計算

### 4. 模型載入要點

- **ModernBERT 專屬設定**（以 `is_modernbert` flag 控制，避免把專屬參數傳給 Ettin 造成錯誤）：
  - `reference_compile=False`：關閉 `torch.compile`，與 HuggingFace Trainer 不相容
  - `classifier_pooling="mean"`：mean pooling 對長文比 CLS token 穩定
- **Attention**：`sdpa`（PyTorch 原生），Ampere GPU 完全支援，無需安裝 flash-attn
- **精度**：`torch_dtype=torch.bfloat16`，Ampere 原生 bf16 Tensor Core

### 5. 類別不平衡處理：Weighted CrossEntropyLoss

對 14,802:1 的不平衡，使用 per-class 權重：

```
w_i = N / (K · n_i)
```

其中 N = 總樣本數、K = label 數（65）、n_i = 第 i 個 label 的樣本數。
（自訂 `WeightedTrainer` 覆寫 `compute_loss`，並對齊 logits 的 device/dtype 避免 bf16 + float32 weights 衝突。）

### 6. 評估指標：MAP@3

```
MAP@3 = (1/N) · Σ AP@3(i)
```

每筆僅有一個真實 label，AP@3 最大為 1（命中 top-1）、最小為 0（未命中 top-3）。

---

## 訓練設定

| 超參數 | 值 | 理由 |
|--------|-----|------|
| `learning_rate` | 5e-5 | encoder fine-tune 標準起點 |
| `num_train_epochs` | 5 | 配合 EarlyStopping（patience=2） |
| `warmup_ratio` | 0.1 | 前 10% steps 線性 warmup |
| `weight_decay` | 0.01 | AdamW L2 正則化 |
| `bf16` | True | Ampere 原生支援、數值範圍大、不易 NaN |
| `tf32` | True | Ampere 矩陣乘法加速 |
| `save_total_limit` | 1 | 只保留最佳 checkpoint |
| `eval_strategy` / `save_strategy` | epoch | 每 epoch 評估，配合 EarlyStopping |
| `metric_for_best_model` | map3 | 以 MAP@3 選最佳 checkpoint |

**Batch size 規劃（2× RTX 3090，DataParallel）**

| 模型 | per-device batch | effective batch (×2 GPU) |
|------|------------------|---------------------------|
| ModernBERT-base | 32 | 64 |
| ModernBERT-large | 16 | 32 |
| Ettin-400m | 32 | 64 |

---

## 訓練環境

| 項目 | 規格 |
|------|------|
| GPU | 2× NVIDIA GeForce RTX 3090（各 ~25.8 GB，Ampere sm86） |
| 並行 | HuggingFace Trainer DataParallel（effective batch = per-GPU batch × 2） |
| 精度 | bfloat16 + tf32 矩陣加速 |
| Attention | sdpa（PyTorch 原生） |
| Framework | PyTorch 2.11.0+cu128 / transformers 5.6.2 / datasets 4.8.5 |

---

## 實驗結果

| 模型 | 設計目的 | 參數量 | effective batch | 訓練時長 | **Val MAP@3** |
|------|---------|--------|-----------------|----------|---------------|
| ModernBERT-base | 最佳化 encoder | 150M | 64 | 16.7 min | 0.6517 |
| Ettin-encoder-400m | 控制變數比較 | 400M | 64 | 33.8 min | 0.8488 |
| **ModernBERT-large** | **最佳化 encoder** | **395M** | **32** | **62.0 min** | **0.8653** |

### 三項關鍵發現

**發現一：規模效應最強（ΔMAP@3 = +0.2136）**
固定架構與訓練配方，base→large 的 MAP@3 從 0.6517 跳到 0.8653。確認參數量是下游性能最強的單一預測因子。

**發現二：任務類型決定相同規模下的勝負（ΔMAP@3 = +0.0165）**
ModernBERT-large vs Ettin-400m（參數量、資料量相近）相差 0.0165，有利 ModernBERT。雖然 Ettin 論文在 GLUE 平均（90.8 vs 90.4）與 MTEB v2（59.4 vs 58.6）略優，但勝負高度依賴任務類型——ModernBERT 在**單句分類**（SST-2）領先、Ettin 在**多句推理**（MRPC/MNLI/RTE）領先。本任務（將學生解釋對應到 65 個標籤）結構更接近單句多類分類，故 ModernBERT 的優勢得以發揮。

**發現三：領域知識的效益取決於任務認知需求**
本任務核心是「識別學生語言表達中的概念誤解模式」，考驗語言理解而非數學知識應用。Ettin 預訓練含科學論文，但對「語言模式分類」型任務優勢不顯著。

> **結論：Scale > Task-Type Alignment > Domain Knowledge > Fine-tuning Recipe**
> 選擇 pre-trained encoder 時，對齊「任務的認知結構（單句 vs 多句、理解 vs 推理）」比「訓練資料的領域相關性」更重要。建議用文獻中的任務類型對照表（如 Ettin 論文 Table 7）判斷模型在目標任務上的勝負，而非僅看整體平均分。

---

## Submission 與 Ensemble

最終提交採用 **3-model weighted ensemble**，以各模型 Val MAP@3 為權重做 logit 加權平均：

```
logits_ensemble = Σ w_m · logits_m ,   w_m = MAP@3_m / Σ MAP@3_j
```

權重：`modernbert_base=0.2755`、`ettin_400m=0.3588`、`modernbert_large=0.3657`。

**提交格式**（每列最多 3 個預測，空格分隔，依 softmax 機率由高到低）：

```
row_id,Category:Misconception
36696,True_Neither:NA True_Correct:NA True_Misconception:WNB
36697,False_Misconception:WNB False_Neither:NA False_Misconception:Incomplete
36698,True_Neither:NA True_Correct:NA False_Neither:NA
```

> 註：在僅 3 筆的本地 test set 上，weighted ensemble 與單一最佳模型（MB-large）產生相同 top-3，因 MB-large 權重最大且 logit 信心足以主導排序；ensemble 的多樣性效益預期在更大 test set 上顯現。

---

## 檔案結構

```
期末報告/
├── README.md
├── 第三組ModernBERT_&_Ettin_Encoder.ipynb  
```

---

## 如何執行

1. **環境**：Python 3.x、CUDA、PyTorch 2.11+cu128、transformers 5.6.2、datasets 4.8.5、scikit-learn、pandas、numpy、matplotlib
   ```bash
   pip install torch transformers datasets scikit-learn pandas numpy matplotlib tqdm
   ```
2. **資料**：將競賽資料放入 `map-charting-student-math-misunderstandings/`（notebook 會自動偵測本機 / Kaggle 環境設定 `DATA_DIR`、`RESULTS_DIR`）。
3. **執行 notebook**：依序執行 `第三組ModernBERT_&_Ettin_Encoder.ipynb` 的所有 cell。訓練順序為 base → ettin_400m → large；已訓練完成（存在 `val_map3.txt`）的模型會自動跳過。
4. **產出**：各模型的 checkpoint、訓練曲線、比較表，以及最終 `submission.csv`。

---

## 未來工作

- **Ettin-1B 的 Scale 對照**：補全 Ettin 系列縱向比較，驗證更大規模能否讓 Ettin 反超 MB-large。
- **Label Smoothing / Focal Loss**：本實驗為控制變數未使用，預期對稀有類別有額外提升（+0.01 ~ 0.02 MAP@3）。
- **Retrieval-augmented**：理論上 +0.03 ~ 0.05 MAP@3，但實作成本較大。

---

## 參考文獻

- ModernBERT — Answer.AI / HuggingFace, Dec 2024
- Ettin — Weller et al., 2025, arXiv:2507.11412
