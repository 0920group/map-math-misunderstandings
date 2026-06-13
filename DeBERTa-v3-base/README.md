# NLP Final Project - Team 3

本專案實作基於 DeBERTa-v3-base 的兩階段文字分類模型，用於參加 Kaggle 競賽：**MAP - Charting Student Math Misunderstandings**。專案共包含四個版本的 Jupyter Notebook 檔案，分別設計於 Google Colab 與 Kaggle Notebook 環境中運行。

## 檔案結構與版本說明

* **`DeBERTa_v3_base_1.ipynb`**：**[Google Colab 環境]** 最初建立的基礎驗證版本。此版本未採用五折交叉驗證，主要用於在 Colab 環境下調用 `kagglehub` API 快速跑通兩階段模型的端到端（End-to-End）常規訓練與推論流程。
* **`DeBERTa_v3_base_2.ipynb`**：**[Kaggle 環境]** 離線五折訓練版本。首次引入五折交叉驗證（5-Fold CV）機制，在 Kaggle 離線環境下對兩階段模型進行完整訓練，執行後共會產生 10 個 `.pt` 模型權重檔案（2 階段 × 5 折）。
* **`DeBERTa_v3_base_3.ipynb`**：**[Kaggle 環境]** 穩定度優化版本。架構延續五折交叉驗證與兩階段離線訓練（同樣產出 10 個 `.pt` 權重檔案），此版本核心在於成功解決並修正了 V2 版本中部分 Fold 模型訓練異常崩潰（Collapse）的問題，確保所有 Fold 皆能穩定收斂。
* **`DeBERTa_v3_base_4_final.ipynb`**：**[Kaggle 環境]** **（最終核心版本）** 本專案預測表現最佳的最終版本。基於 V3 的穩定優化架構，將兩階段模型的訓練輪數（Epoch）由 3 輪調整提升至 4 輪，進一步提升模型在迷思邏輯上的學習深度與最終推論效果。

## 環境與套件要求

### 1. 依賴套件

主要依賴 PyTorch 2.x 深度學習框架與 Hugging Face Transformers 庫。主要套件清單如下：

* `torch` (使用 PyTorch 2.0+ 新版 AMP 語法)
* `transformers` (Hugging Face)
* `pandas`
* `numpy`
* `scikit-learn`
* `scipy`
* `tqdm`
* `kagglehub` (僅限 Colab 版本 `DeBERTa_v3_base_1.ipynb` 運作需要)

### 2. 硬體需求

* **Google Colab**：建議選用 `T4 GPU`。
* **Kaggle Notebook**：建議設定 Accelerator 為 `GPU T4 x2` 或 `GPU P100`。

---

## 執行步驟

### 方案 A：執行 Google Colab 版本 (`DeBERTa_v3_base_1.ipynb`)

1. 將 `DeBERTa_v3_base_1.ipynb` 上傳至 Google Colab，並將執行階段類型變更為 `T4 GPU`。
2. 在第 2 個 Code Cell 中填入您個人的 Kaggle 帳戶憑證金鑰：
```python
os.environ["KAGGLE_USERNAME"] = "您的_KAGGLE_USERNAME"
os.environ["KAGGLE_KEY"]      = "您的_KAGGLE_KEY"

```


3. 點擊「全部執行 (Run All)」。程式會自動調用 API 下載競賽數據、完成 Category (6分類) 與 Misconception (35分類) 模型的訓練，並在最後自動下載產出的 `submission.csv`。

### 方案 B：執行 Kaggle Notebook 版本 (`DeBERTa_v3_base_3.ipynb` 等)

1. 在 Kaggle 平台上新建或匯入對應的 `.ipynb` 檔案。
2. 於右側 **Settings** 面板完成以下環境初始化配置：
* **Accelerator**：切換為 `GPU T4 x2` 或 `GPU P100`。
* **Internet**：切換為 `OFF`（關閉網路以符合競賽離線限制）。


3. 於右側 **Add Input** 欄位手動掛載以下兩項必要的競賽資源：
* **Competitions**：加入 `MAP - Charting Student Math Misunderstandings`
* **Models / Datasets**：搜尋並加入 `microsoft/deberta-v3-base` 離線預訓練模型（確保目錄下含有 `config.json` 與相關的分詞器檔案）。


4. 點擊 **Run All** 執行全部單元格。程式將自動掃描並讀取離線路徑、執行 5-Fold Ensemble 訓練與推論。最終會將最佳策略結果複製並產出為規格要求的 `/kaggle/working/submission.csv`。

---

## 提交與規範說明

1. **模型權重限制**：依據 Checkpoint 4 繳交規範，本專案壓縮檔（`.zip`）內**不包含**任何大型模型權重檔案 (`category_foldX.pt` 或 `miscon_foldX.pt`)。所有模型權重在執行時皆是動態由離線預訓練模型進行 Fine-tuning 訓練生成，或可手動上傳至 Hugging Face Hub 後調用。
2. **最終評分建議**：若助教需檢驗最完整的模型架構與最佳成績，請優先開啟並執行 **`DeBERTa_v3_base_4_final.ipynb`**。