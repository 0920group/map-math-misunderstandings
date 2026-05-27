"""
MAP - Charting Student Math Misunderstandings - Inference v7.0 (Improved RAG)
改進重點 (vs v6.0, 0.57 → higher):
1. BM25 改為索引完整文字 (QuestionText + MC_Answer + StudentExplanation)，提升召回品質。
2. 動態 3-Shot RAG Prompt：取 top-3 相似訓練樣本作為 few-shot context。
3. 生成 label 後接 Label Normalization：對不在已知集合內的輸出做最近鄰修正。
4. 檢索 top-K fallback 候選池：beam 候選不足時從 BM25 得票最高的 label 補齊。
5. MAX_NEW_TOKENS 放寬至 32，避免長 label 被截斷。
"""

import os
import re
import time
import math
import pandas as pd
import numpy as np
import torch
from collections import Counter
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ==== 路徑 ====
BASE_MODEL_PATH = "/kaggle/input/models/google/gemma-3/transformers/gemma-3-1b-it/1"
ADAPTER_PATH    = "/kaggle/input/datasets/alextsai2004/gemma-math-misunderstanding-lora/best_gemma_lora_model"
TEST_CSV        = "/kaggle/input/competitions/map-charting-student-math-misunderstandings/test.csv"
TRAIN_CSV       = "/kaggle/input/competitions/map-charting-student-math-misunderstandings/train.csv"
SAMPLE_CSV      = "/kaggle/input/competitions/map-charting-student-math-misunderstandings/sample_submission.csv"
OUTPUT_CSV      = "/kaggle/working/submission.csv"

# ==== 超參 ====
BATCH_SIZE      = 16    # Phase 1 每批處理筆數
SUB_BATCH_SIZE  = 32    # Phase 2 forward 子批上限
NUM_BEAMS       = 5     # beam width
NUM_RETURN      = 5     # 每筆生成幾個候選
MAX_NEW_TOKENS  = 32    # 放寬至 32，避免長 label 截斷
FEW_SHOT_K      = 3     # 放入 prompt 的 few-shot 範例數
BM25_TOP_K      = 10    # BM25 檢索數，前 FEW_SHOT_K 進 prompt，全部做 fallback 候選

# ==== Sample submission 骨架 ====
sample = pd.read_csv(SAMPLE_CSV)
ROW_ID_COL = sample.columns[0]
PRED_COL   = sample.columns[1]
print(f"Sample shape: {sample.shape}")


# ── BM25 ───────────────────────────────────────────────────────────────────

def tokenize_text(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r'\w+', text.lower())

class LightweightBM25:
    """純 Python 倒排索引 BM25，零外部依賴。"""
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.doc_lens    = [len(d) for d in docs]
        self.avg_doc_len = sum(self.doc_lens) / len(docs) if docs else 1
        self.N = len(docs)

        self.index: dict[str, list[tuple[int, int]]] = {}
        df: dict[str, int] = {}
        for doc_id, doc in enumerate(docs):
            counts = Counter(doc)
            for term, tf in counts.items():
                self.index.setdefault(term, []).append((doc_id, tf))
                df[term] = df.get(term, 0) + 1

        self.idf = {
            term: math.log(1 + (self.N - f + 0.5) / (f + 0.5))
            for term, f in df.items()
        }

    def retrieve_top_k(self, query_tokens: list[str], k: int = BM25_TOP_K) -> list[int]:
        """回傳 top-k 訓練集索引（降序排列）。"""
        scores: dict[int, float] = {}
        for term in set(query_tokens):
            if term not in self.index:
                continue
            idf = self.idf[term]
            for doc_id, tf in self.index[term]:
                num = idf * tf * (self.k1 + 1)
                den = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lens[doc_id] / self.avg_doc_len
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + num / den

        if not scores:
            return list(range(min(k, self.N)))
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        return ranked[:k]


# ── 資料準備 ───────────────────────────────────────────────────────────────

print("Loading train set & building BM25 index...")
train_df = pd.read_csv(TRAIN_CSV)
for col in ["QuestionText", "MC_Answer", "StudentExplanation"]:
    train_df[col] = train_df[col].fillna("")

train_df["target"] = (
    train_df["Category"].astype(str) + ":" +
    train_df["Misconception"].fillna("NA").astype(str)
)

unique_labels     = sorted(train_df["target"].unique().tolist())
unique_labels_set = set(unique_labels)
fallback_labels   = train_df["target"].value_counts().head(3).index.tolist()
if not fallback_labels:
    fallback_labels = ["NA:NA", "NA:NA", "NA:NA"]

# ── 改進 1：BM25 索引完整文字，而非只索引 StudentExplanation ──
train_df["full_text"] = (
    train_df["QuestionText"] + " " +
    train_df["MC_Answer"]    + " " +
    train_df["StudentExplanation"]
)
train_corpus    = [tokenize_text(t) for t in train_df["full_text"]]
bm25_retriever  = LightweightBM25(train_corpus)
print(f"BM25 index built. Training examples: {len(train_df)}")


# ── 改進 2：3-Shot RAG Prompt ──────────────────────────────────────────────

def build_user_prompt(question: str, correct_answer: str, student_explanation: str) -> str:
    return (
        "You are a math misconception classifier.\n"
        "Given the question, the correct answer, and the student's explanation, "
        "predict the final label in the format `Category:Misconception`.\n"
        "If the category is not a misconception type, use `NA` for the misconception part.\n\n"
        f"Question: {question}\n"
        f"Correct answer: {correct_answer}\n"
        f"Student explanation: {student_explanation}\n\n"
        "Return only the label."
    )

def build_dynamic_rag_prompt(row: pd.Series) -> tuple[str, list[str]]:
    """
    Returns (prompt_str, fallback_label_list).
    fallback_label_list = labels from BM25 top-K for candidate supplementing.
    """
    query_tokens = tokenize_text(
        row["QuestionText"] + " " + row["MC_Answer"] + " " + row["StudentExplanation"]
    )
    top_k_idx = bm25_retriever.retrieve_top_k(query_tokens, k=BM25_TOP_K)
    top_k_rows = [train_df.iloc[i] for i in top_k_idx]

    # Fallback candidates from retrieval (for beam 補充用)
    retrieval_labels = [r["target"] for r in top_k_rows]

    # Few-shot examples (top FEW_SHOT_K)
    turns = []
    for ex in top_k_rows[:FEW_SHOT_K]:
        ex_user = build_user_prompt(
            ex["QuestionText"], ex["MC_Answer"], ex["StudentExplanation"]
        )
        turns.append(
            f"<start_of_turn>user\n{ex_user}<end_of_turn>\n"
            f"<start_of_turn>model\n{ex['target']}<end_of_turn>\n"
        )

    current_user = build_user_prompt(
        row["QuestionText"], row["MC_Answer"], row["StudentExplanation"]
    )
    prompt = (
        "".join(turns)
        + f"<start_of_turn>user\n{current_user}<end_of_turn>\n"
        + "<start_of_turn>model\n"
    )
    return prompt, retrieval_labels


# ── 改進 3：Label Normalization ────────────────────────────────────────────

def normalize_label(raw: str) -> str:
    """
    若 raw 已在已知集合內直接返回；
    否則嘗試：
      a) 分拆 Category:Misconception 後按 Category 縮小搜索範圍再做字串相似度比對；
      b) 全域字串相似度最近鄰。
    """
    raw = raw.strip()
    if raw in unique_labels_set:
        return raw

    # a) 以 category 部分縮小搜索範圍
    parts = raw.split(":", 1)
    cat = parts[0].strip() if parts else ""
    category_subset = [l for l in unique_labels if l.startswith(cat + ":")]
    search_space = category_subset if category_subset else unique_labels

    best, best_score = search_space[0], -1.0
    for candidate in search_space:
        score = SequenceMatcher(None, raw, candidate).ratio()
        if score > best_score:
            best, best_score = candidate, score

    return best


# ── 模型載入 ───────────────────────────────────────────────────────────────

print("Loading base model in bf16 ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"  base loaded in {time.time()-t0:.1f}s")

t0 = time.time()
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print(f"  adapter loaded in {time.time()-t0:.1f}s")

print("Merging LoRA into base ...")
t0 = time.time()
model = model.merge_and_unload()
torch.cuda.empty_cache()
print(f"  merge done in {time.time()-t0:.1f}s")

model.eval()
model.config.use_cache = True
device = next(model.parameters()).device

pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = pad_id

end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
stop_ids = [tokenizer.eos_token_id]
if end_of_turn_id and end_of_turn_id != tokenizer.unk_token_id:
    stop_ids.append(end_of_turn_id)


# ── Test 讀取 ──────────────────────────────────────────────────────────────

test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)
for col in ["QuestionText", "MC_Answer", "StudentExplanation"]:
    test_df[col] = test_df[col].fillna("")
print(f"Test size: {len(test_df)}")
assert len(test_df) == len(sample)


# ── Phase 1: Beam Search ───────────────────────────────────────────────────

def clean_label(text: str) -> str:
    if not text:
        return ""
    label = text.splitlines()[0].strip()
    # 取第一個 token（label 不含空格）
    if " " in label:
        label = label.split(" ")[0]
    return label

@torch.no_grad()
def beam_generate_batch(prompts: list[str]) -> list[list[str]]:
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=1400,   # 稍加長以容納 3-shot
    ).to(device)

    outputs = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_RETURN,
        do_sample=False,
        early_stopping=True,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    prompt_len = enc["input_ids"].shape[1]
    outputs = outputs.view(len(prompts), NUM_RETURN, -1)

    results = []
    for i in range(len(prompts)):
        valid, seen = [], set()
        for k in range(NUM_RETURN):
            gen   = outputs[i, k, prompt_len:]
            text  = tokenizer.decode(gen, skip_special_tokens=True).strip()
            raw   = clean_label(text)
            # 改進 3：normalize before dedup
            label = normalize_label(raw) if raw else ""
            if label and label not in seen:
                valid.append(label)
                seen.add(label)
        results.append(valid)
    return results


# ── Phase 2: Log-likelihood Re-ranking ────────────────────────────────────

@torch.no_grad()
def score_candidates_batched(
    prompts: list[str],
    candidates_per_prompt: list[list[str]],
) -> list[list[float]]:
    """全程在 GPU 內計算，避免巨大 logits 搬移至 CPU。"""
    flat_sequences: list[list[int]] = []
    flat_meta: list[tuple[int, int, int]] = []   # (prompt_idx, cand_idx, label_len)

    for pi, (prompt, cands) in enumerate(zip(prompts, candidates_per_prompt)):
        if not cands:
            continue
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        for ci, cand in enumerate(cands):
            cand_ids = tokenizer.encode(cand, add_special_tokens=False)
            cand_ids.append(
                end_of_turn_id if end_of_turn_id else tokenizer.eos_token_id
            )
            flat_sequences.append(prompt_ids + cand_ids)
            flat_meta.append((pi, ci, len(cand_ids)))

    if not flat_sequences:
        return [[] for _ in prompts]

    B       = len(flat_sequences)
    max_len = max(len(s) for s in flat_sequences)

    input_ids      = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    label_starts   = []

    for j, seq in enumerate(flat_sequences):
        input_ids[j, :len(seq)]      = torch.tensor(seq, dtype=torch.long)
        attention_mask[j, :len(seq)] = 1
        label_starts.append(len(seq) - flat_meta[j][2])

    scores_per_prompt = [[0.0] * len(c) for c in candidates_per_prompt]

    for m in range(0, B, SUB_BATCH_SIZE):
        sub_ids   = input_ids[m:m + SUB_BATCH_SIZE].to(device)
        sub_mask  = attention_mask[m:m + SUB_BATCH_SIZE].to(device)
        sub_logits = model(input_ids=sub_ids, attention_mask=sub_mask).logits

        for idx, (pi, ci, L) in enumerate(flat_meta[m:m + SUB_BATCH_SIZE]):
            ls         = label_starts[m + idx]
            slice_lgt  = sub_logits[idx, ls - 1:ls - 1 + L, :].float()
            log_probs  = torch.log_softmax(slice_lgt, dim=-1)
            target_ids = flat_sequences[m + idx][-L:]
            target     = torch.tensor(target_ids, device=device)
            tok_lp     = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            scores_per_prompt[pi][ci] = tok_lp.mean().item()

    return scores_per_prompt


# ── 主推論迴圈 ─────────────────────────────────────────────────────────────

print("\nStart RAG-augmented inference ...")
pred_dict: dict = {}
phase1_time = phase2_time = 0.0
n_empty = total_cands = 0

try:
    for start in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Batch"):
        batch_df = test_df.iloc[start:start + BATCH_SIZE]

        # 改進 2：3-shot prompt + 記錄每筆的 BM25 fallback 候選
        prompts, batch_retrieval_labels = [], []
        for _, row in batch_df.iterrows():
            p, rl = build_dynamic_rag_prompt(row)
            prompts.append(p)
            batch_retrieval_labels.append(rl)

        # Phase 1: Beam Search
        t0 = time.time()
        candidates = beam_generate_batch(prompts)
        phase1_time += time.time() - t0

        # Phase 2: Log-likelihood Re-ranking
        t0 = time.time()
        scores = score_candidates_batched(prompts, candidates)
        phase2_time += time.time() - t0

        # 組合 Top-3
        for i, (_, row) in enumerate(batch_df.iterrows()):
            cands = candidates[i]
            total_cands += len(cands)

            if not cands:
                n_empty += 1
                top3 = list(fallback_labels[:3])
            else:
                ranked = sorted(zip(cands, scores[i]), key=lambda x: -x[1])
                top3 = [c for c, _ in ranked]

                # 改進 4：用 BM25 檢索 label 補足至 3 個
                vote: Counter = Counter(batch_retrieval_labels[i])
                for lbl, _ in vote.most_common():
                    if len(top3) >= 3:
                        break
                    if lbl not in top3:
                        top3.append(lbl)

                # 最終保底 fallback
                for fb in fallback_labels:
                    if len(top3) >= 3:
                        break
                    if fb not in top3:
                        top3.append(fb)

            while len(top3) < 3:
                top3.append(fallback_labels[0])

            pred_dict[row["row_id"]] = " ".join(top3[:3])

except Exception as e:
    print(f"\n[CRITICAL ERROR] {e}")
    print("Generating safe fallback submission ...")
    for _, row in test_df.iterrows():
        if row["row_id"] not in pred_dict:
            pred_dict[row["row_id"]] = " ".join(fallback_labels[:3])


# ── Diagnostic ─────────────────────────────────────────────────────────────

print(f"\nTiming:")
print(f"  Phase 1 (beam):    {phase1_time:.1f}s")
print(f"  Phase 2 (re-rank): {phase2_time:.1f}s")
print(f"  Avg candidates per row: {total_cands / len(test_df):.2f}")
print(f"  Rows with 0 valid beam candidates: {n_empty} "
      f"({n_empty / len(test_df) * 100:.1f}%)")


# ── 產出 Submission ─────────────────────────────────────────────────────────

submission = sample.copy()
submission[PRED_COL] = submission[ROW_ID_COL].map(pred_dict)

if submission[PRED_COL].isna().any():
    submission[PRED_COL] = submission[PRED_COL].fillna(" ".join(fallback_labels[:3]))

print("\nValidation:")
print(f"  Shape       : {submission.shape}")
print(f"  Any NaN     : {submission.isna().any().any()}")
print(f"  All 3 preds : {(submission[PRED_COL].str.split().str.len() == 3).all()}")

assert submission.shape == sample.shape
assert not submission.isna().any().any()
assert (submission[PRED_COL] != "").all()
assert (submission[PRED_COL].str.split().str.len() == 3).all()

submission.to_csv(OUTPUT_CSV, index=False)
print(f"\n[OK] Saved {OUTPUT_CSV}")
