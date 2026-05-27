import os

os.environ["BNB_CUDA_VERSION"] = "130"

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ==== 設定 ====
MODEL_ID         = "google/gemma-3-1b-it"
CSV_PATH         = "train.csv"
OUTPUT_DIR       = "./gemma_math_misunderstanding_results"
BEST_MODEL_PATH  = "./best_gemma_lora_model"
MAX_SEQ_LENGTH   = 1024
OVERSAMPLE_MIN   = 5    # [改進] 低於此數量的 label 會被過採樣


# ==== 工具函式 ====

def make_target_label(category, misconception):
    misconception_text = "NA" if pd.isna(misconception) else str(misconception).strip()
    return f"{str(category).strip()}:{misconception_text}"


def build_user_prompt(question, correct_answer, student_explanation):
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


def to_prompt_completion(example):
    user_prompt = build_user_prompt(
        example["QuestionText"],
        example["MC_Answer"],
        example["StudentExplanation"],
    )
    return {
        "prompt": (
            f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        ),
        "completion": f"{example['target_text']}<end_of_turn>",
    }


def get_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, e)
        for e in os.listdir(output_dir)
        if e.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, e))
    ]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None


# ==== [改進] MAP@3 計算 ====

def compute_map3(true_labels: list, pred_top3_list: list) -> float:
    """競賽指標 MAP@3。"""
    if not true_labels:
        return 0.0
    ap_sum = 0.0
    for true, preds in zip(true_labels, pred_top3_list):
        for k, pred in enumerate(preds[:3]):
            if pred == true:
                ap_sum += 1.0 / (k + 1)
                break
    return ap_sum / len(true_labels)


# ==== [改進] 類別過採樣 ====

def oversample_rare_labels(df: pd.DataFrame, min_samples: int = OVERSAMPLE_MIN) -> pd.DataFrame:
    """對訓練集中樣本數不足 min_samples 的 label 進行重複採樣。"""
    counts = df["target_text"].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    if not rare:
        print(f"  No rare labels (threshold={min_samples}).")
        return df

    extras = []
    for label in rare:
        subset = df[df["target_text"] == label]
        needed = min_samples - len(subset)
        extras.append(subset.sample(needed, replace=True, random_state=42))

    result = pd.concat([df] + extras, ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Oversampled {len(rare)} rare labels: {len(df)} → {len(result)} samples")
    return result


# ==== [改進] 序列長度統計 ====

def log_sequence_length_stats(df: pd.DataFrame, tokenizer, n_samples: int = 500):
    """印出 token 長度分布，協助驗證 MAX_SEQ_LENGTH 是否夠大。"""
    sample = df.head(n_samples)
    lengths = []
    for _, row in sample.iterrows():
        data = to_prompt_completion(row.to_dict())
        full = data["prompt"] + data["completion"]
        lengths.append(len(tokenizer(full, truncation=False)["input_ids"]))

    arr = np.array(lengths)
    truncated = (arr > MAX_SEQ_LENGTH).sum()
    print(
        f"Token length (n={len(arr)}):  "
        f"mean={arr.mean():.0f}  p95={np.percentile(arr, 95):.0f}  "
        f"max={arr.max():.0f}  "
        f"truncated={truncated} ({100 * truncated / len(arr):.1f}%)"
    )
    if truncated / len(arr) > 0.05:
        print(f"  [warn] >5% of samples are truncated — consider raising MAX_SEQ_LENGTH.")


# ==== [改進] MAP@3 Callback：取代 EarlyStoppingCallback ====

class MAP3EvalCallback(TrainerCallback):
    """
    每 eval_every_n 次 evaluation 做一次 beam-search 生成，計算 MAP@3。
    MAP@3 提升時自動存最佳 adapter 權重；連續 patience 次沒有改善則停止訓練。

    取代原本以 eval_loss 選模型的邏輯，直接對齊競賽指標。
    """

    def __init__(
        self,
        val_df: pd.DataFrame,
        tokenizer,
        best_model_path: str,
        eval_samples: int = 200,
        num_beams: int = 3,
        patience: int = 5,
        eval_every_n: int = 2,
    ):
        self.eval_df         = val_df.sample(min(eval_samples, len(val_df)), random_state=42)
        self.tokenizer       = tokenizer
        self.best_model_path = best_model_path
        self.num_beams       = num_beams
        self.patience        = patience
        self.eval_every_n    = eval_every_n
        self.best_map3       = -1.0
        self.no_improve      = 0
        self._call_count     = 0
        self._stop_ids       = None

    def _get_stop_ids(self):
        if self._stop_ids is None:
            stop_ids = [self.tokenizer.eos_token_id]
            eot = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if eot and eot != self.tokenizer.unk_token_id:
                stop_ids.append(eot)
            self._stop_ids = stop_ids
        return self._stop_ids

    def on_evaluate(self, args, state, control, model, **kwargs):
        self._call_count += 1
        if self._call_count % self.eval_every_n != 0:
            return control

        stop_ids = self._get_stop_ids()
        device = next(model.parameters()).device
        self.tokenizer.padding_side = "left"
        model.eval()

        true_labels, pred_top3_list = [], []

        with torch.no_grad():
            for _, row in self.eval_df.iterrows():
                prompt = (
                    f"<start_of_turn>user\n"
                    f"{build_user_prompt(row['QuestionText'], row['MC_Answer'], row['StudentExplanation'])}"
                    f"<end_of_turn>\n<start_of_turn>model\n"
                )
                inputs = self.tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=MAX_SEQ_LENGTH,
                ).to(device)

                out = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    eos_token_id=stop_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    use_cache=True,
                )

                prompt_len = inputs["input_ids"].shape[1]
                preds, seen = [], set()
                for i in range(self.num_beams):
                    text = self.tokenizer.decode(
                        out[i, prompt_len:], skip_special_tokens=True
                    ).strip()
                    label = text.splitlines()[0].strip() if text else ""
                    if label and label not in seen:
                        preds.append(label)
                        seen.add(label)

                true_labels.append(row["target_text"])
                pred_top3_list.append(preds[:3])

        map3 = compute_map3(true_labels, pred_top3_list)
        print(
            f"\n[MAP@3] step={state.global_step}  "
            f"MAP@3={map3:.4f}  best={self.best_map3:.4f}  "
            f"no_improve={self.no_improve}/{self.patience}"
        )

        if map3 > self.best_map3:
            self.best_map3 = map3
            self.no_improve = 0
            model.save_pretrained(self.best_model_path)
            self.tokenizer.save_pretrained(self.best_model_path)
            print(f"  → New best! Adapter saved to {self.best_model_path}")
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                print(f"  → patience={self.patience} reached. Stopping training.")
                control.should_training_stop = True

        # 還原訓練模式設定
        self.tokenizer.padding_side = "right"
        model.train()
        return control


# ==== 資料準備 ====

def load_and_split_data(csv_path):
    print("Loading train.csv ...")
    df = pd.read_csv(csv_path)

    required_columns = {
        "QuestionText", "MC_Answer", "StudentExplanation",
        "Category", "Misconception",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"train.csv 缺少欄位: {sorted(missing)}")

    df = df.copy()
    for col in ["QuestionText", "MC_Answer", "StudentExplanation"]:
        df[col] = df[col].fillna("")
    df["target_text"] = df.apply(
        lambda r: make_target_label(r["Category"], r["Misconception"]), axis=1,
    )

    combined = df["Category"].astype(str) + ":" + df["Misconception"].fillna("NA").astype(str)
    label_counts = combined.value_counts()
    split_strata = combined.where(combined.map(label_counts) >= 2, df["Category"])
    if (split_strata.value_counts() < 2).any():
        print("[warn] 有類別樣本不足 2 筆，改用隨機切分")
        split_strata = None

    df_train, df_val = train_test_split(
        df, test_size=0.1, random_state=42, stratify=split_strata,
    )
    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}")
    return df_train, df_val


# ==== 模型 / Tokenizer ====

def load_model_and_tokenizer():
    print("Loading tokenizer and model ...")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,   # [改進] 0.05 → 0.1，防止對多數類過擬合
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer, use_bf16


# ==== 訓練 ====

def train(model, tokenizer, train_dataset, val_dataset, use_bf16, df_val):
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,              # [改進] 加入 warmup，穩定初期訓練
        lr_scheduler_type="cosine",    # [改進] cosine decay，訓練後期更平滑
        logging_steps=16,
        num_train_epochs=5,            # [改進] 3 → 5，搭配 MAP@3 callback 提前停止
        eval_strategy="steps",
        eval_steps=120,
        save_steps=120,
        save_total_limit=3,
        load_best_model_at_end=False,  # [改進] 改由 MAP@3 callback 選最佳模型
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        loss_type="chunked_nll",
    )

    # [改進] 以 MAP@3 callback 取代 EarlyStoppingCallback
    map3_callback = MAP3EvalCallback(
        val_df=df_val,
        tokenizer=tokenizer,
        best_model_path=BEST_MODEL_PATH,
        eval_samples=200,   # 每次 eval 抽 200 筆計算 MAP@3
        num_beams=3,
        patience=5,         # [改進] 連續 5 次無改善才停（原本 3 次）
        eval_every_n=2,     # 每兩次 eval 才算一次，節省時間
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[map3_callback],
    )

    resume = get_latest_checkpoint(OUTPUT_DIR)
    if resume:
        print(f"Resuming from checkpoint: {resume}")

    print("Start training ...")
    trainer.train(resume_from_checkpoint=resume)

    print(f"\nBest MAP@3 during training: {map3_callback.best_map3:.4f}")
    print(f"Best adapter saved to     : {BEST_MODEL_PATH}")

    plot_loss_curve(trainer.state.log_history)
    return trainer


def plot_loss_curve(history):
    print("Plotting loss curves ...")
    train_loss = [x["loss"] for x in history if "loss" in x]
    eval_loss  = [x["eval_loss"] for x in history if "eval_loss" in x]

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    if eval_loss:
        step_gap = max(1, len(train_loss) // len(eval_loss))
        plt.plot(
            range(step_gap, step_gap * len(eval_loss) + 1, step_gap),
            eval_loss, label="Eval Loss", marker="o",
        )
    plt.title("Gemma 3 1B Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()
    print("Saved loss_curve.png")


# ==== [改進] 評估：beam search + 完整 val set + MAP@3 ====

def evaluate(model, tokenizer, df_val):
    """
    使用 beam search（與推論設定一致）對完整 val set 評估 MAP@3 及 Exact Match。
    """
    print(f"Running evaluation on full val set ({len(df_val)} samples) ...")  # [改進] 全集
    model.eval()
    tokenizer.padding_side = "left"

    stop_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot_id and eot_id != tokenizer.unk_token_id:
        stop_ids.append(eot_id)

    true_labels, pred_top1, pred_top3_list = [], [], []
    input_device = next(model.parameters()).device

    for _, row in df_val.iterrows():
        prompt = (
            f"<start_of_turn>user\n"
            f"{build_user_prompt(row['QuestionText'], row['MC_Answer'], row['StudentExplanation'])}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to(input_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,      # [改進] 64 → 32（與 inference 對齊）
                num_beams=3,            # [改進] 加入 beam search（與 inference 對齊）
                num_return_sequences=3,
                eos_token_id=stop_ids,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        preds, seen = [], set()
        for i in range(3):
            text = tokenizer.decode(
                outputs[i, prompt_len:], skip_special_tokens=True
            ).strip()
            label = text.splitlines()[0].strip() if text else "Unknown"
            if label not in seen:
                preds.append(label)
                seen.add(label)

        true_labels.append(row["target_text"])
        pred_top1.append(preds[0] if preds else "Unknown")
        pred_top3_list.append(preds[:3])

    map3       = compute_map3(true_labels, pred_top3_list)
    exact_acc  = sum(t == p for t, p in zip(true_labels, pred_top1)) / len(true_labels)

    print(f"\n{'='*40}")
    print(f"  MAP@3      : {map3:.4f}")
    print(f"  Exact Match: {exact_acc:.4f}")
    print(f"{'='*40}")

    plot_confusion_matrix(true_labels, pred_top1)
    print("\nClassification report (top-1):")
    all_classes = sorted(set(true_labels + pred_top1))
    print(classification_report(
        true_labels, pred_top1, target_names=all_classes, zero_division=0,
    ))

    tokenizer.padding_side = "right"
    return true_labels, pred_top1, map3


def plot_confusion_matrix(true_labels, pred_labels):
    all_classes = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_classes)

    n = len(all_classes)
    fig_size = max(12, min(n * 0.5, 30))
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm, annot=(n <= 25), fmt="d", cmap="Blues",
        xticklabels=all_classes, yticklabels=all_classes,
        annot_kws={"size": 8},
    )
    plt.title("Confusion Matrix - Math Misunderstanding")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")


# ==== 主流程 ====

def main():
    df_train, df_val = load_and_split_data(CSV_PATH)

    # [改進] 過採樣罕見 label
    print("\nOversampling rare labels ...")
    df_train = oversample_rare_labels(df_train, min_samples=OVERSAMPLE_MIN)

    train_dataset = Dataset.from_pandas(df_train, preserve_index=False).map(
        to_prompt_completion,
        remove_columns=df_train.columns.tolist(),
    )
    val_dataset = Dataset.from_pandas(df_val, preserve_index=False).map(
        to_prompt_completion,
        remove_columns=df_val.columns.tolist(),
    )

    model, tokenizer, use_bf16 = load_model_and_tokenizer()

    # [改進] 印出序列長度統計，確認 MAX_SEQ_LENGTH 設定是否合理
    print("\nChecking sequence lengths ...")
    log_sequence_length_stats(df_train, tokenizer)

    train(model, tokenizer, train_dataset, val_dataset, use_bf16, df_val)

    # 最終評估（當前 checkpoint，最佳 adapter 已存至 BEST_MODEL_PATH）
    print(f"\nNote: best MAP@3 adapter is at {BEST_MODEL_PATH}")
    print("Running final evaluation on current checkpoint ...")
    evaluate(model, tokenizer, df_val)
    print("Done.")


if __name__ == "__main__":
    main()
