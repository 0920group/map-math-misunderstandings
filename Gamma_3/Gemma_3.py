import os

# 強迫 bitsandbytes 使用內建相容的 CUDA 13.0 驅動
os.environ["BNB_CUDA_VERSION"] = "130"

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
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ==== 設定 ====
MODEL_ID = "google/gemma-3-1b-it"
CSV_PATH = "train.csv"
OUTPUT_DIR = "./gemma_math_misunderstanding_results"
BEST_MODEL_PATH = "./best_gemma_lora_model"
MAX_SEQ_LENGTH = 1024
EVAL_ROWS = 100


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
    """trl v1 的 prompt-completion 格式,loss 只算在 completion 段"""
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
    # 罕見組合先 fallback 到 Category
    split_strata = combined.where(combined.map(label_counts) >= 2, df["Category"])
    # 若連 Category 也有單筆,直接放棄 stratify
    if (split_strata.value_counts() < 2).any():
        print("[warn] 有類別樣本不足 2 筆,改用隨機切分")
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
    # Gemma 本身就有 <pad>,不要蓋成 eos_token
    tokenizer.padding_side = "right"  # 訓練用 right padding

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # gradient checkpointing 必需
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer, use_bf16


# ==== 訓練 ====
def train(model, tokenizer, train_dataset, val_dataset, use_bf16):
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,            # LoRA 慣用 1e-4 ~ 2e-4
        logging_steps=16,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=120,
        save_steps=120,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        loss_type="chunked_nll",  # 避免 [B, T, vocab] 全展開,大幅降低 entropy 計算的峰值記憶體
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01,
        )],
    )

    resume = get_latest_checkpoint(OUTPUT_DIR)
    if resume:
        print(f"Resuming from checkpoint: {resume}")

    print("Start training ...")
    trainer.train(resume_from_checkpoint=resume)

    print(f"Saving LoRA weights to: {BEST_MODEL_PATH}")
    trainer.model.save_pretrained(BEST_MODEL_PATH)
    tokenizer.save_pretrained(BEST_MODEL_PATH)

    plot_loss_curve(trainer.state.log_history)
    return trainer


def plot_loss_curve(history):
    print("Plotting loss curves ...")
    train_loss = [x["loss"] for x in history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]

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


# ==== 推論 / 評估 ====
def evaluate(model, tokenizer, df_val, n_samples=EVAL_ROWS):
    print(f"Running evaluation on {n_samples} samples ...")
    model.eval()

    # Gemma 回合結束符,加入 stop tokens
    stop_ids = [tokenizer.eos_token_id]
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id and end_of_turn_id != tokenizer.unk_token_id:
        stop_ids.append(end_of_turn_id)

    true_labels, pred_labels = [], []
    eval_sample = df_val.head(n_samples)
    input_device = next(model.parameters()).device

    for _, row in eval_sample.iterrows():
        prompt = (
            f"<start_of_turn>user\n"
            f"{build_user_prompt(row['QuestionText'], row['MC_Answer'], row['StudentExplanation'])}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,             # 給 misconception 足夠長度
                eos_token_id=stop_ids,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted_label = generated_text.splitlines()[0].strip() if generated_text else "Unknown"

        true_labels.append(row["target_text"])
        pred_labels.append(predicted_label)

    plot_confusion_matrix(true_labels, pred_labels)
    print("\nClassification report:")
    all_classes = sorted(set(true_labels + pred_labels))
    print(classification_report(
        true_labels, pred_labels, target_names=all_classes, zero_division=0,
    ))
    return true_labels, pred_labels


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
    train_dataset = Dataset.from_pandas(df_train, preserve_index=False).map(
        to_prompt_completion,
        remove_columns=df_train.columns.tolist(),
    )
    val_dataset = Dataset.from_pandas(df_val, preserve_index=False).map(
        to_prompt_completion,
        remove_columns=df_val.columns.tolist(),
    )

    model, tokenizer, use_bf16 = load_model_and_tokenizer()
    train(model, tokenizer, train_dataset, val_dataset, use_bf16)
    evaluate(model, tokenizer, df_val, n_samples=EVAL_ROWS)
    print("Done.")


if __name__ == "__main__":
    main()