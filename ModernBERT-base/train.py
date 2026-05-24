import os
import sys
import json
import argparse
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

# Allow importing from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_data, split_data, make_hf_dataset
from model import MODEL_CONFIGS, load_tokenizer, load_model
from evaluate import compute_map3


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "")


def compute_class_weights(labels, num_labels):
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_labels):
        count = counts.get(i, 1)
        weights.append(total / (num_labels * count))
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()  # cast to float32 for loss stability
        weights = self.class_weights.to(device=logits.device, dtype=torch.float32)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def plot_training_curve(log_history, output_path):
    train_steps, train_losses = [], []
    eval_steps, eval_map3 = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_map3" in entry:
            eval_steps.append(entry["step"])
            eval_map3.append(entry["eval_map3"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_steps, train_losses, label="train loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(eval_steps, eval_map3, marker="o", label="val MAP@3")
    ax2.set_xlabel("step")
    ax2.set_ylabel("MAP@3")
    ax2.set_title("Validation MAP@3")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def train(model_name, data_dir, num_epochs=5):
    cfg = MODEL_CONFIGS[model_name]
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        model_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Model ID: {cfg['model_id']}")
    print(f"Output:   {output_dir}")
    print(f"{'='*60}\n")

    # Load data
    print("[1/5] Loading data...")
    train_df, test_df, label2id, id2label = load_data(data_dir)
    num_labels = len(label2id)
    print(f"  Labels: {num_labels}, Train: {len(train_df)}")

    train_split, val_split = split_data(train_df)
    print(f"  Train split: {len(train_split)}, Val split: {len(val_split)}")

    # Save label maps
    with open(os.path.join(output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    # Load tokenizer and tokenize
    print("[2/5] Tokenizing...")
    tokenizer = load_tokenizer(model_name)
    train_ds = make_hf_dataset(train_split, tokenizer)
    val_ds = make_hf_dataset(val_split, tokenizer)

    # Load model
    print("[3/5] Loading model...")
    model = load_model(model_name, num_labels, label2id, id2label)

    # Class weights
    class_weights = compute_class_weights(
        train_split["label"].tolist(), num_labels
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=5e-5,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        dataloader_num_workers=0,  # 0 on Windows to avoid multiprocessing issues
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="map3",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    print("[4/5] Training...")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_map3,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save best model
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"  Best model saved to {best_model_dir}")

    # Save val MAP@3
    best_metric = trainer.state.best_metric
    print(f"\n  Best val MAP@3: {best_metric:.4f}")
    with open(os.path.join(output_dir, "val_map3.txt"), "w") as f:
        f.write(f"best_val_map3: {best_metric:.4f}\n")

    # Save training curve
    plot_training_curve(
        trainer.state.log_history,
        os.path.join(output_dir, "training_curve.png"),
    )
    print(f"  Training curve saved.")

    print(f"[5/5] Done: {model_name}")
    return best_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train(args.model_name, args.data_dir, args.epochs)
