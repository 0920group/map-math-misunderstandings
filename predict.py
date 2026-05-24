import os
import sys
import json
import argparse

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_data, make_test_dataset
from model import MODEL_CONFIGS


def predict(model_name, data_dir):
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        model_name,
    )
    best_model_dir = os.path.join(output_dir, "best_model")

    print(f"\n[predict] Loading model from {best_model_dir}")

    # Load label map
    with open(os.path.join(output_dir, "label2id.json")) as f:
        label2id = json.load(f)
    id2label = {int(i): l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_dir,
        dtype=torch.bfloat16,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load test data
    _, test_df, _, _ = load_data(data_dir)
    test_ds = make_test_dataset(test_df, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(test_ds, batch_size=MODEL_CONFIGS[model_name]["eval_batch_size"], collate_fn=collator)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    top3_indices = np.argsort(logits, axis=1)[:, -3:][:, ::-1]

    predictions = []
    for row_indices in top3_indices:
        labels = [id2label[idx] for idx in row_indices]
        predictions.append(" ".join(labels))

    submission = pd.DataFrame({
        "row_id": test_df["row_id"].values,
        "Category:Misconception": predictions,
    })

    out_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"[predict] Saved submission to {out_path} ({len(submission)} rows)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    args = parser.parse_args()

    predict(args.model_name, args.data_dir)
