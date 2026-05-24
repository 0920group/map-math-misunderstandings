import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset


def make_text(row):
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Explanation: {row['StudentExplanation']}"
    )


def load_data(data_dir):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_df["label_str"] = train_df["Category"] + ":" + train_df["Misconception"].fillna("NA")
    train_df["text"] = train_df.apply(make_text, axis=1)
    test_df["text"] = test_df.apply(make_text, axis=1)

    all_labels = sorted(train_df["label_str"].unique().tolist())
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}

    train_df["label"] = train_df["label_str"].map(label2id)

    return train_df, test_df, label2id, id2label


def split_data(train_df, val_ratio=0.1, seed=42):
    from collections import Counter
    counts = Counter(train_df["label"].tolist())
    # Singleton labels must stay in train
    singleton_mask = train_df["label"].map(lambda l: counts[l] < 2)
    singletons = train_df[singleton_mask]
    splittable = train_df[~singleton_mask]

    train_split, val_split = train_test_split(
        splittable,
        test_size=val_ratio,
        stratify=splittable["label"],
        random_state=seed,
    )
    train_split = pd.concat([train_split, singletons], ignore_index=True)
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)


def make_hf_dataset(df, tokenizer, max_length=64):
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": df["label"].tolist()})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds


def make_test_dataset(df, tokenizer, max_length=64):
    ds = Dataset.from_dict({"text": df["text"].tolist()})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds
