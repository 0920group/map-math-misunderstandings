import os
import json

MODELS = ["modernbert_base", "modernbert_large", "ettin_400m"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    rows = []
    for model_name in MODELS:
        val_path = os.path.join(RESULTS_DIR, model_name, "val_map3.txt")
        if not os.path.exists(val_path):
            rows.append((model_name, "N/A"))
            continue
        with open(val_path) as f:
            line = f.read().strip()
        score = line.split(":")[-1].strip()
        rows.append((model_name, score))

    # Sort by score descending
    def sort_key(r):
        try:
            return float(r[1])
        except ValueError:
            return -1.0

    rows.sort(key=sort_key, reverse=True)

    lines = [
        "# Model Comparison — MAP@3 Results\n",
        "| Model | Val MAP@3 |",
        "|-------|-----------|",
    ]
    for model_name, score in rows:
        lines.append(f"| {model_name} | {score} |")

    lines += [
        "",
        "## Notes",
        "- Trained with weighted CrossEntropyLoss (class-balanced)",
        "- 90/10 train/val split stratified by label",
        "- 5 epochs, lr=5e-5, warmup_ratio=0.1, bf16",
        "- Attention: flash_attention_2 if available, else sdpa",
        "- Early stopping patience=2",
    ]

    out_path = os.path.join(RESULTS_DIR, "comparison.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved comparison to {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
