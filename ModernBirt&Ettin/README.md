# ModernBERT & Ettin Encoder - Python Scripts

This README explains how to use the `.py` files in this folder.

If you want the notebook version, read [`IPYNB_README.md`](./IPYNB_README.md) instead.

## What this folder contains

- `dataset.py`: shared data loading and tokenization helpers
- `model.py`: model and tokenizer loading
- `evaluate.py`: MAP@3 metric helper
- `train.py`: training entry point
- `predict.py`: inference and submission generation
- `make_comparison.py`: validation score summary
- `ModernBERT_&_Ettin_Encoder.ipynb`: notebook version of the pipeline

## What each script does

### `dataset.py`

Shared data utilities used by training and prediction.

Functions:

- `make_text(row)`: formats one competition row into `Question / Answer / Explanation`
- `load_data(data_dir)`: loads `train.csv` and `test.csv`, builds label mappings
- `split_data(train_df, val_ratio=0.1, seed=42)`: stratified train/validation split
- `make_hf_dataset(df, tokenizer, max_length=64)`: tokenizes the training data
- `make_test_dataset(df, tokenizer, max_length=64)`: tokenizes the test data

Expected data files:

- `train.csv`
- `test.csv`

Required columns:

- `QuestionText`
- `MC_Answer`
- `StudentExplanation`
- `Category`
- `Misconception`

### `model.py`

Model presets and loading helpers.

Supported model names:

- `modernbert_base`
- `modernbert_large`
- `ettin_400m`

Main functions:

- `load_tokenizer(model_name)`
- `load_model(model_name, num_labels, label2id, id2label)`

Behavior:

- ModernBERT models use mean pooling when the config supports it.
- The loader tries `flash_attention_2`, then `sdpa`, then the default backend.
- Model dtype is set to `torch.bfloat16`.

### `evaluate.py`

Metric helper.

Function:

- `compute_map3(eval_pred)`: computes MAP@3 from logits and labels

This file is imported by `train.py` and usually does not need to be run by itself.

### `train.py`

Training entry point for the encoder models.

Usage:

```bash
python train.py modernbert_base
python train.py modernbert_large
python train.py ettin_400m
```

Optional arguments:

- `--data_dir`: path that contains `train.csv` and `test.csv`
- `--epochs`: number of training epochs, default `5`

Example:

```bash
python train.py modernbert_large --data_dir .. --epochs 5
```

What it does:

1. Loads the data
2. Splits it into train and validation sets
3. Builds class weights for imbalanced labels
4. Tokenizes the dataset
5. Loads the selected encoder backbone
6. Trains with weighted cross entropy
7. Saves the best checkpoint and training artifacts

Outputs:

- `results/<model_name>/best_model/`
- `results/<model_name>/label2id.json`
- `results/<model_name>/val_map3.txt`
- `results/<model_name>/training_curve.png`

### `predict.py`

Inference and submission generation.

Usage:

```bash
python predict.py modernbert_base
python predict.py modernbert_large
python predict.py ettin_400m
```

Optional arguments:

- `--data_dir`: path that contains `train.csv` and `test.csv`

Example:

```bash
python predict.py modernbert_large --data_dir ..
```

What it does:

1. Loads the saved checkpoint from `results/<model_name>/best_model/`
2. Loads `test.csv`
3. Runs batched inference
4. Writes the submission file

Output:

- `results/<model_name>/submission.csv`

### `make_comparison.py`

Validation summary generator.

Usage:

```bash
python make_comparison.py
```

What it does:

- Reads `results/<model_name>/val_map3.txt`
- Sorts the models by validation MAP@3
- Writes `results/comparison.md`

## Setup

Install the required packages manually:

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib tqdm
```

If you are using GPU training, make sure the PyTorch build matches your CUDA environment.

## Data layout

Put the competition files in the repository root or pass `--data_dir`:

```text
map-math-misunderstandings/
  train.csv
  test.csv
  sample_submission.csv
  ModernBirt&Ettin/
```

## Recommended order

1. Read `IPYNB_README.md` if you want the notebook workflow and hardware configuration.
2. Train one model with `train.py`.
3. Inspect `val_map3.txt`.
4. Generate the submission with `predict.py`.
5. Run `make_comparison.py` if you trained multiple models.

## Outputs

- Training checkpoint: `results/<model_name>/best_model/`
- Label map: `results/<model_name>/label2id.json`
- Validation score: `results/<model_name>/val_map3.txt`
- Training curve: `results/<model_name>/training_curve.png`
- Submission file: `results/<model_name>/submission.csv`
- Comparison report: `results/comparison.md`

## Notes

- This README is only for the Python scripts.
- Hardware configuration and notebook-specific instructions are in [`IPYNB_README.md`](./IPYNB_README.md).
- The notebook file `ModernBERT_&_Ettin_Encoder.ipynb` is the interactive version of the same workflow.
