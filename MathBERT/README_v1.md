# MathBERT Notebook Guide

This file explains how to use the notebook workflow in this folder, especially `MathBERT.ipynb`.

For the exact hardware configuration and package version notes, follow the existing [`README.md`](./README.md). I am not changing that file here.

## Notebook Files

- `MathBERT.ipynb`: main notebook workflow for training and inference
- `mathbert-map-v2.ipynb`: alternate / earlier notebook variant for the same project

## What the Notebook Does

The notebook is designed for the Kaggle competition `MAP - Charting Student Math Misunderstandings`.

Typical workflow:

1. Load the competition data.
2. Build the `Category:Misconception` label.
3. Split the data with stratified K-fold validation.
4. Train MathBERT with the selected configuration.
5. Evaluate with MAP@3.
6. Build the final ensemble submission.

## Input Format

The notebook expects the competition files to follow the same schema described in [`README.md`](./README.md):

- `QuestionText`
- `MC_Answer`
- `StudentExplanation`
- `Category`
- `Misconception`

The input text used by the notebook is typically formed as:

```text
Question: {QuestionText} [SEP] Answer: {MC_Answer} [SEP] Explanation: {StudentExplanation}
```

## How to Run `MathBERT.ipynb`

1. Open `MathBERT.ipynb` in Kaggle Notebook.
2. Make sure the competition data is attached as input.
3. Check that the notebook paths point to the Kaggle input folders.
4. Confirm the hardware setting matches the guidance in [`README.md`](./README.md).
5. Run the cells from top to bottom.
6. After training finishes, run the ensemble / submission cells to generate `submission.csv`.

## Hardware Notes

Use the hardware configuration described in [`README.md`](./README.md).

In short:

- Run this notebook in Kaggle Notebook.
- Use a GPU accelerator.
- Keep the notebook settings aligned with the environment described in the main README.

## Package Notes

Use the package versions described in [`README.md`](./README.md).

If you want to freeze the runtime, you can record the environment from inside the notebook after installation.

## Output Files

The notebook workflow usually produces:

- model checkpoints
- validation logs
- ensemble prediction files
- `submission.csv`

## Practical Tips

- If you are running only inference, skip the training cells and go straight to the prediction / ensemble cells.
- If you are retraining, keep the same split and seed settings as the notebook.
- If a cell depends on a saved fold model, make sure that checkpoint exists before running the ensemble step.

## Relation to `README.md`

`README.md` remains the main folder documentation.

This `README_v1.md` is only a notebook-oriented guide so that the notebook usage is separated from the general folder overview.
