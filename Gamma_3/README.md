# Gamma_3 Math Misconception Pipeline

This folder contains the training, inference, and ensemble workflows for the math misconception classification task.

## What is included

- `Gemma_3.py`: main training and validation script
- `inference_version1.ipynb` ~ `inference_version10.ipynb`: Kaggle inference notebooks
- `essemble_kubeflow_qlora_train.ipynb`: multi-model QLoRA training and RRF ensemble notebook
- `requirements.txt` and `requirement.txt`: Python packages referenced by the code in this folder

## Python packages

The packages used by the code in `Gamma_3` are listed in [`requirements.txt`](./requirements.txt) and [`requirement.txt`](./requirement.txt).

Included packages:

- `torch`
- `transformers`
- `peft`
- `datasets`
- `trl`
- `accelerate`
- `bitsandbytes`
- `huggingface_hub`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `seaborn`
- `sentencepiece`

Standard-library modules such as `os`, `gc`, `pathlib`, `collections`, `math`, `re`, and `time` are not listed because they ship with Python.

## Installation

If you are running locally or in an environment without these packages preinstalled, install them with:

```bash
pip install -r requirements.txt
```

If you prefer the singular file name, `pip install -r requirement.txt` works too because it contains the same package list.

If you are running on Kaggle or Kubeflow, the notebook image/runtime may already provide part of the stack, so only install missing packages if the runtime does not already include them.

## Execution flow

### 1. Train or fine-tune the base model

Run:

```bash
python Gemma_3.py
```

This script:

1. Loads `train.csv`
2. Builds `Category:Misconception` labels
3. Splits the data into train/validation
4. Loads `google/gemma-3-1b-it` with 4-bit QLoRA
5. Trains the model with LoRA adapters
6. Saves the best adapter to `./best_gemma_lora_model/`
7. Saves checkpoints to `./gemma_math_misunderstanding_results/`
8. Produces `loss_curve.png` and `confusion_matrix.png`

### 2. Inference on Kaggle Notebook

The inference notebooks are designed to run on **Kaggle Notebook with T4 GPU**.

Important notes:

- The execution environment is the Kaggle notebook runtime.
- The GPU target is T4.
- The package versions used for inference should follow the versions provided by Kaggle's notebook environment.
- The notebooks read inputs from `/kaggle/input/...` and write outputs to `/kaggle/working/...`.
- The model and adapter paths in the notebooks are already written for Kaggle mount points.

Recommended inference steps:

1. Open one of the `inference_version*.ipynb` notebooks in Kaggle.
2. Confirm the base model path and adapter path.
3. Run the cells in order.
4. Generate `submission.csv` in `/kaggle/working/`.

### 3. Ensemble on Kubeflow

The ensemble workflow is implemented in `essemble_kubeflow_qlora_train.ipynb`.

Important notes:

- The ensemble section is intended to run with the **Kubeflow-side package versions**.
- Use the Kubeflow notebook or pipeline image that matches the package versions required by this ensemble workflow.
- The notebook sequentially fine-tunes multiple base models, saves their adapters, then runs Reciprocal Rank Fusion (RRF) for final prediction fusion.
- When you move this notebook into Kubeflow, keep the package versions aligned with the Kubeflow environment rather than the Kaggle inference runtime.

Recommended ensemble steps:

1. Prepare `train.csv` in the working directory.
2. Set `HF_TOKEN` if the base models require authentication.
3. Run the training cells for each model in `MODEL_LIST`.
4. Save the adapters under `FINAL_WEIGHTS_ROOT`.
5. Run the inference and RRF fusion cells.
6. Review the final MAP@3 score on the holdout set.

## Folder conventions

- Training outputs: `best_gemma_lora_model/`
- Checkpoints: `gemma_math_misunderstanding_results/`
- Kaggle inference outputs: `/kaggle/working/submission.csv`
- Kubeflow ensemble outputs: `./kubeflow_artifacts/`

## Data format

`train.csv` must contain these columns:

- `QuestionText`
- `MC_Answer`
- `StudentExplanation`
- `Category`
- `Misconception`

## Notes

- `Gemma_3.py` is the most direct entry point for local training and validation.
- The Kaggle inference notebooks are focused on generating predictions only.
- The Kubeflow ensemble notebook is intended for multi-model training plus RRF fusion.
