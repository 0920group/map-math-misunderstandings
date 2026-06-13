# MAP - Charting Student Math Misunderstandings

This repository collects several experiments for the Kaggle competition **MAP - Charting Student Math Misunderstandings**.

The project is organized by model family and workflow. Each folder has its own README so you can jump directly to the implementation you need.

## Project Overview

- `DeBERTa-v3-base`: notebook-based DeBERTa experiments for training, cross-validation, and Kaggle submission generation
- `MathBERT`: notebook-based MathBERT experiments for stratified K-fold training and ensemble inference
- `ModernBirt&Ettin`: script-based encoder workflow with separate documentation for Python scripts and notebook usage
- `Gamma_3`: Gemma 3 QLoRA workflow with Kaggle inference notebooks and a Kubeflow-oriented ensemble pipeline

## Folder Guide

| Folder | Main Purpose | Recommended Entry Points |
|---|---|---|
| [`DeBERTa-v3-base`](./DeBERTa-v3-base) | DeBERTa-v3-base notebook experiments for cross-validation and final submission generation. | [`README.md`](./DeBERTa-v3-base/README.md), `DeBERTa_v3_base_1.ipynb`, `DeBERTa_v3_base_4_final.ipynb` |
| [`MathBERT`](./MathBERT) | MathBERT notebook experiments for K-fold training, validation, and ensemble prediction. | [`README.md`](./MathBERT/README.md), [`README_v1.md`](./MathBERT/README_v1.md), `MathBERT.ipynb`, `mathbert-map-v2.ipynb` |
| [`ModernBirt&Ettin`](./ModernBirt%26Ettin) | Encoder-based pipeline with reusable Python modules for data loading, model loading, training, prediction, and comparison. | [`README.md`](./ModernBirt&Ettin/README.md), [`IPYNB_README.md`](./ModernBirt&Ettin/IPYNB_README.md), `train.py`, `predict.py`, `make_comparison.py` |
| [`Gamma_3`](./Gamma_3) | Gemma 3 QLoRA training and inference workflow, including Kaggle notebook inference and Kubeflow ensemble training. | [`README.md`](./Gamma_3/README.md), `Gemma_3.py`, `essemble_kubeflow_qlora_train.ipynb`, `inference_version1.ipynb` |

## Suggested Reading Order

1. [`DeBERTa-v3-base/README.md`](./DeBERTa-v3-base/README.md)
2. [`MathBERT/README.md`](./MathBERT/README.md)
3. [`MathBERT/README_v1.md`](./MathBERT/README_v1.md)
4. [`ModernBirt&Ettin/README.md`](./ModernBirt&Ettin/README.md)
5. [`ModernBirt&Ettin/IPYNB_README.md`](./ModernBirt&Ettin/IPYNB_README.md)
6. [`Gamma_3/README.md`](./Gamma_3/README.md)

## Common Workflow

1. Prepare the competition data files.
2. Train or fine-tune the selected model family.
3. Run inference or cross-validation evaluation.
4. Combine fold outputs or model outputs if the workflow supports ensemble logic.
5. Export the final `submission.csv` for Kaggle.

## Documentation Layout

- Folder-level setup and usage details are documented in each local README.
- `MathBERT` uses `README_v1.md` for notebook-specific guidance.
- `ModernBirt&Ettin` splits script usage and notebook usage into separate documents.
- `Gamma_3` includes both training/inference instructions and environment-specific notes for Kaggle and Kubeflow.

## Dependency Notes

- Each folder may have its own package requirements and runtime assumptions.
- `Gamma_3` includes both [`requirements.txt`](./Gamma_3/requirements.txt) and [`requirement.txt`](./Gamma_3/requirement.txt).
- Notebook workflows may assume Kaggle or other managed GPU environments.
- Script-based workflows may require manual package installation depending on your local environment.

## Repository Notes

- Folder names are preserved as they were originally created.
- The repository is organized by model family rather than by a single shared training framework.
- If you are looking for a specific implementation detail, start with the relevant folder README and then open the notebook or script it references.
