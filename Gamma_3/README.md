# Math Misconception Classifier — Gemma 3 1B QLoRA Fine-tuning

Fine-tunes **Google Gemma-3-1B-IT** with QLoRA (4-bit NF4) + LoRA adapters to classify student math misconceptions into `Category:Misconception` labels.

---

## Hardware Requirements

| Item | Minimum |
|------|---------|
| GPU | NVIDIA GPU with **CUDA 13.x** support |
| VRAM | ≥ 8 GB (fp16 mode) / ≥ 10 GB (bf16 recommended) |
| Disk | ≥ 10 GB for model weights and checkpoints |

---

## Environment Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch==2.12.0+cu132` is pulled from the PyTorch CUDA 13.2 index declared in `requirements.txt`.  
> If your CUDA version differs, replace `cu132` with the matching suffix (e.g. `cu121`).

### 3. Hugging Face authentication

Gemma 3 is a gated model. Log in before running:

```bash
huggingface-cli login
# paste the token from huggingface_key.txt when prompted
```

---

## Project Structure

```
nlp_final_project/
├── Gemma_3.py                          # Main training & evaluation script
├── train.csv                           # Training data (required)
├── test.csv                            # Test data
├── sample_submission.csv               # Submission template
├── requirements.txt                    # Python dependencies
├── best_gemma_lora_model/              # Best LoRA checkpoint (saved after training)
├── gemma_math_misunderstanding_results/# All Trainer checkpoints
├── loss_curve.png                      # Training / eval loss plot
└── confusion_matrix.png                # Evaluation confusion matrix
```

---

## Data Format

`train.csv` must contain the following columns:

| Column | Description |
|--------|-------------|
| `QuestionText` | Math question text |
| `MC_Answer` | The correct answer |
| `StudentExplanation` | Student's written explanation |
| `Category` | Ground-truth misconception category |
| `Misconception` | Specific misconception name (or blank for NA) |

---

## Running the Script

```bash
python Gemma_3.py
```

The script will:

1. Load and split `train.csv` (90 / 10 stratified split)
2. Load `google/gemma-3-1b-it` in 4-bit NF4 quantization
3. Attach LoRA adapters (r=32, alpha=64) to all attention + MLP projection layers
4. Fine-tune for up to 3 epochs with early stopping (patience = 3)
5. Save the best LoRA weights to `./best_gemma_lora_model/`
6. Evaluate on 100 validation samples and print a classification report
7. Save `loss_curve.png` and `confusion_matrix.png`

### Resume from checkpoint

If training is interrupted, re-run the same command. The script automatically detects the latest checkpoint in `gemma_math_misunderstanding_results/` and resumes from it.

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-3-1b-it` |
| Quantization | 4-bit NF4 (double quant) |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-4 |
| Batch size | 2 (× 4 gradient accumulation = effective 8) |
| Max sequence length | 1024 |
| Epochs | 3 (with early stopping) |
| Loss type | chunked NLL (memory-efficient) |

---

## Output Files

| File | Description |
|------|-------------|
| `best_gemma_lora_model/` | LoRA adapter weights + tokenizer |
| `gemma_math_misunderstanding_results/` | All training checkpoints |
| `loss_curve.png` | Train vs. eval loss over steps |
| `confusion_matrix.png` | Predicted vs. true label heatmap |
