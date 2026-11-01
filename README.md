# GenAI Assignment 2 — Clinical Trial Eligibility Screening

Implementation of a Transformer-based system for matching patient descriptions to clinical trial inclusion/exclusion criteria with evidence-based justifications.

## Overview

This project implements the complete pipeline from **Task 1** through **Task 6** of the Generative AI Assignment #2:

- **Task 1**: Research, model selection, and QLoRA fine-tuning on TrialGPT Criterion Annotations
- **Task 2**: Criterion-level eligibility prediction + evidence sentence retrieval
- **Task 3**: KV-cached autoregressive generation with correctness benchmarks
- **Task 4**: Tiled attention with online softmax implementation
- **Task 5**: Speculative decoding with draft model
- **Task 6**: Browser-based screening application

## Hardware

- **GPU**: NVIDIA RTX 4050 (6 GB VRAM)
- **Quantization**: 4-bit (Q4_K_M / NF4) for inference and QLoRA training

## Model Selection

| Component | Model | Parameters | Quantization | VRAM (est.) |
|-----------|-------|------------|--------------|-------------|
| **Target** | Qwen3-4B-Instruct | 4.02B | Q4_K_M | ~2.7 GB |
| **Draft** | Qwen3-0.6B | 0.6B | Q4_K_M | ~0.3 GB |
| **Fine-tuning** | QLoRA (NF4, r=128) | — | 4-bit base + FP16 adapters | ~3.8–4.2 GB |

Both models share the **identical Qwen3 tokenizer** — enabling zero-overhead speculative decoding.

## Dataset

- **Source**: `ncbi/TrialGPT-Criterion-Annotations` (Hugging Face)
- **Size**: 1,015 criterion-level annotations across 53 patients
- **Split**: Patient-grouped, seed 42 → **37 train / 8 val / 8 test**
- **Labels**: 6 expert categories (included, not included, not excluded, excluded, not enough information, not applicable)
- **Evidence**: Expert-annotated sentence indices from patient notes

## Quick Start

```bash
# Install dependencies (uv)
uv sync

# Download & preprocess dataset
uv run python data/download_dataset.py
uv run python data/preprocess.py

# Verify splits
uv run python data/dataset_interface.py
```

## Project Structure

```
GenAI A2/
├── .venv/                      # uv virtual environment (gitignored)
├── data/
│   ├── download_dataset.py     # Download from HF Hub
│   ├── preprocess.py           # Column removal + 37/8/8 split + manifest
│   ├── dataset_interface.py    # Clean split accessor for tasks
│   ├── raw/                    # Raw dataset cache
│   └── processed/              # Processed splits + split_manifest.json
├── task1/
│   ├── research/
│   │   ├── Literature review.md
│   │   └── Model selection.md
│   ├── preprocessing/
│   │   └── tokenizer.py        # Tokenization & data prep (used by train.py)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.yaml         # All hyperparameters (single source of truth)
│   │   ├── requirements.txt    # Exact Colab environment
│   │   ├── train.py            # QLoRA fine-tuning (reproducible)
│   │   └── fine_tune.ipynb     # Original Colab notebook
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Shared metrics (classification + evidence)
│   │   └── eval.py             # Evaluation script
│   └── outputs/
│       ├── adapter/            # LoRA adapter (configs tracked, weights on HF Hub)
│       ├── train/              # Training artifacts
│       │   ├── split_manifest.json
│       │   └── training_log.json
│       └── eval/               # Evaluation results
│           ├── README.md       # Results summary
│           └── test_predictions.json
├── task2/                      # Eligibility + evidence (TODO)
├── task3/                      # KV cache benchmarks (TODO)
├── task4/                      # Tiled attention (TODO)
├── task5/                      # Speculative decoding (TODO)
├── task6/                      # Browser app (TODO)
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## Task 1: QLoRA Fine-tuning & Evaluation

### Reproducible Training
```bash
# 1. Install environment (from Colab)
pip install -r task1/training/requirements.txt

# 2. Prepare tokenized data (uses task1/preprocessing/tokenizer.py)
uv run python task1/preprocessing/tokenizer.py

# 3. Train (reads config from task1/training/config.yaml)
uv run python task1/training/train.py
```
Outputs: `task1/outputs/adapter/adapter_model.safetensors`, `adapter_config.json`, `training_log.json` in `task1/outputs/train/`

### Evaluation
```bash
# Evaluate on test set (requires adapter from training or HF Hub)
uv run python task1/eval/eval.py \
  --adapter-path task1/outputs/adapter \
  --split test \
  --output-dir results/task1/eval
```
Outputs: `test_predictions.json`, `test_metrics.json` with accuracy, macro-F1, per-class P/R/F1, confusion matrix, evidence micro-F1.

### Configuration
All hyperparameters in `task1/training/config.yaml`:
- **Model**: Qwen2.5-3B-Instruct, 2048 seq len
- **QLoRA**: 4-bit NF4, r=128, alpha=256, dropout=0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training**: 3 epochs, batch=1, grad_accum=16, LR=5e-5, cosine, warmup=50
- **Data**: Patient-grouped 37/8/8 split (seed 42), 6 eligibility labels preserved

### Results (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | 0.6782 |
| Macro F1 | 0.4701 |
| Evidence F1 (micro) | 0.4996 |
| Trainable params | ~239M (7.2%) |
| Adapter size | 0.48 GB |

Full metrics in `task1/outputs/eval/README.md` and `test_predictions.json`.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Qwen3-4B + Qwen3-0.6B** | Identical tokenizer; both fit on 6 GB GPU simultaneously; Apache 2.0 |
| **QLoRA (r=128)** | Only method fitting 6 GB; matches LoRA within 2–4 pp on clinical IE (Yim 2024; Estupiñán-Ojeda 2025) |
| **Patient-grouped split** | Prevents leakage; matches assignment spec (seed 42, 37/8/8) |
| **4-class eligibility** | Preserves all expert categories — no binary collapse |
| **Evidence sentence IDs** | Direct span extraction from numbered patient note sentences |

## References

- **TrialGPT**: Jin et al., *Nature Communications* (2024) — base framework
- **QLoRA clinical IE**: Yim et al., medRxiv (2024); Estupiñán-Ojeda et al., JAMIA Open (2025)
- **Small model medical reasoning**: Subaharan (2025); Hariprasad (2026); MedMobile (2024)
- **FlashAttention**: Dao et al., NeurIPS (2022/2023)
- **Speculative Decoding**: Leviathan et al., ICML (2023)

## License

Apache 2.0 (model weights); MIT (code) — see individual model licenses.