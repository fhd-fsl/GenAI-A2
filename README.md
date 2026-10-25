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
│   └── train.py                # QLoRA fine-tuning (TODO)
├── task2/                      # Eligibility + evidence (TODO)
├── task3/                      # KV cache benchmarks (TODO)
├── task4/                      # Tiled attention (TODO)
├── task5/                      # Speculative decoding (TODO)
├── task6/                      # Browser app (TODO)
├── pyproject.toml
├── uv.lock
└── .gitignore
```

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