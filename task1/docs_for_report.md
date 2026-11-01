# Task 1 Documentation for Report

This document consolidates all information needed for the Task 1 section of the IEEE technical report.

---

## 1. Literature Review (Rubric A1)

### 5+ Credible Sources Investigated

| # | Source | Topic | Key Findings |
|---|--------|-------|--------------|
| 1 | Jin et al., *TrialGPT: A Framework for Clinical Trial Eligibility Screening*, Nature Communications (2024) | Clinical eligibility reasoning with LLMs | Introduced criterion-level matching with evidence sentences; GPT-4 baseline; dataset creation |
| 2 | Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS (2023) | QLoRA fine-tuning | 4-bit NF4 quantization + LoRA; matches full fine-tuning; enables 3B+ models on 6GB GPU |
| 3 | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR (2022) | LoRA theory | Low-rank adapters freeze base model; rank r=8-128 typical; alpha=2r common |
| 4 | Dao et al., *FlashAttention-2: Faster Attention with Better Parallelism*, NeurIPS (2023) | Efficient attention | Tiled attention with online softmax; IO-aware; fused kernels |
| 5 | Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML (2023) | Speculative decoding | Draft-verify paradigm; small draft model + large target; greedy/beam acceptance |
| 6 | Yim et al., *Clinical Information Extraction with QLoRA*, medRxiv (2024) | Clinical IE with QLoRA | QLoRA within 2-4 pp of full fine-tuning on clinical tasks; validates our approach |
| 7 | Estupiñán-Ojeda et al., *Parameter-Efficient Fine-tuning for Clinical NLP*, JAMIA Open (2025) | PEFT for clinical NLP | LoRA/QLoRA effective for small clinical datasets; r=128 sufficient |

### Critical Comparison

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Full fine-tuning | Best performance | Needs >24GB VRAM | ❌ Not feasible |
| LoRA (FP16) | Good performance, fewer params | Still needs base model in FP16 | ❌ 6GB insufficient |
| **QLoRA (4-bit NF4)** | **Fits 6GB, matches LoRA** | Slight overhead from dequantization | ✅ **Selected** |
| Adapter/Prefix tuning | Very few params | Lower ceiling on complex tasks | ❌ Not enough capacity |
| Prompt tuning only | Minimal params | Insufficient for structured output | ❌ Not enough capacity |

---

## 2. Model Selection (Rubric A2)

### Candidate Models Evaluated

| Model | Params | Context | License | VRAM (QLoRA) | Tokenizer | Decision |
|-------|--------|---------|---------|--------------|-----------|----------|
| **Qwen2.5-3B-Instruct** | 3.09B | 131K | Apache 2.0 | ~3.8 GB | Qwen2Tokenizer | ✅ **Selected** |
| Qwen3-4B-Instruct | 4.02B | 131K | Apache 2.0 | ~4.5 GB | Qwen2Tokenizer | Backup |
| Llama-3.2-3B-Instruct | 3.2B | 128K | Llama 3.2 Community | ~3.8 GB | Llama3Tokenizer | ❌ License |
| Phi-3.5-mini-Instruct | 3.8B | 128K | MIT | ~4.2 GB | PhiTokenizer | ❌ Smaller context |
| Gemma-2-2B-IT | 2.6B | 8K | Gemma Terms | ~3.2 GB | GemmaTokenizer | ❌ Short context |

### Justification for Qwen2.5-3B-Instruct

- **Architecture**: Standard decoder-only Transformer (Qwen2), well-supported in HF/PEPT
- **Parameters**: 3.09B — fits QLoRA on 6GB GPU with room for gradients/activations
- **Context Window**: 131K tokens — ample for patient notes + criteria (max observed ~1K tokens)
- **License**: Apache 2.0 — permissive, allows commercial use, no restrictions
- **Tokenizer**: Qwen2Tokenizer (BPE, 151K vocab) — handles medical terminology well
- **Instruction Tuning**: Strong base instruction following, crucial for JSON output format
- **Hardware Fit**: Verified on Colab T4 (16GB) and local RTX 4050 (6GB) with QLoRA

### Compute Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | 6 GB VRAM minimum (T4, RTX 4050, or equivalent) |
| RAM | 16 GB system RAM |
| Storage | 10 GB (model + dataset + checkpoints) |
| Training Time | ~45 min on T4 (3 epochs, 603 train samples) |

---

## 3. Data Preprocessing & Tokenization (Rubric B2)

### Dataset
- **Source**: `ncbi/TrialGPT-Criterion-Annotations` (Hugging Face)
- **Size**: 1,015 criterion-level annotations across 53 patients
- **Columns Kept**: `annotation_id`, `patient_id`, `note`, `trial_id`, `criterion_type`, `criterion_text`, `expert_eligibility`, `expert_sentences`
- **Columns Removed**: `trial_title`, `gpt4_explanation`, `explanation_correctness`, `gpt4_sentences`, `gpt4_eligibility`, `training` (GPT-4 predictions excluded per spec)

### Patient-Grouped Split (Seed 42)
```
Total patients: 53 (sorted, shuffled with numpy.random.default_rng(42))
Train:  37 patients → 603 annotations
Val:     8 patients → 206 annotations
Test:   8 patients → 206 annotations
```
- **Manifest**: `task1/outputs/train/split_manifest.json` (identical to `data/processed/split_manifest.json`)
- **No Patient Leakage**: Each patient's annotations in exactly one split

### Tokenization
- **Tokenizer**: `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")`
- **Max Sequence Length**: 2048 tokens
- **Chat Template**: Qwen2.5 default (system + user + assistant)
- **Padding**: Right (training), Left (generation)
- **Truncation Strategy**: Remove sentences from END of patient note (preserves early critical info)
- **Truncation Log**: 0 examples truncated (max input length 1077 tokens < 2048)

### Input Format
```json
{
  "system": "You are a clinical trial eligibility expert...",
  "user": "Patient Description:\n{note}\n\nCriterion Type: {type}\nCriterion Text: {text}",
  "assistant": "{\"criterion_id\": \"...\", \"criterion_type\": \"...\", \"criterion_text\": \"...\", \"prediction\": \"...\", \"evidence_sentence_ids\": [...], \"evidence_text\": [...], \"explanation\": \"...\"}"
}
```

### Output Schema (6 Eligibility Labels)
```python
ELIGIBILITY_LABELS = [
    "included", "not included", "not excluded", "excluded",
    "not enough information", "not applicable"
]
```
All 6 labels preserved — no binary collapse.

### Training Objective
- **Loss**: Causal Language Modeling (next-token prediction)
- **Labels**: `-100` for prompt tokens, actual token IDs for assistant response
- **Target**: Full JSON object as single completion

---

## 4. Fine-tuning Configuration (Rubric A3, B3)

### QLoRA Hyperparameters (config.yaml)

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  max_seq_len: 2048
  trust_remote_code: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: "float16"

lora:
  r: 128
  lora_alpha: 256
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 5e-5
  weight_decay: 0.1
  warmup_steps: 50
  lr_scheduler_type: "cosine"
  eval_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  fp16: true
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
  early_stopping_patience: 1
  seed: 42
```

### Trainable Parameters
- **Total params**: 3.33B
- **Trainable (LoRA)**: ~239M (7.2%)
- **Adapter size**: 479 MB (FP16 safetensors)

### Training Log (3 Epochs)

| Epoch | Train Loss | Val Loss | Val Perplexity | Notes |
|-------|------------|----------|----------------|-------|
| 1 | 0.2071 | **1.3063** | 3.692 | **Best validation** |
| 2 | 0.1271 | 1.6032 | 4.969 | Overfitting begins |
| 3 | 0.0747 | 1.7728 | 5.887 | Continued overfitting |

**Overfitting Analysis**: Train loss decreases monotonically while validation loss increases. Best model at epoch 1. Small dataset (603 train samples) limits generalization. Recommendation: early stopping at epoch 1, higher dropout, or fewer epochs.

### Reproducibility Artifacts
- **Config**: `task1/training/config.yaml` (single source of truth)
- **Requirements**: `task1/training/requirements.txt` (exact Colab env)
- **Train Script**: `task1/training/train.py` (loads tokenized data, applies QLoRA, trains)
- **Tokenized Data**: `task1/preprocessing/tokenizer.py` → `task1/data/tokenized/`
- **Checkpoints**: Saved to `results/task1/checkpoints/` (best model loaded at end)

---

## 5. Before/After Fine-tuning Comparison (Rubric B4)

### Test Set Evaluation (206 samples, 8 patients)

| Metric | Base Qwen2.5-3B | Fine-tuned (LoRA) | Improvement |
|--------|-----------------|-------------------|-------------|
| **Accuracy** | 0.2282 | **0.7330** | **+0.5048** |
| **Macro F1** | 0.1365 | **0.4501** | **+0.3136** |

### Per-Class F1 Comparison

| Label | Base | Fine-tuned | Δ |
|-------|------|------------|---|
| included | 0.297 | 0.783 | +0.486 |
| not included | 0.000 | 0.000 | 0.000 |
| not excluded | 0.182 | 0.840 | +0.658 |
| excluded | 0.000 | 0.000 | 0.000 |
| not enough information | 0.204 | 0.624 | +0.420 |
| not applicable | 0.000 | 0.455 | +0.455 |

### Base Model Behavior
- Predicts "included" for all positive cases (recall=1.0, precision=0.17)
- Predicts "not excluded" for most others (recall=0.11)
- **Zero predictions** for: not included, excluded, not applicable
- Does not output structured JSON with evidence sentences

### Decoding Settings (Identical for Both)
- **Greedy decoding** (`do_sample=False`)
- **Max new tokens**: 512
- **Temperature**: N/A (greedy)
- **Prompt**: Identical system + user template

---

## 6. Validation Performance

### Validation Set (206 samples, 8 patients)

| Epoch | Val Loss | Val Perplexity | Eval Acc | Eval Macro F1 |
|-------|----------|----------------|----------|---------------|
| 1 | 1.3063 | 3.692 | — | — |
| 2 | 1.6032 | 4.969 | — | — |
| 3 | 1.7728 | 5.887 | — | — |

**Note**: Validation metrics during training tracked loss only. Full classification metrics computed on held-out test set post-training.

---

## 7. Truncation Documentation

- **Max sequence length**: 2048 tokens
- **Actual max observed**: 1077 tokens (train), 784 tokens (test)
- **Truncated examples**: 0 / 1015 (0%)
- **Strategy**: Remove sentences from END of patient note (last sentences first)
- **Impact**: No relevant evidence removed — all patient notes fit within context

---

## 8. Files for Submission (Rubric H2)

| Artifact | Location |
|----------|----------|
| Fine-tuning script | `task1/training/train.py` |
| Config | `task1/training/config.yaml` |
| Requirements | `task1/training/requirements.txt` |
| Tokenization | `task1/preprocessing/tokenizer.py` |
| Evaluation script | `task1/eval/eval.py` |
| Metrics | `task1/eval/metrics.py` |
| Adapter weights | Google Drive: `https://drive.google.com/drive/folders/1YEafpplZJwsxZx9rfEQxkuEPuCg8xUrp?usp=sharing` |
| Training logs | `task1/outputs/train/training_log.json` |
| Split manifest | `task1/outputs/train/split_manifest.json` |
| Test predictions (fine-tuned) | `task1/outputs/eval/test_predictions.json` |
| Test predictions (base) | `task1/outputs/base_eval/test_predictions.json` |
| Results summary (fine-tuned) | `task1/outputs/eval/README.md` |
| Results summary (base) | `task1/outputs/base_eval/README.md` |
| Adapter README | `task1/outputs/adapter/README.md` |
| Colab notebooks | `task1/colab/fine_tune-run1(train).ipynb`, `task1/colab/fine-tune-run2(eval).ipynb` |

---

## 9. Key Technical Decisions & Alternatives Rejected (Rubric A3)

| Decision | Alternatives Considered | Why Rejected |
|----------|------------------------|--------------|
| QLoRA over full fine-tuning | Full FT, LoRA-FP16 | VRAM constraints (6GB) |
| r=128, alpha=256 | r=64, r=256 | r=128 standard for 3B; higher overfits more |
| Target all linear layers | Only attention (q,k,v,o) | FFN layers (gate,up,down) critical for knowledge |
| 3 epochs | 5-10 epochs | Overfitting observed at epoch 2 |
| Gradient accumulation=16 | Larger batch size | OOM on T4/4050; accum simulates batch=16 |
| paged_adamw_8bit | adamw_torch | 8-bit optimizer states save VRAM |
| Greedy decoding | Sampling (temp>0) | Deterministic for reproducibility & comparison |
| Remove from END for truncation | Remove from START | Early sentences often contain demographics/chief complaint |

---

## 10. Known Limitations

1. **Overfitting**: Validation loss increases after epoch 1 (1.31 → 1.77)
2. **Small dataset**: 603 train samples from 37 patients
3. **Class imbalance**: "not excluded" (91), "not enough information" (66) dominate; "excluded" (0) absent in test
4. **Base model evidence F1 misleading**: Empty predictions match empty gold → 0.70 F1
5. **No excluded class in test**: Cannot evaluate "excluded" prediction capability

---

## 11. Reproducibility Checklist

- [x] Fixed random seed (42) for data split, training, sampling
- [x] Exact dependency versions in `requirements.txt`
- [x] Single config file (`config.yaml`) for all hyperparameters
- [x] Tokenization script separate from training (`preprocessing/tokenizer.py`)
- [x] Evaluation script loads base+adapter identically for before/after
- [x] Split manifest published with patient IDs and counts
- [x] Adapter weights hosted with download link
- [x] Training curves logged per epoch
- [x] Truncation documented (0 examples affected)

---

*This document serves as the single source of truth for Task 1 report writing.*