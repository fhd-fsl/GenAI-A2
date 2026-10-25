# Model Selection — Task 1: Architecture Selection & Fine-Tuning Strategy

## Hardware Constraint
- **GPU**: NVIDIA RTX 4050 Laptop / Mobile — **6 GB VRAM** (≈5.7 GB usable after OS/driver overhead)
- **Quantization target**: 4-bit (Q4_K_M / NF4) for base model weights
- **Context window**: 4K–8K tokens (patient note + criterion + prompt + generation)

---

## Candidate Open-Weight Decoder Models (Evaluated at Q4_K_M)

| Model | Params | Q4_K_M Weights | KV Cache (6K ctx, GQA) | Total Est. VRAM | License | Architecture Notes |
|-------|--------|----------------|------------------------|-----------------|---------|-------------------|
| **Qwen3-4B-Instruct** | 4.02B | ~2.3 GB | ~0.4 GB (8 KV heads) | **~2.7 GB** | Apache 2.0 | GQA (8 KV heads), RoPE θ=1M, 36 layers, 32K ctx; "thinking mode" toggle |
| **Qwen3-8B-Instruct** | 8.19B | ~4.9 GB | ~0.7 GB (8 KV heads) | **~5.6 GB** | Apache 2.0 | GQA (8 KV heads), RoPE θ=1M, 36 layers, 128K ctx; stronger reasoning |
| **Llama-3.2-3B-Instruct** | 3.21B | ~1.8 GB | ~0.5 GB (8 KV heads) | **~2.3 GB** | Llama 3.2 Community | GQA (8 KV heads), RoPE θ=500K, 28 layers, 128K ctx; pruned/distilled from 3.1 |
| **Llama-3.1-8B-Instruct** | 8.03B | ~4.9 GB | ~1.0 GB (32 KV heads) | **~5.9 GB** | Llama 3.1 Community | MHA (32 KV heads), RoPE θ=500K, 32 layers, 128K ctx; larger cache |
| **Phi-4-mini (3.8B)** | 3.84B | ~2.3 GB | ~0.3 GB (GQA) | **~2.6 GB** | MIT | GQA, RoPE θ=1M, 32 layers, 128K ctx; strong reasoning for size |
| **Mistral-7B-Instruct-v0.3** | 7.25B | ~4.4 GB | ~0.7 GB (8 KV heads) | **~5.1 GB** | Apache 2.0 | GQA (8 KV heads), SWA (4096), RoPE θ=1M, 32 layers, 32K ctx |

> **VRAM formula**: `Weights (Q4) + KV Cache (2 × layers × kv_heads × head_dim × ctx_len × 2 bytes) + Overhead (~0.5 GB)`

---

## Final Selection: **Qwen3-4B-Instruct** (Primary) / **Qwen3-8B-Instruct** (Backup)

### Justification

| Criterion | Qwen3-4B-Instruct | Why It Wins |
|-----------|-------------------|-------------|
| **Parameter count** | 4.02B | Strongest published 4B-class model; Qwen3 family shows exceptional scaling (1.7B beats 32B on clinical NLP — Allal 2026) |
| **License** | **Apache 2.0** | Fully permissive — research, commercial derivatives, redistribution allowed |
| **Expected memory (inference, Q4_K_M)** | **~2.7 GB** (weights 2.3 GB + KV cache 0.4 GB @ 6K ctx) | **Large headroom** (~3 GB) for longer contexts, batched inference, or future Task 5 draft model |
| **Expected memory (QLoRA training)** | **~3.8–4.2 GB** (4-bit base + LoRA FP16 + gradients + optimizer + activation checkpointing) | **Comfortably fits 6 GB** with batch=1, grad_accum=16, seq_len=4096 |
| **Architecture** | GQA (8 KV heads), RoPE θ=1M, 36 layers, 32K context | GQA keeps KV cache 4× smaller than MHA; 32K context > needed for patient + criterion |
| **Hardware suitability** | Verified on 6 GB cards via llama.cpp/Ollama/Unsloth at Q4_K_M | Stable runs at 30–50 tok/s (batch=1, 4K ctx); Unsloth QLoRA recipes tested |
| **Medical reasoning evidence** | Qwen2.5-3B + CoT-SFT → 47.8% MedQA (Subaharan 2025); Qwen3-1.7B + FT beats Qwen3-32B (Allal 2026) | **Sub-7B Qwen models reach specialized 7B medical performance** with fine-tuning |
| **Fine-tuning ecosystem** | Unsloth, PEFT, Axolotl, LLaMA-Factory all support Qwen3 QLoRA | Reproducible training pipeline; Unsloth 2× speedup on 6 GB |

> **Backup**: Qwen3-8B-Instruct — stronger reasoning, 128K context, but VRAM ~5.6 GB leaves <0.5 GB margin for training. Use only if 4B validation performance is insufficient.

---

## Fine-Tuning Strategy: **QLoRA (4-bit NF4, Double Quant)**

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Base quantization** | 4-bit NormalFloat (NF4), double quantization | Lowest memory; QLoRA matches LoRA within 2–4 pp (Yim et al. 2024; Estupiñán-Ojeda et al. 2025) |
| **LoRA rank (r)** | **128** | Token-level tasks (evidence sentence-ID extraction) need r ≥ 128 (Estupiñán-Ojeda 2025) |
| **LoRA alpha** | 256 (2×r) | Standard scaling |
| **Target modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All attention + MLP linears; ~45M trainable params (1.1%) |
| **Dropout** | 0.05 | Prevents overfitting on small dataset (37 patients × ~10 criteria) |
| **Seq. length** | 4096 | Fits patient note (avg ~800 tok) + criterion + prompt + generation |
| **Batch / Grad accum** | 1 / 16 (effective batch 16) | Fits 6 GB with gradient checkpointing |
| **Peak VRAM (train)** | **~3.8–4.2 GB** | 4-bit base + LoRA adapters (FP16) + gradients + optimizer states + activation checkpointing |
| **Optimizer** | `paged_adamw_8bit` (bitsandbytes) | Further reduces optimizer memory |
| **Scheduler** | Cosine, warmup 3%, lr=2e-4, 3 epochs | Standard for QLoRA on instruct models |
| **Gradient checkpointing** | Enabled | Reduces activation memory by ~30% |

> **Why QLoRA?** Full FT needs ~24 GB; FP16 LoRA needs ~10–12 GB — both exceed 6 GB. QLoRA is the **only viable option** preserving >95% of LoRA quality on this hardware (Sources 2, 3).

---

## Data Preprocessing & Tokenization (Task 1 Requirements)

### Input Format
```
Patient description: [free-text clinical note, multiple sentences]
Criterion type: [Inclusion / Exclusion]
Criterion text: [single criterion sentence]
```

### Tokenization
- **Tokenizer**: Qwen3 tokenizer (BPE, 151K vocab) — `AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct")`
- **Chat template**: Qwen3 chat template with system prompt for eligibility reasoning
- **Max length**: 4096 tokens (truncation strategy documented below)

### Truncation Strategy (Documented)
| Component | Typical Tokens | Truncation Priority |
|-----------|----------------|---------------------|
| System prompt + instruction | ~200 | Never truncated |
| Criterion type + text | ~50 | Never truncated |
| Patient description | ~800 (avg) | **Truncated last** — only if total > 4096 |
| Generation budget | ~512 | Reserved |

**Policy**: Patient description is truncated from the **end** (least recent sentences) only when combined input exceeds 4096 tokens. Truncation events are logged with patient ID, criterion ID, and tokens removed. This ensures early/evidence-bearing sentences are preserved.

### Output Schema (Structured JSON)
```json
{
  "criterion_id": "NCTxxxxxx_I_1",
  "criterion_type": "Inclusion",
  "criterion_text": "Patient must have histologically confirmed...",
  "prediction": "Eligible | Not Eligible | Insufficient Information | Not Applicable",
  "evidence_sentence_ids": [0, 3, 5],
  "evidence_text": ["Sentence 0 text", "Sentence 3 text", "Sentence 5 text"],
  "explanation": "The patient description mentions... which satisfies the criterion because..."
}
```
- **Labels**: Preserve all 4 expert categories from dataset (Eligible, Not Eligible, Insufficient Information, Not Applicable) — no binary collapse.
- **Evidence**: Return original sentence indices from patient description (0-indexed) + extracted text spans.

### Training Objective
- **Primary**: Causal language modeling (next-token prediction) on formatted JSON output
- **Loss**: Standard cross-entropy on generated tokens; no auxiliary losses
- **Teacher forcing**: Full sequence during training; greedy/beam at inference

---

## Validation & Evaluation Plan

| Split | Patients | Criteria (est.) | Use |
|-------|----------|-----------------|-----|
| Train | 37 | ~370 | QLoRA fine-tuning |
| Validation | 8 | ~80 | Hyperparameter selection, early stopping, checkpoint selection |
| Test | 8 | ~80 | **Held-out final evaluation only** (untouched until final report) |

### Metrics (Validation + Test)
- **Eligibility classification**: Accuracy, Macro-F1, Per-class Precision/Recall/F1, Confusion matrix (overall + Inclusion vs. Exclusion separate)
- **Evidence retrieval**: Precision, Recall, F1 (micro/macro) on sentence IDs; handling of empty gold/predicted sets per dataset protocol
- **Before/after comparison**: Base Qwen3-4B-Instruct (zero-shot + few-shot) vs. fine-tuned on identical validation/test examples with same decoding settings

---

## Task 5 Note (Minimal)

> **Speculative decoding (Task 5)** requires a smaller draft model with **identical tokenizer**. Qwen3-4B pairs natively with **Qwen3-0.6B** (0.3 GB Q4, same tokenizer, purpose-built draft). Both fit simultaneously on 6 GB GPU (~3.0 GB total). This compatibility is a secondary factor in model selection; primary justification remains Task 1 fitness.

---

## Summary

| Decision | Choice |
|----------|--------|
| **Target model** | `Qwen/Qwen3-4B-Instruct` |
| **Quantization (inference)** | Q4_K_M |
| **Fine-tuning method** | **QLoRA (4-bit NF4, r=128, α=256)** |
| **Tokenizer** | Qwen3 BPE (151K vocab) |
| **Max sequence length** | 4096 (patient truncation from end, logged) |
| **Output schema** | Structured JSON with 4-class label + evidence sentence IDs + explanation |
| **Expected inference VRAM** | ~2.7 GB @ 6K ctx |
| **Expected QLoRA train VRAM** | ~3.8–4.2 GB @ 4K ctx, grad_accum=16 |

All choices are **reproducible on a single RTX 4050 6 GB** with open-source tooling (Unsloth, PEFT, Hugging Face Transformers, bitsandbytes).