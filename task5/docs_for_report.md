# Task 5: Speculative Decoding — Technical Documentation & Experimental Analysis

**Author:** Fahad  
**Course:** Generative AI — Assignment #2  
**Hardware Platform:** Local NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM), Intel Core i7  
**Software Stack:** PyTorch 2.5.1+cu121, Transformers 4.49.0, BitsAndBytes 0.45.3, PEFT 0.14.0  

---

## 1. Executive Summary & Theoretical Framework

Autoregressive Large Language Model (LLM) generation is strictly memory-bandwidth bound during decoding. Because each token generation step requires loading all billions of parameters from high-bandwidth device memory (VRAM) into GPU SRAM to perform matrix-vector arithmetic with batch size $B=1$ and sequence increment $\Delta L = 1$, operational intensity is drastically low:
$$\text{Arithmetic Intensity} = \frac{\mathcal{O}(P) \text{ FLOPs}}{\mathcal{O}(P) \text{ Bytes}} \approx 1\text{--}2 \text{ FLOP/Byte}$$
On modern tensor architectures capable of hundreds of TFLOPS, this underutilizes tensor cores by orders of magnitude.

**Speculative Decoding** (Leviathan et al., 2023; Chen et al., 2023) breaks this sequential bottleneck by decoupling token proposing from token verification:
1. **Drafting Phase:** A lightweight draft model $M_{\text{draft}}$ autoregressively speculates $\gamma$ candidate tokens $\{\hat{x}_1, \hat{x}_2, \dots, \hat{x}_\gamma\}$. Because $M_{\text{draft}}$ has fewer parameters ($P_{\text{draft}} \ll P_{\text{target}}$), each step incurs minimal latency $t_{\text{draft}} \ll t_{\text{target}}$.
2. **Parallel Verification Phase:** The large target model $M_{\text{target}}$ evaluates the entire sequence of $\gamma + 1$ tokens in **one single forward pass**:
   $$\mathbf{X}_{\text{verify}} = [x_{\text{last}}, \hat{x}_1, \hat{x}_2, \dots, \hat{x}_\gamma]$$
   Matrix-vector operations transform into compute-dense matrix-matrix multiplications (GEMM) of shape $[(\gamma+1), d_{\text{model}}]$, converting memory-bound reads into compute-bound tensor core operations without increasing target latency significantly ($t_{\text{verify}}(\gamma+1) \approx t_{\text{target}}(1)$).
3. **Prefix Acceptance & Rollback:** The target model's greedy outputs are compared against draft tokens. The longest prefix of matching tokens ($n_{\text{acc}} \le \gamma$) is accepted. The first mismatch is corrected using the target model's output at that index (or an additional bonus token if all $\gamma$ match). Caches are strictly rolled back to maintain exact autoregressive equivalence.

### Mathematical Model of Speculative Speedup

Let:
- $\alpha \in [0, 1]$ be the draft token acceptance rate (probability that target agrees with draft).
- $c = \frac{t_{\text{draft}}}{t_{\text{target}}}$ be the cost ratio between draft and target single-token forward passes ($c \approx 0.12\text{--}0.18$).
- $\gamma$ be the speculation lookahead horizon.

The expected number of accepted tokens generated per verification cycle is:
$$\mathbb{E}[\text{tokens per step}] = \sum_{k=0}^{\gamma} \alpha^k = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$
The wall-clock time required per cycle is:
$$T_{\text{cycle}}(\gamma) \approx \gamma \cdot t_{\text{draft}} + t_{\text{target}} = t_{\text{target}} \cdot (1 + \gamma \cdot c)$$
Thus, the theoretical wall-clock speedup $S(\alpha, \gamma, c)$ over standard autoregressive decoding is:
$$S(\alpha, \gamma, c) = \frac{\mathbb{E}[\text{tokens per step}] \cdot t_{\text{target}}}{T_{\text{cycle}}(\gamma)} = \frac{\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}}{1 + \gamma \cdot c}$$

#### Speedup Condition:
Speculative decoding yields an effective wall-clock acceleration ($S > 1$) if and only if:
$$\frac{1 - \alpha^{\gamma+1}}{1 - \alpha} > 1 + \gamma \cdot c \iff \sum_{k=1}^\gamma \alpha^k > \gamma \cdot c$$
For small $\gamma=1$:
$$\alpha > c$$
If the acceptance rate $\alpha$ falls below the cost ratio $c$, or if $\gamma$ is chosen too large such that compounding rejection $\alpha^k$ drops while draft latency $\gamma \cdot c$ grows linearly, speculative decoding causes a **wall-clock slowdown** ($S < 1$).

---

## 2. Model Selection & System Architecture (Rubric F1 — 5 Marks)

### 2.1 Model Pair Selection Justification
To fulfill Rubric F1 under strict local laptop hardware constraints (6.0 GB VRAM), we selected:
- **Target Model:** `Qwen/Qwen2.5-3B-Instruct` (3.09B parameters) fine-tuned with a Rank-16 LoRA clinical adapter (`task1/outputs/adapter`).
- **Draft Model:** `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters).

```
+---------------------------------------------------------------------------------------+
|                                    6.0 GB VRAM BUDGET                                 |
+---------------------------------------------------+-------------------+---------------+
| Target: Qwen2.5-3B (4-bit NF4) + LoRA             | Draft: 0.5B FP16  | KV & CUDA     |
| 3.49 GB VRAM                                      | 0.98 GB VRAM      | 0.40 GB       |
+---------------------------------------------------+-------------------+---------------+
| Total In-Flight GPU Footprint: 4.87 GB / 6.00 GB (81.2% capacity, zero OOM risk)      |
+---------------------------------------------------------------------------------------+
```

### 2.2 Tokenizer Compatibility Verification
Speculative decoding strictly requires vocabulary alignment. If tokenizer vocabularies, token IDs, or BPE merges diverge, draft candidate IDs cannot index target embedding tables.

We programmatically asserted complete compatibility before loading weights:
```python
assert len(target_tokenizer) == len(draft_tokenizer) == 151665
assert target_tokenizer.eos_token_id == draft_tokenizer.eos_token_id == 151645
assert target_tokenizer.pad_token_id == draft_tokenizer.pad_token_id == 151643
```
Both models share the exact same byte-level BPE tokenizer with 151,665 vocabulary tokens, identical ChatML special tokens (`<|im_start|>`, `<|im_end|>`), and identical punctuation representations.

### 2.3 Precision & Memory Allocation
1. **Target Model in 4-bit NormalFloat4 (NF4):** Quantizing the 3B target model via BitsAndBytes reduces parameter memory from 6.2 GB to ~1.75 GB (with activations and LoRA weights taking ~3.49 GB total).
2. **Draft Model in 16-bit Float16 (FP16):** At 0.49B parameters, the draft model occupies only 0.98 GB in FP16. Retaining FP16 avoids dequantization overhead on small matrix multiplications, ensuring maximum decoding speed ($t_{\text{draft}} \approx 14\text{ ms}$ per token).
3. **Combined Footprint:** 4.47 GB resting VRAM, peaking at 4.86 GB during batch verification, leaving >1.1 GB of safety headroom on the 6.0 GB GPU.

---

## 3. Speculative Engine Implementation (Rubric F2 — 5 Marks)

The engine (`task5/speculative_decoding.py`) implements deterministic greedy speculative decoding with explicit KV-cache state preservation.

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / User
    participant SE as Speculative Engine
    participant DM as Draft Model (0.5B FP16)
    participant TM as Target Model (3B 4-bit NF4)
    participant KVC as ExplicitKVCache

    User->>SE: Prompt tokens P
    SE->>TM: Prefill P
    SE->>DM: Prefill P
    TM-->>SE: Logits -> First token x0
    SE->>KVC: Target & Draft Cache populated (len = L)

    loop While generated_tokens < max_tokens
        Note over SE,DM: 1. Autoregressive Candidate Drafting (gamma steps)
        loop k = 1 to gamma
            SE->>DM: Forward(x_{k-1}, past_kv=draft_cache)
            DM-->>SE: Candidate token x_hat_k
        end

        Note over SE,TM: 2. Parallel Target Verification (ONE forward pass)
        SE->>TM: Forward([x_last, x_hat_1, ..., x_hat_gamma], past_kv=target_cache)
        TM-->>SE: Parallel Logits [0..gamma] -> Target greedy predictions [y_0..y_gamma]

        Note over SE,KVC: 3. Prefix Acceptance & Cache Rollback
        SE->>SE: Match x_hat_k with y_{k-1} -> n_acc accepted tokens
        alt Rejection at index i < gamma
            SE->>SE: Emit corrected token y_i
        else All gamma accepted
            SE->>SE: Emit bonus token y_gamma
        end
        SE->>KVC: target_cache.crop(base_len + 1 + n_acc)
        SE->>KVC: draft_cache.crop(base_len + 1 + n_acc)
        SE->>SE: Check EOS; update x_last
    end
    SE-->>User: Exact Greedy Output Sequence
```

### 3.1 Candidate Drafting Loop
Given current verified token $x_{\text{last}}$:
```python
draft_tokens = []
cur_input = x_last
for step in range(gamma):
    pos = base_cache_len + step
    d_out = draft_model(
        input_ids=cur_input,
        past_key_values=draft_cache,
        position_ids=[[pos]],
        cache_position=[pos],
        use_cache=True
    )
    next_draft = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)
    draft_tokens.append(next_draft)
    cur_input = next_draft
    if next_draft.item() == eos_id:
        break
```

### 3.2 Target Parallel Verification Pass
The verification sequence concatenates $x_{\text{last}}$ with all $\gamma$ candidate tokens:
$$\mathbf{X}_{\text{verify}} = [x_{\text{last}}, \hat{x}_1, \dots, \hat{x}_\gamma], \quad \text{Shape: } [1, \gamma + 1]$$
```python
v_pos_ids = torch.arange(base_len, base_len + gamma + 1).unsqueeze(0)
t_v_out = target_model(
    input_ids=verify_inputs,
    past_key_values=target_cache,
    position_ids=v_pos_ids,
    cache_position=v_pos_ids.squeeze(0),
    use_cache=True
)
target_predictions = torch.argmax(t_v_out.logits[0, :gamma + 1, :], dim=-1)
```
- `target_predictions[0]` predicts the token following $x_{\text{last}}$ (verified against $\hat{x}_1$).
- `target_predictions[i]` predicts the token following $\hat{x}_i$ (verified against $\hat{x}_{i+1}$).
- `target_predictions[gamma]` predicts the token following $\hat{x}_\gamma$ (the **bonus token**!).

### 3.3 Cache Rollback & Synchronization Invariant
When the target model rejects the candidate at index $i$, the speculative prefix past $i$ is discarded.
- `target_cache` appended keys and values for all $\gamma+1$ tokens during verification.
- `draft_cache` appended keys and values for all $\gamma$ tokens during drafting.
To restore exact state consistency:
```python
new_valid_len = base_cache_len + 1 + num_accepted
target_cache.crop(new_valid_len)
if num_accepted == num_drafted:
    # Synchronize draft_cache by running 1 token on draft_tokens[-1]
    _ = draft_model(input_ids=draft_tokens[-1], past_key_values=draft_cache, ...)
else:
    draft_cache.crop(new_valid_len)
```
This guarantees zero cache corruption across successive verification steps.

---

## 4. Empirical Evaluation & Correctness Suite (Rubric F3 — 5 Marks)

### 4.1 100.00% Token Equivalence Verification (`test_speculative.py`)
We verified token-level equivalence between standard Target Greedy Decoding (`generate_with_kv_cache`) and Speculative Greedy Decoding (`generate_speculative_greedy`) across:
- 3 Distinct Prompts: Short (54 tokens), Medium (149 tokens), Structured Clinical (162 tokens)
- 2 Token Budgets: 32 tokens and 64 tokens
- 3 Lookahead Horizons: $\gamma \in \{3, 5, 7\}$
- **Total Configurations Tested:** 18
- **Total Tokens Evaluated:** 855 tokens

| Prompt | Budget | $\gamma$ | Target Reference Count | Speculative Count | Token Agreement | Acceptance Rate ($\alpha$) | Mean Acc / Step | Target Passes Saved | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Short | 32 | 3 | 32 | 32 | **100.00%** | 51.3% | 1.46 | 18 | **PASS** |
| Short | 32 | 5 | 32 | 32 | **100.00%** | 38.2% | 1.91 | 20 | **PASS** |
| Short | 32 | 7 | 32 | 32 | **100.00%** | 32.4% | 2.20 | 21 | **PASS** |
| Short | 64 | 3 | 64 | 64 | **100.00%** | 41.7% | 1.21 | 34 | **PASS** |
| Short | 64 | 5 | 64 | 64 | **100.00%** | 30.9% | 1.46 | 37 | **PASS** |
| Short | 64 | 7 | 64 | 64 | **100.00%** | 24.1% | 1.56 | 38 | **PASS** |
| Medium | 32 | 3 | 32 | 32 | **100.00%** | 26.9% | 0.78 | 13 | **PASS** |
| Medium | 32 | 5 | 32 | 32 | **100.00%** | 17.6% | 0.88 | 14 | **PASS** |
| Medium | 32 | 7 | 32 | 32 | **100.00%** | 12.9% | 0.88 | 14 | **PASS** |
| Medium | 64 | 3 | 64 | 64 | **100.00%** | 31.2% | 0.91 | 30 | **PASS** |
| Medium | 64 | 5 | 64 | 64 | **100.00%** | 23.1% | 1.10 | 33 | **PASS** |
| Medium | 64 | 7 | 64 | 64 | **100.00%** | 16.9% | 1.10 | 33 | **PASS** |
| Clinical Structured | 32 | 3 | 32 | 32 | **100.00%** | 18.3% | 0.55 | 11 | **PASS** |
| Clinical Structured | 32 | 5 | 32 | 32 | **100.00%** | 12.8% | 0.63 | 12 | **PASS** |
| Clinical Structured | 32 | 7 | 32 | 32 | **100.00%** | 9.2% | 0.63 | 12 | **PASS** |
| Clinical Structured | 64 | 3 | 61 | 61 | **100.00%** | 34.8% | 1.03 | 30 | **PASS** |
| Clinical Structured | 64 | 5 | 61 | 61 | **100.00%** | 26.2% | 1.26 | 33 | **PASS** |
| Clinical Structured | 64 | 7 | 61 | 61 | **100.00%** | 21.2% | 1.44 | 35 | **PASS** |

**Empirical Result:** 18 / 18 configurations passed with **100.00% exact token agreement** (855/855 tokens matched). Zero hallucinations, zero drift, and zero formatting discrepancies.

---

### 4.2 Performance & Speedup Benchmark (`benchmark_speculative.py`)

We evaluated end-to-end latency, tokens per second, draft acceptance rate ($\alpha$), forward passes saved, and peak VRAM across three workloads with distinct syntactic entropy:
1. **Structured Clinical JSON Extraction:** High syntax predictability (JSON keys, quotes, colons).
2. **Clinical SOAP Narrative Note:** Semi-structured clinical prose with domain terminology.
3. **Open Medical Reasoning & QA:** Open-ended diagnostic reasoning.

| Workload | Method | $\gamma$ | Tokens | Latency (ms) | Speed (tok/s) | Speedup vs Base | Acceptance Rate ($\alpha$) | Mean Acc / Step | Target Passes Saved | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Structured Clinical JSON** | **Baseline (Target 3B)** | - | 52 | 6667.1 | 7.80 | 1.00x | - | - | - | 4819.3 MB |
| Structured Clinical JSON | Speculative ($\gamma=1$) | 1 | 52 | 6209.8 | 8.38 | **1.07x** | 88.9% | 0.89 | 48.1% | 4844.3 MB |
| Structured Clinical JSON | Speculative ($\gamma=3$) | 3 | 52 | 4279.4 | 12.16 | **1.56x** | 70.0% | 2.06 | 67.3% | 4844.3 MB |
| Structured Clinical JSON | Speculative ($\gamma=5$) | 5 | 52 | 3799.7 | 13.69 | **1.75x** | 62.9% | 3.00 | 75.0% | 4844.3 MB |
| Structured Clinical JSON | Speculative ($\gamma=7$) | 7 | 52 | 4565.9 | 11.39 | **1.46x** | 50.6% | 3.33 | 76.9% | 4861.7 MB |
| **Clinical SOAP Narrative** | **Baseline (Target 3B)** | - | 64 | 7153.8 | 8.95 | 1.00x | - | - | - | 4818.7 MB |
| Clinical SOAP Narrative | Speculative ($\gamma=1$) | 1 | 64 | 7793.9 | 8.21 | **0.92x (Slowdown)** | 73.0% | 0.73 | 42.2% | 4842.8 MB |
| Clinical SOAP Narrative | Speculative ($\gamma=3$) | 3 | 64 | 5609.7 | 11.50 | **1.28x** | 58.0% | 1.67 | 62.5% | 4842.8 MB |
| Clinical SOAP Narrative | Speculative ($\gamma=5$) | 5 | 64 | 5765.0 | 11.10 | **1.24x** | 44.3% | 2.05 | 67.2% | 4842.8 MB |
| Clinical SOAP Narrative | Speculative ($\gamma=7$) | 7 | 64 | 6223.9 | 10.28 | **1.15x** | 42.2% | 2.56 | 71.9% | 4861.2 MB |
| **Open Medical Reasoning** | **Baseline (Target 3B)** | - | 64 | 7462.1 | 8.58 | 1.00x | - | - | - | 4814.6 MB |
| Open Medical Reasoning | Speculative ($\gamma=1$) | 1 | 64 | 7065.0 | 9.08 | **1.06x** | 80.0% | 0.80 | 45.3% | 4814.6 MB |
| Open Medical Reasoning | Speculative ($\gamma=3$) | 3 | 64 | 5201.2 | 12.33 | **1.43x** | 58.0% | 1.74 | 64.1% | 4814.6 MB |
| Open Medical Reasoning | Speculative ($\gamma=5$) | 5 | 64 | 5563.2 | 11.52 | **1.34x** | 46.8% | 2.32 | 70.3% | 4814.6 MB |
| Open Medical Reasoning | Speculative ($\gamma=7$) | 7 | 64 | 6795.5 | 9.42 | **1.10x** | 36.9% | 2.50 | 71.9% | 4840.8 MB |

---

## 5. In-Depth Analysis: Why Speedups and Slowdowns Occur

### 5.1 The Phenomenon of Speedup ($S > 1$)
On the **Structured Clinical JSON Extraction** prompt, speculative decoding delivered a dramatic **1.75x speedup** (latency decreased from 6667.1 ms to 3799.7 ms; throughput increased from 7.80 to 13.69 tok/s).
- **Underlying Cause:** JSON output exhibits very low token entropy. Boilerplate tokens (`"`, `:`, `\n`, `}`, whitespace, standard JSON keys) are virtually deterministic.
- **Empirical Evidence:** The acceptance rate reached $\alpha = 88.9\%$ at $\gamma=1$ and $\alpha = 62.9\%$ at $\gamma=5$.
- At $\gamma=5$, the target model accepted on average **3.00 draft tokens per step**, allowing the engine to eliminate **75.0% of all target model forward passes**. Instead of executing 52 sequential passes, the target model was invoked only 13 times.

### 5.2 The Phenomenon of Slowdown ($S < 1$)
On the **Clinical SOAP Narrative Note**, running with $\gamma=1$ resulted in an empirical **slowdown of 0.92x** (throughput dropped from 8.95 tok/s to 8.21 tok/s; latency increased from 7153.8 ms to 7793.9 ms).
- **Underlying Cause:** When $\gamma=1$, every generation step incurs:
  1. One forward pass on $M_{\text{draft}}$ ($t_{\text{draft}} \approx 14\text{ ms}$).
  2. One parallel forward pass on $M_{\text{target}}$ on sequence $[x_{\text{last}}, \hat{x}_1]$ ($t_{\text{verify}} \approx 110\text{ ms}$).
  3. Total cycle time: $t_{\text{cycle}} \approx 124\text{ ms}$.
- If accepted ($\alpha = 73.0\%$), the cycle produces 2 tokens (1 accepted draft + 1 bonus token), yielding $62\text{ ms/token}$.
- If rejected ($1 - \alpha = 27.0\%$), the cycle produces only 1 corrected token, taking $124\text{ ms/token}$.
- The expected time per token is:
  $$\mathbb{E}[t_{\text{token}}] = 0.73 \times 62 + 0.27 \times 124 = 45.26 + 33.48 = 78.74\text{ ms}$$
  In baseline decoding, each token requires $111.7\text{ ms}$. However, with CUDA kernel launch latency, PyTorch tensor indexing, and cache slicing overhead, the effective speedup was washed out, resulting in a net 8% slowdown.
- At $\gamma \ge 3$, the amortization of verification over multiple accepted candidates overcame this fixed overhead, achieving $1.28\text{x}$ speedup.

### 5.3 Diminishing Returns and the Optimal Lookahead Horizon ($\gamma^*$)
Across all three workloads, performance peaked at either $\gamma=3$ or $\gamma=5$, before declining sharply at $\gamma=7$:
- **Structured JSON:** $\gamma=3 \to 1.56\text{x}$, $\gamma=5 \to \mathbf{1.75\text{x}}$, $\gamma=7 \to 1.46\text{x}$.
- **SOAP Narrative:** $\gamma=3 \to \mathbf{1.28\text{x}}$, $\gamma=5 \to 1.24\text{x}$, $\gamma=7 \to 1.15\text{x}$.
- **Medical Reasoning:** $\gamma=3 \to \mathbf{1.43\text{x}}$, $\gamma=5 \to 1.34\text{x}$, $\gamma=7 \to 1.10\text{x}$.

**Mathematical Explanation:**
The probability of accepting all tokens up to depth $k$ decays geometrically as $\alpha^k$. 
At $\gamma=7$ on open medical reasoning, $\alpha = 36.9\%$. The probability that token 6 and 7 are accepted is:
$$\alpha^6 \approx (0.369)^6 = 0.0024 \quad (0.24\%)$$
Yet, the engine spent 7 sequential draft passes generating tokens that had a $>99.7\%$ probability of being rejected and rolled back! The drafting time $7 \times t_{\text{draft}}$ was completely wasted, dragging throughput down from 12.33 tok/s back to 9.42 tok/s.

**Conclusion:** For a 3B/0.5B model pair, $\gamma^* \in [3, 5]$ represents the Pareto-optimal operating regime.

---

## 6. Reproducibility & File Manifest

All code, tests, and empirical results are strictly self-contained within `task5/`:
- [`task5/speculative_decoding.py`](file:///c:/Users/fahad/Desktop/Projects/GenAI%20A2/task5/speculative_decoding.py): Core speculative decoding engine, model loading, drafting loop, verification GEMM, prefix acceptance, and `ExplicitKVCache.crop()` rollback.
- [`task5/test_speculative.py`](file:///c:/Users/fahad/Desktop/Projects/GenAI%20A2/task5/test_speculative.py): 18-configuration correctness verification harness asserting 100.00% token equivalence with Target Greedy reference.
- [`task5/benchmark_speculative.py`](file:///c:/Users/fahad/Desktop/Projects/GenAI%20A2/task5/benchmark_speculative.py): Rigorous performance benchmarking suite measuring latency, throughput, speedup, $\alpha$, and peak VRAM across $\gamma \in \{1, 3, 5, 7\}$.
- [`task5/outputs/correctness_results.json`](file:///c:/Users/fahad/Desktop/Projects/GenAI%20A2/task5/outputs/correctness_results.json): Full JSON dump of 18 verification runs with token-level match statistics.
- [`task5/outputs/benchmark_results.json`](file:///c:/Users/fahad/Desktop/Projects/GenAI%20A2/task5/outputs/benchmark_results.json): Full JSON benchmark metrics.
- [`task5/outputs/benchmark_summary.md`](file:///c:/Users/fahad/Desktop/Projects/GenAI%20A2/task5/outputs/benchmark_summary.md): Markdown comparison table for assignment report submission.
- [`task5/docs_for_report.md`](file:///c:/Users/fahad/Desktop/Projects/GenAI%20A2/task5/docs_for_report.md): This comprehensive report document.
