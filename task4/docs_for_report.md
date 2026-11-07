# Task 4: Efficient Tiled Attention with Online Softmax

**Course:** Generative AI — Assignment #2  
**Target Architecture:** Multi-Head Attention ($H=8, d_k=128$) matching Qwen2.5-3B  
**Execution Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB Dedicated GDDR6 VRAM)  
**Status:** Complete, Mathematically Proven & Rigorously Verified  

---

## 1. Executive Summary & Problem Formulation

In modern Transformer architectures, the self-attention mechanism is the fundamental building block enabling tokens to attend across the entire context window. Standard Scaled Dot-Product Attention (SDPA) computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where $Q \in \mathbb{R}^{B \times H \times N \times d}$, $K \in \mathbb{R}^{B \times H \times N \times d}$, and $V \in \mathbb{R}^{B \times H \times N \times d}$.

### 1.1 The Quadratic Memory Wall ($\mathcal{O}(N^2)$)
Standard implementations materialize the unnormalized attention score matrix $S = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times N \times N}$ and the normalized attention probabilities $P = \text{softmax}(S) \in \mathbb{R}^{B \times H \times N \times N}$ directly in GPU High-Bandwidth Memory (HBM).

For long context windows, this intermediate representation becomes a catastrophic bottleneck:
- At sequence length $N = 4,096$ with $H = 8$ heads in FP16 ($2\text{ bytes}$), storing the attention matrix requires:
  $$\text{Memory}_{S} = 1 \times 8 \times 4096 \times 4096 \times 2\text{ bytes} = 268,435,456\text{ bytes} \approx 268.4\text{ MB}$$
- Together with backward pass activations, optimizer states, and multi-layer stacks, this $\mathcal{O}(N^2)$ footprint causes Out-of-Memory (OOM) failures and saturates memory bus bandwidth.

### 1.2 Task 4 Solution
In Task 4, we design, implement, and benchmark:
1. **Reference Scaled Dot-Product Attention (`reference_scaled_dot_product_attention`)**: A baseline materializing full $\mathcal{O}(N^2)$ attention with numerical stabilization and causal masking.
2. **Tiled Attention with Online Softmax (`tiled_online_softmax_attention`)**: A FlashAttention-1-style blocked attention algorithm that computes attention in small tiles ($B_r \times B_c$) using **online softmax**, never materializing the full $N \times N$ matrix and reducing memory complexity to $\mathcal{O}(N)$.
3. **PyTorch Fused Backend Exploration (`pytorch_fused_attention`)**: Profiling native C++/CUDA attention kernels (FlashAttention-2 and Memory-Efficient Attention).
4. **Numerical Verification & Systems Analysis**: Proving numerical equivalence ($< 10^{-6}$ max absolute error) and explaining why educational PyTorch loop tiling incurs Python interpreter overhead compared to fused C++/CUDA SRAM kernels.

---

## 2. Mathematical Foundations: From Standard to Online Softmax

### 2.1 The Classical 3-Pass Softmax Bottleneck
Given a vector of raw scores $x = [x_1, x_2, \dots, x_N] \in \mathbb{R}^N$, the standard numerically stable softmax is computed in three sequential passes over the entire sequence:
1. **Pass 1 (Find Maximum):**
   $$m = \max_{1 \le j \le N} x_j$$
2. **Pass 2 (Compute Exponentials & Sum):**
   $$\ell = \sum_{j=1}^N e^{x_j - m}$$
3. **Pass 3 (Normalize & Project Values):**
   $$p_j = \frac{e^{x_j - m}}{\ell}, \quad O = \sum_{j=1}^N p_j v_j$$

Because Pass 2 requires the global maximum $m$ computed from the *entire* row, and Pass 3 requires the total normalizer $\ell$ summed over the *entire* row, standard softmax cannot proceed until all $N$ elements are computed and stored in memory.

---

### 2.2 The Online Softmax Algorithm (Milakov & Gimelshein 2018; Dao et al. 2022)

Online softmax overcomes this limitation by maintaining **running statistics** ($m$ and $\ell$) that are iteratively updated as new blocks of scores arrive.

Suppose a row $x$ is partitioned into two consecutive blocks: $x^{(1)} \in \mathbb{R}^{B_c}$ and $x^{(2)} \in \mathbb{R}^{B_c}$.

#### Step 1: Processing Block 1
- Local maximum of block 1:
  $$m^{(1)} = \max_{j} x_j^{(1)}$$
- Local normalizer of block 1:
  $$\ell^{(1)} = \sum_{j} e^{x_j^{(1)} - m^{(1)}}$$
- Partial unnormalized output accumulator:
  $$O^{(1)} = \sum_{j} e^{x_j^{(1)} - m^{(1)}} v_j^{(1)}$$

#### Step 2: Processing Block 2 & Rescaling
When block $x^{(2)}$ arrives:
- Local maximum of block 2:
  $$\tilde{m}^{(2)} = \max_{j} x_j^{(2)}$$
- Updated global running maximum:
  $$m^{(2)} = \max\left(m^{(1)}, \tilde{m}^{(2)}\right)$$
- Correction factor for Block 1 terms:
  Because block 1 was scaled by $m^{(1)}$, but the new maximum is $m^{(2)}$, we apply the identity:
  $$e^{x_j^{(1)} - m^{(2)}} = e^{x_j^{(1)} - m^{(1)}} \cdot e^{m^{(1)} - m^{(2)}}$$
  Defining the scalar rescaling factor:
  $$\alpha = e^{m^{(1)} - m^{(2)}}$$
- Unnormalized probabilities for block 2:
  $$\tilde{P}_j^{(2)} = e^{x_j^{(2)} - m^{(2)}}$$
- Updated global running normalizer:
  $$\ell^{(2)} = \ell^{(1)} \cdot \alpha + \sum_{j} \tilde{P}_j^{(2)}$$
- Updated output accumulator:
  $$O^{(2)} = O^{(1)} \cdot \alpha + \sum_{j} \tilde{P}_j^{(2)} v_j^{(2)}$$

#### Step 3: Final Normalization
After all column blocks $T_c$ are processed, the exact output vector is obtained with a single division:
$$O = \frac{O^{(T_c)}}{\ell^{(T_c)}}$$

#### Mathematical Invariance Proof:
Expanding the numerator and denominator:
$$O = \frac{\sum_{j=1}^N e^{x_j - m^{(T_c)}} v_j}{\sum_{j=1}^N e^{x_j - m^{(T_c)}}} = \frac{e^{-m^{(T_c)}} \sum_{j=1}^N e^{x_j} v_j}{e^{-m^{(T_c)}} \sum_{j=1}^N e^{x_j}} = \frac{\sum_{j=1}^N e^{x_j} v_j}{\sum_{j=1}^N e^{x_j}}$$
The factor $e^{-m^{(T_c)}}$ cancels identically between numerator and denominator. Thus, online softmax produces the **exact mathematical result** of classical softmax without ever materializing the full vector or matrix!

---

## 3. Algorithm Implementation (`task4/attention.py`)

### 3.1 Two-Level Tiled Loop Architecture
Our implementation partitions the inputs into row tiles of size $B_r$ and column tiles of size $B_c$:
- $Q$ is split into $T_r = \lceil N_q / B_r \rceil$ row blocks ($Q_1, \dots, Q_{T_r}$)
- $K, V$ are split into $T_c = \lceil N_k / B_c \rceil$ column blocks ($K_1, \dots, K_{T_c}$ and $V_1, \dots, V_{T_c}$)

```
Algorithm: Tiled Online-Softmax Attention
--------------------------------------------------------------------------------
Input: Q, K in R^[B, H, N, d_k], V in R^[B, H, N, d_v], tile sizes B_r, B_c
Initialize: O = zeros([B, H, N, d_v]), l = zeros([B, H, N, 1]), m = fill(-inf, [B, H, N, 1])

For row_block i = 0 to T_r - 1:
    Load Q_i in R^[B, H, B_r, d_k]
    
    For col_block j = 0 to T_c - 1:
        # 1. Causal Block Pruning Optimization:
        If is_causal and (col_start > max_row_in_block):
            Skip col_block j (strictly upper triangular, saves ~50% compute)
            
        Load K_j, V_j in R^[B, H, B_c, d]
        
        # 2. Block Matrix Multiplication (GEMM):
        S_ij = (Q_i @ K_j.T) / sqrt(d_k)       # [B, H, B_r, B_c]
        
        # 3. Local Causal Masking (on diagonal band only):
        If is_causal and (col_end - 1 > row_start):
            Apply local mask where global_col > global_row
            
        # 4. Online Softmax Update:
        m_tilde = max_col(S_ij)                # [B, H, B_r, 1]
        m_new = max(m_i, m_tilde)
        alpha = exp(m_i - m_new)              # Rescaling factor
        P_tilde = exp(S_ij - m_new)           # [B, H, B_r, B_c]
        l_new = l_i * alpha + sum_col(P_tilde)
        
        # 5. Output Accumulator Rescaling:
        O_i = O_i * alpha + (P_tilde @ V_j)
        m_i = m_new
        l_i = l_new
        
    # 6. Final Row Block Normalization:
    O[i] = O_i / l_i
```

### 3.2 Causal Block Pruning Optimization
In autoregressive self-attention, query token $i$ is forbidden from attending to key tokens $j > i$.
In tiled space, if the minimum column index in block $j$ ($k_{\text{start}}$) is strictly greater than the maximum row index in block $i$ ($q_{\text{end}} - 1$), **every single element in block $(i, j)$ is causal $-\infty$**.

Our algorithm evaluates:
```python
if is_causal and (k_start > (q_end - 1 + (n_k - n_q))):
    continue  # Skip entire tile!
```
This pruning completely skips $\approx \frac{T_r(T_r - 1)}{2}$ block evaluations, cutting total matrix multiplications by **$50\%$**.

---

## 4. Correctness Verification Results (Rubric E1 & E3)

To satisfy Rubric E1 and E3, we developed a verification suite in [`task4/test_attention.py`](file:///c:/Users/fahad/Desktop/Projects/GenAI%20A2/task4/test_attention.py).

### 4.1 Verification Methodology
1. **Controlled Toy Matrix Test:** A small $4 \times 4$ problem with hand-verifiable values, ensuring non-causal and causal equivalence and validating that strictly upper-triangular attention probabilities are zero.
2. **Parametric Sweep:** Evaluated across a $6 \times 3 \times 2$ grid:
   - Sequence lengths: $N \in \{16, 64, 128, 256, 512, 1024\}$
   - Head dimensions: $d \in \{32, 64, 128\}$
   - Masking: Standard (Non-Causal) vs. Causal
   - Batch size $B = 2$, Heads $H = 4$
   - Block sizes: $B_r, B_c \in \{16, 32, 64\}$
3. **Irregular & Asymmetric Shapes:** Testing non-power-of-two boundaries ($N = 37, 137, 250$) and asymmetric cross-attention ($N_q = 45, N_k = 128$).
4. **Half-Precision (FP16) Numerical Stability:** Testing numerical precision under standard transformer inference precision.

### 4.2 Empirical Verification Results Table

| Configuration ($N$, $d$) | Mode | Block Size ($B_r, B_c$) | Max Abs Diff ($\|O_{\text{tile}} - O_{\text{ref}}\|_\infty$) | Mean Abs Diff | Relative Frobenius Error | Status |
|---|---|---|---|---|---|---|
| $N=16, d=32$ | Standard | $16 \times 16$ | $2.38 \times 10^{-7}$ | $2.07 \times 10^{-8}$ | $8.51 \times 10^{-8}$ | **PASSED** |
| $N=16, d=32$ | Causal | $16 \times 16$ | $3.58 \times 10^{-7}$ | $2.10 \times 10^{-8}$ | $6.39 \times 10^{-8}$ | **PASSED** |
| $N=64, d=64$ | Standard | $32 \times 32$ | $3.87 \times 10^{-7}$ | $2.69 \times 10^{-8}$ | $1.91 \times 10^{-7}$ | **PASSED** |
| $N=64, d=64$ | Causal | $32 \times 32$ | $4.77 \times 10^{-7}$ | $2.60 \times 10^{-8}$ | $1.12 \times 10^{-7}$ | **PASSED** |
| $N=128, d=128$ | Standard | $32 \times 32$ | $5.96 \times 10^{-7}$ | $2.49 \times 10^{-8}$ | $2.46 \times 10^{-7}$ | **PASSED** |
| $N=128, d=128$ | Causal | $32 \times 32$ | $4.77 \times 10^{-7}$ | $2.56 \times 10^{-8}$ | $1.36 \times 10^{-7}$ | **PASSED** |
| $N=256, d=128$ | Standard | $32 \times 32$ | $4.47 \times 10^{-7}$ | $2.32 \times 10^{-8}$ | $3.21 \times 10^{-7}$ | **PASSED** |
| $N=256, d=128$ | Causal | $32 \times 32$ | $5.96 \times 10^{-7}$ | $2.48 \times 10^{-8}$ | $1.69 \times 10^{-7}$ | **PASSED** |
| $N=512, d=128$ | Standard | $32 \times 32$ | $4.77 \times 10^{-7}$ | $2.23 \times 10^{-8}$ | $4.37 \times 10^{-7}$ | **PASSED** |
| $N=512, d=128$ | Causal | $32 \times 32$ | $7.15 \times 10^{-7}$ | $2.38 \times 10^{-8}$ | $2.09 \times 10^{-7}$ | **PASSED** |
| $N=1024, d=128$ | Standard | $32 \times 32$ | $3.58 \times 10^{-7}$ | $2.19 \times 10^{-8}$ | $6.03 \times 10^{-7}$ | **PASSED** |
| $N=1024, d=128$ | Causal | $32 \times 32$ | $4.77 \times 10^{-7}$ | $2.30 \times 10^{-8}$ | $2.66 \times 10^{-7}$ | **PASSED** |
| Non-Pow2 ($N=137$) | Causal | $32 \times 32$ | $4.77 \times 10^{-7}$ | $2.24 \times 10^{-8}$ | $2.64 \times 10^{-7}$ | **PASSED** |
| Asym ($N_q=45, N_k=128$) | Causal | $16 \times 32$ | $3.28 \times 10^{-7}$ | $2.15 \times 10^{-8}$ | $1.88 \times 10^{-7}$ | **PASSED** |
| FP16 ($N=512, d=128$) | Causal | $64 \times 64$ | $1.95 \times 10^{-3}$ | $5.16 \times 10^{-5}$ | $1.12 \times 10^{-4}$ | **PASSED** |

### Summary of Verification:
- **Total Test Configurations:** 36 parametric + 6 irregular + 1 toy + 1 FP16 = **44 / 44 PASSED**.
- **Float32 Maximum Absolute Error:** $\mathbf{7.75 \times 10^{-7}}$ (well below the $10^{-5}$ rubric threshold).
- **Float32 Mean Absolute Error:** $\mathbf{\approx 2.5 \times 10^{-8}}$.
- **Tripartite Agreement:** Output is identical to PyTorch native SDPA within IEEE floating-point tolerance.

---

## 5. Performance Benchmarks: Latency & Memory Scaling (Rubric E3)

Benchmarks were conducted on the **NVIDIA GeForce RTX 4050 Laptop GPU** with batch size $B=1$, heads $H=8$, head dimension $d=128$, and FP16 precision. Timing was recorded using synchronized CUDA events with repeated warm-up cycles ($R=5$).

### 5.1 Latency Comparison Table

| Sequence Length ($N$) | Reference SDPA (ms) | Tiled Online SDPA (ms) | PyTorch Math (ms) | PyTorch Fused SDPA (ms) | Fused vs Tiled Speedup |
|---|---|---|---|---|---|
| **128** | 0.59 | 1.84 | 0.38 | **0.09** | 21.2x |
| **256** | 0.34 | 7.74 | 0.44 | **0.12** | 63.9x |
| **512** | 0.41 | 18.82 | 1.06 | **0.21** | 91.4x |
| **1024** | 1.58 | 66.03 | 2.96 | **0.32** | 205.7x |
| **2048** | 6.96 | 256.11 | 12.49 | **1.01** | 253.3x |
| **4096** | 26.69 | 1,140.79 | 47.39 | **3.63** | 314.4x |

---

### 5.2 Peak GPU Memory Consumption Table

| Sequence Length ($N$) | Reference SDPA (MB) | Tiled Online SDPA (MB) | PyTorch Math (MB) | PyTorch Fused SDPA (MB) | Memory Saved (Tiled vs Ref) |
|---|---|---|---|---|---|
| **128** | 9.9 | **9.9** | 12.6 | 9.1 | 0.0 MB (0.1%) |
| **256** | 13.2 | **10.9** | 20.4 | 10.1 | -2.3 MB (17.4%) |
| **512** | 24.4 | **12.9** | 46.1 | 12.1 | -11.5 MB (47.1%) |
| **1024** | 65.2 | **16.9** | 138.1 | 16.1 | -48.2 MB (74.0%) |
| **2048** | 220.2 | **24.9** | 484.1 | 24.1 | -195.3 MB (88.7%) |
| **4096** | 824.3 | **41.0** | 1,824.2 | 40.1 | **-783.3 MB (95.0%)** |

---

## 6. Systems Analysis: Why Educational PyTorch Tiling Differs from Fused CUDA Kernels

Rubric E explicitly requires students to:
> *"explain why an educational PyTorch implementation may not outperform a fused kernel."*

Our empirical results above clearly illustrate this:
- At $N = 4,096$, **Tiled Online SDPA achieves the exact same $\mathcal{O}(N)$ memory efficiency as PyTorch Fused** ($41.0\text{ MB}$ vs $40.1\text{ MB}$, a $95\%$ reduction from Reference's $824.3\text{ MB}$).
- However, in execution runtime, Tiled Online SDPA takes $1,140.79\text{ ms}$, whereas PyTorch Fused takes only $3.63\text{ ms}$.

This performance disparity stems from three fundamental architectural and hardware reasons:

### 6.1 Python Interpreter & Kernel Launch Latency
In PyTorch, every high-level tensor operation (`torch.bmm`, `torch.max`, `torch.exp`, `torch.sum`, tensor slicing) generates one or more distinct CUDA kernel launches.
For sequence length $N = 4,096$ with block size $B_r = B_c = 64$:
- Number of row tiles: $T_r = 64$
- Causal column tiles evaluated: $\frac{64 \times 65}{2} = 2,080$ tile iterations.
- Each tile executes $\approx 8$ individual PyTorch operations.
- Total kernel launches: $2080 \times 8 \approx \mathbf{16,640\text{ separate CUDA kernel launches}}$!

On modern GPUs, the launch overhead of a single kernel from the CPU/driver is $\approx 3 - 5\,\mu\text{s}$. Just launching $16,640$ kernels incurs $\approx 50 - 80\text{ ms}$ of pure CPU/driver overhead, during which GPU compute cores frequently stall waiting for work.

In contrast, **Fused FlashAttention-2 compiles the entire algorithm into a single CUDA C++ mega-kernel**. Only **one single kernel launch** is issued for the entire sequence, eliminating driver overhead entirely.

---

### 6.2 The Memory Hierarchy: HBM vs. On-Chip SRAM
The primary bottleneck in modern GPUs is not compute (FLOPs), but **memory bandwidth (I/O)**:
- **RTX 4050 GPU High-Bandwidth Memory (DRAM/HBM):** $\approx 192\text{ GB/s}$
- **RTX 4050 On-Chip Shared Memory / L1 Cache (SRAM):** $\approx 15 - 19\text{ TB/s}$ ($100\times$ faster!)

In our educational PyTorch implementation, even though the full $N \times N$ matrix is not materialized, each intermediate block tensor ($S_{ij} \in \mathbb{R}^{B \times H \times 64 \times 64}$, $P_{ij}$, $m_i$, $\ell_i$) is still written to and read from **GPU DRAM (HBM)** on every loop step.

In a true fused CUDA kernel (FlashAttention):
1. Tiles of $Q_i, K_j, V_j$ are loaded into **fast on-chip SRAM**.
2. Matmul $Q_i K_j^T$, online softmax exponentiation, normalizer accumulation, and value reduction $P V$ are executed **entirely within SRAM and processor registers**.
3. Intermediate matrices $S_{ij}$ and $P_{ij}$ are **never written to DRAM at all**.
4. Only the final output $O_i$ is written back to DRAM.

Because SRAM is $100\times$ faster than DRAM, fused FlashAttention achieves compute saturation ($\approx 3.63\text{ ms}$), while educational PyTorch tiling is bound by DRAM read/write round-trips.

---

### 6.3 Memory Scaling Validation: Achieving the $\mathcal{O}(N)$ Theoretical Goal
Despite the runtime difference caused by language-level kernel boundaries, our educational tiled attention **perfectly achieves the theoretical memory scaling objective**:
$$\text{Memory}_{\text{Tiled}} = \mathcal{O}(B \cdot H \cdot (N \cdot d + B_r \cdot B_c)) = \mathcal{O}(N)$$
At $N = 4,096$:
- Standard attention requires **$824.3\text{ MB}$** (and PyTorch Math requires **$1,824.2\text{ MB}$**).
- Our tiled attention requires only **$41.0\text{ MB}$** (matching the fused kernel's $40.1\text{ MB}$).

This empirically validates that the tiled online softmax algorithm successfully circumvents the quadratic memory wall of self-attention.

---

## 7. Summary of Task 4 Deliverables

All components for Task 4 are complete and self-contained in `task4/`:
- **`task4/__init__.py`**: Public module interface exporting reference and tiled attention routines.
- **`task4/attention.py`**:
  - `reference_scaled_dot_product_attention()`: Full $\mathcal{O}(N^2)$ reference with stable softmax and causal masking.
  - `tiled_online_softmax_attention()`: FlashAttention-1 online softmax tiling with running row maxima, running normalizers, rescaled accumulators, and causal block pruning.
  - `pytorch_fused_attention()`: SDPA backend explorer across FlashAttention, MemEfficient, and Math backends.
- **`task4/test_attention.py`**: Correctness suite across 44 test configurations, achieving $< 7.75 \times 10^{-7}$ maximum error.
- **`task4/benchmark_attention.py`**: CUDA event synchronized latency and peak memory benchmark across $N \in [128, 256, 512, 1024, 2048, 4096]$.
- **`task4/outputs/verification_results.json`**: Machine-readable correctness verification data.
- **`task4/outputs/benchmark_results.json`**: Machine-readable latency and memory benchmark data.
- **`task4/outputs/benchmark_summary.md`**: Markdown summary tables for reports.
- **`task4/docs_for_report.md`**: IEEE research report chapter with proofs and architectural analysis.

