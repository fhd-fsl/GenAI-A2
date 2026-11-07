# Task 4: Attention Benchmark Results Summary

- **Device:** NVIDIA GeForce RTX 4050 Laptop GPU (cuda)
- **Attention Parameters:** Batch=1, Heads=8, HeadDim=128, Precision=torch.float16
- **Timing Protocol:** Repetitions=5, Warm-up=3, Synchronized CUDA Events

## 1. Latency Comparison Table (ms)

| Sequence Length (N) | Reference SDPA (ms) | Tiled Online SDPA (ms) | PyTorch Math (ms) | PyTorch Fused (ms) | Fused vs Tiled Speedup |
|---|---|---|---|---|---|
|   128 | 0.59 | 1.84 | 0.38 | **0.09** | 21.2x |
|   256 | 0.34 | 7.74 | 0.44 | **0.12** | 63.9x |
|   512 | 0.41 | 18.82 | 1.06 | **0.21** | 91.4x |
|  1024 | 1.58 | 66.03 | 2.96 | **0.32** | 205.7x |
|  2048 | 6.96 | 256.11 | 12.49 | **1.01** | 253.3x |
|  4096 | 26.69 | 1140.79 | 47.39 | **3.63** | 314.4x |

## 2. Peak GPU Memory Consumption Table (MB)

| Sequence Length (N) | Reference Peak (MB) | Tiled Online Peak (MB) | PyTorch Math Peak (MB) | PyTorch Fused Peak (MB) | Memory Saved (Tiled vs Ref) |
|---|---|---|---|---|---|
|   128 | 9.9 | **9.9** | 12.6 | 9.1 | -0.0 MB (0.1%) |
|   256 | 13.2 | **10.9** | 20.4 | 10.1 | -2.3 MB (17.4%) |
|   512 | 24.4 | **12.9** | 46.1 | 12.1 | -11.5 MB (47.1%) |
|  1024 | 65.2 | **16.9** | 138.1 | 16.1 | -48.2 MB (74.0%) |
|  2048 | 220.2 | **24.9** | 484.1 | 24.1 | -195.3 MB (88.7%) |
|  4096 | 824.3 | **41.0** | 1824.2 | 40.1 | -783.3 MB (95.0%) |

## 3. Key Observations & Architectural Explanations

1. **Quadratic Memory Explosion of Reference Attention:** As sequence length scales from 128 to 4096, the Reference SDPA peak memory explodes quadratically because it materializes the full `[1, 8, N, N]` score matrix.
2. **Linear Memory Scaling of Tiled Online Attention:** Our tiled implementation maintains strict O(N) memory scaling because it only holds `[B, H, B_r, B_c]` tiles in memory at any moment.
3. **Why Educational PyTorch Tiling is Slower than Fused Kernels:** While our tiled implementation matches FlashAttention's mathematical algorithm and achieves O(N) memory, PyTorch Python-level nested loops suffer from kernel-launch overhead (~5-10 us per tile). In contrast, `torch.nn.functional.scaled_dot_product_attention` (FlashAttention-2) runs within a single fused CUDA C++ mega-kernel entirely in on-chip GPU SRAM.
