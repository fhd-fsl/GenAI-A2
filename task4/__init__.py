"""Task 4: Efficient Tiled Attention with Online Softmax.

Provides:
- reference_scaled_dot_product_attention: Full O(N^2) memory reference SDPA.
- tiled_online_softmax_attention: Blocked FlashAttention-style online softmax attention.
- pytorch_fused_attention: Wrapper around PyTorch native fused/efficient SDPA backends.
"""

from task4.attention import (
    pytorch_fused_attention,
    reference_scaled_dot_product_attention,
    tiled_online_softmax_attention,
)

__all__ = [
    "reference_scaled_dot_product_attention",
    "tiled_online_softmax_attention",
    "pytorch_fused_attention",
]
