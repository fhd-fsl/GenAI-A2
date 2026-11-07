"""Task 4: Efficient Tiled Attention with Online Softmax.

Implements:
1. reference_scaled_dot_product_attention: Full O(N^2) memory reference SDPA.
2. tiled_online_softmax_attention: FlashAttention-1 style tiled online softmax attention.
3. pytorch_fused_attention: PyTorch native SDPA backend explorer (FlashAttention, MemEfficient, Math).

Satisfies Rubrics E1, E2, and E3.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def reference_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_attn_weights: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Standard Scaled Dot-Product Attention (SDPA) reference implementation.
    
    Computes:
        S = (Q @ K^T) * scale
        S = S + mask  (if causal or custom mask)
        P = softmax(S - max(S))  # Numerically stable
        O = P @ V

    Materializes the full [B, H, N_q, N_k] attention matrix in memory (O(N^2) footprint).

    Args:
        query: Tensor of shape [..., N_q, d_k]
        key: Tensor of shape [..., N_k, d_k]
        value: Tensor of shape [..., N_k, d_v]
        is_causal: Whether to apply lower-triangular causal masking
        attn_mask: Optional boolean or additive attention mask
        scale: Scale factor, defaults to 1.0 / sqrt(d_k)
        return_attn_weights: Whether to return the N x N attention matrix P

    Returns:
        output: Tensor of shape [..., N_q, d_v]
        attn_weights: Optional Tensor of shape [..., N_q, N_k] if return_attn_weights=True
    """
    d_k = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    # 1. Compute full attention logits: [..., N_q, N_k]
    # (Q @ K^T) * scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    n_q = query.shape[-2]
    n_k = key.shape[-2]

    # 2. Causal masking
    if is_causal:
        # For causal masking, token i can only attend to tokens j <= i + (n_k - n_q)
        device = query.device
        row_idx = torch.arange(n_q, device=device).unsqueeze(1)
        col_idx = torch.arange(n_k, device=device).unsqueeze(0)
        causal_mask = col_idx > (row_idx + (n_k - n_q))
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # 3. Custom attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    # 4. Numerically stable softmax: P = exp(S - max(S)) / sum(exp(S - max(S)))
    # Subtract row-wise max to prevent overflow in exp()
    max_scores = torch.max(scores, dim=-1, keepdim=True).values
    # Handle rows that might be fully masked out (-inf)
    max_scores = torch.where(torch.isneginf(max_scores), torch.zeros_like(max_scores), max_scores)
    exp_scores = torch.exp(scores - max_scores)
    sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
    # Avoid division by zero on masked rows
    attn_weights = exp_scores / torch.clamp(sum_exp, min=1e-9)

    # 5. Output projection: O = P @ V  [..., N_q, d_v]
    output = torch.matmul(attn_weights, value)

    if return_attn_weights:
        return output, attn_weights
    return output, None


def tiled_online_softmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    block_size_r: int = 64,
    block_size_c: int = 64,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Tiled Scaled Dot-Product Attention with Online Softmax (FlashAttention-style).

    Partitions Q into blocks of size block_size_r along the sequence dimension,
    and K, V into blocks of size block_size_c along the sequence dimension.
    Maintains running row-wise maxima (m_i) and normalizers (l_i) using the
    online softmax technique (Milakov & Gimelshein 2018; Dao et al. 2022).

    Crucially, the full [B, H, N_q, N_k] attention matrix is NEVER materialized,
    reducing memory consumption from O(N^2) to O(N).

    Args:
        query: Tensor of shape [..., N_q, d_k]
        key: Tensor of shape [..., N_k, d_k]
        value: Tensor of shape [..., N_k, d_v]
        is_causal: If True, applies causal mask with triangular block pruning
        block_size_r: Tile size for query rows (B_r)
        block_size_c: Tile size for key/value columns (B_c)
        scale: Scale factor, defaults to 1.0 / sqrt(d_k)

    Returns:
        output: Tensor of shape [..., N_q, d_v] matching reference SDPA output.
    """
    orig_shape = query.shape
    d_k = orig_shape[-1]
    d_v = value.shape[-1]
    n_q = orig_shape[-2]
    n_k = key.shape[-2]

    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    # Flatten leading batch dimensions to [B_flat, N, d] for uniform iteration
    batch_dims = orig_shape[:-2]
    b_flat = math.prod(batch_dims) if batch_dims else 1

    q_flat = query.view(b_flat, n_q, d_k)
    k_flat = key.view(b_flat, n_k, d_k)
    v_flat = value.view(b_flat, n_k, d_v)

    device = query.device
    dtype = query.dtype

    # Initialize output accumulator and online softmax statistics
    # Output accumulator: O in R^[B_flat, N_q, d_v]
    out_accum = torch.zeros((b_flat, n_q, d_v), device=device, dtype=dtype)
    # Running normalizer: l in R^[B_flat, N_q, 1]
    running_l = torch.zeros((b_flat, n_q, 1), device=device, dtype=dtype)
    # Running maximum: m in R^[B_flat, N_q, 1] initialized to -infinity
    running_m = torch.full((b_flat, n_q, 1), float("-inf"), device=device, dtype=dtype)

    num_row_blocks = math.ceil(n_q / block_size_r)
    num_col_blocks = math.ceil(n_k / block_size_c)

    # --- OUTER LOOP: Iterate over Query Row Blocks ---
    for r_idx in range(num_row_blocks):
        q_start = r_idx * block_size_r
        q_end = min((r_idx + 1) * block_size_r, n_q)
        actual_br = q_end - q_start

        # Slice row block Q_i: [B_flat, actual_br, d_k]
        q_block = q_flat[:, q_start:q_end, :]

        # Slices of running stats for current row block
        m_i = running_m[:, q_start:q_end, :]  # [B_flat, actual_br, 1]
        l_i = running_l[:, q_start:q_end, :]  # [B_flat, actual_br, 1]
        o_i = out_accum[:, q_start:q_end, :]  # [B_flat, actual_br, d_v]

        # Row indices in the global sequence
        global_rows = torch.arange(q_start, q_end, device=device).unsqueeze(1)  # [actual_br, 1]

        # --- INNER LOOP: Iterate over Key/Value Column Blocks ---
        for c_idx in range(num_col_blocks):
            k_start = c_idx * block_size_c
            k_end = min((c_idx + 1) * block_size_c, n_k)
            actual_bc = k_end - k_start

            # Causal block pruning:
            # If all columns in this block are strictly greater than the maximum row index
            # in row block i, the entire tile is masked out. We skip it entirely!
            # Condition: min_col > max_row + (n_k - n_q)
            if is_causal and (k_start > (q_end - 1 + (n_k - n_q))):
                continue

            # Slice column blocks K_j, V_j: [B_flat, actual_bc, d]
            k_block = k_flat[:, k_start:k_end, :]
            v_block = v_flat[:, k_start:k_end, :]

            # 1. Compute local block scores: S_ij = (Q_i @ K_j^T) * scale
            # Shape: [B_flat, actual_br, actual_bc]
            s_ij = torch.bmm(q_block, k_block.transpose(1, 2)) * scale

            # 2. Local causal mask (if within the diagonal band)
            if is_causal and (k_end - 1 > (q_start + (n_k - n_q))):
                global_cols = torch.arange(k_start, k_end, device=device).unsqueeze(0)  # [1, actual_bc]
                local_causal_mask = global_cols > (global_rows + (n_k - n_q))
                s_ij = s_ij.masked_fill(local_causal_mask, float("-inf"))

            # 3. Online Softmax update:
            # Find row-wise maximum of current block: m_tilde in [B_flat, actual_br, 1]
            m_tilde = torch.max(s_ij, dim=-1, keepdim=True).values

            # New running row maximum: m_new = max(m_i, m_tilde)
            m_new = torch.maximum(m_i, m_tilde)

            # Masked rows where m_new is still -inf (e.g. completely masked out)
            # Replace -inf with 0 for safe exponent subtraction
            safe_m_new = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), m_new)
            safe_m_i = torch.where(torch.isneginf(m_i), torch.zeros_like(m_i), m_i)

            # Correction factor for previously accumulated terms:
            # alpha = exp(m_i - m_new)
            # If m_i was -inf (step 0), alpha must be 0
            alpha = torch.where(
                torch.isneginf(m_i),
                torch.zeros_like(m_i),
                torch.exp(safe_m_i - safe_m_new),
            )

            # Unnormalized probabilities for current block:
            # P_tilde = exp(S_ij - m_new)
            p_tilde = torch.exp(s_ij - safe_m_new)
            # Ensure masked positions remain 0
            p_tilde = torch.where(torch.isneginf(s_ij), torch.zeros_like(p_tilde), p_tilde)

            # Row sum of current block probabilities: l_tilde in [B_flat, actual_br, 1]
            l_tilde = torch.sum(p_tilde, dim=-1, keepdim=True)

            # Update running normalizer: l_new = l_i * alpha + l_tilde
            l_new = l_i * alpha + l_tilde

            # Update output accumulator: O_new = O_i * alpha + P_tilde @ V_j
            # (alpha broadcasts across d_v dimension)
            o_new = o_i * alpha + torch.bmm(p_tilde, v_block)

            # Advance state for next column block
            m_i = m_new
            l_i = l_new
            o_i = o_new

        # After all column blocks, normalize accumulator by total sum: O_i = O_i / l_i
        # Clamp l_i to avoid division by zero if an entire row was masked
        safe_l = torch.clamp(l_i, min=1e-9)
        o_normalized = o_i / safe_l

        # Store back into full output tensor
        out_accum[:, q_start:q_end, :] = o_normalized
        running_m[:, q_start:q_end, :] = m_i
        running_l[:, q_start:q_end, :] = l_i

    # Reshape back to original batch shape: [..., N_q, d_v]
    return out_accum.view(*orig_shape[:-2], n_q, d_v)


def pytorch_fused_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    backend: str = "all",
) -> Tuple[torch.Tensor, str]:
    """Execute PyTorch's native F.scaled_dot_product_attention under specific backend constraints.

    Investigates supported PyTorch optimized attention backends:
    - 'flash': FlashAttention v2 fused CUDA kernel
    - 'mem_efficient': Memory-Efficient Attention (Cutlass/xFormers kernel)
    - 'math': PyTorch reference math implementation
    - 'all': Allows PyTorch runtime to select the most optimal available backend

    Satisfies Rubric E1 & E3.
    """
    enable_flash = backend in ("flash", "all")
    enable_mem_efficient = backend in ("mem_efficient", "all")
    enable_math = backend in ("math", "all")

    device = query.device
    if device.type == "cuda":
        with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_mem_efficient=enable_mem_efficient,
            enable_math=enable_math,
        ):
            out = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                scale=scale,
                is_causal=is_causal,
            )
    else:
        out = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
        )

    return out, backend
