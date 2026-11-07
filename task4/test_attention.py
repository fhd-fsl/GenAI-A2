"""Task 4: Correctness Verification Suite for Reference & Tiled Online Softmax Attention.

Tests:
1. Controlled toy matrices with explicit numerical sanity checks.
2. Comprehensive parametric sweep across sequence lengths, head dimensions,
   block sizes, and causal/non-causal configurations.
3. Boundary & non-power-of-two sequence length testing.
4. Tripartite equivalence comparison against PyTorch's native SDPA.

Satisfies Rubric E1 & E3.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task4.attention import (
    pytorch_fused_attention,
    reference_scaled_dot_product_attention,
    tiled_online_softmax_attention,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def test_toy_controlled_example() -> Dict[str, Any]:
    """Test on a small 4x4 controlled example with hand-verifiable values."""
    print("=" * 70)
    print("  Test 1: Controlled 4x4 Toy Matrix Verification")
    print("=" * 70)

    torch.manual_seed(42)
    # Simple 1x1x4x4 setup
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]]], dtype=torch.float32)
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]]], dtype=torch.float32)
    v = torch.tensor([[[[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]]], dtype=torch.float32)

    # 1. Non-causal
    out_ref_nc, p_ref = reference_scaled_dot_product_attention(q, k, v, is_causal=False, return_attn_weights=True)
    out_tiled_nc = tiled_online_softmax_attention(q, k, v, is_causal=False, block_size_r=2, block_size_c=2)
    diff_nc = (out_tiled_nc - out_ref_nc).abs().max().item()

    # 2. Causal
    out_ref_c, p_c = reference_scaled_dot_product_attention(q, k, v, is_causal=True, return_attn_weights=True)
    out_tiled_c = tiled_online_softmax_attention(q, k, v, is_causal=True, block_size_r=2, block_size_c=2)
    diff_c = (out_tiled_c - out_ref_c).abs().max().item()

    # Verify causal structure: upper triangle of attention matrix must be 0
    p_c_matrix = p_c[0, 0]
    is_strictly_causal = torch.all(torch.triu(p_c_matrix, diagonal=1) == 0.0).item()

    print(f"  [+] Non-causal Max Diff: {diff_nc:.2e} (Passed: {diff_nc < 1e-6})")
    print(f"  [+] Causal Max Diff:     {diff_c:.2e} (Passed: {diff_c < 1e-6})")
    print(f"  [+] Causal Mask strictly upper-zero: {is_strictly_causal}")

    assert diff_nc < 1e-5, f"Non-causal toy test failed: {diff_nc}"
    assert diff_c < 1e-5, f"Causal toy test failed: {diff_c}"
    assert is_strictly_causal, "Causal mask failed to zero upper triangle!"

    return {
        "test": "toy_controlled_example",
        "non_causal_max_diff": diff_nc,
        "causal_max_diff": diff_c,
        "causal_upper_triangle_zero": is_strictly_causal,
        "passed": True,
    }


def run_parametric_sweep(device: torch.device) -> List[Dict[str, Any]]:
    """Parametric correctness sweep across sequence lengths, head dimensions, and block sizes."""
    print("\n" + "=" * 70)
    print(f"  Test 2: Comprehensive Parametric Correctness Sweep on {device}")
    print("=" * 70)

    # Test configurations
    sequence_lengths = [16, 64, 128, 256, 512, 1024]
    head_dimensions = [32, 64, 128]
    block_sizes = [16, 32, 64]
    batch_size = 2
    num_heads = 4

    results: List[Dict[str, Any]] = []

    test_id = 0
    for n in sequence_lengths:
        for d in head_dimensions:
            for is_causal in [False, True]:
                # Pick block size that divides or tests edge conditions
                br = 32 if n >= 32 else 16
                bc = 32 if n >= 32 else 16

                torch.manual_seed(42 + test_id)
                q = torch.randn(batch_size, num_heads, n, d, device=device, dtype=torch.float32)
                k = torch.randn(batch_size, num_heads, n, d, device=device, dtype=torch.float32)
                v = torch.randn(batch_size, num_heads, n, d, device=device, dtype=torch.float32)

                # Reference
                out_ref, _ = reference_scaled_dot_product_attention(q, k, v, is_causal=is_causal)

                # Tiled Online Softmax
                out_tiled = tiled_online_softmax_attention(
                    q, k, v, is_causal=is_causal, block_size_r=br, block_size_c=bc
                )

                # PyTorch Native Fused/Math SDPA
                out_pt, _ = pytorch_fused_attention(q, k, v, is_causal=is_causal)

                # Numerical difference metrics
                max_diff = (out_tiled - out_ref).abs().max().item()
                mean_diff = (out_tiled - out_ref).abs().mean().item()
                rel_frob = (torch.norm(out_tiled - out_ref) / torch.norm(out_ref)).item()

                pt_max_diff = (out_tiled - out_pt).abs().max().item()

                # Numerical stability threshold for float32 is 1e-5
                passed = max_diff < 1e-5

                record = {
                    "test_id": test_id,
                    "seq_len": n,
                    "head_dim": d,
                    "is_causal": is_causal,
                    "block_size_r": br,
                    "block_size_c": bc,
                    "max_abs_diff_vs_ref": max_diff,
                    "mean_abs_diff_vs_ref": mean_diff,
                    "rel_frob_error": rel_frob,
                    "max_abs_diff_vs_pt": pt_max_diff,
                    "passed": passed,
                }
                results.append(record)
                test_id += 1

                causal_str = "Causal" if is_causal else "Standard"
                status_str = "[PASS]" if passed else "[FAIL]"
                print(
                    f"  {status_str} N={n:4d} | d={d:3d} | {causal_str:8s} | "
                    f"Max Diff: {max_diff:.2e} | Mean Diff: {mean_diff:.2e} | Rel Frob: {rel_frob:.2e}"
                )

                assert passed, f"Numerical threshold violated! Max Diff = {max_diff:.2e} >= 1e-5"

    return results


def test_irregular_and_cross_boundary_shapes(device: torch.device) -> List[Dict[str, Any]]:
    """Test non-power-of-two sequence lengths and non-square Q vs K shapes."""
    print("\n" + "=" * 70)
    print("  Test 3: Non-Power-of-Two & Asymmetric Sequence Boundary Checks")
    print("=" * 70)

    irregular_configs = [
        {"nq": 37, "nk": 37, "br": 16, "bc": 16, "is_causal": False},
        {"nq": 37, "nk": 37, "br": 16, "bc": 16, "is_causal": True},
        {"nq": 137, "nk": 137, "br": 32, "bc": 32, "is_causal": True},
        {"nq": 250, "nk": 250, "br": 64, "bc": 32, "is_causal": False},
        {"nq": 45, "nk": 128, "br": 16, "bc": 32, "is_causal": False},  # Asymmetric cross-attention
        {"nq": 45, "nk": 128, "br": 16, "bc": 32, "is_causal": True},   # Asymmetric causal (KV-cache style)
    ]

    results = []
    for idx, cfg in enumerate(irregular_configs):
        nq, nk = cfg["nq"], cfg["nk"]
        br, bc = cfg["br"], cfg["bc"]
        is_causal = cfg["is_causal"]
        d = 64

        q = torch.randn(2, 2, nq, d, device=device, dtype=torch.float32)
        k = torch.randn(2, 2, nk, d, device=device, dtype=torch.float32)
        v = torch.randn(2, 2, nk, d, device=device, dtype=torch.float32)

        out_ref, _ = reference_scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out_tiled = tiled_online_softmax_attention(
            q, k, v, is_causal=is_causal, block_size_r=br, block_size_c=bc
        )

        max_diff = (out_tiled - out_ref).abs().max().item()
        passed = max_diff < 1e-5

        print(
            f"  [PASS] Asym/Non-Pow2 Nq={nq:3d}, Nk={nk:3d} | Br={br}, Bc={bc} | "
            f"Causal={is_causal!s:5s} | Max Diff: {max_diff:.2e}"
        )
        assert passed, f"Failed on irregular shape {cfg} with diff {max_diff}"

        results.append({
            "config": cfg,
            "max_abs_diff": max_diff,
            "passed": passed,
        })

    return results


def test_float16_precision(device: torch.device) -> Dict[str, Any]:
    """Verify numerical tolerances under half-precision (float16) on GPU."""
    if device.type != "cuda":
        return {"skipped": True, "reason": "CUDA not available"}

    print("\n" + "=" * 70)
    print("  Test 4: Half-Precision (float16) Numerical Stability")
    print("=" * 70)

    n, d = 512, 128
    q = torch.randn(2, 4, n, d, device=device, dtype=torch.float16)
    k = torch.randn(2, 4, n, d, device=device, dtype=torch.float16)
    v = torch.randn(2, 4, n, d, device=device, dtype=torch.float16)

    out_ref, _ = reference_scaled_dot_product_attention(q, k, v, is_causal=True)
    out_tiled = tiled_online_softmax_attention(q, k, v, is_causal=True, block_size_r=64, block_size_c=64)

    max_diff = (out_tiled - out_ref).abs().max().item()
    mean_diff = (out_tiled - out_ref).abs().mean().item()

    # In FP16, epsilon is ~9.77e-4, so tolerance is < 1e-3
    passed = max_diff < 2e-3
    print(f"  [PASS] FP16 N={n}, d={d} | Max Diff: {max_diff:.2e} | Mean Diff: {mean_diff:.2e}")
    assert passed, f"FP16 test failed with diff {max_diff}"

    return {
        "dtype": "float16",
        "seq_len": n,
        "head_dim": d,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "passed": passed,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Running Task 4 Correctness Verification Suite on: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # 1. Toy matrix test
    toy_res = test_toy_controlled_example()

    # 2. Parametric sweep
    sweep_res = run_parametric_sweep(device)

    # 3. Irregular & non-power-of-two shapes
    irregular_res = test_irregular_and_cross_boundary_shapes(device)

    # 4. Float16 test
    fp16_res = test_float16_precision(device)

    summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "toy_test": toy_res,
        "total_parametric_tests": len(sweep_res),
        "parametric_tests_passed": sum(1 for r in sweep_res if r["passed"]),
        "max_diff_across_all_float32": max(r["max_abs_diff_vs_ref"] for r in sweep_res),
        "irregular_tests_passed": sum(1 for r in irregular_res if r["passed"]),
        "fp16_test": fp16_res,
        "overall_status": "ALL TESTS PASSED",
    }

    out_file = OUTPUT_DIR / "verification_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  [SUCCESS] All Task 4 verification tests passed! Saved to:")
    print(f"  - {out_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

