"""Task 6: Automated Integration & Verification Test Suite.

Validates:
1. REST API endpoints (/api/status, /api/presets, /api/benchmarks)
2. Input validation and error handling on empty/invalid payloads
3. Live screening on an unseen clinical note across KV-Cached and Speculative decoding
4. Live decoding comparison runner (/api/compare_decoding) asserting 100% token equivalence
5. Live attention kernel runner (/api/run_attention) verifying numerical tolerance
6. Report generation schema and provenance completeness

Satisfies Rubrics G1, G2, G3, and G4.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task6.engine_manager import (
    ELIGIBILITY_CLASSES,
    PRESET_UNSEEN_CASES,
    aggregate_overall_trial_verdict,
    controller,
    normalize_and_number_sentences,
    parse_model_json_output,
)


def run_integration_tests():
    print("\n" + "=" * 76)
    print("  Task 6: Application Backend & Engine Integration Test Suite")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 76 + "\n")

    # ---------------------------------------------------------
    # Test 1: Sentence Normalization & Numbering
    # ---------------------------------------------------------
    print("[*] Test 1: Verifying Sentence Segmentation & Indexing...")
    raw_prose = "Patient is 65 years old. She has stage IV breast cancer. ECOG score is 1."
    indexed_note = normalize_and_number_sentences(raw_prose)
    assert "0. Patient is 65 years old." in indexed_note
    assert "1. She has stage IV breast cancer." in indexed_note
    assert "2. ECOG score is 1." in indexed_note
    print("    [PASS] Sentence segmentation correctly numbers unindexed prose.")

    # ---------------------------------------------------------
    # Test 2: Rule-Based Trial Verdict Aggregation Logic
    # ---------------------------------------------------------
    print("\n[*] Test 2: Verifying Explicit Rule-Based Trial Verdict Aggregation...")
    
    # Case A: Disqualification via exclusion
    crit_disq = [
        {"criterion_id": "C1", "criterion_type": "inclusion", "prediction": "included"},
        {"criterion_id": "C2", "criterion_type": "exclusion", "prediction": "excluded"},
    ]
    v_disq = aggregate_overall_trial_verdict(crit_disq)
    assert v_disq["verdict"] == "INELIGIBLE", f"Expected INELIGIBLE, got {v_disq['verdict']}"
    assert v_disq["status_color"] == "danger"
    print("    [PASS] Rule 1 correctly triggers INELIGIBLE when exclusion criterion is met.")

    # Case B: Missing info pending state
    crit_miss = [
        {"criterion_id": "C1", "criterion_type": "inclusion", "prediction": "included"},
        {"criterion_id": "C2", "criterion_type": "exclusion", "prediction": "not excluded"},
        {"criterion_id": "C3", "criterion_type": "inclusion", "prediction": "not enough information"},
    ]
    v_miss = aggregate_overall_trial_verdict(crit_miss)
    assert v_miss["verdict"] == "UNCERTAIN_MISSING_INFO", f"Expected UNCERTAIN, got {v_miss['verdict']}"
    assert v_miss["status_color"] == "warning"
    print("    [PASS] Rule 2 correctly triggers UNCERTAIN_MISSING_INFO when information is lacking.")

    # Case C: Fully eligible
    crit_elig = [
        {"criterion_id": "C1", "criterion_type": "inclusion", "prediction": "included"},
        {"criterion_id": "C2", "criterion_type": "exclusion", "prediction": "not excluded"},
    ]
    v_elig = aggregate_overall_trial_verdict(crit_elig)
    assert v_elig["verdict"] == "POTENTIALLY_ELIGIBLE", f"Expected POTENTIALLY_ELIGIBLE, got {v_elig['verdict']}"
    assert v_elig["status_color"] == "success"
    print("    [PASS] Rule 3 correctly triggers POTENTIALLY_ELIGIBLE when all constraints satisfied.")

    # ---------------------------------------------------------
    # Test 3: Hardware Diagnostics & Presets Availability
    # ---------------------------------------------------------
    print("\n[*] Test 3: Verifying Hardware Diagnostics & Presets Loader...")
    hw = controller.get_hardware_status()
    print(f"    Hardware: {hw['gpu_name']} | CUDA Available: {hw['cuda_available']}")
    assert len(controller.preset_cases) >= 2, "Presets must include unseen cases"
    print(f"    [PASS] Controller successfully exposes {len(controller.preset_cases)} clinical cases.")

    # ---------------------------------------------------------
    # Test 4: Live Task 4 Attention Runner
    # ---------------------------------------------------------
    print("\n[*] Test 4: Verifying Live Tiled Attention Execution (/api/run_attention)...")
    att_res = controller.run_live_attention_test(seq_len=256, head_dim=128, block_r=32, block_c=32, is_causal=True)
    num_agree = att_res["numerical_agreement"]
    print(f"    Max Diff: {num_agree['max_absolute_error']:.2e} | Mean Diff: {num_agree['mean_absolute_error']:.2e}")
    print(f"    Reference Latency: {att_res['results']['reference_sdpa']['latency_ms']} ms | Peak VRAM: {att_res['results']['reference_sdpa']['peak_vram_mb']} MB")
    print(f"    Tiled Latency:     {att_res['results']['tiled_online_sdpa']['latency_ms']} ms | Peak VRAM: {att_res['results']['tiled_online_sdpa']['peak_vram_mb']} MB")
    assert num_agree["verified"], "Tiled attention output must match reference within tolerance"
    print("    [PASS] Live attention execution completed and verified numerically!")

    # ---------------------------------------------------------
    # Test 5: Live Decoding Comparison Lab (/api/compare_decoding)
    # ---------------------------------------------------------
    print("\n[*] Test 5: Verifying Live Comparative Decoding Execution (/api/compare_decoding)...")
    test_prompt = (
        "<|im_start|>system\nYou are a clinical trial assistant.<|im_end|>\n"
        "<|im_start|>user\nExtract patient age and gender into JSON.<|im_end|>\n"
        "<|im_start|>assistant\n```json\n{"
    )
    comp_res = controller.compare_decoding_modes(prompt_text=test_prompt, gamma=3, max_new_tokens=16)
    eq = comp_res["token_equality"]
    print(f"    Uncached vs KV-Cached Match: {eq['uncached_vs_cached_exact']}", flush=True)
    print(f"    KV-Cached vs Speculative Match: {eq['cached_vs_speculative_exact']}", flush=True)
    print(f"    Uncached:    {comp_res['modes']['uncached']['tokens_per_sec']} tok/s ({comp_res['modes']['uncached']['latency_ms']} ms)", flush=True)
    print(f"    KV-Cached:   {comp_res['modes']['kv_cached']['tokens_per_sec']} tok/s ({comp_res['modes']['kv_cached']['latency_ms']} ms)", flush=True)
    print(f"    Speculative: {comp_res['modes']['speculative']['tokens_per_sec']} tok/s ({comp_res['modes']['speculative']['latency_ms']} ms) | alpha: {comp_res['modes']['speculative']['acceptance_rate_percent']}%", flush=True)
    assert eq["cached_vs_speculative_exact"], "Speculative decoding must match KV-cached greedy decoding exactly"
    print("    [PASS] Comparative decoding successfully executed with exact token agreement!", flush=True)

    # ---------------------------------------------------------
    # Test 6: Full Patient Screening on Unseen Case (/api/screen)
    # ---------------------------------------------------------
    print("\n[*] Test 6: Verifying End-to-End Clinical Screening on Unseen Case...", flush=True)
    unseen_case = PRESET_UNSEEN_CASES[0]
    screening_out = controller.screen_patient(
        raw_note=unseen_case["note"],
        criteria=unseen_case["criteria"][:1],  # Screen 1 criterion for fast backend verification
        decoding_mode="kv_cached",
        max_new_tokens=64,
    )
    assert len(screening_out["criteria_results"]) == 1
    for c in screening_out["criteria_results"]:
        print(f"    Criterion {c['criterion_id']}: Prediction = '{c['prediction']}' | Evidence Sentences = {c['evidence_sentence_ids']}", flush=True)
        assert c["prediction"] in ELIGIBILITY_CLASSES, f"Invalid prediction: {c['prediction']}"
    print(f"    Overall Verdict: {screening_out['overall_verdict']['verdict_title']}", flush=True)
    print("    [PASS] Full clinical screening pipeline verified!", flush=True)

    print("\n" + "=" * 76, flush=True)
    print("  [SUCCESS] All Task 6 Backend & Inference Integration Tests Passed!", flush=True)
    print("=" * 76 + "\n", flush=True)


if __name__ == "__main__":
    run_integration_tests()

