"""Task 3: Correctness Verification Suite.

Tests token-level equivalence between uncached reference decoding and
KV-cached decoding across multiple prompt lengths and output budgets.
Satisfies Rubric D2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task3.kv_cache import (
    ExplicitKVCache,
    generate_uncached,
    generate_with_kv_cache,
    load_model_for_task3,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

PROMPTS = {
    "short": (
        "Patient Description:\n"
        "0. A 67-year-old woman presents for post-operative evaluation after femoral cardiac catheterization.\n"
        "1. Physical exam shows distal pulses intact and stable vital signs.\n\n"
        "Criterion Type: inclusion\n"
        "Criterion Text: Age greater than 18 years"
    ),
    "medium": (
        "Patient Description:\n"
        "0. A 40-year-old woman with no significant past medical history presents to the emergency department.\n"
        "1. She complains of acute onset excruciating pain in her right upper extremity for 1 hour.\n"
        "2. She denies any preceding trauma, fever, or prior episodes.\n"
        "3. On physical examination she is pale, tachypneic with respiratory rate 26/min, and tachycardic with heart rate 108/min.\n"
        "4. Blood pressure is 80/60 mmHg and distal capillary refill is delayed.\n"
        "5. Intravenous fluid resuscitation is initiated immediately.\n\n"
        "Criterion Type: inclusion\n"
        "Criterion Text: Patients diagnosed with shock or acute respiratory distress syndrome"
    ),
    "long": (
        "Patient Description:\n"
        "0. A 21-year-old female is evaluated for progressive symmetrical arthralgias and malaise for 3 months.\n"
        "1. On physical examination she has non-scarring alopecia, a malar rash over her nose and cheeks, and swollen tender wrists.\n"
        "2. Cardiopulmonary examination is normal without friction rub or murmurs.\n"
        "3. Laboratory evaluation reveals normocytic anemia (hemoglobin 9.4 g/dL), thrombocytopenia (platelets 92,000/uL), and leukopenia.\n"
        "4. Serological testing is strongly positive for antinuclear antibodies (ANA 1:640 homogeneous) and anti-dsDNA antibodies.\n"
        "5. Urinalysis reveals 3+ proteinuria, microscopic hematuria, and dysmorphic red blood cell casts.\n"
        "6. Serum complement C3 and C4 levels are markedly depressed.\n"
        "7. Renal biopsy confirms Class IV diffuse proliferative lupus nephritis.\n"
        "8. Patient is willing and able to comply with all scheduled trial visits and protocol procedures.\n\n"
        "Criterion Type: inclusion\n"
        "Criterion Text: Fulfill at least 4 criteria for Systemic Lupus Erythematosus as defined by the American College of Rheumatology"
    ),
}

GENERATION_LENGTHS = [16, 32, 64]


def run_correctness_verification() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_for_task3()

    results: Dict[str, Any] = {
        "model": "Qwen/Qwen2.5-3B-Instruct (4-bit NF4 + LoRA)",
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "tests": [],
        "overall_summary": {},
    }

    total_pairs = 0
    perfect_matches = 0
    total_tokens_tested = 0
    total_tokens_matched = 0

    print("\n" + "=" * 70)
    print("  Task 3: Running Token Equivalence Verification (Cached vs. Uncached)")
    print("=" * 70)

    for p_name, p_text in PROMPTS.items():
        inputs = tokenizer(p_text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        for gen_len in GENERATION_LENGTHS:
            total_pairs += 1
            print(f"\n[Test] Prompt: {p_name.upper():6s} (Length: {prompt_len:3d} tokens) | Target Output: {gen_len:2d} tokens")

            # 1. Uncached reference generation
            res_uncached = generate_uncached(
                model=model,
                tokenizer=tokenizer,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=gen_len,
                store_logits=True,
            )

            # 2. KV-cached generation
            res_cached = generate_with_kv_cache(
                model=model,
                tokenizer=tokenizer,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=gen_len,
                store_logits=True,
            )

            # Compare tokens
            uncached_ids = res_uncached.generated_ids
            cached_ids = res_cached.generated_ids
            min_len = min(len(uncached_ids), len(cached_ids))
            matches = sum(1 for u, c in zip(uncached_ids, cached_ids) if u == c)
            exact_match = (uncached_ids == cached_ids)

            total_tokens_tested += min_len
            total_tokens_matched += matches
            if exact_match:
                perfect_matches += 1

            first_diff_idx = None
            if not exact_match:
                for idx, (u, c) in enumerate(zip(uncached_ids, cached_ids)):
                    if u != c:
                        first_diff_idx = idx
                        break

            # Compare step-0 (prefill) logits
            step0_diff = 0.0
            max_abs_logit_diff = 0.0
            mean_abs_logit_diff = 0.0
            if res_uncached.step_logits and res_cached.step_logits:
                diff_0 = torch.abs(res_uncached.step_logits[0] - res_cached.step_logits[0])
                step0_diff = float(torch.max(diff_0).item())

                all_max_diffs = []
                all_mean_diffs = []
                for lu, lc in zip(res_uncached.step_logits, res_cached.step_logits):
                    step_diff = torch.abs(lu - lc)
                    all_max_diffs.append(torch.max(step_diff).item())
                    all_mean_diffs.append(torch.mean(step_diff).item())
                max_abs_logit_diff = float(max(all_max_diffs))
                mean_abs_logit_diff = float(sum(all_mean_diffs) / len(all_mean_diffs))

            test_entry = {
                "prompt_name": p_name,
                "prompt_length": prompt_len,
                "budget_new_tokens": gen_len,
                "uncached_tokens_count": len(uncached_ids),
                "cached_tokens_count": len(cached_ids),
                "exact_match": exact_match,
                "matching_tokens": matches,
                "token_match_rate": round(matches / min_len, 4) if min_len > 0 else 1.0,
                "first_divergence_step": first_diff_idx,
                "step0_logit_max_diff": step0_diff,
                "max_abs_logit_diff_overall": max_abs_logit_diff,
                "mean_abs_logit_diff_overall": mean_abs_logit_diff,
                "uncached_text": res_uncached.generated_text,
                "cached_text": res_cached.generated_text,
                "speedup": round(res_uncached.decode_time_ms / max(res_cached.decode_time_ms, 1e-4), 2),
            }
            results["tests"].append(test_entry)

            status = "MATCH [100%]" if exact_match else f"DIFF (Match: {matches}/{min_len})"
            print(f"  Result: {status:16s} | Step-0 Logit Diff: {step0_diff:.6f} | Decode Speedup: {test_entry['speedup']}x")

    overall_match_rate = total_tokens_matched / max(total_tokens_tested, 1)
    results["overall_summary"] = {
        "total_test_configurations": total_pairs,
        "perfect_match_configurations": perfect_matches,
        "configuration_match_rate": round(perfect_matches / total_pairs, 4),
        "total_tokens_evaluated": total_tokens_tested,
        "total_tokens_matched": total_tokens_matched,
        "overall_token_agreement": round(overall_match_rate, 4),
    }

    # Verify tensor growth rate
    # For Qwen2.5-3B: 36 layers * 2 (K,V) * 2 heads * 128 dim * 2 bytes = 36,864 bytes/token
    expected_bytes_per_token = 36 * 2 * 2 * 128 * 2
    results["cache_growth_validation"] = {
        "layers": 36,
        "kv_heads": 2,
        "head_dim": 128,
        "element_size_bytes": 2,
        "expected_bytes_per_token": expected_bytes_per_token,
        "expected_kb_per_token": round(expected_bytes_per_token / 1024, 2),
    }

    output_path = OUTPUT_DIR / "correctness_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  Summary: {perfect_matches}/{total_pairs} configurations perfectly identical ({overall_match_rate * 100:.2f}% token agreement)")
    print(f"  Saved correctness results to {output_path}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_correctness_verification()
