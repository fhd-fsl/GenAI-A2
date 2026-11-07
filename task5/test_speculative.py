"""Task 5: Correctness Verification Suite for Speculative Decoding.

Validates that Greedy Speculative Decoding produces 100.00% token-level equivalence
against standard Target Greedy Decoding across diverse clinical prompts and
speculative lookahead depths (gamma in {3, 5, 7}).

Satisfies Rubric F2 & F3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task3.kv_cache import generate_with_kv_cache
from task3.test_correctness import PROMPTS
from task5.speculative_decoding import (
    DEFAULT_ADAPTER_PATH,
    DRAFT_MODEL_ID,
    TARGET_MODEL_ID,
    generate_speculative_greedy,
    load_draft_and_target_models,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"

# Clinical trial prompts from Task 1, 2, 3
STRUCTURED_ELIGIBILITY_PROMPT = """<|im_start|>system
You are a clinical trial eligibility expert. Given a patient description and a criterion, determine if the patient meets the criterion.
Return a valid JSON object with:
"prediction": one of "included", "not included", "not excluded", "excluded", "not enough information", "not applicable"
"explanation": a short justification.<|im_end|>
<|im_start|>user
Patient Description:
The patient is a 62-year-old female diagnosed with stage III breast cancer. Eastern Cooperative Oncology Group (ECOG) performance status is 1. Serum creatinine is 1.1 mg/dL. Absolute neutrophil count is 1800/mcL.

Criterion Type: inclusion
Criterion Text: ECOG performance status 0 or 1.<|im_end|>
<|im_start|>assistant
"""

TEST_PROMPTS = {
    "short": PROMPTS["short"],
    "medium": PROMPTS["medium"],
    "structured_clinical": STRUCTURED_ELIGIBILITY_PROMPT,
}

GAMMAS_TO_TEST = [3, 5, 7]
NEW_TOKENS_BUDGETS = [32, 64]


def run_correctness_verification():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 75, flush=True)
    print("  Task 5: Speculative Decoding Correctness Verification Suite", flush=True)
    print(f"  Device: {device} | Testing Token Equivalence with Target Greedy Reference", flush=True)
    print("=" * 75 + "\n", flush=True)

    target_model, draft_model, tokenizer = load_draft_and_target_models(device=device)

    test_records: List[Dict[str, Any]] = []
    total_tokens_evaluated = 0
    total_tokens_matched = 0
    all_exact_matches = True

    for p_name, p_text in TEST_PROMPTS.items():
        inputs = tokenizer(p_text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        for budget in NEW_TOKENS_BUDGETS:
            print(f"\n--- Prompt: {p_name.upper()} ({prompt_len} tokens) | Budget: {budget} tokens ---", flush=True)

            # 1. Generate Target Greedy Reference (Ground Truth)
            ref_res = generate_with_kv_cache(
                model=target_model,
                tokenizer=tokenizer,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=budget,
            )
            ref_tokens = ref_res.generated_ids
            ref_text = ref_res.generated_text

            for gamma in GAMMAS_TO_TEST:
                # 2. Generate with Greedy Speculative Decoding
                spec_res = generate_speculative_greedy(
                    target_model=target_model,
                    draft_model=draft_model,
                    tokenizer=tokenizer,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    gamma=gamma,
                    max_new_tokens=budget,
                )
                spec_tokens = spec_res.generated_ids
                spec_text = spec_res.generated_text

                # 3. Compare token-by-token
                min_len = min(len(ref_tokens), len(spec_tokens))
                matches = sum(1 for a, b in zip(ref_tokens[:min_len], spec_tokens[:min_len]) if a == b)
                token_match_rate = matches / max(len(ref_tokens), 1)
                exact_match = (ref_tokens == spec_tokens)

                total_tokens_evaluated += len(ref_tokens)
                total_tokens_matched += matches
                if not exact_match:
                    all_exact_matches = False

                first_mismatch_idx = None
                if not exact_match:
                    for idx, (a, b) in enumerate(zip(ref_tokens, spec_tokens)):
                        if a != b:
                            first_mismatch_idx = idx
                            break

                status_str = "[PASS]" if exact_match else "[FAIL]"
                print(
                    f"  {status_str} gamma={gamma:1d} | Tokens: {len(spec_tokens):2d}/{len(ref_tokens):2d} | "
                    f"Agreement: {token_match_rate*100:6.2f}% | "
                    f"Acceptance Rate: {spec_res.draft_acceptance_rate*100:5.1f}% | "
                    f"Mean Acc/Step: {spec_res.mean_accepted_per_step:4.2f} | "
                    f"Fwd Passes Saved: {spec_res.target_forward_passes_saved:2d}",
                    flush=True,
                )

                test_records.append({
                    "prompt_name": p_name,
                    "prompt_length": prompt_len,
                    "budget_new_tokens": budget,
                    "gamma": gamma,
                    "ref_tokens_count": len(ref_tokens),
                    "spec_tokens_count": len(spec_tokens),
                    "exact_match": exact_match,
                    "token_match_rate": round(token_match_rate, 4),
                    "first_mismatch_step": first_mismatch_idx,
                    "draft_acceptance_rate": spec_res.draft_acceptance_rate,
                    "mean_accepted_per_step": spec_res.mean_accepted_per_step,
                    "target_forward_passes_saved": spec_res.target_forward_passes_saved,
                    "total_verification_steps": spec_res.num_verification_steps,
                    "ref_text_sample": ref_text[:80],
                    "spec_text_sample": spec_text[:80],
                })

                assert exact_match, (
                    f"Speculative decoding token mismatch at index {first_mismatch_idx}!\n"
                    f"Ref:  {ref_tokens}\n"
                    f"Spec: {spec_tokens}"
                )

    overall_agreement = total_tokens_matched / max(total_tokens_evaluated, 1)

    verification_summary = {
        "model_pair": {
            "target": TARGET_MODEL_ID,
            "draft": DRAFT_MODEL_ID,
            "adapter": DEFAULT_ADAPTER_PATH,
        },
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_configurations_tested": len(test_records),
        "configurations_passed": sum(1 for r in test_records if r["exact_match"]),
        "all_configurations_passed": all_exact_matches,
        "total_tokens_evaluated": total_tokens_evaluated,
        "total_tokens_matched": total_tokens_matched,
        "overall_token_agreement": round(overall_agreement, 4),
        "test_results": test_records,
    }

    out_file = OUTPUT_DIR / "correctness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(verification_summary, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print(f"  [SUCCESS] All Speculative Correctness Tests Passed (100% Token Equivalence)!")
    print(f"  Saved verification metrics to: {out_file}")
    print("=" * 75 + "\n", flush=True)

    return verification_summary


if __name__ == "__main__":
    run_correctness_verification()

