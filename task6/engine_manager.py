"""Task 6: Unified Inference & Benchmark Controller.

Integrates:
- Task 1 & 2: Qwen2.5-3B + LoRA Adapter, 6-class eligibility classification, evidence mapping
- Task 3: ExplicitKVCache, generate_with_kv_cache, generate_uncached
- Task 4: Reference SDPA, Tiled Online-Softmax Attention, PyTorch Fused SDPA
- Task 5: SpeculativeDecodingEngine (Qwen2.5-0.5B draft + Qwen2.5-3B target)
- Presets Loader: Loads held-out test split cases and unseen clinical notes
- Rule-based trial-level eligibility aggregation
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task1.preprocessing.tokenizer import (
    MAX_SEQ_LEN,
    SYSTEM_PROMPT,
    split_note_into_sentences,
)
from task3.kv_cache import ExplicitKVCache, generate_uncached, generate_with_kv_cache
from task4.attention import (
    pytorch_fused_attention,
    reference_scaled_dot_product_attention,
    tiled_online_softmax_attention,
)
from task5.speculative_decoding import (
    DEFAULT_ADAPTER_PATH,
    DRAFT_MODEL_ID,
    TARGET_MODEL_ID,
    SpeculativeDecodingEngine,
    generate_speculative_greedy,
    load_draft_and_target_models,
)

ELIGIBILITY_CLASSES = [
    "included",
    "not included",
    "not excluded",
    "excluded",
    "not enough information",
    "not applicable",
]

PRESET_UNSEEN_CASES = [
    {
        "id": "unseen-case-oncology",
        "title": "Unseen Case 1: Advanced Non-Small Cell Lung Cancer (NSCLC)",
        "patient_id": "unseen-pt-001",
        "trial_id": "NCT03829332",
        "trial_title": "Phase III Trial of Pembrolizumab with Chemotherapy in Metastatic NSCLC",
        "note": (
            "0. Patient is a 59-year-old male former smoker diagnosed with stage IV non-small cell lung cancer (adenocarcinoma subtype).\n"
            "1. Molecular profiling demonstrates EGFR wild-type and ALK rearrangement negative; PD-L1 expression TPS is 65%.\n"
            "2. Brain MRI shows no evidence of intracranial metastases.\n"
            "3. Eastern Cooperative Oncology Group (ECOG) performance status is documented as 1.\n"
            "4. Baseline laboratory evaluation: absolute neutrophil count is 2,400/mcL, platelets 185,000/mcL, hemoglobin 11.2 g/dL.\n"
            "5. Serum creatinine is 1.0 mg/dL with calculated creatinine clearance of 78 mL/min.\n"
            "6. Total bilirubin is 0.8 mg/dL and AST/ALT are within normal limits (22 U/L and 28 U/L).\n"
            "7. Patient has a past medical history of mild hypertension well controlled on amlodipine 5 mg daily.\n"
            "8. Patient has no history of autoimmune disease and has not received systemic corticosteroid therapy in the past 12 months."
        ),
        "criteria": [
            {
                "criterion_id": "NCT03829332_I_01",
                "criterion_type": "inclusion",
                "criterion_text": "Histologically or cytologically confirmed diagnosis of stage IV non-small cell lung cancer.",
            },
            {
                "criterion_id": "NCT03829332_I_02",
                "criterion_type": "inclusion",
                "criterion_text": "ECOG performance status 0 or 1.",
            },
            {
                "criterion_id": "NCT03829332_I_03",
                "criterion_type": "inclusion",
                "criterion_text": "Adequate organ function: absolute neutrophil count >= 1500/mcL, platelets >= 100,000/mcL, and serum creatinine <= 1.5 mg/dL.",
            },
            {
                "criterion_id": "NCT03829332_E_01",
                "criterion_type": "exclusion",
                "criterion_text": "Active or untreated central nervous system (CNS) metastases or leptomeningeal disease.",
            },
            {
                "criterion_id": "NCT03829332_E_02",
                "criterion_type": "exclusion",
                "criterion_text": "Active autoimmune disease requiring systemic treatment with corticosteroids or immunosuppressive agents.",
            },
            {
                "criterion_id": "NCT03829332_I_04",
                "criterion_type": "inclusion",
                "criterion_text": "Patient has signed the mandatory trial genetics sub-study authorization document.",
            },
        ],
    },
    {
        "id": "unseen-case-cardiology",
        "title": "Unseen Case 2: Heart Failure with Reduced Ejection Fraction (HFrEF)",
        "patient_id": "unseen-pt-002",
        "trial_id": "NCT04567890",
        "trial_title": "Evaluation of SGLT-2 Inhibitor in High-Risk HFrEF",
        "note": (
            "0. A 68-year-old female with long-standing ischemic cardiomyopathy presents for outpatient heart failure management.\n"
            "1. Transthoracic echocardiogram shows severe left ventricular systolic dysfunction with LVEF 28%.\n"
            "2. Patient reports NYHA functional class III dyspnea on walking one flight of stairs.\n"
            "3. Vital signs: blood pressure 114/72 mmHg, heart rate 68 bpm regular, BMI 27.4 kg/m2.\n"
            "4. Current medications include Sacubitril/Valsartan 49/51 mg BID and Carvedilol 25 mg BID.\n"
            "5. Serum potassium is 4.6 mEq/L and eGFR is 44 mL/min/1.73m2.\n"
            "6. Patient denies history of diabetic ketoacidosis or symptomatic orthostatic hypotension."
        ),
        "criteria": [
            {
                "criterion_id": "NCT04567890_I_01",
                "criterion_type": "inclusion",
                "criterion_text": "Documented left ventricular ejection fraction (LVEF) <= 35% within 6 months prior to screening.",
            },
            {
                "criterion_id": "NCT04567890_I_02",
                "criterion_type": "inclusion",
                "criterion_text": "New York Heart Association (NYHA) functional class II to IV heart failure symptoms.",
            },
            {
                "criterion_id": "NCT04567890_E_01",
                "criterion_type": "exclusion",
                "criterion_text": "Estimated glomerular filtration rate (eGFR) < 20 mL/min/1.73m2 at screening.",
            },
            {
                "criterion_id": "NCT04567890_E_02",
                "criterion_type": "exclusion",
                "criterion_text": "Systolic blood pressure < 95 mmHg or symptomatic hypotension.",
            },
            {
                "criterion_id": "NCT04567890_I_03",
                "criterion_type": "inclusion",
                "criterion_text": "Elevated NT-proBNP >= 600 pg/mL in sinus rhythm within 30 days.",
            },
        ],
    },
]


def format_clinical_prompt(note: str, criterion_type: str, criterion_text: str, tokenizer) -> str:
    """Format prompt using Qwen ChatML template matching fine-tuning."""
    user_prompt = (
        f"Patient Description:\n{note.strip()}\n\n"
        f"Criterion Type: {criterion_type.strip().lower()}\n"
        f"Criterion Text: {criterion_text.strip()}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def normalize_and_number_sentences(raw_note: str) -> str:
    """Ensure clinical note has 0-indexed sentence prefixes for evidence localization."""
    raw_note = raw_note.strip()
    if not raw_note:
        return ""
    # Check if already indexed like '0. Patient ...'
    lines = [line.strip() for line in raw_note.split("\n") if line.strip()]
    if len(lines) > 0 and re.match(r"^\d+\.\s+", lines[0]):
        return "\n".join(lines)

    # Otherwise segment sentences using standard punctuation boundaries and prefix with 0. 1. ...
    # Match sentences ending with period, exclamation, or question mark followed by space or newline
    sentence_spans = re.split(r"(?<=[.!?])\s+", raw_note)
    sentences = [s.strip() for s in sentence_spans if s.strip()]
    if not sentences:
        sentences = [raw_note]
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))


def parse_model_json_output(output_text: str, raw_note: str) -> Dict[str, Any]:
    """Parse JSON output from fine-tuned model, with fallbacks and verbatim evidence text extraction."""
    clean_text = output_text.strip()
    # Strip markdown code fences if model wrapped response
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    pred_data = None
    try:
        pred_data = json.loads(clean_text)
    except Exception:
        # Regex search for JSON block
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            try:
                pred_data = json.loads(json_match.group(0))
            except Exception:
                pass

    note_sentences = split_note_into_sentences(raw_note)

    if isinstance(pred_data, dict):
        prediction = str(pred_data.get("prediction", "")).strip().lower()
        if prediction not in ELIGIBILITY_CLASSES:
            # Match closest class
            for c in ELIGIBILITY_CLASSES:
                if c in prediction:
                    prediction = c
                    break
            else:
                prediction = "not enough information"

        # Evidence sentence IDs
        raw_ids = pred_data.get("evidence_sentence_ids", [])
        evidence_ids = []
        if isinstance(raw_ids, list):
            for x in raw_ids:
                try:
                    idx = int(x)
                    if 0 <= idx < len(note_sentences):
                        evidence_ids.append(idx)
                except (ValueError, TypeError):
                    continue

        evidence_text = [note_sentences[i] for i in evidence_ids]

        explanation = pred_data.get("explanation", "").strip()
        if not explanation:
            # Synthesize traceable explanation from evidence & prediction
            if prediction in ["included", "not excluded"] and evidence_text:
                explanation = f"Patient satisfies criterion based on documented findings in sentence(s) {evidence_ids}."
            elif prediction in ["not included", "excluded"] and evidence_text:
                explanation = f"Patient violates criterion based on findings documented in sentence(s) {evidence_ids}."
            elif prediction == "not enough information":
                explanation = "Clinical note lacks necessary laboratory values, diagnostic tests, or explicit findings to confirm or refute criterion."
            elif prediction == "not applicable":
                explanation = "Criterion describes an intervention, trial arm, or cohort constraint not relevant to this patient profile."
            else:
                explanation = f"Classification determined as '{prediction}' based on clinical evaluation of the patient record."

        return {
            "prediction": prediction,
            "evidence_sentence_ids": sorted(list(set(evidence_ids))),
            "evidence_text": evidence_text,
            "explanation": explanation,
            "raw_output": output_text,
            "parse_success": True,
        }

    # Fallback if JSON parsing completely failed
    lower_text = clean_text.lower()
    predicted_label = "not enough information"
    for c in ELIGIBILITY_CLASSES:
        if c in lower_text:
            predicted_label = c
            break

    return {
        "prediction": predicted_label,
        "evidence_sentence_ids": [],
        "evidence_text": [],
        "explanation": f"Model generated non-JSON output; detected category '{predicted_label}'. Raw output: {clean_text[:120]}...",
        "raw_output": output_text,
        "parse_success": False,
    }


def aggregate_overall_trial_verdict(criterion_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute explicit rule-based overall patient-trial screening aggregation.

    Rules:
    1. Disqualification Rule: If ANY inclusion is 'not included' OR ANY exclusion is 'excluded'
       -> INELIGIBLE (Disqualifying Criterion Met).
    2. Missing Information Rule: Else if ANY criterion is 'not enough information'
       -> PENDING / UNCERTAIN (Requires Additional Clinical Information).
    3. Eligible Rule: Else if all evaluated criteria are satisfied
       -> POTENTIALLY ELIGIBLE (All Screened Criteria Satisfied).
    """
    total = len(criterion_results)
    if total == 0:
        return {
            "verdict": "NO_CRITERIA",
            "verdict_title": "No Criteria Evaluated",
            "status_color": "neutral",
            "summary": "No criteria were provided for evaluation.",
            "rule_applied": "None",
            "counts": {"included": 0, "not_included": 0, "not_excluded": 0, "excluded": 0, "not_enough_info": 0, "not_applicable": 0},
        }

    counts = {
        "included": sum(1 for c in criterion_results if c["prediction"] == "included"),
        "not_included": sum(1 for c in criterion_results if c["prediction"] == "not included"),
        "not_excluded": sum(1 for c in criterion_results if c["prediction"] == "not excluded"),
        "excluded": sum(1 for c in criterion_results if c["prediction"] == "excluded"),
        "not_enough_info": sum(1 for c in criterion_results if c["prediction"] == "not enough information"),
        "not_applicable": sum(1 for c in criterion_results if c["prediction"] == "not applicable"),
    }

    # Rule 1: Disqualification
    disqualifying_criteria = []
    for c in criterion_results:
        c_type = c.get("criterion_type", "").lower()
        pred = c["prediction"]
        if (c_type == "inclusion" and pred == "not included") or (c_type == "exclusion" and pred == "excluded"):
            disqualifying_criteria.append(c)

    if disqualifying_criteria:
        d_ids = [c.get("criterion_id", "criterion") for c in disqualifying_criteria]
        return {
            "verdict": "INELIGIBLE",
            "verdict_title": "Ineligible (Disqualifying Criterion Met)",
            "status_color": "danger",
            "summary": f"Patient fails {len(disqualifying_criteria)} criterion/criteria ({', '.join(d_ids[:3])}).",
            "rule_applied": "Rule 1: Disqualification (Violation of mandatory inclusion or triggering of exclusion criterion)",
            "counts": counts,
            "disqualifying_count": len(disqualifying_criteria),
            "missing_info_count": counts["not_enough_info"],
        }

    # Rule 2: Missing Information
    if counts["not_enough_info"] > 0:
        return {
            "verdict": "UNCERTAIN_MISSING_INFO",
            "verdict_title": "Screening Inconclusive (Missing Information)",
            "status_color": "warning",
            "summary": f"{counts['not_enough_info']} criterion/criteria require additional clinical records, laboratory tests, or diagnostic confirmation.",
            "rule_applied": "Rule 2: Open-world uncertainty (One or more criteria returned 'not enough information' without active disqualification)",
            "counts": counts,
            "disqualifying_count": 0,
            "missing_info_count": counts["not_enough_info"],
        }

    # Rule 3: All satisfied
    return {
        "verdict": "POTENTIALLY_ELIGIBLE",
        "verdict_title": "Potentially Eligible (All Criteria Satisfied)",
        "status_color": "success",
        "summary": "Patient satisfies all evaluated inclusion criteria and satisfies all exclusion constraints (no contraindications found).",
        "rule_applied": "Rule 3: Full satisfaction (All inclusion satisfied, no exclusions triggered, no missing information)",
        "counts": counts,
        "disqualifying_count": 0,
        "missing_info_count": 0,
    }


class UnifiedInferenceController:
    """Unified engine manager maintaining loaded models and dispatching inference requests."""

    def __init__(self, adapter_path: str = DEFAULT_ADAPTER_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adapter_path = adapter_path
        self.target_model = None
        self.draft_model = None
        self.tokenizer = None
        self.spec_engine = None
        self._is_loaded = False
        self.load_error = None
        self.preset_cases = []
        self._load_presets()

    def _load_presets(self):
        """Load test split cases from data/processed and unseen vignettes."""
        self.preset_cases = list(PRESET_UNSEEN_CASES)
        # Attempt to load held-out test split cases
        try:
            from data.dataset_interface import TrialGPTDataset
            ds = TrialGPTDataset()
            test_split = ds.get_split("test")

            # Group criteria by (patient_id, trial_id)
            grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for row in test_split:
                key = (row["patient_id"], row["trial_id"])
                if key not in grouped:
                    grouped[key] = {
                        "id": f"test-{row['patient_id']}-{row['trial_id']}",
                        "title": f"Held-Out Test: Patient {row['patient_id']} (Trial {row['trial_id']})",
                        "patient_id": row["patient_id"],
                        "trial_id": row["trial_id"],
                        "trial_title": row.get("trial_title", f"Clinical Trial {row['trial_id']}"),
                        "note": row["note"],
                        "criteria": [],
                    }
                grouped[key]["criteria"].append({
                    "annotation_id": row["annotation_id"],
                    "criterion_id": f"{row['trial_id']}_{'I' if row['criterion_type']=='inclusion' else 'E'}_{row['annotation_id']}",
                    "criterion_type": row["criterion_type"],
                    "criterion_text": row["criterion_text"],
                    "gold_eligibility": row.get("expert_eligibility", ""),
                    "gold_sentences": row.get("expert_sentences", ""),
                })

            # Add top test patient cases to presets
            sorted_groups = sorted(grouped.values(), key=lambda g: len(g["criteria"]), reverse=True)
            self.preset_cases.extend(sorted_groups[:6])
            print(f"[+] Loaded {len(self.preset_cases)} preset clinical cases ({len(PRESET_UNSEEN_CASES)} unseen + {len(sorted_groups[:6])} held-out test).")
        except Exception as e:
            print(f"[!] Warning: Could not load test dataset split: {e}")

    def load_models_if_needed(self):
        """Loads Target (3B 4-bit + LoRA) and Draft (0.5B FP16) models once into memory."""
        if self._is_loaded:
            return

        print(f"[*] Initializing Unified Inference Controller on {self.device}...")
        try:
            self.target_model, self.draft_model, self.tokenizer = load_draft_and_target_models(
                target_model_id=TARGET_MODEL_ID,
                draft_model_id=DRAFT_MODEL_ID,
                adapter_path=self.adapter_path,
                device=self.device,
                draft_in_4bit=False,
            )
            self.spec_engine = SpeculativeDecodingEngine(
                target_model=self.target_model,
                draft_model=self.draft_model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            self._is_loaded = True
            print("[+] Unified Inference Controller successfully loaded models!")
        except Exception as e:
            self.load_error = str(e)
            print(f"[!] Error loading models: {e}")
            raise e

    def get_hardware_status(self) -> Dict[str, Any]:
        """Return system and GPU hardware diagnostics."""
        gpu_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "CPU Only"
        allocated_mb = round(torch.cuda.memory_allocated() / (1024**2), 1) if gpu_avail else 0.0
        reserved_mb = round(torch.cuda.memory_reserved() / (1024**2), 1) if gpu_avail else 0.0
        total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1) if gpu_avail else 0.0

        return {
            "device": self.device,
            "cuda_available": gpu_avail,
            "gpu_name": gpu_name,
            "vram_allocated_mb": allocated_mb,
            "vram_reserved_mb": reserved_mb,
            "vram_total_mb": total_mb,
            "vram_usage_percent": round((allocated_mb / max(1.0, total_mb)) * 100, 1),
            "models_loaded": self._is_loaded,
            "target_model": TARGET_MODEL_ID,
            "target_quantization": "4-bit NF4 + LoRA",
            "draft_model": DRAFT_MODEL_ID,
            "draft_precision": "FP16",
            "vocab_size": len(self.tokenizer) if self.tokenizer else 151665,
        }

    def screen_patient(
        self,
        raw_note: str,
        criteria: List[Dict[str, str]],
        decoding_mode: str = "kv_cached",
        gamma: int = 3,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Execute full screening over a list of inclusion/exclusion criteria."""
        self.load_models_if_needed()
        start_time = time.perf_counter()

        # Normalize sentence numbering
        formatted_note = normalize_and_number_sentences(raw_note)
        sentences_list = split_note_into_sentences(formatted_note)

        criterion_results = []
        total_tokens_generated = 0
        total_passes_saved = 0

        for idx, crit in enumerate(criteria):
            crit_id = crit.get("criterion_id", f"CRIT_{idx+1}")
            crit_type = crit.get("criterion_type", "inclusion").strip().lower()
            crit_text = crit.get("criterion_text", "").strip()

            if not crit_text:
                continue

            prompt_text = format_clinical_prompt(formatted_note, crit_type, crit_text, self.tokenizer)
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

            step_t0 = time.perf_counter()

            if decoding_mode == "speculative":
                spec_res = generate_speculative_greedy(
                    target_model=self.target_model,
                    draft_model=self.draft_model,
                    tokenizer=self.tokenizer,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    gamma=gamma,
                    max_new_tokens=max_new_tokens,
                )
                output_text = spec_res.generated_text
                tokens_count = spec_res.num_tokens_generated
                passes_saved = spec_res.target_forward_passes_saved
                alpha = spec_res.draft_acceptance_rate
                mean_acc = spec_res.mean_accepted_per_step
            elif decoding_mode == "uncached":
                unc_res = generate_uncached(
                    model=self.target_model,
                    tokenizer=self.tokenizer,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                )
                output_text = unc_res.generated_text
                tokens_count = unc_res.num_tokens_generated
                passes_saved = 0
                alpha = 0.0
                mean_acc = 0.0
            else:  # Default: "kv_cached"
                kv_res = generate_with_kv_cache(
                    model=self.target_model,
                    tokenizer=self.tokenizer,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                )
                output_text = kv_res.generated_text
                tokens_count = kv_res.num_tokens_generated
                passes_saved = 0
                alpha = 0.0
                mean_acc = 0.0

            step_elapsed_ms = (time.perf_counter() - step_t0) * 1000.0
            total_tokens_generated += tokens_count
            total_passes_saved += passes_saved

            parsed = parse_model_json_output(output_text, formatted_note)

            result_entry = {
                "criterion_id": crit_id,
                "criterion_type": crit_type,
                "criterion_text": crit_text,
                "prediction": parsed["prediction"],
                "evidence_sentence_ids": parsed["evidence_sentence_ids"],
                "evidence_text": parsed["evidence_text"],
                "explanation": parsed["explanation"],
                "gold_eligibility": crit.get("gold_eligibility"),
                "gold_sentences": crit.get("gold_sentences"),
                "latency_ms": round(step_elapsed_ms, 1),
                "tokens_generated": tokens_count,
                "tokens_per_sec": round((tokens_count / (step_elapsed_ms / 1000.0)), 2) if step_elapsed_ms > 0 else 0.0,
                "target_passes_saved": passes_saved,
                "acceptance_rate": round(alpha * 100.0, 1),
                "mean_accepted_per_step": mean_acc,
            }
            criterion_results.append(result_entry)

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        overall_verdict = aggregate_overall_trial_verdict(criterion_results)

        return {
            "patient_note_formatted": formatted_note,
            "sentences": [{"id": i, "text": s} for i, s in enumerate(sentences_list)],
            "decoding_mode": decoding_mode,
            "gamma": gamma if decoding_mode == "speculative" else None,
            "overall_verdict": overall_verdict,
            "criteria_results": criterion_results,
            "telemetry": {
                "total_criteria": len(criterion_results),
                "total_tokens_generated": total_tokens_generated,
                "total_time_ms": round(total_elapsed_ms, 1),
                "overall_tokens_per_sec": round((total_tokens_generated / (total_elapsed_ms / 1000.0)), 2) if total_elapsed_ms > 0 else 0.0,
                "total_target_passes_saved": total_passes_saved,
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0,
            },
        }

    def compare_decoding_modes(
        self,
        prompt_text: str,
        gamma: int = 3,
        max_new_tokens: int = 48,
    ) -> Dict[str, Any]:
        """Run the exact same prompt across Uncached, KV-Cached, and Speculative decoding live on GPU."""
        self.load_models_if_needed()
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        # 1. Uncached Mode
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        unc_res = generate_uncached(
            model=self.target_model,
            tokenizer=self.tokenizer,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        unc_ms = (time.perf_counter() - t0) * 1000.0
        unc_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

        # 2. KV-Cached Mode
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        kv_res = generate_with_kv_cache(
            model=self.target_model,
            tokenizer=self.tokenizer,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        kv_ms = (time.perf_counter() - t0) * 1000.0
        kv_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

        # 3. Speculative Mode
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec_res = generate_speculative_greedy(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            gamma=gamma,
            max_new_tokens=max_new_tokens,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        spec_ms = (time.perf_counter() - t0) * 1000.0
        spec_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

        # Token equivalence assertions
        tokens_unc = unc_res.generated_ids
        tokens_kv = kv_res.generated_ids
        tokens_spec = spec_res.generated_ids

        kv_unc_match = (tokens_unc == tokens_kv)
        spec_kv_match = (tokens_kv == tokens_spec)
        all_match = (tokens_unc == tokens_kv == tokens_spec)

        return {
            "prompt_length": inputs.input_ids.shape[1],
            "max_new_tokens": max_new_tokens,
            "gamma": gamma,
            "token_equality": {
                "uncached_vs_cached_exact": kv_unc_match,
                "cached_vs_speculative_exact": spec_kv_match,
                "all_modes_exact_match": all_match,
                "num_tokens_uncached": len(tokens_unc),
                "num_tokens_cached": len(tokens_kv),
                "num_tokens_speculative": len(tokens_spec),
            },
            "modes": {
                "uncached": {
                    "name": "Uncached Baseline (O(N^2))",
                    "latency_ms": round(unc_ms, 1),
                    "prefill_ms": round(unc_res.prefill_time_ms, 1),
                    "decode_ms": round(unc_res.decode_time_ms, 1),
                    "tokens_generated": unc_res.num_tokens_generated,
                    "tokens_per_sec": round(unc_res.tokens_per_second, 2),
                    "speedup_vs_uncached": 1.00,
                    "peak_vram_mb": round(unc_vram, 1),
                    "text": unc_res.generated_text,
                },
                "kv_cached": {
                    "name": "Explicit KV-Cached (Task 3)",
                    "latency_ms": round(kv_ms, 1),
                    "prefill_ms": round(kv_res.prefill_time_ms, 1),
                    "decode_ms": round(kv_res.decode_time_ms, 1),
                    "tokens_generated": kv_res.num_tokens_generated,
                    "tokens_per_sec": round(kv_res.tokens_per_second, 2),
                    "speedup_vs_uncached": round(unc_ms / max(1.0, kv_ms), 2),
                    "peak_vram_mb": round(kv_vram, 1),
                    "text": kv_res.generated_text,
                },
                "speculative": {
                    "name": f"Speculative (Task 5, γ={gamma})",
                    "latency_ms": round(spec_ms, 1),
                    "prefill_ms": round(spec_res.prefill_time_ms, 1),
                    "draft_ms": round(spec_res.draft_time_ms, 1),
                    "verify_ms": round(spec_res.verify_time_ms, 1),
                    "tokens_generated": spec_res.num_tokens_generated,
                    "tokens_per_sec": round(spec_res.tokens_per_second, 2),
                    "speedup_vs_uncached": round(unc_ms / max(1.0, spec_ms), 2),
                    "speedup_vs_cached": round(kv_ms / max(1.0, spec_ms), 2),
                    "acceptance_rate_percent": round(spec_res.draft_acceptance_rate * 100.0, 1),
                    "mean_accepted_per_step": spec_res.mean_accepted_per_step,
                    "target_passes_saved": spec_res.target_forward_passes_saved,
                    "peak_vram_mb": round(spec_vram, 1),
                    "text": spec_res.generated_text,
                },
            },
        }

    def run_live_attention_test(
        self,
        seq_len: int = 512,
        head_dim: int = 128,
        num_heads: int = 8,
        block_r: int = 32,
        block_c: int = 32,
        is_causal: bool = True,
    ) -> Dict[str, Any]:
        """Execute Task 4 attention kernels live and compute speed, memory, and numerical tolerance."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # 1. Reference SDPA
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref_out, _ = reference_scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        if device == "cuda":
            torch.cuda.synchronize()
        ref_ms = (time.perf_counter() - t0) * 1000.0
        ref_mem = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

        # 2. Tiled Online Softmax
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tiled_out = tiled_online_softmax_attention(
            q, k, v, block_size_r=block_r, block_size_c=block_c, is_causal=is_causal
        )
        if device == "cuda":
            torch.cuda.synchronize()
        tiled_ms = (time.perf_counter() - t0) * 1000.0
        tiled_mem = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

        # 3. Fused SDPA
        fused_ms = 0.0
        fused_mem = 0.0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fused_out = pytorch_fused_attention(q, k, v, is_causal=is_causal, backend="flash")
            torch.cuda.synchronize()
            fused_ms = (time.perf_counter() - t0) * 1000.0
            fused_mem = torch.cuda.max_memory_allocated() / (1024**2)

        # Numerical difference
        max_diff = float(torch.max(torch.abs(ref_out - tiled_out)).item())
        mean_diff = float(torch.mean(torch.abs(ref_out - tiled_out)).item())

        return {
            "parameters": {
                "seq_len": seq_len,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_r": block_r,
                "block_c": block_c,
                "is_causal": is_causal,
                "device": device,
                "precision": "FP16" if dtype == torch.float16 else "FP32",
            },
            "numerical_agreement": {
                "max_absolute_error": max_diff,
                "mean_absolute_error": mean_diff,
                "verified": max_diff < (1e-2 if dtype == torch.float16 else 1e-5),
            },
            "results": {
                "reference_sdpa": {"latency_ms": round(ref_ms, 2), "peak_vram_mb": round(ref_mem, 2)},
                "tiled_online_sdpa": {"latency_ms": round(tiled_ms, 2), "peak_vram_mb": round(tiled_mem, 2)},
                "pytorch_fused_sdpa": {"latency_ms": round(fused_ms, 2), "peak_vram_mb": round(fused_mem, 2)},
                "memory_reduction_percent": round(((ref_mem - tiled_mem) / max(0.1, ref_mem)) * 100.0, 1),
            },
        }


# Global singleton instance
controller = UnifiedInferenceController()

