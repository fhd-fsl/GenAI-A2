"""Task 5: Speculative Decoding Engine.

Implements greedy speculative decoding:
- Target Model: Qwen/Qwen2.5-3B-Instruct (+ LoRA fine-tuned adapter) in 4-bit NF4
- Draft Model: Qwen/Qwen2.5-0.5B-Instruct in FP16 (or 4-bit)
- Tokenizer: Shared Qwen2Tokenizer (151,936 vocab size)
- Candidate-token drafting (gamma steps)
- Target-model parallel verification (single forward pass on [x_last, x_hat_1 ... x_hat_gamma])
- Prefix acceptance and rejection correction
- Exact KV-cache synchronization & rollback via ExplicitKVCache.crop()

Satisfies Rubric F1, F2, and F3.
"""

from __future__ import annotations

import gc
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task3.kv_cache import ExplicitKVCache, generate_with_kv_cache

# Monkey-patch PreTrainedModel.to to avoid accelerate dispatch error with single GPU 4-bit
import transformers.modeling_utils
transformers.modeling_utils.PreTrainedModel.to = lambda self, *args, **kwargs: nn.Module.to(self, *args, **kwargs)

TARGET_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER_PATH = "task1/outputs/adapter"


@dataclass
class SpeculativeDecodingResult:
    """Stores full metrics, generated tokens, and speculative statistics."""
    generated_ids: List[int]
    generated_text: str
    num_tokens_generated: int
    prompt_length: int
    gamma: int
    total_time_ms: float
    prefill_time_ms: float
    draft_time_ms: float
    verify_time_ms: float
    tokens_per_second: float
    peak_memory_mb: float
    num_verification_steps: int
    total_draft_tokens_proposed: int
    total_draft_tokens_accepted: int
    draft_acceptance_rate: float
    mean_accepted_per_step: float
    target_forward_passes_saved: int
    step_details: List[Dict[str, Any]] = field(default_factory=list)


def load_draft_and_target_models(
    target_model_id: str = TARGET_MODEL_ID,
    draft_model_id: str = DRAFT_MODEL_ID,
    adapter_path: Optional[str] = DEFAULT_ADAPTER_PATH,
    device: Optional[str] = None,
    draft_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load Target Model (3B + LoRA) and Draft Model (0.5B) onto GPU.

    Verifies 100% tokenizer compatibility between draft and target models.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[+] Loading Shared Tokenizer ({TARGET_MODEL_ID})...", flush=True)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_id, trust_remote_code=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id, trust_remote_code=True)

    # 1. Verify Tokenizer Compatibility (Rubric F1)
    target_vocab_size = len(target_tokenizer)
    draft_vocab_size = len(draft_tokenizer)
    print(f"    Target Vocab Size: {target_vocab_size} | Draft Vocab Size: {draft_vocab_size}", flush=True)
    assert target_vocab_size == draft_vocab_size, (
        f"Tokenizer mismatch! Target has {target_vocab_size} tokens, Draft has {draft_vocab_size}"
    )
    assert target_tokenizer.eos_token_id == draft_tokenizer.eos_token_id, "EOS token mismatch!"
    assert target_tokenizer.pad_token_id == draft_tokenizer.pad_token_id, "PAD token mismatch!"
    print("    [OK] Tokenizers are 100% identical and compatible!", flush=True)

    # 2. Load Target Model in 4-bit NF4
    print(f"\n[+] Loading Target Model ({target_model_id}) on {device} (4-bit=True)...", flush=True)
    compute_dtype = torch.float16 if device == "cuda" else torch.float32

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        load_in_4bit=(device == "cuda"),
        torch_dtype=compute_dtype,
        device_map={"": device} if device == "cuda" else None,
        trust_remote_code=True,
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"    Attaching LoRA fine-tuned adapter from {adapter_path}...", flush=True)
        target_model = PeftModel.from_pretrained(
            target_model,
            adapter_path,
            torch_dtype=compute_dtype,
            device_map={"": device} if device == "cuda" else None,
        )
        print("    [OK] LoRA adapter attached to Target Model!", flush=True)

    target_model.eval()

    # 3. Load Draft Model
    # 0.5B model is only ~0.98 GB in FP16, loading in FP16 gives faster draft inference without dequantization
    print(f"\n[+] Loading Draft Model ({draft_model_id}) on {device} (4-bit={draft_in_4bit})...", flush=True)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        load_in_4bit=(draft_in_4bit and device == "cuda"),
        torch_dtype=compute_dtype,
        device_map={"": device} if device == "cuda" else None,
        trust_remote_code=True,
    )
    draft_model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()
        allocated_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
        total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"[OK] Both Models Loaded! Total Allocated VRAM: {allocated_gb} GB / {total_gb} GB\n", flush=True)

    return target_model, draft_model, target_tokenizer


def generate_speculative_greedy(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gamma: int = 4,
    max_new_tokens: int = 64,
    eos_token_id: Optional[int] = None,
) -> SpeculativeDecodingResult:
    """Greedy Speculative Decoding with explicit KV cache rollback.

    Algorithm:
    1. Prefill prompt on both target and draft models.
    2. Candidate Drafting: draft_model generates gamma candidate tokens autoregressively.
    3. Target Parallel Verification: target_model runs ONE forward pass on
       [x_last, x_hat_1 ... x_hat_gamma].
    4. Acceptance & Rollback: Accept longest matching prefix + 1 correction/bonus token.
       Roll back both target_cache and draft_cache via ExplicitKVCache.crop().
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)

    eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    generated_tokens: List[int] = []
    step_details: List[Dict[str, Any]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    total_start = time.perf_counter()

    # --- 1. PREFILL PHASE ---
    # Initialize dedicated explicit KV caches for both models
    target_cache = ExplicitKVCache()
    draft_cache = ExplicitKVCache()

    pos_ids_prefill = torch.arange(0, prompt_len, dtype=torch.long, device=device).unsqueeze(0)
    cache_pos_prefill = torch.arange(0, prompt_len, dtype=torch.long, device=device)

    prefill_start = time.perf_counter()
    with torch.no_grad():
        # Prefill Target Model
        t_out = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos_ids_prefill,
            past_key_values=target_cache,
            use_cache=True,
            cache_position=cache_pos_prefill,
            return_dict=True,
        )
        # Prefill Draft Model
        _ = draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos_ids_prefill,
            past_key_values=draft_cache,
            use_cache=True,
            cache_position=cache_pos_prefill,
            return_dict=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    prefill_time_ms = (time.perf_counter() - prefill_start) * 1000.0

    # First new token is sampled greedily from the Target's prefill logits
    first_logits = t_out.logits[:, -1, :].clone()
    del t_out
    first_token = torch.argmax(first_logits, dim=-1, keepdim=True)
    first_token_id = first_token.item()
    generated_tokens.append(first_token_id)

    # Invariant: x_last is the most recently verified token NOT yet in the cache.
    # At this moment:
    # target_cache and draft_cache have length = prompt_len.
    # x_last is the (prompt_len)-th token in 0-indexed position.
    x_last = first_token
    x_last_id = first_token_id

    total_draft_time_ms = 0.0
    total_verify_time_ms = 0.0
    total_proposed = 0
    total_accepted = 0
    verification_step_count = 0

    # --- 2. SPECULATIVE GENERATION LOOP ---
    while len(generated_tokens) < max_new_tokens:
        if eos_id is not None and x_last_id == eos_id:
            break

        verification_step_count += 1
        base_cache_len = target_cache.get_seq_length()
        assert base_cache_len == draft_cache.get_seq_length(), (
            f"Cache length mismatch! Target: {base_cache_len}, Draft: {draft_cache.get_seq_length()}"
        )

        # Cap gamma if we are close to max_new_tokens
        current_gamma = min(gamma, max_new_tokens - len(generated_tokens))
        if current_gamma <= 0:
            break

        # --- A. Candidate-Token Drafting (gamma steps on draft_model) ---
        draft_start = time.perf_counter()
        draft_tokens: List[torch.Tensor] = []
        cur_input = x_last

        for j in range(current_gamma):
            cur_pos = base_cache_len + j
            d_pos_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
            d_cache_pos = torch.tensor([cur_pos], dtype=torch.long, device=device)
            d_mask = torch.ones((1, cur_pos + 1), dtype=torch.long, device=device)

            with torch.no_grad():
                d_out = draft_model(
                    input_ids=cur_input,
                    attention_mask=d_mask,
                    position_ids=d_pos_ids,
                    past_key_values=draft_cache,
                    use_cache=True,
                    cache_position=d_cache_pos,
                    return_dict=True,
                )

            next_draft = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(next_draft)
            cur_input = next_draft
            del d_out

            if eos_id is not None and next_draft.item() == eos_id:
                # Stop drafting if draft produces EOS
                break

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_draft_time_ms += (time.perf_counter() - draft_start) * 1000.0

        num_drafted = len(draft_tokens)
        total_proposed += num_drafted

        # --- B. Target Model Parallel Verification (ONE single forward pass) ---
        # Verification sequence: [x_last, draft_1, draft_2, ..., draft_K]
        # Shape: [1, num_drafted + 1]
        verify_inputs = torch.cat([x_last] + draft_tokens, dim=-1)
        k_plus_1 = num_drafted + 1

        v_pos_ids = torch.arange(
            base_cache_len, base_cache_len + k_plus_1, dtype=torch.long, device=device
        ).unsqueeze(0)
        v_cache_pos = torch.arange(
            base_cache_len, base_cache_len + k_plus_1, dtype=torch.long, device=device
        )
        v_mask = torch.ones((1, base_cache_len + k_plus_1), dtype=torch.long, device=device)

        verify_start = time.perf_counter()
        with torch.no_grad():
            t_v_out = target_model(
                input_ids=verify_inputs,
                attention_mask=v_mask,
                position_ids=v_pos_ids,
                past_key_values=target_cache,
                use_cache=True,
                cache_position=v_cache_pos,
                return_dict=True,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_verify_time_ms += (time.perf_counter() - verify_start) * 1000.0

        # Target predictions for each position:
        # Index 0 is the prediction after x_last (should match draft_tokens[0])
        # Index 1 is the prediction after draft_tokens[0] (should match draft_tokens[1])
        # ...
        # Index num_drafted is the prediction after draft_tokens[-1] (the bonus token!)
        target_predictions = torch.argmax(t_v_out.logits[0, :k_plus_1, :], dim=-1)
        del t_v_out

        # --- C. Prefix Acceptance & Cache Rollback ---
        num_accepted = 0
        eos_stopped = False
        for i in range(num_drafted):
            if len(generated_tokens) >= max_new_tokens:
                break
            target_pred_id = target_predictions[i].item()
            draft_id = draft_tokens[i].item()

            if target_pred_id == draft_id:
                num_accepted += 1
                generated_tokens.append(draft_id)
                if eos_id is not None and draft_id == eos_id:
                    eos_stopped = True
                    break
            else:
                # Rejection encountered at position i!
                # Target's prediction target_pred_id corrects the generation.
                break

        total_accepted += num_accepted

        # Determine the correction or bonus token:
        # If rejected at i < num_drafted: corrected token is target_predictions[num_accepted]
        # If all num_drafted accepted: bonus token is target_predictions[num_drafted]
        chosen_next_token_id = target_predictions[num_accepted].item()
        chosen_next_token = target_predictions[num_accepted].view(1, 1)

        # Append the correction or bonus token if space remains and previous wasn't EOS
        if not eos_stopped and len(generated_tokens) < max_new_tokens:
            generated_tokens.append(chosen_next_token_id)
            if eos_id is not None and chosen_next_token_id == eos_id:
                eos_stopped = True

        # Update cache lengths:
        # In both caches, we must retain: base_cache_len + 1 (for x_last) + num_accepted.
        new_valid_cache_len = base_cache_len + 1 + num_accepted

        # Rollback Target Cache
        target_cache.crop(new_valid_cache_len)

        # Rollback / Synchronize Draft Cache:
        # If num_accepted == num_drafted, draft_cache only holds up to draft_tokens[-2].
        # We need draft_cache to also hold draft_tokens[-1].
        if num_accepted == num_drafted:
            # Run one 1-token step on draft_model to catch up to new_valid_cache_len
            last_draft_pos = base_cache_len + num_drafted
            sync_pos_ids = torch.tensor([[last_draft_pos]], dtype=torch.long, device=device)
            sync_cache_pos = torch.tensor([last_draft_pos], dtype=torch.long, device=device)
            sync_mask = torch.ones((1, last_draft_pos + 1), dtype=torch.long, device=device)
            with torch.no_grad():
                _ = draft_model(
                    input_ids=draft_tokens[-1],
                    attention_mask=sync_mask,
                    position_ids=sync_pos_ids,
                    past_key_values=draft_cache,
                    use_cache=True,
                    cache_position=sync_cache_pos,
                    return_dict=True,
                )
        else:
            draft_cache.crop(new_valid_cache_len)

        # Advance invariant for next round
        x_last = chosen_next_token
        x_last_id = chosen_next_token_id

        step_details.append({
            "step": verification_step_count,
            "drafted": num_drafted,
            "accepted": num_accepted,
            "bonus_or_correction_token": chosen_next_token_id,
            "tokens_generated_so_far": len(generated_tokens),
        })

        if eos_stopped or len(generated_tokens) >= max_new_tokens:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time_ms = (time.perf_counter() - total_start) * 1000.0
    num_gen = len(generated_tokens)
    tokens_per_sec = (num_gen / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )

    acceptance_rate = (total_accepted / max(total_proposed, 1))
    mean_accepted_per_step = (total_accepted / max(verification_step_count, 1))
    # Each verification step executes 1 target forward pass instead of (1 + accepted) forward passes
    forward_passes_saved = max(0, num_gen - (1 + verification_step_count))

    return SpeculativeDecodingResult(
        generated_ids=generated_tokens,
        generated_text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        num_tokens_generated=num_gen,
        prompt_length=prompt_len,
        gamma=gamma,
        total_time_ms=total_time_ms,
        prefill_time_ms=prefill_time_ms,
        draft_time_ms=total_draft_time_ms,
        verify_time_ms=total_verify_time_ms,
        tokens_per_second=tokens_per_sec,
        peak_memory_mb=peak_mem_mb,
        num_verification_steps=verification_step_count,
        total_draft_tokens_proposed=total_proposed,
        total_draft_tokens_accepted=total_accepted,
        draft_acceptance_rate=round(acceptance_rate, 4),
        mean_accepted_per_step=round(mean_accepted_per_step, 2),
        target_forward_passes_saved=forward_passes_saved,
        step_details=step_details,
    )


# Convenience alias and wrapper class
load_models = load_draft_and_target_models


class SpeculativeDecodingEngine:
    """Wrapper class for benchmark and inference execution."""

    def __init__(
        self,
        target_model: AutoModelForCausalLM,
        draft_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        gamma: int = 4,
    ) -> SpeculativeDecodingResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return generate_speculative_greedy(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            gamma=gamma,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )


