"""Task 3: Key/Value (KV) Cache Engine & Autoregressive Decoding Loops.

Implements an explicit, instrumented Key/Value cache (ExplicitKVCache)
and both uncached (reference) and cached greedy autoregressive decoding
loops with explicit positional (RoPE) and causal mask handling.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Ensure safe 4-bit dispatch on single-GPU Windows setups
import transformers.modeling_utils

transformers.modeling_utils.PreTrainedModel.to = lambda self, *args, **kwargs: nn.Module.to(
    self, *args, **kwargs
)

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import Cache

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = str(Path(__file__).parent.parent / "task1" / "outputs" / "adapter")


class ExplicitKVCache(Cache):
    """Explicit Key/Value cache managing per-layer K and V tensor state.

    Maintains key_cache and value_cache lists across all decoder layers,
    explicitly updating tensors along the sequence dimension (dim=-2),
    measuring memory consumption, and tracking growth step-by-step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens: int = 0
        self.growth_trace: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self.key_cache)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        raise IndexError(f"Layer {layer_idx} out of range ({len(self.key_cache)} layers present).")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a given layer.

        Args:
            key_states: [batch_size, num_kv_heads, seq_len, head_dim]
            value_states: [batch_size, num_kv_heads, seq_len, head_dim]
            layer_idx: integer index of transformer decoder layer
            cache_kwargs: optional rotary embeddings or cache position metadata
        """
        # Track seen tokens on first layer pass
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            # Prefill phase: initialize layer cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Decode phase: concatenate new token K/V along sequence dimension
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def record_step(self, step_idx: int) -> None:
        """Record instantaneous memory and sequence length for step-by-step profiling."""
        seq_len = self.get_seq_length()
        mem_bytes = self.get_memory_bytes()
        self.growth_trace.append(
            {
                "step": step_idx,
                "seq_len": seq_len,
                "memory_bytes": mem_bytes,
                "memory_mb": round(mem_bytes / (1024**2), 4),
            }
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns sequence length of cached states."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    def crop(self, max_length: int) -> None:
        """Crop cache to max_length tokens along dim=-2.

        Essential for speculative decoding rollback (Task 5).
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def get_memory_bytes(self) -> int:
        """Calculate exact memory occupied by all cached key and value tensors."""
        total = 0
        for k, v in zip(self.key_cache, self.value_cache):
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total

    def get_memory_mb(self) -> float:
        return self.get_memory_bytes() / (1024 * 1024)


@dataclass
class DecodingResult:
    """Stores full metrics, generated tokens, and memory traces from a decoding run."""
    mode: str
    generated_ids: List[int]
    generated_text: str
    num_tokens_generated: int
    prompt_length: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    peak_memory_mb: float
    step_times_ms: List[float] = field(default_factory=list)
    step_logits: List[torch.Tensor] = field(default_factory=list)
    kv_memory_trace: List[Dict[str, Any]] = field(default_factory=list)


def generate_uncached(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 32,
    eos_token_id: Optional[int] = None,
    store_logits: bool = False,
) -> DecodingResult:
    """Deterministic greedy autoregressive decoding WITHOUT key/value cache reuse.

    At each generation step t, the entire input prefix (0 ... L+t-1) is passed
    through the model forward pass, recomputing all activations from scratch (O(N^2) complexity).
    """
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)

    current_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]
    eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    generated_tokens: List[int] = []
    step_times: List[float] = []
    step_logits: List[torch.Tensor] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    total_start = time.perf_counter()
    prefill_time_ms = 0.0

    for step in range(max_new_tokens):
        seq_len = current_ids.shape[1]
        cur_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        step_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=current_ids,
                attention_mask=cur_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_elapsed = (time.perf_counter() - step_start) * 1000.0

        if step == 0:
            prefill_time_ms = step_elapsed
        else:
            step_times.append(step_elapsed)

        # Greedy next-token selection from last logit position
        next_token_logits = outputs.logits[:, -1, :].clone()
        del outputs
        if store_logits:
            step_logits.append(next_token_logits.detach().cpu())

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        generated_tokens.append(token_id)

        current_ids = torch.cat([current_ids, next_token], dim=-1)

        if eos_id is not None and token_id == eos_id:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time_ms = (time.perf_counter() - total_start) * 1000.0
    decode_time_ms = sum(step_times) if step_times else 0.0

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )

    num_gen = len(generated_tokens)
    tokens_per_sec = (num_gen / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    return DecodingResult(
        mode="uncached",
        generated_ids=generated_tokens,
        generated_text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        num_tokens_generated=num_gen,
        prompt_length=prompt_len,
        prefill_time_ms=prefill_time_ms,
        decode_time_ms=decode_time_ms,
        total_time_ms=total_time_ms,
        tokens_per_second=tokens_per_sec,
        peak_memory_mb=peak_mem_mb,
        step_times_ms=step_times,
        step_logits=step_logits,
    )


def generate_with_kv_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 32,
    eos_token_id: Optional[int] = None,
    store_logits: bool = False,
) -> DecodingResult:
    """Deterministic greedy autoregressive decoding WITH explicit KV cache reuse.

    Prefill (step 0): Passes entire prompt of length L, populates ExplicitKVCache.
    Decode (step 1..N-1): Passes single new token [1, 1], explicit position_ids=[[past_len]],
    cache_position=[past_len], and incremental attention mask [1, past_len + 1].
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)

    eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    generated_tokens: List[int] = []
    step_times: List[float] = []
    step_logits: List[torch.Tensor] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    total_start = time.perf_counter()

    # 1. PREFILL PHASE
    kv_cache = ExplicitKVCache()
    position_ids = torch.arange(0, prompt_len, dtype=torch.long, device=device).unsqueeze(0)
    cache_position = torch.arange(0, prompt_len, dtype=torch.long, device=device)

    prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=True,
            cache_position=cache_position,
            return_dict=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    prefill_time_ms = (time.perf_counter() - prefill_start) * 1000.0
    kv_cache.record_step(0)

    # Next token from prefill
    next_token_logits = outputs.logits[:, -1, :].clone()
    del outputs
    if store_logits:
        step_logits.append(next_token_logits.detach().cpu())

    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    token_id = next_token.item()
    generated_tokens.append(token_id)

    # 2. DECODE PHASE (Single token forward passes)
    for step in range(1, max_new_tokens):
        if eos_id is not None and token_id == eos_id:
            break

        past_len = kv_cache.get_seq_length()
        # Explicit positional handling: position_ids is exactly [[past_len]]
        position_ids = torch.tensor([[past_len]], dtype=torch.long, device=device)
        cache_position = torch.tensor([past_len], dtype=torch.long, device=device)
        cur_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)

        step_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=cur_mask,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=True,
                cache_position=cache_position,
                return_dict=True,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_elapsed = (time.perf_counter() - step_start) * 1000.0
        step_times.append(step_elapsed)
        kv_cache.record_step(step)

        next_token_logits = outputs.logits[:, -1, :].clone()
        del outputs
        if store_logits:
            step_logits.append(next_token_logits.detach().cpu())

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        generated_tokens.append(token_id)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time_ms = (time.perf_counter() - total_start) * 1000.0
    decode_time_ms = sum(step_times) if step_times else 0.0

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    num_gen = len(generated_tokens)
    tokens_per_sec = (num_gen / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    return DecodingResult(
        mode="kv_cached",
        generated_ids=generated_tokens,
        generated_text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        num_tokens_generated=num_gen,
        prompt_length=prompt_len,
        prefill_time_ms=prefill_time_ms,
        decode_time_ms=decode_time_ms,
        total_time_ms=total_time_ms,
        tokens_per_second=tokens_per_sec,
        peak_memory_mb=peak_mem_mb,
        step_times_ms=step_times,
        step_logits=step_logits,
        kv_memory_trace=kv_cache.growth_trace,
    )


def load_model_for_task3(
    adapter_path: str = ADAPTER_PATH,
    base_model: str = BASE_MODEL_NAME,
    load_in_4bit: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and adapter with 4-bit quantization on local GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {base_model} on {device} (4-bit={load_in_4bit})...")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if device == "cuda" and load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    if device == "cuda":
        vram = round(torch.cuda.memory_allocated() / (1024**3), 2)
        print(f"[OK] Task 3 Model Ready! Allocated VRAM: {vram} GB / {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")

    return model, tokenizer
