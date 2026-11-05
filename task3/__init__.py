"""Task 3: KV-Cached Autoregressive Generation.

Implements an explicit key/value cache engine, uncached reference generation,
and cache-aware autoregressive decoding for Qwen2.5 models.
"""

from .kv_cache import (
    DecodingResult,
    ExplicitKVCache,
    generate_uncached,
    generate_with_kv_cache,
    load_model_for_task3,
)

__all__ = [
    "DecodingResult",
    "ExplicitKVCache",
    "generate_uncached",
    "generate_with_kv_cache",
    "load_model_for_task3",
]
