"""Task 5: Speculative Decoding.

Implements greedy speculative decoding using Qwen2.5-0.5B-Instruct as a draft
model and Qwen2.5-3B-Instruct (+ LoRA adapter) as the target model.
"""

from task5.speculative_decoding import (
    SpeculativeDecodingResult,
    generate_speculative_greedy,
    load_draft_and_target_models,
)

__all__ = [
    "load_draft_and_target_models",
    "generate_speculative_greedy",
    "SpeculativeDecodingResult",
]

