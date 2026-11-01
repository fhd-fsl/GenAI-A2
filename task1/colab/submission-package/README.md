
# TrialGPT QLoRA Adapter

## Model
- Base: Qwen/Qwen2.5-3B-Instruct
- Method: QLoRA (4-bit NF4, r=128, alpha=256)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable params: ~239M (7.2%)

## Data
- Dataset: ncbi/TrialGPT-Criterion-Annotations
- Split: 37 train / 8 val / 8 test (patient-grouped, seed 42)
- Train: 603 annotations, 37 patients
- Val: 206 annotations, 8 patients  
- Test: 206 annotations, 8 patients

## Training
- Epochs: 3
- Batch size: 1 (grad_accum=8, effective=8)
- LR: 2e-4, cosine, warmup=50 steps
- Seq length: 2048
- Optimizer: paged_adamw_8bit
- Gradient checkpointing: enabled

## Results (Test Set)
- Accuracy: 0.7330
- Macro F1: 0.5402
- Evidence F1 (micro): 0.8284

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "./adapter")
tokenizer = AutoTokenizer.from_pretrained("./adapter")
```
