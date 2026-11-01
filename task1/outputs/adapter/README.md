# TrialGPT QLoRA Adapter

## Download
**Google Drive**: https://drive.google.com/drive/folders/1YEafpplZJwsxZx9rfEQxkuEPuCg8xUrp?usp=sharing

## Contents (Required)
| File | Size | Purpose |
|------|------|---------|
| `adapter_model.safetensors` | 479 MB | LoRA weights |
| `adapter_config.json` | 1.2 KB | LoRA config (r=128, alpha=256) |
| `tokenizer.json` | 11.4 MB | Qwen2.5-3B-Instruct tokenizer |
| `tokenizer_config.json` | 691 B | Tokenizer config |
| `test_predictions.json` | 106 KB | Test set predictions with gold labels |

## Extra Files (Can Ignore)
- `chat_template.jinja` — Jinja template (not needed, tokenizer has it)
- `training_args.bin` — Binary training args (not human-readable)
- `README.md` — This file

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "path/to/adapter/folder")
tokenizer = AutoTokenizer.from_pretrained("path/to/adapter/folder")
```

## Training Details
- **Base model**: Qwen/Qwen2.5-3B-Instruct
- **Method**: QLoRA (4-bit NF4, r=128, alpha=256, dropout=0.05)
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable params**: ~239M (7.2%)
- **Sequence length**: 2048
- **Epochs**: 3 (overfitting observed; early stopping at epoch 1 recommended)
- **Data**: Patient-grouped 37/8/8 split (seed 42)
- **Full config**: `task1/training/config.yaml`

## Results (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | 0.7330 |
| Macro F1 | 0.4501 |
| Evidence F1 (micro) | 0.5871 |
| Inclusion Accuracy | 0.7333 |
| Exclusion Accuracy | 0.7328 |

**Note**: Validation loss increased across epochs (1.31 → 1.77) indicating overfitting on small dataset (37 patients, 603 annotations). Best validation loss at epoch 1. Consider fewer epochs or higher regularization in future runs.