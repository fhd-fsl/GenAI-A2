"""QLoRA Fine-tuning Script for TrialGPT Criterion-Level Eligibility.

Reproducible training script extracted from fine_tune.ipynb.
Uses tokenized data from task1.preprocessing.tokenizer.
"""

import os
import json
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import DatasetDict

from task1.preprocessing.tokenizer import load_tokenized_splits


@dataclass
class TrainConfig:
    """Training configuration loaded from YAML."""
    model_name: str
    max_seq_len: int
    lora: dict
    quantization: dict
    training: dict
    data: dict
    system_prompt: str
    eligibility_labels: list


def load_config(config_path: Path) -> TrainConfig:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(**cfg)


def setup_model_and_tokenizer(config: TrainConfig):
    """Load base model with QLoRA quantization and apply LoRA adapters."""
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization["load_in_4bit"],
        bnb_4bit_quant_type=config.quantization["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config.quantization["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, config.quantization["bnb_4bit_compute_dtype"]),
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=config.trust_remote_code,
    )

    print(f"Model device: {next(model.parameters()).device}")
    print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora["r"],
        lora_alpha=config.lora["lora_alpha"],
        lora_dropout=config.lora["lora_dropout"],
        target_modules=config.lora["target_modules"],
        bias=config.lora["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)

    # Load tokenized datasets (uses task1.preprocessing.tokenizer)
    print("Loading tokenized datasets...")
    tokenized_splits, truncation_log = load_tokenized_splits()
    print(f"Train: {len(tokenized_splits['train'])} examples")
    print(f"Validation: {len(tokenized_splits['validation'])} examples")
    print(f"Test: {len(tokenized_splits['test'])} examples")
    print(f"Truncated examples: {len(truncation_log)}")

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)

    # Training arguments
    train_args = TrainingArguments(
        output_dir=config.training["output_dir"],
        num_train_epochs=config.training["num_train_epochs"],
        per_device_train_batch_size=config.training["per_device_train_batch_size"],
        per_device_eval_batch_size=config.training["per_device_eval_batch_size"],
        gradient_accumulation_steps=config.training["gradient_accumulation_steps"],
        learning_rate=config.training["learning_rate"],
        weight_decay=config.training["weight_decay"],
        warmup_steps=config.training["warmup_steps"],
        lr_scheduler_type=config.training["lr_scheduler_type"],
        logging_steps=config.training["logging_steps"],
        eval_strategy=config.training["eval_strategy"],
        save_strategy=config.training["save_strategy"],
        load_best_model_at_end=config.training["load_best_model_at_end"],
        metric_for_best_model=config.training["metric_for_best_model"],
        greater_is_better=config.training["greater_is_better"],
        save_total_limit=config.training["save_total_limit"],
        fp16=config.training["fp16"],
        bf16=config.training["bf16"],
        gradient_checkpointing=config.training["gradient_checkpointing"],
        gradient_checkpointing_kwargs=config.training["gradient_checkpointing_kwargs"],
        optim=config.training["optim"],
        report_to=config.training["report_to"],
        remove_unused_columns=config.training["remove_unused_columns"],
        dataloader_pin_memory=config.training["dataloader_pin_memory"],
        max_grad_norm=config.training["max_grad_norm"],
        seed=config.training["seed"],
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.training["early_stopping_patience"])],
    )

    print("Starting training...")
    train_result = trainer.train()

    # CRITICAL: Cast adapter weights back to FP16 (trainer saves in FP32)
    print("Casting adapter weights to FP16...")
    trainer.model.to(torch.float16)

    # Save model and tokenizer
    output_dir = Path(config.training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    # Verify adapter size
    adapter_path = output_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        size_gb = adapter_path.stat().st_size / 1e9
        print(f"✅ Training complete! Final adapter size: {size_gb:.2f} GB")
    else:
        print("✅ Training complete!")

    print(f"Model saved to: {output_dir}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()