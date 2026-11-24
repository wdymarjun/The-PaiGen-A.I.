"""
PaiGen AI: Standalone Local QLoRA Fine-Tuning Script
Architect: Arjun Singh

Fine-tunes a local base model on compiled ChatML curriculum datasets using 4-bit NF4
quantization and PEFT LoRA adapters.
Properly formats ChatML messages using the model's official tokenizer chat template
(`apply_chat_template`) rather than passing raw dictionary structures.
"""

import argparse
import os
from pathlib import Path


def format_dataset_with_chat_template(dataset, tokenizer):
    """
    Applies the model's official chat template to the 'messages' array,
    producing a clean, tokenizable 'text' field for SFTTrainer.
    """
    def _format(example):
        formatted_prompt = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": formatted_prompt}

    return dataset.map(_format, remove_columns=["messages"])


def verify_dataset_formatting(base_model_id: str, dataset_path: str):
    """Verifies that the dataset cleanly maps to the model's chat template."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"[!] Dependencies not installed for verification: {e}")
        return False

    print(f"[*] Testing chat template formatting on: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    formatted = format_dataset_with_chat_template(dataset, tokenizer)
    sample_text = formatted[0]["text"]
    print(f"[+] Successfully verified formatting! First sample preview:\n{'-'*60}\n{sample_text[:300]}...\n{'-'*60}")
    print(f"[+] Total verified training samples: {len(formatted)}")
    return True


def run_training(
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    dataset_path: str = "./training/paigen_chatml_dataset.jsonl",
    output_dir: str = "./output/paigen-brain-lora",
    max_steps: int = 150,
):
    """Executes local parameter-efficient fine-tuning if PyTorch/CUDA is available."""
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import trl
        from trl import SFTTrainer
    except ImportError as e:
        print(f"[!] PyTorch/Transformers/PEFT/TRL not installed in current environment: {e}")
        print("[!] Install pinned dependencies with: pip install -r requirements.txt")
        return False

    if not torch.cuda.is_available():
        print("[!] CUDA GPU is not available. Local fine-tuning on CPU is not recommended for LLMs.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[*] Initializing PaiGen Brain Fine-Tuning on: {base_model_id}")
    print(f"[*] Target GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    # 4-bit Quantization Configuration for 8GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    split_ds = raw_dataset.train_test_split(test_size=0.1, seed=42)
    # Apply official ChatML chat template to format clean text
    train_dataset = format_dataset_with_chat_template(split_ds["train"], tokenizer)
    eval_dataset = format_dataset_with_chat_template(split_ds["test"], tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.08,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,
        max_steps=max_steps,
        learning_rate=1.5e-4,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        eval_strategy="steps",
        eval_steps=25,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        report_to="none",  # Strict Invariant: Zero cloud telemetry
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1536,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("[*] Starting Local Training...")
    trainer.train()

    print(f"[*] Training Complete! Saving adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaiGen Local LoRA Fine-Tuner")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Base HuggingFace Model ID")
    parser.add_argument("--dataset", default="./training/paigen_chatml_dataset.jsonl", help="Dataset path")
    parser.add_argument("--output", default="./output/paigen-brain-lora", help="Output directory")
    parser.add_argument("--steps", type=int, default=150, help="Training steps")
    parser.add_argument("--verify-only", action="store_true", help="Only verify chat template formatting")
    args = parser.parse_args()

    if args.verify_only:
        verify_dataset_formatting(args.model, args.dataset)
    else:
        run_training(
            base_model_id=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            max_steps=args.steps
        )
