"""
PaiGen AI: Standalone Local QLoRA Fine-Tuning Script
Architect: Arjun Singh

Runs strictly on local hardware (e.g., NVIDIA RTX 5050 Laptop GPU, 8GB VRAM).
Fine-tunes a local base model on compiled ChatML curriculum datasets using 4-bit NF4
quantization and PEFT LoRA adapters.
Compatible across modern TRL releases (supporting both SFTConfig and TrainingArguments).
"""

import argparse
import os
from pathlib import Path


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
    print(f"[*] TRL Version: {trl.__version__}")

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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Adapt across TRL versions (SFTConfig in >=0.9 vs TrainingArguments in older)
    if hasattr(trl, "SFTConfig"):
        training_args = trl.SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            report_to="none",  # Strict Invariant: Zero cloud telemetry
            max_seq_length=1024,
            dataset_text_field="messages",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            tokenizer=tokenizer,
            args=training_args,
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            dataset_text_field="messages",
            max_seq_length=1024,
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
    args = parser.parse_args()

    run_training(
        base_model_id=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_steps=args.steps
    )
