"""
PaiGen AI: Local QLoRA / LoRA Fine-Tuning Script
Architect: Arjun Singh

Fine-tunes a local LLM (e.g. Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct)
on consumer GPU hardware (NVIDIA RTX 5050 8GB VRAM) using 4-bit quantization,
LoRA adapters, and the compiled PaiGen bilingual curriculum dataset.
Strictly offline and local execution.
"""

import os
from pathlib import Path


FINE_TUNE_SCRIPT = '''# Local Fine-Tuning Script for PaiGen AI Brain
# Runs directly on local GPU with PyTorch, Transformers, and PEFT

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Base model recommendations for 8GB VRAM:
# "Qwen/Qwen2.5-3B-Instruct" (Exceptional Hindi/Devanagari + English reasoning)
# "meta-llama/Llama-3.2-3B-Instruct"
MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
OUTPUT_DIR = "./output/paigen-brain-lora"
DATASET_PATH = "./training/paigen_chatml_dataset.jsonl"

def train():
    print(f"[*] Initializing PaiGen Brain Fine-Tuning on: {MODEL_ID}")
    print(f"[*] GPU Detected: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")

    # 4-bit Quantization Configuration for 8GB VRAM efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA Target Modules
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

    # Load local curriculum dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=150,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        report_to="none",  # Invariant: Zero cloud telemetry
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

    print(f"[*] Training Complete! Saving adapter to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()
'''


def generate_training_runner(target_path: Path):
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(FINE_TUNE_SCRIPT)
