"""
PaiGen AI: Standalone Local QLoRA Fine-Tuning Script
Architect: Arjun Singh

Fine-tunes a local base model on compiled ChatML curriculum datasets using 4-bit NF4
quantization and PEFT LoRA adapters.
Properly formats ChatML messages using the model's official tokenizer chat template
(`apply_chat_template`) rather than passing raw dictionary structures.
"""

import argparse
import inspect
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


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


def _export_dequantized_merged_hf(peft_model, tokenizer, merged_dir: Path, dtype):
    """
    Dequantizes 4-bit bitsandbytes layers, adds trained LoRA deltas in full precision,
    and saves a clean FP16/BF16 HuggingFace safetensors directory for `ollama create -q Q4_K_M`.
    """
    import torch
    import torch.nn as nn
    from peft.tuners.lora.bnb import Linear4bit as PeftBnbLinear4bit
    from peft.utils.integrations import dequantize_bnb_weight

    base = peft_model.get_base_layer() if hasattr(peft_model, "get_base_layer") else peft_model.base_model.model

    def _replace_4bit(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, PeftBnbLinear4bit):
                w_bnb = child.get_base_layer().weight
                w_fp = dequantize_bnb_weight(w_bnb, state=w_bnb.quant_state).to(dtype=dtype)
                for ad in child.active_adapters:
                    if ad in child.lora_A:
                        w_fp = w_fp + child.get_delta_weight(ad).to(dtype=dtype)
                bias_tensor = getattr(child.get_base_layer(), "bias", None)
                new_lin = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=bias_tensor is not None,
                    device="cpu",
                    dtype=dtype,
                )
                with torch.no_grad():
                    new_lin.weight.copy_(w_fp.cpu())
                    if bias_tensor is not None:
                        new_lin.bias.copy_(bias_tensor.detach().to(dtype=dtype).cpu())
                setattr(module, name, new_lin)
                del w_fp, child
            else:
                _replace_4bit(child)

    print(f"[*] Merging trained 7B QLoRA weights into full-precision safetensors -> {merged_dir}")
    _replace_4bit(base)
    torch.cuda.empty_cache()
    base.to("cpu")
    base._weight_conversions = []
    for attr in ("is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized"):
        if hasattr(base, attr):
            setattr(base, attr, False)
    for attr in ("hf_quantizer", "quantization_method"):
        if hasattr(base, attr):
            setattr(base, attr, None)
    if hasattr(base.config, "quantization_config"):
        try:
            delattr(base.config, "quantization_config")
        except Exception:
            base.config.quantization_config = None
    merged_dir.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[+] Saved clean merged 7B HF safetensors to: {merged_dir}")


def run_training(
    base_model_id: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    dataset_path: str = "./training/paigen_chatml_dataset.jsonl",
    output_dir: str = "./output/paigen-7b-lora",
    max_steps: int = 160,
    adapter_init: str = "",
    lr: float = 1.5e-4,
    export_merged_dir: str = "",
):
    """Executes local 4-bit NF4 QLoRA fine-tuning tuned for 8GB RTX VRAM."""
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        import trl
        from trl import SFTTrainer
    except ImportError as e:
        print(f"[!] PyTorch/Transformers/PEFT/TRL not installed in current environment: {e}")
        print("[!] Install dependencies with: pip install -r requirements-train.txt")
        return False

    if not torch.cuda.is_available():
        print("[!] CUDA GPU is not available. Local fine-tuning on CPU is not recommended for LLMs.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[*] Initializing PaiGen Brain 7B QLoRA Fine-Tuning on: {base_model_id}")
    print(f"[*] Target GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    use_bf16 = torch.cuda.is_bf16_supported()
    cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    has_quant_cfg = getattr(cfg, "quantization_config", None) is not None

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if not has_quant_cfg:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    model.config.use_cache = False

    if max_steps <= 0 and adapter_init and export_merged_dir and Path(adapter_init).exists():
        print(f"[*] Fast merge-only mode from adapter: {adapter_init}")
        model = PeftModel.from_pretrained(model, adapter_init, is_trainable=False)
        merge_dtype = torch.bfloat16 if use_bf16 else torch.float16
        _export_dequantized_merged_hf(model, tokenizer, Path(export_merged_dir), merge_dtype)
        return True

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if adapter_init and Path(adapter_init).exists():
        print(f"[*] Continuing QLoRA weights from saved checkpoint: {adapter_init}")
        model = PeftModel.from_pretrained(model, adapter_init, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    split_ds = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = format_dataset_with_chat_template(split_ds["train"], tokenizer)
    eval_dataset = format_dataset_with_chat_template(split_ds["test"], tokenizer)

    SFTConfig = getattr(trl, "SFTConfig", None)
    if SFTConfig is not None:
        sft_params = inspect.signature(SFTConfig.__init__).parameters
        cfg_kwargs = {
            "output_dir": output_dir,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
            "warmup_steps": min(12, max(2, max_steps // 10)),
            "warmup_ratio": 0.08,
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.01,
            "max_grad_norm": 0.3,
            "max_steps": max_steps,
            "learning_rate": lr,
            "bf16": use_bf16,
            "fp16": not use_bf16,
            "eval_strategy": "steps",
            "eval_steps": max(10, min(40, max_steps // 2)),
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": max(20, min(80, max_steps)),
            "load_best_model_at_end": True,
            "report_to": "none",
        }
        if "dataset_text_field" in sft_params:
            cfg_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sft_params:
            cfg_kwargs["max_seq_length"] = 1024
        elif "max_length" in sft_params:
            cfg_kwargs["max_length"] = 1024
        training_args = SFTConfig(**{k: v for k, v in cfg_kwargs.items() if k in sft_params})
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            warmup_ratio=0.08,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=0.3,
            max_steps=max_steps,
            learning_rate=lr,
            bf16=use_bf16,
            fp16=not use_bf16,
            eval_strategy="steps",
            eval_steps=max(10, min(40, max_steps // 2)),
            logging_steps=10,
            save_strategy="steps",
            save_steps=max(20, min(80, max_steps)),
            load_best_model_at_end=True,
            report_to="none",
        )

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = 1024
    # Do not pass peft_config again when model is already wrapped via get_peft_model
    trainer = SFTTrainer(**trainer_kwargs)

    if max_steps > 0:
        print("[*] Starting Local 7B QLoRA Training on RTX GPU...")
        trainer.train()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[*] Training Complete! Saving LoRA adapter to: {out_path}")
    trainer.model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    if export_merged_dir:
        import gc
        merged_model = trainer.model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        merge_dtype = torch.bfloat16 if use_bf16 else torch.float16
        _export_dequantized_merged_hf(merged_model, tokenizer, Path(export_merged_dir), merge_dtype)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaiGen Local 7B QLoRA Fine-Tuner")
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", help="Base HuggingFace Model ID")
    parser.add_argument("--dataset", default="./training/paigen_chatml_dataset.jsonl", help="Dataset path")
    parser.add_argument("--output", default="./output/paigen-7b-lora", help="Output directory")
    parser.add_argument("--steps", type=int, default=160, help="Training steps")
    parser.add_argument("--adapter-init", default="", help="Optional existing LoRA adapter dir to continue from")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--export-merged", default="", help="Optional output path for merged FP16/BF16 safetensors")
    parser.add_argument("--verify-only", action="store_true", help="Only verify chat template formatting")
    args = parser.parse_args()

    if args.verify_only:
        verify_dataset_formatting(args.model, args.dataset)
    else:
        run_training(
            base_model_id=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            max_steps=args.steps,
            adapter_init=args.adapter_init,
            lr=args.lr,
            export_merged_dir=args.export_merged,
        )
