"""
PaiGen AI: Tighter ML + LLM Unified Training Orchestrator
Architect: Arjun Singh

1. Compiles the full 10-level bilingual & psychological curriculum (JSONL + ChatML).
2. Trains and validates the multi-head ML Psycholinguistic & Call-Triage Model (`psychology_ml_model.npz`).
3. Runs tighter QLoRA LLM fine-tuning if PyTorch+CUDA stack is installed, and updates Ollama model when running.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paigen.brain.language.curriculum_generator import CurriculumDatasetGenerator
from paigen.brain.psychology_ml import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    PsychologyMLClassifier,
)


def main():
    print("=" * 70)
    print("  PaiGen AI — Tighter ML + LLM Psychological Training Pipeline")
    print("=" * 70)

    # 1. Compile expanded Curriculum + Level 10 Psychology Datasets
    training_dir = ROOT / "training"
    generator = CurriculumDatasetGenerator(training_dir)
    lang_file = generator.export_all()
    chatml_file = training_dir / "paigen_chatml_dataset.jsonl"

    with open(chatml_file, "r", encoding="utf-8") as f:
        chatml_count = sum(1 for _ in f)
    print(f"[1/3] Compiled curriculum: {lang_file.name} & {chatml_file.name} ({chatml_count} ChatML samples)")

    # 2. Train & Validate ML Psychology Multi-Head Classifier
    clf = PsychologyMLClassifier()
    corpus = clf.load_training_corpus()
    metrics = clf.train(samples=corpus, val_fraction=0.2, seed=42)
    clf.save(model_path=DEFAULT_MODEL_PATH, metrics=metrics, metrics_path=DEFAULT_METRICS_PATH)

    print("[2/3] Trained Multi-Head ML Psychology Classifier:")
    print(f"      - Corpus Size: {metrics['total_samples']} samples (vocab={metrics['vocab_size']})")
    print(f"      - Validation Triage Accuracy:   {metrics['val_triage_accuracy']*100:.1f}%")
    print(f"      - Validation Language Accuracy: {metrics['val_language_accuracy']*100:.1f}%")
    print(f"      - Validation Psych-State Acc:   {metrics['val_state_accuracy']*100:.1f}%")
    print(f"      - Validation Affect MAE:        {metrics['val_affect_mae']:.4f}")
    print(f"      - Full Fit Acc (State/Triage):  {metrics['full_state_accuracy']*100:.1f}% / {metrics['full_triage_accuracy']*100:.1f}%")
    print(f"      - Saved weights -> {DEFAULT_MODEL_PATH}")

    # 3. Check optional PyTorch QLoRA stack
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            from training.train_lora import run_training
            print("[3/3] CUDA GPU detected! Launching tighter QLoRA fine-tuning (r=32, cosine LR)...")
            run_training(dataset_path=str(chatml_file), max_steps=200)
        else:
            print("[3/3] PyTorch found without CUDA; skipped QLoRA training.")
    except ImportError:
        print("[3/3] QLoRA GPU stack (`requirements-train.txt`) not installed in base Python; "
              "ML Psychology weights & Modelfile v4.0 + ChatML dataset are compiled and ready.")


if __name__ == "__main__":
    main()
