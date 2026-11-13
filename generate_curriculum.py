"""
PaiGen AI: Curriculum Dataset Builder Execution Script
Runs the curriculum generator to output JSONL files for training.
"""

from pathlib import Path
from paigen.brain.language.curriculum_generator import CurriculumDatasetGenerator

if __name__ == "__main__":
    output_dir = Path(__file__).parent / "training"
    generator = CurriculumDatasetGenerator(output_dir)
    exported = generator.export_all()
    print(f"[*] Successfully generated bilingual curriculum at: {exported}")
    print(f"[*] Generated ChatML dataset at: {output_dir / 'paigen_chatml_dataset.jsonl'}")
