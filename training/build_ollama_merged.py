"""
PaiGen AI: Builds quantized Q4_K_M Ollama model (`paigen-ai`) from merged 7B HF safetensors.
Architect: Arjun Singh
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Import quantized 7B QLoRA GGUF into Ollama")
    parser.add_argument("--name", default="paigen-ai-qlora", help="Target Ollama model name (default: paigen-ai-qlora)")
    args = parser.parse_args()

    gguf_path = ROOT / "output" / "paigen-7b-q4_k_m.gguf"
    merged_dir = ROOT / "output" / "paigen-7b-merged"
    if not gguf_path.exists() and not merged_dir.exists():
        print(f"[!] Neither GGUF nor merged safetensors found in {ROOT / 'output'}")
        sys.exit(1)

    mf_qwen = subprocess.check_output(["ollama", "show", "qwen2.5:7b", "--modelfile"], text=True, encoding="utf-8")
    m = re.search(r'TEMPLATE\s+""".*?"""', mf_qwen, re.S)
    tpl = m.group(0) if m else ""
    if not tpl:
        print("[!] Warning: Could not extract Qwen2.5 TEMPLATE from ollama show; proceeding with default.")

    mf_paigen = (ROOT / "Modelfile.paigen").read_text(encoding="utf-8")
    from_target = str(gguf_path.resolve()) if gguf_path.exists() else str(merged_dir.resolve())
    merged_mf = mf_paigen.replace("FROM qwen2.5:7b", f"FROM {from_target}\n\n{tpl}", 1)
    out_mf = ROOT / "output" / "Modelfile.paigen.merged"
    out_mf.write_text(merged_mf, encoding="utf-8")
    print(f"[*] Generated {out_mf} (from={from_target}, template={len(tpl)} chars)")

    cmd = ["ollama", "create", args.name, "-f", str(out_mf)]
    print(f"[*] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
