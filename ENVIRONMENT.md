# PaiGen AI: Tested Execution Environment & Dependencies

## 1. Supported Runtime Specifications
- **Operating Systems**: Windows 11 / Linux (Ubuntu 22.04+ / WSL2)
- **Target Python Version**: Python 3.10 – 3.12 (Python 3.10 / 3.11 recommended for C-extension binary wheels)
- **Target GPU**: NVIDIA GeForce RTX 5050 (8GB VRAM) / RTX 3060/4060+ with compute capability >= 7.5
- **CUDA Runtime**: CUDA 12.1 or CUDA 12.4

## 2. Installation Procedure
```powershell
# 1. Create a clean Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install PyTorch with CUDA 12.4 support
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install pinned dependencies
pip install -r requirements.txt
```

## 3. Pinned Dependency Matrix
| Package | Version | Role |
|---|---|---|
| `pydantic` | `2.8.2` | Data contracts, validation schemas |
| `requests` | `2.32.3` | Local HTTP IPC |
| `torch` | `2.4.0` | Tensor backend & GPU execution |
| `transformers` | `4.44.2` | Hugging Face model architecture & tokenization |
| `datasets` | `2.21.0` | Local ChatML dataset loading |
| `peft` | `0.12.0` | Low-Rank Adaptation (LoRA) adapters |
| `trl` | `0.9.6` | Supervised Fine-Tuning (`SFTTrainer` with ChatML template) |
| `bitsandbytes` | `0.43.3` | 4-bit NormalFloat (NF4) quantization |
| `accelerate` | `0.33.0` | Multi-device dispatch and memory allocation |
