# Files Excluded From This Repository

GitHub has a hard 100 MB per-file limit, so several artifacts produced during
training and the Python virtual environment are not tracked here. They are
listed below so anyone cloning this repo knows what is missing and how to
regenerate it.

The exclusion rules live in `.gitignore`.

## Summary

| Category | Path | Approx. size | Regeneratable? |
|---|---|---|---|
| Training runs (checkpoints, optimizer state) | `judge_finetune/runs/` | ~8.3 GB | Yes — rerun training |
| Python virtual environment | `judge_finetune/.venv/` | ~5.7 GB | Yes — `pip install -r judge_finetune/requirements.txt` |

## Individual files larger than 100 MB

### Training checkpoints (`judge_finetune/runs/`)

Produced by `judge_finetune/src/train.py` for each model configuration.
Each run directory contains the HuggingFace `Trainer` checkpoint, including
both the model weights (`model.safetensors`) and the AdamW optimizer state
(`optimizer.pt`).

| File | Size |
|---|---|
| `runs/roberta_large_mnli/checkpoints/checkpoint-4728/optimizer.pt` | 2.6 GB |
| `runs/roberta_large_mnli/checkpoints/checkpoint-4728/model.safetensors` | 1.3 GB |
| `runs/roberta_base/checkpoints/checkpoint-1576/optimizer.pt` | 951 MB |
| `runs/roberta_base_augmented/checkpoints/checkpoint-1588/optimizer.pt` | 951 MB |
| `runs/smoke/checkpoints/checkpoint-56/optimizer.pt` | 951 MB |
| `runs/roberta_base/checkpoints/checkpoint-1576/model.safetensors` | 476 MB |
| `runs/roberta_base_augmented/checkpoints/checkpoint-1588/model.safetensors` | 476 MB |
| `runs/smoke/checkpoints/checkpoint-56/model.safetensors` | 476 MB |

To recreate them, run training with the corresponding config:

```bash
python -m judge_finetune.src.train --config judge_finetune/configs/roberta_base.yaml
python -m judge_finetune.src.train --config judge_finetune/configs/roberta_base_augmented.yaml
python -m judge_finetune.src.train --config judge_finetune/configs/roberta_large_mnli.yaml
python -m judge_finetune.src.train --config judge_finetune/configs/smoke.yaml
```

### Virtual environment libraries (`judge_finetune/.venv/`)

The entire `.venv/` is gitignored. The files below are the individual
binaries inside it that exceed 100 MB — listed for reference, not because
they are special. Recreate the venv with `pip install -r requirements.txt`
and these will all be reinstalled from PyPI.

| File | Size |
|---|---|
| `nvidia/cu13/lib/libcublasLt.so.13` | 517 MB |
| `torch/lib/libtorch_cuda.so` | 435 MB |
| `torch/lib/libtorch_cpu.so` | 430 MB |
| `triton/_C/libtriton.so` | 396 MB |
| `nvidia/cu13/lib/libcufft.so.12` | 273 MB |
| `nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9` | 234 MB |
| `nvidia/cusparselt/lib/libcusparseLt.so.0` | 223 MB |
| `nvidia/nccl/lib/libnccl.so.2` | 208 MB |
| `nvidia/cu13/lib/libcusparse.so.12` | 155 MB |
| `nvidia/cu13/lib/libcusolver.so.12` | 135 MB |
| `nvidia/cu13/lib/libcurand.so.10` | 127 MB |
| `nvidia/cu13/lib/libnvrtc.alt.so.13` | 105 MB |
| `nvidia/cu13/lib/libnvrtc.so.13` | 104 MB |
| `nvidia/cudnn/lib/libcudnn_adv.so.9` | 104 MB |

## Reproducing the environment

```bash
cd judge_finetune
python3.12 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt
```
