#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_CACHE=/root/autodl-tmp/hf_home/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_home/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_home/datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "H800 env loaded"
python - <<'PY'
import torch, sys
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("mem GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
