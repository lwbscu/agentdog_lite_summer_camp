#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_CACHE=/root/autodl-tmp/hf_home/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_home/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_home/datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "L40 env loaded"
python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(idx)
    print(idx, props.name, props.total_memory / 1024**3)
print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
