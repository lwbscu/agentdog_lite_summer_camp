# Volcengine 8xL40 Setup Prompt

你现在在一台新火山服务器上部署 AgentDoG-Lite 项目，用于后续多数据集 full-SFT 和 eval 指标分析。

服务器信息：

- Ubuntu 22.04.5 LTS
- 8 x NVIDIA L40 46GB
- NVIDIA Driver 535.216.01
- nvidia-smi CUDA Version 12.2
- nvcc 不存在，不安装 CUDA Toolkit
- 176 vCPU，944GiB RAM
- 根盘剩余约 87G，必须节省磁盘
- 不要下载或提交 `data/`、`logs/`、`outputs/`、`models/` 到 Git；`data/` 和 `models/` 后续由 U 盘或已有目录拷贝

目标：

1. `git clone` 当前仓库。
2. 创建 Python 3.11 conda 环境。
3. 安装 PyTorch cu121，不要装 cu128，因为 535 驱动不适合 cu128 wheel。
4. 安装项目依赖。
5. 配置 HF/cache/TensorBoard 日志路径。
6. 验证 8xL40 可见。
7. 用 8 卡 DDP full-SFT。
8. eval 必须记录 atbench/rjudge 的 acc、macro_f1、unsafe_f1、token 统计、拒答率、无效输出率、推理耗时、误判案例分析。

执行：

```bash
mkdir -p /root/autodl-tmp
cd /root/autodl-tmp
git clone https://github.com/lwbscu/agentdog_lite_summer_camp.git
cd /root/autodl-tmp/agentdog_lite_summer_camp

conda create -n agentdog311 python=3.11 -y
conda activate agentdog311

python -m pip install -U pip setuptools wheel packaging ninja

python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -U \
  transformers accelerate datasets peft trl huggingface-hub \
  safetensors sentencepiece protobuf scikit-learn pandas pyyaml tqdm pytest tensorboard openai

python -m pip install -e .

source scripts/setup_l40_env.sh

python - <<'PY'
import torch, sys
print("python", sys.version)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("torch cuda", torch.version.cuda)
print("gpu count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 1024**3)
print("bf16", torch.cuda.is_bf16_supported())
PY

nvidia-smi
df -h
python -m pytest tests
```

数据和模型处理：

- 不从 Git 下载数据或模型。
- 复制以下目录到仓库内：
  - `data/2026_summer_camp_teseset`
  - `data/Proessed_train_dataset`
  - 其他待实验训练集目录
  - `models/Qwen3.5-0.8B`
  - `models/AgentDoG1.5-Qwen3.5-0.8B`
  - `models/AgentDoG1.5-FG-Qwen3.5-0.8B`
- 复制后检查：

```bash
find data -maxdepth 3 -type f | sort | head -100
find models -maxdepth 3 \( -name "config.json" -o -name "*.safetensors" -o -name "tokenizer.json" \) | sort
```

8 卡 DDP full-SFT 要求：

- 只有 rank0 保存 checkpoint、调用 eval、写 TensorBoard、更新 registry、删除旧权重。
- 非 rank0 只参与训练，不执行文件清理或评估。
- checkpoint/eval/清理前后使用 distributed barrier。
- effective batch 固定按 `per_device_train_batch_size x gradient_accumulation_steps x world_size` 计算。
- L40 默认配置使用 `configs/train_qwen35_full_sft_llamafactory_l40_ddp.yaml`：
  - `max_seq_len: 8192`
  - `per_device_train_batch_size: 1`
  - `gradient_accumulation_steps: 16`
  - `world_size: 8`
  - effective batch = 128
  - `bf16: true`
  - `tf32: true`
  - `gradient_checkpointing: true`
  - `attn_implementation: sdpa`
- 不要自动降 effective batch。
- 如果 OOM，优先确认没有多进程重复 eval/保存；仍 OOM 再记录日志后调整配置。

启动 smoke：

```bash
AGENTDOG_RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)" \
torchrun --nproc_per_node=8 scripts/train_full_sft.py \
  --config configs/train_qwen35_full_sft_llamafactory_l40_ddp.yaml \
  --max-samples 64 \
  --checkpoint-interval-seconds 300 \
  --eval-limit 20
```

正式训练：

```bash
AGENTDOG_RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)" \
torchrun --nproc_per_node=8 scripts/train_full_sft.py \
  --config configs/train_qwen35_full_sft_llamafactory_l40_ddp.yaml
```

验收：

- 每 30 分钟左右保存一次完整模型。
- 每次保存后 full eval。
- 只保留 best checkpoint 和 latest checkpoint。
- eval 输出包含 atbench/rjudge 独立指标、综合 acc、token 量、拒答率、无效输出率、推理耗时、误判案例分析。
- TensorBoard 写到：
  - `logs/sft/李文博_<run_name>_<YYYYmmdd_HHMMSS>/`
  - `logs/only_eval/李文博_<run_name>_step<step>_<YYYYmmdd_HHMMSS>/`
- 如果用 eval 结果反复优化数据集，报告中必须标注该 eval set 已参与数据集迭代决策，不再作为完全无偏最终测试集。
