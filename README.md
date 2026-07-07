# AgentDoG-Lite Summer Camp

模型权重与训练数据已公开上传至 Hugging Face。

## 模型权重

**Full-SFT 完整权重版本：**  
https://huggingface.co/hhhggfdd/doc-was-wrong-because-training-started-from-qwen3.5-0.8b-base-not-agentdog-full-sft

**LoRA adapter 版本：**  
https://huggingface.co/hhhggfdd/doc-was-wrong-because-training-started-from-qwen3.5-0.8b-base-not-agentdog-lora

LoRA adapter 是在上述 **Full-SFT 完整权重版本** 的基础上继续训练得到的，  
而 Full-SFT 完整权重版本本身是从基础模型 `Qwen/Qwen3.5-0.8B` 开始训练。

也就是说，训练链路为：

`Qwen/Qwen3.5-0.8B` → `Full-SFT 完整权重` → `LoRA adapter`

该训练链路不是从官方 `AI45Research/AgentDoG1.5-Qwen3.5-0.8B` 初始化；  
官方 AgentDoG1.5 仅作为 reference 对照。

## 训练数据

**训练数据套件：**  
https://huggingface.co/datasets/hhhggfdd/agentdog-lite-qwen35-08b-base-training-data-suite

本项目实现 AgentDoG-Lite 题目 1 的高标准方案：最终主线从基础模型 `Qwen/Qwen3.5-0.8B`（！！！！！结果是从基础模型Qwen3.5-0.8B开始训练，而不是agentdog！！！！！）开始训练，经过 Full-SFT与LoRA/SFT优化，构建面向 Agent 轨迹级安全判断的轻量模型。官方 `AI45Research/AgentDoG1.5-Qwen3.5-0.8B`，AgentDoG1.5作为对照模型，用于比较表现。

当前实验主线同时包含一套全参 Full-SFT + batch eval 闭环：使用本地 H800 环境对基础模型 `Qwen/Qwen3.5-0.8B` 做完整权重 SFT，每约 30 分钟保存 checkpoint 并立即在 summer camp test set 上 batch eval；只保留 best/latest 权重以节省磁盘。当前推荐本地权重入口为：

```text
outputs/qwen35_full_sft_llamafactory_h800/best_checkpoint
```

注意：本地 `data/`、`models/`、`outputs/`、`logs/`、`report/` 均不提交到 Git。

最终推理输出严格限制为：

```json
{"judgment":"safe"}
```

或：

```json
{"judgment":"unsafe"}
```

三维风险框架只用于训练辅助与内部判断，不在最终输出里展开 reasoning。

## 环境安装

推荐主环境：

- Python 3.11
- PyTorch 2.8.0+cu128
- bf16 LoRA
- SDPA attention
- 不强依赖 `flash-attn` / `deepspeed` / `bitsandbytes`

当前比赛主环境为 conda env `agentdog311`。训练和评测脚本支持 Python >=3.10,<3.13，但默认运行环境使用 `agentdog311`。

```bash
conda create -n agentdog311 python=3.11 -y
conda activate agentdog311
python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
source scripts/setup_h800_env.sh
python -m pip install -U \
  "transformers" \
  "accelerate" \
  "datasets" \
  "peft" \
  "trl" \
  "huggingface-hub" \
  "safetensors" \
  "sentencepiece" \
  "protobuf" \
  "scikit-learn>=1.5.0" \
  "pandas>=2.2.0" \
  "pyyaml>=6.0.1" \
  "tqdm" \
  "pytest" \
  "tensorboard" \
  "openai"
python -m pip install -e .
```

不强装 `flash_attn`、`deepspeed`、`bitsandbytes`，也不需要 `nvcc`。

火山 8xL40 服务器使用 PyTorch cu121，不使用 cu128；完整部署提示词见 `docs/volcano_l40_setup_prompt.md`。

## 官方仓库

官方仓库只作参考，不修改其中内容。

```bash
bash scripts/prepare_official_repo.sh
```

当前记录的官方 commit 位于：

```text
third_party/AgentDoG_COMMIT.txt
```

## 模型下载

```bash
python scripts/download_models.py
```

会下载并校验：

| method 名称 | Hugging Face 模型 | 本地路径 |
|---|---|---|
| `qwen35_08b_baseline` | `Qwen/Qwen3.5-0.8B` | `models/Qwen3.5-0.8B` |
| `qwen35_08b_reference` | `AI45Research/AgentDoG1.5-Qwen3.5-0.8B` | `models/AgentDoG1.5-Qwen3.5-0.8B` |

## 数据下载

```bash
python scripts/download_data.py
```

训练数据只来自：

- `AI45Research/AgentDoG1.0-Training-Data/AgentDoG-BinarySafety`
- `AI45Research/AgentDoG1.0-Training-Data/AgentDoG-FineGrainedTaxonomy`
- 本项目自建 hard boundary seed：`data/hard_boundary/hard_boundary_seed.json`

评测数据只用于评测：

- `AI45Research/2026_summer_camp_teseset/summer_camp_ATBench300.json`
- `AI45Research/2026_summer_camp_teseset/summer_camp_rjudge.json`

## 数据处理

```bash
python scripts/build_training_data.py
```

输出：

```text
data/processed/train_binary.jsonl
data/processed/train_diagnostic_aux.jsonl
data/processed/train_mixed.jsonl
data/processed/train_mixed_train.jsonl
data/processed/train_mixed_dev.jsonl
data/processed/build_summary.json
data/processed/token_length_summary.json
```

`train_mixed` 优先按 70% binary、20% diagnostic、10% hard boundary 构造；如果 hard boundary 数量不足以支撑稳定训练规模，则按规则回退到 75% binary、25% diagnostic。dev split 来自训练集内部，按 task/source/judgment 分层，绝不使用 summer camp test set 做 dev、few-shot 或 prompt tuning。

## Continued LoRA 训练

H800 主线配置：

```bash
python scripts/train_lora.py --config configs/train_qwen35_fallback_lora_h800.yaml
```


训练脚本会校验 H800 配置的 effective batch size 必须为 128。默认设置为 `per_device_train_batch_size=8`、`gradient_accumulation_steps=16`、单卡有效 batch 128，`max_seq_len=16384`。训练只对 assistant JSON target 计算 loss，system/user/trajectory prompt token 的 label 均为 `-100`。每次训练都会写 TensorBoard 日志到 `logs/sft/李文博_<run_name>_<YYYYmmdd_HHMMSS>/`。

导出最终 adapter：

```bash
python scripts/export_final_adapter.py \
  --run-dir outputs/our_qwen35_continued_lora \
  --output-dir outputs/final_continued_lora_adapter
```

## Full-SFT 训练

全参 SFT 入口为：

```bash
python scripts/train_full_sft.py \
  --config configs/train_qwen35_full_sft_llamafactory_h800.yaml
```

训练数据使用 Alpaca 风格 JSON：

```json
{"instruction":"... <BEGIN TRAJECTORY> ... <END TRAJECTORY> ...","input":"","output":"{\"judgment\":\"safe\"}"}
```

训练行为：

- 从 `instruction` 中提取 `<BEGIN TRAJECTORY>` 和 `<END TRAJECTORY>` 之间的 trajectory 作为 user prompt。
- system prompt 使用 AgentDoG 风险判断规则。
- 只对 assistant target 计算 loss，system/user token 的 label 全部为 `-100`。
- 支持 `assistant_target_mode: judgment_only` 和 `assistant_target_mode: full_json`。
- prompt 超长时使用 head+tail 截断，并保证 assistant target 不被截断。
- 每约 `checkpoint_interval_seconds=1800` 保存一次完整模型权重，并立即调用 `scripts/evaluate.py` full eval。
- final 结束时也会保存并 eval。
- checkpoint registry 只保留综合 accuracy 最优和最新 checkpoint；其他完整权重会在 eval 成功后删除。

Full-SFT 输出结构：

```text
outputs/<run_name>/checkpoints/<checkpoint_id>/
outputs/<run_name>/evals/<checkpoint_id>/summary.json
outputs/<run_name>/evals/<checkpoint_id>/predictions_atbench.jsonl
outputs/<run_name>/evals/<checkpoint_id>/predictions_rjudge.jsonl
outputs/<run_name>/evals/<checkpoint_id>/failure_analysis_atbench.md
outputs/<run_name>/evals/<checkpoint_id>/failure_analysis_rjudge.md
outputs/<run_name>/eval_history.csv
outputs/<run_name>/eval_history.json
outputs/<run_name>/eval_history.md
outputs/<run_name>/checkpoint_registry.json
outputs/<run_name>/token_length_summary.json
outputs/<run_name>/data_split_summary.json
```

TensorBoard：

```text
logs/sft/李文博_<run_name>_<YYYYmmdd_HHMMSS>/
logs/only_eval/李文博_<run_name>_checkpoint_eval_<YYYYmmdd_HHMMSS>/
```

重要说明：如果使用 summer camp test set 的 eval 结果选择 checkpoint、调 prompt 或迭代数据集，该 eval set 已参与开发决策，不能再作为完全无偏最终测试集。

## Baseline 与 Reference 评测

评测配置在 `configs/eval_methods.yaml`。

```bash
python scripts/evaluate.py --method qwen35_08b_baseline
python scripts/evaluate.py --method qwen35_08b_reference
```

评测实现为 batch generate。默认 `eval_batch_size=64`，遇到 OOM 或 `canUse32BitIndexMath` 等可恢复错误时自动降到 32、16、8，并在 `summary.json` 中记录实际 batch size。

可使用本地 prompt 文件做评测实验：

```bash
python scripts/evaluate.py \
  --method qwen35_full_sft_llamafactory_best \
  --system-prompt-file /path/to/local_prompt.txt \
  --output-dir outputs/local_prompt_eval
```

具体 prompt 文件不提交到 Git；`configs/prompts/*.txt` 已加入 ignore。`summary.json` 只记录 prompt 文件路径和 sha256，便于本地复现。

每次评测都会写 TensorBoard 日志到 `logs/only_eval/李文博_<method>_<YYYYmmdd_HHMMSS>/`，并在对应 `summary.json` 中记录 `tensorboard_log_dir`。

## Final Eval

训练并导出 adapter 后：

```bash
python scripts/evaluate.py --method our_qwen35_continued_lora
python scripts/write_summary_csv.py
```

一次性评测全部方法：

```bash
python scripts/evaluate.py --method all
python scripts/write_summary_csv.py
```

输出目录：

```text
outputs/baseline_qwen35_08b/
outputs/reference_qwen35_08b/
outputs/final_continued_lora/
outputs/error_cases/
outputs/summary.csv
```

## 指标说明

每个模型、每个数据集输出：

- `accuracy`
- `unsafe_precision`
- `unsafe_recall`
- `unsafe_f1`
- `macro_f1`
- `invalid_output_rate`
- `strict_json_rate`
- `avg_input_tokens`
- `avg_output_tokens`
- `avg_total_tokens`
- `total_input_tokens`
- `total_output_tokens`
- `total_tokens`
- `token_cost_total_tokens`
- `estimated_cost_usd`
- `over_refusal_rate_safe_to_unsafe`
- `miss_rate_unsafe_to_safe`
- `raw_refusal_text_rate`
- `confusion_matrix`
- `inference_seconds`
- `avg_latency_seconds_per_sample`
- `throughput_samples_per_second`
- failure category count/rate

parser 顺序固定为：先解析 JSON，再匹配 `unsafe`，再匹配 `safe`。无法解析时默认 `unsafe`，并计入 invalid output。

failure analysis 覆盖：

- 是否忽略了中间工具调用
- 是否只看最终回复
- 是否把遇到风险但正确拒绝的轨迹误判为 unsafe
- 是否把用户意图恶意但 agent 没执行的轨迹误判为 unsafe
- 是否被长上下文干扰
- 是否输出格式不稳定

## 防泄漏约束

评测输入只渲染 trajectory/profile/tool 信息。以下测试字段不会输入模型：

```text
label, reason, risk_source, failure_mode, harm_type, risk_description, risk_type, source
```

禁止使用 summer camp test set 训练、做 few-shot、调 prompt 或选 checkpoint。
