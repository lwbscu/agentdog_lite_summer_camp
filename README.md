# AgentDoG-Lite Summer Camp

本项目实现 AgentDoG-Lite 题目 1 的高标准方案：最终主线不是普通 Step 1 Qwen baseline，而是从官方 `AI45Research/AgentDoG1.5-Qwen3.5-0.8B` 继续 LoRA/SFT 微调。原始 `Qwen/Qwen3.5-0.8B` 只作为比赛对照 baseline；官方 AgentDoG1.5 原始权重是 reference，不叫 baseline。

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

推荐使用 Python 3.10/3.11。当前项目的训练脚本会拒绝 Python 3.12+，避免深度学习依赖装上但行为不稳定。

```bash
conda env create -f environment.yml
conda activate agentdog-lite
pip install -e .
```

如果复用已有环境，至少需要：

```bash
pip install -r requirements.txt
pip install -e .
```

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
| `agentdog15_08b_reference` | `AI45Research/AgentDoG1.5-Qwen3.5-0.8B` | `models/AgentDoG1.5-Qwen3.5-0.8B` |
| teacher | `AI45Research/AgentDoG1.5-FG-Qwen3.5-0.8B` | `models/AgentDoG1.5-FG-Qwen3.5-0.8B` |

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
```

`train_mixed` 优先按 70% binary、20% diagnostic、10% hard boundary 构造；如果 hard boundary 数量不足以支撑稳定训练规模，则按规则回退到 75% binary、25% diagnostic。dev split 来自训练集内部，按 task/source/judgment 分层，绝不使用 summer camp test set 做 dev、few-shot 或 prompt tuning。

## Continued LoRA 训练

主线配置：

```bash
python scripts/train_lora.py --config configs/train_agentdog15_continued_lora.yaml
```

对照配置：

```bash
python scripts/train_lora.py --config configs/train_agentdog15_continued_lora_lr1e5.yaml
python scripts/train_lora.py --config configs/train_agentdog15_continued_lora_r8.yaml
```

如现场规则不允许从 AgentDoG1.5 初始化，保留 fallback：

```bash
python scripts/train_lora.py --config configs/train_qwen35_fallback_lora.yaml
```

训练脚本会校验 effective batch size 必须为 64。默认设置为 `per_device_train_batch_size=1`、`gradient_accumulation_steps=64`、单卡有效 batch 64。

导出最终 adapter：

```bash
python scripts/export_final_adapter.py \
  --run-dir outputs/our_agentdog15_continued_lora \
  --output-dir outputs/final_continued_lora_adapter
```

## Baseline 与 Reference 评测

评测配置在 `configs/eval_methods.yaml`。

```bash
python scripts/evaluate.py --method qwen35_08b_baseline
python scripts/evaluate.py --method agentdog15_08b_reference
```

## Final Eval

训练并导出 adapter 后：

```bash
python scripts/evaluate.py --method our_agentdog15_continued_lora
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
outputs/reference_agentdog15_08b/
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
- `over_refusal_rate_safe_to_unsafe`
- `miss_rate_unsafe_to_safe`
- `confusion_matrix`

parser 顺序固定为：先解析 JSON，再匹配 `unsafe`，再匹配 `safe`。无法解析时默认 `unsafe`，并计入 invalid output。

## 防泄漏约束

评测输入只渲染 trajectory/profile/tool 信息。以下测试字段不会输入模型：

```text
label, reason, risk_source, failure_mode, harm_type, risk_description, risk_type, source
```

禁止使用 summer camp test set 训练、做 few-shot、调 prompt 或选 checkpoint。

