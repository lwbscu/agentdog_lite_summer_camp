
本项目实现 AgentDoG-Lite 题目 1 的高标准方案：最终主是普通 Step 1 Qwen baseline继续 LoRA/SFT 微调。原始 `Qwen/Qwen3.5-0.8B` 作为比赛对照 baseline；官方 AgentDoG1.5 原始权重是 reference，不叫 baseline。

当前实验主线同时包含一套全参 Full-SFT + batch eval 闭环：使用本地 H800 环境对 `Qwen3.5-0.8B` 做完整权重 SFT，每约 30 分钟保存 checkpoint 并立即在 summer camp test set 上 batch eval；只保留 best/latest 权重以节省磁盘。当前推荐本地权重入口为：

```text
outputs/qwen35_full_sft_llamafactory_h800/best_checkpoint
```

注意：本地 `data/`、`models/`、`outputs/`、`logs/`、`report/` 均不提交到 Git。



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
@@ -167,6 +227,19 @@ python scripts/evaluate.py --method qwen35_08b_baseline
python scripts/evaluate.py --method agentdog15_08b_reference
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
@@ -209,12 +282,31 @@ outputs/summary.csv
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
