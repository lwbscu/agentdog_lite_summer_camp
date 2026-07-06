# AgentDoG-Lite Summer Camp

本项目实现 AgentDoG-Lite 题目 1 的高标准方案：最终主线从基础模型 `Qwen/Qwen3.5-0.8B` 出发，经过 LoRA/SFT 与 Full-SFT 训练，构建面向 Agent 轨迹级安全判断的轻量模型。

需要特别说明的是：

- 原始 `Qwen/Qwen3.5-0.8B` 是本项目最终主线训练起点，同时也作为比赛对照 baseline。
- 官方 `AI45Research/AgentDoG1.5-Qwen3.5-0.8B` 仅作为 reference 对照模型，用于比较官方 AgentDoG1.5 原始权重表现，不作为本项目最终主线初始化权重。
- 官方 `AI45Research/AgentDoG1.5-FG-Qwen3.5-0.8B` 作为 fine-grained teacher/reference，仅用于辅助分析与对照。

当前实验主线包含一套全参 Full-SFT + batch eval 闭环：使用本地 H800 环境对 `Qwen/Qwen3.5-0.8B` 做完整权重 SFT，每约 30 分钟保存 checkpoint，并立即在 summer camp test set 上 batch eval；只保留 best/latest 权重以节省磁盘。当前推荐本地权重入口为：

```text
outputs/qwen35_full_sft_llamafactory_h800/best_checkpoint
