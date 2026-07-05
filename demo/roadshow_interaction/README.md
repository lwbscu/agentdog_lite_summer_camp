# AI安全可信路演交互演示

这是一个路演 Web 演示。默认推荐用本地后端启动，可以实时调用本地模型做 AgentDoG trajectory safety eval；没有启动后端时，页面仍会保留历史实验指标回放。

## 打开方式

推荐在仓库根目录启动实时后端：

```bash
python demo/roadshow_interaction/server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --preload qwen35_baseline,agentdog15_reference,qwen35_full_sft_llamafactory_checkpoint
```

然后访问：

```text
http://127.0.0.1:8765/demo/roadshow_interaction/
```

如果显存不足，可以只加载 qwen35_full_sft_llamafactory_checkpoint：

```bash
python demo/roadshow_interaction/server.py --preload qwen35_full_sft_llamafactory_checkpoint
```

如只想查看静态效果，也可以直接打开 `demo/roadshow_interaction/index.html`，但实时模型调用不可用。

## 交互说明

- “数据配比实验台”可以拖动 ATBench、R-Judge、Strict JSON、困难负样本四个滑块，页面会匹配最接近的本地实验 checkpoint，并展示平均指标。
- “输入提示词体验”会先在后端解析自然语言场景，再编译成 AgentDoG 轨迹，并实时调用所选本地模型输出 safe/unsafe。
- “实时评估”展示后端场景解释、raw output、parsed judgment、strict JSON、latency 和脱敏轨迹预览。
- 点击“投入竞技场”会把最近一次实时模型输出切到攻防动画里。
- 左侧选择 5 个攻防剧情关卡。
- 顶部展示当前关卡对应的 Accuracy、Macro F1、误报率、漏报率和 Strict JSON。
- 中间区域自动播放攻击者、防守者、评估器三方对话和攻防特效。
- 底部时间轴可点击任一步，支持暂停、重播、语音开关。
- “研究依据”弹层展示本地 AgentDoG 分类体系和轨迹归因图。

## 数据与安全

实时后端只做文本分类推理，不执行任何真实工具或外部 API。演示数据来自本仓库已有报告和评测 summary 的脱敏摘要，不包含真实 token、邮箱、手机号或链上地址。完整素材来源见 `LICENSES.md`。
