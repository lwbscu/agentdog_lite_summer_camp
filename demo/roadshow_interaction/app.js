(function () {
  const cases = window.DEMO_CASES || [];
  const experiments = window.EXPERIMENT_RESULTS || [];
  const weightKeys = [
    ["atbench", "ATBench", "atbench", "覆盖真实工具/API、授权外发和隐藏副作用的安全评估数据。"],
    ["rjudge", "R-Judge", "rjudge", "偏重 prompt injection、越权指令和环境返回污染的轨迹数据。"],
    ["format", "Strict JSON", "format", "模型必须严格输出指定 JSON，便于自动评测解析。"],
    ["hard", "困难负样本", "hard", "容易误判的边界样本，用来区分风险文本和真实风险执行。"]
  ];

  const els = {
    caseButtons: document.getElementById("caseButtons"),
    caseDataset: document.getElementById("caseDataset"),
    caseTitle: document.getElementById("caseTitle"),
    goldLabel: document.getElementById("goldLabel"),
    predLabel: document.getElementById("predLabel"),
    riskType: document.getElementById("riskType"),
    errorDirection: document.getElementById("errorDirection"),
    userRequest: document.getElementById("userRequest"),
    actionSequence: document.getElementById("actionSequence"),
    defenseStrategy: document.getElementById("defenseStrategy"),
    metricAccuracy: document.getElementById("metricAccuracy"),
    metricMacroF1: document.getElementById("metricMacroF1"),
    metricOverRefusal: document.getElementById("metricOverRefusal"),
    metricMissRate: document.getElementById("metricMissRate"),
    metricStrictJson: document.getElementById("metricStrictJson"),
    timeline: document.getElementById("timeline"),
    stepBadge: document.getElementById("stepBadge"),
    stepDetail: document.getElementById("stepDetail"),
    attackerBubble: document.getElementById("attackerBubble"),
    defenderBubble: document.getElementById("defenderBubble"),
    evaluatorBubble: document.getElementById("evaluatorBubble"),
    arena: document.getElementById("arena"),
    playPauseBtn: document.getElementById("playPauseBtn"),
    replayBtn: document.getElementById("replayBtn"),
    voiceBtn: document.getElementById("voiceBtn"),
    expertBtn: document.getElementById("expertBtn"),
    expertModal: document.getElementById("expertModal"),
    closeExpertBtn: document.getElementById("closeExpertBtn"),
    atbenchWeight: document.getElementById("atbenchWeight"),
    rjudgeWeight: document.getElementById("rjudgeWeight"),
    formatWeight: document.getElementById("formatWeight"),
    hardWeight: document.getElementById("hardWeight"),
    atbenchWeightValue: document.getElementById("atbenchWeightValue"),
    rjudgeWeightValue: document.getElementById("rjudgeWeightValue"),
    formatWeightValue: document.getElementById("formatWeightValue"),
    hardWeightValue: document.getElementById("hardWeightValue"),
    mixBars: document.getElementById("mixBars"),
    matchedExperimentName: document.getElementById("matchedExperimentName"),
    labAcc: document.getElementById("labAcc"),
    labMacro: document.getElementById("labMacro"),
    labOver: document.getElementById("labOver"),
    labMiss: document.getElementById("labMiss"),
    promptInput: document.getElementById("promptInput"),
    promptDemoBtn: document.getElementById("promptDemoBtn"),
    promptType: document.getElementById("promptType"),
    promptAdvice: document.getElementById("promptAdvice"),
    sendPromptToArenaBtn: document.getElementById("sendPromptToArenaBtn"),
    templateSelect: document.getElementById("templateSelect"),
    liveModelSelect: document.getElementById("liveModelSelect"),
    liveStatus: document.getElementById("liveStatus"),
    liveLatency: document.getElementById("liveLatency"),
    liveRawOutput: document.getElementById("liveRawOutput"),
    liveParsedOutput: document.getElementById("liveParsedOutput"),
    trajectoryPreview: document.getElementById("trajectoryPreview")
  };

  const state = {
    caseIndex: 0,
    stepIndex: 0,
    playing: true,
    muted: true,
    timer: null,
    speechTimer: null,
    speechToken: 0,
    promptCaseId: "prompt_injection_false_alarm",
    liveModels: [],
    liveTemplates: [],
    apiAvailable: false,
    lastMatchedExperiment: null,
    lastLiveResult: null,
    liveCase: null
  };

  const TEMPLATES_LABELS = {
    scene_compile: "自然语言场景编译",
    direct_intent: "直接提示词安全判定",
    auto_agentdog: "自动执行轨迹",
    injection_readonly: "注入文本未执行",
    credential_exfiltration: "凭证/隐私外泄",
    hidden_side_effect: "隐藏工具副作用",
    authorized_outbound: "授权外发/二维码",
    correct_refusal: "正确拒绝危险动作",
    freeform_trajectory: "高级完整轨迹"
  };

  const AUTO_STEP_MS = 4300;
  const SPEECH_ADVANCE_DELAY_MS = 520;

  function pct(value) {
    if (typeof value !== "number") return "--";
    return `${Math.round(value * 100)}%`;
  }

  function metric(value) {
    if (typeof value !== "number") return "--";
    return value.toFixed(3);
  }

  function labelClass(value) {
    if (value === "safe") return "label-pill safe";
    if (value === "unsafe") return "label-pill unsafe";
    return "label-pill";
  }

  function sceneTypeLabel(type) {
    const labels = {
      agent_refused: "agent 已拒绝",
      agent_executed: "agent 已执行",
      agent_attempted: "agent 尝试执行",
      user_request_only: "仅用户请求",
      unknown: "低置信度场景"
    };
    return labels[type] || "未返回场景解释";
  }

  function confidenceLabel(confidence) {
    const labels = {
      high: "高",
      medium: "中",
      low: "低"
    };
    return labels[confidence] || "--";
  }

  function compiledTypeLabel(compiledType) {
    const value = String(compiledType || "");
    if (value.startsWith("safe_refusal")) return "拒绝/阻断轨迹";
    if (value.startsWith("safe_benign")) return "良性信息轨迹";
    if (value.startsWith("unsafe_default_execution")) return "默认照做风险轨迹";
    if (value.startsWith("unsafe_execution")) return "已执行风险轨迹";
    if (value.startsWith("unknown")) return "低置信度澄清轨迹";
    return value || "--";
  }

  function animationForResult(result) {
    const compiledType = String(result.compiled_trajectory_type || "");
    if (compiledType.startsWith("safe_refusal")) return "shield";
    if (compiledType.startsWith("unsafe")) return "breach";
    if (compiledType.startsWith("unknown")) return "verdict";
    return result.pred === "unsafe" ? "breach" : "shield";
  }

  function themeForResult(result) {
    const compiledType = String(result.compiled_trajectory_type || "");
    if (compiledType.startsWith("unsafe")) return "side_effect";
    if (compiledType.startsWith("safe_refusal")) return "refusal";
    if (compiledType.startsWith("safe_benign")) return "injection";
    return result.pred === "unsafe" ? "side_effect" : "refusal";
  }

  function sceneSummary(result) {
    const scene = result.scene_interpretation;
    if (!scene) {
      const label = TEMPLATES_LABELS[result.template_id] || result.template_title || "--";
      return `模型输入方式：${label}`;
    }
    return [
      "模型输入方式：自然语言场景编译",
      `后端解释：${sceneTypeLabel(scene.type)}`,
      `风险：${scene.risk_label || "--"}`,
      `置信度：${confidenceLabel(scene.confidence)}`
    ].join(" · ");
  }

  function sceneAdvice(result) {
    const scene = result.scene_interpretation;
    if (!scene) {
      return "专家模式直接使用指定输入口径；主路演建议使用自然语言场景编译。";
    }
    const trajectory = compiledTypeLabel(result.compiled_trajectory_type);
    if (scene.confidence === "low") {
      return `${scene.reason} 已构造为「${trajectory}」；现场可提示体验者补充 agent 是拒绝、执行还是调用了工具。`;
    }
    return `${scene.reason} 已构造为「${trajectory}」；模型看到的是右侧实时构造轨迹，而不是前端关键词标签。`;
  }

  function readWeights() {
    return {
      atbench: Number(els.atbenchWeight.value),
      rjudge: Number(els.rjudgeWeight.value),
      format: Number(els.formatWeight.value),
      hard: Number(els.hardWeight.value)
    };
  }

  function normalizeWeights(weights) {
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0) || 1;
    return Object.fromEntries(
      Object.entries(weights).map(([key, value]) => [key, value / total])
    );
  }

  function findClosestExperiment(weights) {
    const candidates = experiments.filter(experimentHasCheckpoint);
    if (!candidates.length) return null;
    const normalized = normalizeWeights(weights);
    let best = candidates[0];
    let bestDistance = Number.POSITIVE_INFINITY;
    candidates.forEach((experiment) => {
      const profile = normalizeWeights(experiment.weights);
      const distance = weightKeys.reduce((sum, [key]) => {
        return sum + Math.pow((normalized[key] || 0) - (profile[key] || 0), 2);
      }, 0);
      if (distance < bestDistance) {
        best = experiment;
        bestDistance = distance;
      }
    });
    return { experiment: best, distance: Math.sqrt(bestDistance) };
  }

  function modelById(modelId) {
    return state.liveModels.find((model) => model.id === modelId);
  }

  function setLiveStatus(text, className) {
    els.liveStatus.textContent = text;
    els.liveStatus.className = className || "";
  }

  function setLatency(text) {
    els.liveLatency.innerHTML =
      `<span class="term" tabindex="0" data-tooltip="Latency：从前端发起请求到后端模型返回判定的耗时。">Latency</span>: ${text}`;
  }

  function modelHasCheckpoint(model) {
    return Boolean(model && model.live_capable && !model.history_only);
  }

  function experimentHasCheckpoint(experiment) {
    if (!experiment || experiment.history_only) return false;
    if (!state.apiAvailable) return true;
    const model = modelById(experiment.live_model_id);
    return modelHasCheckpoint(model);
  }

  function firstCheckpointModel() {
    return state.liveModels.find(modelHasCheckpoint);
  }

  function syncModelSelect(experiment) {
    if (!experiment || !els.liveModelSelect.options.length) return;
    const target = experiment.live_model_id;
    const targetModel = modelById(target);
    if (modelHasCheckpoint(targetModel)) {
      els.liveModelSelect.value = target;
      return;
    }
    const fallback =
      [modelById("qwen35_full_sft_llamafactory_checkpoint"), modelById("full_sft_final"), firstCheckpointModel()]
        .find(modelHasCheckpoint);
    if (fallback) {
      els.liveModelSelect.value = fallback.id;
    }
  }

  async function fetchLiveModels() {
    try {
      const response = await fetch("/api/models", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      state.liveModels = data.models || [];
      state.liveTemplates = data.templates || [];
      state.apiAvailable = true;
      renderModelOptions(data.default_model_id || "qwen35_full_sft_llamafactory_checkpoint");
      const checkpointCount = state.liveModels.filter(modelHasCheckpoint).length;
      const loadedCount = state.liveModels.filter((model) => modelHasCheckpoint(model) && model.loaded).length;
      setLiveStatus(`实时后端：在线，${checkpointCount} 个 checkpoint 可选，已加载 ${loadedCount} 个`, "online");
    } catch (error) {
      state.apiAvailable = false;
      state.liveModels = [];
      els.liveModelSelect.innerHTML = '<option value="">后端未启动</option>';
      setLiveStatus("实时后端：未连接；请通过 VS Code 端口转发访问远程 8765", "offline");
    }
    renderExperimentLab();
  }

  function renderModelOptions(defaultModelId) {
    const checkpointModels = state.liveModels.filter(modelHasCheckpoint);
    const options = checkpointModels.map((model) => {
      const status = model.loaded ? "已加载" : "可加载";
      return `<option value="${model.id}">${model.title} · ${status}</option>`;
    });
    els.liveModelSelect.innerHTML = options.join("") || '<option value="">无 checkpoint 模型</option>';
    if (modelHasCheckpoint(modelById(defaultModelId))) {
      els.liveModelSelect.value = defaultModelId;
    } else if (modelHasCheckpoint(modelById("qwen35_full_sft_llamafactory_checkpoint"))) {
      els.liveModelSelect.value = "qwen35_full_sft_llamafactory_checkpoint";
    } else if (modelHasCheckpoint(modelById("full_sft_final"))) {
      els.liveModelSelect.value = "full_sft_final";
    } else if (checkpointModels[0]) {
      els.liveModelSelect.value = checkpointModels[0].id;
    }
  }

  function clearTimer() {
    if (state.timer) {
      window.clearTimeout(state.timer);
      state.timer = null;
    }
  }

  function clearSpeechTimer() {
    if (state.speechTimer) {
      window.clearTimeout(state.speechTimer);
      state.speechTimer = null;
    }
  }

  function cancelSpeech() {
    state.speechToken += 1;
    clearSpeechTimer();
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  }

  function activeCase() {
    return state.liveCase || cases[state.caseIndex];
  }

  function advanceStep() {
    const current = activeCase();
    if (!current) return;
    state.stepIndex = (state.stepIndex + 1) % current.dialogue.length;
    renderStep();
    scheduleNext();
  }

  function voiceControlsAdvance() {
    return !state.muted && "speechSynthesis" in window;
  }

  function scheduleNext() {
    clearTimer();
    if (!state.playing || voiceControlsAdvance()) return;
    state.timer = window.setTimeout(advanceStep, AUTO_STEP_MS);
  }

  function estimateSpeechTimeout(text) {
    const textLength = [...String(text || "")].length;
    return Math.min(18000, Math.max(5200, textLength * 210));
  }

  function speak(text) {
    if (state.muted || !("speechSynthesis" in window)) return;
    state.speechToken += 1;
    const token = state.speechToken;
    clearSpeechTimer();
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "zh-CN";
    utterance.rate = 0.98;
    utterance.pitch = 1;
    let queued = false;
    const queueNext = () => {
      if (queued || token !== state.speechToken || state.muted || !state.playing) return;
      queued = true;
      clearSpeechTimer();
      state.speechTimer = window.setTimeout(advanceStep, SPEECH_ADVANCE_DELAY_MS);
    };
    utterance.onend = queueNext;
    utterance.onerror = queueNext;
    window.speechSynthesis.speak(utterance);
    state.speechTimer = window.setTimeout(queueNext, estimateSpeechTimeout(text));
  }

  function initCaseButtons() {
    els.caseButtons.innerHTML = "";
    cases.forEach((item, index) => {
      const button = document.createElement("button");
      button.className = "case-button";
      button.type = "button";
      button.innerHTML = `<strong>${item.title}</strong><span>${item.dataset} · ${item.error_direction}</span>`;
      button.addEventListener("click", () => {
        state.liveCase = null;
        state.caseIndex = index;
        state.stepIndex = 0;
        state.playing = true;
        renderCase();
        scheduleNext();
      });
      els.caseButtons.appendChild(button);
    });
  }

  function renderCaseButtons() {
    [...els.caseButtons.children].forEach((button, index) => {
      button.classList.toggle("active", !state.liveCase && index === state.caseIndex);
    });
  }

  function renderCase() {
    const item = activeCase();
    if (!item) return;

    renderCaseButtons();
    els.caseDataset.textContent = item.dataset;
    els.caseTitle.textContent = item.title;
    els.goldLabel.textContent = `Gold: ${item.gold}`;
    els.predLabel.textContent = `Pred: ${item.pred}`;
    els.goldLabel.className = labelClass(item.gold);
    els.predLabel.className = labelClass(item.pred);
    els.riskType.textContent = item.risk_type;
    els.errorDirection.textContent = item.error_direction;
    els.userRequest.textContent = item.user_request_masked;
    els.actionSequence.textContent = item.action_sequence;
    els.defenseStrategy.textContent = item.defense_strategy;
    els.arena.dataset.theme = item.animation_theme;

    const metrics = item.metrics_snapshot;
    els.metricAccuracy.textContent = metric(metrics.accuracy);
    els.metricMacroF1.textContent = metric(metrics.macro_f1);
    els.metricOverRefusal.textContent = pct(metrics.over_refusal_rate_safe_to_unsafe);
    els.metricMissRate.textContent = pct(metrics.miss_rate_unsafe_to_safe);
    els.metricStrictJson.textContent = pct(metrics.strict_json_rate);

    renderTimeline();
    renderStep();
    els.playPauseBtn.textContent = state.playing ? "暂停" : "播放";
  }

  function renderExperimentLab() {
    const weights = readWeights();
    const normalized = normalizeWeights(weights);
    const matched = findClosestExperiment(weights);
    const templateId = "scene_compile";
    els.templateSelect.value = templateId;
    if (!state.lastLiveResult) state.promptCaseId = "prompt_injection_false_alarm";

    els.atbenchWeightValue.textContent = `${weights.atbench}%`;
    els.rjudgeWeightValue.textContent = `${weights.rjudge}%`;
    els.formatWeightValue.textContent = `${weights.format}%`;
    els.hardWeightValue.textContent = `${weights.hard}%`;

    els.mixBars.innerHTML = weightKeys
      .map(([key, label, className, tooltip]) => {
        const width = Math.round((normalized[key] || 0) * 100);
        return `
          <div class="mix-bar">
            <span><span class="term" tabindex="0" data-tooltip="${tooltip}">${label}</span></span>
            <div class="mix-track"><div class="mix-fill ${className}" style="width: ${width}%"></div></div>
            <span>${width}%</span>
          </div>
        `;
      })
      .join("");

    if (!matched) {
      state.lastMatchedExperiment = null;
      els.matchedExperimentName.textContent = "没有可实时调用的 checkpoint";
      els.labAcc.textContent = "--";
      els.labMacro.textContent = "--";
      els.labOver.textContent = "--";
      els.labMiss.textContent = "--";
      els.promptType.textContent = "模型输入方式：自然语言场景编译";
      els.promptAdvice.textContent = "当前没有本地 checkpoint 可选，请先确认模型目录或启动后端。";
      if (state.apiAvailable) {
        setLiveStatus("实时后端：在线；但没有可用 checkpoint 模型", "error");
      }
      return;
    }
    const { experiment, distance } = matched;
    state.lastMatchedExperiment = experiment;
    const similarity = Math.max(0, Math.round((1 - Math.min(distance, 1)) * 100));
    const liveBadge = experimentHasCheckpoint(experiment) ? "可实时调用" : "无 checkpoint";
    els.matchedExperimentName.textContent = `${experiment.title} · ${similarity}% · ${liveBadge}`;
    syncModelSelect(experiment);
    els.labAcc.textContent = metric(experiment.metrics.avg_accuracy);
    els.labMacro.textContent = metric(experiment.metrics.avg_macro_f1);
    els.labOver.textContent = pct(experiment.metrics.over_refusal);
    els.labMiss.textContent = pct(experiment.metrics.miss_rate);

    const templateLabel = TEMPLATES_LABELS[templateId] || "直接提示词安全判定";
    if (state.lastLiveResult) {
      els.promptType.textContent = sceneSummary(state.lastLiveResult);
      els.promptAdvice.textContent = sceneAdvice(state.lastLiveResult);
    } else {
      els.promptType.textContent = `模型输入方式：${templateLabel}`;
      els.promptAdvice.textContent =
        `${experiment.note} 后端会先解析自然语言场景，再构造成 AgentDoG 轨迹；危险文本本身不等于 unsafe，agent 拒绝或阻断应判 safe。`;
    }
    if (state.apiAvailable) {
      const model = modelById(els.liveModelSelect.value);
      const status = model && model.loaded ? "已加载" : "可按需加载";
      setLiveStatus(`实时后端：在线；已限定 checkpoint 模型；${status}`, "online");
    }
  }

  function liveResultToCase(result) {
    const pred = result.pred || "unsafe";
    const strictText = result.strict_json ? "strict JSON" : "非 strict JSON";
    const invalidText = result.invalid_output ? "格式异常" : "格式通过";
    const scene = result.scene_interpretation || {};
    const compiledLabel = compiledTypeLabel(result.compiled_trajectory_type);
    const finalAnimation = animationForResult(result);
    return {
      id: "live_eval_result",
      dataset: result.model_title,
      title: `实时模型评估：${pred.toUpperCase()}`,
      risk_type: scene.risk_label || result.template_title,
      gold: "n/a",
      pred,
      error_direction: `${result.parse_method} · ${strictText}`,
      user_request_masked: els.promptInput.value,
      action_sequence: `${compiledLabel} -> model.generate -> parse_model_output`,
      defense_strategy:
        pred === "unsafe"
          ? "模型判断编译后的轨迹存在危险执行、未授权副作用或高风险外发，需要拦截。"
          : "模型判断编译后的轨迹没有实际危险执行，重点保留 agent 拒绝或阻断证据。",
      animation_theme: themeForResult(result),
      metrics_snapshot: activeCase().metrics_snapshot || cases[0].metrics_snapshot,
      dialogue: [
        {
          speaker: "attacker",
          label: "体验者输入",
          text: els.promptInput.value.slice(0, 110) || "空输入",
          detail: "输入由后端解析为自然语言场景，不由前端关键词决定判定。",
          animation: "packet"
        },
        {
          speaker: "defender",
          label: "实时构造轨迹",
          text: `${sceneTypeLabel(scene.type)} · ${compiledLabel}`,
          detail: result.truncated ? "构造轨迹过长，已按评测逻辑截断。" : (scene.reason || "后端已构造 AgentDoG trajectory。"),
          animation: "scan"
        },
        {
          speaker: "evaluator",
          label: "模型输出",
          text: result.raw_output || "(empty output)",
          detail: `parse=${result.parse_method}，${strictText}，${invalidText}。`,
          animation: result.invalid_output ? "json" : "verdict"
        },
        {
          speaker: pred === "unsafe" ? "defender" : "evaluator",
          label: "现场结论",
          text:
            pred === "unsafe"
              ? "触发 unsafe 告警：需要阻断或人工复核。"
              : "判定 safe：没有看到实际危险副作用。",
          detail: `实时延迟 ${result.latency_seconds.toFixed(2)} 秒，输出 ${result.output_tokens} tokens。`,
          animation: finalAnimation
        }
      ]
    };
  }

  function renderLiveResult(result) {
    state.lastLiveResult = result;
    els.liveRawOutput.textContent = result.raw_output || "(empty output)";
    els.liveParsedOutput.textContent = JSON.stringify(
      {
        judgment: result.pred,
        strict_json: result.strict_json,
        invalid_output: result.invalid_output,
        parse_method: result.parse_method,
        input_mode: result.input_mode,
        compiled_trajectory_type: result.compiled_trajectory_type,
        scene_interpretation: result.scene_interpretation
      },
      null,
      2
    );
    setLatency(`${result.latency_seconds.toFixed(2)}s`);
    els.trajectoryPreview.textContent = result.trajectory_preview || "--";
    els.promptType.textContent = sceneSummary(result);
    els.promptAdvice.textContent = sceneAdvice(result);
    setLiveStatus(
      `实时后端：完成 · ${result.model_title} · ${result.template_title}`,
      result.invalid_output ? "error" : "online"
    );
  }

  function renderLiveError(message) {
    state.lastLiveResult = null;
    els.liveRawOutput.textContent = "--";
    els.liveParsedOutput.textContent = "--";
    setLatency("--");
    els.trajectoryPreview.textContent = "--";
    setLiveStatus(`实时后端：${message}`, "error");
  }

  async function evaluateLivePrompt() {
    state.lastLiveResult = null;
    renderExperimentLab();
    if (!state.apiAvailable) {
      renderLiveError("未连接，先用 server.py 启动本地后端");
      return;
    }
    const modelId = els.liveModelSelect.value;
    const model = modelById(modelId);
    if (!modelHasCheckpoint(model)) {
      renderLiveError("当前没有可实时调用的 checkpoint 模型");
      return;
    }
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 30000);
    els.promptDemoBtn.disabled = true;
    els.promptDemoBtn.textContent = "评估中...";
    setLiveStatus(`实时后端：调用 ${model.title}`, "online");
    try {
      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: modelId,
          user_prompt: els.promptInput.value,
          dataset_mix: readWeights()
        }),
        signal: controller.signal
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`);
      }
      renderLiveResult(data);
      state.liveCase = liveResultToCase(data);
      state.stepIndex = 0;
      state.playing = true;
      renderCase();
      scheduleNext();
    } catch (error) {
      renderLiveError(error.name === "AbortError" ? "请求超过 30 秒" : error.message);
    } finally {
      window.clearTimeout(timeout);
      els.promptDemoBtn.disabled = false;
      els.promptDemoBtn.textContent = "实时评估";
    }
  }

  function renderTimeline() {
    const item = activeCase();
    els.timeline.innerHTML = "";
    item.dialogue.forEach((step, index) => {
      const button = document.createElement("button");
      button.className = "timeline-step";
      button.type = "button";
      button.innerHTML = `<strong>${index + 1}. ${step.label}</strong><span>${step.detail}</span>`;
      button.addEventListener("click", () => {
        state.stepIndex = index;
        renderStep();
        scheduleNext();
      });
      els.timeline.appendChild(button);
    });
  }

  function resetEffects() {
    els.arena.dataset.effect = "";
    document.querySelectorAll(".actor").forEach((actor) => actor.classList.remove("active"));
    document
      .querySelectorAll(".packet, .shield, .scan-line, .tool-lens, .arena-vfx")
      .forEach((node) => {
        node.classList.remove("active");
        void node.offsetWidth;
      });
  }

  function activateEffect(animation) {
    resetEffects();
    els.arena.dataset.effect = animation;
    const actor = document.querySelector(`.actor.${currentStep().speaker}`);
    if (actor) actor.classList.add("active");

    const effectMap = {
      packet: [".packet"],
      inject: [".packet", ".cyber"],
      scan: [".scan-line", ".tool-lens", ".cyber"],
      shield: [".shield", ".magic", ".spark"],
      breach: [".packet", ".spark", ".cyber"],
      verdict: [".tool-lens", ".arena-vfx.magic", ".cyber"],
      qr: [".scan-line", ".tool-lens"],
      json: [".scan-line", ".shield", ".magic"]
    };

    const selectors = effectMap[animation] || effectMap.packet;
    selectors.forEach((selector) => {
      document.querySelectorAll(selector).forEach((node) => node.classList.add("active"));
    });
  }

  function currentStep() {
    return activeCase().dialogue[state.stepIndex];
  }

  function currentSpeechText() {
    const step = currentStep();
    return `${step.label}。${step.text}`;
  }

  function renderStep() {
    const item = activeCase();
    const step = currentStep();

    [...els.timeline.children].forEach((button, index) => {
      button.classList.toggle("active", index === state.stepIndex);
      button.classList.toggle("done", index < state.stepIndex);
    });

    const bubbles = {
      attacker: els.attackerBubble,
      defender: els.defenderBubble,
      evaluator: els.evaluatorBubble
    };

    Object.entries(bubbles).forEach(([speaker, bubble]) => {
      bubble.textContent =
        speaker === step.speaker ? step.text : idleText(speaker, item.animation_theme);
    });

    els.stepBadge.textContent = `STEP ${state.stepIndex + 1} · ${step.label}`;
    els.stepDetail.textContent = step.detail;
    activateEffect(step.animation);
    speak(currentSpeechText());
  }

  function idleText(speaker, theme) {
    const copy = {
      attacker: {
        injection: "载荷藏在工具返回里。",
        refusal: "请求试图推动自动发布。",
        side_effect: "只读外观掩盖副作用。",
        qr_sms: "外发工具名制造误导。",
        json: "旧输出混入多余格式。"
      },
      defender: {
        injection: "检查后续 action 是否执行。",
        refusal: "确认是否停止危险动作。",
        side_effect: "放大环境返回和参数字段。",
        qr_sms: "核对授权、内容和收件人。",
        json: "输出必须是单一 JSON。"
      },
      evaluator: {
        injection: "风险文本不等于风险执行。",
        refusal: "正确拒绝应被识别为防守成功。",
        side_effect: "最终回复不能覆盖中间副作用。",
        qr_sms: "工具名不是唯一证据。",
        json: "格式稳定才有可信评估。"
      }
    };
    return copy[speaker][theme] || "等待下一步证据。";
  }

  function togglePlayback() {
    state.playing = !state.playing;
    els.playPauseBtn.textContent = state.playing ? "暂停" : "播放";
    if (state.playing) {
      if (voiceControlsAdvance()) speak(currentSpeechText());
      scheduleNext();
    } else {
      clearTimer();
      cancelSpeech();
    }
  }

  function replay() {
    clearTimer();
    cancelSpeech();
    state.stepIndex = 0;
    state.playing = true;
    renderCase();
    scheduleNext();
  }

  function toggleVoice() {
    state.muted = !state.muted;
    els.voiceBtn.textContent = state.muted ? "语音关" : "语音开";
    if (state.muted) {
      cancelSpeech();
      scheduleNext();
    } else {
      clearTimer();
      speak(currentSpeechText());
      scheduleNext();
    }
  }

  function bindControls() {
    els.playPauseBtn.addEventListener("click", togglePlayback);
    els.replayBtn.addEventListener("click", replay);
    els.voiceBtn.addEventListener("click", toggleVoice);
    els.expertBtn.addEventListener("click", () => {
      els.expertModal.hidden = false;
      clearTimer();
      cancelSpeech();
    });
    els.closeExpertBtn.addEventListener("click", () => {
      els.expertModal.hidden = true;
      if (voiceControlsAdvance()) speak(currentSpeechText());
      scheduleNext();
    });
    els.expertModal.addEventListener("click", (event) => {
      if (event.target === els.expertModal) {
        els.expertModal.hidden = true;
        if (voiceControlsAdvance()) speak(currentSpeechText());
        scheduleNext();
      }
    });
    [
      els.atbenchWeight,
      els.rjudgeWeight,
      els.formatWeight,
      els.hardWeight
    ].forEach((input) => {
      input.addEventListener("input", renderExperimentLab);
    });
    els.promptInput.addEventListener("input", () => {
      state.lastLiveResult = null;
      renderExperimentLab();
    });
    els.templateSelect.addEventListener("change", renderExperimentLab);
    els.liveModelSelect.addEventListener("change", () => {
      const model = modelById(els.liveModelSelect.value);
      if (modelHasCheckpoint(model) && state.apiAvailable) {
        const status = model.loaded ? "已加载" : "可按需加载";
        setLiveStatus(`实时后端：在线；已限定 checkpoint 模型；${status}`, "online");
      }
    });
    els.promptDemoBtn.addEventListener("click", evaluateLivePrompt);
    els.sendPromptToArenaBtn.addEventListener("click", () => {
      if (state.lastLiveResult) {
        state.liveCase = liveResultToCase(state.lastLiveResult);
        state.stepIndex = 0;
        state.playing = true;
        renderCase();
        scheduleNext();
        return;
      }
      const index = cases.findIndex((item) => item.id === state.promptCaseId);
      if (index >= 0) {
        state.liveCase = null;
        state.caseIndex = index;
        state.stepIndex = 0;
        state.playing = true;
        renderCase();
        scheduleNext();
      }
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !els.expertModal.hidden) {
        els.expertModal.hidden = true;
        if (voiceControlsAdvance()) speak(currentSpeechText());
        scheduleNext();
      }
      if (event.key === " " && event.target === document.body) {
        event.preventDefault();
        togglePlayback();
      }
    });
  }

  function init() {
    initCaseButtons();
    bindControls();
    renderExperimentLab();
    renderCase();
    scheduleNext();
    fetchLiveModels();
  }

  init();
})();
