// 后端地址: 由 api-base.js (window.FBAPI) 在运行时解析 —— 本机控制台为同源 "",
// GitHub Pages 镜像默认连 http://127.0.0.1:8765, 可经 ?backend= / localStorage / 右上角「后端」配置。
const API = window.FBAPI ? window.FBAPI.base : "";
let selectedDataFile = null;
let selectedDataFileBars = 0; // 当前数据文件总根数（供「训练数据范围」默认值/提示）
let selectedSymbol = null;
let selectedStrategyFile = null;
let selectedStrategySymbol = null;
let chart = null;
let chartSymbol = null;
let pollTimer = null;
let clientErrors = [];
let debugMode = false;
let lastDebugViewContent = "";

// 分页与回测状态
let currentPage = "train";
let btActive = false;
let btBuster = "";      // 图表缓存刷新键（用 job 时间戳）
let btPortfolioSig = ""; // 绩效卡签名：变化时才重建 + 播放数字动画，避免每次轮询重播
let lastEquityData = null; // 最近一次资金曲线数据，供绩效卡 sparkline 复用
let btReportRunId = null; // 最近一次回测报告 run_id，用于与资金曲线对齐口径
let lastTrainingActive = false;
let trainActive = false;   // 概览轮询后同步的后端训练活动标志（供自适应轮询用）
let pollIntervalMs = 4000; // 当前轮询间隔：忙 4s / 空闲 12s
let pollWasBusy = true;
let btLastAlertKey = "";
let lastErrorPopupText = "";
let lastErrorPopupAt = 0;

const $ = (id) => document.getElementById(id);

const CPU_TRAINING_NOTE = `暂无报错

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【为什么用 CPU 训练，不用 GPU？】

你可以把 GPU 想象成一辆超大的货车，CPU 想象成一辆灵活的小电瓶车。

我们这个项目的训练，就像要做很多很多道「小题」：
每道题只算一点点数字，算完马上换下一道。
货车虽然一次能装很多，但每装卸一次都要准备很久才能再出发；
电瓶车一次装的少，但说走就走，一道接一道做，反而更快。

再打个比方：
GPU 像很多厨师一起做大锅饭，适合一次炒一大锅；
我们这个训练更像一道道菜分开炒，而且每道菜份量很小。
大锅饭团队每次开火、洗锅、集合都要时间，小菜一碟反而耽误在「准备」上。

所以具体原因是：
1. 每次要算的数据不多，GPU「启动一次计算」的等待，有时比真正算数还久。
2. 训练是一步接一步、一条公式接一条公式地指挥，GPU 经常闲着等下一道题，没法一直满负荷。
3. 数据还要在 CPU 和 GPU 之间来回搬运，也要花时间。

我们实测过（同样训练 50 步）：GPU 大约 4.5 秒一步，CPU 大约 1.9 秒一步。
这不是显卡坏了，也不是没装驱动，而是这个项目的做题方式，更适合 CPU。

说白了就是这个项目用CPU训练的速度比用GPU训练的速度更快`;

function emptyDebugMessage() {
  return debugMode ? "暂无日志" : CPU_TRAINING_NOTE;
}

function formatApiError(data, status, path) {
  const d = data?.detail;
  let detail = "";
  if (Array.isArray(d)) {
    detail = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
  } else if (typeof d === "string") {
    detail = d;
  } else if (d) {
    detail = JSON.stringify(d);
  }
  if (data?.traceback) {
    detail += `\n\n${data.traceback}`;
  }
  return detail || `HTTP ${status} ${path}`;
}

async function logClientError(message, context = {}) {
  const entry = `[${new Date().toLocaleString()}] ${message}`;
  clientErrors.push(entry);
  if (clientErrors.length > 80) clientErrors = clientErrors.slice(-80);
  renderDebugView();
  const silent = !!context.silent;
  if (!silent) {
    const detail = context.detail ? `${message}\n\n${context.detail}` : message;
    showErrorPopup("出错了", detail);
  }
  try {
    await fetch(API + "/api/debug/client-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level: "error", message, context }),
    });
  } catch (_) {
    /* server may be down */
  }
}

function showErrorPopup(title, detail) {
  const modal = $("errorModal");
  const titleEl = $("errorModalTitle");
  const detailEl = $("errorModalDetail");
  if (!modal || !detailEl) {
    window.alert(`${title}\n\n${detail}`);
    return;
  }
  const text = String(detail || "").trim() || "未知错误";
  const now = Date.now();
  if (text === lastErrorPopupText && now - lastErrorPopupAt < 2500) return;
  lastErrorPopupText = text;
  lastErrorPopupAt = now;
  if (titleEl) titleEl.textContent = title || "出错了";
  detailEl.textContent = text;
  modal.hidden = false;
}

function closeErrorPopup() {
  const modal = $("errorModal");
  if (modal) modal.hidden = true;
}

async function copyErrorPopupDetail() {
  const text = $("errorModalDetail")?.textContent || "";
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

function isViewAtBottom(el, threshold = 40) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

let lastServerTail = [];
let lastErrorTail = [];
let debugFilterText = "";
let lastGlobalErrKey = "";
let lastGlobalErrAt = 0;

function renderDebugView(serverLines = lastServerTail, errorLines = lastErrorTail) {
  const q = debugFilterText.trim().toLowerCase();
  const sections = [];
  if (clientErrors.length) {
    sections.push(["=== 前端报错 ===", clientErrors]);
  }
  if (errorLines.length) {
    sections.push(["\n=== 服务端错误日志 (logs/web_errors.log) ===", errorLines]);
  }
  if (serverLines.length) {
    sections.push(["\n=== 服务端运行日志 (logs/web_server.log) ===", serverLines]);
  }
  const parts = [];
  for (const [header, lines] of sections) {
    const kept = q ? lines.filter((l) => l.toLowerCase().includes(q)) : lines;
    if (kept.length) parts.push(header, ...kept);
  }
  const el = $("debugView");
  const atBottom = isViewAtBottom(el);
  let next;
  if (!parts.length) {
    next = q ? "（无匹配日志）" : emptyDebugMessage();
  } else {
    next = parts.join("\n");
  }
  const changed = next !== lastDebugViewContent;
  el.textContent = next;
  if (changed && atBottom && lastDebugViewContent) {
    el.scrollTop = el.scrollHeight;
  }
  lastDebugViewContent = next;
}

async function setDebugMode(enabled) {
  debugMode = !!enabled;
  $("debugModeCheck").checked = debugMode;
  try {
    await fetchJSON("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ debug_mode: debugMode }),
    });
  } catch (e) {
    await logClientError("切换调试模式失败: " + e.message);
  }
  await refreshDebugLogs();
}

async function setBgAnimation(enabled) {
  const on = !!enabled;
  $("bgAnimCheck").checked = on;
  window.__bgAnimationEnabled = on;
  if (window.__bgAnimationSetEnabled) window.__bgAnimationSetEnabled(on);
  try {
    await fetchJSON("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bg_animation: on }),
    });
  } catch (e) {
    await logClientError("切换背景动画失败: " + e.message);
  }
}

async function refreshDebugLogs() {
  try {
    const data = await fetch(API + "/api/debug/logs?lines=400").then(async (res) => {
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(formatApiError(json, res.status, "/api/debug/logs"));
      return json;
    });
    lastServerTail = data.server_tail || [];
    lastErrorTail = data.error_tail || [];
    $("debugLogPaths").textContent = `本地: ${data.error_log}`;
    renderDebugView();
  } catch (e) {
    renderDebugView();
  }
}

// 未捕获错误 / 未处理 Promise 拒绝: 自动进前端报错区并上报服务端错误日志;
// 同类错误 5s 内去抖, 避免轮询期一个坏接口刷屏。
function wireGlobalErrorCapture() {
  window.addEventListener("error", (ev) => {
    captureGlobalError(ev.message || "未知脚本错误", {
      source: ev.filename || "",
      line: ev.lineno || 0,
      col: ev.colno || 0,
    });
  });
  window.addEventListener("unhandledrejection", (ev) => {
    const r = ev.reason;
    const msg = r instanceof Error ? `${r.name}: ${r.message}` : String(r || "Promise rejected");
    captureGlobalError(msg, { promise: true });
  });
}

function captureGlobalError(message, context = {}) {
  const now = Date.now();
  if (message === lastGlobalErrKey && now - lastGlobalErrAt < 5000) return;
  lastGlobalErrKey = message;
  lastGlobalErrAt = now;
  logClientError(`[未捕获] ${message}`, { silent: true, ...context });
}

function wireDebugLogTools() {
  const filterEl = $("debugFilter");
  if (filterEl) {
    filterEl.addEventListener("input", () => {
      debugFilterText = filterEl.value;
      renderDebugView();
    });
  }
  const refreshBtn = $("debugRefreshBtn");
  if (refreshBtn) refreshBtn.addEventListener("click", () => refreshDebugLogs());
  const copyBtn = $("debugCopyBtn");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const text = $("debugView")?.textContent || "";
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
      } catch (_) {
        /* 剪贴板权限被拒时静默 */
      }
    });
  }
}

async function fetchJSON(path, opts = {}) {
  const silent = !!opts.silent;
  // 重试只对瞬时网络抖动有意义：默认 2 次；轮询/静默接口应显式传 retries:0，避免抖动时长时间阻塞
  const maxRetries = opts.retries != null ? Number(opts.retries) : 2;
  const retryDelayMs = opts.retryDelayMs != null ? Number(opts.retryDelayMs) : 2000;
  const fetchOpts = { ...opts };
  delete fetchOpts.silent;
  delete fetchOpts.retries;
  delete fetchOpts.retryDelayMs;
  // 静默 GET（后台轮询类）默认 15s 超时，防止请求悬挂后 tick 堆积；调用方可传 timeoutMs 覆盖
  let timeoutMs = opts.timeoutMs != null ? Number(opts.timeoutMs) : 0;
  delete fetchOpts.timeoutMs;
  const isGet = !fetchOpts.method || String(fetchOpts.method).toUpperCase() === "GET";
  if (!timeoutMs && silent && isGet) timeoutMs = 15000;

  let lastNetworkMsg = null;
  for (let attempt = 1; attempt <= Math.max(1, maxRetries); attempt++) {
    let res;
    let timer = null;
    let timedOut = false;
    if (timeoutMs > 0) {
      const ctrl = new AbortController();
      const extSignal = fetchOpts.signal || null;
      if (extSignal) {
        if (extSignal.aborted) ctrl.abort();
        else extSignal.addEventListener("abort", () => ctrl.abort(), { once: true });
      }
      timer = setTimeout(() => {
        timedOut = true;
        ctrl.abort();
      }, timeoutMs);
      fetchOpts.signal = ctrl.signal;
    }
    try {
      res = await fetch(API + path, fetchOpts);
    } catch (e) {
      // 请求被中止：区分「内部超时」与「外部取消（页面切换/用户取消）」。
      if (e && (e.name === "AbortError" || /abort/i.test(e.message || ""))) {
        if (timedOut) {
          lastNetworkMsg = `请求超时 ${path} (>${timeoutMs}ms)`;
          if (attempt < maxRetries) {
            await new Promise((r) => setTimeout(r, retryDelayMs));
            continue;
          }
          await logClientError(lastNetworkMsg, { path, silent, attempts: attempt });
          throw new Error(lastNetworkMsg);
        }
        const abortErr = new Error("请求已取消（超时或页面切换）");
        abortErr.name = "AbortError";
        throw abortErr;
      }
      lastNetworkMsg = `网络错误 ${path}: ${e.message}`;
      if (attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, retryDelayMs));
        continue;
      }
      await logClientError(lastNetworkMsg, { path, silent, attempts: attempt });
      throw new Error(lastNetworkMsg);
    } finally {
      if (timer) clearTimeout(timer);
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = formatApiError(data, res.status, path);
      await logClientError(`${path} -> ${msg}`, { path, status: res.status, silent });
      if (!silent) await refreshDebugLogs();
      throw new Error(msg);
    }
    return data;
  }
  throw new Error(lastNetworkMsg || `网络错误 ${path}`);
}

function formatScore(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(4);
}

function renderDataFileCard(info) {
  const card = $("dataFileCard");
  const startBtn = $("startBtn");

  if (!info || !info.data_file) {
    card.className = "data-file-card";
    card.innerHTML = '<div class="data-file-empty">尚未选择数据文件</div>';
    selectedDataFile = null;
    selectedDataFileBars = 0;
    selectedSymbol = null;
    startBtn.disabled = true;
    if ($("retrainBtn")) $("retrainBtn").disabled = true;
    if ($("exportBtn")) $("exportBtn").disabled = true;
    if ($("exportTrainingBtn")) $("exportTrainingBtn").disabled = true;
    if ($("importTrainingBtn")) $("importTrainingBtn").disabled = true;
    syncExpRunBtn();
    return;
  }

  selectedDataFile = info.data_file;
  selectedSymbol = info.symbol || null;

  if (info.valid === false) {
    card.className = "data-file-card invalid";
    card.innerHTML = `
      <div class="data-file-error">${info.message || "文件无效"}</div>
      <div class="data-file-path">${info.data_file}</div>
    `;
    startBtn.disabled = true;
    if ($("retrainBtn")) $("retrainBtn").disabled = true;
    if ($("exportTrainingBtn")) $("exportTrainingBtn").disabled = true;
    if ($("importTrainingBtn")) $("importTrainingBtn").disabled = true;
    return;
  }

  card.className = "data-file-card valid";
  const warn = [];
  if (info.duplicate_ts) warn.push(`重复时间戳 ${info.duplicate_ts.toLocaleString()} 根`);
  if (info.missing_columns && info.missing_columns.length) warn.push(`缺少列: ${info.missing_columns.join(", ")}`);
  if (info.checks_ok === false && !warn.length) warn.push("文件检查未通过");
  const warnBlock = warn.length
    ? `<div class="data-file-warn">⚠ ${warn.join("；")}</div>`
    : "";
  const summaryBits = [];
  if (info.bars_unique != null) {
    summaryBits.push(
      `去重后 ${info.bars_unique.toLocaleString()} 根` +
      (info.duplicate_ts ? `（原始 ${info.raw_bars.toLocaleString()}）` : "")
    );
  }
  if (info.start_date && info.end_date) summaryBits.push(`时间 ${info.start_date} → ${info.end_date}`);
  if (!warn.length && info.missing_columns) summaryBits.push("列完整");
  const summaryRow = summaryBits.length
    ? `<div class="data-file-summary">${summaryBits.join(" · ")}</div>`
    : "";
  const yearsText = info.years_h1 != null ? `${info.years_h1} 年` : "—";
  const dlBits = [];
  if (info.download_count != null) dlBits.push(`累计下载 ${Number(info.download_count).toLocaleString()} 次`);
  if (info.first_downloaded_at) dlBits.push(`最早批次 ${String(info.first_downloaded_at).slice(0, 10)}`);
  const dlRow = dlBits.length
    ? `<div class="data-file-summary" title="由 sidecar 元数据记录每次落盘（新建/追加/回溯/覆盖），无新增的重复下载不计数">${dlBits.join(" · ")}</div>`
    : "";
  const srcLabel = { tradingview: "TradingView", binance: "Binance", okx: "OKX", tongdaxin: "通达信" }[info.download_source];
  const srcText = srcLabel
    ? (info.downloaded_at ? `${srcLabel} · ${String(info.downloaded_at).slice(0, 10)}` : srcLabel)
    : "本地文件";
  // 切片/局部窗口标注：与同名全量档案区分（如 data/slices/BTCUSDT_M5.parquet 40,000 根 vs 全量 949,910 根）
  const sliceOf = info.is_slice ? (info.slice_of || {}) : null;
  const sliceBadge = info.is_slice
    ? `<span class="slice-badge">切片/局部窗口</span>`
    : "";
  const sliceTitle = info.is_slice
    ? `切片/局部窗口文件：${info.data_file}${sliceOf && sliceOf.bars != null ? `（全量 ${Number(sliceOf.bars).toLocaleString()} 根）` : "（未找到同名全量档案）"}`
    : (info.download_source ? "联网下载自 " + srcLabel : "本地文件（未标记下载来源）");
  card.innerHTML = `
    <div class="data-file-row">
      <div class="item"><span class="label">品种</span><span class="value sym">${info.symbol}</span></div>
      <div class="item"><span class="label">周期</span><span class="value">${info.timeframe}</span></div>
      <div class="item"><span class="label">K线</span><span class="value">${info.bars?.toLocaleString()}${sliceBadge}</span></div>
      <div class="item"><span class="label">数据年限</span><span class="value">${yearsText}</span></div>
      <div class="item"><span class="label">数据来源</span><span class="value" title="${sliceTitle}">${srcText}</span></div>
      <div class="item"><span class="label">进度</span><span class="value" id="fileProgressPct">—</span></div>
      <div class="item"><span class="label">预计剩余 / 完成</span><span class="value" id="fileEta">—</span></div>
      <div class="item"><span class="label">本次训练时长</span><span class="value" id="fileElapsedTime">—</span></div>
      <div class="item"><span class="label">历史训练总时长</span><span class="value" id="fileHistoryElapsedTime">—</span></div>
      <div class="item"><span class="label">最优分数</span><span class="value score-best" id="fileBestScore">—</span></div>
      <div class="item"><span class="label">验证分数</span><span class="value score-val" id="fileValScore">—</span></div>
      <div class="item"><span class="label">样本外验证</span><span class="value" id="fileHoldout">—</span></div>
    </div>
    ${warnBlock}${summaryRow}${dlRow}
    <div class="path" title="${info.data_file}">${info.filename || info.data_file}${sliceBadge}</div>
  `;
  selectedDataFileBars = info.bars != null ? Number(info.bars) : info.bars_unique || 0;
  startBtn.disabled = false;
  if ($("retrainBtn")) $("retrainBtn").disabled = false;
  refreshTrainRangeUI();
  syncExpRunBtn();
}

// ── 训练数据范围（全部 / 最近 N 根 / 全历史分块 N 根）──────────────
function trDataModeVal() {
  const el = $("trDataMode");
  return (el && el.value) || "full";
}
function trN() {
  const el = $("trNBars");
  const v = Number((el && el.value) || "");
  return Number.isFinite(v) && v > 0 ? v : 0;
}
function trC() {
  const el = $("trChunks");
  const v = Number((el && el.value) || "");
  return Number.isFinite(v) && v >= 2 ? v : null;
}
function trainRangeOk() {
  const mode = trDataModeVal();
  if (mode === "full") return true;
  return trN() >= 5000;
}
function collectTrainRange() {
  const mode = trDataModeVal();
  return {
    data_mode: mode,
    n_bars: mode === "full" ? null : trN() || null,
    n_chunks: mode === "spread" ? trC() : null,
  };
}
function refreshTrainRangeUI() {
  const mode = trDataModeVal();
  const total = selectedDataFileBars || 0;
  const nf = $("trNBarsField");
  const cf = $("trChunksField");
  if (nf) nf.hidden = mode === "full";
  if (cf) cf.hidden = mode !== "spread";
  // 选中文件且未手填时给个默认根数（≤10 万，≤全量）
  const nEl = $("trNBars");
  if (nEl) {
    nEl.max = Math.max(5000, total);
    if (total && nEl.value === "" && mode !== "full") {
      nEl.value = String(Math.min(100000, total));
    }
  }
  const hint = $("trRangeHint");
  if (hint) {
    const fmt = (v) => Number(v).toLocaleString();
    if (mode === "full") {
      hint.textContent = total ? `全部 ${fmt(total)} 根（默认，与现状一致）` : "全部数据（默认）";
    } else if (mode === "tail") {
      hint.textContent = total
        ? `取最近 ${fmt(trN() || 0)} 根（源共 ${fmt(total)} 根）。连续尾部切片，最贴近“近期市场更像未来”，统计最干净。`
        : "选择数据文件后可见总根数；取最近 N 根训练";
    } else {
      const c = trC() || "自动";
      hint.textContent = total
        ? `按年代抽连续块，共约 ${fmt(trN() || 0)} 根（块数 ${c}，覆盖全历史不同行情；末尾自动保留近期连续段做验证/holdout，保证样本外干净）。`
        : "选择数据文件后可见总根数；从全历史按年代抽取 N 根";
    }
  }
  // 只有非训练中才联动开始/重训按钮（训练中由 updateTrainingUI 管理禁用）
  const pill = $("jobPill");
  const running = !!(pill && /训练中/.test(pill.textContent || ""));
  if (!running) {
    const ok = !!selectedDataFile && trainRangeOk();
    const s = $("startBtn");
    const r = $("retrainBtn");
    if (s) s.disabled = !ok;
    if (r) r.disabled = !ok;
  }
  maybeScheduleSubsetPreview(); // 输入签名未变则不重复重算（轮询每 4s 也会走到这里）
}

// ── 取样区间可视化（训练数据范围预览，调后端 preview 接口不写文件）─────
let _trPreviewTimer = null;
// 预览去重：只在 数据文件/模式/N/块数 签名变化时才真正重算；
// 签名不变时（如 overview 轮询/页面重绘）直接跳过，避免 950k 文件反复重新计算。
let __trPreviewSchedSig = ""; // 已调度/已渲染成功的签名
let __trPreviewFailAt = 0; // 最近一次失败时刻：同签名仅在失败后 15s 窗口内允许重试

function trPreviewSig() {
  const mode = trDataModeVal();
  return [
    selectedDataFile || "",
    mode,
    mode === "full" ? "" : String(trN() || ""),
    mode === "spread" ? String(trC() || "") : "",
  ].join("|");
}

function maybeScheduleSubsetPreview() {
  const sig = trPreviewSig();
  if (sig === __trPreviewSchedSig) {
    // 相同签名：若上次请求失败，给一个短重试窗口，否则不再触发重算
    if (__trPreviewFailAt && Date.now() - __trPreviewFailAt < 15000) scheduleSubsetPreview();
    return;
  }
  __trPreviewSchedSig = sig;
  scheduleSubsetPreview();
}
const TR_BLOCK_COLORS = {
  warmup: "#64748b55",     // 核心前的前置行情（滚动特征 warm-up）
  recent: "#a78bfa",       // 末尾近期连续段（验证/holdout 干净区）
  r0: "#22c55e",           // 低波动年代块
  r1: "#eab308",           // 中波动
  r2: "#ef4444",           // 高波动
  rn: "#94a3b8",           // 无 regime 信息（等分模式）
};

function scheduleSubsetPreview() {
  if (_trPreviewTimer) clearTimeout(_trPreviewTimer);
  _trPreviewTimer = setTimeout(() => {
    _trPreviewTimer = null;
    drawSubsetPreview();
  }, 220);
}

function blockColor(b) {
  if (b.kind === "recent") return TR_BLOCK_COLORS.recent;
  if (b.kind === "warmup") return TR_BLOCK_COLORS.warmup;
  if (b.kind === "core") {
    const c = b.regime_class;
    if (c === 0) return TR_BLOCK_COLORS.r0;
    if (c === 1) return TR_BLOCK_COLORS.r1;
    if (c === 2) return TR_BLOCK_COLORS.r2;
    return TR_BLOCK_COLORS.rn;
  }
  return TR_BLOCK_COLORS.rn;
}

function blockLabel(b) {
  const c = b.regime_class;
  if (b.kind === "recent") return "末尾近期段";
  if (b.kind === "warmup") return "前置 warm-up";
  if (b.kind === "core" && c !== null && c !== undefined) {
    return { 0: "低波动", 1: "中波动", 2: "高波动" }[c] || "";
  }
  return b.kind === "core" ? "年代块" : "";
}

async function drawSubsetPreview() {
  const box = $("trPreview");
  if (!box) return;
  const mode = trDataModeVal();
  const file = selectedDataFile;
  const n = trN();
  if (!file || mode === "full" || n < 5000) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const track = $("trPreviewTrack");
  const legend = $("trPreviewLegend");
  const note = $("trPreviewNote");
  track.textContent = "计算取样区间…";
  try {
    const r = await fetchJSON("/api/training/subset-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data_file: file,
        mode,
        n_bars: n || null,
        n_chunks: mode === "spread" ? trC() : null,
        regime: "vol",
      }),
      silent: true,
    });
    const total = r.source_bars || 0;
    if (!total || !r.blocks || !r.blocks.length) {
      box.hidden = true;
      return;
    }
    const fmt = (v) => Number(v).toLocaleString();
    const pct = (s, e) => {
      const w = ((e - s) / total) * 100;
      return w < 0.25 ? 0.25 : w;
    };
    track.innerHTML = r.blocks
      .map((b) => {
        const label = blockLabel(b);
        return `<span class="tr-preview-block" title="${label} · ${fmt(b.start)}–${fmt(b.end)} 根 (${fmt(b.end - b.start)})"
          style="left:${(b.start / total) * 100}%;width:${pct(b.start, b.end)}%;background:${blockColor(b)}"></span>`;
      })
      .join("");
    const legParts = [
      `<span><i style="background:${TR_BLOCK_COLORS.warmup}"></i>前置行情</span>`,
      `<span><i style="background:${TR_BLOCK_COLORS.recent}"></i>末尾近期段</span>`,
    ];
    if (r.regime !== "equal") {
      legParts.unshift(
        `<span><i style="background:${TR_BLOCK_COLORS.r0}"></i>低波动</span>`,
        `<span><i style="background:${TR_BLOCK_COLORS.r1}"></i>中波动</span>`,
        `<span><i style="background:${TR_BLOCK_COLORS.r2}"></i>高波动</span>`,
      );
    }
    legend.innerHTML = legParts.join("");
    const cov = r.regime_coverage || {};
    let txt = `实际取样约 ${fmt(r.total_bars || 0)} 根 · ${r.n_chunks_used || 1} 段`;
    if (r.mode !== "spread") {
      txt += " · ⚠ 数据不足，spread 自动退化为尾部切片";
    } else if (r.regime !== "equal") {
      txt += ` · 波动覆盖 低${cov.low || 0} / 中${cov.mid || 0} / 高${cov.high || 0} 段`;
    }
    note.textContent = txt;
  } catch (_e) {
    box.hidden = true;
    __trPreviewFailAt = Date.now(); // 失败后 15s 窗口内允许同签名重试
  }
}

// ── 范围对比实验（tail vs spread，训练页）──────────────────────────
let cmpActive = false;
let compareChart = null;

function syncExpRunBtn() {
  const run = $("expRunBtn");
  if (!run) return;
  run.disabled = cmpActive || !selectedDataFile;
  const src = $("expSourceName");
  if (src) {
    src.textContent = selectedDataFile
      ? `数据文件：${selectedDataFile.split("/").pop()}` +
        (selectedDataFileBars ? `（${Number(selectedDataFileBars).toLocaleString()} 根）` : "")
      : "先在上方选择数据文件";
  }
  refreshExpMatrixBest(); // 同品种 N×N 最优组合（回测页跑完矩阵后在此并排展示）
}

// ── 训练页对比面板 · 并排展示同品种 N×N 最优组合（资金曲线 + 最大回撤约束）──
let __expMatrixCache = { sym: "", best: null, ts: 0 }; // matrix-best 按品种短缓存
let __expMatrixSig = ""; // 已渲染内容签名
let __expMatrixBusy = false;
// 训练-组合选型同屏联动：训练进度达到阈值时自动预取同品种 N×N 最优组合曲线（每训练任务一次）
const EXP_MX_PREFETCH_PCT = 0.25; // ≥25% 进度（向下取整 ≥6 步、≤40 步封顶）触发
let __expPrefetchKey = ""; // 已触发过的训练任务 key（symbol|train_steps|已到步数）
let __expPrefetchNote = false; // 该任务是否已在面板里挂过「自动预取」角标
let __expPrefetchScrolled = false; // 是否已把对比面板轻滚入视野（每任务一次）

// 训练中途里程碑：强制刷新（绕过 10s TTL）同品种矩阵最优 + 曲线，并排进对比面板；
// 有结果则可见并带「已自动预取」标注，一次软滚动让用户看到训练与组合选型同屏。
async function expMatrixPrefetchOnMilestone(symbol, progress) {
  if (!symbol) return;
  const cur = Number((progress && progress.current_step) || 0);
  const total = Number((progress && progress.train_steps) || 0);
  if (cur < 1) return;
  const target = Math.max(6, Math.min(40, Math.ceil((total || 60) * EXP_MX_PREFETCH_PCT)));
  if (cur < target) return;
  const key = symbol + "|" + (total || cur) + "|" + cur;
  if (key === __expPrefetchKey) return; // 同一训练任务已预取过
  __expPrefetchKey = key;
  __expPrefetchNote = false;
  __expPrefetchScrolled = false;
  const run = () => {
    __expMatrixCache = { sym: "", best: null, ts: 0 }; // 强制重新拉取
    return refreshExpMatrixBest();
  };
  if (__expMatrixBusy) {
    setTimeout(run, 3000); // 与轮询在途请求错峰后重试
    return;
  }
  await run();
  const box = $("expMatrixBest");
  if (!box) return;
  if (!box.hidden && !__expPrefetchNote) {
    __expPrefetchNote = true;
    box.querySelectorAll(".exp-mx-prefetched").forEach((n) => n.remove()); // 换训练任务时去掉旧角标
    const note = document.createElement("div");
    note.className = "exp-mx-prefetched";
    note.textContent = `⚡ 训练进行到第 ${cur} 步已自动预取同品种 N×N 最优组合 — 与下方 tail/spread 结果或训练曲线并排参考`;
    box.appendChild(note);
  }
  if (!box.hidden && !__expPrefetchScrolled && currentPage === "train") {
    __expPrefetchScrolled = true;
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function svgEqMini(labels, equity) {
  if (!labels || !equity || labels.length < 2) return "";
  const pts = labels
    .map((x, i) => ({ x: Number(x), y: Number(equity[i]) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (pts.length < 2) return "";
  let lo = Infinity, hi = -Infinity;
  pts.forEach((p) => { if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y; });
  if (!(hi > lo)) { const m = hi || 1; lo = m * 0.99; hi = m * 1.01; }
  const pad = (hi - lo) * 0.1 || 1e-9;
  lo -= pad; hi += pad;
  const W = 100, T = 8, B = 92;
  const x = (i) => 1.5 + ((W - 30) * i) / (pts.length - 1);
  const y = (v) => T + ((B - T) * (hi - v)) / (hi - lo);
  const d = pts.map((p, i) => (i ? "L" : "M") + x(i).toFixed(2) + " " + y(p.y).toFixed(2)).join("");
  let base = "";
  if (1.0 >= lo && 1.0 <= hi) {
    const by = y(1.0).toFixed(2);
    base = `<line x1="1.5" y1="${by}" x2="${(W - 28).toFixed(2)}" y2="${by}" stroke="#64748b" stroke-width="0.6" stroke-dasharray="3 3"/>`;
  }
  const lastX = x(pts.length - 1).toFixed(2);
  return `<svg class="exp-mx-svg" viewBox="0 0 ${W} 100" preserveAspectRatio="none">` +
    `<path d="M1.5 ${y(pts[0].y).toFixed(2)} L ${lastX} ${y(pts[pts.length - 1].y).toFixed(2)} L ${lastX} 100 L 1.5 100 Z" fill="rgba(56,189,248,0.07)"/>` +
    `<path d="${d}" fill="none" stroke="#38bdf8" stroke-width="1.1" vector-effect="non-scaling-stroke"/>` +
    base +
    `</svg>`;
}

async function refreshExpMatrixBest() {
  const box = $("expMatrixBest");
  if (!box) return;
  const sym = selectedSymbol;
  if (!sym) {
    box.hidden = true;
    __expMatrixSig = "";
    return;
  }
  const now = Date.now();
  let best =
    __expMatrixCache.sym === sym && now - __expMatrixCache.ts < 10000
      ? __expMatrixCache.best
      : null;
  if (!best) {
    if (__expMatrixBusy) return; // 上一轮在途，下一轮轮询再试
    __expMatrixBusy = true;
    try {
      const d = await fetchJSON(
        "/api/backtest/matrix-best?symbol=" + encodeURIComponent(sym),
        { silent: true, retries: 0 }
      );
      best = d && d.exists ? d.best : null;
    } catch (_) {
      best = null;
    }
    __expMatrixBusy = false;
    __expMatrixCache = { sym, best, ts: Date.now() };
  }
  if (!best || !best.combo || (best.symbol && best.symbol !== sym)) {
    box.hidden = true;
    __expMatrixSig = "";
    return;
  }
  box.hidden = false;
  const sig = `${sym}|${best.combo}|${best.generated_at || ""}|${best.window_bars || ""}|${best.window_mode || ""}|${best.sharpe}|${best.max_drawdown}`;
  if (sig === __expMatrixSig) return; // 已渲染（含曲线），无需重建
  __expMatrixSig = sig;
  const meta = $("expMatrixMeta");
  const curve = $("expMatrixCurve");
  if (meta) {
    const ddS =
      best.max_drawdown != null
        ? `最大回撤约束 ≤ ${(Math.abs(Number(best.max_drawdown)) * 100).toFixed(1)}%`
        : "最大回撤 —";
    const sh = best.sharpe != null ? `Sharpe ${Number(best.sharpe).toFixed(2)}` : "Sharpe —";
    const winTxt =
      best.window_mode === "spread"
        ? `窗口 分层抽样 ${best.window_bars || "?"} 根`
        : best.window_bars
          ? `窗口 样本外尾部 ${best.window_bars} 根`
          : "窗口 全部历史";
    meta.innerHTML =
      `<b>${escHtml(comboDisplayName(best.combo))}</b>` +
      `<span>${escHtml(sh)}</span>` +
      `<span>${escHtml(ddS)}</span>` +
      `<span>${escHtml(winTxt)}</span>`;
  }
  const hint = $("expMatrixHint");
  if (hint) {
    hint.textContent = `来自回测页最近一次全组合回测（results/matrix_best_combo.json）；跑新 N×N 后自动更新。与 tail/spread 结果并排参考。`;
  }
  if (curve) {
    curve.innerHTML = '<div class="exp-mx-loading">拉取最优组合资金曲线…</div>';
    try {
      const c = await fetchJSON(
        "/api/backtest/hold-matrix/curve?combo=" + encodeURIComponent(best.combo),
        { silent: true, retries: 0 }
      );
      curve.innerHTML = c && c.available
        ? svgEqMini(c.labels, c.equity)
        : `<div class="metric-empty">${escHtml((c && c.error) || "暂无该组合曲线 — 回测页先跑一次「一键全组合回测」")}</div>`;
    } catch (_) {
      curve.innerHTML = '<div class="metric-empty">曲线拉取失败（矩阵侧车不存在，请先跑 N×N）</div>';
    }
  }
}

async function startCompareExp() {
  if (!selectedDataFile) {
    await logClientError("请先选择数据文件");
    return;
  }
  const n = Number(($("expNBars") || {}).value || 0);
  const steps = Number(($("expSteps") || {}).value || 0);
  if (!Number.isFinite(n) || n < 5000) {
    await logClientError("根数 N 至少 5000");
    return;
  }
  if (!Number.isFinite(steps) || steps < 1) {
    await logClientError("训练步数至少 1");
    return;
  }
  const chunksEl = $("expChunks");
  const chunks = chunksEl && chunksEl.value ? Number(chunksEl.value) : null;
  const seeds = (($("expSeeds") || {}).value || "42").trim();
  const repSel = $("expRepSel");
  const repCriterion = repSel ? repSel.value : "holdout_median";
  try {
    const res = await fetchJSON("/api/experiment/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data_file: selectedDataFile,
        n_bars: Math.round(n),
        chunks: chunks,
        steps: Math.round(steps),
        seeds,
        rep_criterion: repCriterion,
      }),
    });
    cmpActive = true;
    $("expResult").hidden = true;
    $("expLog").hidden = false;
    $("expStatusHint").textContent =
      `对比实验已启动（tail vs spread × steps=${steps}），串行执行，可切页等它跑完…`;
    await refreshCompareExp();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function stopCompareExp() {
  try {
    await fetchJSON("/api/experiment/compare-stop", { method: "POST" });
    await refreshCompareExp();
  } catch (_) {}
}

async function refreshCompareExp() {
  let st;
  try {
    st = await fetchJSON("/api/experiment/compare-status", { silent: true });
  } catch (_) {
    return;
  }
  cmpActive = !!st.active;
  const job = st.job;
  syncExpRunBtn();
  const stop = $("expStopBtn");
  if (stop) stop.disabled = !cmpActive;
  const logEl = $("expLog");
  if (logEl && st.log_tail && st.log_tail.length) {
    logEl.hidden = false;
    logEl.textContent = st.log_tail.join("\n");
    logEl.scrollTop = logEl.scrollHeight;
  }
  // 无运行中任务但已有结果：展示最近一次实验（含 CLI/后台跑的 results/compare_latest.json）
  if (!cmpActive && st.result && !job) {
    renderCompareResult(st.result);
    $("expStatusHint").textContent =
      (st.from_file ? "最近一次后台/CLI 实验" : "对比实验完成 ✓") +
      " 结果如下（未写正式策略，仅作范围选择参考）";
    return;
  }
  if (!job) return;
  if (!cmpActive && st.result && job.state === "done") {
    renderCompareResult(st.result);
    $("expStatusHint").textContent = "对比实验完成 ✓ 结果如下（未写正式策略，仅作范围选择参考）";
  } else if (!cmpActive && job.state === "failed") {
    $("expStatusHint").textContent = `对比实验失败：${job.error || "见日志"}`;
  }
}

function renderCompareResult(result) {
  const box = $("expResult");
  if (!box) return;
  box.hidden = false;
  const tailV = result.variants && result.variants.tail;
  const sprV = result.variants && result.variants.spread;
  const tailBest = tailV && tailV.best;
  const sprBest = sprV && sprV.best;
  const cell = (b, key) => {
    if (!b) return "—";
    const v = b[key];
    if (key === "holdout") {
      const h = b.holdout || {};
      return `${h.passed ? "✓" : "✗"} ${h.val_score != null ? Number(h.val_score).toFixed(4) : "—"}`;
    }
    return v != null ? Number(v).toFixed(4) : "—";
  };
  $("expSummaryTable").innerHTML =
    `<thead><tr><th>变体</th><th>best 分</th><th>holdout 分(✓/✗)</th><th>样本外通过</th></tr></thead>` +
    `<tbody>` +
    `<tr><td>tail（最近 N 根）</td><td>${cell(tailBest, "best_score")}</td>` +
    `<td>${cell(tailBest, "holdout")}</td><td>${tailV ? tailV.holdout_passed + "/" + (tailV.runs || []).length : "—"}</td></tr>` +
    `<tr><td>spread（全历史分块）</td><td>${cell(sprBest, "best_score")}</td>` +
    `<td>${cell(sprBest, "holdout")}</td><td>${sprV ? sprV.holdout_passed + "/" + (sprV.runs || []).length : "—"}</td></tr>` +
    `</tbody>`;
  const concl = $("expConclusion");
  const cc = result.conclusions_by_criterion || {};
  const critLabel = {
    in_sample: "in-sample best",
    holdout_median: "跨 seed 中位 holdout",
    holdout_best: "holdout best",
  };
  const chosenCrit = (result.params && result.params.rep_criterion) || "holdout_median";
  const critChips = Object.keys(cc).length
    ? `<li class="exp-crit-chips"><b>分口径对照：</b>` +
      Object.entries(cc)
        .map(
          ([k, ls]) =>
            `<span class="exp-crit-chip${k === chosenCrit ? " chosen" : ""}" title="${escHtml(critLabel[k] || k)}">${escHtml(critLabel[k] || k)}：${escHtml((ls && ls[0]) || "—")}</span>`
        )
        .join(" ") +
      `</li>`
    : "";
  concl.innerHTML =
    `<li class="exp-crit-note">代表口径：${escHtml(critLabel[chosenCrit] || chosenCrit)}</li>` +
    (result.conclusion || []).map((l) => `<li>${escHtml(l)}</li>`).join("") +
    critChips;
  // 双 best-score 曲线（各自 best run）
  const ctx = $("expChart");
  if (ctx && typeof Chart !== "undefined") {
    if (compareChart) compareChart.destroy();
    const ds = [];
    [
      ["tail（最近N）", tailBest, "#38bdf8"],
      ["spread（全历史分块）", sprBest, "#fbbf24"],
    ].forEach(([label, b, color]) => {
      if (!b || !b.history) return;
      const h = b.history;
      ds.push({
        label,
        data: (h.best_score || []).map((v, i) => ({ x: i + 1, y: v })),
        borderColor: color,
        backgroundColor: color + "22",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      });
    });
    compareChart = new Chart(ctx.getContext("2d"), {
      type: "line",
      data: { datasets: ds },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", title: { display: true, text: "训练步", color: "#7a8a9e" },
               grid: { color: "rgba(255,255,255,0.04)" } },
          y: { grid: { color: "rgba(255,255,255,0.06)" } },
        },
        plugins: {
          legend: { labels: { color: "#cbd5e1" } },
          tooltip: { mode: "index", intersect: false },
        },
      },
    });
  }
}

function updateBtStartBtn() {
  const startBtn = $("btStartBtn");
  if (!startBtn) return;
  startBtn.disabled = btActive || !selectedStrategyFile;
  ["btCommissionInput", "btSlippageInput"].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = btActive;
  });
}

// 回测数据下拉：跟随策略（默认）或指定其他 Parquet（跨品种/跨周期/跨切片）
async function btLoadDataFiles() {
  const sel = $("btDataSelect");
  if (!sel) return;
  let files = [];
  try {
    const d = await fetchJSON("/api/data/files", { silent: true, retries: 0 });
    files = d.files || [];
  } catch (_) {}
  const cur = sel.value;
  const group = (name, rows) =>
    rows.length
      ? `<optgroup label="${escHtml(name)}">` +
        rows
          .map(
            (f) =>
              `<option value="${escHtml(f.data_file)}" title="${escHtml(f.rel)} · ${f.bars ?? "?"} 根 · ${f.start ?? "?"} ~ ${f.end ?? "?"}">` +
              `${escHtml(f.symbol || f.rel)}${f.timeframe ? " · " + escHtml(f.timeframe) : ""} — ${escHtml(f.rel.split("/").pop())}（${f.bars ?? "?"} 根）</option>`
          )
          .join("") +
        "</optgroup>"
      : "";
  const slices = files.filter((f) => f.rel.startsWith("data/slices"));
  const training = files.filter((f) => f.rel.startsWith("data/training"));
  sel.innerHTML =
    '<option value="">— 跟随策略（默认） —</option>' +
    group("data/slices（切片/样本外）", slices) +
    group("data/training（原始下载）", training);
  // 保持当前值；否则默认跟随策略
  if (cur && files.some((f) => f.data_file === cur)) {
    sel.value = cur;
  } else {
    sel.value = "";
  }
}

// 策略切换后：若该策略记录了训练数据且在下拉里，自动选中；否则回到「跟随策略」
function syncBtDataToStrategy(info) {
  const sel = $("btDataSelect");
  if (!sel || !info) return;
  const rec = info.data_file || (info.data_source && info.data_source.data_file) || "";
  if (!rec) return;
  if ([...sel.options].some((o) => o.value === rec)) sel.value = rec;
}

function renderStrategyFileCard(info) {
  const card = $("btStrategyCard");
  if (!card) return;

  if (!info || !info.strategy_file) {
    card.className = "data-file-card";
    card.innerHTML = '<div class="data-file-empty">尚未选择策略文件</div>';
    selectedStrategyFile = null;
    selectedStrategySymbol = null;
    __btStrategyDataFile = "";
    syncBtStrategySelect();
    updateBtStartBtn();
    btSyncDualBtn();
    matrixSymBadgeUpdate();
    return;
  }

  if (info.valid === false) {
    card.className = "data-file-card invalid";
    card.innerHTML = `
      <div class="data-file-error">${info.message || "文件无效"}</div>
      <div class="data-file-path">${info.strategy_file}</div>
    `;
    selectedStrategyFile = null;
    selectedStrategySymbol = null;
    __btStrategyDataFile = "";
    syncBtStrategySelect();
    updateBtStartBtn();
    btSyncDualBtn();
    return;
  }

  selectedStrategyFile = info.strategy_file;
  selectedStrategySymbol = info.symbol || null;
  syncBtStrategySelect();
  syncBtDataToStrategy(info);
  matrixSymBadgeUpdate();
  btLoadMatrixBest();
  btLoadCompare();
  card.className = "data-file-card valid";
  const timeframeItem = info.timeframe
    ? `<div class="item"><span class="label">周期</span><span class="value">${info.timeframe}</span></div>`
    : "";
  const stratDir = info.strategy_file
    ? String(info.strategy_file).split("/").slice(0, -1).join("/") || "—"
    : "—";
  const ds = info.data_source || {};
  const srcText = ds.file
    ? `${ds.file}${ds.start && ds.end ? ` · ${ds.start} → ${ds.end}` : ""}`
    : "—";
  const dataPath = info.data_file || "";
  __btStrategyDataFile = dataPath || ""; // 供「同参双口径」在未另选数据文件时用策略记录的数据
  const dataOk = info.data_file_exists;
  const dataHint = dataPath
    ? (dataOk ? dataPath : `（文件不存在）${dataPath}`)
    : "未记录数据路径 — 回测前请先在训练页选择同品种 Parquet";
  const rangeInfo = trainRangeInfo(info);
  const rangeItem = rangeInfo
    ? `<div class="item"><span class="label">训练数据范围</span>` +
      `<span class="value tr-range-item" title="${escHtml(trainRangeTooltip(rangeInfo, info))}">${escHtml(trainRangeLabel(rangeInfo))}</span></div>`
    : "";
  card.innerHTML = `
    <div class="data-file-row">
      <div class="item"><span class="label">品种</span><span class="value sym">${info.symbol || "—"}</span></div>
      ${timeframeItem}
      <div class="item"><span class="label">最优分数</span><span class="value score-best">${formatScore(info.best_score)}</span></div>
      <div class="item"><span class="label">词表版本</span><span class="value">${info.vocab_version || "—"}</span></div>
      <div class="item"><span class="label">公式长度</span><span class="value">${info.formula_decoded ? info.formula_decoded.split("→").length : "—"}</span></div>
      <div class="item"><span class="label">目录</span><span class="value" title="${stratDir}">${stratDir}</span></div>
      <div class="item"><span class="label">训练数据来源</span><span class="value" title="${ds.data_file || ""}">${srcText}</span></div>
      ${rangeItem}
    </div>
    <div class="path" title="${info.strategy_file}">策略: ${info.filename || info.strategy_file}</div>
    <div class="path ${dataPath && dataOk ? "" : "data-file-missing"}" title="${dataPath || ""}">数据: ${dataHint}</div>
  `;
  updateBtStartBtn();
  btSyncDualBtn();
}

function formatElapsed(startedAtIso, endAtIso) {
  if (!startedAtIso) return "—";
  const started = new Date(startedAtIso).getTime();
  if (Number.isNaN(started)) return "—";
  const end = endAtIso ? new Date(endAtIso).getTime() : Date.now();
  if (Number.isNaN(end)) return "—";
  return formatDurationSeconds(Math.max(0, Math.floor((end - started) / 1000)));
}

function formatDurationSeconds(secs) {
  if (secs == null || secs < 0) return "—";
  const total = Math.floor(secs);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  if (h > 0) return `${h}小时${m}分`;
  if (m > 0) return `${m}分钟`;
  return `${total}秒`;
}

function updateTrainingTimeFields(progress, training) {
  const sessionEl = $("fileElapsedTime");
  const historyEl = $("fileHistoryElapsedTime");
  if (!sessionEl && !historyEl) return;
  updateFileEta(training?.eta, !!training?.active);

  if (historyEl) {
    const hist = progress?.history_total_seconds;
    historyEl.textContent = hist != null ? formatDurationSeconds(hist) : "—";
  }

  const job = training?.job;
  const active = !!training?.active;
  if (!sessionEl) return;

  if (!job || job.state === "idle") {
    sessionEl.textContent = "—";
    return;
  }

  const elapsed = formatElapsed(job.started_at, active ? null : job.finished_at);
  sessionEl.textContent = active || elapsed === "—" ? elapsed : `${elapsed}（已停）`;
}

function updateFileEta(eta, active) {
  const el = document.getElementById("fileEta");
  if (!el) return;
  if (!eta || !active) {
    el.textContent = "—";
    el.title = "";
    return;
  }
  const remain = formatDurationSeconds(eta.remaining_seconds);
  const finish = eta.estimated_finish_local || "—";
  const pace = eta.seconds_per_step != null ? `${eta.seconds_per_step.toFixed(1)}s/步` : "—";
  el.textContent = `剩 ${eta.remaining_steps ?? "?"} 步 ≈ ${remain} · 预计 ${finish}`;
  el.title =
    `估算口径：${eta.pace_source === "session_regression" ? "近期每步耗时回归" : eta.pace_source === "session_mean" ? "本次会话均值" : "同品种历史耗时先验"}` +
    ` · 可信度 ${eta.confidence === "high" ? "高" : eta.confidence === "medium" ? "中" : "低（样本尚少）"} · 当前 ${pace}`;
}

function updateFileProgress(progress) {
  const el = document.getElementById("fileProgressPct");
  if (el && progress) {
    el.textContent = `${progress.current_step} / ${progress.train_steps} (${progress.progress_pct}%)`;
  }
  const bestEl = document.getElementById("fileBestScore");
  if (bestEl) {
    bestEl.textContent = progress ? formatScore(progress.best_score) : "—";
  }
  const valEl = document.getElementById("fileValScore");
  if (valEl) {
    let val = progress?.val_score;
    if (val == null && progress?.history?.val_score?.length) {
      val = progress.history.val_score[progress.history.val_score.length - 1];
    }
    valEl.textContent = progress ? formatScore(val) : "—";
  }
  const hoEl = document.getElementById("fileHoldout");
  if (hoEl) {
    const ho = progress?.holdout;
    if (ho && ho.val_score != null) {
      const ret =
        ho.total_return_pct != null
          ? ` · ${ho.total_return_pct >= 0 ? "+" : ""}${ho.total_return_pct}%`
          : "";
      const ratio =
        ho.score_ratio != null ? ` · 保持率 ${formatScore(ho.score_ratio)}` : "";
      const gate = ho.passed != null
        ? (ho.passed ? " · 闸门 ✓" : " · 闸门 ✗")
        : "";
      hoEl.textContent = `分 ${formatScore(ho.val_score)}${ret}${ratio}${gate}`;
      const gateReasons = Array.isArray(ho.gate) && ho.gate.length
        ? "\n" + ho.gate.map((r) => "  · " + r).join("\n")
        : "";
      hoEl.title =
        `样本外 ${ho.bars} 根（未参与训练/选优）：收益 ${ho.total_return_pct ?? "—"}%，` +
        `Sharpe ${ho.sharpe ?? "—"}，保持率 = 样本外分 ÷ 最优验证分（≈1 泛化好，<0.5 疑似过拟合）` +
        (gateReasons ? `\n闸门判定：${ho.passed ? "通过" : "未过"}${gateReasons}` : "");
    } else {
      hoEl.textContent = "—";
    }
  }
}

const CHART_SERIES = [
  { key: "best_score", label: "最优分数", borderColor: "#34f5c8", fillRGB: "52, 245, 200", yAxisID: "y" },
  { key: "val_score", label: "验证分数", borderColor: "#38bdf8", fillRGB: "56, 189, 248", yAxisID: "y" },
];

// 让曲线在填充区形成竖向渐变
function makeGradient(ctx, area, rgb) {
  if (!area) return `rgba(${rgb}, 0.08)`;
  const g = ctx.createLinearGradient(0, area.top, 0, area.bottom);
  g.addColorStop(0, `rgba(${rgb}, 0.28)`);
  g.addColorStop(0.6, `rgba(${rgb}, 0.06)`);
  g.addColorStop(1, `rgba(${rgb}, 0)`);
  return g;
}

// 发光效果：在每条数据线绘制前设置对应颜色的柔和阴影
const glowPlugin = {
  id: "neonGlow",
  beforeDatasetDraw(chart, args) {
    const color = args?.meta?.dataset?.options?.borderColor;
    const ctx = chart.ctx;
    ctx.save();
    if (typeof color === "string") {
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
    }
  },
  afterDatasetDraw(chart) {
    chart.ctx.restore();
  },
};
if (window.Chart) Chart.register(glowPlugin);

const CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 450, easing: "easeOutQuart" },
  transitions: {
    active: { animation: { duration: 450, easing: "easeOutQuart" } },
  },
  plugins: {
    legend: {
      labels: {
        color: "#a9bccf",
        usePointStyle: true,
        pointStyle: "circle",
        boxWidth: 8,
        boxHeight: 8,
        padding: 16,
        font: { family: "'DM Sans'", size: 12, weight: "600" },
      },
    },
    tooltip: {
      backgroundColor: "rgba(8, 12, 20, 0.92)",
      borderColor: "rgba(94, 234, 212, 0.35)",
      borderWidth: 1,
      titleColor: "#e8edf4",
      bodyColor: "#a9bccf",
      titleFont: { family: "'JetBrains Mono'", size: 11 },
      bodyFont: { family: "'JetBrains Mono'", size: 11 },
      padding: 10,
      cornerRadius: 8,
      displayColors: true,
      usePointStyle: true,
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      title: { display: true, text: "分数（最优 / 验证）", color: "#7dd3fc", font: { family: "'DM Sans'", size: 10, weight: "600" } },
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
  },
};

function buildChartDatasets(history) {
  return CHART_SERIES.filter((s) => history?.[s.key]?.length).map((s) => ({
    label: s.label,
    data: history[s.key],
    borderColor: s.borderColor,
    borderWidth: 2,
    tension: 0.35,
    pointRadius: 0,
    pointHoverRadius: 4,
    pointHoverBackgroundColor: s.borderColor,
    pointHoverBorderColor: "#05070d",
    fill: true,
    backgroundColor: (context) => {
      const { ctx, chartArea } = context.chart;
      return makeGradient(ctx, chartArea, s.fillRGB);
    },
    yAxisID: s.yAxisID,
  }));
}

function destroyChart() {
  if (chart) {
    chart.destroy();
    chart = null;
  }
  chartSymbol = null;
}

function createChart(ctx, steps, history) {
  return new Chart(ctx, {
    type: "line",
    data: { labels: steps, datasets: buildChartDatasets(history) },
    options: CHART_OPTIONS,
  });
}

function updateChartInPlace(steps, history) {
  const prevLen = chart.data.labels.length;
  chart.data.labels = steps;

  const next = buildChartDatasets(history);
  for (const ds of next) {
    const existing = chart.data.datasets.find((d) => d.label === ds.label);
    if (existing) {
      existing.data = ds.data;
    } else {
      chart.data.datasets.push(ds);
    }
  }

  const nextLabels = new Set(next.map((d) => d.label));
  chart.data.datasets = chart.data.datasets.filter((d) => nextLabels.has(d.label));

  const grew = steps.length > prevLen;
  // 轮询场景固定无动画更新：1600+ 点曲线每次带动画重绘是训练页卡顿主因之一
  chart.update("none");
}

function renderChart(history, label, progress) {
  const ctx = $("mainChart").getContext("2d");
  const steps = history?.step || [];
  if (!steps.length) {
    destroyChart();
    if (progress?.current_step > 0) {
      $("chartHint").textContent = `训练中 第 ${progress.current_step}/${progress.train_steps} 步，曲线每步更新`;
    } else {
      $("chartHint").textContent = "暂无历史数据（首步约需 15–30 秒）";
    }
    return;
  }

  const sameSymbol = chart && chartSymbol === label;
  if (sameSymbol) {
    updateChartInPlace(steps, history);
  } else {
    destroyChart();
    chart = createChart(ctx, steps, history);
    chartSymbol = label;
  }

  $("chartTitle").textContent = `${label} 训练曲线`;
  $("chartHint").textContent = `${steps.length} 个记录点`;
}

async function loadSymbolChart(symbol, progress) {
  if (!symbol) return;
  try {
    const data = await fetchJSON(`/api/symbols/${encodeURIComponent(symbol)}`);
    renderChart(data.history, symbol, progress || data);
    $("formulaText").textContent = data.formula_decoded || "—";
  } catch (e) {
    $("formulaText").textContent = "—";
  }
}

// 轮询链增量守卫：训练曲线每步只长一点，4s 轮询 × 全量曲线 JSON 是主要开销。
// 规则：同一 symbol 每 20 步才重拉一次；曲线面板不在视口内时不下载（滚动回
// 来后下一次轮询补拉）。
let lastChartSig = "";
let chartSkippedOffscreen = false;
function chartSigFor(symbol, progress) {
  const step = progress?.current_step ?? progress?.step ?? 0;
  return `${symbol}|${Math.floor(step / 20)}`;
}
function chartPanelInViewport() {
  const el = document.querySelector(".chart-panel");
  if (!el) return true;
  const r = el.getBoundingClientRect();
  return r.bottom > 0 && r.top < window.innerHeight;
}
async function loadSymbolChartGuarded(symbol, progress) {
  if (!symbol) return;
  const sig = chartSigFor(symbol, progress);
  if (sig === lastChartSig && !chartSkippedOffscreen) return;
  if (sig !== lastChartSig) lastChartSig = sig;
  if (!chartPanelInViewport()) {
    chartSkippedOffscreen = true;
    return;
  }
  chartSkippedOffscreen = false;
  await loadSymbolChart(symbol, progress);
}

function debugPanelInViewport() {
  const el = $("debugPanelSection");
  if (!el) return false;
  const r = el.getBoundingClientRect();
  return r.bottom > 0 && r.top < window.innerHeight;
}

function dataSourceLabel(r) {
  const ds = r.data_source || {};
  const file = ds.file || (ds.data_file || "").split("/").pop() || "";
  const span =
    ds.start && ds.end ? `${ds.start.slice(0, 10)} → ${ds.end.slice(0, 10)}` : "";
  if (!file && !span) return "—";
  return span ? `${file} · ${span}` : file;
}

// ── 训练数据范围溯源（full/tail/spread）──────────────────────────────
function trainRangeInfo(r) {
  const tr = r && r.train_range && typeof r.train_range === "object" ? r.train_range : null;
  if (tr && (tr.mode || tr.subset_bars)) return tr;
  // 老文件兜底：子集目录名 tail_12345 / spread_12345_vol_c4 / …
  const path =
    (r && (r.data_file || (r.data_source && r.data_source.data_file))) || "";
  const parent = String(path).split("/").slice(0, -1).pop() || "";
  const m = parent.match(/^(tail|spread)_(\d+)(?:_(vol|trend))?(?:_c(\d+))?$/);
  if (m) {
    return {
      mode: m[1],
      n_bars: Number(m[2]),
      regime: m[3] || "vol",
      n_chunks_used: m[4] ? Number(m[4]) : null,
    };
  }
  return null;
}

function trainRangeLabel(tr) {
  if (!tr) return null;
  const n = tr.n_bars_requested || tr.n_bars || tr.subset_bars;
  const nn = n ? ` ${Number(n).toLocaleString()} 根` : "";
  if (tr.mode === "tail") return `最近${nn}`;
  if (tr.mode === "spread") {
    let s = `全历史分块${nn}`;
    if (tr.regime === "vol") s += "·波动分层";
    else if (tr.regime === "trend") s += "·趋势分层";
    if (tr.n_chunks_used) s += `·${tr.n_chunks_used}段`;
    return s;
  }
  return tr.mode === "full" ? "全部" : "全部";
}

function trainRangeTooltip(tr, r) {
  if (!tr) return "";
  const bits = [`模式: ${tr.mode}`];
  if (tr.source_bars) bits.push(`源 ${Number(tr.source_bars).toLocaleString()} 根`);
  if (tr.subset_bars) bits.push(`子集 ${Number(tr.subset_bars).toLocaleString()} 根`);
  if (tr.n_bars_requested) bits.push(`请求 ${Number(tr.n_bars_requested).toLocaleString()} 根`);
  if (tr.mode === "spread" && tr.regime) bits.push(`regime: ${tr.regime}`);
  if (tr.n_chunks_used) bits.push(`段数: ${tr.n_chunks_used}`);
  if (tr.start && tr.end) bits.push(`覆盖 ${String(tr.start).slice(0, 10)} → ${String(tr.end).slice(0, 10)}`);
  if (tr.regime_coverage) {
    const c = tr.regime_coverage;
    bits.push(`波动覆盖 低${c.low}/中${c.mid}/高${c.high}`);
  }
  const f = tr.data_file || (r && r.data_file) || (r && r.data_source && r.data_source.data_file);
  if (f) bits.push(`文件: ${f}`);
  return bits.join(" · ");
}

function trainRangeBadge(r) {
  const tr = trainRangeInfo(r);
  if (!tr) return "";
  return `<span class="tr-badge" title="${escHtml(trainRangeTooltip(tr, r))}">${escHtml(trainRangeLabel(tr))}</span>`;
}

let _strategiesSig = "";
function renderStrategies(rows) {
  const tbody = $("strategiesBody");
  if (!tbody) return;
  // 签名守卫：内容没变不重建表格（避免每轮轮询 innerHTML 解析 + 丢悬停态）
  const sig = (rows || [])
    .map(
      (r) =>
        [
          r.symbol, r.timeframe, r.best_score, r.strategy_file,
          r.formula_decoded, !!r.train_range,
          r.data_source ? (r.data_source.file || "") : "",
        ].join("~")
    )
    .join("|");
  if (sig === _strategiesSig) return;
  _strategiesSig = sig;
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="6">暂无已保存策略</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map(
      (r) => {
        // 带 train_range 溯源的冠军条目 → 回滚校验按钮（验证子集确定性 + 存档公式 holdout 复验）
        const vrf = r.train_range
          ? `<button type="button" class="btn btn-mini btn-secondary" data-vrf-file="${escHtml(r.strategy_file)}" title="点击跑 scripts/verify_champion_rollback.py：① 按记录的抽样参数重生成子集并与存档逐行对比；② 用存档公式重跑 holdout 复验（可另选带短程重训）。产出报告在页内展示">回滚校验</button>`
          : "";
        return `
    <tr>
      <td>${r.symbol}</td>
      <td>${r.timeframe || "—"}</td>
      <td>${formatScore(r.best_score)}</td>
      <td title="${r.data_source ? (r.data_source.data_file || "") : ""}">${dataSourceLabel(r)} ${trainRangeBadge(r)}</td>
      <td><code>${r.formula_decoded || "—"}</code></td>
      <td>${vrf}</td>
    </tr>`;
      }
    )
    .join("");
}

function updateTrainingUI(training, progress) {
  const job = training?.job;
  const active = training?.active;
  const pill = $("jobPill");
  const startBtn = $("startBtn");
  const retrainBtn = $("retrainBtn");
  const stopBtn = $("stopBtn");

  if (!job || job.state === "idle") {
    pill.innerHTML = '<i class="pill-dot"></i>空闲';
    pill.className = "pill";
    startBtn.disabled = !selectedDataFile || !trainRangeOk();
    if (retrainBtn) retrainBtn.disabled = !selectedDataFile || !trainRangeOk();
    stopBtn.disabled = true;
    $("logHint").textContent = "—";
    updateTrainingTimeFields(progress, training);
    renderTrainInspections(training?.inspections || []);
    return;
  }

  const stateLabel = {
    running: "训练中",
    completed: "已完成",
    failed: "失败",
    stopped: "已停止",
  };
  const label = job.symbol ? `${job.symbol} ${job.timeframe || ""}`.trim() : "训练";
  const stateText = stateLabel[job.state] || job.state;
  pill.innerHTML = `<i class="pill-dot"></i>${stateText} · ${label}`;
  pill.className = "pill " + (job.state === "running" ? "running" : job.state);

  startBtn.disabled = active;
  if (retrainBtn) retrainBtn.disabled = active;
  stopBtn.disabled = !active;
  $("logHint").textContent = job.log_path || "—";
  updateTrainingTimeFields(progress, training);

  const logView = $("logView");
  const atBottom = isViewAtBottom(logView);
  logView.textContent = (training.log_tail || []).join("\n") || "等待输出…";
  if (atBottom) logView.scrollTop = logView.scrollHeight;

  renderTrainInspections(training?.inspections || []);
}

let _overviewInflight = false; // 上一轮概览还没跑完时不重复发起（防慢响应下轮询堆积）
async function refreshOverview() {
  if (_overviewInflight) return;
  _overviewInflight = true;
  try {
    await _refreshOverviewInner();
  } finally {
    _overviewInflight = false;
  }
}

async function _refreshOverviewInner() {
  const trainPageVisible = currentPage === "train";
  let overview = { data_file: null, progress: null };
  let strategies = { strategies: [] };
  let training = { active: false, job: null, log_tail: [] };

  // 训练页专属接口（概览 / 训练状态）只在训练页可见时取
  if (trainPageVisible) {
    try {
      overview = await fetchJSON("/api/overview", { silent: true });
    } catch (_) {}
    try {
      training = await fetchJSON("/api/training/status", { silent: true });
    } catch (_) {}
  }

  // /api/strategies 跨页共享（回测页策略列表也用它），保持每拍取
  try {
    strategies = await fetchJSON("/api/strategies", { silent: true });
  } catch (_) {}

  // 非训练页：只渲染共享部分；曲线/日志/下载历史/AI 探测等训练页请求完全停掉
  renderBtStrategyList(strategies.strategies);
  if (!trainPageVisible) {
    trainActive = false; // 训练状态不再跟踪，pollTick 轮询自然降频
    return;
  }

  if (overview.data_file) renderDataFileCard(overview.data_file);
  updateFileProgress(overview.progress);
  updateFileEta(training?.eta, !!training?.active);
  updateExportBtn(overview.progress, strategies.strategies);
  updateTrainingBtns(overview.progress, training);
  updateTrainingUI(training, overview.progress);
  renderStrategies(strategies.strategies);
  refreshDownloadHistory();

  const sym = overview.progress?.symbol || selectedSymbol || training?.job?.symbol;
  const trainingActive = !!training?.active;
  if (lastTrainingActive && !trainingActive && sym) {
    await applyBestStrategyForBacktest(sym, null);
  }
  lastTrainingActive = trainingActive;
  trainActive = trainingActive;   // 供 pollTick 自适应节奏

  // 训练中途里程碑：达到一定步数自动预取同品种 N×N 最优组合曲线并排进对比面板
  if (trainingActive && sym && overview.progress) {
    await expMatrixPrefetchOnMilestone(sym, overview.progress);
  }

  // 重负载项（200KB 曲线下载 / 读服务端日志 / AI 状态探测）只在训练页可见时取
  if (sym && (training?.active || overview.progress)) {
    await loadSymbolChartGuarded(sym, overview.progress);
  }
  // 54KB 服务端日志只在调试面板滚入视口时才拉，不再每拍无条件取
  if (debugPanelInViewport()) await refreshDebugLogs();
  refreshAiProviderStatus();
}

async function loadConfig() {
  const health = await fetch(API + "/api/health").then((r) => r.json()).catch(() => ({}));
  if (!health.version) {
    await logClientError(
      "后端版本过旧或未启动新版服务。请关闭旧进程后重新运行: python run_web.py",
      { health }
    );
  }

  const cfg = await fetchJSON("/api/config");
  debugMode = !!cfg.debug_mode;
  $("debugModeCheck").checked = debugMode;
  const bgAnim = !!cfg.bg_animation;
  window.__bgAnimationEnabled = bgAnim;
  if ($("bgAnimCheck")) $("bgAnimCheck").checked = bgAnim;
  if (window.__bgAnimationSetEnabled) window.__bgAnimationSetEnabled(bgAnim);
  $("deviceMeta").textContent = `${cfg.train_steps} steps · batch ${cfg.batch_size} · ${cfg.device}`;
  if (cfg.error_log) {
    $("debugLogPaths").textContent = `本地: ${cfg.error_log}`;
  }
  if (cfg.data_file) renderDataFileCard(cfg.data_file);
  if (cfg.strategy_file) renderStrategyFileCard(cfg.strategy_file);
  applyBacktestCostDefaults(cfg);
  await initAiPanel(cfg);
}

function applyBacktestCostDefaults(cfg) {
  const cIn = $("btCommissionInput");
  const sIn = $("btSlippageInput");
  if (cIn && cfg.bt_commission_pct != null) cIn.value = Number(cfg.bt_commission_pct);
  if (sIn && cfg.bt_slippage_pct != null) sIn.value = Number(cfg.bt_slippage_pct);
  updateBtCostHint();
}

function readBacktestCosts() {
  const cRaw = Number($("btCommissionInput")?.value);
  const sRaw = Number($("btSlippageInput")?.value);
  const commission = Number.isFinite(cRaw) && cRaw >= 0 ? cRaw : 0.02;
  const slippage = Number.isFinite(sRaw) && sRaw >= 0 ? sRaw : 0.01;
  return { commission_pct: commission, slippage_pct: slippage };
}

function updateBtCostHint() {
  const hint = $("btCostSumHint");
  if (!hint) return;
  const { commission_pct, slippage_pct } = readBacktestCosts();
  const fee = Number((commission_pct + slippage_pct).toFixed(4));
  hint.textContent = `单边成本 ${fee}%`;
}

async function refreshAiProviderStatus() {
  try {
    const status = await fetchJSON("/api/ai/providers", { silent: true });
    window.__aiProviderStatus = status;
  } catch (_) {
    /* keep previous snapshot */
  }
  updateAiChannelHint();
}

async function initAiPanel(cfg) {
  const keyInput = $("aiApiKeyInput");
  const baseUrlInput = $("aiBaseUrlInput");
  const modelInput = $("aiModelInput");
  if (!keyInput) return;

  if (cfg?.ai_api_key) keyInput.value = cfg.ai_api_key;
  else if (cfg?.ai_provider === "openclaw" || cfg?.ai_provider === "openclaw_wb") {
    keyInput.value = cfg.ai_provider;
  }
  if (baseUrlInput) {
    baseUrlInput.value = cfg?.ai_base_url || "https://api.deepseek.com";
  }
  if (modelInput) {
    modelInput.value = cfg?.ai_model || "deepseek-v4-flash";
  }

  await refreshAiProviderStatus();
  if (!keyInput.dataset.aiStatusBound) {
    keyInput.dataset.aiStatusBound = "1";
    const refreshHint = () => {
      updateAiChannelHint();
      refreshAiProviderStatus();
    };
    keyInput.addEventListener("input", refreshHint);
    if (baseUrlInput) {
      baseUrlInput.addEventListener("input", updateAiChannelHint);
      baseUrlInput.addEventListener("change", updateAiChannelHint);
    }
    if (modelInput) {
      modelInput.addEventListener("input", updateAiChannelHint);
      modelInput.addEventListener("change", updateAiChannelHint);
    }
  }
}

function resolveAiFromKey(raw) {
  const v = (raw || "").trim().toLowerCase();
  // openclaw_wb 必须先于 openclaw，避免前缀误匹配
  if (v === "openclaw_wb" || v.startsWith("openclaw_wb/")) {
    return { provider: "openclaw_wb", apiKey: raw.trim(), isAlias: true };
  }
  if (v === "openclaw" || v.startsWith("openclaw/")) {
    return { provider: "openclaw", apiKey: raw.trim(), isAlias: true };
  }
  return { provider: "deepseek", apiKey: (raw || "").trim(), isAlias: false };
}

function readAiEndpointFields() {
  const baseUrl = ($("aiBaseUrlInput")?.value || "").trim() || "https://api.deepseek.com";
  const model = ($("aiModelInput")?.value || "").trim() || "deepseek-v4-flash";
  return { baseUrl, model };
}

function updateAiChannelHint() {
  const hint = $("aiChannelHint");
  const headHint = $("aiProviderHint");
  const keyInput = $("aiApiKeyInput");
  if (!hint || !keyInput) return;

  const resolved = resolveAiFromKey(keyInput.value);
  const status = window.__aiProviderStatus;
  const row = (status?.providers || []).find((p) => p.id === resolved.provider);
  const { baseUrl, model } = readAiEndpointFields();
  const isSenseNova = /sensenova\.cn/i.test(baseUrl);

  if (resolved.provider === "deepseek") {
    const label = isSenseNova ? "SenseNova" : (/deepseek\.com/i.test(baseUrl) ? "DeepSeek" : "OpenAI 兼容");
    if (headHint) headHint.textContent = `${label} · ${model}`;
    hint.textContent = `当前：${label}（${model} · ${baseUrl}）。可改 Base URL 对接 SenseNova 等兼容网关；Key 填 openclaw / openclaw_wb 可走本地通道。`;
  } else if (resolved.provider === "openclaw") {
    if (headHint) {
      headHint.textContent = row?.available ? "openclaw (QClaw) · 已匹配" : "openclaw (QClaw) · 未就绪";
    }
    hint.textContent = row?.hint || "已匹配 openclaw：将自动使用本地 QClaw token（忽略上方 Base URL）。";
  } else {
    if (headHint) headHint.textContent = row?.available ? "openclaw_wb · 已匹配" : "openclaw_wb · 未就绪";
    hint.textContent = row?.hint || "已匹配 openclaw_wb：将自动使用 WorkBuddy token（忽略上方 Base URL）。";
  }
}

function openUnlimitedModal() {
  const modal = $("aiUnlimitedModal");
  if (modal) modal.hidden = false;
}

function closeUnlimitedModal() {
  const modal = $("aiUnlimitedModal");
  if (modal) modal.hidden = true;
}

async function runAiAnalyze() {
  const btn = $("aiAnalyzeBtn");
  const view = $("aiAnswerView");
  const rawKey = $("aiApiKeyInput")?.value || "";
  const resolved = resolveAiFromKey(rawKey);
  const { baseUrl, model } = readAiEndpointFields();
  if (!view) return;

  if (resolved.provider === "deepseek" && !resolved.apiKey) {
    view.className = "ai-answer error";
    view.textContent = "请填写 API Key";
    return;
  }
  if (resolved.provider === "deepseek") {
    if (!/^https?:\/\//i.test(baseUrl)) {
      view.className = "ai-answer error";
      view.textContent = "Base URL 必须以 http:// 或 https:// 开头（例如 https://token.sensenova.cn/v1）";
      return;
    }
  }

  await refreshAiProviderStatus();

  if (btn) btn.disabled = true;
  view.className = "ai-answer loading";
  view.textContent = `正在通过 ${resolved.provider} 连接并流式分析…`;

  let header = "";
  let answer = "";

  try {
    const res = await fetch(API + "/api/ai/analyze-training", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: resolved.provider,
        api_key: resolved.apiKey,
        base_url: baseUrl,
        model,
        symbol: selectedSymbol || null,
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, "/api/ai/analyze-training"));
    }
    if (!res.body) throw new Error("浏览器不支持流式响应");

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    view.className = "ai-answer streaming";
    view.textContent = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";
      for (const block of chunks) {
        const line = block
          .split("\n")
          .map((l) => l.trim())
          .find((l) => l.startsWith("data:"));
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line.slice(5).trim());
        } catch (_) {
          continue;
        }
        if (event.type === "meta") {
          header =
            `[${event.label || event.provider || resolved.provider} · ${event.model || ""} · ${event.symbol || ""}${event.timeframe ? " " + event.timeframe : ""}]` +
            (event.prior_count
              ? ` · 已带入前 ${event.prior_count} 次同品种同周期分析`
              : " · 首次分析") +
            `\n\n`;
          view.textContent = header;
          view.scrollTop = view.scrollHeight;
        } else if (event.type === "delta") {
          answer += event.text || "";
          view.textContent = header + answer;
          view.scrollTop = view.scrollHeight;
        } else if (event.type === "error") {
          throw new Error(event.message || "分析失败");
        } else if (event.type === "done") {
          answer = event.answer || answer;
          view.className = "ai-answer";
          view.textContent = header + (answer || "（无内容）");
        }
      }
    }
    if (!answer && view.className.includes("streaming")) {
      throw new Error("流式分析中断，未收到完整回复");
    }
    view.className = "ai-answer";
  } catch (e) {
    view.className = "ai-answer error";
    view.textContent = `分析失败: ${e.message}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ── 训练巡检（内置 agent · 自动，无需 API Key）─────────────────────────────
let __aiViewCleared = false;
const AI_LEVEL_LABEL = { ok: "正常", warn: "关注", danger: "⚠ 建议停止", info: "提示" };

function _aiFmtNum(v, digits) {
  if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return (digits === undefined ? n.toPrecision(4) : n.toFixed(digits));
}

function renderTrainInspections(entries) {
  const view = $("aiAnswerView");
  if (!view) return;
  const list = Array.isArray(entries) ? entries.slice().reverse() : [];
  if (!list.length) {
    if (!__aiViewCleared) {
      view.className = "ai-answer";
      view.innerHTML = "开始训练后，巡检会自动在这里给出诊断结论。";
    }
    return;
  }
  __aiViewCleared = false;
  const html = list
    .map((e) => {
      const lv = AI_LEVEL_LABEL[e.level] || e.level || "";
      const lvCls = "ai-lv-" + (e.level || "info");
      const m = e.metrics || {};
      const meta = [];
      if (m.step !== undefined) meta.push(`步 ${m.step}/${m.total ?? "?"}${m.pct !== undefined ? " (" + m.pct + "%)" : ""}`);
      if (m.best !== undefined) meta.push(`最优 ${_aiFmtNum(m.best)}`);
      if (m.stall !== undefined) meta.push(`停滞 ${m.stall} 步`);
      if (m.restarts !== undefined) meta.push(`重启 ${m.restarts}`);
      if (m.ic_now !== undefined) meta.push(`近批 IC ${_aiFmtNum(m.ic_now)}`);
      if (m.val_now !== undefined) meta.push(`近批验证 ${_aiFmtNum(m.val_now)}`);
      if (m.eff_vocab_avg !== undefined) meta.push(`有效词汇 ${_aiFmtNum(m.eff_vocab_avg)}`);
      if (m.champion && m.champion !== "—") meta.push(`线上冠军 ${m.champion}`);
      const checks = (e.checks || [])
        .map((c) => `<div class="ai-check ai-check-${c.level || "info"}">${escHtml(c.text)}</div>`)
        .join("");
      const fmlBlock = (e.formula && typeof e.formula === "object")
        ? `<div class="ai-inspect-fml">` +
          `<div class="ai-fml-title">🧮 最新公式解读</div>` +
          `<div class="ai-fml-expr">${escHtml(e.formula.expression || e.formula.decoded || "")}</div>` +
          (e.formula.summary ? `<div class="ai-fml-summary">${escHtml(e.formula.summary)}</div>` : "") +
          `</div>`
        : "";
      const when = e.ts ? (e.ts || "").replace("T", " ").slice(0, 19) : "";
      return (
        `<div class="ai-inspect-entry">` +
        `<div class="ai-inspect-head"><span class="ai-lv-badge ${lvCls}">${escHtml(lv)}</span>` +
        `<b>${escHtml(e.title || "巡检")}</b>` +
        (e.symbol ? `<span class="hint">${escHtml(e.symbol)}</span>` : "") +
        `</div>` +
        (meta.length ? `<div class="ai-inspect-meta">${meta.map(escHtml).join(" · ")}</div>` : "") +
        (checks ? `<div class="ai-inspect-checks">${checks}</div>` : "") +
        fmlBlock +
        (e.recommendation ? `<div class="ai-inspect-rec">💡 ${escHtml(e.recommendation)}</div>` : "") +
        (when ? `<div class="ai-inspect-ts">${escHtml(when)}</div>` : "") +
        `</div>`
      );
    })
    .join("");
  view.className = "ai-answer has-entries";
  view.innerHTML = html;
}

async function runTrainInspectNow() {
  const view = $("aiAnswerView");
  const btn = $("aiInspectBtn");
  if (view) {
    view.className = "ai-answer loading";
    view.textContent = "正在读取训练日志并巡检…";
  }
  if (btn) btn.disabled = true;
  try {
    const res = await fetchJSON("/api/training/inspect-now", { method: "POST", silent: true });
    const entry = res && res.entry;
    if (entry) renderTrainInspections([entry]);
    else if (view) { view.className = "ai-answer"; view.textContent = "暂无检查结果。"; }
    await refreshOverview();
  } catch (e) {
    if (view) {
      view.className = "ai-answer error";
      view.textContent = "巡检失败: " + (e && e.message ? e.message : e);
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function clearTrainInspections() {
  const view = $("aiAnswerView");
  if (view) {
    __aiViewCleared = true;
    view.className = "ai-answer";
    view.textContent = "已清空；训练中会自动重新生成巡检。";
  }
}

let _btStratListSig = null;
function renderBtStrategyList(rows) {
  const sel = $("btStrategyListSelect");
  if (!sel) return;
  const current = selectedStrategyFile || "";
  const opts = ['<option value="">— 选择策略切换 —</option>'];
  (rows || []).forEach((r) => {
    if (!r.strategy_file) return;
    const tf = r.timeframe ? ` ${r.timeframe}` : "";
    const score = r.best_score != null ? ` · ${Number(r.best_score).toFixed(3)}` : "";
    const ds = r.data_source && r.data_source.file ? ` · ${r.data_source.file}` : "";
    const rng = trainRangeInfo(r);
    const rangeShort = rng ? ` · ${escHtml(trainRangeLabel(rng))}` : "";
    opts.push(
      `<option value="${escHtml(r.strategy_file)}" ${r.strategy_file === current ? "selected" : ""} ` +
        `title="${escHtml((rng ? trainRangeTooltip(rng, r) + " | " : "") + (r.file || ""))}">` +
        `${escHtml(r.symbol || r.file)}${escHtml(tf)}${score}${ds}${rangeShort}</option>`
    );
  });
  // 当前策略不在 best_*.json 列表（如刚导入的外部 json）时，保留为一项，避免选中态丢失
  const hasCur = (rows || []).some((r) => r.strategy_file === current);
  if (current && !hasCur) {
    opts.push(
      `<option value="${escHtml(current)}" selected>当前: ${escHtml(String(current).split("/").pop() || current)}</option>`
    );
  }
  const html = opts.join("");
  if (_btStratListSig === html) {
    if (sel.value !== current) sel.value = current || "";
    return;
  }
  _btStratListSig = html;
  sel.innerHTML = html;
  if (current) sel.value = current;
}

// 卡片渲染后让下拉选中项跟随（下拉选项可能不在 best_*.json 列表，此时补一项保持可见）
function syncBtStrategySelect() {
  const sel = $("btStrategyListSelect");
  const cur = selectedStrategyFile || "";
  if (!sel) return;
  if (cur && ![...sel.options].some((o) => o.value === cur)) {
    const opt = document.createElement("option");
    opt.value = cur;
    opt.textContent = "当前: " + (String(cur).split("/").pop() || cur);
    opt.selected = true;
    sel.appendChild(opt);
  }
  sel.value = cur;
}

async function applyStrategyFilePath(raw) {
  if (!raw) return false;
  try {
    const res = await fetchJSON(
      "/api/strategy-file/browse?path=" + encodeURIComponent(raw),
      { method: "POST", retries: 1 }
    );
    renderStrategyFileCard(res);
    return true;
  } catch (e) {
    $("debugView")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return false;
  }
}

async function browseStrategyFile() {
  let res;
  try {
    res = await fetchJSON("/api/strategy-file/browse", { method: "POST", retries: 0 });
  } catch (e) {
    $("debugView")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  if (!res.dialog || !res.session) {
    if (!res.cancelled) renderStrategyFileCard(res);
    return;
  }
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 800));
      const p = await fetchJSON(
        "/api/strategy-file/browse-poll?session=" + encodeURIComponent(res.session),
        { retries: 0 }
      );
      if (!p.done) continue;
      if (!p.cancelled) renderStrategyFileCard(p);
      return;
    }
  } catch (e) {
    $("debugView")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function applyBestStrategyForBacktest(symbol, strategyFile) {
  if (strategyFile) {
    renderStrategyFileCard(strategyFile);
    return true;
  }
  if (!symbol) return false;
  // Best-effort：无 best_*.json 时后端返回 404，属正常情况，不弹窗、不重试、不回落到
  // loadBacktestStrategyContext（否则会与下方形成无限循环并反复 refreshDebugLogs）。
  try {
    const res = await fetch(
      API + `/api/strategy-file/sync-best?symbol=${encodeURIComponent(symbol)}`,
      { method: "POST" }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return false;
    renderStrategyFileCard(data);
    return true;
  } catch (_) {
    return false;
  }
}

async function loadConfigStrategyFallback() {
  try {
    const cfg = await fetchJSON("/api/config", { silent: true, retries: 1 });
    if (cfg.strategy_file) renderStrategyFileCard(cfg.strategy_file);
  } catch (_) {
    /* ignore */
  }
}

async function loadBacktestStrategyContext() {
  const sym = selectedStrategySymbol || selectedSymbol;
  if (sym) {
    const applied = await applyBestStrategyForBacktest(sym, null);
    if (applied) return;
  }
  await loadConfigStrategyFallback();
}

function setDlHint(text, isError = false) {
  const el = $("dlHint");
  if (!el) return;
  el.textContent = text || "";
  el.classList.toggle("invalid", !!isError && !!text);
  el.classList.toggle("valid", !isError && !!text);
}

const DL_PRESET_FALLBACK = [
  "XAUUSD", "EURUSD", "GBPUSD", "USDJPY",
  "NASDAQ:AAPL", "SSE:600519", "BINANCE:BTCUSDT", "BTCUSDT", "XAGUSD",
];

async function initDlPresets() {
  const dl = $("dlSymbolPresets");
  if (!dl) return;
  const seen = new Set();
  const add = (s) => { const v = (s || "").trim(); if (v && !seen.has(v)) { seen.add(v); dl.appendChild(Object.assign(document.createElement("option"), { value: v })); } };
  let sourcesMeta = {};
  try {
    const res = await fetchJSON("/api/realtime/sources", { retries: 1, silent: true });
    for (const src of res.sources || []) {
      for (const p of src.presets || []) add(p);
      sourcesMeta[src.id] = src;
    }
  } catch (_) { /* 网络错误时用兜底列表 */ }
  for (const p of DL_PRESET_FALLBACK) add(p);

  // 数据源下拉：标注可用性；周期下拉随源重建（各源支持的周期不同）；数量上限/提示随 源×周期 联动
  const TF_LABELS = {
    "1m": "1 分钟", "3m": "3 分钟", "5m": "5 分钟", "15m": "15 分钟", "30m": "30 分钟",
    "1h": "1 小时", "2h": "2 小时", "4h": "4 小时", "6h": "6 小时", "8h": "8 小时", "12h": "12 小时",
    "1d": "1 天", "3d": "3 天", "1w": "1 周", "1M": "1 月",
  };
  const DEEP_TFS = ["1m", "3m", "5m", "15m"]; // Binance 30m 以下支持全量历史
  function refreshDlTfOptions(meta) {
    const tfSel = $("dlTimeframeSelect");
    if (!tfSel) return;
    const prev = tfSel.value;
    const tfs = (meta && meta.timeframes && meta.timeframes.length) ? meta.timeframes : Object.keys(TF_LABELS);
    tfSel.innerHTML = tfs.map((t) => `<option value="${t}">${TF_LABELS[t] || t}</option>`).join("");
    tfSel.value = tfs.includes(prev) ? prev : tfs.includes("1h") ? "1h" : tfs[0];
  }
  function refreshDlBarsLimit() {
    const barsEl = $("dlBarsInput");
    const depthEl = $("dlDepthHint");
    const src = $("dlSourceSelect")?.value || "tradingview";
    const tf = $("dlTimeframeSelect")?.value || "1h";
    const isBinanceDeep = src === "binance" && DEEP_TFS.includes(tf);
    if (barsEl) {
      barsEl.max = isBinanceDeep ? "1000000" : "100000";
      if (parseInt(barsEl.value || "0", 10) > parseInt(barsEl.max, 10)) barsEl.value = barsEl.max;
    }
    if (!depthEl) return;
    if (src === "tradingview") {
      depthEl.textContent = "深度参考（匿名会话实测）：1m≈1 万根(7 天) · 15m≈6 千根(2 个月) · 1h≈1 万根(1.7 年) · 4h/1d≈数万根（多年）";
    } else if (isBinanceDeep) {
      depthEl.textContent = "全量历史可选：1m/3m/5m/15m 上限 100 万根（Binance 自动翻页，后台任务实时显示进度）；更长周期上限 10 万根";
    } else {
      const meta = sourcesMeta[src];
      depthEl.textContent = (meta && meta.hint) ? meta.hint + (meta.available ? "" : "（当前环境不可用）") : "";
    }
  }
  const sel = $("dlSourceSelect");
  if (sel) {
    sel.innerHTML = "";
    const order = ["tradingview", "binance", "okx", "tongdaxin"];
    for (const id of order) {
      const meta = sourcesMeta[id] || { label: id, available: false, hint: "", timeframes: [] };
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = meta.available ? meta.label : `${meta.label}（不可用）`;
      opt.title = meta.hint || "";
      if (!meta.available) opt.classList.add("src-unavailable");
      if (id === "tradingview" && !meta.available) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", () => {
      const meta = sourcesMeta[sel.value];
      refreshDlTfOptions(meta);
      refreshDlBarsLimit();
    });
    sel.dispatchEvent(new Event("change")); // 初始化：重建周期下拉 + 数量上限
  }
  const tfSel = $("dlTimeframeSelect");
  if (tfSel) tfSel.addEventListener("change", refreshDlBarsLimit);
}

async function downloadAndPrepare() {
  const inputEl = $("dlSymbolInput");
  const symbol = (inputEl?.value || "").trim();
  const timeframe = $("dlTimeframeSelect")?.value || "1h";
  const source = $("dlSourceSelect")?.value || "tradingview";
  const sourceLabel = { tradingview: "TradingView", binance: "Binance", okx: "OKX", tongdaxin: "通达信" }[source] || source;
  if (!symbol) {
    setDlHint("请先输入品种名", true);
    inputEl?.focus();
    return;
  }
  const barsEl = $("dlBarsInput");
  let nBars = parseInt(barsEl?.value || "", 10);
  if (!Number.isFinite(nBars)) nBars = 5000;
  const isBinanceDeep = source === "binance" && ["1m", "3m", "5m", "15m"].includes(timeframe);
  nBars = Math.min(isBinanceDeep ? 1000000 : 100000, Math.max(100, nBars));
  if (barsEl) barsEl.value = String(nBars);
  const btn = $("dlFetchBtn");
  const prev = btn?.textContent;
  if (btn) { btn.disabled = true; btn.textContent = "下载中…"; }
  const depthNote = source === "tradingview" ? "（服务端深度上限约 1 万根/次，日线不受限）" : "";
  setDlHint(`正在从 ${sourceLabel} 下载 ${symbol} ${timeframe}，请求 ${nBars.toLocaleString()} 根${depthNote}（大任务后台执行，可随时查看进度）…`);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 15 * 60 * 1000); // 后台任务最长 15 分钟
  try {
    const res = await fetchJSON("/api/data/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, timeframe, source, n_bars: nBars }),
      signal: ctrl.signal,
      retries: 0,
    });
    const info = res.job_id ? await pollDownloadJob(res.job_id, sourceLabel, ctrl.signal) : res;
    const span = info.start_date && info.end_date ? `（${info.start_date} → ${info.end_date}）` : "";
    const total = (info.bars ?? 0).toLocaleString();
    const sum = info.no_change
      ? `档案已是最新，无新增（共 ${total} 根，未改动文件）`
      : info.merged && (info.added_bars || info.new_bars) > 0
        ? `增量追加 ${(info.added_bars || info.new_bars).toLocaleString()} 根，合并后共 ${total} 根`
        : `已下载 ${total} 根 K 线`;
    setDlHint(`✓ [${sourceLabel}] ${sum}${span}，保存为 ${info.filename}`);
    hideDlProgress();
    await applyDataFileResult(info);
    refreshDownloadHistory();
  } catch (e) {
    hideDlProgress();
    setDlHint("下载失败: " + (e?.message || "未知错误"), true);
  } finally {
    clearTimeout(timer);
    if (btn) { btn.disabled = false; btn.textContent = prev || "联网下载数据"; }
  }
}

async function backfillToOrigin() {
  const symbol = ($("dlSymbolInput")?.value || "").trim();
  const timeframe = $("dlTimeframeSelect")?.value || "1h";
  const source = $("dlSourceSelect")?.value || "tradingview";
  const sourceLabel = { tradingview: "TradingView", binance: "Binance", okx: "OKX", tongdaxin: "通达信" }[source] || source;
  if (!symbol) {
    setDlHint("请先输入品种名", true);
    $("dlSymbolInput")?.focus();
    return;
  }
  const barsEl = $("dlBarsInput");
  let nBars = parseInt(barsEl?.value || "", 10);
  if (!Number.isFinite(nBars)) nBars = 5000;
  nBars = Math.min(100000, Math.max(100, nBars));
  const btn = $("dlBackfillBtn");
  const prev = btn?.textContent;
  if (btn) { btn.disabled = true; btn.textContent = "回溯中…"; }
  setDlHint(`正在从 ${sourceLabel} 回溯 ${symbol} ${timeframe} 更早历史（每页 ${nBars.toLocaleString()} 根，直到数据源留存起点）…`);
  try {
    const res = await fetchJSON("/api/data/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, timeframe, source, n_bars: nBars, mode: "backfill" }),
      retries: 0,
    });
    if (!res.job_id) throw new Error(res.message || "回溯任务提交失败");
    const info = await pollDownloadJob(res.job_id, sourceLabel, null);
    const total = (info.bars ?? 0).toLocaleString();
    const span = info.start_date && info.end_date ? `（${info.start_date} → ${info.end_date}）` : "";
    setDlHint(
      `✓ [${sourceLabel}] 回溯${info.reached_origin ? "至数据源起点" : "完成"}：新增 ` +
      `${(info.backfilled_bars ?? 0).toLocaleString()} 根，档案共 ${total} 根${span}（${info.pages ?? 0} 页）`
    );
    hideDlProgress();
    await applyDataFileResult(info);
    refreshDownloadHistory();
  } catch (e) {
    hideDlProgress();
    setDlHint("回溯失败: " + (e?.message || "未知错误"), true);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = prev || "回溯到数据源起点"; }
  }
}

async function pollDownloadJob(jobId, sourceLabel, signal) {
  const started = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 1200));
    const q = await fetchJSON("/api/data/download-queue", { signal, retries: 0 });
    const jobs = q.jobs || [];
    const st = jobs.find((j) => j.job_id === jobId);
    if (!st) throw new Error("任务不存在（服务可能已重启且任务已过期）");
    renderDlJobProgress(st);
    renderDlQueue(jobs, jobId);
    if (st.status === "done") return st.result;
    if (st.status === "error") throw new Error(st.error || "下载失败");
    const secs = Math.round((Date.now() - started) / 1000);
    if (st.status === "queued") {
      setDlHint(`排队中（第 ${st.position || "?"} 位）[${sourceLabel}]：${st.symbol} ${st.timeframe} · ${st.n_bars?.toLocaleString()} 根`);
    } else {
      const bars = st.bars_fetched ? ` · 已取 ${st.bars_fetched.toLocaleString()} 根` : "";
      setDlHint(`下载中 [${sourceLabel}]：${st.phase || "…"}${bars} · ${secs}s`);
    }
  }
}

function renderDlJobProgress(st) {
  const block = $("dlJobProgress");
  if (!block) return;
  if (st.status === "done" || st.status === "error") {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  const pct = st.status === "queued"
    ? 0
    : Math.min(100, Math.round(((st.bars_fetched || 0) / Math.max(st.n_bars || 1, 1)) * 100));
  $("dlJobBar").style.width = pct + "%";
  const label = { tradingview: "TradingView", binance: "Binance", okx: "OKX", tongdaxin: "通达信" }[st.source] || st.source;
  const bars = st.bars_fetched ? `${st.bars_fetched.toLocaleString()} / ${(st.n_bars || 0).toLocaleString()} 根` : `最多 ${(st.n_bars || 0).toLocaleString()} 根`;
  $("dlJobText").textContent = `[${label}] ${st.symbol} ${st.timeframe}　${st.phase}${st.status === "queued" ? `（排队第 ${st.position || "?"} 位）` : ""}　·　${bars}`;
  $("dlJobPct").textContent = st.status === "queued" ? "排队中" : `${pct}%`;
}

function renderDlQueue(jobs, activeJobId) {
  const box = $("dlQueueList");
  if (!box) return;
  const visible = jobs.filter((j) => j.status === "queued" || j.status === "running");
  const recent = jobs.filter((j) => j.status !== "queued" && j.status !== "running").slice(0, 4);
  if (!visible.length && !recent.length) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  box.hidden = false;
  const label = { tradingview: "TradingView", binance: "Binance", okx: "OKX", tongdaxin: "通达信" };
  const rows = [...visible, ...recent].map((j) => {
    const l = label[j.source] || j.source;
    const pos = j.status === "queued" ? `#${j.position || "?"}` : j.status === "running" ? "▶" : j.status === "error" ? "✖" : "✔";
    const info = j.status === "done"
      ? `${l} ${j.symbol} ${j.timeframe} · ${(j.result?.n_bars || 0).toLocaleString()} 根`
      : `${l} ${j.symbol} ${j.timeframe} · ${j.bars_fetched || 0}/${j.n_bars || 0} 根${j.status === "queued" ? ` · ${j.phase}` : ""}`;
    const cls = j.job_id === activeJobId ? " ql-active" : "";
    return `<div class="ql-item${j.status === "done" ? " ql-done" : ""}${j.status === "error" ? " ql-err" : ""}${cls}"><span class="ql-pos">${pos}</span><span>${info}</span></div>`;
  });
  const title = visible.length ? (visible.length > 1 ? "下载队列" : "当前下载") : "最近下载";
  box.innerHTML = `<div style="margin-bottom:4px;opacity:.7">${title}${visible.length > 1 ? ` · ${visible.length} 个任务` : ""}</div>${rows.join("")}`;
}

function dlDateLabel(r) {
  const d = r.downloaded_at ? String(r.downloaded_at).slice(0, 10) : "";
  if (d) return d;
  return r.start ? `${r.start} 起` : "—";
}

function renderDownloadHistory(h) {
  const srcStats = $("dlSourceStats");
  const srcs = (h && h.sources) || [];
  if (srcStats) {
    if (!srcs.length) {
      srcStats.hidden = true;
      srcStats.innerHTML = "";
    } else {
      srcStats.hidden = false;
      srcStats.innerHTML =
        `<span class="dl-src-title">数据目录统计</span>` +
        srcs
          .map(
            (s) =>
              `<span class="dl-src-chip" title="${s.label || s.source || "本地"}：${s.files} 个文件 · ${(s.bars || 0).toLocaleString()} 根${s.latest ? ` · 最近 ${String(s.latest).slice(0, 10)}` : ""}"><b>${s.label || s.source || "本地"}</b><i>${s.files} 文件</i><i>${(s.bars || 0).toLocaleString()} 根</i></span>`
          )
          .join("");
    }
  }

  // 训练数据卡片下方的文件下拉（含来源/根数/时间跨度，点击直接切换）
  const sel = $("dlFileListSelect");
  if (sel) {
    const all = (h && h.downloads) || [];
    const current = selectedDataFile || "";
    sel.innerHTML =
      `<option value="">— 选择 data/training/ 文件切换 —</option>` +
      all
        .map((r) => {
          const span = r.start && r.end ? ` · ${r.start}→${r.end}` : "";
          const bars = r.bars != null ? Number(r.bars).toLocaleString() : "?";
          // 精确路径匹配：切片文件（data/slices/…）不会错选到同名的全量档案选项
          const isCur = !!r.data_file && r.data_file === current;
          return `<option value="${r.data_file || ""}" ${isCur ? "selected" : ""}>` +
            `${r.source_label || "本地"} · ${r.symbol || "?"} ${r.timeframe || ""} · ${bars} 根${span}</option>`;
        })
        .join("");
  }

  const panel = $("dlHistoryPanel");
  if (!panel) return;
  const rows = ((h && h.downloads) || []).slice(0, 20);
  const body = $("dlHistoryBody");
  if (!body) return;
  panel.hidden = !rows.length;
  const count = $("dlHistoryCount");
  if (count) {
    count.textContent = h && h.total_files
      ? `共 ${h.total_files} 文件 · ${(h.total_bars || 0).toLocaleString()} 根（最新 ${rows.length} 条）`
      : "";
  }
  if (!rows.length) {
    body.innerHTML = `<tr class="empty-row"><td colspan="5">暂无下载记录</td></tr>`;
    return;
  }
  body.innerHTML = rows
    .map((r) => {
      const span = r.start && r.end ? `${r.start} ~ ${r.end}` : "";
      const tip = `${r.file}${span ? " · 数据 " + span : ""}`;
      return `<tr title="${tip}">
        <td>${r.source_label || "本地"}</td>
        <td>${r.symbol || "—"}</td>
        <td>${r.timeframe || "—"}</td>
        <td>${r.bars != null ? Number(r.bars).toLocaleString() : "—"}</td>
        <td>${dlDateLabel(r)}</td>
      </tr>`;
    })
    .join("");
}

async function refreshDownloadHistory() {
  try {
    const h = await fetchJSON("/api/data/download-history?limit=100", { silent: true, retries: 0 });
    renderDownloadHistory(h || {});
  } catch (_) {
    /* 面板缺失/接口暂不可用时不打扰用户 */
  }
}

function hideDlProgress() {
  const block = $("dlJobProgress");
  if (block) block.hidden = true;
  const box = $("dlQueueList");
  if (box) {
    const stillActive = [...(box.querySelectorAll(".ql-item"))].some((el) => el.textContent.includes("▶"));
    if (!stillActive) { box.hidden = true; box.innerHTML = ""; }
  }
}

function setManualPathHint(text, isError = false) {
  const el = $("manualPathHint");
  if (!el) return;
  el.textContent = text || "";
  el.classList.toggle("invalid", !!isError && !!text);
  el.classList.toggle("valid", !isError && !!text);
}

function applyDataFileResult(res) {
  renderDataFileCard(res);
  selectedSymbol = res.symbol;
  return loadSymbolChart(res.symbol);
}

async function browseDataFile() {
  setManualPathHint("");
  let res;
  try {
    res = await fetchJSON("/api/data-file/browse", { method: "POST", retries: 0 });
  } catch (e) {
    setManualPathHint("文件选择器不可用（" + (e?.message || "未知错误") + "）— 请手动输入路径", true);
    const inputEl = $("manualPathInput");
    if (inputEl) inputEl.focus();
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  // 兼容旧响应形状
  if (!res.dialog || !res.session) {
    if (res.cancelled) setManualPathHint("已取消选择 — 也可在上方手动输入路径", true);
    else await applyDataFileResult(res);
    return;
  }
  // 原生对话框由服务端进程弹出（默认打开 data/training/ 下载目录）；
  // 轮询直到用户选完/取消——不再有客户端超时竞态，翻目录多久都行
  setManualPathHint(res.already_open ? "已有选择窗口打开，等待完成后自动载入…" : "正在打开文件选择器（默认 data/training/）…");
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 800));
      const p = await fetchJSON(
        "/api/data-file/browse-poll?session=" + encodeURIComponent(res.session),
        { retries: 0 }
      );
      if (!p.done) continue;
      if (p.cancelled) {
        setManualPathHint("已取消选择 — 也可在上方手动输入路径", true);
        return;
      }
      setManualPathHint("");
      await applyDataFileResult(p);
      return;
    }
  } catch (e) {
    setManualPathHint("文件选择失败（" + (e?.message || "未知错误") + "）— 也可在上方手动输入路径", true);
  }
}

async function applyDataFilePath(raw) {
  if (!raw) {
    setManualPathHint("请先输入 Parquet 文件完整路径", true);
    return false;
  }
  setManualPathHint("正在验证路径…");
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 60000);
  try {
    const res = await fetchJSON(
      "/api/data-file/browse?path=" + encodeURIComponent(raw),
      { method: "POST", retries: 1, signal: ctrl.signal }
    );
    setManualPathHint("");
    await applyDataFileResult(res);
    return true;
  } catch (e) {
    if (ctrl.signal.aborted || e.name === "AbortError") {
      // 超时：请检查路径是否为本地 Parquet；页面切换触发的取消则安静处理
      setManualPathHint(
        e.name === "AbortError" && !ctrl.signal.aborted
          ? "验证已取消（页面切换/刷新）— 请重试"
          : "验证超时（60s）— 大文件首次读取较慢，请重试或检查路径",
        true
      );
    } else {
      setManualPathHint("路径无效: " + e.message, true);
    }
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function applyManualPath() {
  await applyDataFilePath(($("manualPathInput")?.value || "").trim());
}

async function startTraining() {
  if (!selectedDataFile) {
    await logClientError("请先选择数据文件");
    return;
  }
  try {
    const res = await fetchJSON("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_file: selectedDataFile, from_scratch: false, ...collectTrainRange() }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol;
    renderDataFileCard(res.data_file);
    await refreshOverview();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function retrainFromScratch() {
  if (!selectedDataFile) {
    await logClientError("请先选择数据文件");
    return;
  }
  const ok = window.confirm(
    "重新训练会清除该品种的检查点，从第 0 步重新搜索。\n" +
      "已有的更优策略会保留，只有挖到更高分才会覆盖。\n\n" +
      "确定要重新训练吗？"
  );
  if (!ok) return;
  try {
    const res = await fetchJSON("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_file: selectedDataFile, from_scratch: true, ...collectTrainRange() }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol;
    renderDataFileCard(res.data_file);
    await refreshOverview();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function updateExportBtn(progress, strategies) {
  const sym = progress?.symbol || selectedSymbol;
  const hasStrategy = progress?.has_strategy || (strategies || []).some((s) => s.symbol === sym);
  const btn = $("exportBtn");
  if (btn) btn.disabled = !sym || !hasStrategy;
}

function updateTrainingBtns(progress, training) {
  const sym = progress?.symbol || selectedSymbol;
  const active = training?.active;
  const hasCheckpoint = Boolean(progress?.has_checkpoint);
  const exportBtn = $("exportTrainingBtn");
  const importBtn = $("importTrainingBtn");

  let exportTitle = "打包 checkpoint、训练曲线与策略为 zip";
  if (!sym) {
    exportTitle = "请先选择数据文件";
  } else if (active) {
    exportTitle = "训练进行中，请停止后再导出";
  } else if (!hasCheckpoint) {
    exportTitle = "该品种尚无检查点：至少训练满 20 步后才会生成（每 20 步保存一次）";
  }

  if (exportBtn) {
    exportBtn.disabled = !sym || !hasCheckpoint || !!active;
    exportBtn.title = exportTitle;
  }
  if (importBtn) {
    importBtn.disabled = !sym || !!active;
    importBtn.title = active ? "训练进行中，请停止后再导入" : "上传 .zip 或 .pt，下次训练断点续训";
  }
}

async function exportTraining() {
  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }
  const path = `/api/training/${encodeURIComponent(sym)}/export`;
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, path));
    }
    const blob = await res.blob();
    const disp = res.headers.get("Content-Disposition") || "";
    const m = /filename="([^"]+)"/.exec(disp);
    const filename = m ? m[1] : `training_${sym.replace(/\./g, "_")}.zip`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    await logClientError(`导出训练失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function triggerImportTraining() {
  const input = $("importTrainingFile");
  if (input) {
    input.value = "";
    input.click();
  }
}

async function handleImportTrainingFile(event) {
  const input = event.target;
  const file = input.files?.[0];
  if (!file) return;

  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/api/training/import?symbol=${encodeURIComponent(sym)}`, {
      method: "POST",
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(formatApiError(data, res.status, "/api/training/import"));
    }
    if (data.symbol && data.symbol !== sym) {
      selectedSymbol = data.symbol;
    }
    clientErrors.push(`[${new Date().toLocaleString()}] ${data.message || "训练文件导入成功"}`);
    if (clientErrors.length > 80) clientErrors = clientErrors.slice(-80);
    renderDebugView();
    await refreshOverview();
  } catch (e) {
    await logClientError(`导入训练失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    input.value = "";
  }
}

function parseContentDispositionFilename(header) {
  if (!header) return null;
  const utf8 = /filename\*=UTF-8''([^;]+)/i.exec(header);
  if (utf8) return decodeURIComponent(utf8[1]);
  const plain = /filename="([^"]+)"/i.exec(header) || /filename=([^;]+)/i.exec(header);
  return plain ? plain[1].trim() : null;
}

async function exportStrategy() {
  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }
  const path = `/api/strategies/${encodeURIComponent(sym)}/export`;
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, path));
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download =
      parseContentDispositionFilename(res.headers.get("Content-Disposition")) ||
      `strategy_${sym.replace(/\./g, "_")}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    await logClientError(`导出策略失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function stopTraining() {
  try {
    const res = await fetchJSON("/api/training/stop", { method: "POST" });
    await refreshOverview();
    const sym = res.training?.job?.symbol || selectedSymbol;
    await applyBestStrategyForBacktest(sym, res.strategy_file);
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ═══════════════════════════════════════════════════════════════════
// 分页切换
// ═══════════════════════════════════════════════════════════════════
let _switchWorkTimer = null;
// 切页动画（pageIn 0.45s）播放期间不跑目标页的重活：拉取 + 整页重渲染单次可占
// 主线程 100-200ms，若和淡入同时进行会掐在动画中间掉帧（观感即「点击卡一下」）。
// 统一延后到动画播完再执行；用户中途连续切页时只认最后一次。
const SWITCH_WORK_DELAY_MS = 500;

function schedulePageWork(page) {
  if (_switchWorkTimer) clearTimeout(_switchWorkTimer);
  _switchWorkTimer = setTimeout(() => {
    _switchWorkTimer = null;
    if (currentPage !== page) return; // 已切走：由最新一次切换接管
    if (page === "backtest") {
      loadBacktestStrategyContext();
      refreshBacktest();
    } else if (page === "realtime") {
      initRealtimeOnce();
      refreshRealtime();
    } else if (page === "paper") {
      initPaperOnce();
      refreshPaper();
    }
  }, SWITCH_WORK_DELAY_MS);
}

function switchPage(page) {
  if (page !== "train" && page !== "backtest" && page !== "realtime" && page !== "paper") return;
  currentPage = page;
  document.querySelectorAll(".stepper .step").forEach((s) => {
    s.classList.toggle("active", s.dataset.page === page);
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("active", p.id === `page-${page}`);
  });
  schedulePageWork(page);
}

// ═══════════════════════════════════════════════════════════════════
// 回测：格式化辅助
// ═══════════════════════════════════════════════════════════════════
function fmtPct(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(digits) + "%";
}
function fmtSigned(v, digits = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + Number(v).toFixed(digits);
}

// ═══════════════════════════════════════════════════════════════════
// 回测：状态轮询 + UI 更新
// ═══════════════════════════════════════════════════════════════════
async function refreshBacktest() {
  let st;
  try {
    st = await fetchJSON("/api/backtest/status", { silent: true });
  } catch (_) {
    return;
  }
  btActive = !!st.active;
  const job = st.job;
  const state = job?.state || "idle";

  // 按钮
  const stopBtn = $("btStopBtn");
  updateBtStartBtn();
  if (stopBtn) stopBtn.disabled = !btActive;

  // 缓存刷新键：用最近一次任务的结束/开始时间
  btBuster = job?.finished_at || job?.started_at || btBuster;

  // 日志
  const logView = $("btLogView");
  const logText = (st.log_tail || []).join("\n") || "等待任务…";
  if (logView) {
    const atBottom = isViewAtBottom(logView);
    logView.textContent = logText;
    if (atBottom) logView.scrollTop = logView.scrollHeight;
  }
  if ($("btLogHint")) $("btLogHint").textContent = job?.log_path || "—";

  // 阶段进度条
  updateBacktestPhase(st, state);

  if (state === "failed") {
    const alertKey = `${job?.log_path || ""}|${job?.finished_at || ""}|${job?.exit_code ?? ""}`;
    if (alertKey && alertKey !== btLastAlertKey) {
      btLastAlertKey = alertKey;
      const errLine = job?.error ? `\n错误: ${job.error}` : "";
      showErrorPopup(
        "回测失败",
        `退出码: ${job?.exit_code ?? "?"}${errLine}\n日志: ${job?.log_path || "—"}\n\n${logText}`
      );
    }
  }

  // 结果报告（非运行态时刷新，运行态保留上次结果）
  if (!btActive) {
    await refreshBacktestReport();
  }
}

const BT_STATE_LABEL = {
  running: "回测中",
  completed: "已完成",
  failed: "失败",
  stopped: "已停止",
  idle: "待机",
};

function updateBacktestPhase(st, state) {
  const fill = $("btPhaseFill");
  const label = $("btPhaseLabel");
  if (!fill || !label) return;

  const total = st.phase_total || 7;
  const idx = st.phase_index || 0;

  let pct;
  if (btActive) {
    pct = Math.min(96, Math.round(((idx + 1) / total) * 100));
    label.textContent = `${st.phase_label || "回测中"}…`;
    fill.classList.add("animate");
  } else if (state === "completed") {
    pct = 100;
    label.textContent = "完成";
    fill.classList.remove("animate");
  } else if (state === "failed" || state === "stopped") {
    pct = Math.min(96, Math.round(((idx + 1) / total) * 100));
    label.textContent = BT_STATE_LABEL[state];
    fill.classList.remove("animate");
  } else {
    pct = 0;
    label.textContent = "待机";
    fill.classList.remove("animate");
  }
  fill.style.width = pct + "%";
}

async function refreshBacktestReport() {
  let data;
  const sym = selectedStrategySymbol || selectedSymbol;
  const url = sym
    ? `/api/backtest/report?symbol=${encodeURIComponent(sym)}`
    : "/api/backtest/report";
  try {
    data = await fetchJSON(url, { silent: true });
  } catch (_) {
    return;
  }
  if (!data.available || !data.report) {
    if ($("btPortfolioHint")) $("btPortfolioHint").textContent = "尚未运行回测";
    lastEquityData = null;
    btPortfolioSig = "";
    renderEquity(null);
    return;
  }
  // 先取资金曲线（写入 lastEquityData），再渲染绩效卡，让 sparkline 用上真实数据
  btReportRunId = data.report.run_id || null;
  await refreshEquityCurve();
  if (btReportRunId && lastEquityData?.run_id && lastEquityData.run_id !== btReportRunId) {
    // 抓到了不同 run 的文件（写盘间隙/并发回测）：等一拍重取一次，仍不一致由 renderEquity 展示警示
    await new Promise((r) => setTimeout(r, 1500));
    await refreshEquityCurve();
  }
  renderPortfolio(data.report);
  renderBacktestTable(data.report.symbols || {});
}

async function refreshEquityCurve() {
  const sym = selectedStrategySymbol || selectedSymbol;
  const url = sym
    ? `/api/backtest/equity?symbol=${encodeURIComponent(sym)}`
    : "/api/backtest/equity";
  try {
    const data = await fetchJSON(url, { silent: true });
    lastEquityData = data?.available ? data.data : null;
    renderEquity(data);
  } catch (_) {
    lastEquityData = null;
    renderEquity(null);
  }
}

// ═══════════════════════════════════════════════════════════════════
// 迷你 sparkline + 数字滚动动画（终端仪表盘质感）
// ═══════════════════════════════════════════════════════════════════
const METRIC_FMT = {
  pct: (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%",
  signed: (v) => (v >= 0 ? "+" : "") + v.toFixed(3),
  ratio: (v) => v.toFixed(3),
  int: (v) => Math.round(v).toLocaleString(),
  winrate: (v) => (v * 100).toFixed(1) + "%",
  strength: (v) => Math.round(v * 100) + "%",
};

function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

// 短促 count-up（≈420ms, easeOutCubic），克制不浮夸
function animateCount(el, to, fmt) {
  const fn = METRIC_FMT[fmt] || ((v) => String(v));
  if (!Number.isFinite(to)) {
    el.textContent = "—";
    return;
  }
  if (prefersReducedMotion()) {
    el.textContent = fn(to);
    return;
  }
  const dur = 420;
  const t0 = performance.now();
  function frame(now) {
    const p = Math.min(1, (now - t0) / dur);
    const e = 1 - Math.pow(1 - p, 3); // easeOutCubic
    el.textContent = fn(to * e);
    if (p < 1) requestAnimationFrame(frame);
    else el.textContent = fn(to);
  }
  requestAnimationFrame(frame);
}

function runCountUp(root) {
  if (!root) return;
  root.querySelectorAll("[data-count]").forEach((el) => {
    animateCount(el, parseFloat(el.dataset.count), el.dataset.fmt || "");
  });
}

// 均匀降采样为 <= target 个有限点
function downsampleSeries(arr, target) {
  const clean = (arr || [])
    .map(Number)
    .filter((v) => Number.isFinite(v));
  if (clean.length <= target) return clean;
  const out = [];
  const step = (clean.length - 1) / (target - 1);
  for (let i = 0; i < target; i++) out.push(clean[Math.round(i * step)]);
  return out;
}

// 生成极小趋势微线（内联 SVG，轻量、清晰）
function sparklineSVG(values, { color = "#5eead4", fillRGB = null, w = 74, h = 22 } = {}) {
  const v = downsampleSeries(values, 56);
  if (v.length < 2) return "";
  const min = Math.min(...v);
  const max = Math.max(...v);
  const range = max - min || 1;
  const n = v.length;
  const x = (i) => (i / (n - 1)) * w;
  const y = (val) => h - 2 - ((val - min) / range) * (h - 4);
  const line = "M" + v.map((val, i) => `${x(i).toFixed(1)} ${y(val).toFixed(1)}`).join(" L ");
  const area = fillRGB
    ? `<path d="${line} L ${w} ${h} L 0 ${h} Z" fill="rgba(${fillRGB},0.14)" stroke="none"/>`
    : "";
  const dot = `<circle cx="${x(n - 1).toFixed(1)}" cy="${y(v[n - 1]).toFixed(1)}" r="1.6" fill="${color}"/>`;
  return `<svg class="spark-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">${area}<path d="${line}" fill="none" stroke="${color}" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"/>${dot}</svg>`;
}

// 取当前主资金曲线序列（组合优先，否则第一个品种）
function mainEquitySeries() {
  const d = lastEquityData;
  if (!d) return null;
  if (d.portfolio) return d.portfolio;
  const syms = d.symbols || {};
  const names = Object.keys(syms);
  return names.length ? syms[names[0]] : null;
}

function renderPortfolio(report) {
  const grid = $("btPortfolioGrid");
  if (!grid) return;
  const p = report.portfolio || {};
  const focus = report.focus_symbol || Object.keys(report.symbols || {})[0] || "";
  const symData = focus ? (report.symbols || {})[focus] : null;

  if (!Object.keys(p).length) {
    grid.innerHTML = '<div class="metric-empty">回测结果无绩效数据</div>';
    btPortfolioSig = "";
    return;
  }

  const plNum = Number(symData?.profit_loss_ratio ?? p.profit_loss_ratio);
  const nTrades = symData?.n_trades ?? p.n_trades;
  const winRate = symData?.win_rate;
  const mdd = symData?.max_drawdown ?? p.max_drawdown;

  // sparkline 数据源：主资金曲线 + 滚动夏普
  const eq = mainEquitySeries();
  const posColor = p.total_return >= 0 ? "#4ade80" : "#f87171";
  const posRGB = p.total_return >= 0 ? "74, 222, 128" : "248, 113, 113";
  const equitySpark = eq ? sparklineSVG(eq.equity, { color: posColor, fillRGB: posRGB }) : "";
  const rollSpark = eq ? sparklineSVG(eq.rolling_sharpe, { color: "#5eead4", fillRGB: "94, 234, 212" }) : "";

  const cards = [
    { label: "总收益", raw: p.total_return, fmt: "pct", cls: p.total_return >= 0 ? "pos" : "neg", spark: equitySpark },
    { label: "Sharpe", raw: p.sharpe, fmt: "signed", cls: "accent", spark: rollSpark },
    { label: "Sortino", raw: p.sortino, fmt: "signed", cls: "accent", spark: rollSpark },
    { label: "最大回撤", raw: Number.isFinite(Number(mdd)) ? Number(mdd) : null, fmt: "pct", cls: Number.isFinite(Number(mdd)) && Number(mdd) < 0 ? "neg" : "accent" },
    { label: "盈亏比", raw: Number.isFinite(plNum) ? plNum : null, fmt: "ratio", cls: Number.isFinite(plNum) ? "accent" : "" },
    { label: "交易数", raw: Number.isFinite(Number(nTrades)) ? Number(nTrades) : null, fmt: "int", cls: "" },
    { label: "胜率", raw: winRate != null ? Number(winRate) : null, fmt: "winrate", cls: "" },
  ];

  // 签名守卫：数值/焦点/资金曲线未变则不重建，避免每次轮询重播动画
  const sig = [focus, btEquitySig, ...cards.map((c) => c.raw)].join("|");
  if (sig === btPortfolioSig) {
    if ($("btPortfolioHint")) $("btPortfolioHint").textContent = focus ? `${focus} 回测绩效` : "回测绩效";
    return;
  }
  btPortfolioSig = sig;

  grid.innerHTML = cards
    .map((c) => {
      const cardCls = c.cls === "pos" || c.cls === "neg" ? c.cls : "";
      const finite = c.raw != null && Number.isFinite(c.raw);
      const finalText = finite ? METRIC_FMT[c.fmt](c.raw) : "—";
      const countAttr = finite ? ` data-count="${c.raw}" data-fmt="${c.fmt}"` : "";
      const spark = c.spark ? `<div class="metric-spark">${c.spark}</div>` : "";
      return `
    <div class="metric-card ${cardCls}">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}"${countAttr}>${finalText}</div>
      ${spark}
    </div>`;
    })
    .join("");

  runCountUp(grid);

  if ($("btPortfolioHint")) {
    $("btPortfolioHint").textContent = focus ? `${focus} 回测绩效` : "回测绩效";
  }
}

function renderBacktestTable(symbols) {
  const tbody = $("btTableBody");
  if (!tbody) return;
  const rows = Object.entries(symbols);
  // 绩效明细已并入「回测绩效」面板(btSymbolBlock):单品种时指标卡已覆盖同口径指标,
  // 明细表只在多品种(>1)回测时展示,避免与卡片重复。
  const wrap = $("btSymbolBlock");
  const showTable = rows.length > 1;
  if (wrap) wrap.hidden = !showTable;
  if (!showTable) {
    if (!rows.length) tbody.innerHTML = '<tr class="empty-row"><td colspan="8">暂无回测结果</td></tr>';
    if ($("btTableHint")) $("btTableHint").textContent = "—";
    return;
  }
  if ($("btTableHint")) $("btTableHint").textContent = `${rows.length} 个品种`;
  tbody.innerHTML = rows
    .map(([sym, d]) => {
      const retCls = (d.total_return || 0) >= 0 ? "pos" : "neg";
      const shCls = (d.sharpe || 0) >= 0 ? "pos" : "neg";
      const dd = d.max_drawdown;
      const ddCls = Number.isFinite(Number(dd)) && Number(dd) < 0 ? "neg" : "";
      return `
      <tr>
        <td class="sym-cell">${sym}</td>
        <td class="${retCls}">${fmtPct(d.total_return)}</td>
        <td class="${shCls}">${fmtSigned(d.sharpe)}</td>
        <td>${fmtSigned(d.sortino)}</td>
        <td class="${ddCls}">${fmtPct(dd)}</td>
        <td>${Number.isFinite(Number(d.profit_loss_ratio)) ? Number(d.profit_loss_ratio).toFixed(3) : "—"}</td>
        <td>${d.n_trades ?? "—"}</td>
        <td>${d.win_rate != null ? (d.win_rate * 100).toFixed(1) + "%" : "—"}</td>
      </tr>`;
    })
    .join("");
}

// ═══════════════════════════════════════════════════════════════════
// 持仓正交组合矩阵（hold_matrix 7×7 · results/hold_matrix_latest.json）
// ═══════════════════════════════════════════════════════════════════
const MATRIX_POLICY_CN = {
  signal: "信号跟随", risk: "止盈止损保护", hybrid: "熔断止损",
  be: "保本追踪", time: "时间止损", chandelier: "吊灯(ATR)", dd: "回撤熔断(DD)",
};

function matrixPolicyCn(id) {
  const k = id || "signal";
  if (MATRIX_POLICY_CN[k]) return MATRIX_POLICY_CN[k];
  const p = holdPolicyById[k];
  return p ? p.name : k;
}

function matrixComboCn(pid) {
  return (pid || "signal").split("+").filter(Boolean).map(matrixPolicyCn).join(" + ");
}

// 单元格定位：与 scripts/hold_matrix.py 的 canon 同规则（按 policies 顺序归一）
function matrixCellPid(policies, a, b) {
  const idx = {};
  policies.forEach((p, i) => (idx[p] = i));
  const parts = [a, b].sort((x, y) => (idx[x] ?? 99) - (idx[y] ?? 99)).filter((x) => x !== "signal");
  const uniq = [...new Set(parts)];
  return uniq.length ? uniq.join("+") : "signal";
}

function matrixNumCls(v, invert) {
  const n = Number(v);
  if (!Number.isFinite(n) || n === 0) return "";
  const good = invert ? n < 0 : n > 0;
  return good ? "pos" : "neg";
}

function matrixFmt(key, v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (key === "total_return") return (n >= 0 ? "+" : "") + (n * 100).toFixed(2) + "%";
  if (key === "max_drawdown") return (n * 100).toFixed(1) + "%";
  if (key === "win_rate") return (n * 100).toFixed(0) + "%";
  if (key === "n_trades") return String(Math.round(n));
  return (n >= 0 ? "+" : "") + n.toFixed(2);
}

const MATRIX_TABLES = [
  { key: "total_return", title: "总收益矩阵（%）", invert: false },
  { key: "sharpe", title: "夏普矩阵", invert: false },
  { key: "max_drawdown", title: "最大回撤矩阵（%）", invert: true },
  { key: "profit_loss_ratio", title: "盈亏比矩阵", invert: false },
];

function matrixTableHtml(policies, cells, cfg) {
  const head = policies.map((p) => `<th>${matrixPolicyCn(p)}</th>`).join("");
  const rows = policies.map((a) => {
    const tds = policies.map((b) => {
      const pid = matrixCellPid(policies, a, b);
      const cell = cells[pid] || {};
      const v = cell[cfg.key];
      const cls = matrixNumCls(v, cfg.invert);
      return `<td class="${cls}">${matrixFmt(cfg.key, v)}</td>`;
    });
    return `<tr><td class="sym-cell">${matrixPolicyCn(a)}</td>${tds.join("")}</tr>`;
  });
  return `<h4 class="bt-matrix-sub">${cfg.title}</h4><div class="table-wrap bt-matrix-wrap"><table class="bt-table matrix-table"><thead><tr><th>行\列</th>${head}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function matrixExitBrief(reasons) {
  const parts = Object.entries(reasons || {})
    .sort((x, y) => y[1] - x[1])
    .map(([k, n]) => `${k}×${n}`);
  const s = parts.join(" ");
  return s.length > 120 ? s.slice(0, 117) + "…" : s;
}

// 多目标帕累托前端兜底（旧结果文件无 pareto 标志时自算）：
// 目标均为越高越好（收益/夏普/盈亏比/最大回撤取负），不被任何其他组合支配 = 帕累托
function matrixParetoFallback(rows) {
  if (!rows || !rows.length) return new Set();
  if (rows.some((r) => "pareto" in r)) return new Set(rows.filter((r) => r.pareto).map((r) => r.combo));
  const keys = ["total_return", "sharpe", "profit_loss_ratio", "max_drawdown"];
  const vec = (r) =>
    keys.map((k) => {
      const v = Number(r[k]);
      return Number.isFinite(v) ? (k === "max_drawdown" ? -v : v) : -Infinity;
    });
  const dom = (a, b) => {
    const va = vec(a);
    const vb = vec(b);
    return va.every((x, i) => x >= vb[i]) && va.some((x, i) => x > vb[i]);
  };
  return new Set(
    rows.filter((r) => !rows.some((o) => o.combo !== r.combo && dom(o, r))).map((r) => r.combo)
  );
}

// 22 组合成本敏感性对比表（读 results/hold_matrix_cost_latest.json 经 /api/backtest/hold-matrix 下发）
function mxCostHtml(cost) {
  if (!cost || !cost.combos || !cost.combos.length) return "";
  const meta = cost.meta || {};
  const mults = meta.cost_mults || [];
  const mcell = (r, m) => {
    const v = r.return_by_mult ? r.return_by_mult[String(m)] : null;
    return v == null
      ? "<td>—</td>"
      : `<td class="${matrixNumCls(v, false)}">${matrixFmt("total_return", v)}</td>`;
  };
  const rows = cost.combos
    .map((r) => ({ r, k: r.insens_rank == null ? 999 : r.insens_rank }))
    .sort((a, b) => a.k - b.k)
    .map(({ r }) => {
      const be = typeof r.breakeven_mult === "number"
        ? `×${(+r.breakeven_mult).toFixed(2)}`
        : (r.breakeven_mult === "∞" ? "∞" : "<0（0成本已亏）");
      const beTitle = escHtml(String(r.breakeven_mult));
      return `<tr data-combo="${escHtml(r.combo)}" class="matrix-rank-row" title="点击绘制该组合资金曲线">
        <td class="sym-cell">${escHtml(r.combo)}</td>
        <td>${escHtml(r.label || "")}</td>
        ${mults.map((m) => mcell(r, m)).join("")}
        <td title="收益由正转负的成本乘数插值点：${beTitle}">${be}</td>
        <td title="成本抬升时收益掉得最少者排前">${r.insens_rank ?? "—"}</td>
        <td title="收益对成本乘数 OLS 斜率，越接近 0 越钝感">${(r.slope_per_mult ?? 0) >= 0 ? "+" : ""}${(+r.slope_per_mult).toExponential(1)}</td>
        <td class="${matrixNumCls(r.mdd_at_1x, true)}">${matrixFmt("max_drawdown", r.mdd_at_1x)}</td>
      </tr>`;
    })
    .join("");
  return `<h4 class="bt-matrix-sub">成本敏感性（22 组合 × ${mults.length} 成本档 · 读 results/hold_matrix_cost_latest.json · 行可点击画曲线）</h4>
    <div class="table-wrap bt-matrix-wrap"><table class="bt-table matrix-table mx-cost-table"><thead><tr>
      <th>组合</th><th>名称</th>${mults.map((m) => `<th>收益@×${m}</th>`).join("")}
      <th>盈亏平衡×</th><th>钝感#</th><th>成本斜率</th><th>回撤@×1</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

// 22 组合五等分热力汇总对比表（读主 JSON 的 heat_summary；旧文件缺省时隐藏）
function mxHeatSummaryHtml(m) {
  const rows = m.heat_summary || [];
  if (!rows.length) return "";
  const cellFmt = (v) => (v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%");
  const trs = rows.map((r) => {
    const buckets = (r.bucket_stats || [])
      .map((s) => `<td class="${(s.sum || 0) >= 0 ? "pos" : "neg"}" title="桶${s.bucket} · ${s.n}根">${cellFmt(s.sum)}</td>`)
      .join("");
    const segCell = (seg) => {
      if (!seg) return "<td>—</td>";
      const reasons = (seg.top_reasons || []).join(" ") || (seg.n_trades ? "—" : "空仓");
      return `<td title="bar ${seg.start_bar}..${seg.end_bar} · ${seg.bars}根 · 出场 ${seg.n_trades} 笔">${cellFmt(seg.cum)}<span class="hint"> ${escHtml(reasons)}</span></td>`;
    };
    return `<tr data-combo="${escHtml(r.combo)}" class="matrix-rank-row" title="点击绘制该组合资金曲线">
      <td class="sym-cell">${escHtml(r.combo)}</td>
      <td>${escHtml(r.label || "")}</td>
      <td class="${matrixNumCls(r.total_return, false)}">${cellFmt(r.total_return)}</td>
      <td>${matrixFmt("sharpe", r.sharpe)}</td>
      ${buckets}
      ${segCell(r.best_segment)}
      ${segCell(r.worst_segment)}
    </tr>`;
  }).join("");
  return `
    <h4 class="bt-matrix-sub">五等分热力汇总（22 组合 · 赚/亏集中度对比，写自 results/hold_matrix_heat_summary.json）</h4>
    <div class="table-wrap bt-matrix-wrap"><table class="bt-table matrix-table mx-heat-table"><thead><tr>
      <th>组合</th><th>名称</th><th>收益</th><th>夏普</th>
      <th title="收益分布最低 20% 分位的 bar 合计">桶0(最亏)</th>
      <th>桶1</th><th>桶2</th><th>桶3</th>
      <th title="收益分布最高 20% 分位的 bar 合计">桶4(最赚)</th>
      <th title="Kadane 最大连续正收益段 · 附段内主出场原因">最赚段</th>
      <th title="Kadane 最大连续负收益段 · 附段内主出场原因/熔断">最亏段</th>
    </tr></thead><tbody>${trs}</tbody></table></div>`;
}

async function loadHoldMatrix() {
  const body = $("btMatrixBody");
  const meta = $("btMatrixMeta");
  if (!body) return;
  let data;
  try {
    data = await fetchJSON("/api/backtest/hold-matrix", { silent: true });
  } catch (_) {
    body.innerHTML = '<div class="metric-empty">加载失败（后端未重启？）</div>';
    return;
  }
  if (!data.available || !data.matrix) {
    body.innerHTML = '<div class="metric-empty">尚无矩阵结果 — 运行 scripts/hold_matrix.py 生成 results/hold_matrix_latest.json</div>';
    return;
  }
  const m = data.matrix;
  const policies = m.policies || [];
  const cells = m.cells || {};
  const base = m.baseline_signal || {};
  const gen = (m.generated_at || "").replace("T", " ").slice(0, 19);
  const rankRowsAll = m.ranking || [];
  const winTxt = (() => {
    const blocks = m.window_blocks || [];
    if (m.window_mode === "spread" && blocks.length) {
      const core = blocks.filter((b) => b.kind !== "warmup") || blocks;
      const rngs = core.map((b) => `bar ${b.start}..${b.end}`).join(" | ");
      return `分层抽样（${m.regime || "vol"} · ${core.length} 块 · ${m.window_bars ?? ""} 根）：${rngs}`;
    }
    if (m.window_bars) {
      let t = `样本外：仅最后 ${m.window_bars} 根（原始序列 bar ${m.window_start ?? 0}..${(m.window_start ?? 0) + (m.bars ?? 0) - 1}`;
      const oos = m.oos;
      if (oos && oos.status !== "unavailable") {
        const S = oos.status;
        const ni = oos.n_in_sample || 0, nh = oos.n_holdout || 0, np2 = oos.n_post_train || 0;
        if (S === "oos-holdout") t += `，训练从未触碰的 holdout 尾部 ✓）`;
        else if (S === "oos-new") t += `，训练截止后的全新数据 ✓）`;
        else if (S === "in-sample") t += `，⚠ 全部 ${ni} 根在训练集内）`;
        else if (S === "partial") t += `，⚠ 部分样本内：${ni} 根在训练集内 · 仅尾 ${nh} 根为真 holdout）`;
        else t += `，⚠ 部分样本内 + 新数据：训练内 ${ni} / holdout ${nh} / 训练后新数据 ${np2}）`;
        const h = oos.honest;
        if (h) t += ` · 诚实 OOS 建议：bar ${h.start_bar}..${h.end_bar}（${h.n_bars} 根${h.kind === "post-train" ? ", 训练后新数据" : ", 真 holdout"}）`;
      } else {
        t += `，无法判定（缺训练溯源））`;
      }
      return t;
    }
    return "全部历史";
  })();
  // 帕累托/关注（服务端算好写入结果文件；旧文件前端兜底）
  const paretoIds = matrixParetoFallback(rankRowsAll);
  const focusRows = (m.focus_list || []).filter((r) => paretoIds.has(r.combo));
  const focusIds = new Set(focusRows.map((r) => r.combo));
  //（窗口描述已在上方 winTxt 计算）
  if (meta) {
    const f = focusRows[0];
    const focusTxt = f
      ? focusRows.map((r) => `★ ${matrixComboCn(r.combo)}（夏普 ${matrixFmt("sharpe", r.sharpe)} · 回撤 ${matrixFmt("max_drawdown", r.max_drawdown)}）`).join("；")
      : "（无组合同时在帕累托前沿且全面优于基线）";
    meta.innerHTML =
      `基线 信号跟随：收益 ${matrixFmt("total_return", base.total_return)} · 夏普 ${matrixFmt("sharpe", base.sharpe)} · ` +
      `最大回撤 ${matrixFmt("max_drawdown", base.max_drawdown)} · 盈亏比 ${matrixFmt("profit_loss_ratio", base.profit_loss_ratio)} · ` +
      `交易 ${base.n_trades ?? "—"} · 胜率 ${matrixFmt("win_rate", base.win_rate)}<br>` +
      `<small>窗口：${winTxt} · 数据：${m.data_file || ""}（${m.bars ?? ""} 根）· 因子 ${m.strategy_file || ""} · ` +
      `成本 手续费 ${m.commission_pct ?? "—"}% / 滑点 ${m.slippage_pct ?? "—"}% · 生成 ${gen || "—"}</small>` +
      `<div class="mx-focus-strip">🎯 值得实盘关注（帕累托 ∩ 优于基线 · 共 ${focusRows.length} 个）：${focusTxt}</div>` +
      `<div class="mx-legend">帕累托=多目标非劣（收益/夏普/回撤/盈亏比四维）；★ 关注=帕累托且在 夏普/最大回撤/盈亏比 上均不劣于基线且至少一项更优</div>`;
  }
  const tables = MATRIX_TABLES.map((cfg) => matrixTableHtml(policies, cells, cfg)).join("");
  const rankRows = rankRowsAll
    .map((r, i) => {
      const comboCls = Number(r.sharpe) >= 0 ? "pos" : "neg";
      const isPareto = paretoIds.has(r.combo);
      const isFocus = focusIds.has(r.combo);
      const badges = isFocus
        ? '<span class="mx-badge focus" title="帕累托前沿 且 夏普/回撤/盈亏比均不劣于基线（至少一项更优）">★ 关注</span>'
        : isPareto
          ? '<span class="mx-badge pareto" title="多目标非劣：未被任何组合在 收益/夏普/回撤/盈亏比 上全面超越">帕累托</span>'
          : "";
      return `<tr data-combo="${escHtml(r.combo)}" class="matrix-rank-row ${isFocus ? "mx-row-focus" : isPareto ? "mx-row-pareto" : ""} ${window.__mxCurveSel === r.combo ? "mx-curve-sel" : ""}">
        <td>${i + 1}</td>
        <td class="sym-cell">${r.combo}</td>
        <td>${matrixComboCn(r.combo)}</td>
        <td class="${matrixNumCls(r.total_return, false)}">${matrixFmt("total_return", r.total_return)}</td>
        <td class="${comboCls}">${matrixFmt("sharpe", r.sharpe)}</td>
        <td>${matrixFmt("sortino", r.sortino)}</td>
        <td class="${matrixNumCls(r.max_drawdown, true)}">${matrixFmt("max_drawdown", r.max_drawdown)}</td>
        <td>${matrixFmt("profit_loss_ratio", r.profit_loss_ratio)}</td>
        <td>${r.n_trades ?? "—"}</td>
        <td>${matrixFmt("win_rate", r.win_rate)}</td>
        <td title="${escHtml(matrixExitBrief(r.exit_reasons))}">${escHtml(matrixExitBrief(r.exit_reasons))}</td>
        <td>${badges}</td>
      </tr>`;
    })
    .join("");
  body.innerHTML =
    tables +
    `<h4 class="bt-matrix-sub">按夏普排名 · 帕累托标注（${rankRowsAll.length} 个组合，共 ${paretoIds.size} 个帕累托 / ${focusRows.length} 个关注）</h4>` +
    `<div class="table-wrap bt-matrix-wrap"><table class="bt-table matrix-table"><thead><tr>` +
    `<th>#</th><th>组合</th><th>名称</th><th>收益</th><th>夏普</th><th>索提诺</th><th>最大回撤</th><th>盈亏比</th><th>交易</th><th>胜率</th><th>出场构成</th><th>多目标</th>` +
    `</tr></thead><tbody>${rankRows}</tbody></table></div>` +
    mxHeatSummaryHtml(m) +
    mxCostHtml(data.cost_sensitivity) +
    `<div class="mx-tip">💡 点击排名表中任意一行（含 信号跟随 基线）高亮该组合；在下方叠加对比中多选 ≥2 个组合查看资金曲线与滚动夏普</div>`;
  bindMatrixCurveRows();
  populateMxOverlay();
  const mxLive = $("btMatrixCurveLive");
  if (mxLive && rankRowsAll.length) mxLive.hidden = false;
}

// ═══════════════════════════════════════════════════════════════════
// 矩阵组合 → 叠加对比（多选 2-3 个组合，读 npz 侧车）
// ═══════════════════════════════════════════════════════════════════
let mxOverlayChart = null;
let mxOverlayRollChart = null;
const __mxCurveCache = {};

// 叠加对比：多选 2-3 个组合，同坐标画多条资金曲线
function populateMxOverlay() {
  const sel = $("btMatrixOverlaySel");
  const bar = $("btMatrixOverlayBar");
  if (!sel || !bar) return;
  const rows = [...document.querySelectorAll("#btMatrixBody tr[data-combo]")];
  bar.hidden = rows.length === 0;
  const prev = new Set([...sel.selectedOptions].map((o) => o.value));
  sel.innerHTML = rows
    .map((r) => {
      const cap = r.dataset.cap || "";
      const thr = r.dataset.thr || "";
      const key = r.dataset.combo + "@" + cap + "@" + thr;   // 叠加缓存/拉取按 组合@上限@阈值 区分
      const extra = cap ? ` · 上限 ${cap}% · t=${thr}` : "";
      return `<option value="${escHtml(key)}">${escHtml(comboDisplayName(r.dataset.combo))}${escHtml(extra)} · ${escHtml(r.dataset.combo)}</option>`;
    })
    .join("");
  [...sel.options].forEach((o) => {
    if (prev.has(o.value)) o.selected = true;
  });
  if (!sel.__mxOvBound) {
    sel.__mxOvBound = true;
    sel.addEventListener("change", mxOverlayRender);
    const clr = $("btMatrixOverlayClear");
    if (clr) clr.addEventListener("click", () => { sel.selectedIndex = -1; mxOverlayRender(); });
  }
  if (prev.size >= 2) mxOverlayRender();
}

async function mxOverlayRender() {
  const sel = $("btMatrixOverlaySel");
  const block = $("btMatrixOverlayBlock");
  const hint = $("btMatrixOverlayHint");
  const picks = [...(sel?.selectedOptions || [])].map((o) => o.value);
  if (!block) return;
  if (picks.length < 2) {
    block.hidden = true;
    if (hint) hint.textContent = picks.length === 1 ? "再选 1-2 个组合即可叠加对比" : "选中 ≥2 个组合叠加对比资金曲线与滚动夏普";
    if (mxOverlayChart) {
      mxOverlayChart.destroy();
      mxOverlayChart = null;
    }
    if (mxOverlayRollChart) {
      mxOverlayRollChart.destroy();
      mxOverlayRollChart = null;
    }
    return;
  }
  block.hidden = false;
  if (hint) hint.textContent = `正在加载 ${picks.length} 条曲线…`;
  try {
    const datas = [];
    for (const c of picks) {
      if (!__mxCurveCache[c]) {
        const [combo, cap, thr] = c.split("@");
        const params = new URLSearchParams({ combo });
        if (cap) params.set("cap_pct", cap);
        if (thr) params.set("threshold", thr);
        const d = await fetchJSON("/api/backtest/combo-sweep/curve?" + params.toString(), {
          silent: true,
          retries: 0,
        });
        if (!d.available) throw new Error(d.error || c);
        __mxCurveCache[c] = d;
      }
      datas.push(__mxCurveCache[c]);
    }
    const palette = ["#56a4f0", "#5eead4", "#f0c25e", "#f07056", "#c084fc", "#7cc576"];
    const labels = datas[0].labels;
    const mkSet = (d, i, key, yOf) => ({
      label: comboDisplayName(d.combo),
      data: (d[key] || []).map((v) => (v == null ? null : yOf(v))),
      borderColor: palette[i % palette.length],
      backgroundColor: palette[i % palette.length] + "18",
      borderWidth: 1.6,
      pointRadius: 0,
      tension: 0.15,
      fill: false,
    });
    const sets = datas.map((d, i) => mkSet(d, i, "equity", (v) => (v - 1) * 100));
    const cv = $("btMatrixOverlayChart");
    if (mxOverlayChart) {
      mxOverlayChart.destroy();
      mxOverlayChart = null;
    }
    mxOverlayChart = new Chart(cv.getContext("2d"), {
      type: "line",
      data: { labels, datasets: sets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { position: "bottom", labels: { color: "#c8d3e0", boxWidth: 14, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: (c) =>
                c.parsed.y == null
                  ? `${c.dataset.label}: —`
                  : `${c.dataset.label}: ${(c.parsed.y >= 0 ? "+" : "") + Number(c.parsed.y).toFixed(2) + "%"}`,
            },
          },
        },
        scales: {
          x: { ticks: { display: false }, grid: { color: "rgba(128,140,160,0.08)" } },
          y: { ticks: { callback: (v) => Number(v).toFixed(0) + "%" }, grid: { color: "rgba(128,140,160,0.08)" } },
        },
      },
    });
    // ── 滚动夏普叠加（第二张图，同一调色板，NaN 断点） ──
    const rollCv = $("btMatrixOverlayRollChart");
    const rollLbl = $("btMatrixOverlayRollLabel");
    if (rollCv) {
      if (mxOverlayRollChart) {
        mxOverlayRollChart.destroy();
        mxOverlayRollChart = null;
      }
      if (rollLbl) rollLbl.hidden = false;
      const rollSets = datas.map((d, i) => ({
        ...mkSet(d, i, "rolling_sharpe", (v) => Number(v)),
        spanGaps: false,
      }));
      mxOverlayRollChart = new Chart(rollCv.getContext("2d"), {
        type: "line",
        data: { labels, datasets: rollSets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          plugins: {
            legend: { position: "bottom", labels: { color: "#c8d3e0", boxWidth: 14, font: { size: 11 } } },
            tooltip: {
              callbacks: {
                label: (c) =>
                  c.parsed.y == null || Number.isNaN(c.parsed.y)
                    ? `${c.dataset.label}: —`
                    : `${c.dataset.label}: ${Number(c.parsed.y).toFixed(2)}`,
              },
            },
          },
          scales: {
            x: { ticks: { display: false }, grid: { color: "rgba(128,140,160,0.08)" } },
            y: { ticks: { callback: (v) => Number(v).toFixed(1) }, grid: { color: "rgba(128,140,160,0.08)" } },
          },
        },
      });
    }
    if (hint) hint.textContent = `叠加中：${picks.map((c) => comboDisplayName(c.split("@")[0])).join(" vs ")}（资金曲线 + 滚动夏普）`;
  } catch (err) {
    if (hint) hint.textContent = "叠加加载失败：" + err.message;
  }
}

function bindMatrixCurveRows() {
  const body = $("btMatrixBody");
  if (!body || body.__mxBound) return;
  body.__mxBound = true;
  body.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-combo]");
    if (tr) selectMatrixCurve(tr);
  });
}

function selectMatrixCurve(tr) {
  const combo = tr.getAttribute("data-combo");
  window.__mxCurveSel = combo;
  document.querySelectorAll("#btMatrixBody tr[data-combo]").forEach((r) =>
    r.classList.toggle("mx-curve-sel", r.getAttribute("data-combo") === combo)
  );
  const stats = $("btMatrixCurveStats");
  if (stats)
    stats.innerHTML =
      `组合 <b>${escHtml(comboDisplayName(combo))}</b>（<code>${escHtml(combo)}</code>）· ` +
      `在下方叠加对比中多选 ≥2 个组合查看资金曲线与滚动夏普`;
}

// ═══════════════════════════════════════════════════════════════════
// N×N 全组合回测（一键跑全部，非手选单格）
// ═══════════════════════════════════════════════════════════════════
let matrixPollTimer = null;
let __mxPrefsTimer = null;
let __mxApplyingPrefs = false;
let __mxPrefsSymbol = null;      // 当前已应用的按品种记忆
let __mxLastRun = null;          // 最近一次启动参数（供轮询提示/记忆）

function setMatrixRunHint(txt, busy) {
  const el = $("btMatrixRunHint");
  if (!el) return;
  el.textContent = txt;
  el.classList.toggle("busy", !!busy);
}

// 窗口模式 UI：spread 时显示 块数/regime，改窗口输入文案
function matrixModeUi() {
  const mode = $("btMatrixModeSelect")?.value || "tail";
  const spread = mode === "spread";
  ["btMatrixChunkLabel", "btMatrixRegimeLabel"].forEach((id) => {
    const el = $(id);
    if (el) el.hidden = !spread;
  });
  const wl = $("btMatrixWinLabel");
  if (wl) {
    const span = wl.firstChild;
    if (span && span.nodeType === 3) span.textContent = spread ? "抽样总量（根）" : "窗口（根）";
  }
}

function matrixPrefsSymbol() {
  return selectedStrategySymbol || window.__lastMatrixSymbol || null;
}

function currentMatrixPrefs() {
  return {
    data_file: $("btMatrixDataSelect")?.value || null,
    window_bars: (() => {
      const v = Number($("btMatrixWindowInput")?.value);
      return Number.isFinite(v) && v > 0 ? Math.round(v) : null;
    })(),
    window_mode: $("btMatrixModeSelect")?.value || "tail",
    regime: $("btMatrixRegimeSelect")?.value || "vol",
    chunks: Math.max(2, Math.round(Number($("btMatrixChunksInput")?.value) || 4)),
  };
}

// 保存当前品种的矩阵设置（防抖；由 run 前与控件变更触发）
function matrixPrefsSave() {
  const sym = matrixPrefsSymbol();
  if (!sym) return;
  window.__lastMatrixSymbol = sym;
  if (__mxApplyingPrefs) return;
  if (__mxPrefsTimer) clearTimeout(__mxPrefsTimer);
  __mxPrefsTimer = setTimeout(async () => {
    try {
      await fetchJSON("/api/backtest/hold-matrix/prefs", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: sym, ...currentMatrixPrefs() }),
        silent: true,
      });
    } catch (_) {}
  }, 500);
}

// 矩阵工具栏品种徽章：与策略文件联动，标明矩阵记忆按哪一品种生效（避免误以为记忆失效）
function matrixSymBadgeUpdate() {
  const badge = $("btMatrixSymBadge");
  if (!badge) return;
  const sym = selectedStrategySymbol;
  const file = selectedStrategyFile || "";
  const fname = String(file).split("/").pop() || "";
  if (!sym || !file) {
    badge.hidden = true;
    return;
  }
  const memTxt = __mxPrefsSymbol === sym ? " · 已带出该品种矩阵记忆" : " · 尚无该品种记忆（默认设置）";
  badge.hidden = false;
  badge.textContent = `🎯 品种 ${sym} · ${fname}${memTxt}`;
}

// 切换品种/回测页打开时：带出该品种上次的矩阵设置
async function matrixPrefsApply(symbol) {
  if (!symbol || symbol === __mxPrefsSymbol) { matrixSymBadgeUpdate(); return; }
  let d;
  try {
    d = await fetchJSON(
      "/api/backtest/hold-matrix/prefs?symbol=" + encodeURIComponent(symbol),
      { silent: true, retries: 0 }
    );
  } catch (_) {
    return;
  }
  const p = (d && d.prefs) || {};
  __mxApplyingPrefs = true;
  try {
    if (p.window_mode) {
      const modeSel = $("btMatrixModeSelect");
      if (modeSel) modeSel.value = p.window_mode;
      const winEl = $("btMatrixWindowInput");
      if (winEl && p.window_bars) winEl.value = String(p.window_bars);
      const ck = $("btMatrixChunksInput");
      if (ck && p.chunks) ck.value = String(p.chunks);
      const rg = $("btMatrixRegimeSelect");
      if (rg && p.regime) rg.value = p.regime;
      const ds = $("btMatrixDataSelect");
      window.__mxPrefsDataFile = p.data_file || null;
      if (ds && p.data_file && [...ds.options].some((o) => o.value === p.data_file)) {
        ds.value = p.data_file;
      } else if (ds && p.data_file && !ds.value) {
        // 列表尚未加载或文件已被移动——等 btLoadMatrixDataFiles 完成后重试
      }
      matrixModeUi();
      const winTxt = p.window_mode === "spread"
        ? `分层抽样（${p.regime || "vol"} · ${p.chunks || 4} 块 · ${p.window_bars || "?"} 根）`
        : p.window_bars ? `样本外尾部 ${p.window_bars} 根` : "全部历史";
      setMatrixRunHint(`已按 ${symbol} 上次设置恢复：${winTxt}${p.data_file ? " · 数据 " + String(p.data_file).split("/").pop() : " · 跟随策略"}（可修改后重跑）`, false);
      __mxPrefsSymbol = symbol;
    } else {
      // 该品种无记忆 → 恢复默认（跟随策略 / 全部历史 / 尾部模式）
      const ds = $("btMatrixDataSelect");
      if (ds) ds.value = "";
      const wi = $("btMatrixWindowInput");
      if (wi) wi.value = "";
      const ms = $("btMatrixModeSelect");
      if (ms) ms.value = "tail";
      matrixModeUi();
      __mxPrefsSymbol = symbol;
    }
  } finally {
    __mxApplyingPrefs = false;
    matrixSymBadgeUpdate();
  }
}

async function btLoadMatrixDataFiles() {
  const sel = $("btMatrixDataSelect");
  if (!sel) return;
  let files = [];
  try {
    const d = await fetchJSON("/api/data/files", { silent: true, retries: 0 });
    files = d.files || [];
  } catch (_) {}
  const cur = sel.value;
  const group = (name, rows) =>
    rows.length
      ? `<optgroup label="${escHtml(name)}">` +
        rows
          .map(
            (f) =>
              `<option value="${escHtml(f.data_file)}">${escHtml(f.symbol || f.rel)}${f.timeframe ? " · " + escHtml(f.timeframe) : ""} — ${escHtml(f.rel.split("/").pop())}（${f.bars ?? "?"} 根）</option>`
          )
          .join("") +
        "</optgroup>"
      : "";
  const slices = files.filter((f) => f.rel.startsWith("data/slices"));
  const training = files.filter((f) => f.rel.startsWith("data/training"));
  sel.innerHTML =
    '<option value="">跟随策略</option>' +
    group("data/slices（切片/样本外）", slices) +
    group("data/training（原始下载）", training);
  if (cur && files.some((f) => f.data_file === cur)) sel.value = cur;
  // 按品种记忆的文件（若应用 prefs 时列表尚未就绪，现在补选）
  const mem = window.__mxPrefsDataFile;
  if (!sel.value && mem && files.some((f) => f.data_file === mem)) sel.value = mem;
}

// 三轴联合回测的 上限%/阈值 档位输入 → 归一化数组（页面输入，非法值回退默认）
function comboAxesInputs() {
  const capsRaw = ($("btMatrixCapsInput")?.value || "").trim();
  const caps = capsRaw
    ? capsRaw.split(/[,，\s]+/)
        .map((s) => Number(s))
        .filter((n) => Number.isFinite(n) && n > 0 && n <= 200)
    : [];
  const thrRaw = ($("btMatrixThrInput")?.value || "").trim();
  const thr = thrRaw
    ? thrRaw.split(/[,，\s]+/)
        .map((s) => Number(s))
        .filter((n) => Number.isFinite(n) && n > 0 && n < 1)
    : [];
  return {
    caps: caps.length ? [...new Set(caps)].sort((a, b) => a - b) : [10, 25, 100],
    thresholds: thr.length ? [...new Set(thr)].sort((a, b) => a - b) : [0.05, 0.3, 0.5, 0.8],
  };
}

async function runHoldMatrix() {
  const btn = $("btMatrixRunBtn");
  const stop = $("btMatrixStopBtn");
  const winEl = $("btMatrixWindowInput");
  const mode = $("btMatrixModeSelect")?.value || "tail";
  const wv = Number(winEl?.value);
  let windowBars = null;
  if (Number.isFinite(wv) && wv > 0) {
    if (wv < 800) {
      setMatrixRunHint("窗口需 ≥800 根（特征 warm-up），请修改后重试", false);
      return;
    }
    windowBars = Math.round(wv);
  }
  if (mode === "spread" && !windowBars) {
    setMatrixRunHint("分层抽样需填写抽样总量（≥800 根）", false);
    return;
  }
  const chunks = Math.max(2, Math.round(Number($("btMatrixChunksInput")?.value) || 4));
  const regime = $("btMatrixRegimeSelect")?.value || "vol";
  const dataFile = $("btMatrixDataSelect")?.value || null;
  const axes = comboAxesInputs();
  window.__lastMatrixSymbol = matrixPrefsSymbol();
  matrixPrefsSave(); // 记住本次设置（按品种）
  if (btn) btn.disabled = true;
  if (winEl) winEl.disabled = true;
  const modeTxt =
    mode === "spread"
      ? `分层抽样（${regime} · ${chunks} 块 · 共 ${windowBars} 根）`
      : windowBars
        ? `样本外：仅尾部 ${windowBars} 根`
        : "全部历史";
  const total = 22 * axes.caps.length * axes.thresholds.length;
  __mxLastRun = { mode, windowBars, regime, chunks, dataFile, axes, total };
  setMatrixRunHint(`启动三轴联合回测（${modeTxt} · 22 组合 × ${axes.caps.length} 档上限 × ${axes.thresholds.length} 档阈值 = ${total} 次回放）…`, true);
  try {
    const res = await fetchJSON("/api/backtest/combo-sweep/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        window_bars: windowBars,
        window_mode: mode,
        regime,
        chunks,
        data_file: dataFile,
        caps: axes.caps,
        thresholds: axes.thresholds,
      }),
    });
    if (!res.ok) throw new Error(res.detail || "启动失败");
  } catch (e) {
    setMatrixRunHint(`启动失败：${e.message}`, false);
    if (btn) btn.disabled = false;
    if (winEl) winEl.disabled = false;
    return;
  }
  if (stop) stop.hidden = false;
  pollHoldMatrix();
}

async function stopHoldMatrix() {
  try {
    await fetchJSON("/api/backtest/combo-sweep/stop", { method: "POST", silent: true });
  } catch (_) {}
  setMatrixRunHint("正在停止…", true);
  pollHoldMatrix();
}

function pollHoldMatrix() {
  if (matrixPollTimer) clearInterval(matrixPollTimer);
  matrixPollTimer = setInterval(async () => {
    let st;
    try {
      st = await fetchJSON("/api/backtest/combo-sweep/status", { silent: true });
    } catch (_) {
      return;
    }
    const job = st.job || {};
    const state = job.state || "idle";
    if (state === "running") {
      const lr = __mxLastRun || {};
      const winTxt = lr.mode === "spread"
        ? ` · 分层抽样 ${lr.regime || "vol"}·${lr.chunks || 4} 块`
        : lr.windowBars ? ` · 样本外尾部 ${lr.windowBars} 根` : "";
      const tot = st.combos_total || lr.total || 22;
      const capsN = (lr.axes && lr.axes.caps.length) || 1;
      const thrN = (lr.axes && lr.axes.thresholds.length) || 1;
      const done = st.combos_done || 0;
      const pct = tot ? Math.round((done / tot) * 100) : 0;
      setMatrixRunHint(`运行中：已完成 ${done}/${tot} 次回放（${pct}%）${winTxt}（组合 × ${capsN} 档上限 × ${thrN} 档阈值）…`, true);
      return;
    }
    clearInterval(matrixPollTimer);
    matrixPollTimer = null;
    const btn = $("btMatrixRunBtn");
    const stopBtn = $("btMatrixStopBtn");
    const winEl = $("btMatrixWindowInput");
    if (btn) btn.disabled = false;
    if (stopBtn) stopBtn.hidden = true;
    if (winEl) winEl.disabled = false;
    if (state === "completed") {
      const lr = __mxLastRun || {};
      const winTxt = lr.mode === "spread"
        ? `（分层抽样 ${lr.regime || "vol"}·${lr.chunks || 4} 块）`
        : lr.windowBars ? `（样本外：仅尾部 ${lr.windowBars} 根）` : "（全部历史）";
      const axes = lr.axes || comboAxesInputs();
      setMatrixRunHint(`三轴联合回测完成 ${winTxt}：下方 全网格排名 + 每档最优 已刷新，矩阵视图默认停在基线切片（上限 ${axes.caps[0]}% × t=${axes.thresholds[0]}）`, false);
      const det = $("btMatrixDetails");
      if (det && !det.open) {
        const sum = det.querySelector("summary");
        if (sum) sum.click();
      }
      await loadComboGrid();
      await loadHoldMatrix();
    } else if (state === "failed") {
      setMatrixRunHint(`三轴联合回测失败：${job.error || "exit " + job.exit_code}（见日志 ${job.log_path || ""}）`, false);
    } else {
      setMatrixRunHint("已停止（上次结果保留）", false);
    }
  }, 2500);
}

// ═══════════════════════════════════════════════════════════════════
// 三轴联合回测结果：全局最优 + 每档阈值/每档上限最优 + 全网格排名表 + 切片矩阵
// ═══════════════════════════════════════════════════════════════════
let __comboGrid = null;        // 最近一次 combo_sweep_latest.json
let __gridSort = { key: "sharpe", desc: true };

const GRID_TABLE_KEYS = [
  { key: "combo", label: "组合", pct: false, sortable: true },
  { key: "cap_pct", label: "上限%", pct: false, sortable: true },
  { key: "threshold", label: "阈值", pct: false, sortable: true },
  { key: "flat_share", label: "观望%", pct: true, sortable: true },
  { key: "total_return", label: "收益", pct: true, sortable: true },
  { key: "sharpe", label: "夏普", pct: false, sortable: true },
  { key: "max_drawdown", label: "最大回撤", pct: true, sortable: true },
  { key: "profit_loss_ratio", label: "盈亏比", pct: false, sortable: true },
  { key: "n_trades", label: "交易数", pct: false, sortable: true },
  { key: "win_rate", label: "胜率", pct: true, sortable: true },
];

function gridFmt(key, v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (key === "total_return") return (n >= 0 ? "+" : "") + (n * 100).toFixed(2) + "%";
  if (key === "max_drawdown") return (n * 100).toFixed(2) + "%";
  if (key === "win_rate" || key === "flat_share") return (n * 100).toFixed(0) + "%";
  if (key === "cap_pct") return String(Math.round(n)) + "%";
  if (key === "threshold") return "t=" + Number(n).toFixed(2);
  if (key === "n_trades") return String(Math.round(n));
  return (n >= 0 ? "+" : "") + Number(n).toFixed(2);
}

function gridNumCls(key, v) {
  const n = Number(v);
  if (!Number.isFinite(n) || n === 0) return "";
  const good = key === "max_drawdown" ? n < 0 : n > 0;
  return good ? "pos" : "neg";
}

function gridBestBrief(b) {
  if (!b || !b.combo) return "—";
  const dd = b.max_drawdown != null ? `回撤 ${(Number(b.max_drawdown) * 100).toFixed(1)}%` : "";
  const sh = b.sharpe != null ? `夏普 ${Number(b.sharpe).toFixed(2)}` : "夏普 —";
  return `${matrixComboCn(b.combo)}（${b.combo}）· 上限 ${Math.round(b.cap_pct)}% · t=${Number(b.threshold).toFixed(2)} · 收益 ${gridFmt("total_return", b.total_return)} · ${sh} · ${dd} · 交易 ${b.n_trades ?? "—"}`;
}

function sliceOf(r) {
  return `${Number(r.cap_pct).toFixed(4)}@${Number(r.threshold).toFixed(4)}`;
}

async function loadComboGrid() {
  const box = $("btComboGrid");
  const row = $("btMatrixSliceRow");
  if (!box) return;
  let data;
  try {
    data = await fetchJSON("/api/backtest/combo-sweep", { silent: true });
  } catch (_) {
    box.innerHTML = '<div class="metric-empty">三轴结果加载失败（后端未重启？）</div>';
    return;
  }
  if (!data.available || !data.grid || !(data.grid.rows || []).length) {
    box.innerHTML = '<div class="metric-empty">尚无三轴联合结果 — 点上方「一键三轴联合回测」生成 results/combo_sweep_latest.json</div>';
    if (row) row.hidden = true;
    return;
  }
  __comboGrid = data.grid;
  renderComboGrid();
  populateSliceSelects();
}

function gridTableHead() {
  const sortMark = (k) => (__gridSort.key === k ? (__gridSort.desc ? " ▼" : " ▲") : "");
  return GRID_TABLE_KEYS.map((c) =>
    c.sortable
      ? `<th data-sort="${c.key}" title="点击排序">${escHtml(c.label)}${sortMark(c.key)}</th>`
      : `<th>${escHtml(c.label)}</th>`
  ).join("");
}

function renderComboGrid() {
  const box = $("btComboGrid");
  if (!box || !__comboGrid) return;
  const g = __comboGrid;
  const rows = g.rows || [];
  const capsN = (g.caps || []).length;
  const thrN = (g.thresholds || []).length;
  const gen = (g.generated_at || "").replace("T", " ").slice(0, 19);
  // 每行标注：全局最高夏普 ★ · 该组合内最优（同组合跨 上限×阈值）✦ · 该 (组合×上限) 内最优阈值 ▲
  const bestOverall = rows.reduce((m, r) => (r.sharpe > (m?.sharpe ?? -1e18) ? r : m), null);
  const bestByCombo = {};
  rows.forEach((r) => {
    const cur = bestByCombo[r.combo];
    if (!cur || (r.sharpe ?? -1e18) > (cur.sharpe ?? -1e18)) bestByCombo[r.combo] = r;
  });
  const bestByComboCap = {};
  rows.forEach((r) => {
    const k = `${r.combo}@${r.cap_pct}`;
    const cur = bestByComboCap[k];
    if (!cur || (r.sharpe ?? -1e18) > (cur.sharpe ?? -1e18)) bestByComboCap[k] = r;
  });
  const sorted = [...rows].sort((a, b) => {
    const k = __gridSort.key;
    if (k === "combo") {
      return __gridSort.desc
        ? String(b.combo).localeCompare(String(a.combo))
        : String(a.combo).localeCompare(String(b.combo));
    }
    const va = Number(a[k]);
    const vb = Number(b[k]);
    if (!Number.isFinite(va) && !Number.isFinite(vb)) return 0;
    if (!Number.isFinite(va)) return 1;
    if (!Number.isFinite(vb)) return -1;
    return __gridSort.desc ? vb - va : va - vb;
  });
  const trHtml = sorted
    .map((r) => {
      const isOverall = bestOverall && r === bestOverall && r.sharpe > 0;
      const isBestCombo = bestByCombo[r.combo] === r && r.sharpe > 0;
      const isBestComboCap = bestByComboCap[`${r.combo}@${r.cap_pct}`] === r && r.sharpe > 0;
      const cls = (isOverall ? " ab-best" : "") + (isBestCombo ? " ab-cur" : "");
      const mark = (isOverall ? " ★" : "") + (isBestCombo ? " ✦" : "") + (isBestComboCap ? " ▲" : "");
      const tds = GRID_TABLE_KEYS.map((c) => {
        if (c.key === "combo") {
          return `<td class="ab-name" title="${escHtml(r.combo_label || "")}">${escHtml(matrixComboCn(r.combo))}${mark}<span class="hint">${escHtml(r.combo)}</span></td>`;
        }
        if (c.key === "cap_pct" || c.key === "threshold") return `<td>${gridFmt(c.key, r[c.key])}</td>`;
        return `<td class="${gridNumCls(c.key, r[c.key])}">${gridFmt(c.key, r[c.key])}</td>`;
      }).join("");
      return `<tr class="${cls.trim()}" title="${isOverall ? "全局最高夏普" : isBestCombo ? "该组合跨 上限×阈值 最优" : isBestComboCap ? "该组合×该上限下最优阈值" : ""}">${tds}</tr>`;
    })
    .join("");
  const bestT = (g.best_per_threshold || []).map((it) => {
    const b = it.best;
    return `<div class="ab-bestline">🎚 阈值 t=${Number(it.threshold).toFixed(2)}：${b ? escHtml(gridBestBrief(b)) : "无有效交易"}</div>`;
  }).join("");
  const bestC = (g.best_per_cap || []).map((it) => {
    const b = it.best;
    return `<div class="ab-bestline">💰 上限 ${Math.round(it.cap_pct)}%：${b ? escHtml(gridBestBrief(b)) : "无有效交易"}</div>`;
  }).join("");
  const over = g.best_overall
    ? `<div class="ab-bestline">🎯 全局最优：${escHtml(gridBestBrief(g.best_overall))}</div>`
    : '<div class="ab-bestline">🎯 全局最优：无有效交易（该窗口/成本下无信号可成交）</div>';
  box.innerHTML =
    over +
    bestT +
    bestC +
    `<h4 class="bt-matrix-sub">全网格排名（${rows.length} 行 = 22 组合 × ${capsN} 档上限 × ${thrN} 档阈值 · 点表头排序 · ★全局最优 ✦组合内最优 ▲该组合×该上限下最优阈值）</h4>` +
    `<div class="table-wrap bt-matrix-wrap"><table class="bt-table matrix-table grid-table"><thead><tr>${gridTableHead()}</tr></thead><tbody>${trHtml}</tbody></table></div>` +
    `<small class="ab-note">口径：离散撮合（与模拟实盘同引擎）· 同一因子一次计算 · 窗口 ${g.window_mode === "spread" ? "分层抽样" : g.window_bars ? `样本外尾部 ${g.window_bars} 根` : "全部历史"} · 成本 手续费 ${g.commission_pct ?? "—"}% / 滑点 ${g.slippage_pct ?? "—"}% · 生成 ${gen}</small>`;
  box.querySelectorAll("th[data-sort]").forEach((th) => {
    if (th.__gridBound) return;
    th.__gridBound = true;
    th.addEventListener("click", () => {
      const k = th.dataset.sort;
      if (__gridSort.key === k) __gridSort.desc = !__gridSort.desc;
      else __gridSort = { key: k, desc: k === "combo" ? false : true };
      renderComboGrid();
    });
  });
}

// 切片选择器：上限% × 阈值 → 切换下方矩阵视图（基线切片走服务端完整渲染，其余前端切片）
function populateSliceSelects() {
  const capSel = $("btSliceCapSelect");
  const thrSel = $("btSliceThrSelect");
  const row = $("btMatrixSliceRow");
  if (!capSel || !thrSel || !__comboGrid) return;
  const caps = [...new Set((__comboGrid.rows || []).map((r) => Number(r.cap_pct)))].sort((a, b) => a - b);
  const thr = [...new Set((__comboGrid.rows || []).map((r) => Number(r.threshold)))].sort((a, b) => a - b);
  const prevCap = capSel.value, prevThr = thrSel.value;
  capSel.innerHTML = caps.map((c) => `<option value="${c}">${Math.round(c)}%</option>`).join("");
  thrSel.innerHTML = thr.map((t) => `<option value="${t}">t=${Number(t).toFixed(2)}</option>`).join("");
  const base = __comboGrid.baseline_slice || {};
  capSel.value = caps.includes(Number(base.cap_pct)) ? String(base.cap_pct) : String(caps[0]);
  thrSel.value = thr.includes(Number(base.threshold)) ? String(base.threshold) : String(thr[0]);
  // 恢复用户上次手动选择的切片（同一份结果内）
  if (prevCap && caps.includes(Number(prevCap))) capSel.value = prevCap;
  if (prevThr && thr.includes(Number(prevThr))) thrSel.value = prevThr;
  row.hidden = false;
  if (!capSel.__sliceBound) {
    capSel.__sliceBound = true;
    capSel.addEventListener("change", renderSliceMatrix);
    thrSel.addEventListener("change", renderSliceMatrix);
  }
  renderSliceMatrix();
}

function renderSliceMatrix() {
  const capSel = $("btSliceCapSelect");
  const thrSel = $("btSliceThrSelect");
  const hint = $("btSliceHint");
  if (!capSel || !thrSel || !__comboGrid) return;
  const cap = Number(capSel.value), t = Number(thrSel.value);
  const base = __comboGrid.baseline_slice || {};
  const isBase = Math.abs(cap - Number(base.cap_pct)) < 1e-9 && Math.abs(t - Number(base.threshold)) < 1e-9;
  if (hint) {
    hint.textContent = isBase
      ? "基线切片：服务端完整渲染（含资金曲线/帕累托/热力），点排名行可画曲线"
      : "非基线切片：前端从全网格结果切片；点排名行按需重放资金曲线/滚动夏普（同引擎同窗口/成本，首次约 1-2 秒）";
  }
  if (isBase) {
    loadHoldMatrix();
    return;
  }
  // 客户端切片：cells / 排名 / 帕累托 / 优于基线
  const policies = __comboGrid.policies || [];
  const cells = {};
  const baseRow = {};
  (__comboGrid.rows || []).forEach((r) => {
    if (Math.abs(Number(r.cap_pct) - cap) > 1e-9 || Math.abs(Number(r.threshold) - t) > 1e-9) return;
    cells[r.combo] = { ...r };
    if (r.combo === "signal") Object.assign(baseRow, r);
  });
  const body = $("btMatrixBody");
  const curveLive = $("btMatrixCurveLive");
  if (!body) return;
  const rankRows = Object.keys(cells)
    .map((combo) => ({ combo, ...cells[combo] }))
    .sort((a, b) => (b.sharpe ?? -999) - (a.sharpe ?? -999));
  const paretoIds = matrixParetoFallback(rankRows);
  const beatsIds = new Set(rankRows.filter((r) => {
    if (!baseRow.sharpe || !r.sharpe) return false;
    const keys = ["sharpe", "max_drawdown", "profit_loss_ratio"];
    if (!keys.every((k) => Number.isFinite(Number(r[k])) && Number.isFinite(Number(baseRow[k])))) return false;
    return keys.every((k) => Number(r[k]) >= Number(baseRow[k])) && keys.some((k) => Number(r[k]) > Number(baseRow[k]));
  }).map((r) => r.combo));
  const focusIds = new Set([...paretoIds].filter((c) => beatsIds.has(c)));
  const tables = MATRIX_TABLES.map((cfg) => matrixTableHtml(policies, cells, cfg)).join("");
  const rankTr = rankRows.map((r, i) => {
    const isPareto = paretoIds.has(r.combo);
    const isFocus = focusIds.has(r.combo);
    const badges = isFocus
      ? '<span class="mx-badge focus">★ 关注</span>'
      : isPareto
        ? '<span class="mx-badge pareto">帕累托</span>'
        : "";
    return `<tr class="matrix-rank-row ${isFocus ? "mx-row-focus" : isPareto ? "mx-row-pareto" : ""}" data-combo="${r.combo}" data-cap="${cap}" data-thr="${t}">
      <td>${i + 1}</td><td class="sym-cell">${r.combo}</td><td>${matrixComboCn(r.combo)}</td>
      <td class="${matrixNumCls(r.total_return, false)}">${matrixFmt("total_return", r.total_return)}</td>
      <td class="${Number(r.sharpe) >= 0 ? "pos" : "neg"}">${matrixFmt("sharpe", r.sharpe)}</td>
      <td>${matrixFmt("sortino", r.sortino)}</td>
      <td class="${matrixNumCls(r.max_drawdown, true)}">${matrixFmt("max_drawdown", r.max_drawdown)}</td>
      <td>${matrixFmt("profit_loss_ratio", r.profit_loss_ratio)}</td>
      <td>${r.n_trades ?? "—"}</td><td>${matrixFmt("win_rate", r.win_rate)}</td>
      <td>${badges}</td></tr>`;
  }).join("");
  body.innerHTML =
    `<div class="mx-focus-strip">切片 上限 ${Math.round(cap)}% × t=${Number(t).toFixed(2)} · ${rankRows.length} 个组合 · 帕累托 ${paretoIds.size} 个 / 关注 ${focusIds.size} 个（基线 signal 收益 ${gridFmt("total_return", baseRow.total_return)} · 夏普 ${gridFmt("sharpe", baseRow.sharpe)}）</div>` +
    tables +
    `<h4 class="bt-matrix-sub">按夏普排名 · 帕累托标注（该切片）</h4>` +
    `<div class="table-wrap bt-matrix-wrap"><table class="bt-table matrix-table"><thead><tr>` +
    `<th>#</th><th>组合</th><th>名称</th><th>收益</th><th>夏普</th><th>索提诺</th><th>最大回撤</th><th>盈亏比</th><th>交易</th><th>胜率</th><th>多目标</th>` +
    `</tr></thead><tbody>${rankTr}</tbody></table></div>` +
    `<div class="mx-tip">💡 点排名行（含 信号跟随 基线）高亮该组合；在下方叠加对比中多选 ≥2 个组合查看资金曲线/滚动夏普。基线切片读回测侧车；非基线切片按需重放（同引擎同窗口/成本，首次约 1-2 秒，之后秒出）</div>`;
  if (curveLive && rankRows.length) curveLive.hidden = false;
  populateMxOverlay();  // 叠加选项按当前切片重建（combo@cap@thr）
}

// ═══════════════════════════════════════════════════════════════════
// 交互式资金曲线（HTML / Chart.js）
// ═══════════════════════════════════════════════════════════════════
let equityChart = null;
let rollingChart = null;
let btEquitySig = "";

const EQUITY_COLORS = [
  { hex: "#5eead4", rgb: "94, 234, 212" },
  { hex: "#38bdf8", rgb: "56, 189, 248" },
  { hex: "#818cf8", rgb: "129, 140, 248" },
  { hex: "#fbbf24", rgb: "251, 191, 36" },
  { hex: "#f472b6", rgb: "244, 114, 182" },
  { hex: "#a3e635", rgb: "163, 230, 53" },
];

function verticalGradient(chart, rgb, topAlpha, bottomAlpha) {
  const { ctx, chartArea } = chart;
  if (!chartArea) return `rgba(${rgb}, ${topAlpha})`;
  const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
  g.addColorStop(0, `rgba(${rgb}, ${topAlpha})`);
  g.addColorStop(0.62, `rgba(${rgb}, ${(topAlpha + bottomAlpha) / 4})`);
  g.addColorStop(1, `rgba(${rgb}, ${bottomAlpha})`);
  return g;
}

const EQUITY_TOOLTIP = {
  backgroundColor: "rgba(8, 12, 20, 0.94)",
  borderColor: "rgba(94, 234, 212, 0.35)",
  borderWidth: 1,
  titleColor: "#e8edf4",
  bodyColor: "#a9bccf",
  titleFont: { family: "'JetBrains Mono'", size: 11 },
  bodyFont: { family: "'JetBrains Mono'", size: 11 },
  padding: 10,
  cornerRadius: 8,
  usePointStyle: true,
};

const EQUITY_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 500, easing: "easeOutQuart" },
  plugins: {
    legend: {
      display: true,
      labels: {
        color: "#a9bccf",
        usePointStyle: true,
        pointStyle: "circle",
        boxWidth: 8,
        boxHeight: 8,
        padding: 14,
        font: { family: "'DM Sans'", size: 12, weight: "600" },
      },
    },
    tooltip: {
      ...EQUITY_TOOLTIP,
      callbacks: {
        label: (c) => ` ${c.dataset.label}: ${Number(c.parsed.y).toFixed(4)}`,
      },
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, maxRotation: 0, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
  },
};

const ROLLING_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 500, easing: "easeOutQuart" },
  spanGaps: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      ...EQUITY_TOOLTIP,
      borderColor: "rgba(251, 191, 36, 0.4)",
      callbacks: {
        label: (c) => {
          const v = c.parsed.y;
          if (v == null || Number.isNaN(v)) return " 滚动夏普: —";
          return ` 滚动夏普: ${Number(v).toFixed(3)}`;
        },
      },
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, maxRotation: 0, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(251,191,36,0.06)" },
      border: { color: "rgba(251,191,36,0.18)" },
    },
  },
};

function destroyEquityCharts() {
  if (equityChart) { equityChart.destroy(); equityChart = null; }
  if (rollingChart) { rollingChart.destroy(); rollingChart = null; }
}

function renderEquityStats(name, series) {
  const el = $("btEquityStats");
  if (!el) return;
  const pl = series.profit_loss_ratio;
  const plText = Number.isFinite(Number(pl)) ? Number(pl).toFixed(3) : "—";
  const roll = series.rolling_sharpe || [];
  let lastRoll = null;
  for (let i = roll.length - 1; i >= 0; i--) {
    const v = Number(roll[i]);
    if (Number.isFinite(v)) {
      lastRoll = v;
      break;
    }
  }
  const plNum = Number(pl);
  const bhRet = series.buy_hold_return;
  const cards = [
    { label: "总收益", raw: series.total_return, fmt: "pct", cls: series.total_return >= 0 ? "pos" : "neg" },
    { label: "夏普", raw: series.sharpe, fmt: "signed", cls: "accent" },
    { label: "索提诺", raw: series.sortino, fmt: "signed", cls: "accent" },
    { label: "盈亏比", raw: Number.isFinite(plNum) ? plNum : null, fmt: "ratio", cls: "accent" },
    {
      label: "最新滚动夏普",
      raw: lastRoll,
      fmt: "signed",
      cls: lastRoll == null ? "" : lastRoll >= 0 ? "accent" : "neg",
    },
    {
      label: "买入持有",
      raw: bhRet,
      fmt: "pct",
      cls: bhRet == null ? "" : bhRet >= 0 ? "accent" : "neg",
    },
  ];
  el.innerHTML =
    `<span class="equity-stat-name">${name}</span>` +
    cards
      .map((c) => {
        const finite = c.raw != null && Number.isFinite(c.raw);
        const finalText = finite ? METRIC_FMT[c.fmt](c.raw) : "—";
        const countAttr = finite ? ` data-count="${c.raw}" data-fmt="${c.fmt}"` : "";
        return `
      <div class="equity-stat">
        <span class="equity-stat-label">${c.label}</span>
        <span class="equity-stat-value ${c.cls}"${countAttr}>${finalText}</span>
      </div>`;
      })
      .join("");
  runCountUp(el);
}

function buildEquityChart(labels, symbols, portfolio) {
  const canvas = $("btEquityChart");
  if (!canvas) return;
  const symNames = Object.keys(symbols);
  const multi = symNames.length > 1;
  const datasets = symNames.map((s, i) => {
    const col = EQUITY_COLORS[i % EQUITY_COLORS.length];
    return {
      label: s,
      data: symbols[s].equity,
      borderColor: col.hex,
      borderWidth: multi ? 1.5 : 2.2,
      tension: 0.25,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: col.hex,
      pointHoverBorderColor: "#05070d",
      fill: !multi,
      backgroundColor: (ctx) => verticalGradient(ctx.chart, col.rgb, 0.3, 0),
    };
  });
  // 买入持有基准（虚线灰，与策略曲线同窗口同口径）
  const bhDatasets = [];
  symNames.forEach((s) => {
    const bh = symbols[s].buy_hold;
    if (bh && bh.length) {
      bhDatasets.push({
        label: `买入持有 ${s}`,
        data: bh,
        borderColor: "#94a3b8",
        borderDash: [6, 4],
        borderWidth: 1.4,
        tension: 0.2,
        pointRadius: 0,
        pointHoverRadius: 3,
        pointHoverBackgroundColor: "#94a3b8",
        pointHoverBorderColor: "#05070d",
        fill: false,
      });
    }
  });
  if (portfolio) {
    datasets.push({
      label: "等权组合",
      data: portfolio.equity,
      borderColor: "#e8edf4",
      borderWidth: 2.4,
      tension: 0.25,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: "#e8edf4",
      pointHoverBorderColor: "#05070d",
      fill: true,
      backgroundColor: (ctx) => verticalGradient(ctx.chart, "232, 237, 244", 0.16, 0),
    });
    if (portfolio.buy_hold && portfolio.buy_hold.length) {
      bhDatasets.push({
        label: "买入持有 等权组合",
        data: portfolio.buy_hold,
        borderColor: "#94a3b8",
        borderDash: [6, 4],
        borderWidth: 1.4,
        tension: 0.2,
        pointRadius: 0,
        pointHoverRadius: 3,
        pointHoverBackgroundColor: "#94a3b8",
        pointHoverBorderColor: "#05070d",
        fill: false,
      });
    }
  }
  datasets.push(...bhDatasets);
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: EQUITY_OPTIONS,
  });
}

function buildRollingChart(labels, series, windowBars) {
  const canvas = $("btRollingChart");
  if (!canvas) return;
  if (rollingChart) rollingChart.destroy();
  const data = series.rolling_sharpe || [];
  const labelEl = $("btRollingLabel");
  if (labelEl) {
    labelEl.textContent = windowBars
      ? `滚动夏普 · ${windowBars} bars`
      : "滚动夏普 · Rolling Sharpe";
  }
  rollingChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "滚动夏普",
          data,
          borderColor: "#fbbf24",
          borderWidth: 1.5,
          tension: 0.2,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "#fbbf24",
          pointHoverBorderColor: "#05070d",
          spanGaps: false,
          fill: {
            target: "origin",
            above: "rgba(251, 191, 36, 0.16)",
            below: "rgba(248, 113, 113, 0.16)",
          },
        },
      ],
    },
    options: ROLLING_OPTIONS,
  });
}

function renderEquity(resp) {
  const live = $("btEquityLive");
  const empty = $("btEquityEmpty");
  const data = resp?.data;
  const symbols = data?.symbols || {};
  const symNames = Object.keys(symbols);
  updateEquityRunWarn(data?.run_id || null);

  if (!resp?.available || !symNames.length) {
    if (live) live.hidden = true;
    if (empty) empty.hidden = false;
    destroyEquityCharts();
    btEquitySig = "";
    return;
  }

  const focus = resp.focus_symbol;
  const sig = [focus, data.total_bars, data.n_points, data.rolling_window, symNames.join(",")].join("|") + "|" + btBuster;
  if (sig === btEquitySig && equityChart) return; // 无变化，避免重建闪烁
  btEquitySig = sig;

  if (live) live.hidden = false;
  if (empty) empty.hidden = true;

  const portfolio = data.portfolio || null;
  let mainName, mainSeries;
  if (portfolio) {
    mainName = "等权组合";
    mainSeries = portfolio;
  } else {
    const key = focus && symbols[focus] ? focus : symNames[0];
    mainName = key;
    mainSeries = symbols[key];
  }

  renderEquityStats(mainName, mainSeries);
  buildEquityChart(data.labels, symbols, portfolio);
  buildRollingChart(data.labels, mainSeries, data.rolling_window);

  if ($("btChartsHint")) {
    $("btChartsHint").textContent = `${mainName} · 交互式资金曲线 · 悬停查看数值`;
  }
}

// 资金曲线与绩效卡必须来自同一次回测（同一 run_id）；不一致时提示（旧文件或写盘间隙）
function updateEquityRunWarn(runId) {
  const el = $("btEquityRunWarn");
  if (!el) return;
  const mismatch = btReportRunId && runId && btReportRunId !== runId;
  el.hidden = !mismatch;
  if (!mismatch) return;
  const cur = $("btEquityRunWarnCur");
  const rep = $("btEquityRunWarnRep");
  if (cur) cur.textContent = runId;
  if (rep) rep.textContent = btReportRunId;
}

async function startBacktest() {
  if (!selectedStrategyFile) {
    await logClientError("请先选择策略文件");
    return;
  }
  const startBtn = $("btStartBtn");
  if (startBtn) startBtn.disabled = true;
  try {
    const costs = readBacktestCosts();
    const res = await fetchJSON("/api/backtest/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        strategy_file: selectedStrategyFile,
        data_file: $("btDataSelect")?.value || null,
        commission_pct: costs.commission_pct,
        slippage_pct: costs.slippage_pct,
        hold_policy: btEffectivePolicy().id || "signal",
        window_bars: (() => {
          const v = Number($("btWindowInput")?.value);
          return Number.isFinite(v) && v >= 800 ? v : null;
        })(),
        max_position_pct: (() => {
          const v = Number($("btMaxPosInput")?.value);
          return Number.isFinite(v) && v >= 1 ? v : 100;
        })(),
        signal_threshold: Number($("btThresholdSelect")?.value) || 0.05,
      }),
    });
    // 同一策略文件不重渲染卡片：避免「一点开始回测就恢复参数」（卡片重渲染会
    // 重新走按品种记忆/矩阵最优的恢复路径）。换模型/重选策略时 renderStrategyFileCard
    // 自己会跑恢复——那是「打开新页面」语义，保留预填充。
    if (res.strategy_file && res.strategy_file !== selectedStrategyFile) renderStrategyFileCard(res.strategy_file);
    await refreshBacktest();
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
    updateBtStartBtn();
  }
}

async function stopBacktest() {
  try {
    await fetchJSON("/api/backtest/stop", { method: "POST" });
    await refreshBacktest();
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
  }
}

// ═══════════════════════════════════════════════════════════════════
// 实时行情分析（信号雷达）
// ═══════════════════════════════════════════════════════════════════
let rtInited = false;
let rtEngineRunning = false;
let rtSources = [];
let rtSourceById = {};
let rtImportedStrategy = null; // {path, name}
let rtGridSig = "";
let rtServerSkew = 0; // server_time - local_now（秒）
let rtCountdownTimer = null;
let rtLiveById = {}; // 最近一次轮询的监控状态（供每秒刷新现价，不重建 DOM）
let rtTvBlockedShownAt = 0;         // 最近一次真正弹出模态的时间（60s 防抖）
let rtTvBlockedEpisodeKey = "";      // 被墙 TradingView 监控项集合签名（id 排序）；集合变化=新剧集
let rtTvBlockedSuppressedKey = "";   // 用户已点「暂不处理」/关掉模态的剧集——同剧集不再被动弹出
let rtTvBlockedSuppressedAt = 0;
let rtTvWikiUrl = "https://my.feishu.cn/wiki/FuqnwkPwdiCLhQkPloKc7r1lntg";
const RT_TV_BLOCKED_MSG =
  "当前设备无法连接 TradingView 数据服务，将无法获取以下 K 线数据：\n" +
  "  · A 股（上证 SSE、深证 SZSE）\n" +
  "  · 港股（HKEX）\n" +
  "  · 美股及指数（NYSE、NASDAQ、SP）\n" +
  "  · 外汇、贵金属、商品期货\n\n" +
  "解决方案：\n" +
  "  · 把你的VPN工具设成全局，并开启TUN(虚拟网卡)模式，如果还不行：\n" +
  "  · 使用云服务器部署本程序（推荐）—— 云服务器可正常连接 TradingView";
const RT_TV_BLOCKED_CODE = "TV_CONNECTIVITY_BLOCKED";

const RT_DIR = {
  LONG: { label: "↑ 预期上涨", cls: "rt-long", color: "#4ade80" },
  SHORT: { label: "↓ 预期下跌", cls: "rt-short", color: "#f87171" },
  FLAT: { label: "— 先观望", cls: "rt-flat", color: "#7a8a9e" },
};
const RT_STATE_LABEL = {
  pending: "等待首次计算",
  ok: "运行中",
  insufficient: "历史不足",
  error: "错误",
};

/** 把 0~1 强度翻成「把握」白话 */
function rtSizePlain(strength, direction) {
  if (direction === "FLAT" || direction == null) {
    return { size: "没把握" };
  }
  const s = Math.max(0, Math.min(1, Number(strength) || 0));
  let size;
  if (s < 0.2) size = "一点把握";
  else if (s < 0.4) size = "把握不大";
  else if (s < 0.6) size = "一半把握";
  else if (s < 0.8) size = "比较有把握";
  else size = "很有把握";
  return { size };
}

function escHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}

function rtClock(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}

function fmtPrice(v) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v);
  const abs = Math.abs(n);
  const dec = abs >= 1000 ? 2 : abs >= 1 ? 4 : 6;
  return n.toLocaleString("en-US", {
    minimumFractionDigits: dec,
    maximumFractionDigits: dec,
  });
}

function rtNowSec() {
  return Date.now() / 1000 + rtServerSkew;
}

function rtFmtCountdown(sec) {
  const s = Math.max(0, Math.floor(sec));
  if (s < 60) return `${s}秒`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}分${String(rs).padStart(2, "0")}秒`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  if (h < 48) return `${h}小时${rm}分`;
  const d = Math.floor(h / 24);
  return `${d}天${h % 24}小时`;
}

function ensureRtCountdownTimer() {
  if (rtCountdownTimer) return;
  rtCountdownTimer = setInterval(tickRtCountdowns, 1000);
}

function tickRtCountdowns() {
  document.querySelectorAll(".rt-countdown").forEach((el) => {
    if (el.dataset.session === "closed") {
      el.textContent = "休市中";
      return;
    }
    const nxt = Number(el.dataset.nextClose);
    if (!Number.isFinite(nxt) || nxt <= 0) {
      el.textContent = "距离下次判断 —";
      return;
    }
    const left = nxt - rtNowSec();
    el.textContent = left <= 0 ? "即将重新判断…" : `距离下次判断 ${rtFmtCountdown(left)}`;
  });
  // 现价：随每次轮询（~4s）跳动，不重建卡片 DOM、不重播动画
  document.querySelectorAll(".rt-px").forEach((el) => {
    const w = rtLiveById[el.dataset.id];
    const live = w ? w.live_price : null;
    const px = live != null ? Number(live) : w && w.last_close != null ? Number(w.last_close) : null;
    const txt = fmtPrice(px);
    if (el.textContent !== txt) el.textContent = txt;
    const card = el.closest(".rt-card");
    if (!card) return;
    const up = live != null && w.last_close != null && Number(live) >= Number(w.last_close);
    const down = live != null && w.last_close != null && Number(live) < Number(w.last_close);
    card.classList.toggle("rt-px-up", up);
    card.classList.toggle("rt-px-down", down);
    // 入场锚芯片：现价（live_price）随轮询刷新 → 偏离%同步更新（内容没变不重写）。
    // rtAnchorChipHtml 返回两个兄弟节点（.rt-anchor + .rt-anchor-hist），仅 outerHTML
    // 替换 .rt-anchor 会把旧的 .rt-anchor-hist 留在原地，每 tick 累积一份 → 先删旧历史再插入。
    const anc = card.querySelector(`.rt-anchor[data-rt-anchor-id="${CSS.escape(w.id)}"]`);
    if (anc) {
      const n = rtAnchorChipHtml(w);
      if (anc.dataset.sig !== n) {
        const oldHist = anc.nextElementSibling;
        const hasOldHist =
          oldHist && oldHist.classList && oldHist.classList.contains("rt-anchor-hist");
        anc.outerHTML = n;
        if (hasOldHist) oldHist.remove();
        const nn = card.querySelector(
          `.rt-anchor[data-rt-anchor-id="${CSS.escape(w.id)}"]`
        );
        if (nn) nn.dataset.sig = n;
      }
    }
  });
  // 形成中实时价尾点：曲线在两根已收盘 bar 之间持续延伸（不重建卡片、不重播动画）
  document.querySelectorAll(".rt-chart").forEach((chart) => {
    if (chart.classList.contains("rt-chart-empty")) return;
    const card = chart.closest(".rt-card");
    if (!card) return;
    const w = rtLiveById[card.dataset.id];
    if (!w || !w.session_live) return;
    const lp = w.live_price != null ? Number(w.live_price) : null;
    const curTail = lp != null && Number.isFinite(lp) ? lp.toFixed(8) : "";
    if (!curTail || curTail === chart.dataset.liveTail) return;
    const svgEl = chart.querySelector("svg");
    if (!svgEl) return;
    chart.dataset.liveTail = curTail;
    svgEl.outerHTML = svgPriceChart(rtChartPts(w), rtChartLevels(w), {});
  });
  const hintCd = $("rtNextHint");
  if (hintCd) {
    if (hintCd.dataset.session === "closed") {
      hintCd.textContent = "休市中";
      return;
    }
    if (hintCd.dataset.nextClose) {
      const nxt = Number(hintCd.dataset.nextClose);
      if (Number.isFinite(nxt) && nxt > 0) {
        const left = nxt - rtNowSec();
        hintCd.textContent =
          left <= 0 ? "即将重新判断" : `距离下次判断 ${rtFmtCountdown(left)}`;
      }
    }
  }
}

async function initRealtimeOnce() {
  if (rtInited) return;
  rtInited = true;
  try {
    const data = await fetchJSON("/api/realtime/sources");
    rtSources = data.sources || [];
    rtSourceById = {};
    rtSources.forEach((s) => (rtSourceById[s.id] = s));
    const sel = $("rtSourceSelect");
    if (sel) {
      sel.innerHTML = rtSources
        .map((s) => `<option value="${s.id}">${escHtml(s.label)}${s.available ? "" : " · 未就绪"}</option>`)
        .join("");
      // 默认选第一个可用数据源
      if (rtSources.length) sel.value = rtSources[0].id;
    }
    if (data.min_exposure != null && $("rtThresholdHint")) {
      $("rtThresholdHint").textContent = `|tanh(因子)| < ${data.min_exposure} → FLAT`;
    }
    if (data.signal_threshold != null) applySigThrSelects(Number(data.signal_threshold));
    onRtSourceChange();
  } catch (e) {
    await logClientError("加载数据源失败: " + e.message);
  }
  await loadRtStrategies();
  await loadRtFeishuSettings();
}

async function loadRtFeishuSettings() {
  try {
    const data = await fetchJSON("/api/realtime/feishu");
    const en = $("rtFeishuEnabled");
    const wh = $("rtFeishuWebhook");
    const sec = $("rtFeishuSecret");
    if (en) en.checked = !!data.enabled;
    if (wh) wh.value = data.webhook_url || "";
    if (sec) sec.value = data.secret || "";
    const dev = $("rtAlertDevPct");
    if (dev) dev.value = data.rt_alert_dev_pct != null ? String(data.rt_alert_dev_pct) : "0.5";
    const stale = $("rtStaleBarsInput");
    if (stale) stale.value = data.rt_alert_stale_bars != null ? String(data.rt_alert_stale_bars) : "10";
  } catch (e) {
    const hint = $("rtFeishuHint");
    if (hint) {
      hint.textContent = "加载飞书设置失败: " + e.message;
      hint.classList.add("bad");
    }
  }
}

async function saveRtFeishuSettings() {
  const hint = $("rtFeishuHint");
  const btn = $("rtFeishuSaveBtn");
  if (btn) btn.disabled = true;
  try {
    await fetchJSON("/api/realtime/feishu", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        enabled: !!$("rtFeishuEnabled")?.checked,
        webhook_url: $("rtFeishuWebhook")?.value || "",
        secret: $("rtFeishuSecret")?.value || "",
        rt_alert_dev_pct: Number($("rtAlertDevPct")?.value),
        rt_alert_stale_bars: Number($("rtStaleBarsInput")?.value),
      }),
    });
    if (hint) {
      hint.textContent = "✓ 已保存，方向转折时会推送到飞书群。";
      hint.classList.remove("bad", "invalid");
      hint.classList.add("valid");
    }
    if (btn) {
      const old = btn.textContent;
      btn.textContent = "已保存";
      setTimeout(() => {
        if (btn.textContent === "已保存") btn.textContent = old || "保存";
      }, 1600);
    }
  } catch (e) {
    if (hint) {
      hint.textContent = "保存失败: " + e.message;
      hint.classList.remove("valid");
      hint.classList.add("bad", "invalid");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function testRtFeishu() {
  const hint = $("rtFeishuHint");
  const btn = $("rtFeishuTestBtn");
  if (btn) btn.disabled = true;
  try {
    await fetchJSON("/api/realtime/feishu/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        webhook_url: $("rtFeishuWebhook")?.value || "",
        secret: $("rtFeishuSecret")?.value || "",
      }),
    });
    if (hint) {
      hint.textContent = "✓ 测试消息已发送，请到飞书群查收。";
      hint.classList.remove("bad", "invalid");
      hint.classList.add("valid");
    }
  } catch (e) {
    if (hint) {
      hint.textContent = "测试失败: " + e.message;
      hint.classList.remove("valid");
      hint.classList.add("bad", "invalid");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function openRtFeishuHelpModal() {
  const modal = $("rtFeishuHelpModal");
  if (modal) modal.hidden = false;
}

function closeRtFeishuHelpModal() {
  const modal = $("rtFeishuHelpModal");
  if (modal) modal.hidden = true;
}

async function loadRtStrategies() {
  const sel = $("rtStrategySelect");
  if (!sel) return;
  let rows = [];
  try {
    // 与回测页/训练页同源：/api/strategies 每行带 strategy_file 完整路径
    const data = await fetchJSON("/api/strategies", { silent: true, retries: 0 });
    rows = data.strategies || [];
  } catch (_) {}
  const opts = ['<option value="">— 选择已保存策略 —</option>'];
  if (rtImportedStrategy) {
    const isym = escHtml(rtImportedStrategy.symbol || "");
    opts.push(
      `<option value="${escHtml(rtImportedStrategy.path)}" data-symbol="${isym}">导入: ${escHtml(rtImportedStrategy.name)}</option>`
    );
  }
  rows.forEach((r) => {
    if (!r.strategy_file) return;
    const score = r.best_score != null ? Number(r.best_score).toFixed(3) : "—";
    const tf = r.timeframe ? ` ${r.timeframe}` : "";
    const ds = r.data_source && r.data_source.file ? ` · ${r.data_source.file}` : "";
    opts.push(
      `<option value="${escHtml(r.strategy_file)}" data-symbol="${escHtml(r.symbol || "")}" ` +
        `title="${escHtml(r.file || "")}">${escHtml(r.symbol || r.file)}${escHtml(tf)} · 分数 ${score}${ds}</option>`
    );
  });
  const prev = sel.value;
  sel.innerHTML = opts.join("");
  if (rtImportedStrategy) sel.value = rtImportedStrategy.path;
  else if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  onRtStrategyChange();
}

function onRtSourceChange() {
  const src = rtSourceById[$("rtSourceSelect")?.value];
  const tfSel = $("rtTimeframeSelect");
  const presets = $("rtSymbolPresets");
  const hint = $("rtSourceHint");
  if (!src) return;
  if (tfSel) {
    const cur = tfSel.value;
    tfSel.innerHTML = (src.timeframes || []).map((t) => `<option value="${t}">${t}</option>`).join("");
    if (src.timeframes && src.timeframes.includes(cur)) tfSel.value = cur;
    else if (src.timeframes && src.timeframes.includes("1h")) tfSel.value = "1h";
  }
  // 品种输入/下拉切换：presets 较多时（如国内期货 60 个品种）用下拉框
  const symbolInput = $("rtSymbolInput");
  const symbolSelect = $("rtSymbolSelect");
  const useSelect = src.id === "domestic_futures" || (src.presets && src.presets.length > 20);
  if (symbolInput && symbolSelect) {
    if (useSelect) {
      symbolInput.hidden = true;
      symbolSelect.hidden = false;
      symbolSelect.innerHTML = (src.presets || [])
        .map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`)
        .join("");
      symbolSelect.onchange = () => { symbolInput.value = symbolSelect.value; };
      if (symbolSelect.value) symbolInput.value = symbolSelect.value;
    } else {
      symbolInput.hidden = false;
      symbolSelect.hidden = true;
      if (presets) {
        presets.innerHTML = (src.presets || [])
          .map((s) => `<option value="${escHtml(s)}"></option>`)
          .join("");
      }
    }
  } else if (presets) {
    presets.innerHTML = (src.presets || [])
      .map((s) => `<option value="${escHtml(s)}"></option>`)
      .join("");
  }
  if (hint) {
    hint.textContent = `${src.label}：${src.hint || ""}`;
    hint.classList.toggle("bad", !src.available);
  }
}

function rtParseSymbolFromFilename(pathOrName) {
  const name = String(pathOrName || "").split(/[/\\]/).pop() || "";
  let m = name.match(/^best_(.+)\.json$/i);
  if (m) return m[1];
  m = name.match(/^strategy_(.+)_step\d+/i);
  if (m) return m[1];
  return "";
}

function rtApplySymbolFromStrategy(sym) {
  const s = String(sym || "").trim();
  if (!s) return;
  const input = $("rtSymbolInput");
  if (input) input.value = s;
  const select = $("rtSymbolSelect");
  if (select && !select.hidden) select.value = s;
}

function onRtStrategyChange() {
  const sel = $("rtStrategySelect");
  const picked = $("rtStrategyPicked");
  if (!sel || !picked) return;
  const opt = sel.options[sel.selectedIndex];
  picked.textContent = sel.value
    ? `因子来源：${opt ? opt.textContent : sel.value}。信号取最后已收盘 bar。`
    : "因子来源：从已保存策略下拉选择，或「导入策略」选本地 JSON。信号取最后已收盘 bar。";
  if (!sel.value) return;
  const fromOpt = (opt && opt.dataset.symbol) || "";
  const fromImport =
    rtImportedStrategy && sel.value === rtImportedStrategy.path
      ? rtImportedStrategy.symbol || ""
      : "";
  const sym = fromOpt || fromImport || rtParseSymbolFromFilename(sel.value);
  rtApplySymbolFromStrategy(sym);
}

async function rtBrowseStrategy() {
  let res;
  try {
    res = await fetchJSON("/api/strategy-file/browse", { method: "POST", retries: 0 });
  } catch (e) {
    await logClientError("导入策略失败: " + e.message);
    return;
  }
  if (!res.dialog || !res.session) {
    if (!res.cancelled) {
      const name0 = res.filename || res.strategy_file;
      rtImportedStrategy = {
        path: res.strategy_file,
        name: name0,
        symbol: (res.symbol || "").trim() || rtParseSymbolFromFilename(name0),
      };
      await loadRtStrategies();
      rtApplySymbolFromStrategy(rtImportedStrategy.symbol);
    }
    return;
  }
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 800));
      const p = await fetchJSON(
        "/api/strategy-file/browse-poll?session=" + encodeURIComponent(res.session),
        { retries: 0 }
      );
      if (!p.done) continue;
      if (!p.cancelled) {
        const name = p.filename || p.strategy_file;
        rtImportedStrategy = {
          path: p.strategy_file,
          name,
          symbol: (p.symbol || "").trim() || rtParseSymbolFromFilename(name),
        };
        await loadRtStrategies();
        rtApplySymbolFromStrategy(rtImportedStrategy.symbol);
      }
      return;
    }
  } catch (e) {
    await logClientError("导入策略失败: " + e.message);
  }
}

async function rtAddWatch() {
  const source = $("rtSourceSelect")?.value;
  const symbol = ($("rtSymbolInput")?.value || "").trim();
  const timeframe = $("rtTimeframeSelect")?.value;
  const strategy_file = $("rtStrategySelect")?.value;
  const picked = $("rtStrategyPicked");
  if (!symbol) {
    if (picked) { picked.textContent = "请填写品种"; picked.classList.add("bad"); }
    return;
  }
  if (!strategy_file) {
    if (picked) { picked.textContent = "请选择或导入策略因子"; picked.classList.add("bad"); }
    return;
  }
  const btn = $("rtAddBtn");
  if (btn) btn.disabled = true;
  try {
    if (source === "tradingview") {
      const reachable = await rtEnsureTradingViewReachable();
      if (!reachable) return;
    }
    const policy_id = effectivePolicy("rtPolicySelect", "rtStackSelect").id;
    await fetchJSON("/api/realtime/watch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, symbol, timeframe, strategy_file, policy_id }),
    });
    if (picked) picked.classList.remove("bad");
    rtEngineRunning = true;
    rtGridSig = "";
    await refreshRealtime();
  } catch (e) {
    if (picked) { picked.textContent = "添加失败: " + e.message; picked.classList.add("bad"); }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function rtEnsureTradingViewReachable() {
  const picked = $("rtStrategyPicked");
  if (picked) {
    picked.textContent = "正在检测 TradingView 连通性…";
    picked.classList.remove("bad");
  }
  try {
    const res = await fetchJSON("/api/realtime/tradingview/probe", {
      method: "POST",
      silent: true,
    });
    if (res && res.ok) {
      if (picked) picked.textContent = "";
      return true;
    }
    if (res?.wiki_url) rtTvWikiUrl = res.wiki_url;
    await showTvBlockedDialog(res?.message || RT_TV_BLOCKED_MSG);
    if (picked) {
      picked.textContent = "TradingView 不可用，请开 VPN 或换云服务器";
      picked.classList.add("bad");
    }
    return false;
  } catch (e) {
    await showTvBlockedDialog(RT_TV_BLOCKED_MSG);
    if (picked) {
      picked.textContent = "TradingView 检测失败: " + (e.message || e);
      picked.classList.add("bad");
    }
    return false;
  }
}

function showTvBlockedDialog(message, opts) {
  opts = opts || {};
  const now = Date.now();
  const modal = $("tvBlockedModal");
  const body = $("tvBlockedBody");
  // 模态已打开时不重复弹（防抖只对“被动触发”生效；manual=用户主动点徽标/探测，始终允许）
  if (modal && !modal.hidden) return Promise.resolve("already-open");
  if (!opts.manual && now - rtTvBlockedShownAt < 60_000) {
    return Promise.resolve("dedupe");
  }
  rtTvBlockedShownAt = now;
  if (!modal || !body) {
    window.alert(message || RT_TV_BLOCKED_MSG);
    return Promise.resolve("alert");
  }
  body.textContent = message || RT_TV_BLOCKED_MSG;
  modal.hidden = false;
  return new Promise((resolve) => {
    modal._tvResolve = resolve;
  });
}

function closeTvBlockedDialog(choice) {
  const modal = $("tvBlockedModal");
  if (modal) modal.hidden = true;
  // 记住“本剧集已被拒绝”，被动轮询不再自动弹（要重新提示需被墙集合变化=新剧集，
  // 或用户主动点徽标/再试一次——这是修复「每隔一分钟反复弹出」的关键）
  if (rtTvBlockedEpisodeKey) {
    rtTvBlockedSuppressedKey = rtTvBlockedEpisodeKey;
    rtTvBlockedSuppressedAt = Date.now();
  }
  const resolve = modal && modal._tvResolve;
  if (resolve) {
    modal._tvResolve = null;
    resolve(choice || "cancel");
  }
}

function onTvBlockedLater() {
  closeTvBlockedDialog("cancel");
}

function onTvBlockedOpenCloud() {
  try {
    window.open(rtTvWikiUrl, "_blank", "noopener,noreferrer");
  } catch (_) {}
  closeTvBlockedDialog("cloud");
}

// 被墙的 TradingView 监控项（tradingview 源且处于连接被墙态）
function rtBlockedTvWatches(watches) {
  return (watches || []).filter(
    (w) => w.source === "tradingview" && (w.tv_blocked || w.message === RT_TV_BLOCKED_CODE)
  );
}

// 页面头部状态行里加一个非阻塞小徽标（可手动点开方案弹窗），被墙期间常驻、不刷屏
function renderRtTvChip(blockedCount) {
  const hint = $("rtStatusHint");
  if (!hint) return;
  let chip = $("rtTvBlockedChip");
  if (!blockedCount) {
    if (chip) chip.remove();
    return;
  }
  if (!chip) {
    chip = document.createElement("button");
    chip.type = "button";
    chip.id = "rtTvBlockedChip";
    chip.className = "rt-tv-chip";
    chip.title = `有 ${blockedCount} 个 TradingView 监控项因无法连接数据服务被挂起 — 点击查看解决方案`;
    chip.textContent = `⛔ TradingView 不可用 ×${blockedCount}`;
    chip.addEventListener("click", () => showTvBlockedDialog(RT_TV_BLOCKED_MSG, { manual: true }));
    hint.appendChild(chip);
  } else if (chip.textContent !== `⛔ TradingView 不可用 ×${blockedCount}`) {
    chip.textContent = `⛔ TradingView 不可用 ×${blockedCount}`;
  }
}

function maybeShowTvBlockedFromWatches(watches) {
  const blocked = rtBlockedTvWatches(watches);
  const key = blocked.map((w) => w.id).sort().join("|");
  // 剧集变化：被墙集合有增/减 → 允许重新提示一次（同剧集被拒后不再骚扰）
  if (key !== rtTvBlockedEpisodeKey) {
    rtTvBlockedEpisodeKey = key;
    if (key) {
      rtTvBlockedSuppressedKey = "";
      rtTvBlockedSuppressedAt = 0;
      rtTvBlockedShownAt = 0;
    }
  }
  renderRtTvChip(blocked.length);
  if (!key) return;
  // 被动自动弹出只在实时分析页；且同剧集已拒绝过就不再弹
  if (currentPage !== "realtime") return;
  if (rtTvBlockedSuppressedKey === key) return;
  const modal = $("tvBlockedModal");
  if (modal && !modal.hidden) return;
  showTvBlockedDialog(RT_TV_BLOCKED_MSG);
}

async function rtRemoveWatch(id) {
  // 乐观移除：先本地重绘（卡立刻消失），服务端异步收尾——unwatch/status 内部
  // 会同步做现价网络拉取，等它会让“关卡片”卡上 1~3 秒
  const prev = Object.assign({}, rtLiveById);
  delete rtLiveById[id];
  rtGridSig = "";
  renderRealtimeGrid(Object.values(rtLiveById));
  try {
    await fetchJSON("/api/realtime/unwatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    refreshRealtime(); // 后台同步剩余卡片的计数/角标/现价，不阻塞点击
  } catch (e) {
    // 服务端失败 → 回滚本地状态，卡片恢复
    Object.assign(rtLiveById, prev);
    rtGridSig = "";
    renderRealtimeGrid(Object.values(rtLiveById));
    await logClientError("移除监控失败: " + e.message);
  }
}

async function refreshRealtime() {
  let st;
  try {
    st = await fetchJSON("/api/realtime/status", { silent: true });
  } catch (_) {
    return;
  }
  // 有监控项却未在跑时自动拉起（不再提供手动开关）；仅 REST 路径做，避免 SSE 推送上反复 POST
  if (st.count > 0 && !st.running) {
    try {
      st = await fetchJSON("/api/realtime/start", { method: "POST", silent: true });
    } catch (_) {}
  }
  applyRealtimeState(st);
}

// 快照 → UI（REST 轮询与 SSE 推送共用同一入口；渲染自带签名守卫）
function applyRealtimeState(st) {
  rtEngineRunning = !!st.running;
  if (typeof st.server_time === "number") {
    rtServerSkew = st.server_time - Date.now() / 1000;
  }

  const nearest = st.nearest_seconds_to_next;
  let nearestClose = null;
  let anyLive = false;
  let anyOk = false;
  for (const w of st.watches || []) {
    if (w.state === "ok") anyOk = true;
    if (w.session_live && w.next_bar_close_at != null) {
      anyLive = true;
      if (nearestClose == null || w.next_bar_close_at < nearestClose) {
        nearestClose = w.next_bar_close_at;
      }
    }
  }

  const hint = $("rtStatusHint");
  if (hint) {
    const base = st.count
      ? `${rtEngineRunning ? "运行中" : "已暂停"} · ${st.count} 项`
      : "暂无监控项";
    if (nearestClose) {
      hint.innerHTML = `${base} · <span id="rtNextHint" data-next-close="${nearestClose}"></span>`;
    } else if (anyOk && !anyLive) {
      hint.innerHTML = `${base} · <span id="rtNextHint" data-session="closed">休市中</span>`;
    } else {
      hint.textContent = base;
    }
  }
  rtLiveById = {};
  (st.watches || []).forEach((w) => (rtLiveById[w.id] = w));
  renderRealtimeGrid(st.watches || []);
  maybeShowTvBlockedFromWatches(st.watches || []);
  ensureRtCountdownTimer();
  tickRtCountdowns();
  loadDdEventsPanel("Rt", false);
}

// ═══════════════════════════════════════════════════════════════════
// 持仓管理方案 + 模型库 + 迷你价格图（回测 / 实时 / 模拟盘共用）
// ═══════════════════════════════════════════════════════════════════
let holdPolicyList = [];
const holdPolicyById = {};

async function loadHoldPolicies() {
  try {
    const data = await fetchJSON("/api/hold-policies", { silent: true, retries: 0 });
    holdPolicyList = data.policies || [];
  } catch (_) {}
  holdPolicyById.signal = { name: "信号跟随", desc: "与回测同口径" };
  holdPolicyList.forEach((p) => (holdPolicyById[p.id] = p));
  bindCostSweep();
  ["btPolicySelect", "rtPolicySelect", "ppPolicySelect", "rsPolicySelect"].forEach((id) => {
    const sel = $(id);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = holdPolicyList
      .map(
        (p) =>
          `<option value="${escHtml(p.id)}" ${p.default ? "selected" : ""} title="${escHtml(p.desc)}">${escHtml(p.name)}${p.default ? "（默认）" : ""}</option>`
      )
      .join("");
    if (prev && holdPolicyList.some((p) => p.id === prev)) sel.value = prev;
  });
  // 回测页 / 实时分析 / 模拟实盘「正交叠加」：同一注册表（含默认 信号跟随）全部可选
  fillBtStackSelect();
  fillStackSelect("rtStackSelect");
  fillStackSelect("ppStackSelect");
  updateBtComboHint();
  updateStackHint("rtPolicyHint", "rtPolicySelect", "rtStackSelect");
  updateStackHint("ppPolicyHint", "ppPolicySelect", "ppStackSelect");
  ["rtPolicySelect", "rtStackSelect"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", () => updateStackHint("rtPolicyHint", "rtPolicySelect", "rtStackSelect"));
  });
  ["ppPolicySelect", "ppStackSelect"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", () => updateStackHint("ppPolicyHint", "ppPolicySelect", "ppStackSelect"));
  });
  if (!window.__btComboWired) {
    window.__btComboWired = true;
    ["btPolicySelect", "btStackSelect"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("change", () => { updateBtComboHint(); scheduleBtPrefsSave(); btLoadMatrixBest(); });
    });
    const winEl = $("btWindowInput");
    if (winEl) winEl.addEventListener("change", scheduleBtPrefsSave);
  }
  restoreBtPrefs();
}

// 主方案 + 正交叠加下拉 → 规范化后的组合 id（与后端 combo_id 同规则：
// 含 signal 时若还有其他模块则剔除 signal；全剔除或空 → signal）
function fillStackSelect(selId) {
  const sel = $(selId);
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML =
    `<option value="" title="只按上方主方案，不加第二层出场">不叠加</option>` +
    holdPolicyList
      .map(
        (p) =>
          `<option value="${escHtml(p.id)}" title="${escHtml(p.desc)}">${escHtml(p.name)}${p.id === "signal" ? "（默认）" : ""}</option>`
      )
      .join("");
  sel.value = prev && [...sel.options].some((o) => o.value === prev) ? prev : "";
}

function fillBtStackSelect() {
  fillStackSelect("btStackSelect");
}

function effectivePolicy(baseSelId, stackSelId) {
  const base = $(baseSelId)?.value || "signal";
  const stk = $(stackSelId)?.value || "";
  const seen = [];
  [base, stk].forEach((v) => {
    if (v && !seen.includes(v)) seen.push(v);
  });
  const mods = seen.filter((v) => v !== "signal");
  return { id: mods.length ? mods.join("+") : "signal", mods };
}

function btEffectivePolicy() {
  return effectivePolicy("btPolicySelect", "btStackSelect");
}

function comboDisplayName(id) {
  const mods = (id || "signal").split("+").filter(Boolean);
  return mods.map((m) => policyName(m)).join(" + ");
}

function updateStackHint(hintId, baseSelId, stackSelId) {
  const hint = $(hintId);
  const sel1 = $(baseSelId);
  if (!hint || !sel1) return;
  const base = sel1.value || "signal";
  const stk = $(stackSelId)?.value || "";
  const eff = effectivePolicy(baseSelId, stackSelId);
  let msg;
  if (eff.id === "signal") {
    msg = `主方案「${policyName(base)}」· 不叠加出场 → 生效：信号跟随（与训练/回测口径一致）`;
  } else if (stk && stk !== "signal" && stk !== base) {
    msg = `正交组合：${policyName(base)} ⊕ ${policyName(stk)} → 生效 ${comboDisplayName(eff.id)}`;
  } else {
    msg = `持仓管理：${comboDisplayName(eff.id)}`;
  }
  hint.textContent = msg;
}

function updateBtComboHint() {
  updateStackHint("btPolicyHint", "btPolicySelect", "btStackSelect");
  renderBtLastCombo();
}

// ═══ 记忆化：最近一次 持仓组合 + 样本外窗口（重启后保留） ═══
let btPrefsSaveTimer = null;
let __btApplyingPrefs = false;

function scheduleBtPrefsSave() {
  if (__btApplyingPrefs) return; // 恢复中不回写
  if (btPrefsSaveTimer) clearTimeout(btPrefsSaveTimer);
  btPrefsSaveTimer = setTimeout(async () => {
    const eff = btEffectivePolicy();
    const wv = Number($("btWindowInput")?.value);
    const windowBars = Number.isFinite(wv) && wv >= 800 ? Math.round(wv) : null;
    // 记忆 + 即时可见标签 + 同步到 模拟实盘/实时分析/历史回放 页
    window.__btMemPolicy = eff.id;
    window.__btMemWindow = windowBars;
    renderBtLastCombo();
    applyBtPrefsToOthers(eff.id, windowBars, Number($("btMaxPosInput")?.value));
    try {
      await fetchJSON("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bt_hold_policy: eff.id, bt_window_bars: windowBars }),
        silent: true,
      });
    } catch (_) {}
    // 按品种记忆：同一品种下次切换模型/重跑时沿用该品种的组合与窗口
    if (selectedStrategySymbol) {
      try {
        await fetchJSON("/api/backtest/prefs", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            bt_prefs_symbol: selectedStrategySymbol,
            bt_hold_policy: eff.id,
            bt_window_bars: windowBars,
          }),
          silent: true,
        });
      } catch (_) {}
    }
  }, 600);
}

// 可见的「最近一次组合」标签：显示已记忆的持仓组合 + 样本外窗口（重启保留）
function renderBtLastCombo() {
  const el = $("btLastCombo");
  if (!el) return;
  const hp = window.__btMemPolicy || "signal";
  const wb = window.__btMemWindow;
  const name = comboDisplayName(hp);
  const winTxt = wb ? `样本外窗口 尾部 ${wb} 根` : "全部历史";
  const tag = hp === "signal" ? "" : " ★";
  el.textContent = `最近一次组合（已记忆 · 重启保留）：${name} · ${winTxt}${tag}`;
}

// ═══ 矩阵最优组合 → 新回测默认持仓管理（同品种；标注最大回撤约束） ═══
let __btMatrixBest = null;
let __btMatrixBestInFlight = false;
let __btMatrixUserApply = false; // 用户在「应用为新回测默认」按钮上的显式点击（其内部 reload 不自动改、不弹通知）
// 用户接受记录（settings.matrix_applied，与手动记忆 bt_prefs/bt_hold_policy 区分开）：
// {品种: {combo, at, sharpe, max_drawdown, ...}} —— 仅用于标识「已应用」态，不再自动恢复参数
let __btMatrixAppliedBySym = {};

async function persistMatrixApplied(symbol, b) {
  if (!symbol || !b || !b.combo) return;
  try {
    const res = await fetchJSON("/api/backtest/matrix-applied", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol,
        combo: b.combo,
        sharpe: b.sharpe != null ? Number(b.sharpe) : null,
        max_drawdown: b.max_drawdown != null ? Number(b.max_drawdown) : null,
        window_bars: b.window_bars != null ? Number(b.window_bars) : null,
        window_mode: b.window_mode || null,
      }),
      silent: true,
      retries: 0,
    });
    if (res && res.ok) __btMatrixAppliedBySym[symbol] = res.applied || null;
  } catch (_) {}
}

async function btLoadMatrixBest() {
  const box = $("btMatrixBest");
  if (!box) return;
  const sym = selectedStrategySymbol;
  if (!sym) {
    box.innerHTML = "";
    __btMatrixBest = null;
    return;
  }
  if (__btMatrixBestInFlight) return;
  __btMatrixBestInFlight = true;
  let best = null;
  try {
    const d = await fetchJSON(
      "/api/backtest/matrix-best?symbol=" + encodeURIComponent(sym),
      { silent: true, retries: 0 }
    );
    best = d && d.exists ? d.best : null;
  } catch (_) {
    best = null;
  }
  // 同品种的「接受记录」（不存在时不阻塞主流程）
  let rec = null;
  if (best && best.combo) {
    try {
      const a = await fetchJSON(
        "/api/backtest/matrix-applied?symbol=" + encodeURIComponent(sym),
        { silent: true, retries: 0 }
      );
      rec = a && a.exists ? a.applied : null;
      __btMatrixAppliedBySym[sym] = rec;
    } catch (_) {}
  }
  __btMatrixBestInFlight = false;
  __btMatrixBest = best;
  if (!best || !best.combo) {
    box.innerHTML = "";
    return;
  }
  const applied = btEffectivePolicy().id === best.combo; // 以当前下拉实际生效组合为准
  const accepted = !!(rec && rec.combo === best.combo); // 这份组合是否被“接受”并记入 settings
  const ddS =
    best.max_drawdown != null
      ? `最大回撤 ${(Number(best.max_drawdown) * 100).toFixed(1)}%`
      : "最大回撤 —";
  const sh = best.sharpe != null ? `夏普 ${Number(best.sharpe).toFixed(2)}` : "夏普 —";
  const winTxt =
    best.window_mode === "spread"
      ? `窗口 分层抽样 ${best.window_bars || "?"} 根`
      : best.window_bars
        ? `窗口 样本外尾部 ${best.window_bars} 根`
        : "窗口 全部历史";
  const recAt = rec && rec.at
    ? new Date(Number(rec.at) * 1000).toLocaleString()
    : "";
  const badgeHtml = applied
    ? accepted
      ? `<span class="bt-mb-badge" title="已记入接受记录（settings.matrix_applied）${recAt ? "（" + recAt + "）" : ""}；记录仅用于标识，不再自动恢复参数">已应用 · 矩阵接受 ✓</span>`
      : `<span class="bt-mb-badge" title="当前生效组合来自手动选择/记忆（非矩阵接受记录）">已作为新回测默认（记忆）</span>`
    : `<button type="button" class="btn btn-mini btn-primary" id="btMatrixBestApplyBtn">应用为新回测默认</button>`;
  box.innerHTML =
    `<div class="bt-matrix-best${applied ? " applied" : ""}" title="来自 results/matrix_best_combo.json（最近一次全组合矩阵；最优=非基线组合按夏普取优，并列取回撤更小；max_drawdown 为该组合在矩阵窗口上的回撤约束参考）。接受记录存于 settings.matrix_applied，与手动记忆分开保存。">` +
    `🎯 矩阵最优：<b>${escHtml(comboDisplayName(best.combo))}</b> · ${sh} · ${ddS} · ${winTxt}` +
    badgeHtml +
    `</div>`;
  const applyBtn = $("btMatrixBestApplyBtn");
  if (applyBtn) applyBtn.addEventListener("click", btApplyMatrixBest);
  // 不再自动应用/自动恢复：只有打开网页时按记忆预填（restoreBtPrefs），绝不因
  // 切模型/点「开始回测」/重启服务而把参数改回矩阵最优；「应用为新回测默认」按钮保持显式可用。
  __btMatrixUserApply = false;
}

// ── 三轴联合回测（合并原 A/B·持仓方案对比 / 阈值敏感性·观望带）──────────
// 见下方 runHoldMatrix / loadComboGrid —— 22 组合 × 上限%档 × 阈值档 全网格子进程任务

// ── 成本敏感性（真实 edge）：同一方案在 0→高成本档位下重跑 ────────────
let btCostChart = null;

function bindCostSweep() {
  const btn = $("btCostRunBtn");
  if (!btn || btn.dataset.bound) return;
  btn.dataset.bound = "1";
  btn.addEventListener("click", runCostSweep);
}

function costSweepReq() {
  const costs = readBacktestCosts();
  const win = Number($("btWindowInput")?.value);
  const pos = Number($("btMaxPosInput")?.value);
  return {
    strategy_file: selectedStrategyFile,
    data_file: $("btDataSelect")?.value || null,
    hold_policy: (btEffectivePolicy ? btEffectivePolicy().id : null) || "signal",
    commission_pct: costs.commission_pct,
    slippage_pct: costs.slippage_pct,
    window_bars: Number.isFinite(win) && win >= 800 ? Math.round(win) : null,
    max_position_pct: Number.isFinite(pos) && pos >= 1 ? pos : 100,
    signal_threshold: Number($("btThresholdSelect")?.value) || 0.05,
  };
}

async function runCostSweep() {
  const stats = $("btCostStats");
  const hint = $("btCostSweepHint");
  const btn = $("btCostRunBtn");
  if (!selectedStrategyFile) {
    if (stats) stats.innerHTML = `<span class="bad">请先在回测页选择策略文件</span>`;
    return;
  }
  if (btn) btn.disabled = true;
  if (hint) hint.textContent = "正在跑（13 个成本档 × 同一次因子）…";
  try {
    const res = await fetchJSON("/api/backtest/cost-sweep", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(costSweepReq()),
    });
    if (!res.ok) throw new Error(res.error || "cost-sweep failed");
    renderCostSweep(res);
  } catch (e) {
    if (stats) stats.innerHTML = `<span class="bad">成本敏感性失败：${escHtml(e && e.message ? e.message : e)}</span>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderCostSweep(d) {
  const stats = $("btCostStats");
  const hint = $("btCostSweepHint");
  const rows = d.rows || [];
  const cur = d.current_cost_pct;
  const fmtC = (v) => (v == null ? "—" : Number(v).toFixed(3) + "%");
  const fmtR = (v) => (v == null ? "—" : ((v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%"));
  const beR = d.break_even_return;
  const beP = d.break_even_plr;
  if (stats) {
    const parts = [
      `方案 <b>${escHtml(d.policy_label || d.policy_id)}</b> · ${rows.length} 档 · 当前每边成本 <b>${fmtC(cur)}</b>`,
    ];
    parts.push(
      beR != null
        ? `净收益由正转负 ≈ <b class="good">${fmtC(beR)}</b>/边`
        : (rows.length && rows[0].total_return != null && rows[0].total_return <= 0
          ? "净收益在所有成本档均 ≤0（模型无正 edge）"
          : "全程净收益 ≥0（该窗口未触到成本上限）")
    );
    if (beP != null) parts.push(`盈亏比跌破 1 ≈ ${fmtC(beP)}/边`);
    stats.innerHTML = parts.join(" · ") + ` — 超过此成本即吃掉全部 alpha；低于它才是真实可行区间。`;
  }
  if (hint) hint.textContent = `每边总成本（手续费+滑点）· 曲线 ${d.bars} 根 · 当前点已标 ⭐`;
  const cv = $("btCostChart");
  if (!cv || !window.Chart) return;
  const bg = "rgba(128,140,160,0.08)";
  const pts = rows.filter((r) => r.total_return != null && r.profit_loss_ratio != null);
  const lineR = pts.map((r) => ({ x: r.cost_pct, y: r.total_return * 100 }));
  const lineP = pts.map((r) => ({ x: r.cost_pct, y: r.profit_loss_ratio }));
  // 当前成本处净收益的线性插值 + 盈亏平衡标记
  let curRet = null;
  for (let i = 0; i < rows.length - 1; i++) {
    if (rows[i].cost_pct <= cur && cur <= rows[i + 1].cost_pct) {
      const a = rows[i], b = rows[i + 1];
      if (a.total_return != null && b.total_return != null && b.cost_pct > a.cost_pct) {
        const f = (cur - a.cost_pct) / (b.cost_pct - a.cost_pct);
        curRet = (a.total_return + f * (b.total_return - a.total_return)) * 100;
      }
      break;
    }
  }
  if (curRet == null && rows.length) curRet = rows[0].total_return != null ? rows[0].total_return * 100 : null;
  const markers = [];
  if (curRet != null) markers.push({ x: cur, y: curRet, style: "star", color: "#5eead4", label: "当前成本" });
  if (beR != null) markers.push({ x: Number(beR), y: 0, style: "triangle", color: "#e74c3c", label: "盈亏平衡" });
  if (btCostChart) {
    btCostChart.destroy();
    btCostChart = null;
  }
  btCostChart = new Chart(cv.getContext("2d"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "净收益 %",
          data: lineR,
          borderColor: "#56a4f0",
          backgroundColor: "rgba(86,164,240,0.08)",
          borderWidth: 1.8,
          pointRadius: 2.5,
          tension: 0.2,
          fill: true,
          yAxisID: "y",
        },
        {
          label: "盈亏比",
          data: lineP,
          borderColor: "#f0c25e",
          borderWidth: 1.6,
          pointRadius: 2.5,
          borderDash: [5, 4],
          tension: 0.2,
          fill: false,
          yAxisID: "y1",
        },
        ...markers.map((m) => ({
          label: m.label,
          data: [{ x: m.x, y: m.y }],
          showLine: false,
          pointRadius: 7,
          pointStyle: m.style,
          pointBackgroundColor: m.color,
          pointHoverRadius: 9,
        })),
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { color: "#c8d3e0", boxWidth: 14, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (c) => {
              if (c.dataset.label === "当前成本") return `当前成本 ${fmtC(c.parsed.x)} · 净收益 ${fmtR(c.parsed.y / 100)}`;
              if (c.dataset.label === "盈亏平衡") return `盈亏平衡：每边总成本 ${fmtC(c.parsed.x)}`;
              if (c.dataset.label === "净收益 %") return `净收益 ${fmtR(c.parsed.y / 100)} @ ${fmtC(c.parsed.x)}`;
              return `盈亏比 ${Number(c.parsed.y).toFixed(2)} @ ${fmtC(c.parsed.x)}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "每边总成本（手续费+滑点 %）", color: "#93a4b8" },
          grid: { color: bg },
          ticks: { color: "#93a4b8", callback: (v) => Number(v).toFixed(3) },
        },
        y: { position: "left", title: { display: true, text: "净收益 %", color: "#56a4f0" }, grid: { color: bg }, ticks: { color: "#93a4b8", callback: (v) => v + "%" } },
        y1: { position: "right", title: { display: true, text: "盈亏比", color: "#f0c25e" }, grid: { drawOnChartArea: false }, ticks: { color: "#93a4b8" } },
      },
    },
  });
}

function abCell(v, digits, pct) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return pct ? (n * 100).toFixed(2) + "%" : (digits ? n.toFixed(digits) : String(n));
}

// ── tail vs spread 对比摘要（回测页默认模型下方）──────────────────────────
let __btCompare = null;
let __btCompareInFlight = false;

async function btLoadCompare() {
  const box = $("btCompare");
  if (!box) return;
  const sym = selectedStrategySymbol;
  if (__btCompareInFlight) return;
  __btCompareInFlight = true;
  let data = null;
  try {
    data = await fetchJSON(
      "/api/backtest/compare-summary?symbol=" + encodeURIComponent(sym || ""),
      { silent: true, retries: 0 }
    );
  } catch (_) {
    data = null;
  }
  __btCompareInFlight = false;
  if (!data || !data.exists || !data.summary) {
    box.innerHTML = "";
    __btCompare = null;
    return;
  }
  __btCompare = data.summary;
  const s = data.summary;
  const vCell = (v) => {
    if (!v) return null;
    const gateTxt = v.passed ? "holdout ✓" : v.passed === false ? "holdout ✗" : "—";
    const parts = [
      v.seed != null ? `代表seed ${v.seed}` : "",
      v.holdout_val != null ? `holdout ${Number(v.holdout_val).toFixed(3)}` : "",
      v.median_holdout != null ? `中位 ${Number(v.median_holdout).toFixed(3)}` : "",
      v.sharpe != null ? `Sharpe ${Number(v.sharpe).toFixed(2)}` : "",
      v.score_ratio != null ? `ratio ${Number(v.score_ratio).toFixed(2)}` : "",
    ].filter(Boolean);
    return { gate: gateTxt, parts };
  };
  const mk = (label, v) => {
    const c = vCell(v);
    if (!c) return "";
    const color = v.passed ? "var(--ok, #4ade80)" : v.passed === false ? "#f87171" : "#7a8a9e";
    return (
      `<div class="bt-cmp-var"><b>${escHtml(label)}</b>` +
      `<span class="bt-cmp-gate" style="color:${color}">${c.gate}</span>` +
      `<span class="bt-cmp-meta">${c.parts.map(escHtml).join(" · ")}</span></div>`
    );
  };
  const mini = s.mini
    ? `<span class="bt-cmp-mini" title="steps<100：仅供方法/管线演示，不能作为正式选型结论；正式需 ≥100 步 + 多 seed 全量跑">迷你(steps=${s.params?.steps})</span>`
    : "";
  const crit = (s.params && s.params.rep_criterion) || "holdout_median";
  const critLabel = {
    in_sample: "in-sample best",
    holdout_median: "跨 seed 中位 holdout",
    holdout_best: "holdout best",
  };
  const basis = `<span class="bt-cmp-mini" title="结论代表口径：in-sample best / 跨 seed 中位 holdout（默认）/ holdout best；报告与下方分口径对照同时给出 in-sample 与 holdout 两种口径的结论">口径：${escHtml(critLabel[crit] || crit)}</span>`;
  const concl = Array.isArray(s.conclusion) && s.conclusion.length
    ? `<span class="bt-cmp-concl">结论：${s.conclusion.map(escHtml).join("；")}</span>`
    : "";
  // 分口径对照（in-sample vs holdout 可能给出相反结论）
  const cc = s.conclusions_by_criterion || {};
  const ccChips = Object.keys(cc).length
    ? `<div class="bt-cmp-row bt-cmp-crit"><b>分口径结论对照：</b>` +
      Object.entries(cc)
        .map(([k, ls]) => `<span class="exp-crit-chip${k === crit ? " chosen" : ""}" title="${escHtml(critLabel[k] || k)}">${escHtml(critLabel[k] || k)}：${escHtml((ls && ls[0]) || "—")}</span>`)
        .join(" ") +
      `</div>`
    : "";
  // 两路代表 run 的 holdout 逐根资金曲线叠加
  const curveHtml = btCompareHoldoutOverlay(s);
  box.innerHTML =
    `<div class="bt-cmp-box">` +
    `<div class="bt-cmp-head">📄 tail vs spread 范围对比 ${mini}${basis}${s.symbol ? ` · ${escHtml(s.symbol)}` : ""}</div>` +
    mk("tail（最近N）", s.tail) + mk("spread（分块）", s.spread) +
    (concl ? `<div class="bt-cmp-row">${concl}</div>` : "") +
    ccChips +
    curveHtml +
    `<details class="bt-cmp-report"><summary>查看完整 docs 报告</summary>` +
    `<pre>${escHtml(data.markdown || "（暂无报告内容）")}</pre></details>` +
    `</div>`;
}

// 两路代表 holdout 资金曲线叠加（tail cyan vs spread amber，同一 holdout bar 轴）
function btCompareHoldoutOverlay(s) {
  const mkds = (v, color) => {
    const hc = v && v.holdout_curve;
    if (!hc || !hc.labels || !hc.equity || hc.labels.length < 2) return null;
    return {
      labels: hc.labels.map(Number),
      equity: hc.equity.map(Number),
      color,
    };
  };
  const dss = [mkds(s.tail, "#38bdf8"), mkds(s.spread, "#fbbf24")].filter(Boolean);
  if (dss.length < 2) return "";
  // 统一到同一 bar 轴（取交集标签；一般同 holdout 窗口）
  let lo = Infinity, hi = -Infinity;
  dss.forEach((d) => d.equity.forEach((y) => { if (y < lo) lo = y; if (y > hi) hi = y; }));
  const pad = (hi - lo) * 0.1 || 1e-9;
  lo -= pad; hi += pad;
  const W = 640, H = 150, L = 46, R = 12, T = 10, B = 22;
  const xmax = Math.max(...dss.map((d) => d.labels[d.labels.length - 1]));
  const xmin = Math.min(...dss.map((d) => d.labels[0]));
  const X = (x) => L + ((x - xmin) / Math.max(1, xmax - xmin)) * (W - L - R);
  const Y = (y) => T + ((hi - y) / (hi - lo)) * (H - T - B);
  const path = (d) =>
    d.labels.map((x, i) => `${i === 0 ? "M" : "L"}${X(x).toFixed(1)},${Y(d.equity[i]).toFixed(1)}`).join(" ");
  return `<div class="bt-cmp-curve"><b>两路代表 holdout 资金曲线叠加</b>` +
    `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px" preserveAspectRatio="none">` +
    dss.map((d, i) => `<path d="${path(d)}" fill="none" stroke="${d.color}" stroke-width="1.6" ${i === 1 ? 'stroke-dasharray="5 4"' : ""}/>`).join("") +
    `<line x1="${L}" y1="${Y(0)}" x2="${W - R}" y2="${Y(0)}" stroke="#4b5563" stroke-width="1" stroke-dasharray="2 3"/>` +
    `</svg>` +
    `<div class="bt-cmp-legend"><span style="color:#38bdf8">─ tail holdout</span><span style="color:#fbbf24">┄ spread holdout</span><span>（0 线 = 起点净值）</span></div>` +
    `</div>`;
}

function btApplyMatrixBest() {
  const b = __btMatrixBest;
  if (!b || !b.combo) return;
  const parts = b.combo.split("+").filter(Boolean);
  const base = $("btPolicySelect");
  const stk = $("btStackSelect");
  if (!base || !stk) return;
  let ok = false;
  if (
    parts.length >= 2 &&
    base.querySelector(`option[value="${parts[0]}"]`) &&
    stk.querySelector(`option[value="${parts[1]}"]`)
  ) {
    base.value = parts[0];
    stk.value = parts[1];
    ok = true;
  } else if (base.querySelector(`option[value="${b.combo}"]`)) {
    base.value = b.combo;
    stk.value = "";
    ok = true;
  }
  if (!ok) return;
  updateBtComboHint();
  scheduleBtPrefsSave();
  // 记录“用户接受”来源（settings.matrix_applied，区别于手动记忆）——重启后芯片仍显示「已应用」
  persistMatrixApplied(selectedStrategySymbol, b);
  __btMatrixUserApply = true; // 内部 reload 不再走「自动恢复」分支/通知
  btLoadMatrixBest();
  btLoadCompare(); // 重渲染为「已作为新回测默认」
}

// ── 冠军回滚校验（训练页「回滚校验」按钮 → verify_champion_rollback）────
let __vrfTimer = null;
let __vrfFile = "";
let __vrfRetrain = false;

async function vrfStart(strategyFile, withRetrain) {
  if (!strategyFile) return;
  __vrfFile = strategyFile;
  __vrfRetrain = !!withRetrain;
  const body = $("vrfBody");
  const hint = $("vrfHint");
  const retr = $("vrfRetrainBtn");
  if (body) body.textContent = "正在启动校验…";
  if (hint) hint.textContent = "提交任务中…";
  if (retr) retr.disabled = true;
  $("vrfModal").hidden = false;
  try {
    const res = await fetchJSON("/api/training/verify-rollback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        strategy_file: strategyFile,
        retrain_steps: withRetrain ? 10 : 0,
      }),
      retries: 0,
    });
    if (res && res.ok) {
      if (hint) hint.textContent = withRetrain
        ? "含短程重训（10 步/记录），较慢；先做确定性+holdout 复验，再重训对照…"
        : "校验运行中：① 子集确定性（重生成 vs 存档逐行对比）② 存档公式 holdout 复验…";
      vrfPoll();
    } else {
      if (body) body.textContent = "启动失败: " + JSON.stringify(res || "无响应");
      if (retr) retr.disabled = false;
    }
  } catch (e) {
    if (body) body.textContent = `启动失败: ${escHtml(e && e.message ? e.message : e)}`;
    if (retr) retr.disabled = false;
  }
}

async function vrfPoll() {
  let st;
  try {
    st = await fetchJSON("/api/training/verify-rollback/status", { silent: true, retries: 0 });
  } catch (_) {
    st = null;
  }
  const body = $("vrfBody");
  if (!body) return;
  const job = st && st.job;
  const retr = $("vrfRetrainBtn");
  if (!job || !st.active) {
    // 完成/失败：展示报告
    if (job && st.report && st.report.markdown) {
      body.textContent = st.report.markdown;
    } else if (st && st.log_tail && st.log_tail.length) {
      body.textContent = (st.log_tail || []).join("\n");
    } else {
      body.textContent = "任务已结束，但未找到报告/日志输出。";
    }
    if (retr) retr.disabled = false;
    return;
  }
  // 运行中：轮询
  const tail = (st.log_tail || []).slice(-8).join("\n");
  body.textContent = "校验运行中（自动刷新）…\n\n" + (tail || "（等待输出…）");
  body.scrollTop = body.scrollHeight;
  __vrfTimer = setTimeout(vrfPoll, 2500);
}

function vrfClose() {
  if (__vrfTimer) {
    clearTimeout(__vrfTimer);
    __vrfTimer = null;
  }
  const m = $("vrfModal");
  if (m) m.hidden = true;
}

function bindVerifyRollback() {
  const tbody = $("strategiesBody");
  if (tbody) {
    tbody.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-vrf-file]");
      if (btn) vrfStart(btn.dataset.vrfFile, false);
    });
  }
  document.querySelectorAll("[data-close-vrf]").forEach((el) => {
    el.addEventListener("click", vrfClose);
  });
  const closeBtn = $("vrfCloseBtn");
  if (closeBtn) closeBtn.addEventListener("click", vrfClose);
  const retr = $("vrfRetrainBtn");
  if (retr) retr.addEventListener("click", () => {
    if (__vrfFile) vrfStart(__vrfFile, true);
  });
}

// 把同一份记忆的主方案/窗口/投入上限 复用到其他页（实时分析/模拟实盘/历史回放）
function applyBtPrefsToOthers(hp, wb, mpp) {
  if (!hp || hp === "signal") return; // 其他页默认即信号跟随
  const parts = hp.split("+").filter(Boolean);
  const baseVal = parts[0];
  const stkVal = parts[1] || "";
  ["rtPolicySelect", "ppPolicySelect", "rsPolicySelect"].forEach((id) => {
    const s = $(id);
    if (s && [...s.options].some((o) => o.value === baseVal)) s.value = baseVal;
  });
  ["rtStackSelect", "ppStackSelect"].forEach((id) => {
    const s = $(id);
    if (!s) return;
    s.value = stkVal && [...s.options].some((o) => o.value === stkVal) ? stkVal : "";
  });
  updateStackHint("rtPolicyHint", "rtPolicySelect", "rtStackSelect");
  updateStackHint("ppPolicyHint", "ppPolicySelect", "ppStackSelect");
  const rw = $("rsWindowInput");
  if (rw && !rw.value && Number.isFinite(wb) && wb >= 800) rw.value = String(Math.round(wb));
  const rmp = $("rsMaxPosInput");
  if (rmp && Number.isFinite(Number(mpp)) && Number(mpp) >= 1) {
    rmp.value = String(Math.round(Math.min(200, Math.max(1, Number(mpp)))));
  }
}

async function restoreBtPrefs() {
  let s;
  try {
    s = await fetchJSON("/api/settings", { silent: true, retries: 0 });
  } catch (_) {
    return;
  }
  const hp = s.bt_hold_policy;
  const wb = Number(s.bt_window_bars);
  __btApplyingPrefs = true;
  try {
    const base = $("btPolicySelect");
    const stk = $("btStackSelect");
    if (hp && hp !== "signal" && base) {
      const parts = hp.split("+").filter(Boolean);
      if (
        parts.length >= 2 &&
        base.querySelector(`option[value="${parts[0]}"]`) &&
        stk &&
        stk.querySelector(`option[value="${parts[1]}"]`)
      ) {
        base.value = parts[0];
        stk.value = parts[1];
      } else if (base.querySelector(`option[value="${hp}"]`)) {
        base.value = hp;
        if (stk) stk.value = "";
      }
      updateBtComboHint();
    }
    if (Number.isFinite(wb) && wb >= 800) {
      const wEl = $("btWindowInput");
      if (wEl) wEl.value = String(Math.round(wb));
      const mxW = $("btMatrixWindowInput");
      if (mxW && !mxW.value) mxW.value = String(Math.round(wb)); // 全组合矩阵沿用最近一次样本外窗口
    }
    const mp = Number(s.max_position_pct);
    const mpEl = $("btMaxPosInput");
    if (mpEl && Number.isFinite(mp) && mp >= 1) mpEl.value = String(Math.round(mp));
    const thr = Number(s.signal_threshold);
    if (Number.isFinite(thr) && thr > 0) applySigThrSelects(thr);
  } finally {
    __btApplyingPrefs = false;
  }
  window.__btMemPolicy = hp && hp !== "signal" ? hp : "signal";
  window.__btMemWindow = Number.isFinite(wb) && wb >= 800 ? Math.round(wb) : null;
  window.__btMemMaxPos = Number.isFinite(Number(s.max_position_pct)) ? Number(s.max_position_pct) : 100;
  renderBtLastCombo();
  applyBtPrefsToOthers(window.__btMemPolicy, window.__btMemWindow, window.__btMemMaxPos);
  btLoadMatrixBest();
  btLoadCompare();
}

// 统一无信号阈值：四个选择框双向同步（回测/回放/实时/模拟实盘共用同一设置）
function applySigThrSelects(thr) {
  const v = String(Number(thr) || 0.05);
  ["btThresholdSelect", "rsThresholdSelect", "rtThresholdSelect", "ppThresholdSelect"].forEach((id) => {
    const el = $(id);
    if (!el || el.value === v) return;
    if ([...el.options].some((o) => o.value === v)) el.value = v;
  });
}

// 观望带角标：因子已连续 N 根处于观望区（|tanh(因子)| < 阈值），
// 显示当前强度相对阈值的百分比与走势（↑=逼近出观望，↓=越陷越深），
// 接近阈值边缘且回升时提示“即将出观望”。
function rtWatchZoneBadge(w, sq) {
  if (!w || w.state !== "ok" || w.factor_value == null) return "";
  const thr = Number(w.threshold != null ? w.threshold : 0.05) || 0.05;
  const hist = Array.isArray(sq && sq.flip_margin_hist)
    ? sq.flip_margin_hist.map(Number).filter(Number.isFinite)
    : [];
  let run = 0; // 连续处于观望带（裕度<0）的根数
  for (let i = hist.length - 1; i >= 0 && hist[i] < 0; i--) run++;
  if (run < 3) return "";
  const pos = Math.abs(Math.tanh(Number(w.factor_value)));
  const ratio = pos / thr; // <1 = 带内；越接近 1 越接近出观望
  const last = hist[hist.length - 1];
  const prev = hist.length >= 2 ? hist[hist.length - 2] : last;
  const rising = last > prev;
  const pct = Math.min(100, Math.round(ratio * 100));
  const reentry = ratio >= 0.8 && rising;
  const tip = reentry
    ? "因子强度已回升至阈值边缘（≥80%），若下一根继续走强将重新出方向"
    : "因子持续处于观望区：|tanh(因子)| < 阈值，暂无方向（↑=回升逼近出观望，↓=越陷越深）";
  return `<span class="rt-watch-run${reentry ? " near" : ""}" title="${tip}">观望 ×${run} · ${pct}%${rising ? " ↑" : " ↓"}</span>`;
}

function rtPctSparkline(hist) {
  // 50 根「因子历史分位」迷你趋势：0-100 的百分位走向。
  // 分位持续走高 = 因子越来越极端（逼近翻转阈值）；回落 = 降温。
  const pts = Array.isArray(hist) ? hist.filter((v) => Number.isFinite(Number(v))).map(Number) : [];
  if (pts.length < 3) return "";
  const view = pts.slice(-50);
  const w = 96, h = 18, pad = 1;
  const min = 0, max = 100;
  const x = (i) => pad + (i / Math.max(1, view.length - 1)) * (w - 2 * pad);
  const y = (v) => pad + (1 - (v - min) / (max - min)) * (h - 2 * pad);
  const line = view.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const last = view[view.length - 1];
  const hot = last >= 90 || last <= 10;
  return `<span class="rt-pct-spark" title="因子历史分位趋势（最近 ${view.length} 根收盘，0-100）：分位逼近 100 或 0 = 因子处于历史极端，可能逼近翻转">
    <svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
      <line x1="${pad}" y1="${y(50).toFixed(1)}" x2="${w - pad}" y2="${y(50).toFixed(1)}" class="rt-spark-mid"/>
      <path d="${line}" class="rt-spark-line${hot ? " rt-spark-hot" : ""}" fill="none"/>
      <circle cx="${x(view.length - 1).toFixed(1)}" cy="${y(last).toFixed(1)}" r="1.6" class="rt-spark-dot${hot ? " rt-spark-hot" : ""}"/>
    </svg>
  </span>`;
}

function rtFlipSparkline(hist, threshold) {
  // 50 根「距翻转裕度」迷你趋势：横线=0（阈值带边缘），柱在 0 下方=观望带内；
  // 近端（最后 5 根）下行且接近 0 → 高亮，提示因子正朝阈值逼近
  const pts = Array.isArray(hist) ? hist.filter((v) => Number.isFinite(Number(v))).map(Number) : [];
  if (pts.length < 3) return "";
  const N = 50;
  const view = pts.slice(-N);
  const W = 120, H = 22, PAD = 2;
  const maxAbs = Math.max(0.05, ...view.map((v) => Math.abs(v)));
  const y0 = H / 2; // 裕度 0 线
  const scale = (H / 2 - PAD) / maxAbs;
  const step = W / Math.max(1, N - 1);
  const x = (i) => i * step;
  const y = (v) => y0 - v * scale;
  const path = view.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const last = view[view.length - 1];
  const near = last >= 0 && last < (Number(threshold) || 0.05) * 3;
  const falling = view.length >= 4 && last < view[view.length - 4];
  const cls = `rt-flip-spark${near && falling ? " rt-flip-spark-hot" : ""}`;
  const lastDot = `<circle cx="${x(view.length - 1).toFixed(1)}" cy="${y(last).toFixed(1)}" r="2" fill="currentColor"/>`;
  const tip = `最近 ${view.length} 根收盘的「距翻转裕度」（|因子|−阈值；横线=0 即阈值带边缘，柱在下方=观望带内）。${near && falling ? "近端正下行逼近 0 —— 可能即将翻向。" : "趋势向下=因子正朝阈值逼近。"}`;
  return `<span class="${cls}" title="${escHtml(tip)}"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true">` +
    `<line x1="0" y1="${y0}" x2="${W}" y2="${y0}" class="rt-flip-spark-zero"/>` +
    `<path d="${path}" fill="none" stroke="currentColor" stroke-width="1.2"/>${lastDot}</svg></span>`;
}

function policyName(id) {
  const p = holdPolicyById[id] || holdPolicyList.find((x) => x.id === id);
  return p ? p.name : id === "signal" ? "信号跟随" : id || "信号跟随";
}

// DD 门控状态角标：熔断中 / 深档熔断中 / 正常（方案叠加了 dd 时显示）
function ddTagHtml(w) {
  if (!w || !w.policy_id) return "";
  if (!String(w.policy_id).split("+").includes("dd")) return "";
  const g = Number(w.dd_gate) || 0;
  const pct = w.dd_pct != null ? ` · 回撤 ${Number(w.dd_pct).toFixed(2)}%` : "";
  const peakTxt = w.dd_peak != null ? `峰值 ${fmtNum(w.dd_peak, 6)}` : "峰值 —";
  if (g === 2) {
    return `<span class="rt-dd rt-dd-deep" title="深档熔断中：暂停新开仓，需收复到回撤 ≤ 阈值才恢复（${peakTxt}${pct}）">⛔ 深档熔断中${pct}</span>`;
  }
  if (g === 1) {
    return `<span class="rt-dd rt-dd-gated" title="熔断中：暂停新开仓，需收复到回撤 ≤ 阈值才恢复（${peakTxt}${pct}）">⛔ 熔断中${pct}</span>`;
  }
  return `<span class="rt-dd rt-dd-ok" title="DD 门控正常 · ${peakTxt}${pct}">DD 正常</span>`;
}

// ── DD 事件回溯小面板（模拟实盘 / 实时分析页共用）────────────────────
// 读 /api/dd-events（logs/dd_gate_events.jsonl，最近 N 条熔断/深档/收复），
// 时间线展示 + 一键导出 CSV。suffix ∈ {"Rt", "Pp"}。
const DD_EV_CLS = { "熔断": "gated", "深档熔断": "deep", "收复": "ok" };
const DD_EV_LAST = {}; // suffix -> { ts, sig, events }
const DD_EV_DISP_MAX = 40; // 过滤后列表最多渲染条数（避免超长时间线）
const DD_EV_STD_TYPES = ["熔断", "深档熔断", "收复"];
// 每面板的过滤状态 { sym: "", scope: "", type: "" }（type 取值同 event，"" = 全部）
const DD_EV_FILT = { Rt: { sym: "", scope: "", type: "" }, Pp: { sym: "", scope: "", type: "" } };
const DD_EV_SCOPE_LABEL = { paper: "模拟盘", realtime: "实时分析" };

function _ddEvSig(events) {
  if (!events.length) return "0";
  return events
    .map((e) => [e.iso || e.ts, e.event, e.symbol, e.timeframe, e.scope, e.dd_pct, e.policy_id].join("~"))
    .join("|");
}

function ddEvLocalTime(e) {
  const ts = e.iso ? Date.parse(e.iso) : Number(e.ts) * 1000;
  if (Number.isFinite(ts)) return new Date(ts).toLocaleString();
  return String(e.iso || "");
}

function ddEvRow(e) {
  const cls = DD_EV_CLS[e.event] || "gated";
  const scope = e.scope === "paper" ? "模拟盘" : e.scope === "realtime" ? "实时分析" : e.scope || "";
  const pct = e.dd_pct != null ? `回撤 ${Number(e.dd_pct).toFixed(2)}%` : "";
  const px = e.mark != null ? `价 ${fmtNum(e.mark, 6)}` : "";
  const pk = e.peak != null ? `峰值 ${fmtNum(e.peak, 6)}` : "";
  const pol = e.policy_id ? comboDisplayName(e.policy_id) : "";
  const detail = e.detail ? ` · ${e.detail}` : "";
  const title = `${ddEvLocalTime(e)} · ${e.symbol || ""} ${e.timeframe || ""} · ${pol} · ${pct}${px ? " · " + px : ""}${pk ? " · " + pk : ""}${detail}`;
  return `<div class="dd-ev dd-ev-${cls}" title="${escHtml(title)}">` +
    `<span class="dd-ev-time">${escHtml(ddEvLocalTime(e))}</span>` +
    `<span class="dd-ev-badge dd-ev-badge-${cls}">${escHtml(e.event || "")}</span>` +
    `<span class="dd-ev-sym">${escHtml(e.symbol || "")} ${escHtml(e.timeframe || "")}</span>` +
    (pol ? `<span class="dd-ev-pol" title="持仓方案">${escHtml(pol)}</span>` : "") +
    `<span class="dd-ev-meta">${pct}${px ? " · " + px : ""}${pk ? " · " + pk : ""}</span>` +
    (scope ? `<span class="dd-ev-scope">${escHtml(scope)}</span>` : "") +
    `</div>`;
}

// 过滤基集：全部已加载事件再按 品种+范围 收窄（类型不在内，供类型计数用）
function ddEvBase(suffix) {
  const all = (DD_EV_LAST[suffix] && DD_EV_LAST[suffix].events) || [];
  const f = DD_EV_FILT[suffix];
  return all.filter((e) => {
    if (f.sym && e.symbol !== f.sym) return false;
    if (f.scope && e.scope !== f.scope) return false;
    return true;
  });
}

// 过滤后可见事件（全部条件），新→旧；导出用不限条数，列表渲染用 DISP_MAX 封顶
function ddEvFiltered(suffix, cap) {
  const f = DD_EV_FILT[suffix];
  const rows = ddEvBase(suffix).filter((e) => {
    if (f.type && e.event !== f.type) return false;
    return true;
  });
  return cap ? rows.slice(0, cap) : rows;
}

// 按事件类型高亮计数（在 品种+范围 过滤集内统计；全部类型恒为基集大小）
function ddEvCounts(suffix) {
  const base = ddEvBase(suffix);
  const types = DD_EV_STD_TYPES.concat(
    [...new Set(base.map((e) => e.event).filter(Boolean))].filter((t) => !DD_EV_STD_TYPES.includes(t))
  );
  return types.map((t) => ({ type: t, n: base.filter((e) => e.event === t).length }));
}

function ddEvRenderFilters(suffix) {
  const symSel = $(`ddEvSym${suffix}`);
  const cntBox = $(`ddEvCounts${suffix}`);
  const all = (DD_EV_LAST[suffix] && DD_EV_LAST[suffix].events) || [];
  // 品种下拉：来源于全部已加载事件，保持当前选择（若仍存在）
  if (symSel) {
    const syms = [...new Set(all.map((e) => e.symbol).filter(Boolean))].sort();
    const cur = DD_EV_FILT[suffix].sym;
    symSel.innerHTML =
      `<option value="">全部品种</option>` +
      syms.map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`).join("");
    if (cur && syms.includes(cur)) symSel.value = cur;
    else DD_EV_FILT[suffix].sym = "";
  }
  if (cntBox) {
    const f = DD_EV_FILT[suffix];
    const counts = ddEvCounts(suffix);
    const total = ddEvBase(suffix).length;
    const chip = (type, label, n) =>
      `<button type="button" class="dd-ev-chip dd-ev-chip-${DD_EV_CLS[type] || "gated"}${f.type === type ? " on" : ""}" data-ev-type="${escHtml(type)}" title="点击只显示该类型；再点一次回到全部">${escHtml(label)} <b>${n}</b></button>`;
    cntBox.innerHTML =
      `<span class="dd-ev-chip-all${f.type === "" ? " on" : ""}" title="全部类型">全部 <b>${total}</b></span>` +
      counts
        .map((c) => chip(c.type, c.type === "熔断" ? "熔断" : c.type === "深档熔断" ? "深档" : c.type === "收复" ? "收复" : c.type, c.n))
        .join("");
    // 绑定类型 chips
    cntBox.querySelectorAll("[data-ev-type]").forEach((b) => {
      b.addEventListener("click", () => {
        const t = b.dataset.evType;
        DD_EV_FILT[suffix].type = DD_EV_FILT[suffix].type === t ? "" : t;
        ddEvRenderFilters(suffix); // 重画计数高亮（不重新拉数据）
        ddEvRenderList(suffix);
      });
    });
  }
}

function ddEvRenderList(suffix) {
  const list = $(`ddEvList${suffix}`);
  if (!list) return;
  const f = DD_EV_FILT[suffix];
  const filtered = ddEvFiltered(suffix, DD_EV_DISP_MAX);
  const all = (DD_EV_LAST[suffix] && DD_EV_LAST[suffix].events) || [];
  let html;
  if (!all.length) {
    html = '<div class="metric-empty">暂无 DD 事件记录 — 模拟盘/实时卡的 DD 方案触发熔断或收复后，这里会出现时间线。</div>';
  } else if (!filtered.length) {
    html = `<div class="metric-empty">无匹配记录 — 当前过滤：品种「${f.sym || "全部"}」· 范围「${f.scope ? DD_EV_SCOPE_LABEL[f.scope] || f.scope : "全部"}」· 类型「${f.type || "全部"}」</div>`;
  } else {
    html = `<div class="dd-ev-col">` + filtered.map(ddEvRow).join("") + `</div>`;
  }
  list.innerHTML = html;
  const hint = $(`ddEvHint${suffix}`);
  if (hint) {
    const total = all.length;
    const shown = filtered.length;
    const filtTxt = (f.sym || f.scope || f.type)
      ? ` · 过滤: 品种${f.sym || "全部"} / 范围${f.scope ? DD_EV_SCOPE_LABEL[f.scope] || f.scope : "全部"} / 类型${f.type || "全部"}`
      : "";
    hint.textContent =
      `最近 ${total} 条${shown < total ? `（列表显示 ${shown}）` : ""} · logs/dd_gate_events.jsonl（15 秒自动刷新）${filtTxt}`;
  }
}

async function loadDdEventsPanel(suffix, force) {
  const list = $(`ddEvList${suffix}`);
  if (!list) return;
  const now = Date.now();
  const last = DD_EV_LAST[suffix];
  if (!force && last && now - last.ts < 15000) return;
  DD_EV_LAST[suffix] = { ts: now, sig: last ? last.sig : "", events: last ? last.events : [] };
  let events = [];
  try {
    const d = await fetchJSON("/api/dd-events?limit=200", { silent: true, retries: 0 });
    events = d.events || [];
  } catch (_) {
    return;
  }
  const sig = _ddEvSig(events);
  DD_EV_LAST[suffix].events = events;
  if (sig === DD_EV_LAST[suffix].sig) return; // 数据未变：过滤器变化由各自的 change 监听单独重画
  DD_EV_LAST[suffix].sig = sig;
  ddEvRenderFilters(suffix);
  ddEvRenderList(suffix);
}

function _csvCell(v) {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

async function exportDdEventsCsv(suffix) {
  let events = DD_EV_LAST[suffix] ? DD_EV_LAST[suffix].events : [];
  try {
    const d = await fetchJSON("/api/dd-events?limit=200", { silent: true, retries: 0 });
    events = d.events || [];
  } catch (_) {}
  const f = DD_EV_FILT[suffix] || { sym: "", scope: "", type: "" };
  // 导出 = 当前筛选条件生效（含类型），与面板所见一致
  const rows = events.filter((e) => {
    if (f.sym && e.symbol !== f.sym) return false;
    if (f.scope && e.scope !== f.scope) return false;
    if (f.type && e.event !== f.type) return false;
    return true;
  });
  const cols = ["ts", "iso", "scope", "symbol", "timeframe", "policy_id", "event", "dd_pct", "mark", "peak", "detail"];
  const filterNote = `筛选: 品种=${f.sym || "全部"} 范围=${f.scope ? DD_EV_SCOPE_LABEL[f.scope] || f.scope : "全部"} 类型=${f.type || "全部"}`;
  const head = cols.map(_csvCell).join(",");
  const body = rows
    .map((e) => cols.map((c) => _csvCell(e[c])).join(","))
    .join("\r\n");
  // 导出时把筛选条件同步带上：文件内注释行 + 文件名后缀
  const blob = new Blob(["\uFEFF# " + filterNote + "\r\n" + head + "\r\n" + body], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const tag = [f.sym, f.scope, f.type].filter(Boolean).join("_") || "all";
  a.download = `dd_events_${new Date().toISOString().slice(0, 10)}_${tag}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 3000);
}

function bindDdEvExport() {
  document.querySelectorAll("[data-dd-ev-export]").forEach((btn) => {
    if (btn.dataset.bound) return;
    btn.dataset.bound = "1";
    const suffix = btn.dataset.ddEvExport;
    btn.addEventListener("click", () => exportDdEventsCsv(suffix));
  });
  // 品种/范围下拉：变更即更新过滤状态并重画（不重新拉数据）
  ["Rt", "Pp"].forEach((suffix) => {
    const symSel = $(`ddEvSym${suffix}`);
    const scopeSel = $(`ddEvScope${suffix}`);
    if (!symSel || !scopeSel) return;
    symSel.addEventListener("change", () => {
      DD_EV_FILT[suffix].sym = symSel.value;
      ddEvRenderFilters(suffix);
      ddEvRenderList(suffix);
    });
    scopeSel.addEventListener("change", () => {
      DD_EV_FILT[suffix].scope = scopeSel.value;
      ddEvRenderFilters(suffix);
      ddEvRenderList(suffix);
    });
  });
  loadDdEventsPanel("Rt", true);
  loadDdEventsPanel("Pp", true);
}

function svgPriceChart(pts, lines, opts) {
  opts = opts || {};
  const maxPts = opts.maxPts || 160;
  const data = (pts || []).slice(-maxPts);
  if (data.length < 2) return "";
  const vals = data.map((d) => Number(d[1]));
  let lo = Math.min.apply(null, vals);
  let hi = Math.max.apply(null, vals);
  if (!(hi > lo)) {
    const m = hi || 1;
    lo = m * 0.99;
    hi = m * 1.01;
  }
  const pad = (hi - lo) * 0.08 || 1e-6;
  lo -= pad;
  hi += pad;
  const W = 100;
  const TOP = 6;
  const BOT = 94;
  const x = (i) => 1.5 + ((W - 30) * i) / (data.length - 1);
  const y = (v) => TOP + ((BOT - TOP) * (hi - v)) / (hi - lo);
  const d = data
    .map((p, i) => (i ? "L" : "M") + x(i).toFixed(2) + " " + y(Number(p[1])).toFixed(2))
    .join("");
  let html = `<svg class="rt-chart-svg" viewBox="0 0 ${W} 100" preserveAspectRatio="none">`;
  html += `<path d="${d}" fill="none" stroke="#56a4f0" stroke-width="1" vector-effect="non-scaling-stroke" opacity="0.95"/>`;
  const seen = {};
  (lines || []).forEach((ln) => {
    const p = Number(ln.price);
    if (!(p > 0) || p < lo || p > hi || seen[ln.label]) return;
    seen[ln.label] = true;
    const yy = y(p);
    const dash = ln.dash || "4 3";
    html +=
      `<line x1="1.5" y1="${yy.toFixed(2)}" x2="${(W - 28).toFixed(2)}" y2="${yy.toFixed(2)}" ` +
      `stroke="${ln.color}" stroke-width="0.9" stroke-dasharray="${dash}" vector-effect="non-scaling-stroke" opacity="0.95"/>` +
      `<text x="${(W - 26).toFixed(2)}" y="${(yy - 2).toFixed(2)}" font-size="5" fill="${ln.color}" font-family="'JetBrains Mono',monospace">${ln.label}</text>`;
  });
  html += "</svg>";
  return html;
}

// ── 实时卡价格图：已收盘 bar + 形成中实时价尾点 ────────────────────
function rtDevThr() {
  // 偏离入场告警阈值（%），读设置输入框（0 = 关闭）
  const el = $("rtAlertDevPct");
  const v = Number(el && el.value);
  return Number.isFinite(v) && v > 0 ? v : 0;
}

// 入场参考锚点：方向建立时锚定的收盘价 + 当前偏离百分比（与告警同口径）
function rtAnchorInfo(w) {
  if (!w) return null;
  const dir = w.direction;
  if (dir !== "LONG" && dir !== "SHORT") return null;
  const entry = w.dir_entry_price != null ? Number(w.dir_entry_price) : null;
  if (!entry || !(entry > 0)) return null;
  const ref =
    w.live_price != null ? Number(w.live_price) : w.last_close != null ? Number(w.last_close) : null;
  if (ref == null || !(ref > 0)) return null;
  return { entry, devPct: (ref / entry - 1) * 100, side: dir === "LONG" ? "多" : "空", thr: rtDevThr() };
}

// 入场锚 → 图上虚线（带方向+当前偏离标签；越线时红色提示告警已触发）
function rtChartLevels(w) {
  const actPos = w.paper_position || null;
  const out = [];
  if (actPos && actPos.entry_price) {
    out.push({ label: "实际入场", price: Number(actPos.entry_price), color: "#fbbf24", kind: "entry" });
    if (actPos.stop_price) out.push({ label: "实际止损", price: Number(actPos.stop_price), color: "#f87171" });
    if (actPos.target_price) out.push({ label: "实际止盈", price: Number(actPos.target_price), color: "#4ade80" });
  } else if (w.plan && w.plan.entry) {
    out.push({ label: "建议入场", price: Number(w.plan.entry), color: "#fbbf24", kind: "entry" });
    if (w.plan.stop_price) out.push({ label: "止损", price: Number(w.plan.stop_price), color: "#f87171" });
    if (w.plan.target_price) out.push({ label: "止盈", price: Number(w.plan.target_price), color: "#4ade80" });
  }
  const an = rtAnchorInfo(w);
  if (an) {
    // 与已在图上的入场价一致时不重复画（实际持仓入场/建议入场就是锚点时）
    const dup = out.some((l) => l.kind === "entry" && Math.abs(Number(l.price) - an.entry) < 1e-9);
    if (!dup) {
      const over = an.thr > 0 && Math.abs(an.devPct) >= an.thr;
      out.push({
        label: `${an.side}入场锚 ${fmtSigned(an.devPct, 2)}%`,
        price: an.entry,
        color: over ? "#f87171" : "#c4b5fd",
        dash: "2 3 7 3",
        kind: "anchor",
        over,
      });
    }
  }
  return out;
}

// 入场锚小条：偏离百分比 + 告警阈值（一眼可见；颜色随是否越线变化）
function rtAnchorChipHtml(w) {
  const a = rtAnchorInfo(w);
  if (!a) return "";
  const over = a.thr > 0 && Math.abs(a.devPct) >= a.thr;
  const thrTxt = a.thr > 0 ? ` · 告警阈值 ±${a.thr}%` : " · 偏离告警已关闭(0)";
  const tip =
    (over
      ? `价格偏离 ${fmtSigned(a.devPct, 2)}% 已越线（阈值 ±${a.thr}%）——该监控会推送飞书；回到阈值一半以内才允许再次提醒`
      : `入场参考 = ${a.side === "多" ? "LONG" : "SHORT"} 方向建立时锚定的收盘价 ${fmtPrice(a.entry)}；当前偏离 ${fmtSigned(a.devPct, 2)}%，与飞书偏离告警同口径`);
  return `<div class="rt-anchor${over ? " over" : ""}" data-rt-anchor-id="${escHtml(w.id)}" title="${escHtml(tip)}">` +
    `入场锚(${a.side}) ${fmtPrice(a.entry)} · 当前偏离 <b>${fmtSigned(a.devPct, 2)}%</b>${thrTxt}` +
    `</div>` +
    rtAnchorDevHistHtml(w, a);
}

// 入场锚偏离历史迷你图：每根收盘 bar 记一次价格相对锚的偏离%，画最近 ≤50 根。
// 一眼看出是 缓涨慢慢越过阈值 还是 单根跳穿：+/-阈值画虚线，末点越线时红标并注明成因。
function rtAnchorDevHistHtml(w, a) {
  const hist = Array.isArray(w && w.anchor_dev_hist)
    ? w.anchor_dev_hist.map(Number).filter(Number.isFinite)
    : [];
  if (!a || hist.length < 1) return "";
  const thr = a.thr || 0;
  let lo = Math.min.apply(null, hist);
  let hi = Math.max.apply(null, hist);
  if (thr > 0) { lo = Math.min(lo, -thr); hi = Math.max(hi, thr); }
  if (!(hi > lo)) { const m = Math.max(Math.abs(hi), Math.abs(lo), 1e-6); lo = -m; hi = m; }
  const pad = (hi - lo) * 0.15 || 0.05;
  lo -= pad; hi += pad;
  const W = 100, T = 4, B = 26;
  const n = hist.length;
  const x = (i) => 1.5 + ((W - 8) * i) / Math.max(1, n - 1);
  const y = (v) => T + ((B - T) * (hi - v)) / (hi - lo);
  const d = hist.map((v, i) => (i ? "L" : "M") + x(i).toFixed(2) + " " + y(v).toFixed(2)).join("");
  const zeroY = y(0).toFixed(2);
  // 末点是否越线 + 成因（单根跳穿 vs 缓涨越过）
  const last = hist[hist.length - 1];
  const prev = hist.length >= 2 ? hist[hist.length - 2] : null;
  const crossed = thr > 0 && Math.abs(last) >= thr;
  const jump =
    crossed && prev != null && Math.abs(prev) < thr
      ? "单根跳穿"
      : crossed
        ? "缓涨越过（已在阈值外持续）"
        : null;
  const yT = (v) => y(v).toFixed(2);
  const thrLines =
    thr > 0
      ? `<line x1="1.5" y1="${yT(thr)}" x2="${(W - 6).toFixed(2)}" y2="${yT(thr)}" stroke="rgba(239,68,68,0.5)" stroke-width="0.5" stroke-dasharray="2 2"/>` +
        `<line x1="1.5" y1="${yT(-thr)}" x2="${(W - 6).toFixed(2)}" y2="${yT(-thr)}" stroke="rgba(239,68,68,0.5)" stroke-width="0.5" stroke-dasharray="2 2"/>`
      : "";
  const col = crossed ? "#f87171" : "#c4b5fd";
  const lastX = x(n - 1).toFixed(2);
  const lastY = y(last).toFixed(2);
  const dot = crossed
    ? `<circle cx="${lastX}" cy="${lastY}" r="1.4" fill="#f87171"/>`
    : `<circle cx="${lastX}" cy="${lastY}" r="1.1" fill="#c4b5fd"/>`;
  const tip =
    `每根已收盘 bar 记一次价格相对入场锚的偏离%（最近 ${n} 根，0 线=锚价）。` +
    (jump ? `末点 ${fmtSigned(last, 2)}% ${jump} 阈值 ±${thr}%。` : thr > 0 ? `尚未越线（阈值 ±${thr}%）。` : "") +
    `缓涨=逐根逼近阈值；单根跳穿=一步跨过（多为瞬时插针/剧烈行情）。`;
  return (
    `<div class="rt-anchor-hist" title="${escHtml(tip)}">` +
    `<svg viewBox="0 0 ${W} 30" preserveAspectRatio="none">` +
    `<line x1="1.5" y1="${zeroY}" x2="${(W - 6).toFixed(2)}" y2="${zeroY}" stroke="rgba(148,163,184,0.55)" stroke-width="0.5" stroke-dasharray="3 3"/>` +
    thrLines +
    `<path d="${d}" fill="none" stroke="${col}" stroke-width="0.8" vector-effect="non-scaling-stroke"/>` +
    dot +
    `</svg>` +
    `<span class="rt-anchor-hist-note">偏离史 · ${n} 根${jump ? " · " + jump : ""}</span>` +
    `</div>`
  );
}

// 已收盘 bar 序列 + 正在形成的实时价尾点：两 bar 之间曲线也持续延伸
function rtChartPts(w) {
  const pts = ((w && w.chart && w.chart.pts) || []).map((p) => [p[0], Number(p[1])]);
  if (w && pts.length && w.session_live && w.live_price != null) {
    const lp = Number(w.live_price);
    if (Number.isFinite(lp)) pts.push([Math.floor(Date.now() / 1000), lp]);
  }
  return pts;
}

// 因子僵化诊断角标：连续 >=5 根新 bar 因子几乎不变（变化 < 0.001%）
// 带一键「查看轨迹」：点开最近 50 根 因子 vs 价格 双轴小图，区分 真钝化 还是 区间震荡
function rtStaleHtml(w, sq) {
  if (!w || !sq || !(sq.window > 0) || !(sq.factor_flat_run >= 5)) return "";
  const moved = (sq.price_moved_run || 0) >= sq.factor_flat_run;
  const why = moved
    ? "行情价格在动而因子几乎不动：公式处于饱和/钝化区，小波动不足以改变次 bar 信号（引擎每次都在正常重算，不是卡死）"
    : "价格也几乎没动：除因子钝化外，数据源可能停更或休市，请检查行情推送";
  const hasTr = Array.isArray(w.trace_hist) && w.trace_hist.length >= 2;
  return (
    `<div class="rt-stale rt-stale-btn" data-rt-stale="${escHtml(w.id)}" title="${why}。连续 ${sq.factor_flat_run} 根新 bar 因子变化 &lt; 0.001%。` +
    `${hasTr ? "\n\n▸ 点击查看最近 50 根 因子 vs 价格 双轴轨迹：真·贴边钝化 还是 区间震荡，一眼可见" : "\n\n等待服务端累计 ≥2 根新 bar 的因子轨迹后，可点开双轴图区分 真钝化 还是 区间震荡"}">` +
    `因子僵化 · ${sq.factor_flat_run} 根几乎不变${hasTr ? " · 轨迹 ▸" : ""}` +
    `</div>`
  );
}

// ── 僵化轨迹小弹窗：最近 ~50 根 因子 vs 价格 双轴（归一化叠加） ──────
function normSeriesPts(vals) {
  const nums = vals.filter((v) => Number.isFinite(v));
  if (!nums.length) return { pts: [], lo: NaN, hi: NaN };
  let lo = Math.min.apply(null, nums);
  let hi = Math.max.apply(null, nums);
  if (!(hi > lo)) { const m = hi || 1; lo = m * 0.99; hi = m * 1.01; }
  const pts = vals.map((v) => (Number.isFinite(v) ? ((v - lo) / (hi - lo)) * 100 : null));
  return { pts, lo, hi };
}

function staleTraceVerdict(trace) {
  const fs = trace.map((p) => Number(p[2])).filter(Number.isFinite);
  if (fs.length < 2) return "数据不足：至少需要 2 根带因子的新收盘 bar。";
  const uniq = new Set(fs.map((v) => v.toFixed(9)));
  const lo = Math.min.apply(null, fs);
  const hi = Math.max.apply(null, fs);
  if (uniq.size === 1) {
    return `因子 ${fs.length} 根完全相同（= ${hi}）：真·硬钝化——双峰饱和型公式贴在一侧，regime 切换前它自己不会动；不是引擎卡死。`;
  }
  let sameAdj = 0;
  for (let i = 1; i < fs.length; i++) if (fs[i] === fs[i - 1]) sameAdj++;
  const zeroFrac = sameAdj / (fs.length - 1);
  const spread = Math.abs(hi - lo) / Math.max(1e-9, Math.max(Math.abs(hi), Math.abs(lo), 1e-9));
  if (zeroFrac >= 0.6) {
    return `因子主要在少数几个台阶间切换（相邻 bar ${(zeroFrac * 100).toFixed(0)}% 完全不变）：阶梯式贴边/饱和，只在极端时跳一档。值域 [${lo}, ${hi}]。`;
  }
  if (spread < 1e-6) {
    return `因子极小幅摆动（值域 [${lo}, ${hi}]）：模型对该行情已无新信息，价格再动它也只在 ±微小 内震荡。`;
  }
  return `因子在区间内正常震荡（值域 [${lo}, ${hi}]，非恒定）：属于区间震荡而非死钝化——僵化计数只盯“相邻变化 < 0.001%”的段，配合左侧曲线看它是否正重新拉开。`;
}

function svgDualNorm(fpts, ppts) {
  const n = Math.max(fpts.length, ppts.length);
  if (n < 2) return "";
  const W = 100, T = 6, B = 94;
  const x = (i) => 1.5 + ((W - 8) * i) / (n - 1);
  const y = (v) => T + ((B - T) * (100 - v)) / 100;
  const path = (ptsArr, color, wdt) => {
    let d = "";
    for (let i = 0; i < ptsArr.length; i++) {
      const v = ptsArr[i];
      if (v == null) continue;
      d += (d ? "L" : "M") + x(i).toFixed(2) + " " + y(v).toFixed(2);
    }
    return d
      ? `<path d="${d}" fill="none" stroke="${color}" stroke-width="${wdt}" vector-effect="non-scaling-stroke"/>`
      : "";
  };
  return `<svg class="rt-stale-svg" viewBox="0 0 ${W} 100" preserveAspectRatio="none">` +
    path(fpts, "#38bdf8", 1.2) +
    path(ppts, "#fbbf24", 0.9) +
    `</svg>`;
}

function showStaleTraceModal(w) {
  const old = document.querySelector(".rt-stale-overlay");
  if (old) old.remove();
  const trace = Array.isArray(w && w.trace_hist)
    ? w.trace_hist.filter((p) => p && p.length >= 3)
    : [];
  const ov = document.createElement("div");
  ov.className = "rt-stale-overlay";
  const sq = (w && w.signal_quality) || {};
  const flat = sq.factor_flat_run != null ? sq.factor_flat_run : "—";
  const pmov = sq.price_moved_run != null ? sq.price_moved_run : "—";
  const sub = w
    ? `${escHtml(w.symbol)} ${escHtml(w.timeframe)} · ${escHtml(w.strategy_name)} · 连续不变 ${flat} 根 · 价格同时动 ${pmov} 根`
    : "";
  let body = "";
  if (trace.length < 2) {
    body = `<div class="metric-empty">因子轨迹数据不足（需 ≥2 根带因子的新收盘 bar）。服务端每次新 bar 记录一点，稍候自动可看。</div>`;
  } else {
    const fs = trace.map((p) => Number(p[2]));
    const ps = trace.map((p) => (p[1] != null ? Number(p[1]) : NaN));
    const fN = normSeriesPts(fs);
    const pN = normSeriesPts(ps);
    const verdict = staleTraceVerdict(trace);
    const win = trace.length;
    const lab = (v, nd = 6) => (Number.isFinite(v) ? Number(v).toPrecision(nd) : "—");
    body =
      `<div class="rt-stale-sub">${sub}</div>` +
      `<div class="rt-stale-chart">${svgDualNorm(fN.pts, pN.pts)}</div>` +
      `<div class="rt-stale-legend">` +
      `<span class="f">━ 因子（最近 ${win} 根 · 值域 [${lab(fN.lo)}, ${lab(fN.hi)}]）</span>` +
      `<span class="p">━ 价格（归一 · 值域 [${lab(pN.lo, 8)}, ${lab(pN.hi, 8)}]）</span>` +
      `</div>` +
      `<div class="rt-stale-verdict">💡 ${escHtml(verdict)}</div>`;
  }
  ov.innerHTML =
    `<div class="rt-stale-modal">` +
    `<div class="rt-stale-head"><h3>因子轨迹 · 最近 ${Math.max(2, trace.length)} 根</h3>` +
    `<button type="button" class="rt-stale-close" title="关闭">×</button></div>` +
    `<div class="rt-stale-body">${body}</div>` +
    `</div>`;
  const close = () => ov.remove();
  ov.addEventListener("mousedown", (e) => {
    if (e.target === ov) close();
  });
  ov.querySelector(".rt-stale-close").addEventListener("click", close);
  const onKey = (e) => {
    if (e.key === "Escape") close();
  };
  document.addEventListener("keydown", onKey, { once: true });
  ov.addEventListener("click", () => document.removeEventListener("keydown", onKey));
  document.body.appendChild(ov);
}

let _modelRowsCache = null;
let _modelRowsLoading = false;
async function ensureModelRows() {
  if (_modelRowsCache) return _modelRowsCache;
  try {
    const data = await fetchJSON("/api/strategies", { silent: true, retries: 0 });
    _modelRowsCache = data.strategies || [];
  } catch (_) {
    _modelRowsCache = [];
  }
  return _modelRowsCache;
}

function renderModelLibrary(gridId, activeFile) {
  const grid = $(gridId);
  if (!grid) return;
  // 懒加载：缓存未就绪时先取一次 /api/strategies，就绪后把两个模型库一起重绘
  if (_modelRowsCache === null) {
    if (!_modelRowsLoading) {
      _modelRowsLoading = true;
      ensureModelRows().finally(() => {
        _modelRowsLoading = false;
        if ($("btModelGrid")) renderModelLibrary("btModelGrid", selectedStrategyFile);
        if ($("ppModelGrid")) renderModelLibrary("ppModelGrid", $("ppStrategySelect")?.value || "");
      });
    }
    grid.innerHTML = '<div class="metric-empty">加载模型库…</div>';
    return;
  }
  const rows = _modelRowsCache || [];
  if (!rows.length) {
    grid.innerHTML = '<div class="metric-empty">暂无已保存模型（先训练生成 best_*.json）</div>';
    return;
  }
  // 每品种分数最高者标记为「最佳」（训练中·未验证的 live 侧车不参与，冠军星标仍归已部署文件）
  const bestBySym = {};
  rows.forEach((r) => {
    if (r.live_only) return;
    const s = r.symbol || "";
    const sc = Number(r.best_score);
    if (!(s in bestBySym) || sc > bestBySym[s].score) bestBySym[s] = { file: r.strategy_file, score: sc };
  });
  grid.innerHTML = rows
    .map((r) => {
      const sym = escHtml(r.symbol || r.file || "");
      const tf = r.timeframe ? `<span class="model-tf">${escHtml(r.timeframe)}</span>` : "";
      const score =
        r.best_score != null ? `<span class="model-score">分数 ${Number(r.best_score).toFixed(3)}</span>` : "";
      const isBest = bestBySym[r.symbol] && bestBySym[r.symbol].file === r.strategy_file;
      const liveScoreTxt =
        r.live && r.live.best_score != null ? Number(r.live.best_score).toFixed(3) : "—";
      const liveTimeTxt = r.live && r.live.updated_at ? String(r.live.updated_at).slice(5) : "";
      const liveFormulaTxt = (r.live && (r.live.formula_decoded || "")) || "";
      const liveBadge = r.live
        ? `<span class="model-live" title="训练中实时保存 best-so-far（未过冠军闸门 → 绝不覆盖部署的 strategies/best_*.json）\n实时分数：${liveScoreTxt} · 更新：${r.live.updated_at || "—"}\n公式：${escHtml(liveFormulaTxt || r.live.formula || "—")}\n侧车：${escHtml(r.live.live_file || "")}">⏳ 训练中 ${liveScoreTxt}${liveTimeTxt ? " · " + escHtml(liveTimeTxt) : ""}</span>`
        : "";
      const decoded = String(r.formula_decoded || "");
      const active =
        activeFile && String(r.strategy_file).replace(/\\/g, "/") === String(activeFile).replace(/\\/g, "/")
          ? " active"
          : "";
      const liveRowCls = r.live_only ? " model-live-row" : "";
      return `<div class="model-row${active}${liveRowCls}" title="${escHtml(r.strategy_file)}">
        <div class="model-main">
          <b>${sym}</b> ${tf}${isBest ? '<span class="model-best">★ 最佳</span>' : ""}${liveBadge} ${score}
          <div class="model-meta">${escHtml(r.file)} ${trainRangeBadge(r)} · ${escHtml(decoded.slice(0, 110))}${decoded.length > 110 ? "…" : ""}</div>
        </div>
        <button type="button" class="btn btn-mini btn-secondary" data-model-pick="${escHtml(r.strategy_file)}">选用</button>
      </div>`;
    })
    .join("");
}

function wireModelLibrary() {
  const wire = (gridId, onPick) => {
    const grid = $(gridId);
    if (!grid) return;
    grid.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-model-pick]");
      if (btn) onPick(btn.dataset.modelPick, grid);
    });
  };
  wire("btModelGrid", async (file) => {
    await applyStrategyFilePath(file);
    renderModelLibrary("btModelGrid", selectedStrategyFile);
  });
  wire("ppModelGrid", (file) => {
    const sel = $("ppStrategySelect");
    if (sel) {
      sel.value = file;
      onPaperStrategyChange();
      renderModelLibrary("ppModelGrid", sel.value);
    }
  });
}

// 半环表盘（180° 上半环，值弧按强度填充）
const RT_ARC_LEN = 150.8; // π * 48
function halfRingGauge(strength, colorHex) {
  const s = Math.max(0, Math.min(1, strength || 0));
  const off = RT_ARC_LEN * (1 - s);
  return `<svg class="rt-gauge-svg" viewBox="0 0 120 74" aria-hidden="true">
    <path class="rt-gauge-track" d="M12 62 A 48 48 0 0 1 108 62" />
    <path class="rt-gauge-val" d="M12 62 A 48 48 0 0 1 108 62"
      style="stroke:${colorHex};stroke-dasharray:${RT_ARC_LEN};stroke-dashoffset:${off.toFixed(1)};" />
  </svg>`;
}

function renderRealtimeGrid(watches) {
  const grid = $("rtGrid");
  if (!grid) return;
  if (!watches.length) {
    grid.innerHTML =
      '<div class="metric-empty">尚无监控项。添加「数据源 + 品种 + 周期 + 因子」后开始实时分析。</div>';
    rtGridSig = "";
    return;
  }

  // 签名：只在信号相关字段变化时重建（避免每次轮询重播动画）
  const sig = watches
    .map((w) =>
      [
        w.id,
        w.state,
        w.direction,
        w.strength,
        w.warn,
        w.message,
        w.last_bar_ts,
        w.updated_at,
        w.factor_value != null ? Number(w.factor_value).toFixed(4) : "",
        w.threshold != null ? Number(w.threshold) : "",
        w.session_live ? 1 : 0,
        w.next_bar_close_at || "",
        w.policy_id || "",
        w.dd_gate || "",
        w.dd_pct != null ? Number(w.dd_pct).toFixed(2) : "",
        w.last_close || "",
        (w.chart && w.chart.pts && w.chart.pts.length) || 0,
        (w.paper_position && w.paper_position.entry_price) || "",
        (w.paper_position && w.paper_position.stop_price) || "",
        (w.paper_position && w.paper_position.target_price) || "",
        (w.signal_quality && w.signal_quality.dir_run) || "",
        (w.signal_quality && w.signal_quality.factor_pct) || "",
        (w.signal_quality && w.signal_quality.flat_pct) || "",
        (w.signal_quality && w.signal_quality.factor_flat_run) || "",
        (w.signal_quality && w.signal_quality.price_moved_run) || "",
        w.dir_entry_price || "",
      ].join("~")
    )
    .join("|");
  // 签名未变时仍同步休市/倒计时锚点
  if (sig === rtGridSig) {
    watches.forEach((w) => {
      const el = grid.querySelector(`.rt-card[data-id="${CSS.escape(w.id)}"] .rt-countdown`);
      if (!el) return;
      if (w.session_live && w.next_bar_close_at) {
        el.dataset.session = "";
        el.dataset.nextClose = String(w.next_bar_close_at);
      } else if (w.state === "ok") {
        el.dataset.nextClose = "";
        el.dataset.session = "closed";
      } else {
        el.dataset.nextClose = "";
        el.dataset.session = "";
      }
    });
    return;
  }
  rtGridSig = sig;

  grid.innerHTML = watches
    .map((w) => {
      const dir = w.state === "ok" ? RT_DIR[w.direction] || RT_DIR.FLAT : null;
      const color = dir ? dir.color : "#7a8a9e";
      const strength = w.state === "ok" ? w.strength || 0 : 0;
      const dirKey = w.state === "ok" ? w.direction : null;
      const plain = w.state === "ok" ? rtSizePlain(strength, dirKey) : null;
      const dirLabel = dir ? dir.label : RT_STATE_LABEL[w.state] || w.state;
      const dirCls = dir ? dir.cls : "rt-flat";
      const srcLabel = (rtSourceById[w.source] || {}).label || w.source;
      const factorText = w.factor_value != null ? Number(w.factor_value).toFixed(4) : "—";
      // 距翻转裕度：|因子| 越过阈值才翻向，展示与阈值带的距离；靠近时高亮提醒
      const flipChip = (() => {
        if (w.state !== "ok" || w.factor_value == null) return "";
        const fv = Number(w.factor_value);
        const thr = Number(w.threshold != null ? w.threshold : 0.05) || 0.05;
        if (!Number.isFinite(fv)) return "";
        const abs = Math.abs(fv);
        if (abs <= thr) {
          return `<span class="rt-flip rt-flip-flat" title="因子已回落至 ±${thr} 阈值带内（观望区），随时可能翻向">已入观望区 · 距翻转 0</span>`;
        }
        const dist = abs - thr;
        const to = fv > 0 ? "空" : "多";
        const near = dist < thr * 3;
        return `<span class="rt-flip${near ? " rt-flip-near" : ""}" title="方向翻转需因子越过 ±${thr}；当前距翻${to}还差 ${dist.toFixed(3)}（阈值带 ±${thr}，靠近时高亮）">距翻${to} ${dist.toFixed(2)}${near ? " · 接近翻转" : ""}</span>`;
      })();
      const livePx = w.live_price != null ? Number(w.live_price) : null;
      const pxShow = livePx != null ? livePx : w.last_close;
      const pxUp = livePx != null && w.last_close != null && livePx >= Number(w.last_close);
      const pxDown = livePx != null && w.last_close != null && livePx < Number(w.last_close);
      const livePxHtml = fmtPrice(pxShow);
      const liveArrowHtml = pxUp ? "▲" : pxDown ? "▼" : "";
      const warn = w.warn ? `<div class="rt-warn" title="${escHtml(w.warn)}">⚠ ${escHtml(w.warn)}</div>` : "";
      const displayMsg =
        w.message === RT_TV_BLOCKED_CODE || w.tv_blocked
          ? "无法连接 TradingView：请开启全局 VPN（TUN）或使用云服务器"
          : w.message;
      const msg =
        w.state !== "ok" && displayMsg
          ? `<div class="rt-msg">${escHtml(displayMsg)}</div>`
          : "";
      const sizeText = plain ? plain.size : "—";
      // 价格走势图 + 入场/止损/止盈参考线（有模拟盘实际持仓画实际线，否则画建议线）
      // 曲线 = 已收盘 bar + 形成中实时价尾点（数据-live-tail 供每秒 tick 增量刷新）
      const chartPts = rtChartPts(w);
      const rtLevels = rtChartLevels(w);
      const liveTailStr =
        w.session_live && w.live_price != null ? Number(w.live_price).toFixed(8) : "";
      const chartHtml =
        chartPts.length > 1
          ? `<div class="rt-chart" data-live-tail="${escHtml(liveTailStr)}">${svgPriceChart(chartPts, rtLevels, {})}` +
            (rtLevels.length
              ? `<div class="rt-chart-tags">${rtLevels
                  .map((l) => `<span style="color:${l.color}">${l.label}</span>`)
                  .join("")}</div>`
              : "") +
            `</div>`
          : '<div class="rt-chart rt-chart-empty">等待 K 线…</div>';
      const anchorHtml = rtAnchorChipHtml(w); // 入场参考锚点：虚线+方向标签+当前偏离+阈值
      const policyTag =
        w.policy_id && w.policy_id !== "signal"
          ? `<span class="rt-policy-chip" title="持仓管理：${escHtml(policyName(w.policy_id))}">${escHtml(policyName(w.policy_id))}</span>`
          : "";
      const ddTag = ddTagHtml(w);
      // 信号质量角标：连续根数 / 多空观占比 / 因子历史分位 + 因子僵化诊断
      const sq = w.signal_quality;
      let sqHtml = "";
      if (sq && sq.window > 0) {
        const sqParts = [];
        if (sq.dir_run != null) sqParts.push(`连续 ${sq.dir_run} 根`);
        if (sq.long_pct != null) sqParts.push(`多 ${sq.long_pct}% · 空 ${sq.short_pct}% · 观 ${sq.flat_pct}%`);
        if (sq.factor_pct != null) sqParts.push(`分位 ${sq.factor_pct}%`);
        sqHtml =
          `<div class="rt-sq" title="信号质量（最近 ${sq.window} 次收盘判断）：连续=当前方向已持续根数；多/空/观=各方向占比；分位=当前因子处于自身历史的分位（越接近 100 越极端）。方向久不变但分位在回落，说明在看涨惯性中降温">${sqParts.join(" · ")}</div>` +
          rtStaleHtml(w, sq);
      }
      const flipSpark = rtFlipSparkline(sq && sq.flip_margin_hist, w.threshold);
      const pctSpark = rtPctSparkline(sq && sq.factor_pct_hist);
      const watchZoneBadge = rtWatchZoneBadge(w, sq);
      return `
    <div class="rt-card ${dirCls}${pxUp ? " rt-px-up" : pxDown ? " rt-px-down" : ""}" data-id="${escHtml(w.id)}">
      <button class="rt-remove" data-remove="${escHtml(w.id)}" title="移除监控">×</button>
      <div class="rt-card-head">
        <span class="rt-sym">${escHtml(w.symbol)}</span>
        <span class="rt-tf">${escHtml(w.timeframe)}</span>
        <span class="rt-src">${escHtml(srcLabel)}</span>
      </div>
      <div class="rt-gauge">
        ${halfRingGauge(strength, color)}
        <div class="rt-gauge-center">
          <div class="rt-strength">${escHtml(sizeText)}</div>
          <div class="rt-dir ${dirCls}">${dirLabel}</div>
        </div>
      </div>
      ${chartHtml}${anchorHtml}
      <div class="rt-live">
        <span class="rt-live-label">现价</span>
        <b class="rt-px" data-id="${escHtml(w.id)}">${livePxHtml}</b>
        ${liveArrowHtml ? `<span class="rt-px-arrow">${liveArrowHtml}</span>` : ""}
        <span class="rt-live-note">因子按已收盘 bar 更新 · 价格实时跳动</span>
      </div>
      <div class="rt-meta">
        <span class="rt-meta-item">因子 <b>${factorText}</b>${flipChip}${watchZoneBadge}${flipSpark}${pctSpark}</span>
        <span class="rt-meta-item">${escHtml(w.strategy_name)}${policyTag}${ddTag}</span>
      </div>
      ${sqHtml}
      <div class="rt-foot">
        <span class="rt-state ${w.state}">${RT_STATE_LABEL[w.state] || w.state}</span>
        <span class="rt-time">更新 ${rtClock(w.updated_at)}</span>
        <span class="rt-countdown"${
          w.session_live && w.next_bar_close_at
            ? ` data-next-close="${w.next_bar_close_at}"`
            : w.state === "ok"
              ? ` data-session="closed"`
              : ""
        }>${
          w.session_live && w.next_bar_close_at
            ? "距离下次判断 …"
            : w.state === "ok"
              ? "休市中"
              : "距离下次判断 —"
        }</span>
      </div>
      ${warn}
      ${msg}
    </div>`;
    })
    .join("");

  runCountUp(grid);
}

// ═══════════════════════════════════════════════════════════════════
// 模拟实盘（纸上交易）页
// ═══════════════════════════════════════════════════════════════════
let ppInited = false;
let ppRunning = false;
let ppSources = [];
let ppSourceById = {};
let ppImportedStrategy = null;
let ppCfgApplied = false;
let ppEquityChart = null;
let ppEquitySig = "";
let ppAccountSig = "";
let ppWatchesSig = "";
let ppTradesSig = "";
let ppWatchesById = {}; // 最近一次状态轮询的监控行（乐观移除直接改这里，行秒级消失）
const ppRemovingIds = new Set(); // 正在后台移除中的 id：渲染时过滤，防止轮询“复活”已删行

const PP_DIR = {
  LONG: { label: "多", cls: "pos" },
  SHORT: { label: "空", cls: "neg" },
  FLAT: { label: "观望", cls: "" },
};
const PP_SIDE = {
  LONG: { label: "多", action: "做多" },
  SHORT: { label: "空", action: "做空" },
};
const PP_STATE_LABEL = {
  pending: "待首评",
  ok: "运行中",
  insufficient: "历史不足",
  error: "错误",
};

function fmtMoney(v, { signed = false } = {}) {
  if (v == null || Number.isNaN(v)) return "—";
  const n = Number(v);
  const sign = signed && n > 0 ? "+" : "";
  const neg = n < 0 ? "-" : "";
  return `${sign}${neg}${Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtNum(v, digits = 4) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: digits });
}

function ppClock(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  const pad = (x) => String(x).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function ppPnlCls(v) {
  if (v == null || Number.isNaN(v) || v === 0) return "";
  return v > 0 ? "pos" : "neg";
}

async function initPaperOnce() {
  if (ppInited) return;
  ppInited = true;
  try {
    const data = await fetchJSON("/api/realtime/sources");
    ppSources = data.sources || [];
    ppSourceById = {};
    ppSources.forEach((s) => (ppSourceById[s.id] = s));
    const sel = $("ppSourceSelect");
    if (sel) {
      sel.innerHTML = ppSources
        .map((s) => `<option value="${escHtml(s.id)}">${escHtml(s.label)}${s.available ? "" : " · 未就绪"}</option>`)
        .join("");
      if (ppSources.length) sel.value = ppSources[0].id;
    }
    if (data.min_exposure != null && $("ppThresholdHint")) {
      $("ppThresholdHint").textContent = `|tanh(因子)| < ${data.min_exposure} → FLAT`;
    }
    if (data.signal_threshold != null) applySigThrSelects(Number(data.signal_threshold));
    onPaperSourceChange();
  } catch (e) {
    await logClientError("加载数据源失败: " + e.message);
  }
  await loadPaperStrategies();
}

function onPaperSourceChange() {
  const src = ppSourceById[$("ppSourceSelect")?.value];
  const tfSel = $("ppTimeframeSelect");
  const presets = $("ppSymbolPresets");
  const hint = $("ppSourceHint");
  if (!src) return;
  if (tfSel) {
    const cur = tfSel.value;
    tfSel.innerHTML = (src.timeframes || []).map((t) => `<option value="${escHtml(t)}">${escHtml(t)}</option>`).join("");
    if (src.timeframes && src.timeframes.includes(cur)) tfSel.value = cur;
    else if (src.timeframes && src.timeframes.includes("1h")) tfSel.value = "1h";
  }
  const symbolInput = $("ppSymbolInput");
  const symbolSelect = $("ppSymbolSelect");
  const useSelect = src.id === "domestic_futures" || (src.presets && src.presets.length > 20);
  if (symbolInput && symbolSelect) {
    if (useSelect) {
      symbolInput.hidden = true;
      symbolSelect.hidden = false;
      symbolSelect.innerHTML = (src.presets || [])
        .map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`)
        .join("");
      symbolSelect.onchange = () => { symbolInput.value = symbolSelect.value; };
      if (symbolSelect.value) symbolInput.value = symbolSelect.value;
    } else {
      symbolInput.hidden = false;
      symbolSelect.hidden = true;
      if (presets) {
        presets.innerHTML = (src.presets || [])
          .map((s) => `<option value="${escHtml(s)}"></option>`)
          .join("");
      }
    }
  } else if (presets) {
    presets.innerHTML = (src.presets || [])
      .map((s) => `<option value="${escHtml(s)}"></option>`)
      .join("");
  }
  if (hint) {
    hint.textContent = `${src.label}：${src.hint || ""}`;
    hint.classList.toggle("bad", !src.available);
  }
}

function onPaperStrategyChange() {
  const sel = $("ppStrategySelect");
  const picked = $("ppStrategyPicked");
  if (!sel || !picked) return;
  const opt = sel.options[sel.selectedIndex];
  picked.textContent = sel.value
    ? `因子来源：${opt ? opt.textContent : sel.value}。信号取最后已收盘 bar，方向翻转才成交。`
    : "因子来源：从已保存策略下拉选择，或「导入策略」选本地 JSON。";
  if (!sel.value) return;
  const fromOpt = (opt && opt.dataset.symbol) || "";
  const fromImport = ppImportedStrategy && sel.value === ppImportedStrategy.path ? ppImportedStrategy.symbol || "" : "";
  const sym = fromOpt || fromImport || rtParseSymbolFromFilename(sel.value);
  const input = $("ppSymbolInput");
  if (input && !input.hidden) input.value = sym;
  const select = $("ppSymbolSelect");
  if (select && !select.hidden) select.value = sym;
}

async function loadPaperStrategies() {
  const sel = $("ppStrategySelect");
  if (!sel) return;
  let rows = [];
  try {
    const data = await fetchJSON("/api/strategies", { silent: true, retries: 0 });
    rows = data.strategies || [];
  } catch (_) {}
  const opts = ['<option value="">— 选择已保存策略 —</option>'];
  if (ppImportedStrategy) {
    opts.push(`<option value="${escHtml(ppImportedStrategy.path)}" data-symbol="${escHtml(ppImportedStrategy.symbol || "")}">导入: ${escHtml(ppImportedStrategy.name)}</option>`);
  }
  rows.forEach((r) => {
    if (!r.strategy_file) return;
    const score = r.best_score != null ? Number(r.best_score).toFixed(3) : "—";
    const tf = r.timeframe ? ` ${r.timeframe}` : "";
    opts.push(`<option value="${escHtml(r.strategy_file)}" data-symbol="${escHtml(r.symbol || "")}">${escHtml(r.symbol || r.file)}${escHtml(tf)} · 分数 ${score}</option>`);
  });
  const prev = sel.value;
  sel.innerHTML = opts.join("");
  if (ppImportedStrategy) sel.value = ppImportedStrategy.path;
  else if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  onPaperStrategyChange();
}

async function ppBrowseStrategy() {
  let res;
  try {
    res = await fetchJSON("/api/strategy-file/browse", { method: "POST", retries: 0 });
  } catch (e) {
    await logClientError("导入策略失败: " + e.message);
    return;
  }
  const apply = async (info) => {
    const name = info.filename || info.strategy_file;
    ppImportedStrategy = {
      path: info.strategy_file,
      name,
      symbol: (info.symbol || "").trim() || rtParseSymbolFromFilename(name),
    };
    await loadPaperStrategies();
    const input = $("ppSymbolInput");
    if (input && !input.hidden) input.value = ppImportedStrategy.symbol;
    const select = $("ppSymbolSelect");
    if (select && !select.hidden) select.value = ppImportedStrategy.symbol;
  };
  if (!res.dialog || !res.session) {
    if (!res.cancelled) await apply(res);
    return;
  }
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 800));
      const p = await fetchJSON("/api/strategy-file/browse-poll?session=" + encodeURIComponent(res.session), { retries: 0 });
      if (!p.done) continue;
      if (!p.cancelled) await apply(p);
      return;
    }
  } catch (e) {
    await logClientError("导入策略失败: " + e.message);
  }
}

async function ppAddWatch() {
  const source = $("ppSourceSelect")?.value;
  const symbol = ($("ppSymbolInput")?.value || "").trim();
  const timeframe = $("ppTimeframeSelect")?.value;
  const strategy_file = $("ppStrategySelect")?.value;
  const picked = $("ppStrategyPicked");
  if (!symbol) {
    if (picked) { picked.textContent = "请填写品种"; picked.classList.add("bad"); }
    return;
  }
  if (!strategy_file) {
    if (picked) { picked.textContent = "请选择或导入策略因子"; picked.classList.add("bad"); }
    return;
  }
  const btn = $("ppAddBtn");
  if (btn) btn.disabled = true;
  try {
    const policy_id = effectivePolicy("ppPolicySelect", "ppStackSelect").id;
    await fetchJSON("/api/paper/watch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, symbol, timeframe, strategy_file, policy_id }),
    });
    if (picked) picked.classList.remove("bad");
    ppRunning = true;
    ppAccountSig = ""; ppWatchesSig = ""; ppTradesSig = ""; ppEquitySig = "";
    await refreshPaper();
  } catch (e) {
    if (picked) { picked.textContent = "加入失败: " + e.message; picked.classList.add("bad"); }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function ppUnwatch(id) {
  // 乐观移除：先本地删行立即重绘（行秒级消失），服务端异步收尾——
  // 不再等 unwatch + status 两个串行往返（unwatch 会先平仓，可能较慢）。
  // 墓碑过滤防止等待期间的轮询把行“复活”；失败则回滚恢复。
  if (ppRemovingIds.has(id)) return;
  const prev = Object.assign({}, ppWatchesById);
  ppRemovingIds.add(id);
  delete ppWatchesById[id];
  // 用哨兵而非空串：空结果集时 sig 也是 ""，若重置成 "" 会被签名守卫
  // 误判为“未变化”提前返回，导致最后一行移除后 DOM 不更新。
  ppWatchesSig = "∅";
  ppRenderWatchesTable();
  try {
    await fetchJSON("/api/paper/unwatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    ppAccountSig = "";
    await refreshPaper(); // 后台同步账户/持仓/明细；renderPaperWatches 里确认消失后清墓碑
  } catch (e) {
    // 服务端失败 → 回滚本地状态，行恢复；恢复的行闪红提示 + 记录原因
    ppRemovingIds.delete(id);
    Object.assign(ppWatchesById, prev);
    ppWatchesSig = "∅";
    ppRenderWatchesTable();
    const reason = e && e.message ? e.message : String(e);
    const restored = [...document.querySelectorAll("#ppWatchesBody tr")].find((tr) => {
      const btn = tr.querySelector("[data-pp-unwatch]");
      return btn && btn.dataset.ppUnwatch === id;
    });
    if (restored) {
      restored.classList.add("pp-unwatch-fail");
      const hint = document.createElement("span");
      hint.className = "pp-unwatch-fail-hint";
      hint.textContent = "⚠ 移除失败（行已恢复）: " + reason;
      const firstTd = restored.querySelector("td");
      if (firstTd) firstTd.appendChild(hint);
      window.setTimeout(() => {
        restored.classList.remove("pp-unwatch-fail");
        hint.remove();
      }, 6000);
    }
    await logClientError("移除模拟监控失败: " + reason);
  }
}

async function ppClosePosition(id) {
  try {
    await fetchJSON("/api/paper/close", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    ppTradesSig = ""; ppAccountSig = ""; ppWatchesSig = ""; ppEquitySig = "";
    await refreshPaper();
  } catch (e) {
    await logClientError("平仓失败: " + e.message);
  }
}

async function ppCloseAll() {
  try {
    await fetchJSON("/api/paper/close-all", { method: "POST" });
    ppTradesSig = ""; ppAccountSig = ""; ppWatchesSig = ""; ppEquitySig = "";
    await refreshPaper();
  } catch (e) {
    await logClientError("全平失败: " + e.message);
  }
}

// ── DD 演练：隔离账户注入合成暴跌→收复，当场走真实 dd 熔断路径 ──
async function ppRunDrill() {
  const btn = $("ppDrillBtn");
  const panel = $("ppDrillPanel");
  const body = $("ppDrillBody");
  if (!btn || !panel || !body) return;
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = "演练运行中…";
  panel.hidden = false;
  body.innerHTML = '<div class="metric-empty">注入合成行情（105 → 100.5 → 104.6）…</div>';
  try {
    const d = await fetchJSON("/api/paper/dd-drill", { method: "POST", retries: 0 });
    renderDdDrill(body, d);
  } catch (e) {
    body.innerHTML = `<div class="dd-drill-err">演练失败：${escHtml(String(e && e.message ? e.message : e))}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function renderDdDrill(body, d) {
  const stepRows = (d.steps || [])
    .map(
      (s) =>
        `<tr><td>${escHtml(s.step)}</td><td>${Number(s.n_open)}</td>` +
        `<td>${s.dd_gate === 2 ? "深档熔断" : s.dd_gate === 1 ? "熔断中" : "正常"}</td>` +
        `<td>${s.dd_pct != null ? Number(s.dd_pct).toFixed(3) + "%" : "—"}</td></tr>`
    )
    .join("");
  const evRows = (d.events || [])
    .map(
      (e) =>
        `<tr><td>${escHtml(e.event)}</td><td>${e.dd_pct != null ? Number(e.dd_pct).toFixed(2) + "%" : "—"}</td>` +
        `<td>${e.mark != null ? fmtNum(e.mark, 6) : "—"}</td><td>${e.peak != null ? fmtNum(e.peak, 6) : "—"}</td></tr>`
    )
    .join("");
  const trRows = (d.trades || [])
    .map((t) => `<tr><td>${escHtml(t.action)}</td><td>${escHtml(t.reason)}</td><td>${t.price != null ? fmtNum(t.price, 6) : "—"}</td></tr>`)
    .join("");
  const feishuLines = (d.feishu || []).length
    ? d.feishu
        .map((f) => `<div class="dd-drill-feishu">${escHtml(f.step)} → ${escHtml(f.line)}</div>`)
        .join("")
    : '<div class="dd-drill-feishu dim">（本次演练未产生飞书打印行）</div>';
  body.innerHTML =
    `<div class="dd-drill-grid">
       <div class="dd-drill-col"><h4>步骤（隔离账户）</h4>
         <table class="bt-table"><thead><tr><th>步骤</th><th>在途</th><th>闸门</th><th>回撤</th></tr></thead>
         <tbody>${stepRows || '<tr class="empty-row"><td colspan="4">无</td></tr>'}</tbody></table></div>
       <div class="dd-drill-col"><h4>DD 事件（已写入 logs/dd_gate_events.jsonl）</h4>
         <table class="bt-table"><thead><tr><th>事件</th><th>回撤</th><th>现价</th><th>峰值</th></tr></thead>
         <tbody>${evRows || '<tr class="empty-row"><td colspan="4">无</td></tr>'}</tbody></table></div>
       <div class="dd-drill-col"><h4>成交流水（含组合触发原因）</h4>
         <table class="bt-table"><thead><tr><th>动作</th><th>原因</th><th>价格</th></tr></thead>
         <tbody>${trRows || '<tr class="empty-row"><td colspan="3">无</td></tr>'}</tbody></table></div>
     </div>
     <div class="dd-drill-feishu-box">
       <h4>飞书告警结果</h4>${feishuLines}
     </div>
     <small class="field-hint">${escHtml(d.note || "")} · ${escHtml(d.account || "")}</small>`;
}

async function ppResetAccount() {
  const balRaw = Number($("ppStartBalanceInput")?.value);
  const bal = Number.isFinite(balRaw) && balRaw > 0 ? balRaw : null;
  const text = bal
    ? `确定重置模拟账户？将清空持仓、成交记录与资金曲线，起始资金按 ${bal.toLocaleString()} 重新开始（保留监控项）。`
    : "确定重置模拟账户？将清空持仓、成交记录与资金曲线（保留监控项）。";
  if (!window.confirm(text)) return;
  try {
    await fetchJSON("/api/paper/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bal != null ? { starting_balance: bal } : {}),
    });
    ppAccountSig = ""; ppWatchesSig = ""; ppTradesSig = ""; ppEquitySig = "";
    const hint = $("ppCfgHint");
    if (hint) {
      hint.textContent = "✓ 账户已重置。";
      hint.classList.remove("bad", "invalid");
      hint.classList.add("valid");
    }
    await refreshPaper();
  } catch (e) {
    await logClientError("重置账户失败: " + e.message);
  }
}

async function ppSaveCfg() {
  const hint = $("ppCfgHint");
  const btn = $("ppSaveCfgBtn");
  const read = (id, d) => {
    const v = Number($(id)?.value);
    return Number.isFinite(v) && v >= 0 ? v : d;
  };
  if (btn) btn.disabled = true;
  try {
    await fetchJSON("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        paper_starting_balance: read("ppStartBalanceInput", 100000),
        paper_notional: read("ppNotionalInput", 10000),
        paper_commission_pct: read("ppCommissionInput", 0.02),
        paper_slippage_pct: read("ppSlippageInput", 0.01),
        max_position_pct: (() => {
          const v = Number($("ppMaxPosInput")?.value);
          return Number.isFinite(v) && v >= 1 ? Math.min(200, Math.max(1, Math.round(v))) : 100;
        })(),
      }),
    });
    if (hint) {
      hint.textContent = "✓ 设置已保存（新成交按新参数；起始资金点「重置账户」生效）。";
      hint.classList.remove("bad", "invalid");
      hint.classList.add("valid");
    }
  } catch (e) {
    if (hint) {
      hint.textContent = "保存失败: " + e.message;
      hint.classList.remove("valid");
      hint.classList.add("bad", "invalid");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ═══ 历史回放（Parquet × 模型 × 持仓管理，离散引擎） ═══
// ── 可复用组件：同参「回放 vs 回测」双曲线 ──────────────────────
// 抽取自回放页「同参数回测对比」，供 回放页 / 回测页 共用：
// 画两条资金曲线进任意 canvas + 填并排统计摘要（registry 支持多图并存）。
const __eqChartRegistry = {}; // canvasId -> Chart 实例
const EQ_DUAL_COLORS = ["#5eead4", "#fbbf24", "#f87171", "#60a5fa", "#a78bfa"];
let __btStrategyDataFile = ""; // 回测页当前策略记录的默认数据文件（btDataSelect 为空时兜底）

// ── drawEquityChartTo 注册表生命周期：统一释放入口 ──────────────────
// 切换品种/策略、换数据文件、改参、清空回测视图时调用，避免长会话里
// Chart 实例/ResizeObserver/动画帧 堆在隐藏或已失效的 canvas 上。
function disposeEquityChart(canvasId) {
  const ch = __eqChartRegistry[canvasId];
  if (!ch) return;
  try {
    ch.destroy();
  } catch (_) {
    /* 已损坏实例也直接丢弃 */
  }
  delete __eqChartRegistry[canvasId];
}

function disposeAllEquityCharts() {
  Object.keys(__eqChartRegistry).forEach((id) => disposeEquityChart(id));
}

// 把注册表里挂在「不在文档中/非当前策略上下文」的旧图也顺手回收：
// canvas 被整块重建（innerHTML 替换）而没人重画时，注册表仍持有旧实例。
function sweepOrphanEquityCharts() {
  Object.keys(__eqChartRegistry).forEach((id) => {
    const ch = __eqChartRegistry[id];
    if (ch && (!ch.canvas || !ch.canvas.isConnected)) disposeEquityChart(id);
  });
}

function drawEquityChartTo(cfg) {
  const canvas = $(cfg.canvasId);
  if (!canvas) {
    // canvas 已不存在（视图被重建/清空）→ 先释放旧实例再返回
    disposeEquityChart(cfg.canvasId);
    return null;
  }
  disposeEquityChart(cfg.canvasId);
  const ds = (cfg.datasets || []).map((d, i) => ({
    label: d.label || `曲线 ${i + 1}`,
    data: d.data || [],
    borderColor: d.color || EQ_DUAL_COLORS[i % EQ_DUAL_COLORS.length],
    backgroundColor: (d.color || EQ_DUAL_COLORS[i % EQ_DUAL_COLORS.length]) + "14",
    fill: !!d.fill && i === 0,
    pointRadius: 0,
    borderWidth: 1.6,
    borderDash: d.dash || [],
    tension: 0.15,
  }));
  const ch = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels: cfg.labels || [], datasets: ds },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: {
          display: ds.length > 1 && cfg.legend !== false,
          labels: { color: "#9fb0c8", boxWidth: 14, font: { size: 11 } },
        },
      },
      scales: {
        x: { ticks: { color: "#6b7d92", maxTicksLimit: 8, maxRotation: 0, font: { size: 10 } } },
        y: { ticks: { color: "#6b7d92", font: { size: 10 } } },
      },
    },
  });
  __eqChartRegistry[cfg.canvasId] = ch;
  return ch;
}

function dualCompareSummaryHTML(stR, stB, note) {
  const pct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
  const mkCol = (title, st) =>
    `<div class="rs-cmp-col">
       <div class="rs-cmp-title">${title} <small>${escHtml(st.engine || "")}</small></div>
       <div class="metric-grid rs-cmp-grid">
         ${rsStatCard("总收益", st.total_return, pct, st.total_return > 0 ? "pos" : st.total_return < 0 ? "neg" : "")}
         ${rsStatCard("夏普", st.sharpe, (v) => v.toFixed(2), "")}
         ${rsStatCard("最大回撤", st.max_drawdown, pct, "")}
         ${rsStatCard("交易数", st.n_trades, (v) => String(v), "")}
         ${rsStatCard("胜率", st.win_rate, (v) => (v * 100).toFixed(1) + "%", "")}
         ${rsStatCard("盈亏比", st.profit_loss_ratio, (v) => v.toFixed(2), "")}
       </div>
     </div>`;
  return (
    `<div class="rs-cmp-cols">${mkCol("回放（离散撮合）", stR)}${mkCol("回测口径", stB)}</div>` +
    (note ? `<div class="rs-cmp-note">${escHtml(note)}</div>` : "")
  );
}

// 组件入口：把 回放+回测 双曲线画进指定 canvas 并填摘要容器
function renderDualCompareView(cfg) {
  drawEquityChartTo({
    canvasId: cfg.canvasId,
    labels: cfg.labels || [],
    datasets: [
      { label: cfg.replayLabel || "回放（离散撮合）", data: cfg.replayData || [], color: cfg.replayColor || "#5eead4", fill: true },
      { label: cfg.btLabel || "回测口径", data: cfg.btData || [], color: cfg.btColor || "#fbbf24", dash: cfg.btDash || [5, 3] },
    ],
    legend: cfg.legend,
  });
  if (cfg.axisEl && cfg.axisLabel) cfg.axisEl.textContent = cfg.axisLabel;
  if (cfg.summaryEl) cfg.summaryEl.innerHTML = dualCompareSummaryHTML(cfg.replayStats || {}, cfg.btStats || {}, cfg.note || "");
}

async function loadReplayStrategies() {
  const sel = $("rsStrategySelect");
  if (!sel) return;
  let rows = [];
  try {
    const d = await fetchJSON("/api/strategies", { silent: true, retries: 0 });
    rows = d.strategies || [];
  } catch (_) {}
  const cur = sel.value;
  sel.innerHTML = ['<option value="">— 选择模型 —</option>']
    .concat(
      rows.map(
        (r) =>
          `<option value="${escHtml(r.strategy_file)}">${escHtml(r.symbol || r.file)}${r.timeframe ? " " + escHtml(r.timeframe) : ""} · 分数 ${r.best_score != null ? Number(r.best_score).toFixed(3) : "—"}</option>`
      )
    )
    .join("");
  if (cur && rows.some((r) => r.strategy_file === cur)) sel.value = cur;
  if (!sel.value && rows.length) sel.value = rows[0].strategy_file;
}

// 回放数据文件下拉：data/training + data/slices（含一级子目录）
async function rsLoadDataFiles() {
  const sel = $("rsDataSelect");
  if (!sel) return;
  let files = [];
  try {
    const d = await fetchJSON("/api/data/files", { silent: true, retries: 0 });
    files = d.files || [];
  } catch (_) {}
  const cur = sel.value;
  const group = (name, rows) =>
    rows.length
      ? `<optgroup label="${escHtml(name)}">` +
        rows
          .map(
            (f) =>
              `<option value="${escHtml(f.data_file)}" title="${escHtml(f.rel)} · ${f.bars ?? "?"} 根 · ${f.start ?? "?"} ~ ${f.end ?? "?"} · ${f.size_mb ?? "?"} MB">` +
              `${escHtml(f.symbol || f.rel)}${f.timeframe ? " · " + escHtml(f.timeframe) : ""} — ${escHtml(f.rel.split("/").pop())}` +
              `${f.bars ? `（${f.bars} 根）` : ""}</option>`
          )
          .join("") +
        "</optgroup>"
      : "";
  const slices = files.filter((f) => f.rel.startsWith("data/slices"));
  const training = files.filter((f) => f.rel.startsWith("data/training"));
  sel.innerHTML =
    '<option value="">— 选择数据文件 —</option>' +
    group("data/slices（切片/样本外）", slices) +
    group("data/training（原始下载）", training);
  // 优先保留当前值 → 最近一次记忆的文件 → 最新文件
  if (cur && files.some((f) => f.data_file === cur)) {
    sel.value = cur;
  } else {
    const mem = await fetchJSON("/api/settings", { silent: true, retries: 0 }).catch(() => ({}));
    const last = (mem && mem.last_data_file) || "";
    if (last && files.some((f) => f.data_file === last)) sel.value = last;
    else if (slices.length) sel.value = slices[0].data_file;
    else if (files.length) sel.value = files[0].data_file;
  }
}

async function rsBrowseDataFile() {
  let res;
  try {
    res = await fetchJSON("/api/data-file/browse", { method: "POST", retries: 0 });
  } catch (_) {
    return;
  }
  const apply = (p) => {
    const sel = $("rsDataSelect");
    if (!sel || !p || !p.data_file) return;
    if (![...sel.options].some((o) => o.value === p.data_file)) {
      const name = p.data_file.split("/").pop() || p.data_file;
      sel.add(new Option(`自定义：${name}`, p.data_file));
    }
    sel.value = p.data_file;
  };
  if (!res.dialog || !res.session) {
    apply(res);
    return;
  }
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 800));
      const p = await fetchJSON(
        "/api/data-file/browse-poll?session=" + encodeURIComponent(res.session),
        { retries: 0 }
      );
      if (!p.done) continue;
      apply(p);
      return;
    }
  } catch (_) {}
}

async function rsRunReplay() {
  const data_file = ($("rsDataSelect")?.value || "").trim();
  const strategy_file = $("rsStrategySelect")?.value || "";
  const btn = $("rsReplayBtn");
  const hint = $("rsHint");
  if (!data_file) {
    if (hint) {
      hint.textContent = "请选择 Parquet 数据文件";
      hint.classList.add("bad");
    }
    return;
  }
  if (!strategy_file) {
    if (hint) {
      hint.textContent = "请选择模型";
      hint.classList.add("bad");
    }
    return;
  }
  if (btn) btn.disabled = true;
  if (hint) {
    hint.textContent = "回放中（大文件需数十秒）…";
    hint.classList.remove("bad");
  }
  try {
    const w = Number($("rsWindowInput")?.value);
    const mp = Number($("rsMaxPosInput")?.value);
    const res = await fetchJSON("/api/paper/replay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data_file,
        strategy_file,
        policy_id: $("rsPolicySelect")?.value || "signal",
        commission_pct: Number($("rsCommInput")?.value) || 0.02,
        slippage_pct: Number($("rsSlipInput")?.value) || 0.01,
        window_bars: Number.isFinite(w) && w >= 800 ? w : null,
        max_position_pct: Number.isFinite(mp) && mp >= 1 ? Math.min(200, Math.max(1, mp)) : 100,
        signal_threshold: Number($("rsThresholdSelect")?.value) || 0.05,
      }),
    });
    rsRenderReplayResult(res);
  } catch (e) {
    if (hint) {
      hint.textContent = "回放失败: " + e.message;
      hint.classList.add("bad");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

// 当前回放参数记忆（供「同参数回测对比」原样复用）
let __rsLast = null;

function rsCurrentParams() {
  const w = Number($("rsWindowInput")?.value);
  const mp = Number($("rsMaxPosInput")?.value);
  return {
    data_file: ($("rsDataSelect")?.value || "").trim(),
    strategy_file: $("rsStrategySelect")?.value || "",
    policy_id: $("rsPolicySelect")?.value || "signal",
    commission_pct: Number($("rsCommInput")?.value) || 0.02,
    slippage_pct: Number($("rsSlipInput")?.value) || 0.01,
    window_bars: Number.isFinite(w) && w >= 800 ? w : null,
    max_position_pct: Number.isFinite(mp) && mp >= 1 ? Math.min(200, Math.max(1, mp)) : 100,
    signal_threshold: Number($("rsThresholdSelect")?.value) || 0.05,
  };
}

// 回放页资金曲线（rsEquityChart）也按参数签名管理生命周期：
// 切换模型/数据/窗口/组合/成本后旧曲线不再代表当前设置 → 释放并收起。
let __rsDrawnKey = "";

function rsInvalidateEquityView() {
  disposeEquityChart("rsEquityChart");
  __rsDrawnKey = "";
  const live = $("rsEquityLive");
  if (live) live.hidden = true;
  const cmp = $("rsCompareSummary");
  if (cmp) cmp.hidden = true;
  const grid = $("rsStatsGrid");
  if (grid && !grid.querySelector(".metric-empty")) {
    grid.innerHTML = '<div class="metric-empty">运行回放后显示绩效（参数已变更，旧结果已释放）</div>';
  }
}

function rsSyncCompareBtn() {
  const btn = $("rsCompareBtn");
  if (!btn) return;
  const p = rsCurrentParams();
  btn.disabled = !(p.data_file && p.strategy_file);
  const key = JSON.stringify(p);
  if (__rsDrawnKey && key !== __rsDrawnKey) rsInvalidateEquityView();
}

// 同参数回测对比：后端同因子一次算出「回放（离散）+ 回测口径」两条曲线，叠加到同一张图
async function rsRunCompare() {
  const btn = $("rsCompareBtn");
  const hint = $("rsHint");
  if (btn) btn.disabled = true;
  if (hint) {
    hint.textContent = "回测对比中（同因子跑回放+回测，需数十秒）…";
    hint.classList.remove("bad");
  }
  try {
    const res = await fetchJSON("/api/paper/replay-compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rsCurrentParams()),
    });
    rsRenderCompare(res);
    __rsDrawnKey = JSON.stringify(rsCurrentParams()); // 记录成图参数签名
    sweepOrphanEquityCharts();
  } catch (e) {
    if (hint) {
      hint.textContent = "回测对比失败: " + e.message;
      hint.classList.add("bad");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function rsDrawEquity(datasets, opts) {
  opts = opts || {};
  const live = $("rsEquityLive");
  if (!live) return null;
  live.hidden = false;
  if (opts.axisLabel) {
    const lb = $("rsEquityAxisLabel");
    if (lb) lb.textContent = opts.axisLabel;
  }
  return drawEquityChartTo({
    canvasId: "rsEquityChart",
    labels: opts.labels || [],
    datasets: datasets || [],
    legend: opts.legend,
  });
}

function rsStatCard(label, raw, fmt, extraCls) {
  const text = raw == null || Number.isNaN(raw) ? "—" : fmt(raw);
  return `<div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value ${extraCls || ""}">${text}</div></div>`;
}

function rsRenderCompare(res) {
  const hint = $("rsHint");
  const stR = (res.replay && res.replay.stats) || {};
  const stB = (res.backtest && res.backtest.stats) || {};
  const pol = policyName(res.policy_id);
  if (hint) {
    hint.textContent = `对比完成 · ${pol} · ${res.bars} 根${res.window_start ? `（尾部 ${res.bars} 根）` : "（全量）"} · ${res.policy_note || ""}`;
    hint.classList.remove("bad");
  }
  const sum = $("rsCompareSummary");
  if (sum) sum.hidden = false;
  renderDualCompareView({
    canvasId: "rsEquityChart",
    labels: res.labels || [],
    replayLabel: "回放（离散撮合）",
    replayData: (res.replay && res.replay.equity) || [],
    btLabel: res.policy_id === "signal" ? "回测（训练连续口径）" : "回测（同离散引擎）",
    btData: (res.backtest && res.backtest.equity) || [],
    axisEl: $("rsEquityAxisLabel"),
    axisLabel: "累计净值（1.0 起步）· 同参数：回放 vs 回测",
    summaryEl: sum,
    replayStats: stR,
    btStats: stB,
    note: res.policy_note || "",
    legend: true,
  });
}

// ── 回测页「同一模型 回测 vs 回放 双曲线视图」：复用同参对比组件 ────
function btReplayCompareParams() {
  const w = Number($("btWindowInput")?.value);
  const mp = Number($("btMaxPosInput")?.value);
  const costs = readBacktestCosts();
  const dataFile = ($("btDataSelect")?.value || "").trim() || __btStrategyDataFile || "";
  return {
    data_file: dataFile,
    strategy_file: selectedStrategyFile || "",
    policy_id: btEffectivePolicy().id,
    commission_pct: costs.commission_pct,
    slippage_pct: costs.slippage_pct,
    window_bars: Number.isFinite(w) && w >= 800 ? w : null,
    max_position_pct: Number.isFinite(mp) && mp >= 1 ? Math.min(200, Math.max(1, mp)) : 100,
    signal_threshold: Number($("btThresholdSelect")?.value) || 0.05,
  };
}

// 同参双口径视图：记录最近一次成图时的参数签名；任一参数/品种变化 → 释放旧图并收起
let __btDualDrawnKey = "";

function btInvalidateDualView() {
  disposeEquityChart("btDualChart");
  __btDualDrawnKey = "";
  const live = $("btDualLive");
  if (live) live.hidden = true;
  const empty = $("btDualEmpty");
  if (empty) empty.hidden = false;
  const axis = $("btDualAxis");
  if (axis) axis.textContent = "累计净值（1.0 起步）· 同一模型：回测 vs 回放";
  const sum = $("btDualSummary");
  if (sum) sum.innerHTML = "";
}

function btSyncDualBtn() {
  const btn = $("btDualBtn");
  if (!btn) return;
  const p = btReplayCompareParams();
  btn.disabled = !(p.data_file && p.strategy_file);
  // 切换品种/策略/数据/窗口/组合/成本 已使旧图失真 → 统一释放注册表实例并收起视图
  const key = JSON.stringify(p);
  if (__btDualDrawnKey && key !== __btDualDrawnKey) btInvalidateDualView();
}

async function btRunDualCompare() {
  const btn = $("btDualBtn");
  const hint = $("btDualHint");
  const empty = $("btDualEmpty");
  const live = $("btDualLive");
  if (btn) btn.disabled = true;
  if (hint) {
    hint.textContent = "同参回放叠加中（同一因子跑 回放 + 回测 两条曲线）…";
    hint.classList.remove("bad");
  }
  try {
    const res = await fetchJSON("/api/paper/replay-compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(btReplayCompareParams()),
    });
    if (empty) empty.hidden = true;
    if (live) live.hidden = false;
    if (hint) {
      hint.textContent = `对比完成 · ${comboDisplayName(res.policy_id)} · ${res.bars} 根 · ${res.policy_note || ""}`;
      hint.classList.remove("bad");
    }
    renderDualCompareView({
      canvasId: "btDualChart",
      labels: res.labels || [],
      replayLabel: "回放（模拟实盘离散）",
      replayData: (res.replay && res.replay.equity) || [],
      btLabel: res.policy_id === "signal" ? "回测（训练连续口径）" : "回测（同离散引擎）",
      btData: (res.backtest && res.backtest.equity) || [],
      axisEl: $("btDualAxis"),
      axisLabel: "累计净值（1.0 起步）· 同一模型：回测口径 vs 回放（离散撮合）",
      summaryEl: $("btDualSummary"),
      replayStats: (res.replay && res.replay.stats) || {},
      btStats: (res.backtest && res.backtest.stats) || {},
      note: res.policy_note || "",
      legend: true,
    });
    __btDualDrawnKey = JSON.stringify(btReplayCompareParams()); // 记录成图时的完整参数签名
    sweepOrphanEquityCharts();
  } catch (e) {
    if (hint) {
      hint.textContent = "同参回放叠加失败: " + e.message;
      hint.classList.add("bad");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function rsRenderReplayResult(res) {
  const st = res.stats || {};
  const hint = $("rsHint");
  if (hint) {
    hint.textContent = `回放完成 · ${policyName(res.policy_id)} · ${res.bars} 根${res.window_start ? `（尾部 ${res.bars} 根）` : "（全量）"}`;
    hint.classList.remove("bad");
  }
  rsRenderOosWarn(res);
  rsRenderYearly(res);
  const grid = $("rsStatsGrid");
  if (grid) {
    const mk = (label, raw, fmt, cls) => {
      const text = raw == null || Number.isNaN(raw) ? "—" : fmt(raw);
      return `<div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value ${cls}">${text}</div></div>`;
    };
    const pct = (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
    grid.innerHTML =
      mk("总收益", st.total_return, pct, st.total_return > 0 ? "pos" : st.total_return < 0 ? "neg" : "") +
      mk("Sharpe", st.sharpe, (v) => v.toFixed(2), "") +
      mk("Sortino", st.sortino, (v) => v.toFixed(2), "") +
      mk("交易数", st.n_trades, (v) => String(v), "") +
      mk("胜率", st.win_rate, (v) => (v * 100).toFixed(1) + "%", "") +
      mk("盈亏比", st.profit_loss_ratio, (v) => v.toFixed(2), "") +
      mk("平均持仓", st.avg_hold_bars, (v) => v.toFixed(1) + " 根", "") +
      mk("手续费合计", st.fees_total, (v) => v.toFixed(4), "");
  }
  const hideSum = $("rsCompareSummary");
  if (hideSum) hideSum.hidden = true; // 仅回放时隐藏对比摘要
  rsDrawEquity(
    [{ label: "账户净值（离散撮合）", data: (res.equity && res.equity.equity) || [], color: "#5eead4", fill: true }],
    { labels: (res.equity && res.equity.labels) || [], axisLabel: "账户净值（1.0 起步，离散撮合同模拟盘口径）", legend: false }
  );
  __rsDrawnKey = JSON.stringify(rsCurrentParams()); // 先记签名再同步，避免刚画完被误判为“参数已变”
  rsSyncCompareBtn();
  sweepOrphanEquityCharts();
}

// 窗口超出训练范围 → 弹「样本外部署」警告 + 诚实 OOS 建议
function rsRenderOosWarn(res) {
  const el = $("rsOosWarn");
  if (!el) return;
  const oos = res.oos;
  const st = oos && oos.status;
  const needsWarn = st && st !== "oos-holdout" && st !== "oos-new";
  if (!oos || !needsWarn) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  const ni = oos.n_in_sample || 0;
  const nh = oos.n_holdout || 0;
  const npost = oos.n_post_train || 0;
  const train = oos.train || {};
  let headline = "";
  if (st === "in-sample") headline = `⚠ 样本内回放：窗口 ${res.bars} 根全部在训练集内（模型见过这些数据）`;
  else if (st === "partial") headline = `⚠ 部分样本内：${ni} 根在训练集内，仅尾 ${nh} 根为真 holdout`;
  else if (st === "mixed") headline = `⚠ 混合窗口：样本内 ${ni} / holdout ${nh} / 训练截止后新数据 ${npost}`;
  else headline = "⚠ 无法判定窗口的样本外状态（缺训练溯源）";
  const trainTxt = train.n_bars
    ? `训练范围：${train.n_bars} 根 · 截止 ${train.train_end_date || "—"} · holdout 尾 ${train.holdout_bars ?? "—"} 根（${train.source === "train_range" ? "精确溯源" : "推断溯源"}）`
    : "训练范围：未知（策略文件缺 train_range/data_source）";
  const h = oos.honest;
  const honestTxt = h
    ? `诚实 OOS 建议：从 bar ${h.start_bar} 起 ${h.n_bars} 根（${h.note}）`
    : "诚实 OOS 建议：无法计算（回放文件与训练文件不同或无时间戳）";
  el.hidden = false;
  el.innerHTML =
    `<div class="rs-oos-head">${escHtml(headline)} — 这是<b>样本内部署</b>，绩效会系统性偏乐观</div>` +
    `<div class="rs-oos-sub">${escHtml(trainTxt)}<br>${escHtml(honestTxt)}</div>`;
}

// 逐年度绩效拆解表
function rsRenderYearly(res) {
  const block = $("rsYearlyBlock");
  const table = $("rsYearlyTable");
  if (!block || !table) return;
  const rows = res.yearly || [];
  if (!rows.length) {
    block.hidden = true;
    table.innerHTML = "";
    return;
  }
  block.hidden = false;
  const bestY = res.best_year && res.best_year.year;
  const worstY = res.worst_year && res.worst_year.year;
  const pct = (v) => (v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%");
  const head = `<thead><tr><th>年份</th><th>根数</th><th>收益</th><th>最大回撤</th><th>交易</th><th>夏普</th><th>标注</th></tr></thead>`;
  const trs = rows.map((r) => {
    const cls = r.return > 0 ? "pos" : r.return < 0 ? "neg" : "";
    let tag = "";
    if (r.year === bestY && rows.length > 1) tag = `<span class="mx-badge focus" title="全部年份中收益最高">★ 贡献最多</span>`;
    if (r.year === worstY && rows.length > 1) tag += ` <span class="mx-badge pareto" title="全部年份中收益最低">▼ 损失最多</span>`;
    return `<tr>
      <td class="sym-cell">${escHtml(r.year)}</td>
      <td>${r.bars}</td>
      <td class="${cls}">${pct(r.return)}</td>
      <td class="neg">${r.max_drawdown != null ? (r.max_drawdown * 100).toFixed(2) + "%" : "—"}</td>
      <td>${r.n_trades ?? "—"}</td>
      <td>${r.sharpe != null ? r.sharpe.toFixed(2) : "—"}</td>
      <td>${tag}</td>
    </tr>`;
  }).join("");
  table.innerHTML = head + `<tbody>${trs}</tbody>`;
}

function renderPaperAccount(st) {
  const grid = $("ppAccountGrid");
  if (!grid) return;
  const a = st.account || {};
  const eq = a.equity;
  const pnlColor = eq - st.config.starting_balance >= 0 ? "#4ade80" : "#f87171";
  const cards = [
    { label: "账户净值", raw: a.equity, cls: "accent" },
    { label: "可用现金", raw: a.cash, cls: "" },
    { label: "总盈亏", raw: a.total_pnl, cls: ppPnlCls(a.total_pnl) },
    { label: "总收益率", raw: a.total_return_pct != null ? (a.total_return_pct / 100).toFixed(4) : null, cls: ppPnlCls(a.total_pnl), pct: true },
    { label: "已实现盈亏", raw: a.realized_pnl, cls: ppPnlCls(a.realized_pnl) },
    { label: "未实现盈亏", raw: a.unrealized_pnl, cls: ppPnlCls(a.unrealized_pnl) },
    { label: "已付手续费", raw: a.fees_paid, cls: "" },
    { label: "平仓次数", raw: a.n_trades, cls: "" },
  ];
  const sig = cards.map((c) => `${c.label}:${c.raw}`).join("|");
  if (sig === ppAccountSig && st.count > 0) {
    if ($("ppStatusHint")) $("ppStatusHint").textContent = ppStatusText(st);
    return;
  }
  ppAccountSig = sig;
  grid.innerHTML = cards
    .map((c) => {
      let text;
      if (c.raw == null || Number.isNaN(c.raw)) {
        text = "—";
      } else if (c.pct) {
        const n = Number(c.raw);
        text = (n >= 0 ? "+" : "") + (n * 100).toFixed(2) + "%";
      } else {
        text = fmtMoney(c.raw, { signed: /盈亏|收益/.test(c.label) });
      }
      return `
    <div class="metric-card">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}">${text}</div>
    </div>`;
    })
    .join("");
  if ($("ppStatusHint")) $("ppStatusHint").textContent = ppStatusText(st);
}

function ppStatusText(st) {
  const cfg = st.config || {};
  const base = st.count
    ? `${st.running ? "运行中" : "已暂停"} · ${st.count} 项 · 起始 ${fmtMoney(cfg.starting_balance)} · 满仓名义 ${fmtMoney(cfg.notional)}`
    : "暂无监控项";
  return base;
}

function renderPaperWatches(st) {
  // 服务端已确认消失的移除项 → 清除墓碑（unwatch 200 后的 status 不再含它）
  const serverIds = new Set((st.watches || []).map((w) => w.id));
  for (const id of Array.from(ppRemovingIds)) {
    if (!serverIds.has(id)) ppRemovingIds.delete(id);
  }
  // 本地缓存 = 服务端最近一次真值（乐观移除在此基础上删行、过滤墓碑）
  ppWatchesById = {};
  (st.watches || []).forEach((w) => (ppWatchesById[w.id] = w));
  ppRenderWatchesTable();
}

function ppRenderWatchesTable() {
  const tbody = $("ppWatchesBody");
  if (!tbody) return;
  const rows = Object.values(ppWatchesById).filter((w) => !ppRemovingIds.has(w.id));
  const sig = rows.map((w) => [w.id, w.state, w.direction, w.strength, w.message, w.last_close, w.policy_id, w.dd_gate || "", w.dd_pct != null ? Number(w.dd_pct).toFixed(2) : ""].join("~")).join("|");
  if (sig === ppWatchesSig) return;
  ppWatchesSig = sig;
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="9">暂无监控项，加入后引擎自动按收盘信号撮合</td></tr>';
    $("ppWatchesHint").textContent = "—";
    return;
  }
  $("ppWatchesHint").textContent = `${rows.length} 个监控项`;
  tbody.innerHTML = rows
    .map((w) => {
      const dir = PP_DIR[w.direction];
      const dirHtml = dir ? `<span class="${dir.cls}">${dir.label}</span>` : "—";
      const strength = w.strength != null ? (w.strength * 100).toFixed(0) + "%" : "—";
      const srcLabel = (ppSourceById[w.source] || {}).label || w.source;
      const msg = w.state === "ok" ? "" : `<span class="rt-warn" title="${escHtml(w.message)}">${escHtml((w.message || "").slice(0, 26))}</span>`;
      return `
      <tr>
        <td class="sym-cell">${escHtml(w.symbol)}</td>
        <td>${escHtml(w.timeframe)}</td>
        <td>${escHtml(srcLabel)}</td>
        <td title="${escHtml(w.strategy_file)}">${escHtml(w.strategy_name)}${w.policy_id && w.policy_id !== "signal" ? `<span class="rt-policy-chip" title="持仓管理：${escHtml(policyName(w.policy_id))}">${escHtml(policyName(w.policy_id))}</span>` : ""}${ddTagHtml(w)}</td>
        <td>${dirHtml}</td>
        <td>${strength}</td>
        <td><span class="rt-state ${w.state}">${PP_STATE_LABEL[w.state] || w.state}</span> ${msg}</td>
        <td>${fmtMoney(w.notional)}</td>
        <td><button type="button" class="btn btn-mini btn-danger" data-pp-unwatch="${escHtml(w.id)}">移除</button></td>
      </tr>`;
    })
    .join("");
}

function renderPaperPositions(st) {
  const tbody = $("ppPositionsBody");
  if (!tbody) return;
  const rows = st.positions || [];
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="12">暂无持仓（空仓）</td></tr>';
  } else {
    tbody.innerHTML = rows
      .map((p) => {
        const side = PP_SIDE[p.side];
        const sideCls = p.side === "LONG" ? "pos" : "neg";
        const polChip =
          p.policy_id && p.policy_id !== "signal"
            ? `<span class="rt-policy-chip" title="持仓管理方案">${escHtml(policyName(p.policy_id))}</span>`
            : "—";
        return `
      <tr>
        <td class="sym-cell">${escHtml(p.symbol)}</td>
        <td>${escHtml(p.timeframe)}</td>
        <td class="${sideCls}"><b>${side ? side.label : "—"}</b></td>
        <td>${polChip}</td>
        <td>${fmtMoney(p.notional_value)}</td>
        <td>${fmtNum(p.qty, 8)}</td>
        <td>${fmtNum(p.entry_price, 6)}</td>
        <td>${p.mark_price != null ? fmtNum(p.mark_price, 6) : "—"}</td>
        <td>${p.stop_price != null ? fmtNum(p.stop_price, 6) : "—"}</td>
        <td>${p.target_price != null ? fmtNum(p.target_price, 6) : "—"}</td>
        <td class="${ppPnlCls(p.unrealized_pnl)}">${fmtMoney(p.unrealized_pnl, { signed: true })}</td>
        <td><button type="button" class="btn btn-mini btn-secondary" data-pp-close="${escHtml(p.watch_id)}">平仓</button></td>
      </tr>`;
      })
      .join("");
  }
  const closeAll = $("ppCloseAllBtn");
  if (closeAll) closeAll.disabled = !rows.length;
  const empty = $("ppPositionsEmptyHint");
  if (empty) empty.textContent = rows.length ? "" : "";
}

const PP_ACTION_CLS = {
  "开多": "pos",
  "平多": "pos",
  "开空": "neg",
  "平空": "neg",
};

function renderPaperTrades(st) {
  const tbody = $("ppTradesBody");
  if (!tbody) return;
  const trades = st.trades || [];
  const sig = (trades[0] ? trades[0].seq + ":" + trades[0].cash_after : "none") + ":" + trades.length;
  if (sig === ppTradesSig) return;
  ppTradesSig = sig;
  if (!trades.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="11">暂无成交记录</td></tr>';
    $("ppTradesHint").textContent = "—";
    return;
  }
  $("ppTradesHint").textContent = `最近 ${trades.length} 笔（最新在上）`;
  tbody.innerHTML = trades
    .map((t) => {
      const actionCls = PP_ACTION_CLS[t.action] || "";
      return `
      <tr>
        <td>${ppClock(t.ts)}</td>
        <td class="${actionCls}"><b>${escHtml(t.action)}</b></td>
        <td class="sym-cell">${escHtml(t.symbol)}</td>
        <td>${escHtml(t.timeframe)}</td>
        <td>${fmtNum(t.price, 6)}</td>
        <td>${fmtNum(t.qty, 8)}</td>
        <td>${fmtMoney(t.notional_value)}</td>
        <td>${fmtMoney(t.fee)}</td>
        <td class="${ppPnlCls(t.pnl)}">${fmtMoney(t.pnl, { signed: true })}</td>
        <td>${fmtMoney(t.cash_after)}</td>
        <td title="${escHtml(t.reason || "")}">${escHtml(t.reason || "")}</td>
      </tr>`;
    })
    .join("");
}

const PP_EQUITY_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 400, easing: "easeOutQuart" },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: "rgba(8, 12, 20, 0.94)",
      borderColor: "rgba(94, 234, 212, 0.35)",
      borderWidth: 1,
      titleColor: "#e8edf4",
      bodyColor: "#a9bccf",
      titleFont: { family: "'JetBrains Mono'", size: 11 },
      bodyFont: { family: "'JetBrains Mono'", size: 11 },
      padding: 10,
      cornerRadius: 8,
      callbacks: {
        label: (c) => ` 净值 ${fmtMoney(c.parsed.y)}`,
      },
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, maxRotation: 0, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
  },
};

function renderPaperEquity(st) {
  const live = $("ppEquityLive");
  const empty = $("ppEquityEmpty");
  const eq = (st.equity || {});
  const ts = eq.ts || [];
  const vals = eq.equity || [];
  if (!ts.length) {
    if (live) live.hidden = true;
    if (empty) empty.hidden = false;
    if (ppEquityChart) { ppEquityChart.destroy(); ppEquityChart = null; }
    ppEquitySig = "";
    if ($("ppChartHint")) $("ppChartHint").textContent = "—";
    return;
  }
  const sig = `${ts.length}:${vals[0]}:${vals[vals.length - 1]}`;
  if (sig === ppEquitySig && ppEquityChart) {
    if ($("ppChartHint")) $("ppChartHint").textContent = `${ts.length} 个资金点`;
    return;
  }
  ppEquitySig = sig;
  if (live) live.hidden = false;
  if (empty) empty.hidden = true;
  const labels = ts.map((t) => ppClock(t));
  const data = vals.map(Number);
  const canvas = $("ppEquityChart");
  if (ppEquityChart) ppEquityChart.destroy();
  if (canvas && window.Chart) {
    ppEquityChart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "账户净值",
            data,
            borderColor: "#34f5c8",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: "#34f5c8",
            pointHoverBorderColor: "#05070d",
            fill: true,
            backgroundColor: (ctx) => verticalGradient(ctx.chart, "52, 245, 200", 0.3, 0),
          },
        ],
      },
      options: PP_EQUITY_OPTIONS,
    });
  }
  if ($("ppChartHint")) $("ppChartHint").textContent = `${ts.length} 个资金点 · 账户净值 ${fmtMoney(vals[vals.length - 1])}`;
}

async function refreshPaper() {
  let st;
  try {
    st = await fetchJSON("/api/paper/status", { silent: true });
  } catch (_) {
    return;
  }
  // 有监控项却未在跑时自动拉起引擎；仅 REST 路径做
  if (st.count > 0 && !st.running) {
    try {
      st = await fetchJSON("/api/paper/start", { method: "POST", silent: true });
    } catch (_) {}
  }
  applyPaperState(st);
}

// 快照 → UI（REST 轮询与 SSE 推送共用同一入口）
function applyPaperState(st) {
  ppRunning = !!st.running;
  if (!ppCfgApplied && st.config) {
    ppCfgApplied = true;
    const setVal = (id, v) => { const el = $(id); if (el && v != null) el.value = Number(v); };
    setVal("ppStartBalanceInput", st.config.starting_balance);
    setVal("ppNotionalInput", st.config.notional);
    setVal("ppCommissionInput", st.config.commission_pct);
    setVal("ppSlippageInput", st.config.slippage_pct);
    setVal("ppMaxPosInput", st.config.max_position_pct);
  }
  renderPaperAccount(st);
  renderPaperWatches(st);
  renderPaperPositions(st);
  renderPaperTrades(st);
  renderPaperEquity(st);
  loadDdEventsPanel("Pp", false);
}

// 绑定模拟盘页事件（元素都在静态 HTML 中，脚本执行时已可访问）
function bindPaperEvents() {
  if (document.querySelector(".stepper [data-page=paper]")) {
    // switchPage 已由统一监听器处理；此处只绑定页内控件
  }
  const bind = (id, evt, fn) => {
    const el = $(id);
    if (el) el.addEventListener(evt, fn);
  };
  bind("ppSourceSelect", "change", onPaperSourceChange);
  bind("ppStrategySelect", "change", onPaperStrategyChange);
  bind("ppBrowseStrategyBtn", "click", ppBrowseStrategy);
  bind("ppAddBtn", "click", ppAddWatch);
  bind("ppSaveCfgBtn", "click", ppSaveCfg);
  bind("ppResetBtn", "click", ppResetAccount);
  bind("ppCloseAllBtn", "click", ppCloseAll);
  bind("ppSymbolInput", "keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      ppAddWatch();
    }
  });
  const watchBody = $("ppWatchesBody");
  if (watchBody) {
    watchBody.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-pp-unwatch]");
      if (btn) ppUnwatch(btn.dataset.ppUnwatch);
    });
  }
  const posBody = $("ppPositionsBody");
  if (posBody) {
    posBody.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-pp-close]");
      if (btn) ppClosePosition(btn.dataset.ppClose);
    });
  }
}
bindPaperEvents();

// 图表库(Chart.js)加载失败提示：CDN 被墙/断网时曲线图不可用但页面其余功能正常。
// 用一次性顶部提示而非弹窗（用户明确反感被动弹层）。
function ensureChartLib() {
  if (window.Chart || document.getElementById("chartLibNotice")) return;
  const strip = document.createElement("div");
  strip.id = "chartLibNotice";
  strip.style.cssText =
    "position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:400;" +
    "display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:10px;" +
    "border:1px solid rgba(251,191,36,.35);background:rgba(30,27,5,.92);" +
    "color:#fde68a;font-size:12px;box-shadow:0 6px 24px rgba(0,0,0,.4);";
  strip.innerHTML =
    '<span>📊</span><span>图表库 Chart.js 未能从 CDN 加载，曲线图暂时不可用。请检查网络/代理后刷新页面重试；其余功能不受影响。</span>';
  (document.body || document.documentElement).appendChild(strip);
  setTimeout(() => strip.remove(), 12000);
}

// 一次性全量同步（SSE 重连 / 回到前台时用；未经 SSE 门控，走 REST 保真）
function syncNow() {
  if (document.hidden) return;
  refreshOverview();
  if (currentPage === "backtest" || btActive) refreshBacktest();
  if (currentPage === "realtime" || rtEngineRunning) refreshRealtime();
  if (currentPage === "paper" || ppRunning) refreshPaper();
  syncExpRunBtn();
  if (currentPage === "train") refreshCompareExp();
}

function sseEnabled() {
  return !!(window.FBStream && window.FBStream.connected && !window.FBStream.unsupported);
}

function pollTick() {
  // 标签页隐藏：不轮询（切回可见时由 visibilitychange 立即补一次）
  if (document.hidden) return;
  const sseOk = sseEnabled();
  refreshOverview();
  if (currentPage === "backtest" || btActive) refreshBacktest();
  // 实时/模拟盘：SSE 连接且引擎在跑 → 状态由推送接管，REST 轮询让位
  if (!(sseOk && rtEngineRunning)) {
    if (currentPage === "realtime" || rtEngineRunning) refreshRealtime();
  }
  if (!(sseOk && ppRunning)) {
    if (currentPage === "paper" || ppRunning) refreshPaper();
  }
  syncExpRunBtn();
  // 对比实验状态面板在训练页：页面不可见时同样停轮询
  if (currentPage === "train") refreshCompareExp();

  // 自适应：训练/回测/对比在跑才 4s；SSE 在线且无任务 → 30s 慢调；否则 12s
  const busy = btActive || cmpActive || trainActive || (!sseOk && (rtEngineRunning || ppRunning));
  const nextMs = busy ? 4000 : (sseOk ? 30000 : 12000);
  window.__bgBusy = busy; // 背景动效引擎据此自动降帧/降耗
  if (busy !== pollWasBusy || nextMs !== pollIntervalMs) {
    pollWasBusy = busy;
    pollIntervalMs = nextMs;
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollTick, pollIntervalMs);
  }
}

// SSE 事件流接线：实时/模拟盘快照只在其页面可见时应用（省电）；
// 重连成功后做一次全量 resync 兜底；后端无 /api/events 时 FBStream 自动停用回退轮询
function wireSseStream() {
  if (!window.FBStream) return;
  FBStream.on("realtime", (st) => {
    if (st && currentPage === "realtime") applyRealtimeState(st);
  });
  FBStream.on("paper", (st) => {
    if (st && currentPage === "paper") applyPaperState(st);
  });
  FBStream.onConnect(() => {
    if (!document.hidden) syncNow();
  });
  FBStream.start();
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollWasBusy = true;
  pollIntervalMs = 4000;
  pollTimer = setInterval(pollTick, pollIntervalMs);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) pollTick();   // 回到前台立即刷新一次
  });
  wireSseStream();
}

// ─────────────────────────────────────────────────────────────
// 前后端版本一致性检测：新前端 + 旧后端时高亮提示，而非静默失效
// 后端接口清单（新增功能所依赖的 route，缺失即视为后端为旧版）
const COMPAT_REQUIRED_ROUTES = [
  { path: "/api/hold-policies", label: "持仓管理方案下拉（回测 / 模拟实盘 / 实时）" },
  { path: "/api/training/subset-preview", label: "训练数据范围与取样区间可视化" },
  { path: "/api/experiment/compare-status", label: "训练范围对比实验面板" },
  { path: "/api/paper/replay", label: "模拟实盘历史回放" },
];
let compatTimer = null;
let compatDismissed = false;

function missingCompatRoutes(routes) {
  const have = new Set((routes || []).map((r) => r.path));
  return COMPAT_REQUIRED_ROUTES.filter((r) => !have.has(r.path));
}

async function checkBackendCompat() {
  if (compatDismissed) return;
  const banner = $("verBanner");
  if (!banner) return;
  let missing = null;
  let unreachable = false;
  try {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), 8000);
    const res = await fetch(`${API}/api/routes`, { signal: ctrl.signal });
    clearTimeout(to);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    missing = missingCompatRoutes(data.routes || []);
  } catch (e) {
    unreachable = true;
  }

  const show = unreachable || (missing && missing.length > 0);
  if (!show) {
    if (compatTimer) {
      clearInterval(compatTimer);
      compatTimer = null;
    }
    banner.hidden = true;
    banner.innerHTML = "";
    return;
  }

  if (unreachable) {
    banner.innerHTML = `
      <span class="ver-banner-ico">⛔</span>
      <div class="ver-banner-body">
        <div class="ver-banner-title">无法连接后端服务（/api/routes 无响应）</div>
        <div class="ver-banner-hint">请确认 Web 服务已启动（<span class="retry-hint">本提示每 12 秒自动重试，恢复后消失</span>）</div>
      </div>
      <button class="ver-banner-close" data-close title="关闭">×</button>`;
  } else {
    const chips = missing
      .map((m) => `<code>${escHtml(m.path)}</code>（${escHtml(m.label)}）`)
      .join("、");
    banner.innerHTML = `
      <span class="ver-banner-ico">⚠️</span>
      <div class="ver-banner-body">
        <div class="ver-banner-title">页面已是新版本，但<b>后端仍为旧版本</b> —— 请重启 Web 服务</div>
        <div class="ver-banner-missing">后端缺少接口：${chips}</div>
        <div class="ver-banner-hint">上述功能当前不可用。重启服务后本提示会自动消失（<span class="retry-hint">每 12 秒自动检测</span>），持仓管理下拉、训练范围与回放等会立即生效。</div>
      </div>
      <button class="ver-banner-close" data-close title="关闭（本次不再提示）">×</button>`;
  }
  banner.hidden = false;
  const closeBtn = banner.querySelector(".ver-banner-close");
  if (closeBtn) {
    closeBtn.onclick = () => {
      compatDismissed = true;
      if (compatTimer) {
        clearInterval(compatTimer);
        compatTimer = null;
      }
      banner.hidden = true;
      banner.innerHTML = "";
    };
  }
  if (!compatTimer) {
    compatTimer = setInterval(checkBackendCompat, 12000);
  }
}

async function init() {
  checkBackendCompat(); // 非阻塞：前后端版本不匹配时尽早亮横幅
  try {
    await loadConfig();
    await refreshOverview();
  } catch (e) {
    await logClientError("初始化失败: " + e.message);
  }
  $("browseBtn").addEventListener("click", browseDataFile);
  $("dlFetchBtn").addEventListener("click", downloadAndPrepare);
  $("dlBackfillBtn").addEventListener("click", backfillToOrigin);
  $("dlSymbolInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      downloadAndPrepare();
    }
  });
  initDlPresets();
  $("manualPathBtn").addEventListener("click", applyManualPath);
  $("manualPathInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      applyManualPath();
    }
  });
  const dlFileList = $("dlFileListSelect");
  if (dlFileList) {
    dlFileList.addEventListener("change", (e) => {
      const v = e.target.value;
      if (v) applyDataFilePath(v);
      else e.target.value = "";
    });
  }
  $("startBtn").addEventListener("click", startTraining);
  if ($("retrainBtn")) $("retrainBtn").addEventListener("click", retrainFromScratch);
  ["trDataMode", "trNBars", "trChunks"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", refreshTrainRangeUI);
  });
  if ($("trNBars")) $("trNBars").addEventListener("input", refreshTrainRangeUI);
  refreshTrainRangeUI();
  if ($("expRunBtn")) $("expRunBtn").addEventListener("click", startCompareExp);
  if ($("expStopBtn")) $("expStopBtn").addEventListener("click", stopCompareExp);
  syncExpRunBtn();
  $("stopBtn").addEventListener("click", stopTraining);
  $("exportBtn").addEventListener("click", exportStrategy);
  $("exportTrainingBtn").addEventListener("click", exportTraining);
  $("importTrainingBtn").addEventListener("click", triggerImportTraining);
  $("importTrainingFile").addEventListener("change", handleImportTrainingFile);
  $("debugModeCheck").addEventListener("change", (e) => setDebugMode(e.target.checked));
  $("bgAnimCheck").addEventListener("change", (e) => setBgAnimation(e.target.checked));
  wireGlobalErrorCapture();
  wireDebugLogTools();
  if ($("aiApiKeyInput")) {
    $("aiApiKeyInput").addEventListener("input", updateAiChannelHint);
    $("aiApiKeyInput").addEventListener("change", updateAiChannelHint);
  }
  if ($("aiBaseUrlInput")) {
    $("aiBaseUrlInput").addEventListener("input", updateAiChannelHint);
    $("aiBaseUrlInput").addEventListener("change", updateAiChannelHint);
  }
  if ($("aiModelInput")) {
    $("aiModelInput").addEventListener("input", updateAiChannelHint);
    $("aiModelInput").addEventListener("change", updateAiChannelHint);
  }
  if ($("aiInspectBtn")) $("aiInspectBtn").addEventListener("click", runTrainInspectNow);
  if ($("aiInspectClearBtn")) $("aiInspectClearBtn").addEventListener("click", clearTrainInspections);
  document.querySelectorAll("[data-close-error]").forEach((el) => {
    el.addEventListener("click", closeErrorPopup);
  });
  bindVerifyRollback();
  if ($("errorModalCopyBtn")) {
    $("errorModalCopyBtn").addEventListener("click", copyErrorPopupDetail);
  }

  // 步骤导航
  document.querySelectorAll(".stepper .step").forEach((btn) => {
    btn.addEventListener("click", () => switchPage(btn.dataset.page));
  });

  // 回测控制
  if ($("btBrowseStrategyBtn")) $("btBrowseStrategyBtn").addEventListener("click", browseStrategyFile);
  btLoadDataFiles();
  const btStratList = $("btStrategyListSelect");
  if (btStratList) {
    btStratList.addEventListener("change", (e) => {
      const v = e.target.value;
      if (v) applyStrategyFilePath(v);
      else e.target.value = "";
      renderModelLibrary("btModelGrid", selectedStrategyFile);
    });
  }
  if ($("btStartBtn")) $("btStartBtn").addEventListener("click", startBacktest);
  if ($("btStopBtn")) $("btStopBtn").addEventListener("click", stopBacktest);
  ["btCommissionInput", "btSlippageInput"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("input", updateBtCostHint);
    el.addEventListener("change", () => { updateBtCostHint(); btSyncDualBtn(); }); // 成本也是双曲线参数：变了即释放旧图
  });
  // 每笔投入上限（%）：改动即持久化，供回测/矩阵/A·B/回放/模拟实盘共用
  ["btMaxPosInput", "ppMaxPosInput"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("change", async () => {
      let v = Number(el.value);
      if (!Number.isFinite(v) || v < 1) v = 100;
      v = Math.min(200, Math.max(1, Math.round(v)));
      el.value = String(v);
      try {
        await fetchJSON("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ max_position_pct: v }),
        });
        const other = $(id === "btMaxPosInput" ? "ppMaxPosInput" : "btMaxPosInput");
        if (other) other.value = String(v);
      } catch (_) {}
    });
  });

  // 统一无信号阈值：任一页面改动即持久化，并同步其余三个选择框（回测/回放/实时/模拟盘）
  ["btThresholdSelect", "rsThresholdSelect", "rtThresholdSelect", "ppThresholdSelect"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("change", async () => {
      const v = Number(el.value) || 0.05;
      applySigThrSelects(v);
      try {
        await fetchJSON("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ signal_threshold: v }),
        });
      } catch (_) {}
    });
  });

  // 实时分析控制
  if ($("rtSourceSelect")) $("rtSourceSelect").addEventListener("change", onRtSourceChange);
  if ($("rtStrategySelect")) $("rtStrategySelect").addEventListener("change", onRtStrategyChange);
  if ($("rtBrowseStrategyBtn")) $("rtBrowseStrategyBtn").addEventListener("click", rtBrowseStrategy);
  if ($("rtAddBtn")) $("rtAddBtn").addEventListener("click", rtAddWatch);
  if ($("tvBlockedLaterBtn")) $("tvBlockedLaterBtn").addEventListener("click", onTvBlockedLater);
  if ($("tvBlockedCloudBtn")) $("tvBlockedCloudBtn").addEventListener("click", onTvBlockedOpenCloud);
  document.querySelectorAll("[data-close-tv-blocked]").forEach((el) => {
    el.addEventListener("click", () => closeTvBlockedDialog("cancel"));
  });
  if ($("rtFeishuSaveBtn")) $("rtFeishuSaveBtn").addEventListener("click", saveRtFeishuSettings);
  if ($("rtFeishuTestBtn")) $("rtFeishuTestBtn").addEventListener("click", testRtFeishu);
  if ($("rtFeishuHelpBtn")) $("rtFeishuHelpBtn").addEventListener("click", openRtFeishuHelpModal);
  document.querySelectorAll("[data-close-feishu-help]").forEach((el) => {
    el.addEventListener("click", closeRtFeishuHelpModal);
  });
  if ($("rtGrid")) {
    $("rtGrid").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-remove]");
      if (btn) {
        rtRemoveWatch(btn.dataset.remove);
        return;
      }
      // 点“因子僵化”徽标 → 最近 50 根 因子 vs 价格 双轴轨迹
      const st = e.target.closest("[data-rt-stale]");
      if (st) showStaleTraceModal(rtLiveById[st.dataset.rtStale]);
    });
  }

  // 持仓管理方案 + 模型库 + 历史回放初始化
  loadHoldPolicies();
  const md = $("btMatrixDetails");
  if (md) {
    md.addEventListener("toggle", () => {
      if (md.open && !window.__btMatrixLoaded) {
        window.__btMatrixLoaded = true;
        loadComboGrid();
        loadHoldMatrix();
      }
    });
  }
  if ($("btMatrixRunBtn")) $("btMatrixRunBtn").addEventListener("click", runHoldMatrix);
  if ($("btMatrixStopBtn")) $("btMatrixStopBtn").addEventListener("click", stopHoldMatrix);
  ["btMatrixModeSelect", "btMatrixWindowInput", "btMatrixChunksInput", "btMatrixRegimeSelect", "btMatrixDataSelect"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", () => { matrixModeUi(); matrixPrefsSave(); });
  });
  // 上限%档位 / 阈值档位：只有初次打开网页才自动填充默认值；用户改过就用记忆值
  try {
    const saved = JSON.parse(localStorage.getItem("am.comboAxes") || "null");
    if (saved && typeof saved === "object") {
      const capEl = $("btMatrixCapsInput");
      const thrEl = $("btMatrixThrInput");
      if (capEl && saved.caps) capEl.value = saved.caps;
      if (thrEl && saved.thresholds) thrEl.value = saved.thresholds;
    }
  } catch (_) {}
  ["btMatrixCapsInput", "btMatrixThrInput"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("input", () => {
      try {
        const v = { caps: $("btMatrixCapsInput")?.value || "", thresholds: $("btMatrixThrInput")?.value || "" };
        if (v.caps.trim() || v.thresholds.trim()) localStorage.setItem("am.comboAxes", JSON.stringify(v));
        else localStorage.removeItem("am.comboAxes");
      } catch (_) {}
    });
  });
  matrixModeUi();
  btLoadMatrixDataFiles();
  matrixPrefsApply(selectedStrategySymbol);
  btLoadMatrixBest();
  btLoadCompare();
  if ($("ppDrillBtn")) $("ppDrillBtn").addEventListener("click", ppRunDrill);
  wireModelLibrary();
  renderModelLibrary("btModelGrid", selectedStrategyFile);
  renderModelLibrary("ppModelGrid", $("ppStrategySelect")?.value || "");
  loadReplayStrategies();
  rsLoadDataFiles();
  if ($("rsBrowseDataBtn")) $("rsBrowseDataBtn").addEventListener("click", rsBrowseDataFile);
  if ($("rsReplayBtn")) $("rsReplayBtn").addEventListener("click", rsRunReplay);
  if ($("rsCompareBtn")) $("rsCompareBtn").addEventListener("click", rsRunCompare);
  ["rsDataSelect", "rsStrategySelect", "rsPolicySelect", "rsWindowInput", "rsCommInput", "rsSlipInput"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", rsSyncCompareBtn);
  });
  // 回放页上限输入框：与回测/实盘共用同一设置（改动即持久化 + 双向同步）
  ["rsMaxPosInput"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("change", async () => {
      let v = Number(el.value);
      if (!Number.isFinite(v) || v < 1) v = 100;
      v = Math.min(200, Math.max(1, Math.round(v)));
      el.value = String(v);
      try {
        await fetchJSON("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ max_position_pct: v }),
        });
        ["btMaxPosInput", "ppMaxPosInput"].forEach((oid) => {
          const o = $(oid);
          if (o) o.value = String(v);
        });
      } catch (_) {}
    });
  });
  rsSyncCompareBtn();

  bindDdEvExport(); // DD 事件回溯面板：绑定导出 + 首拉（模拟实盘/实时分析页共用）

  // 回测页「同参双口径（回测 vs 回放）」：绑定按钮 + 参数变化时同步可用性
  const btDualBtn = $("btDualBtn");
  if (btDualBtn) {
    btDualBtn.addEventListener("click", btRunDualCompare);
    ["btPolicySelect", "btStackSelect", "btWindowInput", "btDataSelect", "btMaxPosInput", "btThresholdSelect"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("change", btSyncDualBtn);
    });
    btSyncDualBtn();
  }

  ensureChartLib();
  startPolling();
}

init();

// 入场动画一次性锁：初始加载时 panel/split 的渐入（最晚 stagger 延迟 .3s + 0.6s ≈ .9s 播完），
// 之后再给 <body> 加 anim-locked（style.css 据此禁掉 panel/split 动画重放），
// 保证来回切页时只有 .page 容器的轻量 pageIn 在播、不会被十余个 panel 的重淡入拖到掉帧。
setTimeout(() => {
  if (document.body) document.body.classList.add("anim-locked");
}, 1200);
