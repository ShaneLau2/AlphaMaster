/* =========================================================================
   FBAPI · 后端服务地址运行时配置
   -------------------------------------------------------------------------
   背景: AlphaMaster 控制台既跑在本机 (http://127.0.0.1:8765, 同源), 也有
   GitHub Pages 公开静态镜像 (https://shanelau2.github.io/alphamaster-web/)。
   同一台 Mac 上打开镜像页时, 让前端连回本机后端即可「真正操作」。
   浏览器不允许公网 HTTPS 页面直连私有网段, 因此:
     · CORS 由后端放开 (web/app.py, allow_origins=["*"])
     · Private Network Access 预检由后端应答 Access-Control-Allow-Private-Network
     · 后端地址做成本页的运行时配置 (自动判断 + ?backend= 参数 + localStorage)

   解析优先级:
     1. URL 参数  ?backend=<url> 或 ?api=<url>   (显式指定, 最优先)
     2. localStorage 「alphamaster:apiBase」     (右上角「后端」弹层保存)
     3. 页面由本机控制台提供 (端口 8765 / localhost / 127.0.0.1) → 同源 ""
     4. 其它来源 (GitHub Pages 镜像等) → 默认 http://127.0.0.1:8765

   浏览器说明: Chrome / Edge / Firefox 允许 HTTPS 页面访问 http://127.0.0.1
   (环回地址被视为可信); Chrome 首次会弹一次「允许访问本地网络」授权。
   Safari 仍拦截混合内容 → 镜像页直连本机请用 Chrome/Edge/Firefox。
   ========================================================================= */
(() => {
  "use strict";
  if (window.FBAPI) return;

  const LS_KEY = "alphamaster:apiBase";
  const LS_TOKEN_KEY = "alphamaster:apiToken";
  const REMOTE_DEFAULT = "http://127.0.0.1:8765";
  const PROBE_INTERVAL_MS = 12000;
  const PROBE_TIMEOUT_MS = 4000;

  const $ = (id) => document.getElementById(id);

  // 只接受合法的 http(s) 绝对地址, 去掉结尾斜杠; 其它一律视为「无效/未配置」。
  function norm(v) {
    if (!v) return "";
    v = String(v).trim().replace(/\/+$/, "");
    return /^https?:\/\//i.test(v) ? v : "";
  }

  // 页面本身是否由本机控制台提供(同源直连, 无需跨域)。
  function isLocalOrigin() {
    const port = location.port || (location.protocol === "https:" ? 443 : 80);
    if (String(port) === "8765") return true;
    const h = (location.hostname || "").toLowerCase();
    return h === "localhost" || h === "127.0.0.1" || h === "::1" || h === "0.0.0.0";
  }

  const qs = new URLSearchParams(location.search);
  const fromQuery = norm(qs.get("backend") || qs.get("api"));
  const stored = norm(localStorage.getItem(LS_KEY));

  let state;
  if (fromQuery) {
    state = { base: fromQuery, kind: "query", desc: "由 URL 参数指定" };
  } else if (stored) {
    state = { base: stored, kind: "custom", desc: "手动配置(本地保存)" };
  } else if (isLocalOrigin()) {
    state = { base: "", kind: "local", desc: "本机控制台 · 同源自动" };
  } else {
    state = { base: REMOTE_DEFAULT, kind: "remote", desc: "默认连接本机后端" };
  }

  // base === "" 表示「同源」: API 地址 = 页面所在主机(本机控制台场景)。
  // ── API 令牌: 所有 /api 请求需携带 Authorization: Bearer <token> ──────
  // 令牌由后端生成 (web_token.txt), 首次访问 GET /api/auth/token 自动获取并
  // 保存在本浏览器; 该端点按来源放行(同源/局域网/已知镜像), 其它来源拿不到。
  let token = localStorage.getItem(LS_TOKEN_KEY) || "";

  async function fetchToken() {
    // 注意: 可能被 FBAPI 对象字面量构造期间(ready:)调用, 此时 window.FBAPI
    // 尚未赋值, 只能读闭包里的 state, 不能读 window.FBAPI。
    const base = state.base;
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 4000);
      const res = await origFetch(base + "/api/auth/token", {
        signal: ctrl.signal,
        cache: "no-store",
      });
      clearTimeout(t);
      if (res.ok) {
        const body = await res.json().catch(() => ({}));
        if (body && body.token) {
          token = String(body.token);
          localStorage.setItem(LS_TOKEN_KEY, token);
          return token;
        }
      }
    } catch (_) {
      /* 后端未启动 / 来源被拒: 稍后由探针/弹层重试 */
    }
    return "";
  }

  // 先抓原始 fetch, 再替换: 包装器为所有 API 请求补 Authorization 头
  // (app.js 的 fetchJSON 与各裸 fetch 调用全部经由 window.fetch, 一处覆盖)。
  const origFetch = window.fetch.bind(window);
  window.FBAPI = {
    base: state.base,
    kind: state.kind,
    get isRemote() {
      return !!state.base;
    },
    desc: state.desc,
    get token() {
      return token;
    },
    // 首次令牌获取: 页面加载即触发, 后续请求 await 它再发
    ready: token ? Promise.resolve(token) : fetchToken(),
  };

  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    const base = state.base;
    const isApi = base ? url.startsWith(base + "/api") : url.startsWith("/api");
    if (!isApi) return origFetch(input, init);
    return window.FBAPI.ready.then(() => {
      const headers = new Headers((init && init.headers) || {});
      if (token) headers.set("Authorization", "Bearer " + token);
      return origFetch(input, Object.assign({}, init, { headers }));
    });
  };

  /* ── 探针: 报告目标后端是否存活(仅跨域/远程时真正发请求) ─────────── */
  let dotEl = null;
  let textEl = null;
  let statusEl = null;
  let lastOk = null;
  let lastVersion = "";

  function hostLabel(base) {
    if (!base) return "本机";
    try {
      return new URL(base).host;
    } catch (_) {
      return base.replace(/^https?:\/\//, "");
    }
  }

  function setDot(cls) {
    if (dotEl) {
      dotEl.classList.remove("ok", "err");
      if (cls) dotEl.classList.add(cls);
    }
    lastOk = cls === "ok" ? true : cls === "err" ? false : null;
  }

  function statusHTML(inner) {
    if (statusEl) statusEl.innerHTML = inner;
  }

  async function probe() {
    const base = window.FBAPI.base;
    if (!base) {
      // 同源: 页面能打开 = 后端就在同端口; 只显示信息, 不发请求。
      setDot("ok");
      const p = textEl ? textEl.parentElement : null;
      if (p) p.title = "数据来自本机控制台(同源)";
      statusHTML('本机控制台提供页面与数据(<span class="ok">同源 · 正常</span>)');
      return;
    }
    // 令牌缺失(如后端稍后启动)时先补一次获取, 避免探针 401 误报
    if (!token) await fetchToken();
    let ok = false;
    let version = "";
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), PROBE_TIMEOUT_MS);
      const res = await fetch(base + "/api/health", {
        signal: ctrl.signal,
        cache: "no-store",
      });
      clearTimeout(t);
      if (res.ok) {
        const body = await res.json().catch(() => ({}));
        ok = true;
        version = body.version || "";
      }
    } catch (_) {
      /* 未连通 / 超时 */
    }
    setDot(ok ? "ok" : "err");
    const pill = textEl ? textEl.parentElement : null;
    if (pill) pill.title = ok ? `已连接 ${base}` : `无法连接 ${base}`;
    if (ok) {
      statusHTML(
        `已连接 <b>${hostLabel(base)}</b>${version ? " · 后端 v" + escapeHtml(version) : ""}<span class="ok"> ✓</span>`
      );
    } else {
      statusHTML(
        `无法连接 <b>${hostLabel(base)}</b><span class="err"> ✗</span> — 本机后端未启动?<br>` +
          `<span style="opacity:.7">请确认 LaunchAgent 已运行, 或换用本机控制台 http://127.0.0.1:8765</span>`
      );
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /* ── 右上角 pill + 配置弹层 ──────────────────────────────────────── */
  function refreshPillText() {
    if (!textEl) return;
    const base = window.FBAPI.base;
    textEl.textContent = "后端 · " + hostLabel(base);
  }

  function initUI() {
    const pillWrap = document.getElementById("apiPillWrap");
    if (!pillWrap) return;
    dotEl = document.getElementById("apiDot");
    textEl = document.getElementById("apiPillText");
    const popEl = document.getElementById("apiPop");
    const inputEl = document.getElementById("apiBaseInput");
    statusEl = document.getElementById("apiPopStatus");
    const tokenInput = document.getElementById("apiTokenInput");
    const tokenStatus = document.getElementById("apiTokenStatus");

    // 令牌行的展示函数(initUI 作用域, pill 点击处理器也会调用)
    const refreshTokenUI = () => {
      if (!tokenInput) return;
      tokenInput.value = token;
      if (tokenStatus) {
        tokenStatus.innerHTML = token
          ? `本机令牌已<span class="ok">配置</span> · 保存在本浏览器, /api 请求自动携带`
          : '<span class="err">未配置令牌</span> · 首次访问会自动获取; 失败时可粘贴本机 <code>web_token.txt</code> 内容';
      }
    };

    refreshPillText();
    const pill = document.getElementById("apiPill");
    if (pill) {
      pill.title = `数据后端: ${window.FBAPI.desc} — 点击配置`;
      pill.addEventListener("click", (e) => {
        e.stopPropagation();
        const willOpen = popEl.classList.contains("hidden");
        popEl.classList.toggle("hidden", !willOpen);
        if (pill) pill.setAttribute("aria-expanded", String(willOpen));
        if (willOpen) {
          inputEl.value = window.FBAPI.isRemote ? window.FBAPI.base : "";
          if (!token) fetchToken().then(refreshTokenUI);  // 后端恢复后自动补令牌
          refreshTokenUI();
          probe();
          inputEl.focus();
        }
      });
    }
    document.addEventListener("mousedown", (e) => {
      if (popEl.classList.contains("hidden")) return;
      if (!pillWrap.contains(e.target)) popEl.classList.add("hidden");
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") popEl.classList.add("hidden");
    });

    document.getElementById("apiSaveBtn").addEventListener("click", () => {
      const raw = inputEl.value.trim();
      const v = norm(raw);
      if (raw && !v) {
        statusHTML('<span class="err">地址无效</span> — 仅支持 http(s):// 开头, 例如 http://127.0.0.1:8765');
        return;
      }
      if (v) localStorage.setItem(LS_KEY, v);
      else localStorage.removeItem(LS_KEY);
      const url = new URL(location.href);
      url.searchParams.delete("backend");
      url.searchParams.delete("api");
      location.replace(url); // 带 ?backend= 时一并清掉, 重新按新配置加载
    });
    document.getElementById("apiResetBtn").addEventListener("click", () => {
      localStorage.removeItem(LS_KEY);
      const url = new URL(location.href);
      url.searchParams.delete("backend");
      url.searchParams.delete("api");
      location.replace(url);
    });

    /* ── API 令牌行 ─────────────────────────────────────────────── */
    refreshTokenUI();
    if (tokenInput) {
      document.getElementById("apiTokenSaveBtn").addEventListener("click", () => {
        const raw = tokenInput.value.trim();
        if (raw && raw.length < 16) {
          if (tokenStatus) tokenStatus.innerHTML = '<span class="err">令牌太短</span> · 请粘贴完整令牌';
          return;
        }
        token = raw;
        if (raw) localStorage.setItem(LS_TOKEN_KEY, raw);
        else localStorage.removeItem(LS_TOKEN_KEY);
        location.reload();
      });
      document.getElementById("apiTokenRegenBtn").addEventListener("click", async () => {
        if (!confirm("重新生成后, 本机其它设备/已保存的旧令牌立即失效。继续?")) return;
        const base = window.FBAPI.base;
        try {
          const res = await origFetch(base + "/api/auth/rotate", {
            method: "POST",
            headers: { Authorization: "Bearer " + token },
          });
          const body = await res.json().catch(() => ({}));
          if (res.ok && body && body.token) {
            token = String(body.token);
            localStorage.setItem(LS_TOKEN_KEY, token);
            if (tokenStatus) tokenStatus.innerHTML = '已<span class="ok">重新生成</span>并保存 ✓ 即将刷新';
            refreshTokenUI();
            setTimeout(() => location.reload(), 600);
          } else {
            if (tokenStatus) {
              tokenStatus.innerHTML = `<span class="err">重新生成失败</span> · ${escapeHtml(body && body.detail ? body.detail : "HTTP " + res.status)}`;
            }
          }
        } catch (_) {
          if (tokenStatus) tokenStatus.innerHTML = '<span class="err">网络错误</span> · 后端未启动?';
        }
      });
    }

    // 探针: 立即一次, 之后仅页面可见时定时(后台页不打扰)。
    probe();
    let timer = null;
    const arm = () => {
      if (timer) clearInterval(timer);
      timer = setInterval(probe, PROBE_INTERVAL_MS);
    };
    const disarm = () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    };
    const onVis = () => (document.hidden ? disarm() : (arm(), probe()));
    document.addEventListener("visibilitychange", onVis);
    arm();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initUI);
  } else {
    initUI();
  }
})();
