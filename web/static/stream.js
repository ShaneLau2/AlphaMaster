/* =========================================================================
   AlphaMaster · SSE 事件流客户端（web/events.py 的后端推送）
   ------------------------------------------------------------------------
   职责：
     · 连接 GET /api/events（EventSource），订阅实时/模拟盘状态变更推送
     · 事件名 = 域（realtime / paper / hello…），data = JSON，直接喂给
       既有渲染函数（它们自带签名守卫，重复/未变数据零成本）
     · 断线指数退避自动重连；连上后执行一次全量 resync（注册的 onConnect）
     · 45s 无任何消息 → 强制重建连接（服务端可能静默死掉）
     · 标签页隐藏时主动断开，回到前台再重连（省电）
   后端无订阅者时 watcher 自动停止——本脚本只在页面开着时才连，天然对齐。
   ========================================================================= */
(() => {
  "use strict";

  const WATCHDOG_MS = 15000;      // 看门狗检查周期
  const STALE_MS = 45000;         // 超过该时长无任何数据帧视为连接僵死
  const RECONNECT_BASE_MS = 1000;
  const RECONNECT_MAX_MS = 8000;
  // 后端推送的都是「命名事件」（event: realtime / paper / hello…）。按 SSE 规范，
  // 命名事件只派发给 addEventListener(type) 监听器，永远不会触发 onmessage ——
  // 只挂 onmessage 会静默丢掉全部推送，必须逐一注册命名监听。
  const NAMED_EVENTS = ["hello", "realtime", "paper"];

  let es = null;
  let started = false;
  let closedByUs = false;
  let connected = false;
  let reconnectDelay = RECONNECT_BASE_MS;
  let lastMsgAt = 0;
  let watchdogTimer = null;
  let failureStreak = 0;      // 连续失败次数（后端无 /api/events 时不再无限重试）
  const FAILURE_LIMIT = 6;
  const handlers = {};    // eventName -> [fn(data)]
  const connectFns = [];  // 每次（重）连成功后执行一次全量 resync

  function notifyHandlers(evt, data) {
    const fns = handlers[evt];
    if (!fns) return;
    for (const fn of fns.slice()) {
      try {
        fn(data);
      } catch (e) {
        // 单个处理器异常不影响事件流
      }
    }
  }

  async function open() {
    if (!started || closedByUs || es) return;
    // 与 REST 一致: 后端地址取 api-base.js 解析的运行时 base (同源为空串)。
    // EventSource 无法带自定义请求头, 令牌走 ?token= 查询参数 (后端同样校验)。
    const fb = window.FBAPI;
    const base = fb ? fb.base : "";
    await (fb && fb.ready ? fb.ready : Promise.resolve());
    if (!started || closedByUs || es) return;  // await 期间可能已被 stop/重开
    const tok = fb && fb.token ? fb.token : "";
    const qs = tok ? "?token=" + encodeURIComponent(tok) : "";
    try {
      es = new EventSource(base + "/api/events" + qs);
    } catch (_) {
      scheduleReconnect();
      return;
    }
    es.onopen = () => {
      connected = true;
      failureStreak = 0;
      reconnectDelay = RECONNECT_BASE_MS;
      notifyHandlers("hello", { connected: true });
      for (const fn of connectFns.slice()) {
        try { fn(); } catch (_) { /* resync 失败由下一次推送/轮询兜底 */ }
      }
    };
    // 任何数据帧（命名事件或无名 message）都更新 liveness 并按事件名分发。
    // 心跳以无名 data 帧发送（会触发 message 事件）；纯注释 `: keepalive`
    // 在 JS 层不可见，无法用来判断连接死活，故后端不再发注释心跳。
    // hb 无名帧经上面的 "message" 监听路由，同时刷新 lastMsgAt 喂看门狗。
    const route = (ev) => {
      lastMsgAt = Date.now();
      let data = null;
      try { data = JSON.parse(ev.data || "{}"); } catch (_) { data = {}; }
      notifyHandlers(ev.type || "message", data);
    };
    for (const name of NAMED_EVENTS) {
      es.addEventListener(name, route);
    }
    es.addEventListener("message", route);
    es.onerror = () => {
      if (es) { es.close(); es = null; }
      connected = false;
      failureStreak += 1;
      if (failureStreak >= FAILURE_LIMIT) {
        // 连续多次失败（多半是旧后端没有 /api/events）：停止自动重连，
        // 前端 pollTick 自然回退到 REST 轮询；不打扰用户、不无限打请求。
        if (!FBStream.unsupported) {
          FBStream.unsupported = true;
          stop();
        }
        return;
      }
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (!started || closedByUs || es) return;
    connected = false;
    const delay = reconnectDelay;
    reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
    setTimeout(open, delay);
  }

  function watchdog() {
    if (!started || closedByUs) return;
    if (!es) {
      // 一直没连上：按退避重试（onerror 已负责，这里兜底）
      open();
      return;
    }
    if (Date.now() - lastMsgAt > STALE_MS) {
      es.close();
      es = null;
      connected = false;
      scheduleReconnect();
    }
  }

  function onVisibility() {
    if (document.hidden) {
      if (es) {
        closedByUs = true;
        es.close();
        es = null;
        connected = false;
      }
    } else if (started && !es) {
      closedByUs = false;
      reconnectDelay = RECONNECT_BASE_MS;
      open();
    }
  }

  function start() {
    if (started) return;
    started = true;
    lastMsgAt = Date.now();
    open();
    watchdogTimer = setInterval(watchdog, WATCHDOG_MS);
    document.addEventListener("visibilitychange", onVisibility);
  }

  function stop() {
    started = false;
    closedByUs = true;
    if (es) { es.close(); es = null; }
    connected = false;
    if (watchdogTimer) { clearInterval(watchdogTimer); watchdogTimer = null; }
    document.removeEventListener("visibilitychange", onVisibility);
  }

  const FBStream = {
    get connected() { return connected; },
    on(evt, fn) {
      (handlers[evt] = handlers[evt] || []).push(fn);
      return FBStream;
    },
    onConnect(fn) {
      connectFns.push(fn);
      return FBStream;
    },
    start,
    stop,
  };

  window.FBStream = FBStream;
})();
