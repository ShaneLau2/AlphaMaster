"""×→消失延迟回归：真实浏览器里点「实时分析卡 ×」/「模拟实盘行 移除」，量 DOM 消失耗时。

背景：移除监控曾因前端等服务端往返（unwatch + status，且 status 内串行拉现价）
而卡顿 ~1s。修复 = 乐观移除：先本地删行/删卡（同步），服务端异步收尾。

本文件把“点 × 到卡片消失”的耗时做成自动化回归：若未来有人把移除改回
「先 await 服务端再更新 DOM」，点击后的同一帧内卡片仍在 → gone=False /
耗时超阈值 → 测试直接 FAIL，而不是等网络超时才暴露。

放在独立目录 tests/perf（默认全量回归 tests/unit tests/smoke 不含它），依赖：
    .venv/bin/python -m playwright --version        # Python Playwright 已装
    本机 Chrome（channel=chrome，无需下载浏览器）
跑法：
    .venv/bin/python -m pytest tests/perf/test_unwatch_latency.py -q
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS = PROJECT_ROOT / "results" / "_perf_web_settings.json"
_STRATEGY = PROJECT_ROOT / "strategies" / "best_BTCUSDT.json"
# 宽容阈值：乐观路径是同步 DOM 更新（毫秒级），500ms 只拦“又变回等服务端”。
_LATENCY_MS = 500


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


@pytest.fixture(scope="session")
def server() -> str:
    """隔离的测试服务：重定向 settings 到临时文件，不污染真实 web_settings.json。"""
    pytest.importorskip("playwright", reason="需要 Python Playwright")
    if not _STRATEGY.exists():
        pytest.skip(f"缺少策略文件 {_STRATEGY}")
    port = _free_port()
    if _SETTINGS.exists():
        _SETTINGS.unlink()
    wrapper = (
        "import web.settings as s\n"
        "from pathlib import Path\n"
        f"s.SETTINGS_PATH = Path(r'{_SETTINGS}')\n"
        "import uvicorn\n"
        "from web.app import app\n"
        f"uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", wrapper],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=2) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(1)
    if not ready:
        proc.kill()
        raise RuntimeError(f"测试服务未在 {port} 就绪（exit={proc.poll()}），日志被丢弃——请手动 uvicorn 排查")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    if _SETTINGS.exists():
        _SETTINGS.unlink()


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Chrome 启动失败（channel=chrome）：{exc}") from exc
        yield b
        b.close()


def _measure_remove(page, q_selector: str):
    """同帧量：点击移除按钮到对应 DOM 节点消失的耗时（ms）。"""
    js = """
    (q) => {
      const btn = document.querySelector(q);
      if (!btn) return {gone: false, ms: -1, why: 'no-btn'};
      const t0 = performance.now();
      btn.click();
      const gone = !document.querySelector(q);
      return {gone: gone, ms: performance.now() - t0};
    }
    """
    return page.evaluate(js, q_selector)


def _print_measurement(name: str, res: dict) -> None:
    """发版检查清单里要打印实测毫秒数（-s 运行可见）。"""
    print(f"[perf] {name}: 点击→DOM消失 = {res.get('ms', -1):.1f} ms")


def test_realtime_card_remove_latency(server: str, browser) -> None:
    """信号雷达卡点 × → 卡消失耗时 < 阈值（乐观移除）。"""
    payload = {
        "source": "binance", "symbol": "BTCUSDT", "timeframe": "5m",
        "strategy_file": str(_STRATEGY),
    }
    watch = _post(server, "/api/realtime/watch", payload)["watch"]
    wid = watch["id"]
    page = None
    try:
        page = browser.new_page()
        page.goto(server + "/", timeout=60000)
        page.evaluate("switchPage('realtime')")
        page.wait_for_selector(f'.rt-card[data-id="{wid}"]', timeout=30000)
        q = f'.rt-card[data-id="{wid}"] .rt-remove'
        res = _measure_remove(page, q)
        _print_measurement("信号雷达 ×→消失", res)
        assert res["gone"], f"点击后卡片仍在（乐观移除失效？）: {res}"
        assert res["ms"] >= 0, f"按钮未找到: {res}"
        assert res["ms"] < _LATENCY_MS, (
            f"×→消失耗时 {res['ms']:.0f}ms 超过 {_LATENCY_MS}ms —— 移除被服务端往返拖慢了"
        )
    finally:
        if page is not None:
            page.close()
        try:
            _post(server, "/api/realtime/unwatch", {"id": wid})
        except Exception:
            pass


def test_paper_row_remove_latency(server: str, browser) -> None:
    """模拟实盘行点「移除」→ 行消失耗时 < 阈值（乐观移除）。"""
    payload = {
        "source": "binance", "symbol": "BTCUSDT", "timeframe": "5m",
        "strategy_file": str(_STRATEGY), "policy_id": "signal",
    }
    watch = _post(server, "/api/paper/watch", payload)["watch"]
    wid = watch["id"]
    page = None
    try:
        page = browser.new_page()
        page.goto(server + "/", timeout=60000)
        page.evaluate("switchPage('paper')")
        page.wait_for_selector(f'[data-pp-unwatch="{wid}"]', timeout=30000)
        q = f'[data-pp-unwatch="{wid}"]'
        res = _measure_remove(page, q)
        _print_measurement("模拟实盘行 移除→消失", res)
        assert res["gone"], f"点击后行仍在（乐观移除失效？）: {res}"
        assert res["ms"] >= 0, f"按钮未找到: {res}"
        assert res["ms"] < _LATENCY_MS, (
            f"移除→行消失耗时 {res['ms']:.0f}ms 超过 {_LATENCY_MS}ms —— 移除被服务端往返拖慢了"
        )
    finally:
        if page is not None:
            page.close()
        try:
            _post(server, "/api/paper/unwatch", {"id": wid})
        except Exception:
            pass
