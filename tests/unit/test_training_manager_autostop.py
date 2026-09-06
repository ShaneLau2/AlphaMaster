"""TrainingManager 巡检接线单测：终态 flush + 「建议停止」自动停训。"""
from __future__ import annotations

from web import training_manager as tm_mod

DANGER = {
    "ts": "2026-09-05T01:00:00+00:00",
    "symbol": "BTCUSDT",
    "level": "danger",
    "title": "建议停止：465 步零提升 + 分布塌缩",
    "recommendation": "停滞超过阈值，可停止本轮。",
    "checks": [{"level": "danger", "text": "最优已停滞 465 步。"}],
    "metrics": {"step": 900, "total": 9000, "best": 2.9, "stall": 465},
}


class _FakeJob:
    symbol = "BTCUSDT"
    log_path = "logs/train_BTCUSDT_x.log"


class _FakeProc:
    """poll() 恒 None = 永远活着；terminate 置标记。"""

    def __init__(self) -> None:
        self._terminated = False

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self._terminated = True


class _DeadProc(_FakeProc):
    def poll(self) -> int | None:
        return 0


def _make_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(tm_mod, "PROJECT_ROOT", tmp_path)
    m = tm_mod.TrainingManager()
    m._job = _FakeJob()
    return m


def test_should_auto_stop_only_once_per_run(tmp_path, monkeypatch) -> None:
    m = _make_manager(tmp_path, monkeypatch)
    m._proc = _FakeProc()
    assert m._should_auto_stop(DANGER, alive_snapshot=True)
    assert m._should_auto_stop(DANGER, alive_snapshot=False) is False  # 进程死了不越权
    m._auto_stop_done = True
    assert m._should_auto_stop(DANGER, alive_snapshot=True) is False


def test_auto_stop_terminates_notifies_and_records(tmp_path, monkeypatch) -> None:
    import web.feishu_notify as fn
    import web.local_notify as ln

    m = _make_manager(tmp_path, monkeypatch)
    m._proc = _FakeProc()
    sent = []
    monkeypatch.setattr(fn, "send_text", lambda text, **kw: (sent.append(text), (True, "ok"))[1])
    ln_calls = []
    monkeypatch.setattr(ln, "notify", lambda title, message: ln_calls.append((title, message)) or {})

    m._auto_stop_for_entry(DANGER)
    assert m._proc._terminated
    assert m._auto_stop_done
    assert any(x.get("auto_stopped") for x in m._inspections)
    assert sent and "建议停止" in sent[0]


def test_auto_stop_noop_when_proc_dead(tmp_path, monkeypatch) -> None:
    import web.feishu_notify as fn

    m = _make_manager(tmp_path, monkeypatch)
    m._proc = _DeadProc()
    monkeypatch.setattr(fn, "send_text", lambda text, **kw: (False, "no webhook"))

    m._auto_stop_for_entry(DANGER)
    assert not m._proc._terminated
    assert m._auto_stop_done is False  # 进程已结束 → 不标记、不重复终止


def test_terminal_flush_writes_history(tmp_path, monkeypatch) -> None:
    m = _make_manager(tmp_path, monkeypatch)
    m._proc = _DeadProc()
    m._inspections = [DANGER]
    m._flush_terminal_history()
    hist = tmp_path / "training_history_BTCUSDT.json"
    assert hist.exists()
    import json

    data = json.loads(hist.read_text(encoding="utf-8"))
    assert data["inspections"][0]["event"] == "train_inspect"
    # 幂等：第二次 flush 不重复
    n = len(data["inspections"])
    m._flush_terminal_history()
    assert len(json.loads(hist.read_text(encoding="utf-8"))["inspections"]) == n
