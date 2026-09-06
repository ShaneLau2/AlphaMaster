"""Unit tests for scripts/train_gonogo.py.

History is shaped like real engine output: the engine atomically refreshes the
JSON every HISTORY_LIVE_EVERY_STEPS=50 steps, and batch-mean val_score sits at
real magnitudes (~0.2-0.5) while best_score is the frozen elite ceiling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_gonogo import evaluate, load_history  # noqa: E402


def _mk_history(
    max_step: int,                      # 最后一个写入的 step(含)
    best: float,
    val: float,
    top1: float,
    eff_vocab: float,
    best_frozen_at: int = 0,            # best 从该步起冻结(此前微量抬升)
    climb: float = 0.0,                 # 每步 batch-mean val 增量(模拟批次平均漂移)
    improve_at: int | None = None,      # 若给出:该步 best 再次刷新(+0.05)
) -> dict:
    """构造真实形状 history。

    引擎只在 step 是 refresh_every 整倍数时落盘整份 JSON(本函数不再模拟刷新
    滞后,直接返回满数组);best_score 是逐公式验证奖励的 running max,
    val_score 是 batch-mean(真实量级 ~0.3)。
    """
    n = max_step + 1
    steps = list(range(n))
    # best:best_frozen_at 之前每步微量抬升,之后冻结;improve_at 处再次刷新
    best_series = []
    for s in range(n):
        if improve_at is not None and s >= improve_at:
            best_series.append(float(best) + 0.05)
        else:
            best_series.append(float(best) if s >= best_frozen_at else float(min(best, 0.5 + s * 0.01)))
    # val: batch-mean 基础量 + 爬升项(独立于 best,模拟批次平均漂移)
    val_series = [float(val) + climb * s for s in range(n)]
    return {
        "step": steps,
        "best_score": best_series,
        "val_score": val_series,
        "top1_prob": [float(top1)] * n,
        "eff_vocab": [float(eff_vocab)] * n,
    }


def _write(tmp_path: Path, h: dict, symbol: str = "X") -> None:
    (tmp_path / f"training_history_{symbol}.json").write_text(json.dumps(h))


# ── 50 步刷新 + 边界时序 ─────────────────────────────────────────────────

def test_fires_at_exact_target_step(tmp_path: Path) -> None:
    # step 数组恰在 1000 收尾 → 必须判定,不能 wait
    h = _mk_history(1000, best=1.605, val=0.30, top1=0.99, eff_vocab=1.2,
                    best_frozen_at=100)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, history_dir=tmp_path)
    assert r["verdict"] == "kill"
    assert r["exit"] == 1


def test_jump_999_then_1049_across_refreshes(tmp_path: Path) -> None:
    # 引擎每 50 步刷新,on-disk step 以 49/99/.../999/1049 收尾(跳过 1000)。
    # 999 帧必须 wait(未到目标),下一帧 1049(>1000)必须 fire。
    h_before = _mk_history(999, best=1.605, val=0.30, top1=0.99, eff_vocab=1.2,
                           best_frozen_at=100)
    _write(tmp_path, h_before)
    r_before = evaluate("X", target_step=1000, history_dir=tmp_path)
    assert r_before["verdict"] == "wait"
    assert r_before["exit"] == 2

    h_at = _mk_history(1049, best=1.605, val=0.30, top1=0.99, eff_vocab=1.2,
                       best_frozen_at=100)
    _write(tmp_path, h_at)
    r_at = evaluate("X", target_step=1000, history_dir=tmp_path)
    assert r_at["verdict"] == "kill"
    assert r_at["exit"] == 1


def test_no_premature_fire_before_target(tmp_path: Path) -> None:
    # 即便条件已满足 GO,只要未到目标步就必须 wait(不会提前 kill/go)
    h = _mk_history(900, best=1.85, val=0.30, top1=0.99, eff_vocab=1.2,
                    best_frozen_at=100)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, history_dir=tmp_path)
    assert r["verdict"] == "wait"
    assert r["exit"] == 2


def test_no_premature_kill_before_target(tmp_path: Path) -> None:
    h = _mk_history(999, best=1.605, val=0.30, top1=0.99, eff_vocab=1.2,
                    best_frozen_at=100)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, history_dir=tmp_path)
    assert r["verdict"] == "wait"
    assert r["exit"] == 2


# ── 判定口径(GO/KILL) ───────────────────────────────────────────────────

def test_go_on_best_breakout(tmp_path: Path) -> None:
    h = _mk_history(1100, best=1.85, val=0.30, top1=0.5, eff_vocab=40.0,
                    best_frozen_at=500)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, threshold_best=1.8, history_dir=tmp_path)
    assert r["verdict"] == "go"
    assert r["exit"] == 0


def test_no_go_when_batch_val_climbs_but_best_flat(tmp_path: Path) -> None:
    # 失败模式:batch-mean val 爬升(climb>0)但 best_score 冻结(flat)。
    # 旧逻辑会误判 go;新逻辑只看 best_score 是否刷新 → 不得 go。
    # top1 低 → 非 kill,应为 wait(证据不足)。
    h = _mk_history(1100, best=1.605, val=0.30, top1=0.4, eff_vocab=45.0,
                    best_frozen_at=100, climb=0.0002)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, threshold_best=1.8,
                 threshold_top1=0.95, history_dir=tmp_path)
    assert r["verdict"] != "go"
    assert r["exit"] != 0


def test_no_go_when_batch_val_climbs_but_best_flat_top1_high(tmp_path: Path) -> None:
    # 同失败模式 + 系数塌缩 → kill(而非 go)
    h = _mk_history(1100, best=1.605, val=0.30, top1=0.99, eff_vocab=1.2,
                    best_frozen_at=100, climb=0.0002)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, threshold_best=1.8,
                 threshold_top1=0.95, history_dir=tmp_path)
    assert r["verdict"] == "kill"
    assert r["exit"] == 1


def test_go_when_best_improved_recently(tmp_path: Path) -> None:
    # 门控系列(best_score)近期刷新 → go(即使 batch-mean val 平坦)
    h = _mk_history(1100, best=1.605, val=0.30, top1=0.5, eff_vocab=40.0,
                    best_frozen_at=100, climb=0.0, improve_at=1020)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, threshold_best=1.8,
                 threshold_top1=0.95, history_dir=tmp_path)
    assert r["verdict"] == "go"
    assert r["exit"] == 0
    assert r["best_recent"] is True


def test_no_go_when_best_improved_too_long_ago(tmp_path: Path) -> None:
    # best 上次刷新在 improve_window(默认 200)之前 → 不再视为近期,不得 go
    h = _mk_history(1100, best=1.605, val=0.30, top1=0.5, eff_vocab=40.0,
                    best_frozen_at=100, climb=0.0002, improve_at=800)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, threshold_best=1.8,
                 threshold_top1=0.95, history_dir=tmp_path)
    assert r["verdict"] == "wait"  # top1 低 → 非 kill
    assert r["exit"] == 2


def test_kill_frozen_collapsed(tmp_path: Path) -> None:
    # best 冻结(≤阈值)+ top1 高(系数塌缩)→ kill
    h = _mk_history(1100, best=1.605, val=0.30, top1=0.99, eff_vocab=1.2,
                    best_frozen_at=100)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, threshold_best=1.8,
                 threshold_top1=0.95, history_dir=tmp_path)
    assert r["verdict"] == "kill"
    assert r["exit"] == 1


def test_kill_reason_points_to_family_not_gate_tuning(tmp_path: Path) -> None:
    h = _mk_history(1100, best=1.605, val=0.30, top1=0.99, eff_vocab=1.2,
                    best_frozen_at=100)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, history_dir=tmp_path)
    assert r["verdict"] == "kill"
    # 不得再给出「放松门/降步数」的死胡同建议
    assert "放松" not in r["reason"]
    assert "降 TRAIN_STEPS" not in r["reason"]
    # 应指向公式族 + 诊断脚本
    assert "公式族" in r["reason"]
    assert "vol_gate_diag.py" in r["reason"]


# ── 数据缺失 / 边界 ─────────────────────────────────────────────────────

def test_no_data(tmp_path: Path) -> None:
    r = evaluate("MISSING", target_step=1000, history_dir=tmp_path)
    assert r["verdict"] == "no-data"
    assert r["exit"] == 3


def test_missing_signals_after_target(tmp_path: Path) -> None:
    h = {"step": list(range(1001))}
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, history_dir=tmp_path)
    assert r["verdict"] == "no-data"
    assert r["exit"] == 3


def test_ambiguous_at_target_wait_manual(tmp_path: Path) -> None:
    # 目标步已到但证据不足以二分(如 top1 中间值 + best 冻结)→ wait(人工复核)
    h = _mk_history(1100, best=1.605, val=0.30, top1=0.6, eff_vocab=40.0,
                    best_frozen_at=100, climb=0.0)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, history_dir=tmp_path)
    assert r["verdict"] == "wait"
    assert r["exit"] == 2


def test_improve_window_parameter(tmp_path: Path) -> None:
    # 显式 improve_window 覆盖默认:best 在 300 步前刷新,窗口 500 → 仍算近期
    h = _mk_history(1100, best=1.605, val=0.30, top1=0.5, eff_vocab=40.0,
                    best_frozen_at=100, climb=0.0, improve_at=800)
    _write(tmp_path, h)
    r = evaluate("X", target_step=1000, threshold_best=1.8,
                 threshold_top1=0.95, improve_window=500, history_dir=tmp_path)
    assert r["verdict"] == "go"
    assert r["exit"] == 0


def test_load_history_handles_corrupt(tmp_path: Path) -> None:
    (tmp_path / "training_history_X.json").write_text("{ not json")
    assert load_history("X", tmp_path) == {}