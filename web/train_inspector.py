"""训练巡检（内置 Agent，无需外部 API Key）。

替代旧的「AI 分析当前训练情况」（DeepSeek/openclaw Key 接入）：
在训练过程中由后台守护线程**不定时**读取训练日志输出，用确定性规则
解析 步进/奖励/验证/IC/熵/多样性/停滞/重启 等指标并给出诊断：

- 进度：step / 总步数
- 停滞：自上次最优以来的步数占比 → 平台期提示
- 分布塌缩：有效词汇过低 / 最高概率过高 / KL上步≈0 → 策略确定性退化
- 与线上冠军对比：读 strategies/best_{symbol}.json 的 best_score，
  best 持平/超越/落后 → “是否值得继续”的判断依据
- 结论：recommend_continue（继续 / 可继续观察 / 建议停止）

本模块无任何外部依赖（stdlib + pathlib），可单测。
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROGRESS_RE = re.compile(
    r"\[(?P<step>\d+)/(?P<total>\d+)\]\s*"
    r"(?:市场=\[[^\]]*\]\s*)?"
    r"新公式=\d+\s+精英=\d+\s*\|\s*"
    r"有效=(?P<valid>\d+)\s+无效=\d+\s+常数=\d+\s*\|\s*"
    r"奖励=(?P<reward>-?[\d.]+)\s+验证=(?P<val>[\d.]+)[\s|]+"
    r"IC=(?P<ic>[\d.]+)[\s|]+"
    r"熵=(?P<entropy>[\d.]+)(?:\(系数=(?P<ent_coef>[\d.]+)\))?\s*\|\s*"
    r"最优=(?P<best>[\d.]+)\s+停滞=(?P<stall>\d+)\s+精英池=\d+\s+重启=(?P<restarts>\d+)"
)

_DIST_RE = re.compile(
    r"分布:\s*初始熵=(?P<init_entropy>[\d.]+)\s+"
    r"KL均匀=(?P<kl_uniform>[\d.]+)\s+"
    r"KL上步=(?P<kl_prev>[\d.]+)"
    r"(?:\s+最高概率=(?P<top_p>[\d.]+))?"
    r"(?:\s+前五概率=(?P<top5_p>[\d.]+))?"
    r"(?:\s+有效词汇=(?P<eff_vocab>[\d.]+))?"
    r"(?:\s+标准差=(?P<std>[\d.]+))?"
    r"(?:\s*\|\s*本批:\s*唯一符号=[\d.]+/[\d.]+\s+唯一公式=[\d.]+/[\d.]+\s+多样性=(?P<diversity>[\d.]+))?"
)

CHAMPION_THRESHOLD_REL = 0.005  # best 与冠军相对差 < 0.5% 视为“持平/同公式”
STALL_WARN_STEPS = 200
STALL_DANGER_STEPS = 400
COLLAPSE_EFF_VOCAB = 2.0
COLLAPSE_TOP_P = 0.95


def parse_progress(line: str) -> dict[str, Any] | None:
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    d = m.groupdict()
    return {k: float(v) if k in ("reward", "val", "ic", "entropy", "ent_coef", "best") else int(v)
            for k, v in d.items() if v is not None}


def parse_distribution(line: str) -> dict[str, Any] | None:
    m = _DIST_RE.search(line)
    if not m:
        return None
    d = m.groupdict()
    return {k: float(v) for k, v in d.items() if v is not None}


def read_champion_score(symbol: str, strategies_dir: Path) -> float | None:
    """读已部署冠军 best_{symbol}.json 的 best_score；无文件/解析失败返回 None。"""
    path = strategies_dir / f"best_{symbol}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        v = data.get("best_score")
        return float(v) if isinstance(v, (int, float)) else None
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _recent(items: list[dict[str, Any]], key: str, n: int = 8) -> list[float]:
    return [x[key] for x in items[-n:] if x.get(key) is not None]


def diagnose(
    rows: list[dict[str, Any]],
    *,
    champion_score: float | None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """对解析出的最近日志行做一次巡检，返回结构化结论。

    rows: parse_progress 的输出（可含 merge 的 dist 字段）。
    """
    checks: list[dict[str, str]] = []
    now = datetime.now(timezone.utc).isoformat()

    if not rows:
        return {
            "ts": now,
            "level": "info",
            "title": "暂无解析到训练进度（日志尚未输出或已结束）",
            "checks": [],
            "recommendation": "等待训练输出，或重开训练后自动巡检。",
            "metrics": {},
        }

    last = rows[-1]
    step = int(last.get("step", 0))
    total = int(last.get("total", step) or step)
    best = last.get("best")
    stall = int(last.get("stall", 0))
    restarts = int(last.get("restarts", 0))
    pct = round(100.0 * step / total, 1) if total else None

    metrics: dict[str, Any] = {
        "step": step,
        "total": total,
        "pct": pct,
        "best": best,
        "stall": stall,
        "restarts": restarts,
        "entropy": last.get("entropy"),
        "ent_coef": last.get("ent_coef"),
        "ic": last.get("ic"),
        "val": last.get("val"),
        "reward": last.get("reward"),
    }
    dist = last.get("_dist") or {}
    for k in ("eff_vocab", "top_p", "kl_prev", "init_entropy", "diversity"):
        if dist.get(k) is not None:
            metrics[k] = dist[k]

    # ── 1) 停滞（平台期）────────────────────────────────────────────
    level = "ok"
    if stall >= STALL_DANGER_STEPS:
        level = "danger"
        checks.append({
            "level": "danger",
            "text": f"最优已停滞 {stall} 步（占已跑 {step} 步的 {100.0 * stall / max(step, 1):.0f}%），"
                    f"超过 {STALL_DANGER_STEPS} 步阈值，处于明显平台期。",
        })
    elif stall >= STALL_WARN_STEPS:
        level = "warn"
        checks.append({
            "level": "warn",
            "text": f"最优已停滞 {stall} 步（占已跑 {step} 步的 {100.0 * stall / max(step, 1):.0f}%）。",
        })
    if restarts >= 5:
        checks.append({
            "level": "warn",
            "text": f"已重启 {restarts} 次仍未突破当前最优 —— 大概率是配置/数据层面的瓶颈，而非偶发。",
        })

    # ── 2) 分布塌缩 ────────────────────────────────────────────────
    eff = _mean(_recent([r for r in rows if r.get("_dist")], "eff_vocab")) or dist.get("eff_vocab")
    top_p = _mean(_recent([r for r in rows if r.get("_dist")], "top_p")) or dist.get("top_p")
    kl_prev = _mean(_recent([r for r in rows if r.get("_dist")], "kl_prev")) or dist.get("kl_prev")
    diversity = _mean(_recent([r for r in rows if r.get("_dist")], "diversity")) or dist.get("diversity")
    collapsed = (
        eff is not None and eff <= COLLAPSE_EFF_VOCAB
        and top_p is not None and top_p >= COLLAPSE_TOP_P
        and (kl_prev is None or kl_prev < 0.01)
    )
    if collapsed:
        checks.append({
            "level": "warn",
            "text": f"分布塌缩：有效词汇≈{eff:.1f}/127、最高概率 {top_p:.3f}、KL上步≈{kl_prev:.4f}"
                    f" —— 采样几乎确定，难以再探索到新公式。",
        })
    if diversity is not None and diversity < 0.5:
        checks.append({
            "level": "warn",
            "text": f"批内多样性低（{diversity:.2f}），本批公式高度雷同。",
        })

    # ── 3) 与线上冠军对比 ───────────────────────────────────────────
    champion_txt = "—"
    if champion_score is not None and best is not None:
        rel = (best - champion_score) / max(abs(champion_score), 1e-9)
        champion_txt = f"{champion_score:.4f}"
        if rel > CHAMPION_THRESHOLD_REL:
            checks.append({
                "level": "ok",
                "text": f"当前最优 {best:.4f} 已超过线上冠军 {champion_score:.4f}（+{100 * rel:.2f}%），"
                        f"通过闸门后即可部署。",
            })
        elif abs(rel) <= CHAMPION_THRESHOLD_REL:
            checks.append({
                "level": "warn",
                "text": f"当前最优 {best:.4f} 与线上冠军 {champion_score:.4f} 持平/同公式 —— "
                        f"继续跑出的公式若无法显著超越，不会带来新价值。",
            })
        else:
            checks.append({
                "level": "info",
                "text": f"当前最优 {best:.4f} 仍落后线上冠军 {champion_score:.4f}（{-100 * rel:.2f}%）。",
            })

    # ── 4) 近期 IC / 验证趋势（最新 vs 前半段）──────────────────────
    ic_now = _mean(_recent(rows, "ic", 6))
    val_now = _mean(_recent(rows, "val", 6))
    ic_early = _mean([r["ic"] for r in rows[:6] if r.get("ic") is not None])
    if ic_now is not None and ic_early is not None and ic_now < ic_early * 0.5:
        checks.append({
            "level": "info",
            "text": f"IC 近期下滑（早期≈{ic_early:.4f} → 近批≈{ic_now:.4f}），信号质量在变弱。",
        })

    metrics.update({
        "ic_now": ic_now, "val_now": val_now, "champion": champion_txt,
        "eff_vocab_avg": eff, "top_p_avg": top_p, "kl_prev_avg": kl_prev,
        "diversity_avg": diversity,
    })

    # ── 5) 结论：是否值得继续 ───────────────────────────────────────
    danger = any(c["level"] == "danger" for c in checks)
    warns = [c for c in checks if c["level"] == "warn"]
    if danger and collapsed:
        verdict_level = "danger"
        title = f"建议停止：{stall} 步零提升 + 分布塌缩"
        recommendation = ("继续跑到结束大概率只是空转。建议停止本轮；若想超越线上冠军，"
                          "换 seed / 提高熵系数上限 / 换数据范围（tail/spread）或调整词汇后再开一轮。")
    elif danger:
        verdict_level = "danger"
        title = f"建议停止：{stall} 步无新最优"
        recommendation = "停滞超过阈值，重启多次无果。可停止本轮并保留已部署冠军。"
    elif not warns and champion_score is not None and best is not None and best < champion_score:
        verdict_level = "ok"
        title = f"运行正常 · 尚未超越冠军（步 {step}/{total}）"
        recommendation = "进度健康且仍在提升/保持探索，可继续跑；以闸门是否通过为部署依据。"
    elif warns:
        verdict_level = "warn"
        title = f"值得关注：已停滞 {stall} 步（步 {step}/{total}）"
        recommendation = "若停滞继续扩大或出现塌缩信号，建议停止；否则观察重启后熵是否回升。"
    else:
        verdict_level = level if level in ("ok", "warn", "danger") else "ok"
        title = f"运行正常（步 {step}/{total}{'，' + str(pct) + '%' if pct is not None else ''}）"
        recommendation = "无异常信号，可继续训练。"

    return {
        "ts": now,
        "symbol": symbol,
        "level": verdict_level,
        "title": title,
        "checks": checks,
        "recommendation": recommendation,
        "metrics": metrics,
    }


def inspect_log_tail(
    lines: list[str],
    *,
    champion_score: float | None,
    symbol: str | None = None,
) -> dict[str, Any] | None:
    """解析日志尾部（进度行 + 紧随的分布行配对），跑一次诊断。无进度行返回 None。"""
    rows: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        prog = parse_progress(lines[i])
        if prog:
            if i + 1 < len(lines):
                dist = parse_distribution(lines[i + 1])
                if dist:
                    prog["_dist"] = dist
            rows.append(prog)
        i += 1
    if not rows:
        return None
    # 收敛到最多 ~60 行样本，避免全量扫描噪音
    if len(rows) > 60:
        rows = rows[-60:]
    return diagnose(rows, champion_score=champion_score, symbol=symbol)


def now_ms() -> int:
    return int(time.time() * 1000)


# ── 巡检结论持久化（供回溯：champion_history / training_history 侧车）────────
#
# 只写“紧凑事件”，不把超大 metrics 全量塞进档案；写入方（TrainingManager）负责
# 编排时机与文件锁：
#   - champion_history.json：训练运行中即可追加重载结论（deploy/reject 之外的
#     event="train_inspect" 事件，append-only）；
#   - training_history_{symbol}.json：仅引擎退出后（terminal）flush 一次，避免
#     与引擎逐 step 的整文件写入互踩。


def _compact_metrics(m: dict) -> dict:
    """只挑巡检结论里值得回溯的指标子集，避免塞爆档案。"""
    keys = (
        "step", "total", "pct", "best", "stall", "restarts",
        "champion", "ic_now", "val_now", "eff_vocab_avg", "top_p_avg",
        "kl_prev_avg", "diversity_avg", "entropy", "ic", "val", "reward",
    )
    return {k: m[k] for k in keys if m.get(k) is not None}


def build_inspect_event(entry: dict) -> dict | None:
    """把一次巡检 entry 转成紧凑可回溯事件（无 symbol 或非巡检结果则 None）。"""
    symbol = entry.get("symbol") if isinstance(entry.get("symbol"), str) else None
    if not symbol:
        return None
    return {
        "ts": entry.get("ts") or datetime.now(timezone.utc).isoformat(),
        "event": "train_inspect",
        "symbol": symbol,
        "level": entry.get("level"),
        "title": entry.get("title"),
        "recommendation": entry.get("recommendation"),
        "checks": [c.get("text") for c in (entry.get("checks") or []) if isinstance(c, dict) and c.get("text")],
        "metrics": _compact_metrics(entry.get("metrics") or {}),
    }
    # 回溯时保留公式解读（紧凑字段，不塞 parts 大数组）
    fml = entry.get("formula")
    if isinstance(fml, dict):
        if fml.get("expression"):
            ev["formula_expression"] = fml["expression"]
        if fml.get("summary"):
            ev["formula_summary"] = fml["summary"]
        if fml.get("decoded"):
            ev["formula_decoded"] = fml["decoded"]
    return ev


def _inspect_fs_lock(path: Path):
    """对 `<path>.lock` 加排他 flock（与引擎 _champion_lock_ctx 同锁名，跨进程互斥）。

    无 fcntl 平台（Windows）降级为无锁。
    """
    import contextlib

    try:
        import fcntl
    except Exception:  # noqa: BLE001
        fcntl = None

    @contextlib.contextmanager
    def _cm():
        if fcntl is None:
            yield
            return
        lock_path = Path(str(path) + ".lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a+") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except OSError:
            yield

    return _cm()


def append_champion_inspect_event(champion_path: Path, entry: dict, *, max_inspect: int = 200) -> bool:
    """向 strategies/champion_history.json 追加一次巡检事件（原子读改写）。

    只裁剪旧的 event="train_inspect" 条目（保留 deploy/reject 及其它历史事件），
    防止训练巡检高频追加把档案撑爆，同时不回滚冠军部署记录。
    """
    ev = build_inspect_event(entry)
    if ev is None:
        return False
    try:
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        with _inspect_fs_lock(champion_path):
            hist: list[dict] = []
            if champion_path.exists():
                try:
                    data = json.loads(champion_path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        hist = data
                except (json.JSONDecodeError, OSError):
                    hist = []
            hist.append(ev)
            # 只裁剪 train_inspect：保留全部非巡检条目（deploy/reject/其他）
            others = [e for e in hist if e.get("event") != "train_inspect"]
            inspects = [e for e in hist if e.get("event") == "train_inspect"]
            if len(inspects) > max_inspect:
                inspects = inspects[-max_inspect:]
            tmp = str(champion_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(others + inspects, fp, ensure_ascii=False, indent=2)
            os.replace(tmp, champion_path)
        return True
    except OSError:
        return False


def merge_inspections_into_history(history_path: Path, entries: list[dict], *, max_keep: int = 50) -> int:
    """把巡检结论合并进 training_history_{symbol}.json 顶层 key "inspections"。

    保留原文件全部曲线 key；只增改 inspections 数组（新→旧倒序，cap 后写回）。
    仅在引擎已退出的终态调用，避免与引擎逐 step 整文件覆盖互踩。返回写入条数。
    """
    evs = []
    for e in entries:
        ev = build_inspect_event(e)
        if ev:
            evs.append(ev)
    if not evs:
        return 0
    history_path = Path(history_path)
    try:
        with _inspect_fs_lock(history_path):
            payload: dict = {}
            if history_path.exists():
                try:
                    data = json.loads(history_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        payload = data
                except (json.JSONDecodeError, OSError):
                    payload = {}
            merged = evs + list(payload.get("inspections") or [])[: max_keep - len(evs)]
            payload["inspections"] = merged[:max_keep]
            history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(history_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False)
            os.replace(tmp, history_path)
        return len(evs)
    except OSError:
        return 0


def verdict_recommends_stop(entry: dict | None) -> bool:
    """该巡检结论是否明确给出「建议停止」（危险级 + 标题含 建议停止）。"""
    if not isinstance(entry, dict):
        return False
    if entry.get("level") != "danger":
        return False
    return bool(entry.get("title")) and "建议停止" in str(entry.get("title"))


def build_stop_notice(entry: dict, symbol: str | None = None) -> str:
    """把一次「建议停止」巡检整理成人类可读的停止说明（飞书推送 + 记录共用）。"""
    sym = symbol or entry.get("symbol") or "?"
    m = entry.get("metrics") or {}
    step = m.get("step")
    total = m.get("total")
    step_s = f"{step}/{total}" if step is not None else "—"
    lines = [
        f"【AlphaMaster 训练巡检 · 自动停止】",
        f"{sym} · 步 {step_s}",
        f"结论：{entry.get('title') or '建议停止'}",
    ]
    checks = [c.get("text") for c in (entry.get("checks") or []) if isinstance(c, dict) and c.get("text")]
    for c in checks[:4]:
        lines.append(f"• {c}")
    rec = entry.get("recommendation")
    if rec:
        lines.append(f"建议：{rec}")
    return "\n".join(lines)
