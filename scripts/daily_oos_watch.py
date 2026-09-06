"""每日巡检：增量下载 M5 新 bar 累积「训练截止后」数据，攒够 N 根自动重跑真 OOS 报告。

设计（与 results/true_oos_REPORT.md 的结论衔接）：
- 训练截止（cutoff）= 冠军策略 data_source.end / 其训练切片的最后一根 ts；
- 「训练截止后」数据累积在 data/oos_daily/BTCUSDT_M5.parquet（只存 ts > cutoff 的 bar），
  每次运行先种子化（从 data/training 全历史里已有的 post-cutoff bar 导入，避免重复下载），
  再增量拉 Binance 尾部窗口合并去重；
- 距上次报告之后新增 ≥ min-new 根 → 重建 [训练切片 + 累积新 bar] 的 OOS 文件，在
  新段（start_idx = 切片长度）上重跑 5 方案 × 阈值 真 OOS 回放，区域统计只统计新段，
  产出 results/true_oos_daily_<时间戳>.json/.md 并飞书推送摘要（未配 webhook 时降级
  macOS 本地通知 + logs/daily_oos_watch.log，链路不静默失效）；
- 幂等：重复运行不重复报告（reported_upto_ts 推进），--force 强制重跑。

用法：
  # 每天一次（launchd / crontab）：
  .venv/bin/python scripts/daily_oos_watch.py
  # 只打印状态不拉网不动盘：
  .venv/bin/python scripts/daily_oos_watch.py --dry-run --offline
  # 攒够了立即出一份：
  .venv/bin/python scripts/daily_oos_watch.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYMBOL = "BTCUSDT"
TIMEFRAME = "M5"
STATE_DIR = ROOT / "data" / "oos_daily"
ACCUM_FILE = STATE_DIR / "BTCUSDT_M5.parquet"
RUN_FILE = STATE_DIR / "run" / "BTCUSDT_M5.parquet"  # 文件名须 {sym}_{tf}.parquet
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = ROOT / "logs" / "daily_oos_watch.log"
POLICIES = ["signal", "risk", "dd", "chandelier", "dd+chandelier"]
THRESHOLDS = [0.05, 0.8]
COMMISSION, SLIPPAGE = 0.02, 0.01


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")


def _ts_col(df) -> str:
    return "timestamp" if "timestamp" in df.columns else df.columns[0]


def _load_accum() -> tuple | None:
    """读累积文件（新 bar 只存 OHLCV，time 秒）。返回 df 或 None。"""
    import pandas as pd

    if not ACCUM_FILE.exists():
        return None
    df = pd.read_parquet(ACCUM_FILE)
    t = _ts_col(df)
    if df[t].dtype != "int64":
        df[t] = df[t].astype("int64")
    return df.drop_duplicates(t).sort_values(t).reset_index(drop=True)


def _resolve_cutoff(strategy_file: Path, slice_file: Path) -> tuple[float, str]:
    """返回 (cutoff 秒, 来源说明)。优先 data_source.end ISO；否则取训练切片最后一根 ts。"""
    import pandas as pd

    try:
        strat = json.loads(strategy_file.read_text(encoding="utf-8"))
        end = ((strat.get("data_source") or {}).get("end")
               or (strat.get("train_range") or {}).get("end"))
        if end:
            # 支持 '2026-04-17T12:45:00+00:00' 与 '2026-09-03'
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(end, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.timestamp(), f"data_source.end({end})"
                except ValueError:
                    continue
    except Exception:  # noqa: BLE001
        pass
    df = pd.read_parquet(slice_file)
    ts = df[_ts_col(df)].to_numpy(dtype=np.float64)
    return float(ts[-1]), f"训练切片末根 ts（{slice_file.name}）"


def _seed_from_training_archive(cutoff: float, state: dict) -> dict:
    """把 data/training 全历史里 ts > cutoff 的 bar 一次性导入累积文件。"""
    import pandas as pd

    import pandas as pd

    arc = ROOT / "data" / "training" / f"{SYMBOL}_{TIMEFRAME}.parquet"
    if not arc.exists():
        return state
    full = pd.read_parquet(arc)
    tcol = _ts_col(full)
    if full[tcol].dtype != "int64":
        full[tcol] = full[tcol].astype("int64")
    new_mask = full[tcol].to_numpy(dtype=np.float64) > cutoff
    if int(new_mask.sum()) == 0:
        return state
    new = full.loc[new_mask].copy()
    accum = _load_accum()
    merged = new if accum is None else pd.concat([accum, new], ignore_index=True)
    merged = (merged.drop_duplicates(tcol).sort_values(tcol).reset_index(drop=True)
              .astype({tcol: "int64", "volume": "int64"}))
    _write_accum(merged)
    state["seeded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state["seed_bars"] = int(len(new))
    _log(f"[种子] 从全历史导入 ts>cutoff 的 {len(new)} 根到累积文件")
    return state


def _write_accum(df) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ACCUM_FILE, index=False)


def _fetch_and_merge(fetch_bars: int) -> int:
    """拉 Binance 尾部窗口并入累积文件；返回新增根数。"""
    from web.data_download import _fetch_source_bars

    df = _fetch_source_bars("binance", SYMBOL, TIMEFRAME, int(fetch_bars))
    tcol = _ts_col(df)
    if df[tcol].dtype != "int64":
        df[tcol] = df[tcol].astype("int64")
    accum = _load_accum()
    merged = df if accum is None else pd.concat([accum, df], ignore_index=True)
    before = 0 if accum is None else int(len(accum))
    merged = (merged.drop_duplicates(tcol).sort_values(tcol).reset_index(drop=True)
              .astype({tcol: "int64", "volume": "int64"}))
    added = int(len(merged)) - before
    if added > 0:
        _write_accum(merged)
    _log(f"[拉取] Binance 尾窗 {len(df)} 根 → 新增 {added} 根（累积 {len(merged)}）")
    return added


def _new_since(accum_df, ts0: float) -> int:
    tcol = _ts_col(accum_df)
    return int((accum_df[tcol].to_numpy(dtype=np.float64) > ts0).sum())


def _build_run_file(slice_file: Path, cutoff: float) -> int:
    """重建 OOS 回放文件 = 训练切片 + 累积新 bar；返回新段起始 bar。"""
    import pandas as pd

    slice_df = pd.read_parquet(slice_file)
    accum = _load_accum()
    tcol = _ts_col(slice_df)
    if slice_df[tcol].dtype != "int64":
        slice_df[tcol] = slice_df[tcol].astype("int64")
    if accum is None:
        raise RuntimeError("累积文件为空，无法重建 OOS 文件")
    at = _ts_col(accum)
    if accum[at].dtype != "int64":
        accum[at] = accum[at].astype("int64")
    out = pd.concat([slice_df, accum], ignore_index=True)
    out = (out.drop_duplicates(tcol).sort_values(tcol).reset_index(drop=True)
           .astype({tcol: "int64", "volume": "int64"}))
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RUN_FILE, index=False)
    ts = out[tcol].to_numpy(dtype=np.float64)
    start = int(np.argmax(ts > cutoff))
    _log(f"[OOS 文件] {RUN_FILE.name}: {len(out)} 根（切片 {len(slice_df)} + 新 {len(out) - len(slice_df)}），新段从 bar {start} 起")
    return start


def _region_stats(rep: dict, s0: int) -> dict:
    eq = np.asarray(rep["equity"], dtype=float)
    act = eq[s0:]
    peak = np.maximum.accumulate(act)
    mdd = float((act / np.where(peak > 0, peak, 1.0) - 1.0).min()) if act.size else 0.0
    eq_r = eq[max(0, s0 - 1):]
    ret = float(eq[-1] / eq_r[0] - 1.0) if eq_r[0] > 0 else 0.0
    trs = [t for t in rep["trades"] if int(t.get("bar") or 0) >= s0 - 1]
    wins = [t for t in trs if t["pnl"] > 0]
    losses = [t for t in trs if t["pnl"] < 0]
    pl = None
    if wins and losses:
        pl = float(np.mean([t["pnl"] for t in wins]) / abs(np.mean([t["pnl"] for t in losses])))
    return {
        "region_return": round(ret, 6),
        "region_mdd": round(mdd, 6),
        "n_trades": len(trs),
        "win_rate": round(len(wins) / len(trs), 4) if trs else None,
        "profit_loss_ratio": round(pl, 4) if pl is not None else None,
        "sharpe": rep["stats"]["sharpe"],
        "sortino": rep["stats"]["sortino"],
    }


def _run_report(strategy_file: Path, s0: int, cutoff: float, n_new: int) -> dict:
    from data_pipeline.parquet_manager import ParquetDataManager  # noqa: E402
    from model_core.backtest import estimate_periods_per_year  # noqa: E402
    from model_core.features import FeatureEngineer  # noqa: E402
    from model_core.vm import StackVM  # noqa: E402
    from web.paper_replay import run_replay  # noqa: E402

    strat = json.loads(strategy_file.read_text(encoding="utf-8"))
    formula = [int(t) for t in strat["formula"]]
    pm = ParquetDataManager(str(RUN_FILE))
    pm.load()
    raw_d = pm.raw_dict
    feats = FeatureEngineer.compute_features(raw_d)
    vm = StackVM()
    import torch  # noqa: E402

    with torch.no_grad():
        factor = vm.execute(formula, feats)[0].cpu().numpy().astype(float)
    o = raw_d["open"][0].numpy().astype(float)
    h = raw_d["high"][0].numpy().astype(float)
    l = raw_d["low"][0].numpy().astype(float)
    c = raw_d["close"][0].numpy().astype(float)
    times = raw_d.get("time")
    times = times[0].numpy().astype(float) if times is not None else None
    ppy = estimate_periods_per_year(times) if times is not None else 105195.0

    rows = []
    for pol in POLICIES:
        for thr in THRESHOLDS:
            rep = run_replay(factor=factor, open_p=o, high_p=h, low_p=l, close_p=c,
                             commission_pct=COMMISSION, slippage_pct=SLIPPAGE,
                             policy_id=pol, max_position_pct=100.0, start_idx=s0,
                             threshold=thr, periods_per_year=ppy)
            st = _region_stats(rep, s0)
            rows.append({"policy": pol, "threshold": thr, **st})
            wr = "—" if st["win_rate"] is None else f"{st['win_rate'] * 100:.0f}%"
            print(f"  {pol:14s} t={thr}: 收益 {st['region_return'] * 100:+.2f}% "
                  f"夏普 {st['sharpe']:+.2f} 回撤 {st['region_mdd'] * 100:+.2f}% "
                  f"交易 {st['n_trades']:>3d}", flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "strategy": str(strategy_file), "run_file": str(RUN_FILE),
           "cutoff": cutoff, "cutoff_iso": datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(),
           "n_new_total": int(n_new), "start_idx": int(s0), "rows": rows,
           "policies": POLICIES, "thresholds": THRESHOLDS,
           "commission": COMMISSION, "slippage": SLIPPAGE}
    (ROOT / "results" / f"true_oos_daily_{stamp}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: (r.get("sharpe") or -999))
    md_lines = [f"# 真·样本外自动巡检（{stamp}）",
                "",
                f"- 冠军 `{Path(strategy_file).name}` · 训练截止 "
                f"{datetime.fromtimestamp(cutoff, tz=timezone.utc)} UTC",
                f"- 训练后新 bar 共 {n_new} 根（>cutoff 累积），本次统计段 bar {s0}..（仅新段计收益/回撤）",
                f"- 成本 佣金 {COMMISSION}% / 滑点 {SLIPPAGE}%；t 为无信号阈值 |tanh(因子)|",
                "",
                "| 方案 | 阈值 | OOS 段收益 | 夏普 | 最大回撤 | 交易 | 胜率 | 盈亏比 |",
                "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: -(x.get("sharpe") or -999)):
        wr = "—" if r["win_rate"] is None else f"{r['win_rate'] * 100:.0f}%"
        plr = "—" if r["profit_loss_ratio"] is None else f"{r['profit_loss_ratio']:.2f}"
        md_lines.append(f"| {r['policy']} | {r['threshold']} | {r['region_return'] * 100:+.2f}% | "
                        f"{r['sharpe']:+.2f} | {r['region_mdd'] * 100:+.2f}% | {r['n_trades']} | "
                        f"{wr} | {plr} |")
    md_lines += ["",
                 "## 解读",
                 f"- 样本量 {n_new} 根 M5 ≈ {n_new / 288:.1f} 天——仍属小样本，结论是方向性 sanity check。",
                 f"- 本段最佳方案：`{best['policy']}`（t={best['threshold']}，Sharpe {best['sharpe']:+.2f}）。",
                 "- 与旧结论对照见 results/true_oos_REPORT.md；趋势由多次巡检累积判断，勿用单次报告下结论。"]
    (ROOT / "results" / f"true_oos_daily_{stamp}.md").write_text("\n".join(md_lines) + "\n",
                                                                 encoding="utf-8")
    return {"file_md": f"true_oos_daily_{stamp}.md", "file_json": f"true_oos_daily_{stamp}.json",
            "rows": rows, "best": best, "stamp": stamp}


def _notify(text: str) -> tuple[bool, str]:
    ok, fb = False, ""
    try:
        from web.feishu_notify import send_text

        ok, fb = send_text(text)
    except Exception as exc:  # noqa: BLE001
        fb = str(exc)
    if ok:
        return True, "飞书已推送"
    _log(f"[通知] 飞书不可用（{fb}），降级 macOS 本地通知")
    try:
        import subprocess

        safe = text.replace('"', "'")[:900]
        r = subprocess.run(
            ["osascript", "-e", f'display notification "{safe}" with title "AlphaMaster 真OOS巡检"'],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:120])
    except Exception as exc:  # noqa: BLE001
        _log(f"[通知] macOS 通知也失败: {exc}")
    return False, f"飞书未推送（{fb}），已写日志/本地通知"


def main() -> int:
    ap = argparse.ArgumentParser(description="每日增量下载 + 真 OOS 自动巡检")
    ap.add_argument("--strategy-file", default="strategies/best_BTCUSDT.json")
    ap.add_argument("--slice-file", default="data/slices/BTCUSDT_M5.parquet")
    ap.add_argument("--min-new", type=int, default=5000, help="距上次报告新增多少根才重跑")
    ap.add_argument("--fetch-bars", type=int, default=3000, help="每次拉 Binance 尾窗根数")
    ap.add_argument("--force", action="store_true", help="不管够不够先跑一份报告")
    ap.add_argument("--offline", action="store_true", help="跳过 Binance 拉取（只合并已有数据）")
    ap.add_argument("--dry-run", action="store_true", help="只打印状态，不动盘不推送")
    args = ap.parse_args()

    strategy_file = (ROOT / args.strategy_file) if not Path(args.strategy_file).is_absolute() else Path(args.strategy_file)
    slice_file = (ROOT / args.slice_file) if not Path(args.slice_file).is_absolute() else Path(args.slice_file)
    if not strategy_file.exists():
        print(f"错误: 找不到策略 {strategy_file}")
        return 1
    if not slice_file.exists():
        print(f"错误: 找不到训练切片 {slice_file}")
        return 1

    cutoff, cut_src = _resolve_cutoff(strategy_file, slice_file)
    state: dict = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    reported_upto = float(state.get("reported_upto_ts") or cutoff)

    if args.dry_run:
        accum = _load_accum()
        n_have = 0 if accum is None else _new_since(accum, cutoff)
        pending = 0 if accum is None else _new_since(accum, reported_upto)
        print(f"[状态] 训练截止 {datetime.fromtimestamp(cutoff, tz=timezone.utc)}（{cut_src}）")
        print(f"[状态] 累积 post-cutoff bar: {n_have}  距上次报告新增: {pending} / 阈值 {args.min_new}")
        if accum is not None:
            tcol = _ts_col(accum)
            print(f"[状态] 累积区间 {accum[tcol].min()} .. {accum[tcol].max()}（{len(accum)} 根）")
        print(f"[dry-run] 触发条件满足? {'是' if pending >= args.min_new or args.force else '否'}")
        print(f"[dry-run] 重跑命令: .venv/bin/python scripts/daily_oos_watch.py "
              f"{'--force' if args.force else ''}".strip())
        return 0

    # 1) 种子化（仅一次）
    if not state.get("seeded_at"):
        state = _seed_from_training_archive(cutoff, state)

    # 2) 增量拉取
    if not args.offline:
        try:
            _fetch_and_merge(args.fetch_bars)
        except Exception as exc:  # noqa: BLE001 拉取失败不阻断（离线时也能跑报告）
            _log(f"[拉取失败] {exc}")

    accum = _load_accum()
    if accum is None or len(accum) == 0:
        _log("[跳过] 仍无任何 post-cutoff 数据")
        return 0
    pending = _new_since(accum, reported_upto)
    n_new_total = _new_since(accum, cutoff)
    _log(f"[状态] post-cutoff 共 {n_new_total} 根 · 距上次报告 {pending} / {args.min_new}")

    if not args.force and pending < args.min_new:
        _log("[跳过] 未到触发阈值（cron 每天跑，攒数据中）")
        return 0

    # 3) 重跑真 OOS 报告
    s0 = _build_run_file(slice_file, cutoff)
    rep = _run_report(strategy_file, s0, cutoff, n_new_total)
    state["reported_upto_ts"] = float(accum[_ts_col(accum)].max())
    state["last_report"] = rep["stamp"]
    state["last_report_pending"] = pending
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[完成] 报告 results/{rep['file_md']}（最佳 {rep['best']['policy']} "
         f"Sharpe {rep['best']['sharpe']:+.2f}）")

    # 4) 推送
    b = rep["best"]
    lines = [
        f"【AlphaMaster 真OOS 自动巡检】{SYMBOL} 新数据 {n_new_total} 根（M5）",
        f"训练截止 {datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime('%m-%d %H:%M')} UTC",
        f"本段最佳：{b['policy']}（t={b['threshold']}）收益 {b['region_return'] * 100:+.2f}% · "
        f"Sharpe {b['sharpe']:+.2f} · 交易 {b['n_trades']}",
        f"报告：results/{rep['file_md']}",
    ]
    ok, fb = _notify("\n".join(lines))
    _log(f"[推送] {fb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
