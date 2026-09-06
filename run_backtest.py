"""
run_backtest.py — 多因子组合回测（含手续费/滑点、夏普、资金曲线）

训练/回测一律使用本地 Parquet 数据。

用法：
    python run_backtest.py --strategy-file strategies/best_ADAUSD.json --data-file D:\\K线数据\\ADAUSD_H1.parquet
    python run_backtest.py --strategy-file path\\to\\strategy.json
        # 若策略 JSON 内含 data_file 字段，可省略 --data-file
    python run_backtest.py --commission 0.02 --slippage 0.01
        # 单边手续费/滑点（单位 %），默认 0.02 / 0.01
"""

import json, sys, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager
from backtest_viz import BacktestEngine
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import FeatureEngineer
from model_core.backtest import estimate_periods_per_year
from strategy_manager.signal import compute_target_positions_stateless

_H1_PER_YEAR = 6240
DEFAULT_COMMISSION_PCT = 0.02  # 单边手续费 %
DEFAULT_SLIPPAGE_PCT = 0.01    # 单边滑点 %


def decode_formula(tokens: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in tokens)


def _factor_series(formula: list[int], feat: np.ndarray | torch.Tensor) -> np.ndarray:
    """把公式在整个特征序列上执行一次，得到逐 bar 因子值（因果，无 look-ahead）。

    离散撮合引擎需要逐 bar 的因子来决定方向/强度；实时/模拟盘每次只用最新一根
    bar 的信号，口径一致（特征与 VM 输出均只依赖 ≤t 的数据）。
    """
    vm = StackVM()
    with torch.no_grad():
        out = vm.execute([int(t) for t in formula], feat)
    if out is None or out.ndim != 2 or out.shape[1] == 0:
        return np.zeros(feat.shape[-1], dtype=np.float64)
    return out[0].cpu().numpy().astype(np.float64)


def load_strategy(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"formula": data, "vocab_version": "legacy", "symbol": None}
    return data


# ── 统计指标 ──────────────────────────────────────────────────────────────────

def calc_sharpe(pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """年化 Sharpe（无风险利率=0）。"""
    m = pnl.mean()
    s = pnl.std(ddof=0)
    if s < 1e-10:
        return 0.0
    return float(m / s * math.sqrt(periods_per_year))


def calc_sortino(pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """年化 Sortino（下行标准差）。"""
    m    = pnl.mean()
    down = pnl[pnl < 0]
    ds   = down.std(ddof=0) if len(down) > 0 else 1e-10
    ds   = max(ds, abs(m), 1e-10)
    return float(np.clip(m / ds * math.sqrt(periods_per_year), -20, 20))


def calc_max_drawdown(cum: np.ndarray) -> float:
    """从累计资金曲线算最大回撤（负数；如 -0.123 = -12.3%）。

    输入 cum 为累计收益序列（等权/离散引擎口径：equity = 1 + cum），
    输出 = min(equity/滚动峰值 - 1)，单调上涨时回撤为 0（<=0 不会为正）。
    """
    arr = np.asarray(cum, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    eq = 1.0 + arr
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min()) if np.isfinite(dd).all() else 0.0


def calc_rolling_sharpe(
    pnl: np.ndarray,
    window: int = 500,
    periods_per_year: int = _H1_PER_YEAR,
) -> np.ndarray:
    """滚动年化夏普；窗口不足处为 nan。"""
    T = len(pnl)
    out = np.full(T, np.nan, dtype=np.float64)
    if T == 0 or window <= 1:
        return out
    w = min(window, T)
    # 累积和 / 累积平方和 → O(T) 滑动窗口
    csum = np.concatenate([[0.0], np.cumsum(pnl, dtype=np.float64)])
    csq = np.concatenate([[0.0], np.cumsum(pnl.astype(np.float64) ** 2)])
    for i in range(w - 1, T):
        s = csum[i + 1] - csum[i + 1 - w]
        sq = csq[i + 1] - csq[i + 1 - w]
        mean = s / w
        var = sq / w - mean * mean
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.clip(mean / std * math.sqrt(periods_per_year), -20, 20))
    return out


def _fmt_pl_ratio(results_map: dict) -> str:
    vals = [
        d["profit_loss_ratio"]
        for d in results_map.values()
        if d.get("profit_loss_ratio") is not None
    ]
    if not vals:
        return "—"
    return f"{sum(vals) / len(vals):.3f}"


# ── 资金曲线图 ────────────────────────────────────────────────────────────────

def _setup_chinese_font() -> None:
    """让 matplotlib 能正确显示中文（Windows 优先微软雅黑）。"""
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_equity_curves(results_map: dict, output_dir: str, times_arr: np.ndarray | None = None,
                       periods_per_year: int = _H1_PER_YEAR):
    """绘制各品种 + 等权组合的资金曲线（中文标注）。

    Args:
        results_map: {symbol: {"pnl": np.array, "cum_pnl": np.array, ...}}
        output_dir:  输出目录
        times_arr:   时间戳数组（Unix秒），用于 X 轴刻度
        periods_per_year: 年化因子（每年 bar 数），由调用方按数据周期估计
    """
    _setup_chinese_font()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    syms   = list(results_map.keys())
    n_syms = len(syms)

    fig, ax_eq = plt.subplots(figsize=(18, 7), dpi=110)

    colors = ["#1565c0", "#00897b", "#e65100", "#6a1b9a", "#558b2f", "#b71c1c"]

    # 等权组合 PnL
    all_pnls = np.stack([results_map[s]["pnl"] for s in syms], axis=0)
    port_pnl = all_pnls.mean(axis=0)
    port_cum = np.cumsum(port_pnl)

    T = len(port_cum)
    x = np.arange(T)

    if n_syms == 1:
        sym = syms[0]
        cum = results_map[sym]["cum_pnl"]
        ax_eq.plot(
            x, cum, linewidth=2.0, color="#1565c0",
            label=f"{sym}（索提诺 {results_map[sym]['sortino']:+.2f}）",
        )
        ax_eq.fill_between(x, cum, 0, where=cum >= 0, alpha=0.08, color="#1565c0")
        ax_eq.fill_between(x, cum, 0, where=cum < 0,  alpha=0.08, color="#b71c1c")
        title_head = f"{sym} 资金曲线"
        show_pnl, show_cum = results_map[sym]["pnl"], cum
    else:
        for i, sym in enumerate(syms):
            cum = results_map[sym]["cum_pnl"]
            ax_eq.plot(
                x, cum, linewidth=0.8, alpha=0.65, color=colors[i % len(colors)],
                label=f"{sym}（索提诺 {results_map[sym]['sortino']:+.2f}）",
            )
        ax_eq.plot(
            x, port_cum, linewidth=2.2, color="black",
            label=f"等权组合（索提诺 {calc_sortino(port_pnl, periods_per_year):+.2f}）",
        )
        ax_eq.fill_between(x, port_cum, 0, where=port_cum >= 0, alpha=0.06, color="#1565c0")
        ax_eq.fill_between(x, port_cum, 0, where=port_cum < 0,  alpha=0.06, color="#b71c1c")
        title_head = "多因子组合资金曲线"
        show_pnl, show_cum = port_pnl, port_cum

    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("累计对数收益", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(
        f"{title_head}  |  "
        f"总收益={show_cum[-1]:+.3f}  "
        f"夏普={calc_sharpe(show_pnl, periods_per_year):+.2f}  "
        f"索提诺={calc_sortino(show_pnl, periods_per_year):+.2f}  "
        f"盈亏比={_fmt_pl_ratio(results_map)}",
        fontsize=11, pad=8,
    )

    # X 轴时间刻度
    if times_arr is not None and len(times_arr) == T:
        from datetime import datetime, timezone
        step  = max(1, T // 10)
        ticks = x[::step]
        labels = [
            datetime.fromtimestamp(int(times_arr[i]), tz=timezone.utc).strftime("%Y-%m-%d")
            for i in range(0, T, step)
        ]
        ax_eq.set_xticks(ticks)
        ax_eq.set_xticklabels(labels[:len(ticks)], fontsize=8, rotation=20)
    ax_eq.set_xlabel("日期", fontsize=9)

    path = str(Path(output_dir) / "portfolio_equity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  资金曲线图已保存 → {path}")
    return path


def _real_of_series_end(kinds: list[str], last: float) -> float:
    """把累计曲线的末点换算成真实简单收益率。

    - kind="log"  : 曲线是累计对数收益 (连续/signal 口径), 真实收益 = expm1(last)
    - kind="simple": 曲线已是简单收益 (离散撮合口径), 直接用
    组合层用品种 kind 集合判断（同一 hold_policy 下所有品种同构）。
    """
    if "log" in kinds:
        return float(np.expm1(last))
    return float(last)


def export_equity_json(
    results_map: dict,
    output_dir: str,
    times_arr: np.ndarray | None = None,
    max_points: int = 1500,
    rolling_window: int = 500,
    periods_per_year: int = _H1_PER_YEAR,
):
    """导出资金曲线原始数据为 JSON，供前端渲染交互式 HTML 图表。

    结构：
        {
          "labels": [...时间标签],
          "n_points": int, "total_bars": int,
          "rolling_window": int,
          "symbols": { sym: { equity, rolling_sharpe, sharpe, sortino,
                              total_return, profit_loss_ratio } },
          "portfolio": { ... }   # 多品种时才有
        }
    """
    syms = list(results_map.keys())
    if not syms:
        return None

    all_pnls = np.stack([results_map[s]["pnl"] for s in syms], axis=0)
    port_pnl = all_pnls.mean(axis=0)
    port_cum = np.cumsum(port_pnl)
    T = len(port_cum)

    # 均匀降采样，保证首尾点在内，避免 JSON 过大导致前端卡顿
    if T > max_points:
        idx = np.unique(np.linspace(0, T - 1, max_points).astype(int))
    else:
        idx = np.arange(T)

    def _sample(arr: np.ndarray) -> list[float | None]:
        out = []
        for i in idx:
            v = arr[i]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(None)
            else:
                out.append(round(float(v), 6))
        return out

    if times_arr is not None and len(times_arr) == T:
        from datetime import datetime, timezone

        labels = [
            datetime.fromtimestamp(int(times_arr[i]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            for i in idx
        ]
    else:
        labels = [str(int(i)) for i in idx]

    out: dict = {
        "labels": labels,
        "n_points": int(len(idx)),
        "total_bars": int(T),
        "rolling_window": int(rolling_window),
        "symbols": {},
    }
    for s in syms:
        cum = results_map[s]["cum_pnl"]
        roll = calc_rolling_sharpe(results_map[s]["pnl"], window=rolling_window,
                                   periods_per_year=periods_per_year)
        pl = results_map[s].get("profit_loss_ratio")
        bh = results_map[s].get("buy_hold")  # 买入持有基准（与策略曲线同口径，见 run 处）
        bh_kind = results_map[s].get("buy_hold_kind", "simple")
        out["symbols"][s] = {
            "equity": _sample(cum),
            "buy_hold": _sample(bh) if bh is not None else None,
            # 统计卡展示真实收益率: log 口径 (signal) 用 expm1 还原, 曲线数组仍保持原生口径
            "buy_hold_return": round(_real_of_series_end([bh_kind], float(bh[-1])), 6)
            if bh is not None and len(bh) else None,
            "rolling_sharpe": _sample(roll),
            "sharpe": round(float(results_map[s]["sharpe"]), 4),
            "sortino": round(float(results_map[s]["sortino"]), 4),
            "total_return": round(float(results_map[s]["total_return"]), 6),
            "profit_loss_ratio": round(float(pl), 4) if pl is not None else None,
        }

    if len(syms) > 1:
        pl_vals = [
            results_map[s]["profit_loss_ratio"]
            for s in syms
            if results_map[s].get("profit_loss_ratio") is not None
        ]
        port_pl = float(sum(pl_vals) / len(pl_vals)) if pl_vals else None
        bh_arrs = [results_map[s]["buy_hold"] for s in syms
                   if results_map.get(s, {}).get("buy_hold") is not None]
        port_kinds = [results_map[s].get("buy_hold_kind", "simple") for s in syms
                      if results_map.get(s, {}).get("buy_hold") is not None]
        if bh_arrs:
            port_bh = np.mean(np.stack(bh_arrs), axis=0)
        else:
            port_bh = None
        out["portfolio"] = {
            "equity": _sample(port_cum),
            "total_return": round(_real_of_series_end(port_kinds, float(port_cum[-1])), 6),
            "buy_hold": _sample(port_bh) if port_bh is not None else None,
            "buy_hold_return": round(_real_of_series_end(port_kinds, float(port_bh[-1])), 6)
            if port_bh is not None and len(port_bh) else None,
            "rolling_sharpe": _sample(calc_rolling_sharpe(port_pnl, window=rolling_window,
                                                         periods_per_year=periods_per_year)),
            "sharpe": round(float(calc_sharpe(port_pnl, periods_per_year)), 4),
            "sortino": round(float(calc_sortino(port_pnl, periods_per_year)), 4),
            "profit_loss_ratio": round(port_pl, 4) if port_pl is not None else None,
        }

    path = Path(output_dir) / "equity_curve.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  资金曲线数据已保存 → {path}")
    return str(path)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR  = "backtest_output"
    single_mode = "--single" in sys.argv
    # 回测强制离线：只用本地 Parquet
    if "--online" in sys.argv:
        print("[ERROR] 回测已禁用在线拉数。请使用本地 Parquet（--data-file 或策略内 data_file）。")
        sys.exit(1)

    strategy_file = None
    data_file_arg = None
    commission_pct = DEFAULT_COMMISSION_PCT
    slippage_pct = DEFAULT_SLIPPAGE_PCT
    hold_policy = "signal"
    window_bars = None
    max_position_pct = 100.0
    signal_threshold = None
    for i, arg in enumerate(sys.argv):
        if arg == "--strategy-file" and i + 1 < len(sys.argv):
            strategy_file = sys.argv[i + 1]
        elif arg == "--data-file" and i + 1 < len(sys.argv):
            data_file_arg = sys.argv[i + 1]
        elif arg == "--commission" and i + 1 < len(sys.argv):
            commission_pct = float(sys.argv[i + 1])
        elif arg == "--slippage" and i + 1 < len(sys.argv):
            slippage_pct = float(sys.argv[i + 1])
        elif arg == "--hold-policy" and i + 1 < len(sys.argv):
            hold_policy = sys.argv[i + 1].strip().lower()
        elif arg == "--window-bars" and i + 1 < len(sys.argv):
            window_bars = int(sys.argv[i + 1])
        elif arg == "--max-position-pct" and i + 1 < len(sys.argv):
            max_position_pct = float(sys.argv[i + 1])
        elif arg == "--threshold" and i + 1 < len(sys.argv):
            signal_threshold = float(sys.argv[i + 1])
    from web.hold_policy import HOLD_POLICIES, combo_id, combo_label, combo_parts
    raw_l = hold_policy.strip().lower()
    if raw_l != "signal" and combo_parts(raw_l) == ["signal"]:
        opts = "/".join(list(HOLD_POLICIES) + ["A+B 组合（如 dd+chandelier）"])
        print(f"[ERROR] 未知持仓管理方案: {hold_policy}（可选 {opts}）")
        sys.exit(1)
    hold_policy = combo_id(raw_l)

    if commission_pct < 0 or slippage_pct < 0:
        print("[ERROR] 手续费/滑点不能为负"); sys.exit(1)
    cost_rate_all = (commission_pct + slippage_pct) / 100.0
    print(
        f"\n交易成本（单边）: "
        f"手续费={commission_pct:g}%  滑点={slippage_pct:g}%  "
        f"→ cost_rate={cost_rate_all:.8f}"
    )
    max_position_pct = min(200.0, max(1.0, max_position_pct))
    print(f"每笔投入上限: {max_position_pct:g}%（占权益；仓位 = |tanh(因子)| × 上限%）")
    if signal_threshold is not None:
        signal_threshold = float(signal_threshold)
        print(f"无信号阈值: |tanh(因子)| < {signal_threshold:g} → 观望 FLAT")
    print(f"持仓管理: {combo_label(hold_policy)}（{hold_policy}）")
    if hold_policy != "signal":
        print("  非 signal 方案走离散撮合引擎（与模拟实盘同口径：收盘信号翻转 + 止损/止盈/移动止损）")
    print("数据模式: 本地 Parquet（强制离线）")

    # ── 2. 加载策略 ─────────────────────────────────────────────────
    strategy_data_file = None
    print(f"\n{'='*62}")
    if strategy_file:
        data = load_strategy(Path(strategy_file))
        if data is None:
            print(f"[ERROR] 找不到: {strategy_file}"); sys.exit(1)
        strategy_data_file = data.get("data_file")
        sym = data.get("symbol")
        if not sym:
            stem = Path(strategy_file).stem
            if stem.startswith("best_"):
                sym = stem.replace("best_", "", 1)
            elif stem.startswith("strategy_"):
                # strategy_ADAUSD_step0084_score2.4021 / strategy_ADAUSD (1)
                rest = stem.replace("strategy_", "", 1)
                sym = rest.split("_step")[0].split(" ")[0]
        if not sym:
            print("[ERROR] 策略文件未包含 symbol，且无法从文件名识别"); sys.exit(1)
        symbol_formulas = {sym: data["formula"]}
        sc = data.get("best_score", "N/A")
        score_txt = f"{sc:.3f}" if isinstance(sc, (int, float)) else str(sc)
        print(f"  模式: 单策略文件 ({Path(strategy_file).name})")
        print(f"  {sym}: score={score_txt}  {decode_formula(data['formula'])}")
        if strategy_data_file:
            print(f"  策略记录数据: {strategy_data_file}")
    elif single_mode:
        data = load_strategy(Path(Config.STRATEGY_FILE))
        if data is None:
            print(f"[ERROR] 找不到: {Config.STRATEGY_FILE}"); sys.exit(1)
        strategy_data_file = data.get("data_file")
        symbol_formulas = {sym: data["formula"] for sym in Config.SYMBOLS}
        print("  模式: 单公式（所有品种共用）")
    else:
        symbol_formulas = {}
        for sym in Config.SYMBOLS:
            path = Path("strategies") / f"best_{sym}.json"
            data = load_strategy(path)
            if data is None:
                print(f"  [缺失] {sym}")
                continue
            ver = data.get("vocab_version", "unknown")
            if ver != VOCAB_VERSION:
                print(f"  [跳过] {sym}: vocab_version 不符 ({ver} vs {VOCAB_VERSION})")
                continue
            symbol_formulas[sym] = data["formula"]
            if not strategy_data_file and data.get("data_file"):
                strategy_data_file = data.get("data_file")
            sc = data.get("best_score", "N/A")
            print(f"  {sym}: score={sc:.3f}  {decode_formula(data['formula'])}")

    if not symbol_formulas:
        print("[ERROR] 没有有效策略，请先运行训练"); sys.exit(1)

    cost_rates = {sym: cost_rate_all for sym in symbol_formulas}
    print(f"{'='*62}\n")

    # ── 3. 加载数据（仅本地 Parquet）────────────────────────────────
    if not data_file_arg and strategy_data_file:
        data_file_arg = str(strategy_data_file).strip() or None

    if not data_file_arg:
        print(
            "[ERROR] 未指定本地 Parquet。\n"
            "请传入 --data-file PATH\\TO\\SYMBOL_TF.parquet，\n"
            "或使用包含 data_file 字段的策略 JSON（本软件训练生成）。\n"
            "回测使用本地 Parquet，不使用在线行情。"
        )
        sys.exit(1)

    parquet_path = Path(data_file_arg)
    if not parquet_path.exists():
        print(f"[ERROR] Parquet 不存在: {parquet_path}")
        sys.exit(1)

    print(f"正在加载数据（离线 Parquet: {parquet_path}）...")
    pm = ParquetDataManager(str(parquet_path))
    pm.load()
    raw_dict = pm.raw_dict
    syms = pm.symbols
    # 策略品种名与 Parquet 品种不一致时（单策略 + 单品种文件），映射公式到数据品种
    if strategy_file and len(symbol_formulas) == 1 and len(syms) == 1:
        strat_sym = next(iter(symbol_formulas))
        data_sym = syms[0]
        if strat_sym != data_sym:
            formula = symbol_formulas[strat_sym]
            print(f"  [映射] 策略品种 {strat_sym} → 数据品种 {data_sym}")
            symbol_formulas = {data_sym: formula}
            cost_rates = {data_sym: cost_rate_all}

    T = raw_dict["open"].shape[1]
    times_all = raw_dict.get("time", None)
    # 数据驱动年化因子：按实际时间戳估计每年 bar 数，替代写死的 H1=6240。
    # A 股日线(~244)、A 股 15min(~3904)、外汇 H1(6240)、加密日线(365) 都会被正确年化。
    ppy = estimate_periods_per_year(times_all) if times_all is not None else _H1_PER_YEAR
    # 样本外窗口：只回测尾部最后 N 根（如训练从未见过的 holdout 之后数据）
    window_start = 0
    if window_bars:
        if window_bars < 800:
            print(f"[ERROR] 窗口过小（{window_bars} 根）：特征 warm-up 需 ≥800 根，无法稳定计算")
            sys.exit(1)
        if window_bars >= T:
            window_bars = None
        else:
            window_start = T - window_bars
            print(f"  样本外窗口: 只回测最后 {window_bars} 根（bar {window_start}..{T-1}，占总数据 {window_bars/T:.1%}）")

    if window_start > 0:
        raw_dict = {k: v[:, window_start:] for k, v in raw_dict.items()}
        times_all = raw_dict.get("time")
        T = raw_dict["open"].shape[1]
        ppy = estimate_periods_per_year(times_all) if times_all is not None else _H1_PER_YEAR
        print(f"  窗口切分后: T={T} bars  年化因子={ppy} bar/年\n")
    else:
        print(f"  品种: {syms}  T={T} bars  年化因子={ppy} bar/年\n")

    # ── 4. 为每品种计算因子 + 回测 ───────────────────────────────
    vm   = StackVM()
    # 因果特征化：_robust_norm 现为滚动窗口实现，传入全量序列是安全的
    # 每个时间步 t 的归一化参数只依赖 [t-w+1..t]，无 look-ahead
    feat = FeatureEngineer.compute_features(raw_dict)  # [N, F, T]，因果安全

    results_map = {}
    backtest_results = []

    for i, sym in enumerate(syms):
        if sym not in symbol_formulas:
            print(f"  [跳过] {sym}（无策略）")
            continue

        formula   = symbol_formulas[sym]
        cost_rate = cost_rates.get(sym, cost_rate_all)
        feat_i    = feat[i:i+1]
        raw_i     = {k: v[i:i+1] for k, v in raw_dict.items()}

        if hold_policy != "signal":
            # 离散撮合引擎：与模拟实盘同口径（含止损/止盈/移动止损）
            from web.paper_replay import run_replay

            factor = _factor_series(formula, feat_i)
            T = factor.shape[0]
            rep = run_replay(
                factor=factor,
                open_p=raw_i["open"][0].numpy(),
                high_p=raw_i["high"][0].numpy(),
                low_p=raw_i["low"][0].numpy(),
                close_p=raw_i["close"][0].numpy(),
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
                policy_id=hold_policy,
                notional=1.0,
                max_position_pct=max_position_pct,
                threshold=signal_threshold,
                periods_per_year=ppy,
            )
            st = rep["stats"]
            pnl_arr = rep["pnl"]
            cum_arr = rep["equity"] - 1.0
            # 买入持有基准（离散口径：简单收益，与 equity-1.0 同单位）
            _closes = raw_i["close"][0].numpy()
            buy_hold = _closes / _closes[0] - 1.0
            results_map[sym] = {
                "pnl":          pnl_arr,
                "cum_pnl":      cum_arr,
                "buy_hold":     buy_hold,
                "buy_hold_kind": "simple",
                "total_return": st["total_return"],
                "sharpe":       st["sharpe"],
                "sortino":      st["sortino"],
                "max_drawdown": st["max_drawdown"],
                "n_trades":     st["n_trades"],
                "win_rate":     st["win_rate"],
                "avg_hold":     st["avg_hold_bars"],
                "profit_loss_ratio": st["profit_loss_ratio"],
                "cost_rate":    cost_rate,
                "hold_policy":  hold_policy,
            }
            print(f"  [离散] {sym}: 总收益={st['total_return']:+.3f} "
                  f"Sharpe={st['sharpe']:+.2f} 交易={st['n_trades']} "
                  f"胜率={st['win_rate']:.1%} 出场费合计={st['fees_total']:.4f}")
        else:
            engine    = BacktestEngine(formula=formula, cost_rate=cost_rate,
                                       periods_per_year=ppy,
                                       max_position_pct=max_position_pct,
                                       min_abs=signal_threshold)
            sym_res   = engine.run(raw_i, feat_i, [sym])
            backtest_results.extend(sym_res)

            r = sym_res[0]
            pnl_arr = r.pnl
            cum_arr = r.cum_pnl
            sharpe  = calc_sharpe(pnl_arr, ppy)
            sortino = calc_sortino(pnl_arr, ppy)
            pl_ratio = r.profit_loss_ratio
            # 买入持有基准（连续口径：累计对数收益，与 cum_pnl 同单位）
            _closes = raw_i["close"][0].numpy()
            buy_hold = np.log(_closes / _closes[0])

            results_map[sym] = {
                "pnl":          pnl_arr,
                "cum_pnl":      cum_arr,
                "buy_hold":     buy_hold,
                "buy_hold_kind": "log",
                # 引擎 total_return 是累计对数收益末点; 对外统一成真实简单收益率 (expm1),
                # 与离散撮合口径 / acceptance 报告的 % 一致。曲线数组仍保持 log 口径。
                "total_return": float(np.expm1(float(r.total_return))),
                "sharpe":       sharpe,
                "sortino":      sortino,
                "max_drawdown": calc_max_drawdown(cum_arr),
                "n_trades":     r.n_trades,
                "win_rate":     r.win_rate,
                "avg_hold":     r.avg_hold_bars,
                "profit_loss_ratio": pl_ratio,
                "cost_rate":    cost_rate,
            }

    # ── 5. 打印各品种统计 ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  多因子回测报告")
    print(f"{'='*62}")
    header = f"{'品种':12s} {'PnL':>8} {'Sharpe':>8} {'Sortino':>8} {'盈亏比':>8} {'Trades':>7} {'WinRate':>8} {'AvgH':>6}"
    print(f"  {header}")
    print(f"  {'─'*72}")
    for sym, d in results_map.items():
        pl = d["profit_loss_ratio"]
        pl_s = f"{pl:8.3f}" if pl is not None else f"{'—':>8}"
        print(f"  {sym:12s} "
              f"{d['total_return']:+8.3f} "
              f"{d['sharpe']:+8.3f} "
              f"{d['sortino']:+8.3f} "
              f"{pl_s} "
              f"{d['n_trades']:7d} "
              f"{d['win_rate']:8.1%} "
              f"{d['avg_hold']:6.1f}h")

    # 等权组合
    p_pl_ratio = None
    if results_map:
        all_pnls = np.stack([d["pnl"] for d in results_map.values()], axis=0)
        port_pnl = all_pnls.mean(axis=0)
        port_cum = np.cumsum(port_pnl)
        p_sharpe  = calc_sharpe(port_pnl, ppy)
        p_sortino = calc_sortino(port_pnl, ppy)
        pl_vals = [d["profit_loss_ratio"] for d in results_map.values()
                   if d["profit_loss_ratio"] is not None]
        p_pl_ratio = float(sum(pl_vals) / len(pl_vals)) if pl_vals else None
        pl_s = f"{p_pl_ratio:8.3f}" if p_pl_ratio is not None else f"{'—':>8}"
        port_kinds = [d.get("buy_hold_kind", "simple") for d in results_map.values()]
        port_total = _real_of_series_end(port_kinds, float(port_cum[-1]))
        print(f"  {'─'*72}")
        print(f"  {'Portfolio':12s} "
              f"{port_total:+8.3f} "
              f"{p_sharpe:+8.3f} "
              f"{p_sortino:+8.3f} "
              f"{pl_s}")
        print(f"\n  正收益品种: {sum(1 for d in results_map.values() if d['total_return']>0)}/{len(results_map)}")
        print(f"  Sharpe>1 品种: {sum(1 for d in results_map.values() if d['sharpe']>1)}/{len(results_map)}")
    print(f"{'='*62}\n")

    # ── 6. 资金曲线图 ─────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if results_map:
        times_np = times_all[0].numpy() if times_all is not None else None
        plot_equity_curves(results_map, OUTPUT_DIR, times_np, periods_per_year=ppy)
        export_equity_json(results_map, OUTPUT_DIR, times_np, periods_per_year=ppy)

    # ── 7. 资金曲线图已在步骤 6 生成；跳过 K 线/逐笔交易图以加快回测 ─────

    # ── 8. 保存 JSON 报告 ─────────────────────────────────────────────
    report = {
        "mode": "single" if single_mode else "multi_factor",
        "cost_rates": cost_rates,
        "hold_policy": hold_policy,
        "window_bars": window_bars,
        "window_start": window_start,
        "symbols": {},
        "portfolio": {},
    }
    for sym, d in results_map.items():
        formula = symbol_formulas.get(sym, [])
        pl = d["profit_loss_ratio"]
        report["symbols"][sym] = {
            "formula":      formula,
            "readable":     decode_formula(formula),
            "cost_rate":    d["cost_rate"],
            "total_return": round(d["total_return"], 6),
            "sharpe":       round(d["sharpe"], 4),
            "sortino":      round(d["sortino"], 4),
            "max_drawdown": round(d["max_drawdown"], 4),
            "n_trades":     d["n_trades"],
            "win_rate":     round(d["win_rate"], 4),
            "avg_hold_bars":round(d["avg_hold"], 2),
            "profit_loss_ratio": round(pl, 4) if pl is not None else None,
        }
    if results_map:
        report["portfolio"] = {
            "total_return": round(float(port_cum[-1]), 6),
            "sharpe":       round(p_sharpe, 4),
            "sortino":      round(p_sortino, 4),
            "max_drawdown": round(calc_max_drawdown(port_cum), 4),
            "profit_loss_ratio": round(p_pl_ratio, 4) if p_pl_ratio is not None else None,
        }
    rp = f"{OUTPUT_DIR}/multi_factor_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 报告已保存 → {rp}")
    print("完成。\n")


if __name__ == "__main__":
    main()
