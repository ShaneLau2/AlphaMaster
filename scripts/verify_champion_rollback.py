"""冠军回滚校验：验证「记录在案的版本」是否可复现。

背景（为什么需要它）：
- 训练/对比实验会在磁盘上留下溯源记录：data/slices/*/train_range.json
  （子集 sidecar）、results/compare_*.json（对比实验每程的 best 运行）、
  strategies/best_*.json 与 training_history_*.json（引擎写入 data_source +
  train_range + holdout）。
- 日后想回滚/复核某个冠军版本时，必须先确认两件事可复现：
  A. 抽样确定性 —— 用记录里的 模式/参数/源文件 重新生成子集，应得到与
     当时训练完全一致的文件（行数 + OHLC 数值一致；采样器无随机种子，
     同参数 → 同字节）。
  B. 存档公式的样本外分数 —— 用记录的公式 + best_score 在同一子集的
     同一条尾部 holdout 窗口上重跑生产口径验证（engine end_step=0 的
     _verify_holdout 路径），存档的 val_score/Sharpe/闸门应原样复现。

脚本对每条记录输出三档检查：
  det     确定性（子集文件在盘 → 与重生成逐行对比；文件已删 → 行数 parity）
  rescore 存档公式 holdout 复验（记录带公式 → 引擎重算 holdout）
  retrain 短程重训（--retrain-steps>0 才跑；与记录收敛幅度对照，仅供参考）

用法:
  python scripts/verify_champion_rollback.py                 # 自动发现全部记录
  python scripts/verify_champion_rollback.py --records <json> # 只查指定记录
  python scripts/verify_champion_rollback.py --retrain-steps 10 --seed 42
  python scripts/verify_champion_rollback.py --no-restore      # 缺失子集只进 tmp，不写回 data/slices

输出: results/verify_champion_rollback_<时间戳>.json / .md；任何 det/rescore FAIL
时进程以退出码 1 结束。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"


# ── 记录发现 ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def record_from_sidecar(sidecar: Path) -> dict | None:
    """data/slices/<xxx>/train_range.json → subset 记录。"""
    meta = _load_json(sidecar)
    if not meta.get("subset"):
        return None
    parquet = sidecar.parent / (meta.get("data_file") and Path(meta["data_file"]).name or "")
    rec = {
        "kind": "subset",
        "symbol": meta.get("symbol") or "?",
        "label": f"subset:{meta.get('mode')}",
        "sidecar": str(sidecar),
        "source_file": meta.get("source_file"),
        "mode": meta.get("mode"),
        "n_bars": meta.get("n_bars_requested"),
        "n_chunks": meta.get("n_chunks"),
        "regime": meta.get("regime") or "vol",
        "recorded_file": str(parquet) if parquet.exists() else None,
        "recorded_subset_bars": meta.get("subset_bars"),
        "archived": {
            "formula": None, "best": None, "holdout": None,
            "seed": None, "steps": None,
        },
    }
    return rec


def _pick_run(variant: dict) -> dict | None:
    """variant = {best, holdout_passed, runs:[...]}（对比实验格式）。取 in-sample best 那程。"""
    if isinstance(variant, dict):
        best = variant.get("best")
        if isinstance(best, dict):
            return best
        runs = variant.get("runs") or []
        if isinstance(runs, list) and runs:
            return runs[0]
    if isinstance(variant, list) and variant:
        return variant[0]
    return None


def record_from_compare(path: Path) -> list[dict]:
    """results/compare_*.json → 每 variant 的 best 运行 = 一条 champion 记录。"""
    d = _load_json(path)
    variants = d.get("variants") or {}
    out: list[dict] = []
    params = d.get("params") or {}
    for vkey, v in variants.items():
        run = _pick_run(v) if isinstance(v, (dict, list)) else None
        if not isinstance(run, dict):
            continue
        formula = run.get("best_formula") or run.get("formula")
        if not formula:
            continue
        ho = run.get("holdout")
        rec = {
            "kind": "champion",
            "symbol": run.get("symbol") or "?",
            "label": f"compare:{vkey}",
            "origin": str(path),
            "source_file": d.get("source_file"),
            "mode": "tail" if str(vkey).startswith("tail") else
                    ("spread" if str(vkey).startswith("spread") else
                     (str(run.get("variant") or "").split("_")[0] or vkey)),
            "n_bars": params.get("n_bars"),
            "n_chunks": params.get("chunks"),
            "regime": params.get("regime") or "vol",
            "recorded_file": None,          # 对比实验子集通常已清理
            "recorded_subset_bars": run.get("source_bars"),
            "archived": {
                "formula": [int(t) for t in formula],
                "best": run.get("best_score"),
                "holdout": ho if isinstance(ho, dict) else None,
                "seed": run.get("seed"),
                "steps": run.get("steps"),
            },
        }
        out.append(rec)
    return out


def record_from_strategy(path: Path) -> dict | None:
    """strategies/best_*.json / training_history_*.json（引擎写入，带公式）→ champion 记录。"""
    try:
        d = _load_json(path)
    except Exception:  # noqa: BLE001
        return None
    formula = d.get("formula")
    is_history = "step" in d and isinstance(d.get("step"), list)
    if not formula and not is_history:
        return None
    if is_history and not formula:
        return None
    tr = d.get("train_range")
    data_file = (tr or {}).get("data_file") or d.get("data_file") or \
        ((d.get("data_source") or {}).get("data_file"))
    if not data_file:
        return None
    pf = Path(data_file)
    rec = {
        "kind": "champion",
        "symbol": d.get("symbol") or pf.stem.replace("best_", "", 1) or "?",
        "label": path.stem[:40],
        "origin": str(path),
        "source_file": (tr or {}).get("source_file"),
        "mode": (tr or {}).get("mode") or "full",
        "n_bars": (tr or {}).get("n_bars_requested"),
        "n_chunks": (tr or {}).get("n_chunks"),
        "regime": (tr or {}).get("regime") or "vol",
        "recorded_file": str(pf) if pf.exists() else None,
        "recorded_subset_bars": (tr or {}).get("subset_bars"),
        "archived": {
            "formula": [int(t) for t in formula] if formula else None,
            "best": d.get("best_score"),
            "holdout": d.get("holdout") if isinstance(d.get("holdout"), dict) else None,
            "seed": d.get("seed"),
            "steps": d.get("steps") or (d.get("train_steps") if not is_history else None),
        },
    }
    return rec


def discover_records() -> list[dict]:
    recs: list[dict] = []
    seen = set()
    # 1) 训练子集 sidecar（当前盘上真实存在的抽样记录）
    for sidecar in sorted(Path(PROJECT_ROOT / "data" / "slices").glob("*/train_range.json")):
        r = record_from_sidecar(sidecar)
        if r:
            recs.append(r)
    # 2) 对比实验结果（带存档公式 + holdout 数字）
    for cp in sorted(Path(RESULTS_DIR).glob("compare_*.json")):
        for r in record_from_compare(cp):
            key = (r["label"], r["archived"]["seed"])
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
    # 3) 已部署/历史冠军（引擎写 data_source/train_range + holdout 的版本）
    for pat in (str(PROJECT_ROOT / "strategies" / "best_*.json"),
                str(PROJECT_ROOT / "training_history_*.json")):
        for p in sorted(glob.glob(pat)):
            r = record_from_strategy(Path(p))
            if r:
                recs.append(r)
    return recs


# ── 子集解析与确定性 ─────────────────────────────────────────────────────

def regen_params(rec: dict) -> dict:
    n_chunks = rec.get("n_chunks")
    return {
        "mode": rec.get("mode") or "full",
        "n_bars": int(rec["n_bars"]) if rec.get("n_bars") else None,
        "n_chunks": int(n_chunks) if n_chunks else None,
        "regime": rec.get("regime") or "vol",
    }


def regenerate_subset(rec: dict, root_dir: Path) -> dict | None:
    """按记录参数重生成子集到 root_dir。返回 {data_file, subset_bars, ok, err}。"""
    from data_pipeline.train_sampler import prepare_training_subset

    src = rec.get("source_file")
    if not src or not Path(src).exists():
        return {"ok": False, "err": f"源文件缺失: {src}"}
    if (rec.get("mode") or "full") == "full":
        # full = 原文件直用，无抽样
        return {"ok": True, "data_file": str(Path(src).resolve()),
                "subset_bars": None, "full": True}
    try:
        r = prepare_training_subset(
            src, root_dir=root_dir, overwrite=True, **regen_params(rec)
        )
        return {"ok": True, "data_file": r["data_file"],
                "subset_bars": r.get("subset_bars"), "full": False}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}"}


def _parquet_close_and_len(path: Path) -> tuple[int, object, str | None]:
    import pandas as pd

    try:
        df = pd.read_parquet(path)
        closes = df["close"].to_numpy(dtype="float64")
        return len(df), closes, None
    except Exception as exc:  # noqa: BLE001
        return -1, None, f"{type(exc).__name__}: {exc}"


def check_determinism(rec: dict, regen: dict) -> dict:
    """确定性：重生成文件 vs 盘上记录文件 逐行一致；记录文件已删 → 行数 parity。"""
    out = {"status": "skip", "note": ""}
    rec_file = rec.get("recorded_file")
    if rec_file and Path(rec_file).exists():
        n_rec, c_rec, err_rec = _parquet_close_and_len(Path(rec_file))
        n_re, c_re, err_re = _parquet_close_and_len(Path(regen["data_file"]))
        if err_rec or err_re:
            out.update({"status": "FAIL", "note": f"读取失败 rec={err_rec} regen={err_re}"})
            return out
        if n_rec != n_re:
            out.update({"status": "FAIL",
                        "note": f"行数不一致: 记录 {n_rec} ≠ 重生成 {n_re}"})
            return out
        import numpy as np

        if c_rec is not None and c_re is not None and c_rec.shape == c_re.shape:
            delta = float(np.max(np.abs(c_rec - c_re))) if n_rec else 0.0
        else:
            delta = -1.0
        same = delta == 0.0
        out.update({
            "status": "PASS" if same else "FAIL",
            "bars": n_rec,
            "max_close_delta": delta,
            "note": f"{n_rec} 根；close 逐根一致" if same
                    else f"{n_rec} 根但 close 最大差 {delta:.6g}（采样器代码可能已改版）",
        })
        return out
    # 记录文件不在盘（已清理）→ 与记录的 subset_bars parity
    expect = rec.get("recorded_subset_bars")
    n_re = regen.get("subset_bars")
    if regen.get("full"):
        out.update({"status": "skip", "note": "full 模式直用源文件，无抽样可校验"})
        return out
    if n_re is None or expect is None:
        out.update({"status": "skip", "note": "无记录行数可对比"})
        return out
    ok = int(n_re) == int(expect)
    out.update({
        "status": "PASS" if ok else "FAIL",
        "bars_regen": n_re, "bars_recorded": expect,
        "note": f"记录文件已清理；重生成 {n_re} 根 == 记录 {expect} 根"
                if ok else f"重生成 {n_re} 根 ≠ 记录 {expect} 根",
    })
    return out


# ── 存档公式 holdout 复验（复用生产 _verify_holdout 路径）──────────────

def _cleanup_tagged(symbol: str, tag: str) -> None:
    for pat in (str(PROJECT_ROOT / f"training_history_{symbol}_{tag}.json"),
                str(PROJECT_ROOT / "checkpoints" / f"ckpt_{symbol}_{tag}_step_*.pt")):
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except OSError:
                pass


def rescore_archived(rec: dict, subset_file: str, tag: str) -> dict:
    """在子集上用生产引擎重跑存档公式的 holdout 验证（不训练、不部署）。"""
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.engine import AlphaEngine

    formula = rec["archived"]["formula"]
    try:
        mgr = ParquetDataManager(subset_file)
        mgr.load()
        eng = AlphaEngine(data_manager=mgr, target_symbol=rec["symbol"],
                          seed=int(rec["archived"]["seed"] or 42))
        eng.run_tag = tag
        eng._suppress_deploy = True
        eng.best_formula = list(formula)
        eng.best_score = float(rec["archived"]["best"] or 1.0)
        eng.train(start_step=0, end_step=0, verbose_header=False)  # 预热 + _verify_holdout
        ho = dict(eng.holdout) if eng.holdout else None
        return {"ok": True, "holdout": ho, "tag": tag}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}", "tag": tag}
    finally:
        try:
            _cleanup_tagged(rec["symbol"], tag)
        except Exception:  # noqa: BLE001
            pass


def grade_rescore(rec: dict, got: dict | None, tol: float) -> dict:
    """存档 holdout vs 复验结果。无存档 → 只给状态读数。"""
    arch = rec["archived"]["holdout"]
    if got is None:
        return {"status": "FAIL", "note": "引擎复验无输出"}
    if not isinstance(arch, dict):
        v = got.get("val_score")
        note = (f"存档无 holdout 记录 → 现状读数：val={v:.4f} "
                f"sharpe={got.get('sharpe')} ratio={got.get('score_ratio')} "
                f"passed={got.get('passed')}")
        return {"status": "info", "note": note, "holdout": got}
    dv = abs(float(got.get("val_score") or 0.0) - float(arch.get("val_score") or 0.0))
    base = max(1e-9, abs(float(arch.get("val_score") or 0.0)))
    rel = dv / base
    ds = abs(float(got.get("sharpe") or 0.0) - float(arch.get("sharpe") or 0.0))
    passed_same = bool(got.get("passed")) == bool(arch.get("passed"))
    ok = rel <= tol and abs(ds) <= tol * 20 and passed_same
    note = (f"存档 val={arch.get('val_score')} (ratio {arch.get('score_ratio')}, "
            f"passed={arch.get('passed')}) → 复验 val={got.get('val_score')} "
            f"(ratio {got.get('score_ratio')}, passed={got.get('passed')})；"
            f"|Δval| 相对 {rel:.4f} · |Δsharpe| {ds:.4f}")
    return {
        "status": "PASS" if ok else "FAIL",
        "note": note + ("（一致）" if ok else "（不一致）"),
        "holdout": got,
    }


# ── 短程重训（可选，--retrain-steps>0）──────────────────────────────────

def short_retrain(rec: dict, subset_file: str, steps: int, seed: int, tag: str) -> dict:
    try:
        from scripts.train_variant import run_variant

        res = run_variant(subset_file, tag, steps, seed)
        return {"ok": True, "result": res}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}"}


# ── 主流程 ────────────────────────────────────────────────────────────────

def verify_records(recs: list[dict], steps: int, seed: int, tol: float,
                   restore_missing: bool, do_retrain: bool) -> list[dict]:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    verdicts: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="vrb_regen_") as td:
        tmp_root = Path(td)
        for i, rec in enumerate(recs):
            tag = f"vrb{i}_{stamp}"
            entry = {"record": rec, "checks": {}}
            # 确保子集存在（记录文件在盘则直接复用；缺失时按记录参数重生成）
            subset_file = rec.get("recorded_file")
            regen = None
            if not subset_file and (rec.get("mode") or "full") != "full":
                target = PROJECT_ROOT if restore_missing else tmp_root
                regen = regenerate_subset(rec, target)
                if not regen["ok"]:
                    entry["checks"]["det"] = {"status": "FAIL",
                                              "note": f"子集重生成失败: {regen.get('err')}"}
                    verdicts.append(entry)
                    continue
                subset_file = regen["data_file"]
                if restore_missing and rec.get("recorded_file"):
                    pass  # 已写回原始位置
            # det —— 确定性（full=直用源文件，无抽样可校验，跳过）
            if (rec.get("mode") or "full") == "full":
                entry["checks"]["det"] = {"status": "skip",
                                            "note": "full 模式直用源文件，无抽样可校验"}
            elif rec.get("recorded_file") and Path(rec["recorded_file"]).exists():
                rgen = regenerate_subset(rec, tmp_root)
                if rgen["ok"]:
                    entry["checks"]["det"] = check_determinism(rec, rgen)
                else:
                    entry["checks"]["det"] = {"status": "FAIL",
                                              "note": f"重生成失败: {rgen.get('err')}"}
            elif regen is not None:
                entry["checks"]["det"] = check_determinism(rec, regen)
            elif rec.get("recorded_file"):
                entry["checks"]["det"] = {"status": "FAIL",
                                          "note": f"记录子集缺失: {rec['recorded_file']}"}
            else:
                entry["checks"]["det"] = {"status": "skip", "note": "无子集可校验"}

            # rescore —— 存档公式 holdout 复验
            arch = rec.get("archived") or {}
            if arch.get("formula") and subset_file and Path(subset_file).exists():
                got = rescore_archived(rec, subset_file, tag)
                if got["ok"]:
                    entry["checks"]["rescore"] = grade_rescore(rec, got["holdout"], tol)
                else:
                    entry["checks"]["rescore"] = {"status": "FAIL",
                                                  "note": f"复验失败: {got.get('err')}"}
            else:
                entry["checks"]["rescore"] = {"status": "skip",
                                              "note": "记录无存档公式（纯子集记录）"}

            # retrain —— 可选短程重训
            if do_retrain and subset_file and Path(subset_file).exists():
                rt = short_retrain(rec, subset_file, steps, seed, tag + "_rt")
                if rt["ok"]:
                    r2 = rt["result"]
                    n = {"status": "info",
                         "note": f"短训 {steps} 步 best={r2.get('best_score')} "
                                 f"holdout={ (r2.get('holdout') or {}).get('passed') }",
                         "best": r2.get("best_score"),
                         "best_formula": r2.get("best_formula"),
                         "holdout": r2.get("holdout")}
                    if arch.get("best") and arch.get("steps") == steps and \
                            arch.get("seed") == seed and arch["best"]:
                        gap = abs(float(n["best"] or 0) - float(arch["best"])) / abs(float(arch["best"]))
                        n["note"] += f"；存档 best={arch['best']} 差距 {gap:.2%}"
                    entry["checks"]["retrain"] = n
                else:
                    entry["checks"]["retrain"] = {"status": "FAIL",
                                                  "note": f"短训失败: {rt.get('err')}"}
            verdicts.append(entry)
    return verdicts


def _fmt(v: dict) -> str:
    s = v.get("status", "skip")
    icons = {"PASS": "✅", "FAIL": "❌", "info": "ℹ️", "skip": "—"}
    return f"{icons.get(s, '?')} {s:<5} {v.get('note', '')}"


def render_markdown(verdicts: list[dict], argv: str) -> str:
    lines = ["# 冠军回滚校验报告",
             "",
             f"- 生成: {_dt.datetime.now().isoformat(timespec='seconds')}",
             f"- 命令: `{argv}`",
             f"- 记录数: {len(verdicts)}",
             ""]
    for entry in verdicts:
        rec = entry["record"]
        lines += [f"## {rec['label']} · {rec['symbol']}",
                  ""]
        meta = f"- kind={rec['kind']} origin=`{rec.get('origin') or rec.get('sidecar') or '—'}`"
        if rec.get("recorded_file"):
            meta += f"\n- 记录子集: `{rec['recorded_file']}`"
        if rec.get("archived") and rec["archived"].get("best") is not None:
            a = rec["archived"]
            meta += (f"\n- 存档: best={a['best']} seed={a['seed']} steps={a['steps']} "
                     f"holdout={ (a.get('holdout') or {}).get('val_score') if a.get('holdout') else '无' }")
        lines += [meta, ""]
        for ck, v in entry["checks"].items():
            lines += [f"- **{ck}**: {_fmt(v)}"]
        lines += [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="冠军回滚校验（子集确定性 + 存档公式 holdout 复验）")
    ap.add_argument("--records", nargs="*", default=None,
                    help="显式指定记录 JSON（train_range.json / compare_*.json / best_*.json）；缺省自动发现")
    ap.add_argument("--retrain-steps", type=int, default=0,
                    help=">0 时对每条记录额外跑短程重训（默认 0=跳过，只做确定性+复验）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="rescore 相对容差（val_score/存档 的相对偏差上限，默认 2%%）")
    ap.add_argument("--no-restore", action="store_true",
                    help="缺失子集只重生成到临时目录，不写回 data/slices")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    if args.records:
        recs: list[dict] = []
        for raw in args.records:
            p = Path(raw)
            if not p.exists():
                print(f"[skip] 记录不存在: {p}")
                continue
            if p.name == "train_range.json":
                r = record_from_sidecar(p)
                if r:
                    recs.append(r)
            elif "compare_" in p.name:
                recs.extend(record_from_compare(p))
            else:
                r = record_from_strategy(p)
                if r:
                    recs.append(r)
    else:
        recs = discover_records()
    if not recs:
        print("没有找到任何可校验的记录（data/slices/*/train_range.json、"
              "results/compare_*.json、带公式的 strategies/training_history JSON）")
        sys.exit(2)

    verdicts = verify_records(recs, args.retrain_steps, args.seed, args.tol,
                              restore_missing=not args.no_restore,
                              do_retrain=args.retrain_steps > 0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_p = out_dir / f"verify_champion_rollback_{stamp}.json"
    md_p = out_dir / f"verify_champion_rollback_{stamp}.md"

    simple = [{"record": {k: e["record"][k] for k in ("label", "symbol", "kind", "origin", "sidecar")
                         if k in e["record"]},
               "checks": e["checks"]} for e in verdicts]
    json_p.write_text(json.dumps(simple, ensure_ascii=False, indent=2), encoding="utf-8")
    md_p.write_text(render_markdown(verdicts, " ".join(sys.argv)), encoding="utf-8")

    n_fail = 0
    print("\n=== 冠军回滚校验 ===")
    for entry in verdicts:
        rec = entry["record"]
        print(f"\n◆ {rec['label']} · {rec['symbol']}")
        for ck, v in entry["checks"].items():
            print(f"   {_fmt(v)}")
            if v.get("status") == "FAIL":
                n_fail += 1
    print(f"\n报告: {md_p}")
    print(f"结果: {json_p}")
    print(f"{'❌ 存在 FAIL' if n_fail else '✅ 全部 PASS/INFO'}（{n_fail} 个 FAIL）")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
