"""exp4_bc_pretrain.py — E4 实验: 行为克隆(BC)预热 + RL 微调 vs 从零 RL。

两阶段:
  1) pretrain: 用 strategies/best_*.json + champion_history.json(含 rejected/approved
     candidate)里的真实公式语料,对 AlphaGPT 做 next-token 行为克隆(teacher forcing,
     状态 = 采样时同款前缀),保存权重到 --bc-pt。
  2) rl: 相同训练预算下对比两条 arm:
       --init scratch = 从随机初始化直接 RL(现状)
       --init bc      = 用 BC 权重初始化(engine.model + _best_snapshot 都替换),
                        再 RL 微调
  输出 val/best 曲线 + 达到指定 val 分数所需步数 + 墙钟时间。

实验纪律(与 train_variant/exp1 一致): run_tag 隔离 + _suppress_deploy=True,
不写 best_*.json / *.live.json,不消费 holdout 单次批准,结束清理 tagged 残留。

用法:
  # 1) BC 预热(语料自动收集)
  .venv/bin/python scripts/exp4_bc_pretrain.py --mode pretrain --bc-pt results/exp4_bc.pt \
      --bc-steps 300 --out-json results/exp4_pretrain.json
  # 2a) 从零 RL arm
  .venv/bin/python scripts/exp4_bc_pretrain.py --mode rl --init scratch \
      --data-file <parquet> --steps 200 --seed 42 --tag exp4_scratch \
      --out-json results/exp4_scratch.json
  # 2b) BC 预热后 RL 微调 arm(同预算)
  .venv/bin/python scripts/exp4_bc_pretrain.py --mode rl --init bc \
      --bc-pt results/exp4_bc.pt --data-file <parquet> --steps 200 --seed 42 \
      --tag exp4_bc --out-json results/exp4_bc.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_core.config import ModelConfig  # noqa: E402
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION  # noqa: E402


# ── 语料收集 ────────────────────────────────────────────────────────────────

_VAL_KEYS = ("best_score", "val_score", "candidate_val", "val", "score",
             "val_score_out")
_FML_KEYS = ("formula", "best_formula", "candidate_formula", "fml", "tokens",
             "champion_formula")


def _add_item(seqs: dict[tuple[int, ...], float | None],
              f: object, val: object | None) -> None:
    """把 (公式, val) 收进池: 非法序列丢弃; 已存在只保留更高 val。"""
    if not isinstance(f, list) or not (2 <= len(f) <= 20):
        return
    if not all(isinstance(t, int) and 0 <= t < FORMULA_VOCAB.size for t in f):
        return
    key = tuple(f)
    v = val if isinstance(val, (int, float)) and not isinstance(val, bool) \
        else None
    v = None if v is None else float(v)
    old = seqs.get(key, None)
    if old is None or (v is not None and v > old):
        seqs[key] = v if v is not None else old


def _collect_corpus(root: Path, elite_ckpt: str | None = None
                    ) -> dict[tuple[int, ...], float | None]:
    """收集真实公式语料 → {公式: val}(去重, 取最高 val)。

    来源: strategies/best_*.json(部署冠军 + best_score),
    strategies/finalists_*.json / champion_history.json(带候选 val),以及
    --ckpt 里已部署 820 步冠军运行的 elite_pool(60 条 top 公式, 带分数)。
    旧 3 条语料(best + champion_history)严格是新池子集 → 干净消融。
    """
    seqs: dict[tuple[int, ...], float | None] = {}

    def add_scored(d: dict, fml_keys=_FML_KEYS, val_keys=_VAL_KEYS) -> None:
        val = next((d.get(k) for k in val_keys if isinstance(d.get(k), (int, float))
                    and not isinstance(d.get(k), bool)), None)
        for k in fml_keys:
            _add_item(seqs, d.get(k), val)

    for p in glob.glob(str(root / "strategies" / "best_*.json")):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(d, dict):
                add_scored(d)
        except Exception:  # noqa: BLE001
            pass
    for p in glob.glob(str(root / "strategies" / "finalists_*.json")):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            items = d if isinstance(d, list) else [d]
            for it in items:
                if isinstance(it, dict):
                    add_scored(it)
        except Exception:  # noqa: BLE001
            pass
    p = root / "strategies" / "champion_history.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, list):
                for ev in d:
                    if isinstance(ev, dict):
                        add_scored(ev)
        except Exception:  # noqa: BLE001
            pass
    if elite_ckpt:
        try:
            ck = torch.load(elite_ckpt, map_location="cpu", weights_only=False)
            pool = ck.get("elite_pool") or []
            for e in pool:
                if not isinstance(e, (tuple, list)) or len(e) < 3:
                    continue
                _add_item(seqs, e[2], e[0])   # (score, counter, tokens, birth)
            print(f"[exp4:collect] elite 池来源: {elite_ckpt} "
                  f"({len(pool)} 条, 含冠军运行 top-60)")
        except Exception as exc:  # noqa: BLE001
            print(f"[exp4:collect] 警告: 读 elite ckpt 失败 -> {exc}")
    return seqs


def _val_weights(items: list[tuple[list[int], float]], temp: float):
    """val 加权采样权重: 先 z-score(减中位/除 std)再 softmax(z/temp)。

    抵抗不同运行间 val 尺度的漂移; temp 越大越接近均匀。
    """
    vals = torch.tensor([v for _, v in items], dtype=torch.float64)
    if vals.numel() == 1:
        return torch.ones(1, dtype=torch.float64)
    med = vals.median()
    sd = vals.std().clamp_min(1e-6)
    z = (vals - med) / sd
    w = torch.softmax(z / max(float(temp), 1e-6), dim=0)
    return w


# ── BC 预热 ─────────────────────────────────────────────────────────────────

def run_pretrain(bc_pt: str, bc_steps: int, out_json: str,
                 elite_ckpt: str | None, per_step: int, temp: float) -> None:
    from model_core.alphagpt import AlphaGPT

    raw = _collect_corpus(PROJECT_ROOT, elite_ckpt)
    if not raw:
        raise SystemExit("语料为空: strategies/best_*.json 或 champion_history.json 里没有公式")
    items: list[tuple[list[int], float]] = []
    for k, v in raw.items():
        if v is None:
            continue
        items.append((list(k), v))
    items.sort(key=lambda kv: -kv[1])
    n_valued = len(items)
    vals = [v for _, v in items]
    print(f"[exp4:pretrain] 语料公式数={len(raw)}(有 val 可加权 {n_valued}) "
          f"val: min={min(vals):.3f} med={sorted(vals)[n_valued//2]:.3f} "
          f"max={max(vals):.3f}")
    print(f"[exp4:pretrain] 采样: per_step={per_step} "
          f"temp={temp}  {'加权' if per_step < n_valued else '全池(等权)'}")
    names = FORMULA_VOCAB.token_names
    for s, v in items[:6]:
        print(f"  val={v:.3f} {len(s)} tokens: "
              + " ".join(names[t] for t in s))
    if len(items) > 6:
        print(f"  ... 其余 {len(items)-6} 条略")

    torch.manual_seed(0)
    model = AlphaGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses: list[float] = []
    t0 = time.time()
    w_all = _val_weights(items, temp)
    for step in range(max(1, int(bc_steps))):
        # val 加权采样: 每步抽 per_step 条(无放回); per_step>=n 时退化为全池
        if per_step >= n_valued:
            chosen = items
        else:
            idx = torch.multinomial(w_all, int(per_step), replacement=False)
            chosen = [items[i] for i in idx.tolist()]
        opt.zero_grad()
        tot = torch.zeros((), device=ModelConfig.DEVICE)
        n = 0
        for seq, _v in chosen:
            # 状态 = 采样时同款前缀: 起始 token 0 + tokens[0..i-1], 预测 tokens[i]
            for i in range(1, len(seq)):
                prefix = torch.tensor([[0] + seq[:i]], dtype=torch.long,
                                      device=ModelConfig.DEVICE)
                logits, _, _ = model(prefix)   # [1, vocab]
                tgt = torch.tensor([seq[i]], dtype=torch.long,
                                   device=ModelConfig.DEVICE)
                tot = tot + torch.nn.functional.cross_entropy(logits, tgt)
                n += 1
        loss = tot / max(1, n)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == bc_steps - 1:
            print(f"[exp4:pretrain] step {step+1}/{bc_steps} loss={losses[-1]:.4f} "
                  f"({time.time()-t0:.1f}s)")
    wall_s = time.time() - t0

    payload = {
        "model_state_dict": model.state_dict(),
        "vocab_version": VOCAB_VERSION,
        "meta": {
            "n_formulas": len(raw),
            "n_valued": n_valued,
            "per_step": int(per_step), "temp": float(temp),
            "elite_ckpt": str(elite_ckpt) if elite_ckpt else None,
            "val_min": round(min(vals), 4), "val_med": round(float(
                sorted(vals)[n_valued // 2]), 4), "val_max": round(max(vals), 4),
            "bc_steps": max(1, int(bc_steps)),
            "final_loss": round(losses[-1], 6),
            "loss_curve": [round(x, 6) for x in losses],
            "wall_s": round(wall_s, 1),
            "corpus": [{"f": list(s), "val": v} for s, v in items],
        },
    }
    out = Path(bc_pt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    Path(out_json).write_text(
        json.dumps(payload["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp4:pretrain] 权重已存 -> {bc_pt} (wall={wall_s:.1f}s)")


# ── RL arm(scratch / bc)─────────────────────────────────────────────────────

def _cleanup_artifacts(tag: str) -> None:
    for pat in (str(PROJECT_ROOT / f"training_history_*{tag}*.json"),
                str(PROJECT_ROOT / "checkpoints" / f"ckpt_*{tag}*_step_*.pt")):
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except OSError:
                pass


def _apply_vol_gate(vg: str) -> None:
    """显式钉住选优层 vol 覆盖口径(default=跟随 ModelConfig)。"""
    if vg == "default":
        return
    if vg == "off":
        ModelConfig.VOL_COVERAGE_ENABLED = False
        return
    ModelConfig.VOL_COVERAGE_ENABLED = True
    ModelConfig.VOL_COVERAGE_GRID = (vg == "grid")


def run_rl(data_file: str, init: str, bc_pt: str | None, tag: str,
           steps: int, seed: int, vol_gate: str = "default") -> dict:
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.engine import AlphaEngine

    assert init in ("scratch", "bc")
    _apply_vol_gate(vol_gate)
    old_steps = ModelConfig.TRAIN_STEPS
    ModelConfig.TRAIN_STEPS = max(1, int(steps))
    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        symbol, timeframe = mgr.symbol, mgr.timeframe
        engine = AlphaEngine(data_manager=mgr, target_symbol=symbol, seed=int(seed))
        engine.timeframe = timeframe
        engine.data_file = str(Path(data_file).resolve())
        engine.run_tag = tag
        engine._suppress_deploy = True

        if init == "bc":
            if not bc_pt or not Path(bc_pt).exists():
                raise SystemExit(f"--init bc 需要存在的 --bc-pt: {bc_pt!r}")
            ck = torch.load(bc_pt, map_location=ModelConfig.DEVICE)
            engine.model.load_state_dict(ck["model_state_dict"])
            engine._best_snapshot = copy.deepcopy(ck["model_state_dict"])
            print(f"[exp4:rl] BC 权重注入完成 (vocab v={ck.get('vocab_version')})")

        t0 = time.time()
        engine.train(verbose_header=False)
        wall_s = time.time() - t0

        hist = engine.training_history
        best_curve = hist.get("best_score", [])
        step_ids = hist.get("step", [])
        best = engine.best_score if engine.best_score not in (None, -float("inf")) else None
        n_steps = len(step_ids)
        result = {
            "tag": tag, "init": init, "seed": int(seed),
            "steps_requested": max(1, int(steps)), "steps_done": n_steps,
            "wall_s": round(wall_s, 1),
            "s_per_step": round(wall_s / max(1, n_steps), 3),
            "symbol": symbol, "timeframe": timeframe,
            "data_file": str(Path(data_file).resolve()),
            "source_bars": int(mgr.raw_dict["close"].shape[1]),
            "restart_count": int(getattr(engine, "_restart_count", 0) or 0),
            "best_score": round(float(best), 6) if best is not None else None,
            "best_formula": engine.best_formula,
            "formula_decoded": engine._decode_formula(engine.best_formula),
            "holdout": (engine.holdout or None),
            "first_step_ge_1p0": next(
                (int(s) for s, v in zip(step_ids, best_curve)
                 if isinstance(v, (int, float)) and v >= 1.0), None),
            "first_step_ge_0p5": next(
                (int(s) for s, v in zip(step_ids, best_curve)
                 if isinstance(v, (int, float)) and v >= 0.5), None),
            "history": {
                "step": list(step_ids),
                "val_score": list(hist.get("val_score", [])),
                "best_score": list(best_curve),
                "avg_reward": list(hist.get("avg_reward", [])),
            },
        }
        return result
    finally:
        ModelConfig.TRAIN_STEPS = old_steps


def main() -> None:
    ap = argparse.ArgumentParser(description="E4: BC 预热 + RL 微调 vs 从零 RL")
    ap.add_argument("--mode", required=True, choices=["pretrain", "rl"])
    ap.add_argument("--init", choices=["scratch", "bc"], default="scratch")
    ap.add_argument("--bc-pt", default="results/exp4_bc.pt")
    ap.add_argument("--bc-steps", type=int, default=300)
    ap.add_argument("--ckpt", default="checkpoints/ckpt_BTCUSDT_step_0820.pt",
                    help="已部署运行 elite_pool 来源(全池扩展); None 关掉")
    ap.add_argument("--per-step", type=int, default=12,
                    help="每步 val 加权采样的公式条数; >= 池大则退化为全池")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="val 加权温度(z/softmax); 越大越均匀")
    ap.add_argument("--data-file", default="data/training/BINANCE_BTCUSDT_H1.parquet")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--vol-gate", choices=["default", "tier", "grid", "off"],
                    default="default",
                    help="选优层 vol 覆盖口径钉住(default=跟随 ModelConfig; "
                         "tier=段级3段, grid=vol×er 9格, off=关)")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    if args.mode == "pretrain":
        ckpt = None if args.ckpt in ("None", "none", "") else args.ckpt
        run_pretrain(args.bc_pt, args.bc_steps, args.out_json,
                     ckpt, args.per_step, args.temp)
        return
    if not args.tag:
        ap.error("--tag 在 --mode rl 下必填")

    res = run_rl(args.data_file, args.init, args.bc_pt, args.tag,
                 args.steps, args.seed, args.vol_gate)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _cleanup_artifacts(args.tag)
    except Exception:  # noqa: BLE001
        pass
    print(f"[exp4:{args.tag}] done -> {out}")
    print(f"  init={res['init']} steps={res['steps_done']} "
          f"wall={res['wall_s']}s ({res['s_per_step']}s/step) best_val={res['best_score']}")
    print(f"  first best>=0.5 @ step {res['first_step_ge_0p5']} | "
          f">=1.0 @ step {res['first_step_ge_1p0']}")
    if res["best_formula"]:
        print(f"  best={res['best_formula']}  =>  {res['formula_decoded']}")
    if res.get("holdout"):
        h = res["holdout"]
        print(f"  holdout: val={h.get('val_score')} sharpe={h.get('sharpe')} "
              f"passed={h.get('passed')}")


if __name__ == "__main__":
    main()
