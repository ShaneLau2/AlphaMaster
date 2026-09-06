"""
train_file.py — 从单个 Parquet K 线文件训练

用法:
    python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet
    python train_file.py --data-file AAPL_H1.parquet --seed 42
    python train_file.py --data-file AAPL_H1.parquet --seeds 17,23,31   # 双/多 seed 升级纪律

文件名格式: {品种}_{周期}.parquet，例如 AAPL_H1.parquet、US30.cash_H1.parquet

冠军闸门（P0/P1）：
- val 只用于入围 top-K；最终冠军在从未参与选优的 holdout 尾部窗口上按生产配置
  （成本 + tanh 仓位）跑真实回测选出；
- holdout 闸门（分>0、保持率≥0.30、Sharpe≥0、同数据版本单次消费）不过时
  【不覆盖】 strategies/best_{symbol}.json，保留原冠军并记录 champion_history；
- --seeds A,B：每个 seed 完整训练一轮，全部 seed 的 val 都超过旧冠军 +
  CHAMPION_UPGRADE_MARGIN 才允许部署（任意 seed 不过则保留旧冠军）。
"""
from __future__ import annotations

import glob as _glob
import json
import pathlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.vocab import VOCAB_VERSION


def train_from_file(data_file: str, *, from_scratch: bool = False,
                    seed: int | None = None, deploy: bool = True) -> AlphaEngine | None:
    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    timeframe = info["timeframe"]

    print(f"\n{'='*60}")
    print(f"  AlphaGPT 文件训练 — {info['filename']}")
    print(f"{'='*60}")
    print(f"  品种: {symbol}")
    print(f"  周期: {timeframe}")
    print(f"  数据: 本地 Parquet（强制离线）")
    print(f"  文件: {Path(data_file).resolve()}")
    print(f"  训练步数: {ModelConfig.TRAIN_STEPS}")
    print(f"  K线数: {info['bars']}")
    print(f"  模式: {'重新训练（从头）' if from_scratch else '自动续训'}")
    if seed is not None:
        print(f"  seed: {seed}")
    if not deploy:
        print(f"  部署: 抑制（双 seed 模式，全部 seed 通过后统一部署）")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线（指纹 {mgr.fingerprint}）")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    engine = AlphaEngine(data_manager=mgr, target_symbol=symbol, seed=seed)
    engine.timeframe = timeframe
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = "parquet_file"
    engine.train_steps = ModelConfig.TRAIN_STEPS
    if not deploy:
        engine._suppress_deploy = True

    ckpt_pattern = str(pathlib.Path("checkpoints") / f"ckpt_{symbol}_step_*.pt")
    ckpt_files = sorted(_glob.glob(ckpt_pattern))
    start_step = 0

    if from_scratch:
        removed = 0
        for p in ckpt_files:
            try:
                pathlib.Path(p).unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                print(f"  [警告] 无法删除检查点 {p}: {e}")
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            try:
                hist_path.unlink()
            except OSError:
                pass
        print(f"  [重新训练] 已清除 {removed} 个检查点，从第 0 步开始")
        # 保留已有最优策略作为分数下限，避免开局弱公式覆盖 strategies/best_*.json
        _seed_best_from_strategy(engine, symbol)
        ckpt_files = []
    elif ckpt_files:
        latest = ckpt_files[-1]
        try:
            start_step = engine.load_checkpoint(latest)
            print(f"  [续训] 从 {latest} 恢复，起始步={start_step}")
        except Exception as e:
            print(f"  [警告] 检查点加载失败: {e}，将从头开始")

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [完成] {symbol} 已完成全部 {ModelConfig.TRAIN_STEPS} 步，跳过训练")
        _verify_holdout_or_warn(engine)
        if not getattr(engine, "champion_outcome", None):
            # 续训完成路径：同样走冠军闸门（真实回测选优 + 单次消费）
            engine._finalize_champion()
        _save_strategy(engine, symbol, timeframe, data_file)
        return engine

    if start_step == 0 and not from_scratch:
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            hist_path.unlink()
        print("  [新训] 从第 0 步开始")

    if start_step > 0:
        engine._save_training_history_live(force=True)   # 训练收尾快照恒写

    engine.train(start_step=start_step)
    _verify_holdout_or_warn(engine)
    _save_strategy(engine, symbol, timeframe, data_file)
    return engine


def train_multi_seed(data_file: str, seeds: list[int], from_scratch: bool = False,
                     deploy: bool = True) -> list[AlphaEngine | None]:
    """双/多 seed 升级纪律：每个 seed 完整训练一轮（互不干扰），\n\n    全部 seed 的 val 都超过旧冠军 + CHAMPION_UPGRADE_MARGIN 才部署最佳者。\n    """
    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    margin = float(ModelConfig.CHAMPION_UPGRADE_MARGIN)

    old_path = pathlib.Path("strategies") / f"best_{symbol}.json"
    old_champ: dict | None = None
    old_score = float("-inf")
    if old_path.exists():
        try:
            old_champ = json.loads(old_path.read_text(encoding="utf-8"))
            old_score = float(old_champ.get("best_score") or float("-inf"))
        except Exception:
            old_champ = None
    print(f"\n[多seed] 旧冠军 best_score={old_score if old_score != float('-inf') else '—'}"
          f"，升级裕度={margin}")

    results: list[AlphaEngine | None] = []
    for seed in seeds:
        _stash_symbol_state(symbol, seed)
        print(f"\n{'#'*20} seed={seed} 开始（检查点已隔离）{'#'*20}")
        try:
            eng = train_from_file(data_file, from_scratch=from_scratch,
                                  seed=seed, deploy=False)
        finally:
            _unstash_symbol_state(symbol, seed)
        results.append(eng)
        if eng is not None:
            ok = (eng.best_formula is not None and old_score != float("-inf")
                  and eng.best_score > old_score + margin) or \
                 (eng.best_formula is not None and old_score == float("-inf"))
            print(f"  [seed={seed}] best={eng.best_score:.4f} "
                  f"{'✓ 超过旧冠军+裕度' if ok else '✗ 未达升级线'}")

    passed = [e for e in results
              if e is not None and e.best_formula is not None
              and (old_score == float("-inf") or e.best_score > old_score + margin)]
    if len(passed) == len(seeds) and passed:
        best = max(passed, key=lambda e: e.best_score)
        print(f"\n[多seed] 全部 {len(seeds)} 个 seed 通过升级线，部署最佳者 "
              f"best={best.best_score:.4f}")
        if deploy:
            best._suppress_deploy = False
            best._finalize_champion()
        return results
    print(f"\n[多seed] 仅 {len(passed)}/{len(seeds)} 个 seed 通过升级线——"
          f"保留旧冠军，不部署（避免单次 argmax 上生产）")
    return results


def _stash_symbol_state(symbol: str, seed: int) -> None:
    """把该品种的检查点/历史挪到隔离目录（各 seed 互不干扰）。"""
    backup = pathlib.Path("checkpoints") / f".seed_backup_{symbol}_{seed}"
    backup.mkdir(parents=True, exist_ok=True)
    for pattern, dest in (
        (f"checkpoints/ckpt_{symbol}_step_*.pt", backup),
        (f"training_history_{symbol}.json", backup),
    ):
        for p in pathlib.Path(".").glob(pattern):
            try:
                shutil.move(str(p), str(backup / p.name))
            except OSError as e:
                print(f"  [警告] 隔离 {p} 失败: {e}")


def _unstash_symbol_state(symbol: str, seed: int) -> None:
    """把隔离目录里的检查点/历史挪回来（当前 seed 自己的除外——由下次训练覆盖）。"""
    backup = pathlib.Path("checkpoints") / f".seed_backup_{symbol}_{seed}"
    if not backup.exists():
        return
    for p in backup.iterdir():
        try:
            shutil.move(str(p), str(pathlib.Path(".") / p.name))
        except OSError as e:
            print(f"  [警告] 恢复 {p} 失败: {e}")


def _verify_holdout_or_warn(engine: AlphaEngine) -> None:
    """训练结束后触发样本外验证（幂等）；无预留/无公式时打印提示。"""
    result = engine._verify_holdout()
    if result is None:
        if engine.holdout_bars > 0 and engine.best_formula is None:
            print("  [holdout] 未找到最优公式，跳过样本外验证")
    else:
        ratio = result.get("score_ratio")
        ratio_str = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "—"
        verdict = "✓ 闸门通过" if result.get("passed") else "✗ 闸门未过"
        print(
            f"  [holdout] 样本外验证: 尾部 {result['bars']} 根 "
            f"收益={result['total_return_pct']:+.2f}% "
            f"Sharpe={result['sharpe']} 保持率={ratio_str} → {verdict}"
        )
        for r in (result.get("gate") or []):
            print(f"      - {r}")


def _seed_best_from_strategy(engine: AlphaEngine, symbol: str) -> None:
    """把已有 best_{symbol}.json 当作重新训练的分数下限。"""
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [警告] 读取已有策略失败: {e}")
        return
    formula = data.get("formula")
    score = data.get("best_score")
    if not formula or score is None:
        return
    try:
        engine.best_formula = [int(t) for t in formula]
        engine.best_score = float(score)
        print(f"  [重新训练] 保留已有最优分数下限={engine.best_score:.4f}，仅更好时才会覆盖策略文件")
    except (TypeError, ValueError) as e:
        print(f"  [警告] 已有策略无法用作下限: {e}")


def _load_train_range(data_file: str) -> dict[str, Any] | None:
    """读取数据文件旁的 train_range.json（训练数据范围溯源）。"""
    try:
        from data_pipeline.train_sampler import read_train_range

        return read_train_range(data_file)
    except Exception:  # noqa: BLE001 溯源缺失不影响策略保存
        return None


def _save_strategy(engine: AlphaEngine, symbol: str, timeframe: str, data_file: str) -> None:
    """保存策略。冠军闸门启用时以 engine.champion_outcome 为准（引擎已原子写入/恢复）。"""
    oc = getattr(engine, "champion_outcome", None)
    if oc:
        action = oc.get("action")
        if action == "deploy":
            print(f"  [策略] 已由冠军闸门部署（holdout 真实回测 + 统计闸门通过）: "
                  f"{oc.get('save_path')} best={oc.get('best_score')}")
            return
        if action == "reject_restore":
            print(f"  [策略] 闸门拒绝部署，已恢复旧冠军: {oc.get('save_path')}")
            for r in (oc.get("reasons") or []):
                print(f"      - {r}")
            return
        if action == "suppressed":
            print(f"  [策略] 部署被抑制（双 seed 模式，由主流程统一部署）")
            return
        # no_champion：落入下方常规逻辑

    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    path.parent.mkdir(exist_ok=True)
    if engine.best_formula is None:
        print("  未发现有效公式，策略未保存")
        return
    # 若磁盘上已有更高分，不要用更弱结果覆盖
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old_score = old.get("best_score")
            if old_score is not None and float(old_score) > float(engine.best_score):
                print(
                    f"  [策略] 保留磁盘更优结果 {float(old_score):.4f} "
                    f"> 本次 {float(engine.best_score):.4f}，未覆盖 {path}"
                )
                merged = dict(old)
                for key, val in (
                    ("timeframe", timeframe),
                    ("data_file", str(Path(data_file).resolve())),
                    ("mode", "parquet_file"),
                    ("train_steps", ModelConfig.TRAIN_STEPS),
                    ("train_range", _load_train_range(data_file)),
                ):
                    if val is not None and not merged.get(key):
                        merged[key] = val
                if merged != old:
                    path.write_text(
                        json.dumps(merged, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"  [策略] 已补全数据路径等元数据: {path}")
                return
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": str(Path(data_file).resolve()),
        "mode": "parquet_file",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
        if engine.best_formula
        else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
        "holdout_bars": int(getattr(engine, "holdout_bars", 0) or 0),
        "holdout": (getattr(engine, "holdout", None) or None),
        "train_range": _load_train_range(data_file),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  策略已保存: {path}")


if __name__ == "__main__":
    ModelConfig.REWARD_MODE = "ftmo"

    if "--data-file" not in sys.argv:
        print("用法: python train_file.py --data-file PATH\\TO\\SYMBOL_TF.parquet "
              "[--from-scratch] [--seed N] [--seeds A,B,C] [--steps N] [--holdout-bars N]")
        print("示例: python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet")
        print("      python train_file.py --data-file AAPL_H1.parquet --seeds 17,23")
        print("      python train_file.py --data-file BTCUSDT_M5.parquet --steps 300 --holdout-bars 3000")
        sys.exit(1)

    idx = sys.argv.index("--data-file")
    if idx + 1 >= len(sys.argv):
        print("错误: --data-file 后需要文件路径")
        sys.exit(1)

    data_file = sys.argv[idx + 1]
    from_scratch = "--from-scratch" in sys.argv
    no_deploy = "--no-deploy" in sys.argv
    if no_deploy:
        print("[短训] --no-deploy：抑制冠军闸门部署，不写入 strategies/（仅验证流程）")
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
        if steps <= 0:
            print("错误: --steps 必须为正整数")
            sys.exit(1)
        default_steps = ModelConfig.TRAIN_STEPS
        ModelConfig.TRAIN_STEPS = steps
        print(f"[短训] --steps {steps}：本轮只训练 {steps} 步（默认 {default_steps}）")
    seed = None
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    seeds = None
    if "--seeds" in sys.argv:
        seeds = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
    # 样本外预留根数覆盖（默认 500；大 holdout 如 3000 让闸门分数更可信）
    if "--holdout-bars" in sys.argv:
        hb = int(sys.argv[sys.argv.index("--holdout-bars") + 1])
        if hb < 0:
            print("错误: --holdout-bars 必须 >= 0")
            sys.exit(1)
        ModelConfig.HOLDOUT_BARS = hb
        print(f"[训练] --holdout-bars {hb}：样本外预留根数覆盖默认 500（随引擎写入溯源）")
    # 环境变量覆盖（实验不用改代码）：AM_NEUTRAL_W 开中性带正则、AM_REWARD_MODE 切奖励模式
    import os as _os

    for _env, _attr, _cast in (("AM_NEUTRAL_W", "NEUTRAL_BAND_W", float),
                               ("AM_REWARD_MODE", "REWARD_MODE", str)):
        _v = _os.getenv(_env)
        if _v is not None:
            setattr(ModelConfig, _attr, _cast(_v))
            print(f"[训练] 环境变量 {_env}={_v} → ModelConfig.{_attr}")
    # 防终端回收假中止：孤儿化 SIGTERM → 重挂 launchd 继续跑
    try:
        from model_core.supervise import install_orphan_sigterm_handler
        install_orphan_sigterm_handler("train_file",
                                       log_path=str(Path(__file__).parent / "logs" / "train_file_guard.log"))
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()

    if seeds:
        engs = train_multi_seed(data_file, seeds, from_scratch=from_scratch,
                                deploy=not no_deploy)
        eng = next((e for e in engs if e is not None), None)
        label = f"多seed {seeds}"
    else:
        eng = train_from_file(data_file, from_scratch=from_scratch, seed=seed,
                              deploy=not no_deploy)
        label = f"seed={seed}" if seed is not None else "默认seed"
    elapsed = time.time() - t0

    if eng:
        sym = eng.target_symbol or "?"
        print(f"\n<<< [{sym}] 训练完成({label}): 最优分数={eng.best_score:.4f}，耗时 {elapsed/3600:.2f} 小时")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
    else:
        print("\n<<< 训练失败")
        sys.exit(1)