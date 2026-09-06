"""M2 多市场训练器单元测试（Mode C）。

覆盖: registry 读取/训练腿过滤、_aggregate 组级等权聚合与 min 门槛、
市场批次采样(calib 步=全市场)、MarketEnv 批量评估形状、端到端 1 步训练。
慢用例(真实 parquet 加载)控制在 ~15s。
"""
from __future__ import annotations

import random

import pytest
import torch

from model_core.config import ModelConfig
from data_pipeline.market_registry import load_registry, train_markets, min_bars_for


# ── registry ──────────────────────────────────────────────────────────────
def test_registry_load_and_policies():
    reg = load_registry()
    assert "crypto" in reg["policies"]
    assert "equity" in reg["policies"]
    assert len(reg["oos_review"]) >= 5  # PAXG/XAUUSD/EURUSD/600519/AAPL(+D1)


def test_registry_train_markets_filter():
    reg = load_registry()
    ms = train_markets(reg, "crypto")
    syms = [m["symbol"] for m in ms if m.get("role") == "train"]
    assert "BTCUSDT" in syms
    assert "ETHUSDT" in syms
    assert min_bars_for(reg, "crypto") == 8000


# ── 聚合逻辑(纯函数) ──────────────────────────────────────────────────────
def test_aggregate_group_mean_and_min_gate(monkeypatch):
    from model_core.multimarket import MultiMarketEngine

    eng = MultiMarketEngine.__new__(MultiMarketEngine)
    monkeypatch.setattr(ModelConfig, "MARKET_GROUP_MEAN", True)
    monkeypatch.setattr(ModelConfig, "MARKET_MIN_SHARPE", 0.0)

    dev = ModelConfig.DEVICE
    mk1 = {"env": type("E", (), {"group": "crypto"})(),
           "rewards": torch.tensor([2.0, 1.0, -1.0], device=dev),
           "vals": torch.tensor([1.8, 0.9, -1.2], device=dev),
           "status": ["ok", "ok", "ok"]}
    mk2 = {"env": type("E", (), {"group": "crypto"})(),
           "rewards": torch.tensor([4.0, -2.0, -1.0], device=dev),
           "vals": torch.tensor([3.6, -1.8, -1.2], device=dev),
           "status": ["ok", "ok", "ok"]}
    mk3 = {"env": type("E", (), {"group": "commodity"})(),
           "rewards": torch.tensor([0.0, 0.5, 2.0], device=dev),
           "vals": torch.tensor([0.0, 0.4, 1.9], device=dev),
           "status": ["ok", "ok", "ok"]}
    agg_r, agg_v, min_fail = eng._aggregate([mk1, mk2, mk3], 3)

    # 组级等权: crypto 组 = mean(mk1, mk2), 再与 commodity 组均分
    crypto_r = (torch.tensor([2.0, 1.0, -1.0]) + torch.tensor([4.0, -2.0, -1.0])) / 2
    assert torch.allclose(agg_r.cpu(), (crypto_r + torch.tensor([0.0, 0.5, 2.0])) / 2)
    # min 门槛: mk2 第 2 个市场 val=-1.8 < 0 → 该公式 min_fail
    assert min_fail.cpu().tolist() == [False, True, True]  # i=1: mk2 -1.8; i=2: mk1 -1.2


def test_aggregate_plain_mean(monkeypatch):
    from model_core.multimarket import MultiMarketEngine

    eng = MultiMarketEngine.__new__(MultiMarketEngine)
    monkeypatch.setattr(ModelConfig, "MARKET_GROUP_MEAN", False)
    monkeypatch.setattr(ModelConfig, "MARKET_MIN_SHARPE", 0.0)

    dev = ModelConfig.DEVICE
    per_env = [
        {"env": type("E", (), {"group": "crypto"})(),
         "rewards": torch.tensor([2.0, 4.0], device=dev),
         "vals": torch.tensor([2.0, 4.0], device=dev),
         "status": ["ok", "ok"]},
        {"env": type("E", (), {"group": "commodity"})(),
         "rewards": torch.tensor([0.0, 8.0], device=dev),
         "vals": torch.tensor([0.0, 8.0], device=dev),
         "status": ["ok", "ok"]},
    ]
    agg_r, _, _ = eng._aggregate(per_env, 2)
    assert torch.allclose(agg_r.cpu(), torch.tensor([1.0, 6.0]))


# ── 市场批次采样 ──────────────────────────────────────────────────────────
def test_sample_market_batch_calib_and_weighted(monkeypatch):
    from model_core.multimarket import MultiMarketEngine

    eng = MultiMarketEngine.__new__(MultiMarketEngine)
    fake = [type("E", (), {"name": f"m{i}"})() for i in range(3)]
    eng.envs = fake
    eng._market_weights = [1.0 / 3.0] * 3
    monkeypatch.setattr(ModelConfig, "MARKET_CALIB_EVERY", 2)
    monkeypatch.setattr(ModelConfig, "MARKET_BATCH_SIZE", 1)

    random.seed(7)
    calib = eng._sample_market_batch(0)   # 0 % 2 == 0 → 全市场
    assert len(calib) == 3
    normal = eng._sample_market_batch(1)  # 非 calib → batch=1 加权采样
    assert len(normal) == 1 and normal[0].name in ("m0", "m1", "m2")


# ── MarketEnv 集成(真实 parquet, 小文件) ──────────────────────────────────
def test_market_env_eval_shapes():
    from data_pipeline.market_env import MarketEnv

    env = MarketEnv({"symbol": "XAUUSD", "timeframe": "H1", "group": "commodity",
                     "role": "oos", "file": "data/training/XAUUSD_H1.parquet"})
    fmls = [[4, 80, 61, 104], [31, 114, 93, 89], [8, 79, 8, 3]]
    rr = env.eval_formulas(fmls, collect_res=True)
    assert rr["rewards"].shape == (3,)
    assert rr["vals"].shape == (3,)
    assert rr["vol_ok"].shape == (3,)
    assert len(rr["res_list"]) == 3
    assert all(r is None or r.shape[1] == env.bars_train for r in rr["res_list"])
    # 覆盖诊断（格级口径汇总）必须随评估返回（status ok 的候选才带 vol_info）
    cs = rr["cov_stats"]
    assert cs is not None and cs["mode"] == "grid"
    assert 0.0 <= cs["coverage_rate"] <= 1.0
    assert 1 <= cs["n"] <= 3
    assert cs["eff_cells_median"] is not None


# ── 端到端 1 步训练(2 小市场) ─────────────────────────────────────────────
def test_multimarket_engine_one_step(monkeypatch):
    from model_core.multimarket import MultiMarketEngine
    from data_pipeline.market_registry import load_registry

    monkeypatch.setattr(ModelConfig, "BATCH_SIZE", 4)
    monkeypatch.setattr(ModelConfig, "MAX_FORMULA_LEN", 4)
    monkeypatch.setattr(ModelConfig, "MARKET_BATCH_SIZE", 2)
    monkeypatch.setattr(ModelConfig, "MARKET_CALIB_EVERY", 1)   # 每步 calib
    monkeypatch.setattr(ModelConfig, "MARKET_MIN_SHARPE", -100.0)
    monkeypatch.setattr(ModelConfig, "TRAIN_STEPS", 1)

    reg = load_registry()
    all_m = ([m for pol in reg["policies"].values()
              for m in pol.get("markets", [])] + reg.get("oos_review", []))
    specs = [m for m in all_m if m["symbol"] in ("XAUUSD", "EURUSD")]
    eng = MultiMarketEngine(policy="crypto", seed=42, markets=specs,
                            run_tag="utest")
    eng.train(0, 1)
    h = eng.training_history
    assert h["step"] == [0]
    assert h["is_calib"] == [True]
    assert len(h["val_score"]) == 1
    assert eng._suppress_deploy is True
    # 聚合后的奖励应为标量均值（非 NaN）
    assert h["avg_reward"][0] == h["avg_reward"][0]  # not NaN
    # 覆盖健康记录已随步写入（格级汇总，与步对齐）
    assert len(h["cov_rate"]) == 1 and len(h["cov_detail"]) == 1
    assert h["cov_detail"][0] is not None  # step 0 是 calib 步


# ── Phase-1 部署语义（2026-09-05 钉死）──────────────────────────────────
def test_cs_token_ids_are_derived():
    """CS_* 词元 id 必须能由词表名推导（mask 用它屏蔽采样）。"""
    from model_core.multimarket import MultiMarketEngine
    from model_core.vocab import FORMULA_VOCAB

    names = FORMULA_VOCAB.token_names
    ids = {i for i, n in enumerate(names)
           if n in MultiMarketEngine._CS_NAMES}
    assert ids == {109, 110, 111}, ids          # CS_RANK/CS_SCALE/CS_NEUTRALIZE
    assert all(i >= FORMULA_VOCAB.operator_offset for i in ids)


def test_mask_cs_tokens_forbids_sampling():
    """采样 logits 中 CS_* 词元被置 -inf → 采样永不产出横截面词元。"""
    import torch

    from model_core.multimarket import MultiMarketEngine

    eng = MultiMarketEngine.__new__(MultiMarketEngine)
    eng._cs_token_ids = [109, 110, 111]
    lg = torch.zeros(64, 127)                    # 64 条公式 × 词表
    masked = eng._mask_cs_tokens(lg)
    assert torch.isinf(masked[:, 109]).all()
    assert torch.isinf(masked[:, 110]).all()
    assert torch.isinf(masked[:, 111]).all()
    d = torch.distributions.Categorical(logits=masked[0])   # 单行：logits [127]
    assert set(d.sample((5000,)).tolist()) & {109, 110, 111} == set()


def test_prune_checkpoints_keeps_recent_milestone_and_other_runs(monkeypatch, tmp_path):
    """checkpoint 治理：保留最近 3 + 千步里程碑；不误删其他 run/单市场文件。"""
    from model_core import engine as engine_mod
    from model_core.multimarket import MultiMarketEngine

    names = ["ckpt_MULTI_crypto_t_step_0020.pt", "ckpt_MULTI_crypto_t_step_0040.pt",
             "ckpt_MULTI_crypto_t_step_0060.pt", "ckpt_MULTI_crypto_t_step_0080.pt",
             "ckpt_MULTI_crypto_t_step_0100.pt", "ckpt_MULTI_crypto_t_step_1000.pt",
             "ckpt_MULTI_crypto_other_step_0020.pt", "ckpt_BTCUSDT_step_0200.pt"]
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    monkeypatch.setattr(engine_mod, "_CHECKPOINT_DIR", tmp_path)

    eng = MultiMarketEngine.__new__(MultiMarketEngine)
    eng.policy = "crypto"
    eng.run_tag = "t"
    eng._prune_checkpoints(end_step=150)

    left = sorted(p.name for p in tmp_path.iterdir())
    # 本 run：最近 3 个 = 0080/0100/1000（1000 同时是里程碑）；0020/0040/0060 删
    assert "ckpt_MULTI_crypto_t_step_0020.pt" not in left
    assert "ckpt_MULTI_crypto_t_step_0040.pt" not in left
    assert "ckpt_MULTI_crypto_t_step_0060.pt" not in left
    assert "ckpt_MULTI_crypto_t_step_0080.pt" in left
    assert "ckpt_MULTI_crypto_t_step_0100.pt" in left
    assert "ckpt_MULTI_crypto_t_step_1000.pt" in left
    # 其他 run / 单市场文件不碰
    assert "ckpt_MULTI_crypto_other_step_0020.pt" in left
    assert "ckpt_BTCUSDT_step_0200.pt" in left


def test_save_training_history_live_throttle(monkeypatch, tmp_path):
    """实时曲线 JSON 节流：首步写、中间整倍数才写、force 恒写（原子写保留）。"""
    import json as _json

    from model_core import engine as engine_mod
    from model_core.config import ModelConfig

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ModelConfig, "HISTORY_LIVE_EVERY_STEPS", 50)
    eng = engine_mod.AlphaEngine.__new__(engine_mod.AlphaEngine)
    eng.target_symbol = "TEST"
    eng.run_tag = ""
    eng.training_history = {"step": []}
    live = tmp_path / "training_history_TEST.json"

    def snap():
        eng.training_history["step"].append(len(eng.training_history["step"]) + 1)

    snap()
    eng._save_training_history_live()          # n==1 → 首步恒写
    assert _json.loads(live.read_text())["step"] == [1]
    for _ in range(4):
        snap()
        eng._save_training_history_live()      # n=2..5 非整倍数 → 跳过
    assert _json.loads(live.read_text())["step"] == [1]
    snap()
    eng._save_training_history_live(force=True)  # force → 终局/收尾恒写
    assert len(_json.loads(live.read_text())["step"]) == 6


# ── 终局闸门:部署范围口径（gate follows deployment scope, OOS report-only）──
class _StubEnv:
    """finalize 用的最小环境替身:只提供 name/symbol/holdout/coverage。"""

    def __init__(self, name, cov_ok=True, ho_adj=1.5, sharpe=3.0):
        self.name = name
        self.symbol = name
        self._cov_ok = cov_ok
        self._ho = {"ho_adj": ho_adj, "sharpe": sharpe,
                    "sortino": 4.0, "total_return_pct": 5.0}

    def holdout(self, fml):
        return dict(self._ho)

    def coverage(self, fml):
        return (self._cov_ok, {"grid": True, "cells": "stub"})


def _finalize_engine(oos_oks=(False, True), train_ok=True):
    from model_core.multimarket import MultiMarketEngine
    eng = MultiMarketEngine.__new__(MultiMarketEngine)
    eng.envs = [_StubEnv("BTCUSDT_H1", cov_ok=train_ok)]
    eng.oos_legs = [_StubEnv(f"OOS{i}", cov_ok=ok) for i, ok in enumerate(oos_oks)]
    eng.target_symbol = "MULTI_crypto"
    eng.policy = "crypto"
    eng.seed = 42
    eng.champion_outcome = None
    eng._top_finalists = lambda k: [{"fml": [1, 2, 3], "val": 1.7, "step": 1}]
    eng._deploy_market = lambda env, champ: f"strategies/{env.symbol}.json"
    return eng


def test_finalize_gate_follows_deployment_scope(monkeypatch):
    """部署腿(BTC)全过 + OOS 腿不过 → 仍部署;跨域标记 partial、report-only 不 veto。"""
    from model_core import multimarket as mm
    eng = _finalize_engine(oos_oks=(False, True))
    monkeypatch.setattr(mm, "_record_champion_event", lambda d: None)
    out = eng._finalize_champion_multi()
    assert out["action"] == "deploy"
    assert out["deployment_eligible"] is True
    assert out["deployment_scope"] == ["BTCUSDT_H1"]
    assert out["cross_domain_validation"] == "partial"
    assert out["cross_domain"]["OOS0"]["coverage_ok"] is False
    assert out["cross_domain"]["OOS1"]["coverage_ok"] is True
    assert out["save_path"] == ["strategies/BTCUSDT_H1.json"]


def test_finalize_gate_rejects_on_deploy_leg_failure(monkeypatch):
    """部署腿(BTC)覆盖不过 → 拒绝;OOS 结果仍记录但不进入 reasons。"""
    from model_core import multimarket as mm
    eng = _finalize_engine(oos_oks=(True, True), train_ok=False)
    monkeypatch.setattr(mm, "_record_champion_event", lambda d: None)
    out = eng._finalize_champion_multi()
    assert out["action"] == "reject_restore"
    assert out["deployment_eligible"] is False
    assert any("vol 覆盖未过" in r for r in out["reasons"])
    assert all("OOS" not in r for r in out["reasons"])
    assert out["cross_domain_validation"] == "full"


def test_finalize_gate_cross_domain_full_and_none(monkeypatch):
    from model_core import multimarket as mm
    monkeypatch.setattr(mm, "_record_champion_event", lambda d: None)
    eng = _finalize_engine(oos_oks=(True, True))
    assert eng._finalize_champion_multi()["cross_domain_validation"] == "full"
    eng = _finalize_engine(oos_oks=(False, False))
    assert eng._finalize_champion_multi()["cross_domain_validation"] == "none"


# ── _deploy_market 文件名：timeframe 限定（best_{name}.json）防跨口径覆盖 ──
class _DeployStubEnv:
    """_deploy_market 的真实路径用 env：name=符号_周期、symbol=裸符号。"""

    def __init__(self, name, symbol, tf="H1", file_path="data/training/BTCUSDT_H1.parquet"):
        self.name = name
        self.symbol = symbol
        self.timeframe = tf
        self.bars_train = 78000
        self.bars_full = 78500        # bars_train + holdout_bars（整份数据窗口）
        self.holdout_bars = 500

        class _Mgr:
            def __init__(self, fp):
                self.file_path = fp

        self.mgr = _Mgr(file_path)


def test_deploy_market_filename_is_timeframe_qualified(tmp_path, monkeypatch):
    """H1 冠军部署 → best_BTCUSDT_H1.json（不是 best_BTCUSDT.json，避免压掉 M5 制品）。"""
    import json as _json
    from model_core import multimarket as mm

    env = _DeployStubEnv("BTCUSDT_H1", "BTCUSDT")
    eng = mm.MultiMarketEngine.__new__(mm.MultiMarketEngine)
    eng.envs = [env]
    eng.policy = "crypto"
    eng.seed = 43
    eng._decode_formula = lambda f: "AMIHUD_ILLIQ -> SQRT"

    def _fake_path(name):
        assert name == "BTCUSDT_H1", f"期望 env.name 限定文件名，实际 {name!r}"
        return str(tmp_path / f"best_{name}.json")

    monkeypatch.setattr(mm, "_strategy_file_for_symbol", _fake_path)
    # 旧文件残留过期的 train_range（in-sample 根数口径 + tail）——部署必须
    # 用本 env 覆盖，不得随写随带（这正是 2026-09-06 E2E 抓到的溯源错配）。
    stale = {"symbol": "BTCUSDT", "formula": [1], "best_score": 1.0,
             "train_range": {"data_file": "x.parquet", "n_bars": 78000,
                             "mode": "tail", "holdout_bars": 500}}
    (tmp_path / "best_BTCUSDT_H1.json").write_text(
        _json.dumps(stale), encoding="utf-8")
    champ = {"fml": [41, 125], "val": 1.789,
             "per_market": {"BTCUSDT_H1": {"sharpe": 4.15, "sortino": 5.87,
                                           "ho_adj": 1.77, "total_return_pct": 6.7}}}
    out = eng._deploy_market(env, champ)
    assert out == str(tmp_path / "best_BTCUSDT_H1.json")
    data = _json.loads((tmp_path / "best_BTCUSDT_H1.json").read_text(encoding="utf-8"))
    assert data["symbol"] == "BTCUSDT"           # 裸符号仍是交易对标识
    assert data["timeframe"] == "H1"             # 周期随制品自描述（UI/溯源不靠猜）
    assert data["data_file"].endswith("BTCUSDT_H1.parquet")
    assert data["deployment_scope"] == ["BTCUSDT_H1"]
    # 溯源以本 env 为准（覆盖旧残留），不随写随带过期 train_range。
    # n_bars = 整份数据窗口长（bars_full = 训练 + holdout 尾），与
    # web/oos_provenance.py 读取口径一致（窗口尾 holdout_bars 根为 holdout）；
    # 不是 in-sample 根数 bars_train。
    assert data["train_range"]["data_file"].endswith("BTCUSDT_H1.parquet")
    assert data["train_range"]["n_bars"] == 78500
    assert data["train_range"]["mode"] == "full"
