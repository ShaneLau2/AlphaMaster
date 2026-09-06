"""M2: MultiMarketEngine —— 共享 policy 多市场训练器（Mode C）。

设计（docs/design_modeC_multi_market.md v2）：
- 一份 AlphaGPT policy 权重；训练腿来自 data/market_registry.json（role=train，
  min_bars 门槛；不足的腿自动降级 eval-only/OOS）。
- 普通步：按 P ∝ N^alpha 加权随机采样 MARKET_BATCH_SIZE 个市场（N=训练 bar 数，
  与数据量脱钩）；每 MARKET_CALIB_EVERY 步全市场 calibration（冠军晋升只在
  calibration 步——那时每条公式在所有市场上的分数完整、无采样噪声）。
- 奖励聚合：每市场年化 Sharpe → MARKET_GROUP_MEAN=True 时按 group 等权
  （组内 mean 再组间 mean，防相关 ticker 重复计票）；min 门槛
  （任一市场聚合分 < MARKET_MIN_SHARPE）否决冠军晋升，不改 policy 梯度。
- 终局（M2b）：冠军须在【每个】训练腿的 holdout 生产口径回测 + vol 覆盖
  以及全部 OOS 复核腿的 vol 覆盖上通过，才逐 best_{sym}_{tf}.json 部署
  （2026-09-06 起按 env.name=symbol+timeframe 限定文件名，防同交易对不同
  周期的冠军互覆；见 _deploy_market）；任一腿失败 → 拒绝部署并记录
  champion_history（保留旧冠军）。

与单市场 AlphaEngine 的关系：继承 model/sampler/opt/elite/factor/重启/检查点，
仅替换 train() 的 Part C（评估 → 多市场批量聚合）与终局（_finalize_champion_multi）。
单市场路径零改动；本类只被 scripts/train_multi.py 使用。
"""
from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import random
import sys
import time
from pathlib import Path

import torch
from torch.distributions import Categorical
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.market_env import MarketEnv  # noqa: E402
from data_pipeline.market_registry import (  # noqa: E402
    load_registry,
    oos_review_markets,
    train_markets,
)
from model_core.config import ModelConfig  # noqa: E402
from model_core.engine import (  # noqa: E402
    AlphaEngine,
    _record_champion_event,
    _strategy_file_for_symbol,
)
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION  # noqa: E402
from strategy_manager.signal import compute_target_positions_stateless  # noqa: E402


class MultiMarketEngine(AlphaEngine):
    """多市场共享 policy 训练器。

    用法（scripts/train_multi.py）：
        eng = MultiMarketEngine(policy="crypto", seed=42, run_tag="exp1")
        eng.train(0, steps)

    部署语义（Phase-1 钉死，2026-09-05）：
      - 每训练腿按单品种 [N=1] 评估 → CS_*（CS_RANK/CS_SCALE/CS_NEUTRALIZE）
        在该上下文是恒等退化（见 model_core/ops.py），既无意义又误导部署。
      -        因此本引擎采样时屏蔽 CS_* 词元，保证产出的共享公式是诚实的
        per-symbol 时序公式 → best_{sym}_{tf}.json 可独立部署（文件名含
        timeframe）；真正的横截面语义留给 Phase-3 Meta policy（等长 cohort
        + context 元数据）。
      - 加载含 CS 词元的旧口径 checkpoint 时，自动从精英池剔除并告警。
    """

    # 横截面词元（N=1 下恒等退化，多市场 Phase-1 禁用于公式）
    _CS_NAMES = frozenset({"CS_RANK", "CS_SCALE", "CS_NEUTRALIZE"})

    def __init__(self, policy: str | None = None, seed: int | None = None,
                 markets: list[dict] | None = None,
                 registry: str | None = None,
                 n_folds: int = 5, run_tag: str = "",
                 suppress_deploy: bool | None = None) -> None:
        self.policy = policy or str(ModelConfig.MARKET_POLICY)
        reg = load_registry(registry)
        specs = train_markets(reg, self.policy) if markets is None else markets
        if not specs:
            raise ValueError(f"policy '{self.policy}' 无训练腿")

        envs = [MarketEnv(s, seed=seed) for s in specs]
        self.envs: list[MarketEnv] = [e for e in envs if e.trainable]
        self.eval_only: list[MarketEnv] = [e for e in envs if not e.trainable]
        if not self.envs:
            names = ", ".join(e.name for e in envs)
            raise ValueError(
                f"无达标训练腿（min_bars={ModelConfig.MARKET_MIN_BARS}）：{names}")
        for e in self.eval_only:
            print(f"[multimarket] {e.name} bar={e.bars_train} 低于训练门槛，"
                  f"降级 eval-only（可作 OOS 复核）")

        # 宿主引擎：挂第一个训练腿的数据（model/sampler/opt/检查点全继承）
        host = self.envs[0]
        super().__init__(data_manager=host.mgr,
                         target_symbol=f"MULTI_{self.policy}",
                         seed=seed, n_folds=n_folds)

        self.registry = reg
        self.oos_legs: list[MarketEnv] = []
        for m in oos_review_markets(reg):
            try:
                self.oos_legs.append(MarketEnv(m, seed=seed))
            except Exception as exc:  # noqa: BLE001 OOS 腿失败不阻断训练
                print(f"[multimarket] OOS 腿 {m.get('symbol')} 加载失败，跳过: {exc}")
        # 去重：训练腿中已有的标的不再重复复核
        train_names = {e.name for e in self.envs}
        self.oos_legs = [e for e in self.oos_legs if e.name not in train_names]

        # √N 采样权重（P ∝ N^alpha）
        ns = torch.tensor([float(e.bars_train) for e in self.envs])
        ws = ns ** float(ModelConfig.MARKET_SAMPLE_ALPHA)
        self._market_weights = (ws / ws.sum()).tolist()

        self.run_tag = run_tag
        # 实验 run 不部署；生产 run（run_tag=""）由 _finalize_champion_multi 决定
        self._suppress_deploy = (suppress_deploy if suppress_deploy is not None
                                 else bool(run_tag))
        # CS 词元 id（采样屏蔽用）
        _names = FORMULA_VOCAB.token_names
        self._cs_token_ids: list[int] = [
            i for i, n in enumerate(_names) if n in self._CS_NAMES]
        if self._cs_token_ids:
            print(f"[multimarket] Phase-1 部署语义: 采样屏蔽 CS_* 词元 "
                  f"{sorted(self._cs_token_ids)} (N=1 恒等退化，避免误导部署)")
        if getattr(ModelConfig, "ACC_CRITIC_GAE", False):
            print("[multimarket] 警告: ACC_CRITIC_GAE 未适配多市场，已忽略（走 EMA baseline）")

    # ── 市场采样与奖励聚合 ───────────────────────────────────────────────
    def _sample_market_batch(self, step: int) -> list[MarketEnv]:
        if step % ModelConfig.MARKET_CALIB_EVERY == 0:
            return list(self.envs)                      # 全市场 calibration
        k = min(int(ModelConfig.MARKET_BATCH_SIZE), len(self.envs))
        if k <= 0:
            return list(self.envs)
        return random.choices(self.envs, weights=self._market_weights, k=k)

    def _aggregate(self, per_env: list[dict], tot: int
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """per_env: [{env, rewards, vals, vol_ok, status}]（各 [tot] 长度）
        返回 (agg_reward, agg_val, min_fail)：
          agg_reward — REINFORCE 信号（训练窗年化 Sharpe 的组级等权聚合）
          agg_val    — 晋升信号（验证折分数聚合）
          min_fail   — 任一市场（有效评估）聚合分 < MARKET_MIN_SHARPE
        """
        agg_r = torch.zeros(tot, device=ModelConfig.DEVICE)
        agg_v = torch.zeros(tot, device=ModelConfig.DEVICE)
        min_fail = torch.zeros(tot, dtype=torch.bool, device=ModelConfig.DEVICE)
        if ModelConfig.MARKET_GROUP_MEAN:
            groups: dict[str, list[dict]] = {}
            for rr in per_env:
                groups.setdefault(rr["env"].group, []).append(rr)
            for g, rrs in groups.items():
                gr = torch.stack([rr["rewards"] for rr in rrs]).mean(dim=0)
                gv = torch.stack([rr["vals"] for rr in rrs]).mean(dim=0)
                agg_r += gr / len(groups)
                agg_v += gv / len(groups)
                for rr in rrs:
                    ok = torch.tensor([s != "error" for s in rr["status"]],
                                      dtype=torch.bool,
                                      device=ModelConfig.DEVICE)
                    min_fail |= ok & (rr["vals"] < ModelConfig.MARKET_MIN_SHARPE)
        else:
            agg_r = torch.stack([rr["rewards"] for rr in per_env]).mean(dim=0)
            agg_v = torch.stack([rr["vals"] for rr in per_env]).mean(dim=0)
            for rr in per_env:
                ok = torch.tensor([s != "error" for s in rr["status"]],
                                  dtype=torch.bool,
                                  device=ModelConfig.DEVICE)
                min_fail |= ok & (rr["vals"] < ModelConfig.MARKET_MIN_SHARPE)
        return agg_r, agg_v, min_fail

    # ── CS 词元屏蔽与 checkpoint 治理 ──────────────────────────────────
    def _mask_cs_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """采样时屏蔽 CS_* 词元（Phase-1：多市场共享公式禁止横截面词元）。"""
        if not self._cs_token_ids:
            return logits
        lg = logits.clone()
        lg[:, self._cs_token_ids] = float("-inf")
        return lg

    def load_checkpoint(self, path: str) -> int:
        """续训加载；顺带清理旧口径 checkpoint 中残留的 CS_* 公式（N=1 惰性）。"""
        completed = super().load_checkpoint(path)
        if self._cs_token_ids:
            cs = set(self._cs_token_ids)
            before = len(self._elite_pool)
            self._elite_pool = [e for e in self._elite_pool
                                if not (cs & set(e[2]))]
            if len(self._elite_pool) != before:
                print(f"[multimarket] 续训清理: 剔除 {before - len(self._elite_pool)} 条"
                      f"含 CS_* 的陈旧精英公式（N=1 惰性，已禁用于多市场）")
            if self.best_formula and (cs & set(self.best_formula)):
                print("[multimarket] 警告: checkpoint 的 best_formula 含 CS_* 词元"
                      "（旧 tier/N=1 口径产物），其评分不含真实截面语义；"
                      "建议从该步之后由格级+屏蔽口径重新晋升冠军。")
        return completed

    def _prune_checkpoints(self, end_step: int) -> None:
        """控制磁盘：本 run 保留最近 3 个 + 千步里程碑 + 终点，其余删除。

        9000 步按每 20 步存 1 个会到 450 个文件（~2.2GB）；此策略把常驻
        文件压到 ~10 个（5MB/个）。里程碑步（step % 1000 == 0）保留作
        人工审视点（如 3000/6000/9000）。
        """
        import re as _re

        from model_core import engine as _engine_mod
        d = _engine_mod._CHECKPOINT_DIR
        sym_tag = f"_MULTI_{self.policy}"
        tag = self._run_tag_suffix()
        prefix = f"ckpt{sym_tag}{tag}_step_"
        pat = _re.compile(_re.escape(prefix) + r"(\d+)\.pt$")
        found: list[tuple[int, pathlib.Path]] = []
        if d.is_dir():
            for p in d.iterdir():
                m = pat.match(p.name)
                if m:
                    found.append((int(m.group(1)), p))
        if len(found) <= 3:
            return
        found.sort(key=lambda t: t[0], reverse=True)
        keep: set[int] = {st for st, _ in found[:3]}
        for st, _ in found:
            if st % 1000 == 0 or st == end_step:
                keep.add(st)
        for st, p in found:
            if st not in keep:
                try:
                    p.unlink()
                except OSError:
                    pass

    # ── 主训练循环（替代 base.train()：Part A/B/D/E/G 同构，Part C 多市场）──
    def train(self, start_step: int = 0, end_step: int | None = None,
              migration_hook=None, verbose_header: bool = True):
        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS
        if verbose_header:
            print(f"[multimarket] 共享 policy 训练: policy={self.policy} "
                  f"训练腿={[e.name for e in self.envs]} "
                  f"OOS复核腿={[e.name for e in self.oos_legs]}")
            print(f"   采样: batch={ModelConfig.MARKET_BATCH_SIZE} "
                  f"α={ModelConfig.MARKET_SAMPLE_ALPHA} "
                  f"calib每{ModelConfig.MARKET_CALIB_EVERY}步 | "
                  f"聚合={'组级等权' if ModelConfig.MARKET_GROUP_MEAN else '平权mean'} "
                  f"min门槛={ModelConfig.MARKET_MIN_SHARPE}")
            for e in self.envs:
                print(f"   {e.name}: bars={e.bars_train} group={e.group} "
                      f"w={self._market_weights[self.envs.index(e)]:.3f}")

        remaining = end_step - start_step
        if remaining <= 0:
            print(f"[multimarket] 起始步 {start_step} 已达目标步 {end_step}。")
            return

        bs = ModelConfig.BATCH_SIZE
        n_elite = max(1, int(bs * ModelConfig.ELITE_REPLAY_FRAC))
        n_new = bs - n_elite
        dev = ModelConfig.DEVICE

        pbar = tqdm(range(start_step, end_step), total=end_step,
                    initial=start_step, disable=not sys.stderr.isatty(),
                    leave=False, mininterval=5.0)
        low_entropy_streak = 0
        prev_init_dist = None
        t_start = time.time()

        for step in pbar:
            # ── Part A: 采样 n_new 条新公式（与单市场同构）────────────────
            inp_new = torch.zeros((n_new, 1), dtype=torch.long, device=dev)
            lp_new, tok_new, ent_new, v_new = [], [], [], []
            sd_new = [0] * n_new
            prev_tokens_new: list[int | None] = [None] * n_new
            infected_chain_new: list[int] = [0] * n_new
            for si in range(ModelConfig.MAX_FORMULA_LEN):
                lg, val_now, _ = self.model(inp_new)
                if val_now is not None:
                    v_new.append(val_now)
                lg = self.sampler.apply_mask_to_logits(
                    lg, sd_new, si, ModelConfig.MAX_FORMULA_LEN,
                    prev_tokens=prev_tokens_new,
                    infected_chain_lens=infected_chain_new)
                lg = self._mask_cs_tokens(lg)
                d = Categorical(logits=lg)
                a = d.sample()
                lp_new.append(d.log_prob(a))
                tok_new.append(a)
                ent_new.append(d.entropy())
                inp_new = torch.cat([inp_new, a.unsqueeze(1)], dim=1)
                for b in range(n_new):
                    sd_new[b] += self.sampler.delta[a[b].item()]
                    prev_tokens_new[b] = a[b].item()
                    infected_chain_new[b] = self.sampler.update_infection(
                        a[b].item(), infected_chain_new[b])
            seqs_new = torch.stack(tok_new, dim=1)

            # ── Part B: 精英回放（与单市场同构）──────────────────────────
            elite_formulas: list[list[int]] = []
            if self._elite_pool and n_elite > 0:
                ps, pt, weights = [], [], []
                for sc, cnt, toks, birth in self._elite_pool:
                    age = max(0, step - birth)
                    decay = 1.0
                    if ModelConfig.ELITE_DECAY:
                        half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                        decay = 0.5 ** (age / half)
                    ps.append(sc); pt.append(toks); weights.append(decay)
                ps_min, ps_max = min(ps), max(ps)
                if ps_max > ps_min:
                    normalized = [(s - ps_min) / (ps_max - ps_min + 1e-8) for s in ps]
                else:
                    normalized = [1.0] * len(ps)
                temp = 0.5
                exp_s = [weights[i] * (2.0 ** (normalized[i] / temp))
                         for i in range(len(ps))]
                exp_sum = sum(exp_s)
                probs = [e / exp_sum for e in exp_s]
                idx_e = random.choices(range(len(self._elite_pool)),
                                       weights=probs, k=n_elite)
                elite_formulas = [pt[i] for i in idx_e]
            else:
                elite_formulas = seqs_new[:n_elite].tolist()

            lp_elite, ent_elite, v_elite = [], [], []
            if elite_formulas:
                ne = len(elite_formulas)
                inp_e = torch.zeros((ne, 1), dtype=torch.long, device=dev)
                sd_e = [0] * ne
                prev_tokens_elite: list[int | None] = [None] * ne
                infected_chain_elite: list[int] = [0] * ne
                tok_e_t = torch.tensor(elite_formulas, dtype=torch.long,
                                       device=dev)
                for si in range(ModelConfig.MAX_FORMULA_LEN):
                    lg_e, val_e, _ = self.model(inp_e)
                    if val_e is not None:
                        v_elite.append(val_e)
                    lg_e = self.sampler.apply_mask_to_logits(
                        lg_e, sd_e, si, ModelConfig.MAX_FORMULA_LEN,
                        prev_tokens=prev_tokens_elite,
                        infected_chain_lens=infected_chain_elite)
                    d_e = Categorical(logits=lg_e)
                    tk = tok_e_t[:, si]
                    lp_elite.append(d_e.log_prob(tk))
                    ent_elite.append(d_e.entropy())
                    inp_e = torch.cat([inp_e, tk.unsqueeze(1)], dim=1)
                    for b in range(ne):
                        sd_e[b] += self.sampler.delta[tk[b].item()]
                        prev_tokens_elite[b] = tk[b].item()
                        infected_chain_elite[b] = self.sampler.update_infection(
                            tk[b].item(), infected_chain_elite[b])

            # ── Part C: 多市场批量评估 + 聚合 ────────────────────────────
            all_fmls = seqs_new.tolist() + elite_formulas
            tot = len(all_fmls)
            mkt_batch = self._sample_market_batch(step)
            factor_pool_snapshot = list(self.factor_pool)
            per_env: list[dict] = []
            for env in mkt_batch:
                rr = env.eval_formulas(all_fmls, collect_res=True)
                rr["env"] = env
                per_env.append(rr)
            agg_r, agg_v, min_fail = self._aggregate(per_env, tot)
            is_calib = (step % ModelConfig.MARKET_CALIB_EVERY == 0)

            # 暴露度检查用因子：取批次首市场的 res（与单市场同语义）
            res_list = per_env[0].get("res_list") if per_env else None

            ok_cnt = none_cnt = const_cnt = 0
            step_max_val = -float('inf'); step_best_f = None
            bic, bis, bsor = [], [], []
            for i, fml in enumerate(all_fmls):
                statuses = [rr["status"][i] for rr in per_env]
                st_any_ok = any(s == "ok" for s in statuses)
                st_all_err = all(s == "error" for s in statuses)
                if st_any_ok:
                    ok_cnt += 1
                    bic.append(per_env[0]["ic_full"][i].item())
                    bis.append(per_env[0]["ic_stab"][i].item())
                    bsor.append(agg_v[i].item())
                elif st_all_err:
                    none_cnt += 1
                    continue
                else:
                    const_cnt += 1
                    continue

                final_val = agg_v[i].item()
                if final_val > step_max_val:
                    step_max_val = final_val; step_best_f = fml

                # vol 覆盖：批次内所有市场都过才算过
                vol_ok = all(bool(rr["vol_ok"][i]) for rr in per_env)
                gate = bool(vol_ok) and not bool(min_fail[i])

                # 精英池：每步更新（batch 聚合分数带采样噪声，可接受）
                self._update_elite_pool(final_val, fml, step)

                # 冠军/入围表：只在 calibration 步（全市场分数完整）晋升
                if gate and is_calib:
                    self._update_finalists(final_val, fml, step)
                    if final_val > self.best_score:
                        train_val = agg_r[i].item()
                        if train_val > 0.5 and final_val < train_val * 0.5:
                            tqdm.write(
                                f"[过拟合跳过 @ 第{step}步] 验证={final_val:.3f} "
                                f"训练={train_val:.3f} | 聚合过拟合")
                        else:
                            exposure = self._batch_exposure(res_list, i)
                            if exposure < 0.05:
                                tqdm.write(
                                    f"[稀疏跳过 @ 第{step}步] 验证={final_val:.3f} "
                                    f"暴露度={exposure:.1%}")
                            else:
                                old_best = self.best_score
                                self.best_score = final_val
                                self.best_formula = fml
                                self._best_snapshot = copy.deepcopy(
                                    self.model.state_dict())
                                self._best_update_step = step
                                self._stagnation_steps = 0
                                self._save_strategy_live()
                                tqdm.write(
                                    f"[!] 新最优 @ 第{step}步(calib): 验证={final_val:.3f} "
                                    f"(原 {old_best:.3f}) | {fml}")
                elif not gate and is_calib and final_val > self.best_score:
                    why = ("vol覆盖未过" if not vol_ok else "min门槛未过")
                    tqdm.write(
                        f"[{why}跳过 @ 第{step}步] 验证={final_val:.3f} 不更新最优")

            # ── Part D: REINFORCE（EMA baseline，与单市场同构）────────────
            batch_mean = agg_r.mean().item()
            batch_std = agg_r.std().clamp(min=0.1)
            if (ModelConfig.REWARD_EMA_BASELINE
                    and self._reward_ema_step >= ModelConfig.REWARD_EMA_WARMUP):
                baseline = self._reward_ema
                adv = (agg_r - baseline) / (batch_std + 1e-5)
            else:
                adv = (agg_r - batch_mean) / (batch_std + 1e-5)
            if self._reward_ema is None:
                self._reward_ema = batch_mean
            else:
                self._reward_ema = (ModelConfig.REWARD_EMA_DECAY * self._reward_ema
                                    + (1.0 - ModelConfig.REWARD_EMA_DECAY) * batch_mean)
            self._reward_ema_step += 1
            adv_new = adv[:n_new]
            adv_elite = adv[n_new:]

            policy_loss = torch.zeros(1, device=dev)
            for ti in range(len(lp_new)):
                policy_loss += (-lp_new[ti] * adv_new).mean()
            if lp_elite and adv_elite.shape[0] > 0:
                for ti in range(len(lp_elite)):
                    lpe = lp_elite[ti]
                    if lpe.shape[0] == adv_elite.shape[0]:
                        policy_loss += (-lpe * adv_elite
                                        * ModelConfig.ELITE_REWARD_SCALE).mean()

            mean_ent_new = torch.stack(ent_new).mean()
            if ent_elite:
                mean_ent_elite = torch.stack(ent_elite).mean()
                mean_ent = (mean_ent_new * n_new + mean_ent_elite * n_elite) / bs
            else:
                mean_ent = mean_ent_new
            ent_val = mean_ent.item()
            ent_coeff = ModelConfig.ENTROPY_COEFF_MAX / (
                (1.0 + ent_val) ** ModelConfig.ENTROPY_COEFF_POWER)
            ent_floor_loss = torch.zeros(1, device=dev)
            if ModelConfig.ENTROPY_FLOOR and ent_val < ModelConfig.ENTROPY_FLOOR_THRESH:
                floor_gap = ModelConfig.ENTROPY_FLOOR_THRESH - ent_val
                ent_floor_loss = ModelConfig.ENTROPY_FLOOR_LAMBDA * torch.tensor(
                    floor_gap, device=dev, dtype=mean_ent.dtype)
            loss = policy_loss - ent_coeff * mean_ent + ent_floor_loss

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            if self.use_lord:
                self.lord_opt.step()

            # ── Part D2: 分布指标 ───────────────────────────────────────
            dst = self._distribution_stats(prev_init_dist)
            prev_init_dist = dst['dist']
            with torch.no_grad():
                uniq_tokens = seqs_new.unique().numel()
                uniq_fmls = torch.unique(seqs_new, dim=0).shape[0]
                fml_div = uniq_fmls / max(1, n_new)

            # ── Part E: 日志与历史 ──────────────────────────────────────
            avg_rew = agg_r.mean().item()
            avg_val = agg_v.mean().item()
            bim = sum(bic) / len(bic) if bic else 0.0
            self._stagnation_steps = step - self._best_update_step
            mkt_desc = ",".join(e.name for e in mkt_batch)
            per_mkt = {e.name: float(rr["rewards"].mean().item())
                       for e, rr in zip(mkt_batch, per_env)}
            tqdm.write(
                f"[{step+1}/{end_step}] 市场=[{mkt_desc}] "
                f"新公式={n_new} 精英={n_elite} | "
                f"有效={ok_cnt} 无效={none_cnt} 常数={const_cnt} | "
                f"奖励={avg_rew:.3f} 验证={avg_val:.3f} IC={bim:.4f} "
                f"熵={ent_val:.3f} | 最优={self.best_score:.3f} "
                f"停滞={self._stagnation_steps} 精英池={len(self._elite_pool)} "
                f"重启={self._restart_count} {('CALIB' if is_calib else '')}"
            )
            tqdm.write(
                f"   市场奖励: {(' '.join(f'{k}={v:.2f}' for k, v in per_mkt.items()))}"
                f" | 分布: 初始熵={dst['entropy']:.3f} KL均匀={dst['kl_uniform']:.3f} "
                f"KL上步={dst['kl_prev']:.4f} 唯一公式={uniq_fmls}/{n_new}"
            )
            pbar.set_postfix({'验证': f"{avg_val:.3f}", '最优': f"{self.best_score:.3f}",
                              '熵': f"{ent_val:.2f}", '停滞': f"{self._stagnation_steps}"})

            # 覆盖健康（格级汇总，来自本轮已评估市场的 cov_stats）
            cov_list = [rr.get("cov_stats") for rr in per_env
                        if rr.get("cov_stats") is not None]
            cov_rate = (round(sum(c["coverage_rate"] for c in cov_list)
                              / len(cov_list), 4) if cov_list else None)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_rew)
            self.training_history['val_score'].append(avg_val)
            self.training_history['best_score'].append(self.best_score)
            self.training_history.setdefault('entropy', []).append(ent_val)
            self.training_history.setdefault('ic_mean', []).append(bim)
            self.training_history.setdefault('elite_pool_size', []).append(
                len(self._elite_pool))
            self.training_history.setdefault('init_entropy', []).append(dst['entropy'])
            self.training_history.setdefault('kl_uniform', []).append(dst['kl_uniform'])
            self.training_history.setdefault('kl_prev', []).append(dst['kl_prev'])
            self.training_history.setdefault('batch_uniq_fmls', []).append(uniq_fmls)
            self.training_history.setdefault('markets', []).append(
                [e.name for e in mkt_batch])
            self.training_history.setdefault('market_rewards', []).append(per_mkt)
            self.training_history.setdefault('is_calib', []).append(is_calib)
            self.training_history.setdefault('cov_rate', []).append(cov_rate)
            self.training_history.setdefault('cov_detail', []).append(
                {rr["env"].name: rr.get("cov_stats") for rr in per_env}
                if is_calib else None)
            # 实时曲线按 HISTORY_LIVE_EVERY_STEPS 节流；run 终点强制落盘
            self._save_training_history_live(force=(step + 1) == end_step)

            if is_calib and cov_list:
                tqdm.write(
                    "  [覆盖 grid] " + " | ".join(
                        f"{rr['env'].name} rate={rr['cov_stats']['coverage_rate']:.2f} "
                        f"ok={rr['cov_stats']['n_ok']}/{rr['cov_stats']['n']} "
                        f"cells_med={rr['cov_stats']['eff_cells_median']} "
                        f"bars_min={rr['cov_stats']['cell_bars_min']}"
                        for rr in per_env if rr.get("cov_stats")))

            if (step + 1) % 20 == 0 or (step + 1) == end_step:
                ckpt = self.save_checkpoint(step + 1)
                tqdm.write(f"[检查点] → {ckpt} (最优={self.best_score:.3f})")
                self._prune_checkpoints(end_step)

            if migration_hook is not None and (step + 1) % ModelConfig.MIGRATION_INTERVAL == 0:
                migration_hook(self, step + 1)

            # ── Part G: 熵坍塌检测与重启（与单市场同构）───────────────────
            if ent_val < ModelConfig.ENTROPY_COLLAPSE_THRESH:
                low_entropy_streak += 1
            else:
                low_entropy_streak = 0
            if low_entropy_streak >= ModelConfig.ENTROPY_COLLAPSE_STEPS:
                self._stagnation_steps = step - self._best_update_step
                stagnation_ratio = self._stagnation_steps / max(1, ModelConfig.STAGNATION_WINDOW)
                base_noise = ModelConfig.RESTART_NOISE
                if ModelConfig.ADAPTIVE_NOISE:
                    raw_noise = (base_noise + ModelConfig.NOISE_BOOST_FACTOR * 0.1
                                 * min(stagnation_ratio, 3.0))
                    noise = max(ModelConfig.NOISE_MIN,
                                min(ModelConfig.NOISE_MAX, raw_noise))
                else:
                    noise = base_noise
                max_r = ModelConfig.MAX_RESTARTS
                if self._restart_count < max_r:
                    improved_since_prev = (
                        self._last_restart_step < 0
                        or self._best_update_step > self._last_restart_step)
                    escalate_full = bool(ModelConfig.RESTART_ESCALATE_FULL) and (
                        self._restart_count > 0 and not improved_since_prev)
                    self._restart_count += 1
                    low_entropy_streak = 0
                    self._last_restart_step = step
                    do_full_reset = (
                        self._restart_count % ModelConfig.FULL_RESET_EVERY == 0
                        or ent_val < 0.3 or escalate_full)
                    if do_full_reset:
                        for layer in self.model.modules():
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                        _why = ("升级:上次重启后无刷新" if escalate_full else
                                ("固定周期 full reset" if
                                 self._restart_count % ModelConfig.FULL_RESET_EVERY == 0
                                 else "深度坍塌 H<0.3"))
                        tqdm.write(f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                                   f"模式=完全重置（{_why}）")
                    elif self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                        with torch.no_grad():
                            for nm, p in self.model.named_parameters():
                                if any(k in nm for k in ModelConfig.PARTIAL_RESET_LAYERS):
                                    p.add_(torch.randn_like(p) * noise)
                        tqdm.write(f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                                   f"模式=部分层 噪声={noise:.4f}")
                    else:
                        with torch.no_grad():
                            for p in self.model.parameters():
                                p.add_(torch.randn_like(p) * noise)
                        tqdm.write(f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                                   f"模式=全参数 噪声={noise:.4f}")
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                else:
                    low_entropy_streak = 0
                    hard_noise = min(ModelConfig.NOISE_MAX, noise * 2.0)
                    if self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                    with torch.no_grad():
                        for p in self.model.parameters():
                            p.add_(torch.randn_like(p) * hard_noise)
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                    tqdm.write(f"[强重启 @ 第{step}步] 已达最大重启次数={max_r} "
                               f"熵={ent_val:.3f} 强噪声={hard_noise:.4f}")

        # ── 训练收尾 ────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        self.training_history["wall_seconds"] = elapsed
        if end_step == ModelConfig.TRAIN_STEPS:
            self._finalize_champion_multi()
        hist_path = (f"training_history_{self.target_symbol}"
                     f"{self._run_tag_suffix()}.json")
        tmp_hist = hist_path + ".tmp"
        with open(tmp_hist, "w", encoding="utf-8") as fp:
            json.dump(self.training_history, fp, ensure_ascii=False)
        os.replace(tmp_hist, hist_path)
        print(f"\n[完成] {self.target_symbol} 多市场训练结束！({elapsed:.1f}s)")
        print(f"  最优验证分数 : {self.best_score:.4f}")
        print(f"  最优公式令牌 : {self.best_formula}")
        print(f"  可读公式     : {self._decode_formula(self.best_formula)}")
        print(f"  重启次数     : {self._restart_count}")

    # ── 暴露度检查（批次首市场因子）──────────────────────────────────────
    def _batch_exposure(self, res_list: list | None, i: int) -> float:
        try:
            if not res_list or res_list[i] is None:
                return 1.0
            pos = compute_target_positions_stateless(res_list[i])
            return float(pos.abs().mean().item())
        except Exception:  # noqa: BLE001 fail-open
            return 1.0

    # ── M2b: 终局闸门 —— 每市场 holdout + 覆盖 + OOS 腿，全过才逐市场部署 ──
    def _finalize_champion_multi(self) -> dict:
        sym = self.target_symbol
        outcome = {"action": "no_champion", "symbol": sym, "save_path": None}
        if getattr(self, "_suppress_deploy", False):
            self.champion_outcome = {"action": "suppressed", "symbol": sym,
                                     "save_path": None}
            return self.champion_outcome

        candidates = self._top_finalists(ModelConfig.FINALIST_TOP_K)
        if not candidates:
            self.champion_outcome = outcome
            return outcome

        # 逐腿 holdout 生产口径指标（与单市场 _finalize_champion 同函数）
        scored: list[dict] = []
        for cand in candidates:
            per_mkt: dict[str, dict] = {}
            for env in self.envs:
                m = env.holdout(cand["fml"])
                if m is None:
                    m = {"error": "holdout 无结果"}
                per_mkt[env.name] = m
            scored.append({"fml": cand["fml"], "val": cand["val"],
                           "step": cand["step"], "per_market": per_mkt})

        def _mkt_ok(m: dict) -> bool:
            if "error" in m:
                return False
            return (float(m.get("ho_adj", -99)) > ModelConfig.HOLDOUT_MIN_SCORE
                    and float(m.get("sharpe", -99)) >= ModelConfig.HOLDOUT_MIN_SHARPE)

        def _rank_key(s: dict):
            sharps = [float(v.get("sharpe", -99))
                      for v in s["per_market"].values() if "error" not in v]
            if not sharps:
                return (-99.0, -99.0)
            return (min(sharps), sum(sharps) / len(sharps))

        scored.sort(key=_rank_key, reverse=True)
        champ = scored[0]
        reasons: list[str] = []

        # 1) 每训练腿 holdout 通过
        for name, m in champ["per_market"].items():
            if not _mkt_ok(m):
                reasons.append(f"{name} holdout 未过 (ho_adj="
                               f"{m.get('ho_adj', 'err')}, sharpe="
                               f"{m.get('sharpe', 'err')})")
        # 2) 每部署腿 vol 覆盖（训练窗，与选优闸门同口径）——闸门跟随部署范围
        #    （deployment_scope = 实际写出 best_{sym}.json 的腿，不是训练/复核全集）
        for env in self.envs:
            ok, info = env.coverage(champ["fml"])
            if not ok:
                reasons.append(f"{env.name} vol 覆盖未过 {info}")
        # 3) OOS 复核腿 → report-only（部署范围外，不 veto；供迁移性/域稳健性报告）
        cross_domain: dict[str, dict] = {}
        for env in self.oos_legs:
            ok, info = env.coverage(champ["fml"])
            cross_domain[env.name] = {"coverage_ok": bool(ok), "coverage": info}
        n_ok = sum(1 for v in cross_domain.values() if v["coverage_ok"])
        n_tot = len(cross_domain)
        cross_domain_validation = (
            "full" if n_tot and n_ok == n_tot else
            "partial" if n_ok > 0 else
            "none" if n_tot else "na")

        scope = [env.name for env in self.envs]
        gate_ok = len(reasons) == 0
        if gate_ok:
            # 逐部署腿写出 best_{sym}.json（P0 纪律：部署范围全过才写部署路径）
            saved = []
            for env in self.envs:
                p = self._deploy_market(env, champ)
                saved.append(str(p))
            outcome = {"action": "deploy", "symbol": sym, "save_path": saved,
                       "formula": champ["fml"],
                       "deployment_scope": scope, "deployment_eligible": True,
                       "cross_domain_validation": cross_domain_validation,
                       "cross_domain": cross_domain,
                       "per_market_holdout": champ["per_market"]}
            _record_champion_event({
                "event": "deploy", "symbol": sym, "source": f"MULTI_{self.policy}",
                "formula": champ["fml"],
                "formula_decoded": self._decode_formula(champ["fml"]),
                "per_market": {k: {kk: vv for kk, vv in v.items()
                                   if kk in ("sharpe", "ho_adj", "sortino")}
                               for k, v in champ["per_market"].items()},
                "gate": "deployment_scope_all_pass",
                "deployment_scope": scope,
                "cross_domain_validation": cross_domain_validation,
            })
            print(f"[多市场冠军闸门] 部署 ✓ 逐市场: {saved}")
        else:
            outcome = {"action": "reject_restore", "symbol": sym,
                       "save_path": None, "reasons": reasons,
                       "formula": champ["fml"],
                       "deployment_scope": scope, "deployment_eligible": False,
                       "cross_domain_validation": cross_domain_validation,
                       "cross_domain": cross_domain}
            _record_champion_event({
                "event": "reject", "symbol": sym, "source": f"MULTI_{self.policy}",
                "formula": champ["fml"],
                "reasons": reasons,
                "deployment_scope": scope,
            })
            print("[多市场冠军闸门] 拒绝部署 ✗ 保留旧冠军：")
            for r in reasons:
                print(f"    - {r}")
        if cross_domain:
            _cd = ", ".join(f"{k}={'✓' if v['coverage_ok'] else '✗'}"
                            for k, v in cross_domain.items())
            print(f"[跨域 OOS 报告] {cross_domain_validation}（report-only，不 veto）: {_cd}")
        self.champion_outcome = outcome
        return outcome

    def _deploy_market(self, env: MarketEnv, champ: dict) -> str:
        """把多市场冠军写入单个市场的部署文件（strategies/best_{sym}_{tf}.json）。

        文件名按 env.name（symbol+timeframe）限定，而不是裸 symbol：同一交易对
        不同 timeframe 的训练（如 BTCUSDT M5 与 H1）产出的是不同冠军，若共用
        best_{symbol}.json 会被 web 的 sync-best（跨 timeframe 取 max best_score）
        互相覆盖，并留下"M5 公式 + H1 train_range"这类溯源错配（2026-09-06 实测
        事故）。H1 部署 → best_BTCUSDT_H1.json；旧 M5 制品继续占用
        best_BTCUSDT.json，互不干扰。
        """
        m = champ["per_market"].get(env.name, {})
        strategy_data = {
            "vocab_version": VOCAB_VERSION,
            "symbol": env.symbol,
            "timeframe": env.timeframe,
            "data_file": str(env.mgr.file_path),
            "formula": champ["fml"],
            "best_score": round(float(champ["val"]), 6),
            "formula_decoded": self._decode_formula(champ["fml"]),
            "deployed_from": f"MULTI_{self.policy}",
            "holdout": {k: m.get(k) for k in ("sharpe", "sortino", "ho_adj",
                                              "total_return_pct")},
            "deployment_scope": [e.name for e in self.envs],
            "seed": self.seed,
            "per_market_holdout": {
                name: {kk: v.get(kk) for kk in ("sharpe", "ho_adj", "sortino")}
                for name, v in champ["per_market"].items()},
        }
        # train_range 溯源：以本训练 env 为准（部署的冠军就是在该 env 上训的），
        # 不用旧文件里可能残留的过期口径——与 engine.py 的语义一致
        # （“部署路径会用引擎本程覆盖为更精确值；live 保存才保旧”）。
        # 若旧文件曾带别的周期/数据口径（如 M5/H1 同 symbol 事故），随写随带会
        # 把过期溯源带到新冠军上。
        # n_bars = 训练窗口总长（含预留 holdout 尾），与 web/oos_provenance.py
        # 读取口径一致：provenance 把窗口尾部 holdout_bars 根当 holdout。
        # 不能用 env.bars_train（= in-sample 根数）——那会把真正的 holdout
        # 后移一个周期，把样本内窗误标成 holdout/新数据。
        # 本训练器(MarketEnv)用整份文件：in-sample=[0..bars_full-hb)，
        # holdout=文件尾部 hb 根 → n_bars = env.bars_full。
        strategy_data["train_range"] = {
            "data_file": str(env.mgr.file_path),
            "n_bars": int(env.bars_full),
            "mode": "full",
            "holdout_bars": int(env.holdout_bars),
        }
        save_path = _strategy_file_for_symbol(env.name)
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = save_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(strategy_data, fp, indent=2, ensure_ascii=False)
        os.replace(tmp, save_path)
        return save_path
