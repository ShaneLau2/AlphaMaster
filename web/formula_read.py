"""巡检用「最新公式解读」：把训练中的 best-so-far token 公式翻译成人话。

读取顺序（与进度面板一致）：优先训练中的 *.live.json 侧车（best-so-far，
未过闸门），否则回退已部署的 best_{symbol}.json。token 序列是 StackVM 的
后缀式程序（特征叶子 + 算子按 arity 弹栈），据此重建一棵表达式树，输出：

  - decoded:     token 名链（与策略卡的 formula_decoded 同口径）
  - expression:  带括号的中缀表达式（如 TS_MEAN_5(RET20) − TS_MEAN_20(RET20)）
  - parts:       逐 token {id, name, kind, note}
  - summary:     一句话（用了哪些类别的特征 / 哪些算子家族）

本模块不重（lazy import model_core），巡检线程调用不会拖慢训练。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FEATURE_CATEGORY_CN = {
    "trend": "趋势",
    "momentum": "动量",
    "volatility": "波动率",
    "volume": "量能",
    "reversal": "反转",
    "channel": "通道/位置",
    "statistical": "统计分布",
    "cross_sectional": "横截面",
}

# 常见算子的人类可读简写（用于中缀表达式 + note）。未列出的回退到原名。
_OP_GLYPH = {
    "ADD": "+", "SUB": "−", "MUL": "×", "DIV": "÷",
    "NEG": "−()", "ABS": "|·|", "SIGN": "sign",
    "GATE": "if(·>0, a, b)", "IF_GT": "if(a>b, a, b)",
    "JUMP": "zscore跳变", "DECAY": "指数衰减", "DECAY_LINEAR_5": "线性衰减5",
    "TS_DECAY_EXP_5": "指数衰减5", "DELAY1": "滞后1", "DELAY4": "滞后4",
    "DELTA": "一阶差分", "DELTA_5": "5期差分", "MOMENTUM_5": "5期动量",
    "MOMENTUM_10": "10期动量", "WMA": "加权均线", "EMA_5": "EMA5",
    "EMA_20": "EMA20", "MAX3": "3期最大", "SCALE": "缩放",
    "PRODUCT_5": "5期乘积", "SIGNED_POWER_2": "带符号平方",
    "POWER": "带符号平方", "SIGNED_LOG": "带符号对数", "SQRT": "带符号开方",
    "TS_CORR_10": "10期相关", "COVARIANCE_10": "10期协方差",
    "TS_SUM_5": "5期和", "TS_SUM_10": "10期和", "TS_SUM_20": "20期和",
    "TS_ARG_MAX_5": "5期最大值位置", "TS_ARG_MIN_5": "5期最小值位置",
    "MIN": "取小", "MAX": "取大", "WINSORIZE": "缩尾", "CLIP": "截断",
    "SIGMOID": "sigmoid", "TANH_SQUASH": "tanh压扁",
    "CS_RANK": "横截面排名", "CS_SCALE": "横截面缩放", "CS_NEUTRALIZE": "横截面中性化",
}

_OP_FAMILY = {
    "TS_MEAN_": "滚动均值", "TS_STD_": "滚动标准差", "TS_RANK_": "滚动排名",
    "TS_ZSCORE_": "滚动z值", "TS_QUANTILE_": "滚动分位", "TS_SKEW_": "滚动偏度",
    "TS_MIN_": "滚动最小", "TS_MAX_": "滚动最大", "TS_SUM_": "滚动求和",
}


def _op_glyph(name: str) -> str:
    if name in _OP_GLYPH:
        return _OP_GLYPH[name]
    for prefix, cn in _OP_FAMILY.items():
        if name.startswith(prefix):
            n = name[len(prefix):]
            return f"{cn}{n}" if n else cn
    return name


def _load_context() -> dict[str, Any]:
    """返回 {token_names, operator_offset, op_arity_by_token, feature_cat_by_name}。"""
    from model_core.vocab import FORMULA_VOCAB
    from model_core.registry import Registry  # noqa: F401 触发注册表构建
    from model_core.ops import OPERATOR_REGISTRY
    from model_core.features import FEATURE_REGISTRY

    names = FORMULA_VOCAB.token_names
    offset = FORMULA_VOCAB.operator_offset
    op_arity: dict[int, int] = {}
    for i, spec in enumerate(OPERATOR_REGISTRY.operator_specs):
        op_arity[offset + i] = int(spec.arity)
    feat_cat = {spec.name: str(spec.category) for spec in FEATURE_REGISTRY.feature_specs}
    return {
        "token_names": list(names),
        "operator_offset": offset,
        "op_arity_by_token": op_arity,
        "feature_cat_by_name": feat_cat,
    }


def read_formula_file(strategies_dir: Path, symbol: str) -> list[int] | None:
    """读该品种当前最优公式 token；live 侧车优先，其次已部署 best。"""
    for name in (f"best_{symbol}.live.json", f"best_{symbol}.json"):
        p = Path(strategies_dir) / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        fml = data.get("formula")
        if isinstance(fml, list) and fml:
            return [int(t) for t in fml]
    return None


def _build_expression(tokens: list[int], ctx: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """后缀式 → 中缀表达式 + 逐 token parts。返回 (expr, parts)。"""
    names = ctx["token_names"]
    offset = ctx["operator_offset"]
    feat_cat = ctx["feature_cat_by_name"]
    op_arity = ctx["op_arity_by_token"]

    stack: list[str] = []
    parts: list[dict[str, Any]] = []
    for tid in tokens:
        if tid < 0 or tid >= len(names):
            stack.append(f"#{tid}")
            parts.append({"id": tid, "name": f"#{tid}", "kind": "unknown", "note": ""})
            continue
        name = names[tid]
        if tid < offset:  # 特征叶子
            cat = feat_cat.get(name, "")
            cat_cn = _FEATURE_CATEGORY_CN.get(cat, cat or "特征")
            stack.append(name)
            parts.append({"id": tid, "name": name, "kind": "feature",
                          "note": f"{cat_cn}类特征"})
        else:  # 算子：按 arity 弹栈
            arity = int(op_arity.get(tid, 1))
            glyph = _op_glyph(name)
            if arity <= 0 or len(stack) < arity:
                # 无法解析（理论不会发生，防御）——整条表达式退回 token 链
                stack.append(name)
                parts.append({"id": tid, "name": name, "kind": "operator",
                              "arity": arity, "note": ""})
                continue
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            if name == "SUB" and arity == 2:
                expr = f"{operands[0]} − {operands[1]}"
            elif name == "DIV" and arity == 2:
                expr = f"{operands[0]} ÷ {operands[1]}"
            elif name == "GATE" and arity == 3:
                expr = f"门控(cond={operands[0]}, x={operands[1]}, y={operands[2]})"
            elif arity == 1:
                expr = f"{glyph}({operands[0]})"
            else:
                expr = f"{glyph}({' , '.join(operands)})"
            stack.append(expr)
            parts.append({"id": tid, "name": name, "kind": "operator",
                          "arity": arity, "note": glyph})
    expr = stack[0] if stack else "(空)"
    return expr, parts


def _build_summary(tokens: list[int], parts: list[dict[str, Any]]) -> str:
    cats: dict[str, int] = {}
    op_names: list[str] = []
    for p in parts:
        if p["kind"] == "feature":
            note = str(p.get("note") or "")
            cat = note.replace("类特征", "")
            cats[cat] = cats.get(cat, 0) + 1
        elif p["kind"] == "operator":
            op_names.append(str(p.get("note") or p.get("name") or ""))
    cat_s = "、".join(f"{k}×{v}" for k, v in sorted(cats.items(), key=lambda x: -x[1])) if cats else "无特征"
    op_s = "、".join(op_names) if op_names else "无算子"
    return f"共 {len(tokens)} 个 token：用到 {cat_s}；算子组合：{op_s}"


def interpret_formula(tokens: list[int] | None) -> dict[str, Any] | None:
    """把 token 公式解释成 {decoded, expression, parts, summary}；读不到则 None。"""
    if not tokens:
        return None
    try:
        ctx = _load_context()
    except Exception:  # noqa: BLE001 model_core 不可用时优雅降级
        return None
    names = ctx["token_names"]
    decoded = " → ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in tokens)
    expr, parts = _build_expression(tokens, ctx)
    return {
        "decoded": decoded,
        "expression": expr,
        "parts": parts,
        "summary": _build_summary(tokens, parts),
        "token_count": len(tokens),
    }


def reading_for_symbol(strategies_dir: Path, symbol: str) -> dict[str, Any] | None:
    """一站读取：live/best 公式 → 解读 dict（或 None）。"""
    if not symbol:
        return None
    tokens = read_formula_file(Path(strategies_dir), symbol)
    return interpret_formula(tokens)
