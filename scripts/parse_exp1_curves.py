"""parse_exp1_curves.py — 从 E1 运行日志解析两条 arm 的逐步曲线。

输出 results/exp1_curves_parsed.json: {A:{...}, B:{...}}
字段: steps/ent/rew/val/best/stg/rst/ic (与日志行一一对应)
只读。
"""
from __future__ import annotations

import json
import re

LOG = "logs/exp1_run_200_s42.log"
OUT = "results/exp1_curves_parsed.json"

# 实际行序: 奖励 验证 | IC | 熵(系数) | 最优 停滞 | 精英池 | 重启
PAT = re.compile(
    r"^\[(\d+)/(\d+)\].*?奖励=(-?[\d.]+) 验证=(-?[\d.]+).*?"
    r"IC=([\d.eE+-]+).*?熵=([\d.]+)\(系数=([\d.]+)\).*?"
    r"最优=(-?[\d.]+) 停滞=(\d+).*?重启=(\d+)"
)


def parse_block(block: list[str]) -> dict | None:
    steps, ent, rew, val, best, stg, rst, ic = ([] for _ in range(8))
    for l in block:
        m = PAT.match(l.strip())
        if not m:
            continue
        steps.append(int(m.group(1)))
        rew.append(float(m.group(3)))
        val.append(float(m.group(4)))
        ic.append(float(m.group(5)))
        ent.append(float(m.group(6)))
        best.append(float(m.group(8)))
        stg.append(int(m.group(9)))
        rst.append(int(m.group(10)))
    if not steps:
        return None
    return {"steps": steps, "ent": ent, "rew": rew, "val": val,
            "best": best, "stg": stg, "rst": rst, "ic": ic}


def main() -> None:
    lines = open(LOG, encoding="utf-8").read().splitlines()
    seps = [i for i, x in enumerate(lines) if "done ->" in x and "exp1_" in x]
    assert len(seps) >= 2, f"expected 2 arm separators, got {seps}"
    A = parse_block(lines[: seps[0] + 1])
    B = parse_block(lines[seps[0] + 1: seps[1] + 1])
    assert A and B, "parse failed for one arm"
    json.dump({"A": A, "B": B}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"parsed: baseline={len(A['steps'])}步 critic={len(B['steps'])}步 -> {OUT}")
    for tag, d in (("baseline", A), ("critic", B)):
        n = len(d["steps"])
        bmax = max(d["best"])
        first = d["steps"][d["best"].index(bmax)]
        h = max(1, n // 4)
        print(f"  [{tag}] 步数={n} 熵首/末/均={d['ent'][0]:.2f}/{d['ent'][-1]:.2f}/"
              f"{sum(d['ent'])/n:.2f} 熵前{h}均={sum(d['ent'][:h])/h:.2f} "
              f"后{h}均={sum(d['ent'][-h:])/h:.2f}")
        print(f"         reward均={sum(d['rew'])/n:+.3f} val均={sum(d['val'])/n:+.3f} "
              f"best末={d['best'][-1]:.3f} best最高={bmax:.3f}@{first}步 "
              f"重启={d['rst'][-1]} IC均={sum(d['ic'])/n:.4f}")


if __name__ == "__main__":
    main()
