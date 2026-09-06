"""gonogo_monitor.py — 每 30 分钟把 ETH 训练 go/no-go 判定追加到日志。

供 launchd(com.alphamaster.gonogo.plist,StartInterval=1800)调度。
日志: logs/gonogo.log(逐行 JSON)。不做决策动作,只留证据;人类/后续据此行动。
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_gonogo import evaluate  # noqa: E402


def main() -> int:
    res = evaluate("ETHUSDT", target_step=1000)
    line = json.dumps(
        {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            **res,
        },
        ensure_ascii=False,
    )
    log = ROOT / "logs" / "gonogo.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")
    print(line, flush=True)
    return int(res["exit"])


if __name__ == "__main__":
    sys.exit(main())