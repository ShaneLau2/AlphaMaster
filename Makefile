# ── AlphaMaster 常用检查入口 ──────────────────────────────────────────────
# 每次发版前的检查清单（release-check）：
#   1. 单测/冒烟/属性测试（tests/unit tests/smoke tests/property）
#   2. 交互延迟回归（tests/perf：×→消失 乐观移除毫秒实测，失败即 FAIL）
#   3. 可选压测（make load：10-20 监控项下 status 轮询/移除延迟/价格刷新预算封顶）
#
# 用法：
#   make check          # 快速全量逻辑回归（不含浏览器 perf）
#   make perf           # ×→消失延迟回归（需要本机 Chrome + playwright；-s 打印实测毫秒）
#   make load           # 实时监控压测（无需浏览器）
#   make release-check  # 发版清单：check + perf

.PHONY: check perf load release-check

check:
	.venv/bin/python -m pytest tests/unit tests/smoke tests/property -q

perf:
	@echo "==> 发版检查清单 #2：×→消失延迟回归（乐观移除毫秒实测）"
	.venv/bin/python -m pytest tests/perf -q -s

load:
	@echo "==> 实时监控压测：10-20 监控项下 status 轮询 / 单卡移除 / 价格刷新预算封顶"
	.venv/bin/python scripts/perf_load_test.py

release-check: check perf
	@echo "✅ 发版检查通过：逻辑回归绿 + ×→消失延迟回归绿"