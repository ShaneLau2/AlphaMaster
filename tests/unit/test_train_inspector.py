"""内置训练巡检（web/train_inspector）单测：解析 + 停滞/塌缩/冠军对比诊断。"""
from __future__ import annotations

import json

import pytest

from web.train_inspector import (
    diagnose,
    inspect_log_tail,
    parse_distribution,
    parse_progress,
    read_champion_score,
)

S1 = "[829/9000] 新公式=144 精英=48 | 有效=192 无效=0 常数=0 | 奖励=-0.252 验证=0.871 | IC=0.0046 | 熵=1.623(系数=0.381) | 最优=2.982 停滞=465 精英池=60 重启=6"
D1 = "   分布: 初始熵=0.179 KL均匀=4.665 KL上步=0.0005 最高概率=0.980 前五概率=0.983 有效词汇=1.20 标准差=0.0866 | 本批: 唯一符号=45/127 唯一公式=144/144 多样性=1.00"
S2 = "[120/9000] 新公式=144 精英=48 | 有效=190 无效=2 常数=1 | 奖励=0.88 验证=1.41 | IC=0.0110 | 熵=1.9(系数=0.45) | 最优=1.4 停滞=8 精英池=60 重启=0"
D2 = "   分布: 初始熵=0.55 KL均匀=3.1 KL上步=0.09 最高概率=0.31 前五概率=0.62 有效词汇=18.2 标准差=0.12 | 本批: 唯一符号=88/127 唯一公式=144/144 多样性=0.97"
S3 = "[1606/9000] 市场=[BTCUSDT_H1] 新公式=144 精英=48 | 有效=192 无效=0 常数=0 | 奖励=1.602 验证=1.368 IC=0.0182 熵=0.935 | 最优=1.789 停滞=775 精英池=60 重启=16 CALIB"
D3 = "   市场奖励: BTCUSDT_H1=1.60 | 分布: 初始熵=1.573 KL均匀=3.271 KL上步=0.2355 唯一公式=113/144"


def test_parse_progress_fields():
    p = parse_progress(S1)
    assert p is not None
    assert p["step"] == 829 and p["total"] == 9000
    assert p["best"] == pytest.approx(2.982)
    assert p["stall"] == 465 and p["restarts"] == 6
    assert p["entropy"] == pytest.approx(1.623)
    assert p["ic"] == pytest.approx(0.0046)
    assert p["val"] == pytest.approx(0.871)


def test_parse_distribution_fields():
    d = parse_distribution(D1)
    assert d is not None
    assert d["eff_vocab"] == pytest.approx(1.20)
    assert d["top_p"] == pytest.approx(0.980)
    assert d["diversity"] == pytest.approx(1.00)


def test_parse_multimarket_format():
    # multimarket 引擎日志：市场前缀、奖励/验证/IC/熵 同段（无 系数 后缀）
    p = parse_progress(S3)
    assert p is not None
    assert p["step"] == 1606 and p["total"] == 9000
    assert p["best"] == pytest.approx(1.789)
    assert p["stall"] == 775 and p["restarts"] == 16
    assert p["entropy"] == pytest.approx(0.935)
    assert p["ic"] == pytest.approx(0.0182)
    assert p["val"] == pytest.approx(1.368)
    assert "ent_coef" not in p  # 多市场格式无 系数 后缀

    d = parse_distribution(D3)
    assert d is not None
    assert d["init_entropy"] == pytest.approx(1.573)
    assert d["kl_prev"] == pytest.approx(0.2355)
    assert "eff_vocab" not in d and "diversity" not in d  # 多市场格式缺省字段


def test_multimarket_plateau_is_danger():
    rows = [_row(S3, D3)]
    e = diagnose(rows, champion_score=2.9819, symbol="BTCUSDT_H1")
    assert e["level"] == "danger"
    assert "停止" in e["title"]


def test_parse_rejects_unrelated_line():
    assert parse_progress("启动训练 … loading data") is None
    assert parse_distribution("  最优=1.0") is None


def _row(s: str, d: str) -> dict:
    r = parse_progress(s)
    r["_dist"] = parse_distribution(d)
    return r


def test_plateau_collapse_is_danger():
    rows = [_row(S1, D1)]
    e = diagnose(rows, champion_score=2.9819, symbol="BTCUSDT")
    assert e["level"] == "danger"
    assert "停止" in e["title"]
    texts = " ".join(c["text"] for c in e["checks"])
    assert "分布塌缩" in texts
    assert "持平/同公式" in texts


def test_healthy_early_run_is_ok():
    rows = [_row(S2, D2)]
    e = diagnose(rows, champion_score=2.9, symbol="BTCUSDT")
    assert e["level"] in ("ok", "warn")  # 落后冠军属 info，不升级
    assert not any(c["level"] == "danger" for c in e["checks"])


def test_beats_champion_green():
    rows = [_row(S2, D2)]
    rows[-1]["best"] = 3.02
    e = diagnose(rows, champion_score=2.9, symbol="BTCUSDT")
    assert any("已超过线上冠军" in c["text"] for c in e["checks"])


def test_empty_rows_info():
    e = diagnose([], champion_score=None)
    assert e["level"] == "info"


def test_inspect_log_tail_pairs_distribution():
    e = inspect_log_tail([S1, D1], champion_score=2.9819, symbol="BTCUSDT")
    assert e is not None
    assert e["metrics"]["step"] == 829
    assert e["metrics"]["eff_vocab"] == pytest.approx(1.20)


def test_read_champion_score(tmp_path):
    (tmp_path / "best_BTCUSDT.json").write_text(
        json.dumps({"best_score": 2.9818}), encoding="utf-8"
    )
    assert read_champion_score("BTCUSDT", tmp_path) == pytest.approx(2.9818)
    assert read_champion_score("ETHUSDT", tmp_path) is None
    (tmp_path / "best_XXX.json").write_text("broken", encoding="utf-8")
    assert read_champion_score("XXX", tmp_path) is None
