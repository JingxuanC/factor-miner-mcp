#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_feedback.py — factor_feedback 反馈层测试

契约（feedback.py docstring）：
  - 容忍三种结果形状（evaluate/backtest/oos），自动识别
  - 同一种失败 → 同一种诊断与建议（确定性，非 LLM 自由发挥）
  - verdict ∈ success/fixable/rejected；metrics 可跨轮传递做进化追踪
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feedback  # noqa: E402


def _eval_result(ic=0.031, annualized=0.15):
    return {"ok": True, "eval_ok": True,
            "tearsheet": {"ic": {"mean": ic, "ir": 0.4},
                          "long_short": {"1d": {"annualized": annualized}}},
            "monotonicity": {"1d": True}}


def _bt_result(ic=0.031, dedup=False, corr=0.3):
    return {"ok": True, "dedup_dropped": dedup, "sota_broken": [],
            "metrics": {"IC": ic, "Rank IC": ic, "ICIR": 0.4,
                        "1day.excess_return_with_cost.annualized_return": 0.15,
                        "1day.excess_return_with_cost.max_drawdown": -0.1},
            "correlations": {"f": [corr]}, "error": ""}


def _oos_result(decay=0.2):
    return {"ok": True, "decay": decay,
            "mining": {"ic": 0.031, "annualized_return": 0.15, "max_drawdown": -0.1},
            "oos": {"ic": 0.031 * (1 - decay), "annualized_return": 0.1,
                    "max_drawdown": -0.08}}


def test_shape_detection():
    assert feedback._detect_shape(_eval_result()) == "evaluate"
    assert feedback._detect_shape(_bt_result()) == "backtest"
    assert feedback._detect_shape(_oos_result()) == "oos"
    assert feedback._detect_shape({}) == "unknown"
    assert feedback._detect_shape({"__unparseable__": True}) == "unknown"


def test_extract_evaluate_uses_real_keys():
    m = feedback._extract(_eval_result(), "evaluate")
    assert m["ic"] == 0.031
    assert m["annualized"] == 0.15
    assert m["icir"] == 0.4


def test_verdict_success_strong_ic():
    r = feedback.build_feedback("动量因子", _bt_result(ic=0.06))
    assert r["verdict"] == "success"
    assert "进入组合层" in r["feedback"]


def test_verdict_weak_ic_rejected():
    r = feedback.build_feedback("h", _bt_result(ic=0.01))
    assert r["verdict"] == "rejected"
    assert "信号弱" in r["feedback"]


def test_verdict_decay_overfit():
    r = feedback.build_feedback("h", _oos_result(decay=0.8))
    assert r["verdict"] == "rejected"
    assert "过拟合" in r["feedback"]
    assert "80.0%" in r["feedback"]


def test_verdict_dedup_dropped():
    r = feedback.build_feedback("h", _bt_result(dedup=True, corr=0.995))
    assert r["verdict"] == "rejected"
    assert "去重闸门" in r["feedback"]
    assert "0.995" in r["feedback"]


def test_verdict_execution_failure_fixable():
    r = feedback.build_feedback("h", {"ok": False, "error": "boom",
                                      "traceback": "MemoryError: x\nline2\nlast-line"})
    assert r["verdict"] == "fixable"
    assert "last-line" in r["feedback"]
    assert "先修代码" in r["feedback"]


def test_evolution_tracking_with_prev():
    prev = feedback.build_feedback("h", _bt_result(ic=0.02))["metrics"]
    r = feedback.build_feedback("h", _bt_result(ic=0.04), prev=prev)
    assert "0.0200 → 0.0400" in r["feedback"]
    assert "方向正确" in r["feedback"]
    r2 = feedback.build_feedback("h", _bt_result(ic=0.01), prev=prev)
    assert "方向错误" in r2["feedback"]


def test_accepts_json_string():
    r = feedback.build_feedback("h", json.dumps(_bt_result(ic=0.06)))
    assert r["verdict"] == "success"


def test_degenerate_inputs():
    for bad in ("not json", None, [1, 2], 42, ""):
        r = feedback.build_feedback("", bad)
        assert r["verdict"] == "fixable"
        assert r["shape"] == "unknown"
        assert "无法识别" in r["feedback"]


def test_evaluate_error_branch():
    r = feedback.build_feedback("h", {"ok": False, "error": "截面标的数不足",
                                      "tearsheet": None})
    assert r["verdict"] == "fixable"


def test_sota_broken_noted():
    r = feedback.build_feedback("h", {**_bt_result(), "sota_broken": ["s1"]})
    assert "s1" in r["feedback"]
    assert "隔离" in r["feedback"]


# ── 负 IC 分支 ──

def test_negative_ic_direction_flipped():
    r = feedback.build_feedback("h", _bt_result(ic=-0.06))
    assert r["verdict"] == "fixable"
    assert "方向反了" in r["feedback"]
    assert "取负" in r["feedback"]
    assert "0.0600" in r["feedback"]  # 预期取负后 IC


def test_negative_weak_ic_still_weak():
    # 负但绝对值小 → 仍是弱信号，不误判为方向问题
    r = feedback.build_feedback("h", _bt_result(ic=-0.01))
    assert r["verdict"] == "rejected"
    assert "信号弱" in r["feedback"]


def test_negative_ic_priority_over_decay():
    # 强负 IC + 高衰减：方向分支优先（取负后 decay 会换个算法重新看）
    r = feedback.build_feedback("h", _oos_result(decay=0.8))
    r = feedback.build_feedback("h", {**_bt_result(ic=-0.07), "decay_hint": None})
    assert "方向反了" in r["feedback"]


# ── SQLite 闭环记忆 ──

def test_log_round_and_trend(tmp_path=None):
    import os
    import tempfile

    d = tempfile.mkdtemp()
    db = os.path.join(d, "fb.sqlite3")
    assert feedback.log_round(db, "h1", "rejected", {"ic": 0.01}) == {"round": 1}
    assert feedback.log_round(db, "h2", "success", {"ic": 0.06, "decay": 0.2}) == {"round": 2}
    t = feedback.history_trend(db)
    assert t["n"] == 2 and t["window"] == 2
    assert t["ic_first"] == 0.01 and t["ic_last"] == 0.06
    assert abs(t["ic_mean"] - 0.035) < 1e-12
    # window 限制
    assert feedback.history_trend(db, n=1)["window"] == 1
    # 无 IC 的记录不影响趋势
    feedback.log_round(db, "h3", "fixable", {})
    assert feedback.history_trend(db)["n"] == 2
    # 库不存在 → None
    assert feedback.history_trend(os.path.join(d, "none.sqlite3")) is None


def test_log_round_bad_path_no_raise():
    assert feedback.log_round("/nonexistent-dir/x/y.sqlite3", "h", "v", {"ic": 0.1}) == {"round": None}
