#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_report.py — report.py 渲染层测试

渲染层契约（report.py 模块 docstring）：
  - 纯标准库（HTML + 内联 SVG），无第三方依赖
  - 永不抛异常：任何畸形输入都产出占位文本
  - 用户可控字符串全部 escape
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report  # noqa: E402


def _ok_result():
    return {
        "ok": True,
        "dedup_dropped": False,
        "sota_broken": [],
        "metrics": {
            "IC": 0.0312,
            "Rank IC": 0.0288,
            "ICIR": 0.42,
            "1day.excess_return_with_cost.annualized_return": 0.153,
            "1day.excess_return_with_cost.max_drawdown": -0.112,
        },
        "correlations": {"mom20": [0.31, 0.87]},
        "net_values": [1.0, 1.01, 1.02],
        "net_curve": [
            {"date": "2024-01-02", "i": 0, "value": 1.0},
            {"date": "2024-01-03", "i": 1, "value": 1.01},
            {"date": "2024-01-04", "i": 2, "value": 1.02},
        ],
        "error": "",
        "traceback": "",
    }


def _meta():
    return {"sota": [{"name": "mom5", "code": "c"}, {"name": "vol10", "code": "c"}],
            "new_factors": [{"name": "mom20", "code": "c"}],
            "profile": "smoke", "data_version": "sha256:abc"}


def test_basic_render_contains_key_sections():
    html = report.render_backtest_report(_ok_result(), _meta())
    assert "<!doctype html>" in html
    for kw in ("净值曲线", "回撤", "去重闸门", "买卖点", "mom20", "vol10"):
        assert kw in html, kw
    # 指标值要出现
    assert "0.0312" in html
    # 自包含：不允许任何外部资源引用
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html
    assert "<script" not in html


def test_status_banners():
    ok_html = report.render_backtest_report(_ok_result(), _meta())
    assert 'class="banner ok"' in ok_html

    dedup = dict(_ok_result(), ok=False, dedup_dropped=True)
    dedup_html = report.render_backtest_report(dedup, _meta())
    assert 'class="banner warn"' in dedup_html
    assert "FactorEmptyError" in dedup_html

    fail = dict(_ok_result(), ok=False, error="boom", traceback="tb-line")
    fail_html = report.render_backtest_report(fail, _meta())
    assert 'class="banner bad"' in fail_html
    assert "boom" in fail_html and "tb-line" in fail_html


def test_xss_escaped():
    evil = _meta()
    evil["new_factors"] = [{"name": "<script>alert(1)</script>", "code": "c"}]
    bad = _ok_result()
    bad["error"] = "<img src=x onerror=alert(1)>"
    html = report.render_backtest_report(bad, evil)
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_degenerate_inputs_never_raise():
    # 空结果 / None / 缺字段 / 全零净值 / 单点曲线
    for payload in ({}, {"ok": False}, {"net_curve": []},
                    {"net_curve": [{"date": None, "i": 0, "value": 0.0}] * 5},
                    {"net_curve": [{"date": "x", "value": float("nan")},
                                   {"date": "y", "value": float("inf")}]},
                    {"metrics": {"IC": None, "Rank IC": "oops"}},
                    {"correlations": {"f": [None, "x", 0.5]}}):
        html = report.render_backtest_report(payload, None)  # meta 也是 None
        assert "<!doctype html>" in html
    # None 整个传入也不抛
    assert "<!doctype html>" in report.render_backtest_report(None, None)


def test_net_values_fallback_when_no_dates():
    r = _ok_result()
    r["net_curve"] = []
    html = report.render_backtest_report(r, _meta())
    assert "数据点不足" not in html
    assert "<svg" in html  # 从 net_values 画出曲线


def test_drawdown_math():
    dd = report._drawdown([1.0, 1.2, 0.9, 1.3])
    assert dd[0] == 0.0
    assert abs(dd[2] - (0.9 / 1.2 - 1)) < 1e-12
    assert dd[3] == 0.0  # 创新高后回撤归零


def test_downsample_bounds():
    pts = list(range(5000))
    out = report._downsample(pts, 600)
    assert len(out) == 600
    assert out[0] == 0 and out[-1] == 4999
    assert report._downsample([1, 2, 3], 600) == [1, 2, 3]


def test_corr_gate_line_and_bad_bar():
    html = report.render_backtest_report(_ok_result(), _meta())
    assert "dedup gate 0.99" in html
    # mom20 × vol10 = 0.87 通过（蓝色 bar），mom20 × mom5 = 0.31 通过
    assert 'class="bar bad"' not in html
    # 高相关（撞闸门）→ 红色 bar
    hot = _ok_result()
    hot["correlations"] = {"mom20": [0.995, 0.1]}
    hot_html = report.render_backtest_report(hot, _meta())
    assert 'class="bar bad"' in hot_html


def test_render_exception_isolated():
    # 即使 _render 内部炸掉，render_backtest_report 也要回退化错误页
    original = report._render
    try:
        report._render = lambda *a, **k: 1 / 0
        html = report.render_backtest_report(_ok_result(), _meta())
        assert "报告渲染失败" in html
    finally:
        report._render = original


# ── OOS 报告 ──

def _oos_result(decay=0.22):
    return {
        "ok": True,
        "oos": {"ic": 0.024, "annualized_return": 0.11, "max_drawdown": -0.09},
        "mining": {"ic": 0.031, "annualized_return": 0.153, "max_drawdown": -0.112},
        "decay": decay,
        "mining_net_curve": [{"date": None, "i": i, "value": 1.0 + i * 0.001} for i in range(60)],
        "oos_net_curve": [{"date": None, "i": i, "value": 1.0 + i * 0.0007} for i in range(40)],
        "error": "",
    }


def test_oos_render_two_curves_and_verdict():
    html = report.render_oos_report(_oos_result(0.22), {"name": "mom20"})
    assert "<!doctype html>" in html
    assert "挖掘窗口" in html and "样本外窗口" in html
    assert html.count("<polyline") == 2          # 两条净值曲线
    assert 'class="banner ok"' in html
    assert "22.0%" in html
    # 指标对比表
    assert "<th>挖掘窗口</th>" in html and "0.0310" in html and "0.0240" in html


def test_oos_decay_grading():
    assert 'class="banner ok"' in report.render_oos_report(_oos_result(0.0), {})
    assert 'class="banner warn"' in report.render_oos_report(_oos_result(0.5), {})
    assert 'class="banner bad"' in report.render_oos_report(_oos_result(0.8), {})
    # decay 不可计算 → warn（不可证伪拒绝）
    r = _oos_result(None)
    r["ok"] = True
    assert 'class="banner warn"' in report.render_oos_report(r, {})
    # 检验失败 → bad
    f = dict(_oos_result(0.1), ok=False, error="mining-window backtest failed: x")
    assert 'class="banner bad"' in report.render_oos_report(f, {})


def test_oos_degenerate_never_raises():
    for payload in ({}, {"ok": True}, {"ok": True, "decay": "oops"},
                    {"ok": True, "mining_net_curve": [{"value": 1.0}],  # 单点
                     "oos_net_curve": []},
                    {"ok": True, "decay": float("nan"),
                     "mining_net_curve": [{"value": 0.0}] * 5,   # 全零被滤掉
                     "oos_net_curve": [{"value": -1.0}] * 5}):   # 负值被滤掉
        html = report.render_oos_report(payload, None)
        assert "<!doctype html>" in html
    assert "<!doctype html>" in report.render_oos_report(None, None)


def test_oos_xss_escaped():
    r = _oos_result(0.1)
    r["error"] = "<script>alert(1)</script>"
    html = report.render_oos_report(r, {"name": "<img src=x onerror=alert(1)>"})
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html


def test_oos_render_exception_isolated():
    original = report._render_oos
    try:
        report._render_oos = lambda *a, **k: 1 / 0
        html = report.render_oos_report(_oos_result(0.1), {"name": "x"})
        assert "报告渲染失败" in html
    finally:
        report._render_oos = original
