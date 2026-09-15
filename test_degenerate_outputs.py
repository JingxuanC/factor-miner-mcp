"""(c) 两个退化输出的回归测试。

1. `vol_forecast` 此前只有纯 EWMA —— 数学上**天然平坦**（无均值回复），
   所以永远拿不到期限结构，无法用于按 horizon 缩放仓位。
   现已补上均值回复路径（φ/LR 由滚动已实现方差的 AR(1) 估计）。
2. `predict` 此前缺输入时恒返回 signal=0.0 / confidence=0.5，
   会被上层误当成真实模型输出。现已返回 insufficient_input + null。
"""

import json

import numpy as np

import analytics
import models

DATES = ["2026-%02d-%02d" % (1 + i // 28, 1 + i % 28) for i in range(300)]


def _bars(vol_lo=0.004, vol_hi=0.030, n_hi=60, n=300, seed=7):
    """前段低波动、末段高波动 —— 这样当前方差 > 长期方差，预测应随 horizon 下降。"""
    rng = np.random.default_rng(seed)
    r = np.concatenate([rng.normal(0, vol_lo, n - n_hi), rng.normal(0, vol_hi, n_hi)])
    px = 100.0 * np.exp(np.cumsum(r))
    return [
        {"date": DATES[i], "open": c, "high": c * 1.002, "low": c * 0.998,
         "close": c, "volume": 1_000_000}
        for i, c in enumerate(px)
    ]


def _j(s):
    return json.loads(s)


# ────────────────────────────── vol_forecast


def test_ewma_mr_has_term_structure():
    """显式 ewma_mr：不尝试 GARCH，强制走均值回复路径。

    （本地/容器若装了 arch，method="auto" 会走 GARCH —— 那是 GARCH 自己的期限
    结构，不是这条路径，所以这里必须显式指定。）
    """
    r = _j(analytics.vol_forecast(_bars(), horizon=5, method="ewma_mr"))
    assert "error" not in r, r
    assert r["method"] == "ewma_mr", r
    assert r["term_structure"] == "mean_reverting"
    assert 0 < r["phi"] < 1
    assert r["long_run_vol"] > 0
    assert r["half_life_days"] is not None and r["half_life_days"] > 0
    vols = [d["vol"] for d in r["forecast"]]
    assert len(vols) == 5
    assert len(set(round(v, 12) for v in vols)) > 1, "期限结构不该是平的"
    assert r["horizon_ratio"] != 1.0


def test_auto_picks_a_method_with_term_structure():
    """auto 必须给出期限结构（装了 arch 走 GARCH，否则走 ewma_mr），
    不能是平坦的纯 EWMA。"""
    r = _j(analytics.vol_forecast(_bars(), horizon=5))
    assert "error" not in r, r
    assert r["method"] in ("garch", "ewma_mr"), r
    assert r["term_structure"] == "mean_reverting"


def test_vol_decays_toward_long_run_when_above():
    """当前波动显著高于长期水平 → 预测应随 horizon **下降**并趋于 long_run_vol。"""
    r = _j(analytics.vol_forecast(_bars(), horizon=10, method="ewma_mr"))
    assert "error" not in r, r
    vols = [d["vol"] for d in r["forecast"]]
    assert vols[0] > vols[-1], vols
    assert vols[0] >= r["long_run_vol"] - 1e-12
    # 单调趋近（不要求严格单调，但不能反向）
    for a, b in zip(vols, vols[1:]):
        assert b <= a + 1e-12, vols


def test_vol_rises_toward_long_run_when_below():
    """反向情形：当前波动显著低于长期水平 → 预测应上升。"""
    r = _j(analytics.vol_forecast(_bars(vol_lo=0.030, vol_hi=0.004, seed=11),
                                  horizon=10, method="ewma_mr"))
    assert "error" not in r, r
    if r["method"] == "ewma_mr":
        vols = [d["vol"] for d in r["forecast"]]
        assert vols[-1] > vols[0], vols


def test_explicit_ewma_stays_flat_and_says_so():
    """显式 ewma 必须保持 RiskMetrics 的平坦语义，并如实标注 —— 平坦不是 bug，
    但要让人知道那不是模型推断出来的期限结构。"""
    r = _j(analytics.vol_forecast(_bars(), horizon=5, method="ewma"))
    assert "error" not in r, r
    assert r["method"] == "ewma"
    assert r["term_structure"] == "flat"
    vols = [d["vol"] for d in r["forecast"]]
    assert len(set(vols)) == 1
    assert r["horizon_ratio"] == 1.0
    assert "note" in r and "平坦" in r["note"]


def test_short_series_falls_back_flat_with_note():
    """样本不足以估计 AR(1) 时必须回退平坦并说明，而不是静默给个假期限结构。"""
    r = _j(analytics.vol_forecast(_bars()[:35], horizon=5, method="ewma_mr"))
    assert "error" not in r, r
    assert r["method"] == "ewma"
    assert r["term_structure"] == "flat"
    assert "note" in r


def test_vol_forecast_shape_still_tolerant():
    """(a) 的形状容差不能被 (c) 的改动弄坏。"""
    bars = _bars()
    a = _j(analytics.vol_forecast(bars, horizon=5))
    b = _j(analytics.vol_forecast({"X": bars}, horizon=5))
    assert a == b, "flat bars 与 keyed bars 必须逐位相同"
    # columnar 只给了 date/close → 没有 high/low，按契约不该编造 parkinson_vol
    c = _j(analytics.vol_forecast(
        {"columns": ["date", "close"], "rows": [[d["date"], d["close"]] for d in bars]},
        horizon=5))
    assert "error" not in c
    assert c["forecast"] == a["forecast"]
    assert c["current_vol"] == a["current_vol"]
    assert c["method"] == a["method"]
    assert "parkinson_vol" not in c, "缺少 high/low 时不该有 parkinson_vol"
    assert "parkinson_vol" in a, "给了 high/low 时应当有 parkinson_vol"


# ────────────────────────────── predict


def test_predict_missing_input_is_explicit():
    """缺输入必须是 insufficient_input + null，而不是 0.0/0.5 这种假输出。"""
    r = models.ModelEngine().predict({"symbol": "sh600519", "factors": {}})
    assert r["status"] == "insufficient_input"
    assert r["signal"] is None and r["confidence"] is None
    assert r["is_mock"] is True
    assert r["missing"] == ["roc_5", "volume_ratio"]
    assert "ml_predict" in r["error"]


def test_predict_partial_input_also_explicit():
    r = models.ModelEngine().predict({"symbol": "x", "factors": {"roc_5": 0.02}})
    assert r["status"] == "insufficient_input"
    assert r["missing"] == ["volume_ratio"]


def test_predict_with_input_is_labelled_mock():
    r = models.ModelEngine().predict(
        {"symbol": "x", "factors": {"roc_5": 0.02, "volume_ratio": 1.5, "std_20": 0.01,
                                    "mfv": 1.0}})
    assert r["status"] == "ok"
    assert r["signal"] is not None and r["confidence"] is not None
    assert r["is_mock"] is True and r["method"] == "phase0_mock"
    assert "note" in r


def test_predict_never_silently_returns_zero():
    """回归：任何输入组合下都不能再出现「无标注的 0.0/0.5」。"""
    for fv in ({}, {"roc_5": 0.0}, {"ma_20": 1850.5}, {"typical_price": 100.0}):
        r = models.ModelEngine().predict({"symbol": "x", "factors": fv})
        assert r["is_mock"] is True
        if r["status"] == "ok":
            assert r.get("method") == "phase0_mock" and "note" in r
        else:
            assert r["signal"] is None and r["confidence"] is None
