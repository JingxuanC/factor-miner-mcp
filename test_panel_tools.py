"""形状容差验收测试：真实分析工具对 5 种等价形状必须给出**逐位相同**的结果。

这是 panel 契约的落地证明 —— 之前 `factor_tearsheet` 只吃 `{symbol: bars}`、
`vol_forecast` 只吃 `[bars]`，同一个 agent 要在多种形状间反复转换。
"""

import json

import numpy as np
import pytest

import analytics

N_SYM, N_DAY = 12, 120
SYMS = ["sh6005%02d" % i for i in range(N_SYM)]
DATES = ["2026-%02d-%02d" % (1 + (i // 28), 1 + (i % 28)) for i in range(N_DAY)]

rng = np.random.default_rng(20260915)
CLOSES = {
    s: list(100.0 + np.cumsum(rng.normal(0, 1.2, N_DAY)) + i * 5.0)
    for i, s in enumerate(SYMS)
}


def _bars(sym):
    return [
        {"date": d, "open": c, "high": c * 1.01, "low": c * 0.99,
         "close": c, "volume": 1000 + i}
        for i, (d, c) in enumerate(zip(DATES, CLOSES[sym]))
    ]


KEYED = {s: _bars(s) for s in SYMS}

MULTI_SHAPES = {
    "keyed_bars": KEYED,
    "columnar": {
        "columns": ["date", "symbol", "close"],
        "rows": [[d, s, c] for s in SYMS for d, c in zip(DATES, CLOSES[s])],
    },
    "wide_symbol_outer": {s: dict(zip(DATES, CLOSES[s])) for s in SYMS},
    "wide_date_outer": {
        d: {s: CLOSES[s][i] for s in SYMS} for i, d in enumerate(DATES)
    },
    "long_frame": [
        {"date": d, "symbol": s, "close": c}
        for s in SYMS for d, c in zip(DATES, CLOSES[s])
    ],
}

ONE = SYMS[0]
SINGLE_SHAPES = {
    "flat_bars": _bars(ONE),
    "columnar": {
        "columns": ["date", "close"],
        "rows": [[d, c] for d, c in zip(DATES, CLOSES[ONE])],
    },
    "wide_symbol_outer": {ONE: dict(zip(DATES, CLOSES[ONE]))},
    "long_frame": [
        {"date": d, "symbol": ONE, "close": c}
        for d, c in zip(DATES, CLOSES[ONE])
    ],
}


def _j(s):
    return json.loads(s)


def test_all_shapes_normalize_to_one_frame():
    frames = {k: analytics._klines_to_close_frame(v) for k, v in MULTI_SHAPES.items()}
    ref = frames["keyed_bars"]
    assert ref.shape == (N_DAY, N_SYM)
    for k, f in frames.items():
        assert list(f.columns) == list(ref.columns), k
        np.testing.assert_allclose(f.to_numpy(), ref.to_numpy(), err_msg=k)


# ------------------------------------------------------------ 真实工具

def test_factor_tearsheet_shape_invariant():
    """tearsheet 原本只接受 keyed bars。"""
    fvals = [
        {"date": d, "symbol": s, "value": float(np.sin((i + si * 13) / 5.0))}
        for si, s in enumerate(SYMS) for i, d in enumerate(DATES)
    ]
    out = {}
    for k, kl in MULTI_SHAPES.items():
        out[k] = _j(analytics.factor_tearsheet(fvals, kl, quantiles=5, periods=[1, 5]))
        assert "error" not in out[k], (k, out[k].get("error"))
    ref = out["keyed_bars"]
    assert ref["quantile_returns"]["5d"]["mean_period_return"]["q1"] is not None
    for k, v in out.items():
        assert v == ref, k  # 完整结果逐位相同（同一份数据 → 同一个 frame → 同一结果）


def test_factor_tearsheet_accepts_panel_factor_values():
    """factor_values 也走同一契约：{columns,rows} 面板应当可用。"""
    long_fv = [
        {"date": d, "symbol": s, "value": float(np.cos(i / 7.0))}
        for s in SYMS for i, d in enumerate(DATES)
    ]
    col_fv = {
        "columns": ["date", "symbol", "value"],
        "rows": [[r["date"], r["symbol"], r["value"]] for r in long_fv],
    }
    a = _j(analytics.factor_tearsheet(long_fv, KEYED, periods=[1, 5]))
    b = _j(analytics.factor_tearsheet(col_fv, KEYED, periods=[1, 5]))
    assert a.get("error") is None and b.get("error") is None
    assert a["quantile_returns"] == b["quantile_returns"]


def test_portfolio_optimize_shape_invariant():
    out = {}
    for k, kl in MULTI_SHAPES.items():
        out[k] = _j(analytics.portfolio_optimize(SYMS, kl, method="hrp", lookback=100))
        assert "error" not in out[k], (k, out[k].get("error"))
    ref = out["keyed_bars"]
    for k, v in out.items():
        assert v["weights"] == pytest.approx(ref["weights"]), k


def test_regime_detect_shape_invariant():
    out = {
        k: _j(analytics.regime_detect(kl, n_regimes=3))
        for k, kl in SINGLE_SHAPES.items()
    }
    for k, v in out.items():
        assert "error" not in v, (k, v.get("error"))
    ref = out["flat_bars"]["current_regime"]
    for k, v in out.items():
        assert v["current_regime"] == ref, k


def test_vol_forecast_shape_invariant():
    out = {
        k: _j(analytics.vol_forecast(kl, horizon=5))
        for k, kl in SINGLE_SHAPES.items()
    }
    for k, v in out.items():
        assert "error" not in v, (k, v.get("error"))
    ref = out["flat_bars"]["forecast"]
    for k, v in out.items():
        assert v["forecast"] == pytest.approx(ref), k


def test_change_point_accepts_series_shapes():
    """change_point 的 series 现在也接受裸浮点数组 / {date: value}。"""
    vals = CLOSES[ONE]
    a = _j(analytics.change_point([{"date": d, "value": v} for d, v in zip(DATES, vals)]))
    b = _j(analytics.change_point(dict(zip(DATES, vals))))
    assert "error" not in a and "error" not in b, (a.get("error"), b.get("error"))
    assert a["change_points"] == b["change_points"]

    c = _j(analytics.change_point(list(vals)))  # 裸数组（合成日期）
    assert "error" not in c


def test_multi_series_rejected_with_actionable_message():
    """单序列工具收到多序列必须给出可行动的报错，而不是静默取第一列。"""
    r = _j(analytics.vol_forecast(KEYED, horizon=5))
    assert "error" in r
    assert "单序列" in r["error"] and "sh6005" in r["error"]


def test_bad_shape_message_names_the_shape():
    r = _j(analytics.factor_tearsheet([], {"nonsense": 1}, periods=[1]))
    assert "error" in r
    assert "无法解析" in r["error"] or "为空" in r["error"]
