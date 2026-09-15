"""panel 契约测试：核心性质是**形状不变性** ——
同一份数据用 5 种受支持形状喂进去，归一化结果必须完全一致。
"""

import numpy as np
import pandas as pd
import pytest

from panel import (
    PanelError,
    as_bar_dict,
    as_bars,
    as_close_frame,
    as_dataframe,
    as_features,
    as_matrix,
    as_series,
    detect_shape,
)

# 两个标的、3 个交易日 —— 小到能人工核对，又足以暴露对齐错误
DATES = ["2026-01-05", "2026-01-06", "2026-01-07"]
A = [10.0, 11.0, 12.0]
B = [20.0, 19.0, 18.0]


def _flat(sym, vals):
    return [{"date": d, "open": v, "high": v, "low": v, "close": v, "volume": 100}
            for d, v in zip(DATES, vals)]


SHAPES = {
    # 1. keyed bars（tearsheet/portfolio_optimize 的原生形状）
    "keyed_bars": {"AAA": _flat("AAA", A), "BBB": _flat("BBB", B)},
    # 2. flat bars（kronos/regime_detect 的原生形状）
    "flat_bars": _flat("AAA", A),
    # 3. columnar panel（learn_graph/pcmci_discover 的原生形状）
    "columnar": {
        "columns": ["date", "symbol", "close"],
        "rows": [
            [d, "AAA", a] for d, a in zip(DATES, A)
        ] + [
            [d, "BBB", b] for d, b in zip(DATES, B)
        ],
    },
    # 4a. wide close，外层 symbol
    "wide_sym_outer": {"AAA": dict(zip(DATES, A)), "BBB": dict(zip(DATES, B))},
    # 4b. wide close，外层 date
    "wide_date_outer": {
        d: {"AAA": a, "BBB": b} for d, a, b in zip(DATES, A, B)
    },
    # 5. long frame
    "long_frame": [
        {"date": d, "symbol": "AAA", "close": a} for d, a in zip(DATES, A)
    ] + [
        {"date": d, "symbol": "BBB", "close": b} for d, b in zip(DATES, B)
    ],
}

MULTI = ["keyed_bars", "columnar", "wide_sym_outer", "wide_date_outer", "long_frame"]


def test_detect_shape():
    assert detect_shape(SHAPES["keyed_bars"]) == "keyed_bars"
    assert detect_shape(SHAPES["flat_bars"]) == "flat_bars"
    assert detect_shape(SHAPES["columnar"]) == "columnar_panel"
    assert detect_shape(SHAPES["wide_sym_outer"]) == "wide_close"
    assert detect_shape(SHAPES["wide_date_outer"]) == "wide_close"
    assert detect_shape(SHAPES["long_frame"]) == "long_frame"


@pytest.mark.parametrize("name", MULTI)
def test_close_frame_is_shape_invariant(name):
    """★ 契约核心：5 种形状 → 同一个 close frame。"""
    got = as_close_frame(SHAPES[name])
    assert list(got.columns) == ["AAA", "BBB"], (name, got.columns)
    assert [str(d.date()) for d in got.index] == DATES
    np.testing.assert_allclose(got["AAA"].to_numpy(), A)
    np.testing.assert_allclose(got["BBB"].to_numpy(), B)


@pytest.mark.parametrize("name", MULTI)
def test_bar_dict_is_shape_invariant(name):
    bd = as_bar_dict(SHAPES[name])
    assert set(bd) == {"AAA", "BBB"}
    assert [r["date"] for r in bd["AAA"]] == DATES
    np.testing.assert_allclose([r["close"] for r in bd["BBB"]], B)


@pytest.mark.parametrize("name", MULTI)
def test_dataframe_is_shape_invariant(name):
    df = as_dataframe(SHAPES[name])
    assert set(["date", "symbol", "close"]) <= set(df.columns)
    assert len(df) == 6
    for f in ("open", "high", "low"):  # 缺省用 close 补齐
        np.testing.assert_allclose(df[f].to_numpy(), df["close"].to_numpy())


def test_flat_bars_single_series():
    b = as_bars(SHAPES["flat_bars"])
    assert len(b) == 3
    np.testing.assert_allclose([r["close"] for r in b], A)
    s = as_series(SHAPES["flat_bars"])
    assert [r["date"] for r in s] == DATES
    np.testing.assert_allclose([r["value"] for r in s], A)


def test_multi_series_rejected_not_silently_truncated():
    """多序列喂给单序列工具必须报错 —— 静默取第一个是 09-15 那批错配难查的根源。"""
    with pytest.raises(PanelError) as e:
        as_bars(SHAPES["keyed_bars"])
    assert "单序列" in str(e.value)
    with pytest.raises(PanelError):
        as_series(SHAPES["keyed_bars"])


def test_matrix_multi_column():
    cols, rows = as_matrix(SHAPES["keyed_bars"])
    assert cols == ["AAA", "BBB"]          # 不含日期列
    assert len(rows) == 3
    assert rows[0][0] == pytest.approx(10.0)
    assert rows[2][1] == pytest.approx(18.0)


def test_matrix_with_dates_side_channel():
    """日期走独立字段 —— rows 必须能直接 dtype=float 转成数组。"""
    cols, rows, dates = as_matrix(SHAPES["keyed_bars"], with_dates=True)
    assert cols == ["AAA", "BBB"]
    assert dates == DATES
    import numpy as np
    np.asarray(rows, dtype=float)  # 不能抛 could not convert string to float


def test_matrix_single_column_ok():
    """单序列矩阵化不算错误（因果图工具需要 >=2 列时会自己再校验）。

    flat bars 不含 symbol 字段 → 落到 default_symbol（"asset"）。
    """
    cols, rows = as_matrix(SHAPES["flat_bars"])
    assert cols == ["asset"]
    assert len(rows) == 3


def test_features_passthrough_and_expand():
    native = {"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]}
    assert as_features(native) == native
    expanded = as_features(SHAPES["columnar"])
    assert set(expanded) == {"close"}, expanded
    assert len(expanded["close"]) == 6
    feats = as_features(SHAPES["keyed_bars"])
    assert set(feats) == {"AAA", "BBB"}
    np.testing.assert_allclose(feats["BBB"], B)


def test_value_list_gets_synthetic_dates():
    """裸浮点数组也要能用（kronos 的 klines 有时只传收盘价）。"""
    df = as_dataframe([1.0, 2.0, 3.0])
    assert len(df) == 3
    np.testing.assert_allclose(df["close"].to_numpy(), [1.0, 2.0, 3.0])


def test_series_dict_shape():
    """{date: value} 也要能识别。"""
    df = as_dataframe({"2026-01-05": 1.0, "2026-01-06": 2.0})
    assert len(df) == 2
    assert set(df["symbol"]) == {"asset"}


def test_columnar_row_width_mismatch_raises():
    with pytest.raises(PanelError) as e:
        as_dataframe({"columns": ["date", "close"], "rows": [["2026-01-05", 1.0, 99.0]]})
    assert "行宽" in str(e.value)


def test_errors_carry_shape_info():
    """报错必须点名实际收到的形状 —— 这是能自诊断的前提。"""
    with pytest.raises(PanelError) as e:
        as_dataframe({"weird": 123})
    assert "weird" in str(e.value)
    with pytest.raises(PanelError) as e:
        as_dataframe("hello")
    assert "str" in str(e.value)
    with pytest.raises(PanelError):
        as_dataframe(None)
    with pytest.raises(PanelError):
        as_dataframe([])


def test_series_fill_forward_gap():
    """停牌缺口按 ffill 处理（tearsheet 依赖这个行为）。"""
    rows = [
        {"date": "2026-01-05", "symbol": "AAA", "close": 10.0},
        {"date": "2026-01-06", "symbol": "AAA", "close": None},
        {"date": "2026-01-07", "symbol": "AAA", "close": 12.0},
    ]
    px = as_close_frame(rows)
    assert len(px) == 3
    assert px["AAA"].iloc[1] == pytest.approx(10.0)


def test_is_flat_bars_detection():
    """快速路径判定：带 symbol 的 list of dicts 不算 flat bars（否则多标的会被静默合并）。"""
    from panel import is_flat_bars
    assert is_flat_bars(SHAPES["flat_bars"]) is True
    assert is_flat_bars(SHAPES["long_frame"]) is False      # 带 symbol 列
    assert is_flat_bars(SHAPES["keyed_bars"]) is False
    assert is_flat_bars([{"date": "2026-01-01", "close": 1.0}]) is False  # 缺 open
    assert is_flat_bars([1.0, 2.0]) is False
    assert is_flat_bars(None) is False
