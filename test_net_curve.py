"""净值曲线两个纯函数的单测：日期提取与抽稀。

这两件事都有"错了也看不出来"的失败模式：

  - 日期解析失败若默默返回空曲线，图上什么都没有，而那与"这个因子没跑成"
    长得一模一样 —— 所以约定是**退回下标**而不是消失；
  - 抽稀若丢掉了首尾点，曲线起点会漂到第二天、终点到倒数第二天，而那不是
    数据的问题，是画的问题；
  - 抽稀若按时间聚合而不是按下标等距，回撤的局部极值会被抹平，抽稀后的曲线
    与 metrics 里的 max_drawdown 对不上 —— 这两个数字本应互相印证。

不需要 pytables/qlib，任何装了 pandas 的环境都能跑。
"""

import math

import numpy as np
import pandas as pd

import factor_worker as fw
from factor_executor import parse as ep
from factor_miner import factor_backtest as fb


def _report(idx, values, extra_col: str = "return"):
    """造一份形状与 qlib report_normal_1day 相同的报告。"""
    return pd.DataFrame({"value": values, extra_col: [0.0] * len(values)}, index=idx)


DATES = pd.DatetimeIndex(["2021-01-04", "2021-01-05", "2021-01-06"], name="datetime")


# ── _net_curve ──────────────────────────────────────────────────────────────

def test_net_curve_takes_dates_from_the_report_index():
    rep = _report(DATES, [1.0, 1.02, 0.99])
    for fn, label in ((fb._net_curve, "factor_backtest"), (ep._net_curve, "executor.parse")):
        out = fn(rep)
        assert [p["date"] for p in out] == ["2021-01-04", "2021-01-05", "2021-01-06"], label
        assert [p["i"] for p in out] == [0, 1, 2], label
        assert out[1]["value"] == 1.02, label


def test_both_net_curve_implementations_agree():
    """两份实现是刻意重复的（执行器不能 import miner 业务模块），口径必须一致。

    数值口径不一致是这次拆分最危险的失败模式：同一个因子在"本地回测"和"远程
    回测"下会给出不同的图，而那种不一致极难被发现。
    """
    rep = _report(DATES, [1.0, 1.02, 0.99])
    assert fb._net_curve(rep) == ep._net_curve(rep)


def test_net_curve_falls_back_to_index_when_dates_are_unparseable():
    """索引不是日期时给 i，而不是丢掉整条曲线。

    给一条 x 轴是"第 N 个交易日"的曲线，远好于整条曲线消失 —— 后者看起来与
    "因子没跑成"没有区别。
    """
    rep = _report(pd.RangeIndex(3, name="datetime"), [1.0, 1.1, 1.2])
    for fn in (fb._net_curve, ep._net_curve):
        out = fn(rep)
        assert len(out) == 3
        assert [p["i"] for p in out] == [0, 1, 2]
        assert all(p["date"] is None for p in out)


def test_net_curve_skips_non_finite_and_uncoercible_values():
    rep = _report(DATES, [1.0, float("nan"), 1.2])
    for fn in (fb._net_curve, ep._net_curve):
        out = fn(rep)
        assert [p["value"] for p in out] == [1.0, 1.2]

    rep2 = _report(DATES, ["nope", 1.1, 1.2])
    for fn in (fb._net_curve, ep._net_curve):
        assert [p["value"] for p in fn(rep2)] == [1.1, 1.2]


def test_net_curve_drops_the_leading_qlib_placeholder_zero():
    """首值是 qlib 的占位 0（生产 7/7 份 report 都如此），必须丢掉。

    留着它，曲线从 0 起步、读起来像"亏光了本金"，而且会把 y 轴压扁到看不见变化。
    ``i`` 保持原值，所以时间轴不会错位（第一个点的 i 应当是 1 而不是 0）。
    """
    rep = _report(DATES, [0.0, 8.7e7, 9.9e7])
    for fn in (fb._net_curve, ep._net_curve):
        out = fn(rep)
        assert [p["value"] for p in out] == [8.7e7, 9.9e7]
        assert [p["i"] for p in out] == [1, 2]
        assert out[0]["date"] == "2021-01-05"


def test_net_curve_keeps_zeros_that_are_not_leading():
    """序列中间的 0 是真实净值，不能动。"""
    idx4 = pd.DatetimeIndex(
        ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"], name="datetime")
    rep = _report(idx4, [0.0, 8.7e7, 0.0, 9.0e7])
    for fn in (fb._net_curve, ep._net_curve):
        assert [p["value"] for p in fn(rep)] == [8.7e7, 0.0, 9.0e7]


def test_net_curve_keeps_an_all_zero_series_intact():
    """整条都是 0 时原样返回 —— 用"没有曲线"代替"曲线是平的"是更糟的失败。"""
    rep = _report(DATES, [0.0, 0.0, 0.0])
    for fn in (fb._net_curve, ep._net_curve):
        assert [p["value"] for p in fn(rep)] == [0.0, 0.0, 0.0]


def test_net_curve_returns_empty_without_a_value_column():
    """报告结构变化时返回空，而不是抛异常炸掉整次 OOS 检查。"""
    rep = pd.DataFrame({"account": [0.1, 0.2]}, index=DATES[:2])
    for fn in (fb._net_curve, ep._net_curve):
        assert fn(rep) == []
        assert fn(None) == []


# ── _downsample_curve ───────────────────────────────────────────────────────

def _curve(n):
    return [{"date": f"d{i}", "i": i, "value": float(i)} for i in range(n)]


def test_downsample_keeps_first_and_last():
    """首尾必须保留：丢首点会让曲线起点漂到第二天。"""
    out = fw._downsample_curve(_curve(1000), 10)
    assert len(out) == 10
    assert out[0]["i"] == 0
    assert out[-1]["i"] == 999


def test_downsample_is_capped_and_evenly_spaced():
    out = fw._downsample_curve(_curve(2200), 400)
    assert len(out) == 400
    idxs = [p["i"] for p in out]
    assert idxs == sorted(set(idxs)), "下标必须严格递增且无重复"
    # 等距：相邻间隔最多相差 1（首尾修正造成的）
    gaps = {b - a for a, b in zip(idxs, idxs[1:])}
    assert max(gaps) - min(gaps) <= 1, gaps


def test_downsample_is_a_noop_below_the_cap():
    c = _curve(50)
    assert fw._downsample_curve(c, 400) == c
    assert len(fw._downsample_curve(c, 400)) == 50


def test_downsample_degenerate_inputs():
    assert fw._downsample_curve(None) == []
    assert fw._downsample_curve([]) == []
    # max_points=1 → 只留最后一点（最近的净值最有意义）
    out = fw._downsample_curve(_curve(10), 1)
    assert [p["i"] for p in out] == [9]
    # 非法/缺失 max_points 退回**默认上限**而不是抛异常，也不是"不设限"：
    # 默认上限存在的理由就是别把 2200 点原样塞进 MCP 响应。
    assert len(fw._downsample_curve(_curve(1000), 0)) == 1000  # 0 = 不设限（显式）
    assert len(fw._downsample_curve(_curve(500), "nope")) == fw.NET_CURVE_MAX_POINTS
    assert len(fw._downsample_curve(_curve(500))) == fw.NET_CURVE_MAX_POINTS


def test_downsample_does_not_invent_points():
    """抽稀只能少点，不能造点 —— 每个返回点都必须来自输入。"""
    src = _curve(777)
    out = fw._downsample_curve(src, 100)
    values = {p["value"] for p in src}
    assert all(p["value"] in values for p in out)
    assert len({p["i"] for p in out}) == len(out)


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}", flush=True)
            except Exception:
                failures += 1
                print(f"FAIL  {name}", flush=True)
                traceback.print_exc()
    print("FAILURES:", failures)
    raise SystemExit(1 if failures else 0)
