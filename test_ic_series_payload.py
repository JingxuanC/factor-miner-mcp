"""`_ic_series_payload` 的单测：逐日 IC 序列的**序列化契约**。

它不需要 pytables/qlib —— 只吃一个 pandas Series（`groupby(level="datetime")`
的产物），所以这个文件在任何装了 pandas 的环境都能跑，不像
test_recent_ic_memory.py 那样必须进镜像。

钉住的是几件容易悄悄错、而错了以后图上"看着也像条曲线"的事：

  - 日期格式统一成 YYYY-MM-DD（调用方要拿它跟行情日期对齐）
  - NaN 点被丢掉，而不是变成 JSON 里的 null / Infinity（前端会把 JSON 的 null
    当 0 处理，那会在图上凭空多出一个"IC 归零"的假信号）
  - 重复日期时 `.loc` 返回 Series —— 必须取标量而不是把 Series 塞进 payload
  - `n`（当日有效样本数）跟着同一个日期走；取不到时为 None 而不是丢键
  - 空输入返回空列表，不抛异常
"""

import numpy as np
import pandas as pd

import factor_worker as fw


def _ics(pairs) -> pd.Series:
    """[(date, ic)] → 以 datetime 为索引的 Series（模拟 groupby 产物）。"""
    idx = pd.Index([pd.Timestamp(d) for d, _ in pairs], name="datetime")
    return pd.Series([v for _, v in pairs], index=idx)


def test_dates_are_iso_and_ordered_as_given():
    ics = _ics([
        ("2025-01-02", 0.031),
        ("2025-01-03", -0.012),
        ("2025-01-06", 0.044),
    ])
    out = fw._ic_series_payload(ics)
    assert [p["date"] for p in out] == ["2025-01-02", "2025-01-03", "2025-01-06"]
    assert [p["ic"] for p in out] == [0.031, -0.012, 0.044]
    # 不带 counts 时不应凭空造出 n
    assert all("n" not in p for p in out)


def test_nan_points_are_dropped_not_serialised_as_null():
    """NaN → 丢掉整点。若序列化成 null，前端画出来就是"IC 归零"的假信号。"""
    ics = _ics([
        ("2025-01-02", 0.031),
        ("2025-01-03", np.nan),
        ("2025-01-06", 0.044),
    ])
    out = fw._ic_series_payload(ics)
    assert [p["date"] for p in out] == ["2025-01-02", "2025-01-06"]
    assert all(isinstance(p["ic"], float) for p in out)


def test_duplicate_dates_yield_one_point_with_scalar_ic():
    """重复日期只能产出一个点。

    否则时间轴上会出现两个同日期点（渲染器画出回折），任何按日期合并的调用方
    也会拿到重复键。`.loc[dt]` 遇到重复索引返回的是 Series，直接塞进 payload
    甚至不是合法 JSON —— 所以要取标量并去重。
    """
    idx = pd.Index([pd.Timestamp("2025-01-02")] * 2, name="datetime")
    ics = pd.Series([0.031, 0.099], index=idx)
    out = fw._ic_series_payload(ics)
    assert len(out) == 1
    assert out[0]["date"] == "2025-01-02"
    assert isinstance(out[0]["ic"], float)
    assert out[0]["ic"] == 0.031  # 保留首个，而不是均值/Series


def test_dates_are_unique_even_with_mixed_sources():
    """同一日期以不同精度出现（Timestamp vs 字符串）也只保留一个点。"""
    idx = pd.Index(["2025-01-02", pd.Timestamp("2025-01-02"),
                    pd.Timestamp("2025-01-03")], name="datetime")
    ics = pd.Series([0.01, 0.02, 0.03], index=idx)
    out = fw._ic_series_payload(ics)
    dates = [p["date"] for p in out]
    assert len(dates) == len(set(dates)), dates
    assert "2025-01-02" in dates and "2025-01-03" in dates


def test_counts_ride_along_with_their_date():
    ics = _ics([("2025-01-02", 0.031), ("2025-01-03", 0.012)])
    counts = pd.Series([1200, 1180],
                       index=pd.Index([pd.Timestamp("2025-01-02"),
                                       pd.Timestamp("2025-01-03")], name="datetime"))
    out = fw._ic_series_payload(ics, counts)
    assert [(p["date"], p["n"]) for p in out] == [("2025-01-02", 1200), ("2025-01-03", 1180)]


def test_missing_count_becomes_none_instead_of_keyerror():
    """counts 里缺某天时给 None，不要 KeyError，也不要偷偷丢掉那个 IC 点。"""
    ics = _ics([("2025-01-02", 0.031), ("2025-01-03", 0.012)])
    counts = pd.Series([1200], index=pd.Index([pd.Timestamp("2025-01-02")], name="datetime"))
    out = fw._ic_series_payload(ics, counts)
    assert len(out) == 2
    assert out[0]["n"] == 1200
    assert out[1]["n"] is None


def test_empty_input_returns_empty_list():
    assert fw._ic_series_payload(pd.Series(dtype=float)) == []


def test_unparseable_date_is_skipped_not_raised():
    """坏日期跳过该点，整条序列仍要返回 —— 一个坏索引不该让看板没有曲线。"""
    idx = pd.Index(["not-a-date", pd.Timestamp("2025-01-02")], name="datetime")
    ics = pd.Series([0.05, 0.031], index=idx)
    out = fw._ic_series_payload(ics)
    # 取决于 pandas 能否把该字符串当日期解析；要么被跳过，要么被规范化。
    # 关键在于：不抛异常，且保留下来的点日期都是 ISO 或已丢弃。
    assert isinstance(out, list)
    for p in out:
        assert len(p["date"]) == 10 and p["date"][4] == "-"


if __name__ == "__main__":
    import traceback

    failures = 0
    for fn in (test_dates_are_iso_and_ordered_as_given,
               test_nan_points_are_dropped_not_serialised_as_null,
               test_duplicate_dates_yield_one_point_with_scalar_ic,
               test_dates_are_unique_even_with_mixed_sources,
               test_counts_ride_along_with_their_date,
               test_missing_count_becomes_none_instead_of_keyerror,
               test_empty_input_returns_empty_list,
               test_unparseable_date_is_skipped_not_raised):
        try:
            fn()
            print(f"PASS  {fn.__name__}", flush=True)
        except Exception:
            failures += 1
            print(f"FAIL  {fn.__name__}", flush=True)
            traceback.print_exc()
    print("FAILURES:", failures)
    raise SystemExit(1 if failures else 0)
