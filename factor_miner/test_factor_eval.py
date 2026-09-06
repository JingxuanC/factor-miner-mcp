#!/usr/bin/env python3
"""test_factor_eval.py — factor_eval battery 合成 DataFrame 用例（DESIGN-FACTOR-MINER.md §13）

pytest 或直跑均可：
  python3 -m pytest factor_miner/test_factor_eval.py
  python3 -m factor_miner.test_factor_eval
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_eval import evaluate


def _daily_df(values=None) -> pd.DataFrame:
    """合规样本：MultiIndex(datetime, instrument) 单列 float64 日频。"""
    dates = pd.date_range("2019-01-01", periods=10, freq="B")
    instruments = ["SH600000", "SH600004", "SH600006"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    if values is None:
        values = np.random.default_rng(0).normal(size=len(idx))
    return pd.DataFrame({"factor": np.asarray(values, dtype="float64")}, index=idx)


def test_pass_case():
    res = evaluate(_daily_df())
    assert res.ok, res.feedback()
    assert res.failed == []


def test_series_input_accepted():
    res = evaluate(_daily_df().iloc[:, 0])
    assert res.ok, res.feedback()


def test_inf_case():
    df = _daily_df()
    df.iloc[0, 0] = float("inf")
    df.iloc[5, 0] = float("-inf")
    res = evaluate(df)
    assert not res.ok
    assert "no_inf" in res.failed
    assert "single_column" not in res.failed


def test_multi_column_case():
    df = _daily_df()
    df["second"] = 1.0
    res = evaluate(df)
    assert not res.ok
    assert "single_column" in res.failed


def test_intraday_case():
    dates = pd.date_range("2019-01-01 09:30", periods=20, freq="min")
    instruments = ["SH600000", "SH600004"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    df = pd.DataFrame({"factor": np.zeros(len(idx))}, index=idx)
    res = evaluate(df)
    assert not res.ok
    assert "daily_frequency" in res.failed


def test_missing_datetime_level_case():
    df = _daily_df().reset_index(drop=True)
    res = evaluate(df)
    assert not res.ok
    assert "datetime_index" in res.failed


def test_feedback_names_failed_checks():
    df = _daily_df()
    df.iloc[0, 0] = float("inf")
    df["second"] = 1.0
    res = evaluate(df)
    text = res.feedback()
    assert "single_column" in text and "no_inf" in text


def test_load_missing_file():
    res = evaluate("/tmp/definitely_not_exists_result.h5")
    assert not res.ok
    assert res.failed == ["load"]


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"OK {fn.__name__}")
    print(f"{len(fns)} tests passed")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
