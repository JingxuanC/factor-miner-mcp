#!/usr/bin/env python3
"""test_factor_eval.py — factor_eval battery 合成 DataFrame 用例（DESIGN-FACTOR-MINER.md §13）

pytest 或直跑均可：
  python3 -m pytest factor_miner/test_factor_eval.py
  python3 -m factor_miner.test_factor_eval
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


# ═══════════════════════════════════════════════════════════════════
# 沙箱输入截窗（factor_worker.stage_h5_for_sandbox）
#
# 背景：全量 daily_pv_all.h5 是 6000+ 标的 × 4300+ 交易日（实测 15,095,115 行），
# 因子代码 read_hdf 读它会在沙箱 RLIMIT_AS=2GB 下炸：
#   numpy._core._exceptions._ArrayMemoryError: Unable to allocate 345. MiB
# 而 recent_ic / daily_compute 都只用尾部窗口 —— 截窗必须"结果不变"。
# ═══════════════════════════════════════════════════════════════════

def _mk_pv(n_days: int, n_inst: int = 3) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd
    dts = pd.date_range("2024-01-01", periods=n_days, freq="B")
    insts = ["sh60000%d" % i for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dts, insts], names=["datetime", "instrument"])
    return pd.DataFrame({
        "$close": np.arange(len(idx), dtype="float64"),
        "$open": np.arange(len(idx), dtype="float64"),
        "$high": np.arange(len(idx), dtype="float64") + 1,
        "$low": np.arange(len(idx), dtype="float64") - 1,
        "$volume": np.ones(len(idx)),
        "$factor": np.ones(len(idx)),
    }, index=idx)


def test_stage_h5_window_keeps_tail_only(tmp_path):
    pytest.importorskip("tables")   # HDF5 依赖：未装则跳过，不误报
    """截窗后只剩最近 N 天，且尾部数据逐值不变（IC/最新截面口径不变）。"""
    import pandas as pd
    from factor_worker import stage_h5_for_sandbox
    src = tmp_path / "daily_pv_all.h5"
    df = _mk_pv(100)
    df.to_hdf(src, key="data", mode="w")

    job = tmp_path / "job"
    job.mkdir()
    out = stage_h5_for_sandbox(src, job, window_days=30)

    staged = pd.read_hdf(out, key="data")
    assert staged.index.get_level_values("datetime").nunique() == 30
    # 尾部 30 天必须与原数据逐值一致
    tail = df.loc[df.index.get_level_values("datetime").isin(
        df.index.get_level_values("datetime").unique().sort_values()[-30:])]
    pd.testing.assert_frame_equal(staged.sort_index(), tail.sort_index())
    # key / 层级名 / 列名保持（因子代码惯用单 key 读取 + level="instrument"）
    assert list(staged.columns) == list(df.columns)
    assert staged.index.names == ["datetime", "instrument"]


def test_stage_h5_window_larger_than_data_is_noop(tmp_path):
    pytest.importorskip("tables")   # HDF5 依赖：未装则跳过，不误报
    from factor_worker import stage_h5_for_sandbox
    src = tmp_path / "daily_pv_all.h5"
    df = _mk_pv(10)
    df.to_hdf(src, key="data", mode="w")
    job = tmp_path / "job"
    job.mkdir()
    out = stage_h5_for_sandbox(src, job, window_days=400)
    import pandas as pd
    pd.testing.assert_frame_equal(pd.read_hdf(out, key="data").sort_index(),
                                  df.sort_index())


def test_stage_h5_window_zero_keeps_symlink(tmp_path):
    pytest.importorskip("tables")   # HDF5 依赖：未装则跳过，不误报
    """window_days<=0 保持旧的 symlink 全量行为（需要全历史时用）。"""
    from factor_worker import stage_h5_for_sandbox
    src = tmp_path / "daily_pv_all.h5"
    _mk_pv(5).to_hdf(src, key="data", mode="w")
    job = tmp_path / "job"
    job.mkdir()
    out = stage_h5_for_sandbox(src, job, window_days=0)
    assert out.is_symlink()
    assert out.resolve() == src.resolve()


def test_window_days_default_is_bounded():
    """默认窗口必须是有界的（否则又会把全量喂进 2GB 沙箱）。"""
    import factor_worker as fw
    assert fw.WINDOW_DAYS > 0
    assert fw.WINDOW_DAYS <= 1000


# ═══════════════════════════════════════════════════════════════════
# 回测路径的沙箱暂存（_exec_factor_worker）
#
# 上面那组用例覆盖的是 factor_worker.stage_h5_for_sandbox —— 它们全绿，却漏掉了
# 真正的故障：_exec_factor_worker 曾经自己 inline 一句
#     link.symlink_to(Path(data_h5).resolve())
# 不走窗口截取，于是**回测路径永远喂全量**。全量读 + pandas 中间结果顶破沙箱的
# RLIMIT_AS **硬**上限（sandbox.MAX_AS_BYTES，默认 4GiB，虚拟地址空间含 mmap），
# 子进程被 SIGKILL，调用方只看到一句 `exit -9` / "new factor 'x' failed"。
#
# 2026-09-19 生产实测：factor_oos_check 8 秒失败、exit -9、qrun 根本没启动，
# 而同一因子走窗口截取的路径只要 1166MiB 就轻松跑通。两条路径行为不一致就是根因，
# 所以这里钉住"回测路径也必须窗口化"。
# ═══════════════════════════════════════════════════════════════════

def test_exec_factor_worker_stages_a_window_not_the_full_symlink(tmp_path):
    """回测路径必须走窗口截取：暂存文件是真实文件、行数受窗口约束。"""
    pytest.importorskip("tables")
    import pandas as pd
    from factor_miner.factor_backtest import _exec_factor_worker

    src = tmp_path / "daily_pv_all.h5"
    df = _mk_pv(120)                     # 120 个交易日
    df.to_hdf(src, key="data", mode="w")

    job = tmp_path / "job"
    code = (
        "import pandas as pd\n"
        "df = pd.read_hdf('daily_pv.h5', key='data')\n"
        "s = df['$close'].sort_index()\n"
        "f = s.groupby(level='instrument').pct_change(2)\n"
        "f.name = 'factor'\n"
        "f.to_frame().to_hdf('result.h5', key='data')\n"
    )
    name, out_df, err = _exec_factor_worker(
        ("probe", code, str(src), str(job), 120, 30))   # window_days=30

    assert err == "", err
    assert out_df is not None and not out_df.empty
    staged = job / "daily_pv.h5"
    assert not staged.is_symlink(), \
        "回测路径又变回 symlink 全量了 —— 这正是被 SIGKILL 的那条路"
    assert staged.is_file()
    # 窗口之外的日期不应出现（120 天数据只留尾部 30 天）
    kept = out_df.index.get_level_values("datetime").nunique()
    assert 0 < kept <= 30, kept


def test_exec_factor_worker_accepts_window_zero_for_full_history(tmp_path):
    """window_days=0 是显式的"要全历史"，此时才允许 symlink。"""
    pytest.importorskip("tables")
    from factor_miner.factor_backtest import _exec_factor_worker

    src = tmp_path / "daily_pv_all.h5"
    _mk_pv(40).to_hdf(src, key="data", mode="w")
    job = tmp_path / "job"
    code = (
        "import pandas as pd\n"
        "df = pd.read_hdf('daily_pv.h5', key='data')\n"
        "df['$close'].to_frame('factor').to_hdf('result.h5', key='data')\n"
    )
    name, out_df, err = _exec_factor_worker(
        ("probe", code, str(src), str(job), 120, 0))
    assert err == "", err
    assert (job / "daily_pv.h5").is_symlink()


def test_backtest_default_window_is_bounded():
    """回测模块的默认窗口也必须有界，且与 factor_worker 同源同值。"""
    import factor_worker as fw
    from factor_miner import factor_backtest as fb
    assert fb.WINDOW_DAYS > 0
    assert fb.WINDOW_DAYS == fw.WINDOW_DAYS, "两处窗口默认值分叉了"
