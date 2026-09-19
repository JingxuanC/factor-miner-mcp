"""factor_recent_ic 的内存回归：整读一次，而不是两次。

背景（2026-09-18 生产实测）：`daily_pv_all.h5` 是 pandas 的 Fixed 格式，
PyTables 里是 `block0_values` 数组、**没有 table**，所以无法按列或按行部分读
（`columns=['$close']` 直接 TypeError）。原实现因此整读两次全量：

    fac  = _run_factor_df(...)   # 内部 stage_h5_for_sandbox 读第 1 次全量
    pv   = pd.read_hdf(data_h5)  # 第 2 次
    ...

逐行画像：710MiB（第 1 次读）→ 733（第 2 次读起）→ 1127（ret1）→ 1375MiB
（concat）。容器的 mem_limit 是 1536MiB，于是**内核 memcg OOM 杀掉的不是那个
任务，而是整个服务容器**（dmesg: oom_memcg=/system.slice/docker-<id>.scope），
`unless-stopped` 再把它拉起来——表现为客户端的 RemoteProtocolError。

修法：因子值和收盘价都取自**同一次**暂存窗口。这里的两个断言就是那次修复的
契约：只从窗口取数，且窗口取到的 close 与"整读全量再过滤"逐值相同。
"""

import json

import numpy as np
import pandas as pd

import factor_worker as fw

# pytables 不是全环境都装（例如系统 python3.9 就没有 pandas 的 HDF 后端），
# 而本文件的两个用例都要写/读 h5。在 pytest 下**优雅跳过**而不是让默认测试
# 运行报 collection error；作为脚本直接跑时仍会真跑（镜像里有 pytables）。
try:  # pragma: no cover - 取决于环境
    import tables  # noqa: F401
    HAVE_HDF = True
except ImportError:  # pragma: no cover
    HAVE_HDF = False

try:  # pragma: no cover
    import pytest
    if not HAVE_HDF:
        pytest.skip("pytables 未安装：h5 用例需要 pandas 的 HDF 后端",
                    allow_module_level=True)
except ImportError:  # pragma: no cover - 镜像里没有 pytest，由 __main__ 跑
    pytest = None

# 面板要够算 IC：窗口 30 + 因子回看 2 + shift(-1)，再留出 lookback 的交易日。
# 取 3 年 × 8 个标的，既够大又仍是一次秒级的小测试。
DATES = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-01", periods=750, freq="B")]
INSTRUMENTS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
FACTOR_CODE = """import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")
close = df["$close"].sort_index()
del df

factor = -close.groupby(level="instrument").pct_change(2)
factor.name = "factor"

factor.to_frame().to_hdf("result.h5", key="data")
"""


def _panel() -> pd.DataFrame:
    """(datetime, instrument) 两级索引 + $ 前缀列，与 qlib 导出同构。"""
    idx = pd.MultiIndex.from_product([DATES, INSTRUMENTS],
                                     names=["datetime", "instrument"])
    rng = np.random.default_rng(7)
    close = 100 + rng.normal(0, 1, len(idx)).cumsum()
    return pd.DataFrame({
        "$open": close, "$high": close, "$low": close,
        "$close": close, "$volume": 1000.0, "$factor": 1.0,
    }, index=idx)


def _setup(data_root):
    """把模块级路径指向一份合成数据集，返回 (data_dir, restore)。

    手工设置/还原，而不是 pytest fixture：这个仓库的 h5 测试要在带 pytables 的
    解释器里跑（镜像内有），而镜像没装 pytest。所以文件顶部不 import pytest，
    两个用例只收一个普通参数，任何 runner 都能调用。
    """
    d = data_root / "factor_mining"
    d.mkdir(parents=True, exist_ok=True)
    _panel().to_hdf(d / "daily_pv_all.h5", key="data")
    saved = (fw.DATA_DIR, fw.JOBS_ROOT, fw.WINDOW_DAYS)
    fw.DATA_DIR = d
    fw.JOBS_ROOT = data_root / "jobs"
    fw.WINDOW_DAYS = 30
    return d, saved


def _restore(saved):
    fw.DATA_DIR, fw.JOBS_ROOT, fw.WINDOW_DAYS = saved


def test_staged_close_matches_full_panel_window(data_root):
    """窗口里取到的 close 必须与整读全量再按同窗口过滤逐值相同。

    这是这次修复的核心等价性：不再整读第二遍，数值口径就不能变。若哪天有人把
    窗口截取逻辑改得与全量过滤不一致（例如丢了排序、或窗口取错区间），各因子
    的 IC 会**静默**漂移——这个断言就是拦那种漂移的。
    """
    data_dir, saved = _setup(data_root)
    try:
        _factor, close = fw._run_factor_window(FACTOR_CODE, data_dir / "daily_pv_all.h5")
        full = pd.read_hdf(data_dir / "daily_pv_all.h5", key="data").sort_index()
    finally:
        _restore(saved)
    expected = full["$close"].rename("close")
    keep = sorted(set(full.index.get_level_values("datetime")))[-30:]
    expected = expected[expected.index.get_level_values("datetime").isin(list(keep))]

    pd.testing.assert_index_equal(close.index, expected.index)
    pd.testing.assert_series_equal(close, expected, check_names=True)


def test_factor_recent_ic_does_not_read_the_full_file_twice(data_root):
    """整读全量只允许发生一次。

    两次整读正是那次 OOM 的直接原因（710MiB × 2 份 + concat 峰值 1375MiB），
    所以用"全量 h5 被 read_hdf 了几次"来钉住它——比断言某个 RSS 数字稳定得多。
    """
    data_dir, saved = _setup(data_root)
    full_h5 = data_dir / "daily_pv_all.h5"
    real_read_hdf = pd.read_hdf
    reads: list[str] = []

    def counting_read_hdf(path, *a, **kw):
        if str(path) == str(full_h5):
            reads.append(str(path))
        return real_read_hdf(path, *a, **kw)

    fw.pd.read_hdf = counting_read_hdf
    try:
        # lookback 取 30，断言才对得上：工具会把 pair 截到最近 lookback 个交易日，
        # 传 5 就只能有 5 天，与"有足够交易日"的断言自相矛盾。
        out = json.loads(fw.factor_recent_ic(FACTOR_CODE, "count_reads", lookback_days=30))
    finally:
        fw.pd.read_hdf = real_read_hdf
        _restore(saved)

    assert reads == [str(full_h5)], f"全量 h5 被整读了 {len(reads)} 次：{reads}"
    # 顺带确认工具本身仍然正常工作（沙箱跑通、有足够交易日）
    assert out["ok"] is True, out
    assert out["days"] >= 10, out


def test_recent_ic_returns_a_usable_daily_series(data_root):
    """逐日 IC 序列的端到端契约（看板画衰减曲线就靠它）。

    这条只有镜像里能跑（要 pytables 读 h5）。断言的是**调用方真正依赖的**几件事，
    而不是内部实现细节：

      - series 是 list，且每个点都有 date/ic
      - 日期是 ISO（看板要拿它跟行情日期对齐）
      - 日期唯一且升序（时间轴上不能回折）
      - 点数与 days 一致，且这些点的均值等于标量 ic
        （同一个 groupby 的两个视图，分开算就说明有一边错了）
      - n 是当日有效样本数，为正整数
    """
    import datetime as _dt

    data_dir, saved = _setup(data_root)
    try:
        out = json.loads(fw.factor_recent_ic(FACTOR_CODE, "t", lookback_days=60))
        assert out["ok"] is True, out

        series = out["series"]
        assert isinstance(series, list) and series, out
        assert len(series) == out["days"], (len(series), out["days"])

        dates = [p["date"] for p in series]
        assert len(dates) == len(set(dates)), f"日期重复：{dates}"
        assert dates == sorted(dates), "日期必须升序"
        for d in dates:
            _dt.date.fromisoformat(d)  # 非法 ISO 会抛

        for p in series:
            assert isinstance(p["ic"], float), p
            assert isinstance(p["n"], int) and p["n"] > 0, p

        # 逐日序列的均值必须与标量 ic 一致 —— 两者来自同一个 groupby
        mean_of_series = sum(p["ic"] for p in series) / len(series)
        assert abs(mean_of_series - out["ic"]) < 1e-9, (mean_of_series, out["ic"])
    finally:
        _restore(saved)


if __name__ == "__main__":
    # 自带 runner：镜像里没有 pytest，而 pytables 只在镜像里。
    #   docker run --rm -v $PWD:/app -w /app factor-miner-mcp:latest \
    #       python3 test_recent_ic_memory.py
    import pathlib
    import tempfile
    import traceback

    failures = 0
    for fn in (test_staged_close_matches_full_panel_window,
               test_factor_recent_ic_does_not_read_the_full_file_twice,
               test_recent_ic_returns_a_usable_daily_series):
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(pathlib.Path(td))
                print(f"PASS  {fn.__name__}", flush=True)
            except Exception:
                failures += 1
                print(f"FAIL  {fn.__name__}", flush=True)
                traceback.print_exc()
    print("FAILURES:", failures)
    raise SystemExit(1 if failures else 0)
