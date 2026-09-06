"""update_data 单测：mock 数据源，覆盖 staging 成功切换 / 校验失败保留 / 数据源全挂三条路径。

不依赖 qlib / mootdx / 网络：fake provider 目录手工构造最小 bin 树，
DumpDataUpdate 用真实逻辑跑（只依赖 numpy/pandas）。
"""

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_miner import update_data as ud

CAL = ["2026-08-24", "2026-08-25"]  # 老日历，d0 = 08-25
NEW_DAYS = ["2026-08-26", "2026-08-27"]


def _write_bin(path: Path, start_idx: int, values: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.hstack([[start_idx], np.array(values, dtype=np.float32)])
    arr.astype("<f").tofile(str(path))


def make_provider(root: Path) -> Path:
    """最小 cn_data: 2 只存量股（sh600519/sz000001），各 2 天数据。"""
    prov = root / "cn_data"
    (prov / "calendars").mkdir(parents=True)
    (prov / "calendars" / "day.txt").write_text("\n".join(CAL) + "\n")
    (prov / "instruments").mkdir()
    (prov / "instruments" / "all.txt").write_text(
        "SH600519\t2020-01-02\t2026-08-25\nSZ000001\t2020-01-02\t2026-08-25\n"
    )
    # sh600519: close 317.14, factor 0.2432（约定: close=前复权价, factor=qfq/raw）
    base = {
        "sh600519": {"close": [316.0, 317.14], "factor": [0.2432, 0.2432]},
        "sz000001": {"close": [10.0, 10.1], "factor": [0.8, 0.8]},
    }
    for fname, fields in base.items():
        for field in ud.DUMP_FIELDS:
            vals = fields.get(field, [1.0, 1.0])
            _write_bin(prov / "features" / fname / f"{field}.day.bin", 0, vals)
    return prov


def _df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _klines(code: str, days: list, close_map: dict, vol: float = 1000.0) -> pd.DataFrame:
    """days 内每天一行；close_map 之外的日子 close=NaN 直接不生成（模拟停牌）。"""
    rows = []
    for d in days:
        if d in close_map:
            c = close_map[d]
            rows.append({"date": d, "open": c, "high": c, "low": c, "close": c, "volume": vol})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


class FakeFetcher:
    """data: code -> (raw_df, qfq_df)；index_days: 指数日历。"""

    name = "fake"
    last_source = "fake"

    def __init__(self, index_days: list, data: dict, fail_all: bool = False):
        self.index_days = index_days
        self.data = data
        self.fail_all = fail_all

    def alive(self) -> bool:
        return not self.fail_all

    def fetch_index_days(self, start, end):
        if self.fail_all:
            raise ud.FetchError("fake: all sources down")
        return self.index_days

    def fetch_stock(self, code, start, end):
        if self.fail_all or code not in self.data:
            raise ud.FetchError(f"fake: {code} fetch failed")
        return self.data[code]


def _read_bin(path: Path):
    return np.fromfile(str(path), dtype="<f")


def _normal_fetcher():
    days = CAL + NEW_DAYS
    # 无除权: qfq == raw（近日期 qfq 基准 == 原始价）
    data = {
        # sh600519 raw close 08-25 = 317.14/0.2432 ≈ 1304.0
        "600519": (
            _klines("600519", days, {"2026-08-25": 1304.0, "2026-08-26": 1300.0, "2026-08-27": 1297.4}),
            _klines("600519", days, {"2026-08-25": 1304.0, "2026-08-26": 1300.0, "2026-08-27": 1297.4}),
        ),
        "000001": (
            _klines("000001", days, {"2026-08-25": 12.625, "2026-08-26": 12.6, "2026-08-27": 12.7}),
            _klines("000001", days, {"2026-08-25": 12.625, "2026-08-26": 12.6, "2026-08-27": 12.7}),
        ),
    }
    return FakeFetcher(days, data)


def test_success_staging_swap(tmp_path):
    prov = make_provider(tmp_path)
    out = tmp_path / "mining"
    rc = ud.run(str(prov), str(out), fetcher=_normal_fetcher(), symbols=["000001", "600519"], skip_h5=True)
    assert rc == ud.EXIT_OK

    # 日历扩展到 08-27，staging/prev 已清理
    assert ud.read_calendar(prov)[-1] == pd.Timestamp("2026-08-27")
    assert not (tmp_path / "cn_data.__staging__").exists()
    assert not (tmp_path / "cn_data.__prev__").exists()

    # bin 行数与新日历对齐：4 天数据 + 1 个头
    arr = _read_bin(prov / "features" / "sh600519" / "close.day.bin")
    assert len(arr) == 1 + 4 and arr[0] == 0.0
    # scale = 317.14/1304 = 0.24320...；新 close = qfq×scale
    scale = 317.14 / 1304.0
    assert arr[-1] == pytest.approx(1297.4 * scale, rel=1e-5)
    # factor 连续（无除权 → 新 factor == 老 factor）
    fac = _read_bin(prov / "features" / "sh600519" / "factor.day.bin")
    assert fac[-1] == pytest.approx(0.2432, rel=1e-4)
    # volume = raw_vol / factor
    vol = _read_bin(prov / "features" / "sh600519" / "volume.day.bin")
    assert vol[-1] == pytest.approx(1000.0 / 0.2432, rel=1e-3)

    # instruments end 推进
    inst = (prov / "instruments" / "all.txt").read_text()
    assert "SH600519\t2020-01-02\t2026-08-27" in inst

    # manifest 字段完整（skip_h5 → h5_sha256 为 null）
    m = json.loads((out / "manifest.json").read_text())
    assert m["last_trading_day"] == "2026-08-27"
    assert m["instrument_count"] == 2
    assert m["updated_symbols"] == 2
    assert m["h5_sha256"] is None
    assert m["version"] and m["generated_at"]


def test_scale_continuity_across_split(tmp_path):
    """除权（1拆2，08-27 除权日）场景: qfq 基准变化，scale 保证 close 边界连续。"""
    prov = make_provider(tmp_path)
    days = CAL + NEW_DAYS
    # raw: 08-25=1304, 08-26=1300(拆前), 08-27=650(拆后)
    # qfq(新基准): 08-25=652, 08-26=650, 08-27=650 → scale = 317.14/652 ≈ 0.4864
    data = {
        "600519": (
            _klines("600519", days, {"2026-08-25": 1304.0, "2026-08-26": 1300.0, "2026-08-27": 650.0}),
            _klines("600519", days, {"2026-08-25": 652.0, "2026-08-26": 650.0, "2026-08-27": 650.0}),
        ),
        "000001": (
            _klines("000001", days, {"2026-08-25": 12.625, "2026-08-26": 12.6, "2026-08-27": 12.7}),
            _klines("000001", days, {"2026-08-25": 12.625, "2026-08-26": 12.6, "2026-08-27": 12.7}),
        ),
    }
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=FakeFetcher(days, data),
                         symbols=["000001", "600519"], skip_h5=True)
    assert rc == ud.EXIT_OK
    close = _read_bin(prov / "features" / "sh600519" / "close.day.bin")
    scale = 317.14 / 652.0
    assert close[-2] == pytest.approx(650.0 * scale, rel=1e-5)  # 08-26
    assert close[-1] == pytest.approx(650.0 * scale, rel=1e-5)  # 08-27
    # 边界收益率 == qfq 序列的收益率（650/652-1）；若 scale 错了会错成 -50%
    assert close[-1] / close[-3] - 1 == pytest.approx(650.0 / 652.0 - 1, abs=1e-4)
    # factor 跳变到约 2 倍（真实反映除权）
    fac = _read_bin(prov / "features" / "sh600519" / "factor.day.bin")
    assert fac[-1] == pytest.approx(650.0 * scale / 650.0, rel=1e-5)


def test_validation_failure_keeps_old(tmp_path):
    """指数已到 08-28 但个股数据只到 08-27 → 校验失败，旧数据原样保留。"""
    prov = make_provider(tmp_path)
    days = CAL + NEW_DAYS
    fetcher = FakeFetcher(CAL + NEW_DAYS + ["2026-08-28"], _normal_fetcher().data)
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=fetcher,
                         symbols=["000001", "600519"], skip_h5=True)
    assert rc == ud.EXIT_VALIDATE
    # 旧日历/bin/manifest 均未变
    assert ud.read_calendar(prov)[-1] == pd.Timestamp("2026-08-25")
    assert len(_read_bin(prov / "features" / "sh600519" / "close.day.bin")) == 3
    assert not (tmp_path / "mining" / "manifest.json").exists()
    assert not (tmp_path / "cn_data.__staging__").exists()


def test_all_sources_down_keeps_old(tmp_path):
    prov = make_provider(tmp_path)
    fetcher = FakeFetcher([], {}, fail_all=True)
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=fetcher,
                         symbols=["000001", "600519"], skip_h5=True)
    assert rc == ud.EXIT_NO_SOURCE
    assert ud.read_calendar(prov)[-1] == pd.Timestamp("2026-08-25")
    assert not (tmp_path / "cn_data.__staging__").exists()


def test_new_stock_and_suspension(tmp_path):
    """新上市股只写真值行；停牌日补 NaN 行保持日历对齐。"""
    prov = make_provider(tmp_path)
    days = CAL + NEW_DAYS
    data = dict(_normal_fetcher().data)
    # 600519 在 08-26 停牌（raw/qfq 均无该行）
    data["600519"] = (
        _klines("600519", days, {"2026-08-25": 1304.0, "2026-08-27": 1297.4}),
        _klines("600519", days, {"2026-08-25": 1304.0, "2026-08-27": 1297.4}),
    )
    # 新上市股 300999：08-26 才有数据
    data["300999"] = (
        _klines("300999", days, {"2026-08-26": 20.0, "2026-08-27": 21.0}),
        _klines("300999", days, {"2026-08-26": 20.0, "2026-08-27": 21.0}),
    )
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=FakeFetcher(days, data),
                symbols=["000001", "600519", "300999"], skip_h5=True)
    assert rc == ud.EXIT_OK
    # 600519: 4 行，08-26 为 NaN
    close = _read_bin(prov / "features" / "sh600519" / "close.day.bin")
    assert len(close) == 5 and np.isnan(close[-2]) and not np.isnan(close[-1])
    # 300999: 新 bin，start_idx = 2（08-26 在新日历中的下标），2 天数据
    new_close = _read_bin(prov / "features" / "sz300999" / "close.day.bin")
    assert new_close[0] == 2.0 and len(new_close) == 3
    assert new_close[-1] == pytest.approx(21.0, rel=1e-5)
    inst = (prov / "instruments" / "all.txt").read_text()
    assert "SZ300999\t2026-08-26\t2026-08-27" in inst


def test_compute_scale_no_overlap():
    assert ud.compute_scale({}, pd.DataFrame(), pd.Timestamp("2026-08-25")) == 1.0


def test_backfill_missed_days(tmp_path):
    """instrument end 落后于日历（上次运行该股抓取失败）→ 本轮按 end 回补缺口。"""
    prov = make_provider(tmp_path)
    # sz000001 的 end 停在 08-24（日历已 08-25），bin 只有 08-24 一行（真实约定: bin 末行==end）
    (prov / "instruments" / "all.txt").write_text(
        "SH600519\t2020-01-02\t2026-08-25\nSZ000001\t2020-01-02\t2026-08-24\n"
    )
    for field in ud.DUMP_FIELDS:
        vals = {"close": [10.0], "factor": [0.8]}.get(field, [1.0])
        _write_bin(prov / "features" / "sz000001" / f"{field}.day.bin", 0, vals)
    days = CAL + NEW_DAYS  # CAL 已含 08-24（重叠日）
    data = dict(_normal_fetcher().data)
    closes = {"2026-08-24": 12.5, "2026-08-25": 12.625, "2026-08-26": 12.6, "2026-08-27": 12.7}
    data["000001"] = (_klines("000001", days, closes), _klines("000001", days, closes))
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=FakeFetcher(days, data),
                symbols=["000001", "600519"], skip_h5=True)
    assert rc == ud.EXIT_OK
    close = _read_bin(prov / "features" / "sz000001" / "close.day.bin")
    scale = 10.0 / 12.5  # 重叠日 08-24: 老 close / qfq
    assert len(close) == 1 + 4  # 08-24 + 回补 08-25 + 新增 08-26/27，与新日历对齐
    assert close[-3] == pytest.approx(12.625 * scale, rel=1e-5)  # 08-25 回补
    assert close[-1] == pytest.approx(12.7 * scale, rel=1e-5)
    inst = (prov / "instruments" / "all.txt").read_text()
    assert "SZ000001\t2020-01-02\t2026-08-27" in inst


def _write_bin_roundtrip_check():
    """防回归: 确认 fixture 的 bin 写入格式与 qlib 约定一致（float32 小端, 头=start_idx）。"""
    arr = np.hstack([[0], [1.0, 2.0]]).astype("<f")
    assert struct.unpack("<f", arr.tobytes()[:4])[0] == 0.0
