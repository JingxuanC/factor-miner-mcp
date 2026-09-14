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


# ═══════════ 回归: "局部落库"事故（2026-09-08 起 4 天只剩 2 只票）═══════════
#
# 事故链条：东财 clist 静默截断分页（pz=500 只回 100 行，接口异常时只回 2 行）
# → 旧代码 `if len(diff) < pz: break` 把短页当成"列表到底了"
# → "全市场"退化成 2 只 → 全局日历照常前移，但只有这 2 只拿到新 bar
# → staging 校验只抽"刚更新的那几只"（必然通过），污染被 atomic_swap 正式发布。


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _clist_payload(codes, total):
    return {"data": {"total": total, "diff": [{"f12": c} for c in codes]}}


def test_list_symbols_pages_by_total_not_by_short_page(tmp_path, monkeypatch):
    """第一页被截断成 100 行、但 total=250 → 必须继续翻页，而不是就此收工。"""
    prov = make_provider(tmp_path)
    pages = {}
    for market in ("m:1+t:2,m:1+t:23", "m:0+t:6,m:0+t:80"):
        base = "600" if market.startswith("m:1") else "000"
        pages[market] = [
            [f"{base}{i:03d}" for i in range(100)],          # 页1: 100 行（被截断）
            [f"{base}{100 + i:03d}" for i in range(100)],    # 页2: 100 行
            [f"{base}{200 + i:03d}" for i in range(50)],     # 页3: 50 行 → 到底
        ]
    seen = []

    def fake_em_get(url, params=None, timeout=None):
        market = params["fs"]
        pn = int(params["pn"])
        seen.append((market, pn))
        page = pages[market][pn - 1] if pn <= len(pages[market]) else []
        return _FakeResp(_clist_payload(page, total=250))

    monkeypatch.setattr("factor_miner.fetchers.em_get", fake_em_get)
    monkeypatch.setattr(ud, "MIN_UNIVERSE", 3)

    out = ud.list_symbols(prov)
    # 3 页共 500 只，再并入 all.txt 里接口没覆盖到的 600519
    assert len(out) == 501, "必须按 total 翻满 3 页，而不是停在第一页的 100 行"
    assert ("m:1+t:2,m:1+t:23", 3) in seen and ("m:0+t:6,m:0+t:80", 3) in seen


def test_list_symbols_rejects_truncated_response_and_keeps_all_txt(tmp_path, monkeypatch):
    """接口只回 2 行（成功、无异常）→ 判为退化，以 all.txt 为准，绝不缩成 2 只。"""
    prov = make_provider(tmp_path)  # all.txt: SH600519 / SZ000001
    monkeypatch.setattr("factor_miner.fetchers.em_get",
                        lambda url, params=None, timeout=None: _FakeResp(
                            _clist_payload(["000002", "000003"], total=5500)))
    out = ud.list_symbols(prov)
    assert set(out) == {"000001", "600519"}, "退化时必须回退到 all.txt 存量 universe"
    assert "000002" not in out


def test_list_symbols_union_keeps_old_on_partial_page(tmp_path, monkeypatch):
    """取到的码与 all.txt 求并集：接口漏掉的存量票不会被丢掉。"""
    prov = make_provider(tmp_path)
    monkeypatch.setattr("factor_miner.fetchers.em_get",
                        lambda url, params=None, timeout=None: _FakeResp(
                            _clist_payload(["300999", "000002"], total=2)))
    monkeypatch.setattr(ud, "MIN_UNIVERSE", 2)
    out = ud.list_symbols(prov)
    assert set(out) == {"000001", "600519", "000002", "300999"}


def test_validate_rejects_partial_landing(tmp_path):
    """日历前移、但只有 1/4 只票的 bin 对齐到日历尾 → 校验必须拒绝。"""
    prov = make_provider(tmp_path)
    cal = ud.read_calendar(prov) + [pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27")]
    (prov / "calendars" / "day.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in cal) + "\n")
    (prov / "instruments" / "all.txt").write_text(
        "".join(f"{c}\t2020-01-02\t2026-08-27\n" for c in
                ("SH600519", "SZ000001", "SZ000002", "SZ000003")))
    # 只有 sh600519 真拿到了新 bar（4 行对齐 4 天日历），其余 3 只停在老的一天
    _write_bin(prov / "features" / "sh600519" / "close.day.bin", 0, [1.0, 2.0, 3.0, 4.0])
    for fname in ("sz000001", "sz000002", "sz000003"):
        _write_bin(prov / "features" / fname / "close.day.bin", 0, [1.0])

    with pytest.raises(ud.ValidateError, match="覆盖面过低"):
        ud.validate_staging(prov, "2026-08-27", 4, ["600519"])


def test_validate_accepts_full_landing(tmp_path):
    """全部对齐时正常放行（阈值用 min_instruments 定标，小 universe 单测不受影响）。"""
    prov = make_provider(tmp_path)
    for fname in ("sh600519", "sz000001"):
        _write_bin(prov / "features" / fname / "close.day.bin", 0, [1.0, 2.0])
    ud.validate_staging(prov, "2026-08-25", 2, ["600519", "000001"])
    aligned, checked = ud.bin_alignment(prov, 2)
    assert (aligned, checked) == (2, 2)


def test_run_refuses_degenerate_auto_universe(tmp_path, monkeypatch):
    """自动发现 universe 只有 2 只 → 拒绝更新并保留旧日历（零副作用）。"""
    prov = make_provider(tmp_path)
    monkeypatch.setattr(ud, "list_symbols", lambda p: ["000001", "600519"])
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=_normal_fetcher(), skip_h5=True)
    assert rc == ud.EXIT_NO_SOURCE
    assert ud.read_calendar(prov)[-1] == pd.Timestamp("2026-08-25")
    assert not (tmp_path / "cn_data.__staging__").exists()


def test_run_allows_explicit_small_symbols(tmp_path):
    """调用方显式传 symbols（联调/定向回补）不受 MIN_UNIVERSE 限制。"""
    prov = make_provider(tmp_path)
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=_normal_fetcher(),
                symbols=["000001", "600519"], skip_h5=True)
    assert rc == ud.EXIT_OK
    assert ud.read_calendar(prov)[-1] == pd.Timestamp("2026-08-27")


# ═══════════ 回归: 断点续抓（进程被杀/容器重启后不白费已抓的一小时）═══════════

def test_resume_reuses_existing_csv(tmp_path, monkeypatch):
    """FACTOR_MINER_CSV_RESUME=1 时已存在的 CSV 直接复用，不再重复抓取。"""
    prov = make_provider(tmp_path)
    csv_dir = tmp_path / "csv_resume"
    csv_dir.mkdir()
    # 上一轮已抓好的 600519（新日历 08-26/08-27 两行）。
    # 口径与真实抓取路径一致：close = qfq × 复权连续因子，factor = qfq/raw。
    q = 1297.4 * 0.2432
    (csv_dir / "sh600519.csv").write_text(
        "date,open,high,low,close,volume,factor\n"
        "2026-08-26,%.4f,%.4f,%.4f,%.4f,1000.0,0.2432\n"
        "2026-08-27,%.4f,%.4f,%.4f,%.4f,1000.0,0.2432\n" % ((q,) * 4 + (q,) * 4))
    monkeypatch.setenv("FACTOR_MINER_CSV_DIR", str(csv_dir))
    monkeypatch.setenv("FACTOR_MINER_CSV_RESUME", "1")

    fetched = []
    base = _normal_fetcher()

    class TrackingFetcher(FakeFetcher):
        def fetch_stock(self, code, start, end):
            fetched.append(code)
            if code == "600519":
                raise AssertionError("已复用的票不该再被抓取")
            return base.fetch_stock(code, start, end)

    rc = ud.run(str(prov), str(tmp_path / "mining"),
                fetcher=TrackingFetcher(base.index_days, base.data),
                symbols=["600519", "000001"], skip_h5=True)
    assert rc == ud.EXIT_OK
    assert fetched == ["000001"], "只该抓缺失的票"
    # 复用的 CSV 必须真的进了库：600519 的 close 有 4 行（含新两天）
    close = _read_bin(prov / "features" / "sh600519" / "close.day.bin")
    assert len(close) == 5 and close[-1] == pytest.approx(q, rel=1e-4)


def test_resume_off_ignores_existing_csv(tmp_path):
    """不开 resume 时沿用临时目录：同名 CSV 不会被复用（避免静默用旧数据）。"""
    prov = make_provider(tmp_path)
    csv_dir = tmp_path / "csv_resume"
    csv_dir.mkdir()
    (csv_dir / "sh600519.csv").write_text(
        "date,open,high,low,close,volume,factor\n"
        "2026-08-26,1.0,1.0,1.0,1.0,1.0,1.0\n")
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=_normal_fetcher(),
                symbols=["600519", "000001"], skip_h5=True)
    assert rc == ud.EXIT_OK
    close = _read_bin(prov / "features" / "sh600519" / "close.day.bin")
    # 真实抓取口径：close = qfq × 连续因子 = 1297.4 × 0.2432
    assert close[-1] == pytest.approx(1297.4 * 0.2432, rel=1e-4), \
        "应走真实抓取，而非那个 1.0 的旧 CSV"


# ═══════════ 回归: 抽样不能"猜"市场（北交所目录名是 bj，不是 sz）═══════════

def test_validate_handles_bj_instrument_sample(tmp_path):
    """样本里含北交所票时不能误报缺失——market_of 只会给 sh/sz，必须用真实目录反查。"""
    prov = make_provider(tmp_path)
    (prov / "instruments" / "all.txt").write_text(
        "SH600519\t2020-01-02\t2026-08-25\n"
        "SZ000001\t2020-01-02\t2026-08-25\n"
        "BJ836414\t2020-01-02\t2026-08-25\n")
    for fname in ("sh600519", "sz000001", "bj836414"):
        _write_bin(prov / "features" / fname / "close.day.bin", 0, [1.0, 2.0])
    dirs = ud.feature_instruments(prov)
    assert dirs["836414"] == "bj836414", "必须能用代码反查到北交所目录"
    # 旧逻辑会算出 sz836414 → 误报缺失把落库拦下来
    ud.validate_staging(prov, "2026-08-25", 3, ["836414", "600519", "000001"])


def test_validate_reports_real_missing_bin(tmp_path):
    """真缺失时必须照旧拦截（不能为了修误报而放水）。"""
    prov = make_provider(tmp_path)
    (prov / "instruments" / "all.txt").write_text(
        "SH600519\t2020-01-02\t2026-08-25\nSZ000001\t2020-01-02\t2026-08-25\n")
    for fname in ("sh600519", "sz000001"):
        _write_bin(prov / "features" / fname / "close.day.bin", 0, [1.0, 2.0])
    # 数据集里没有的代码（模拟 all.txt 有条目但目录缺失）
    with pytest.raises(ud.ValidateError, match="close.day.bin 缺失"):
        ud.validate_staging(prov, "2026-08-25", 2, ["600519", "836414"])


def test_explicit_csv_dir_survives_normal_exit(tmp_path, monkeypatch):
    """调用方显式给的抓取目录不能在正常退出时被清掉（否则校验失败会白丢一次全量抓取）。"""
    prov = make_provider(tmp_path)
    csv_dir = tmp_path / "csv_keep"
    csv_dir.mkdir()
    monkeypatch.setenv("FACTOR_MINER_CSV_DIR", str(csv_dir))
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=_normal_fetcher(),
                symbols=["000001", "600519"], skip_h5=True)
    assert rc == ud.EXIT_OK
    kept = sorted(p.name for p in csv_dir.glob("*.csv"))
    assert kept == ["sh600519.csv", "sz000001.csv"], "显式给的目录必须留着"


def test_explicit_csv_dir_survives_validation_failure(tmp_path, monkeypatch):
    """校验失败这条路径更要留：已抓好的 CSV 是重试的唯一本钱。"""
    prov = make_provider(tmp_path)
    csv_dir = tmp_path / "csv_keep"
    csv_dir.mkdir()
    monkeypatch.setenv("FACTOR_MINER_CSV_DIR", str(csv_dir))
    monkeypatch.setattr(ud, "MIN_BIN_COVERAGE", 1.01)  # 强制覆盖面校验失败
    rc = ud.run(str(prov), str(tmp_path / "mining"), fetcher=_normal_fetcher(),
                symbols=["000001", "600519"], skip_h5=True)
    assert rc == ud.EXIT_VALIDATE
    assert sorted(p.name for p in csv_dir.glob("*.csv")) == ["sh600519.csv", "sz000001.csv"]
    assert ud.read_calendar(prov)[-1] == pd.Timestamp("2026-08-25"), "旧数据必须保留"


def test_validate_tolerates_legitimately_stopped_instrument(tmp_path, monkeypatch):
    """退市/长期停牌票的 bin 天然落后于日历尾（自身 end 更早）→ 必须容忍，
    否则一次正确的落库会被误杀（2026-09-14：000413 因历史长期停牌被拦下）。"""
    prov = make_provider(tmp_path)
    cal = ud.read_calendar(prov) + [pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27")]
    (prov / "calendars" / "day.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in cal) + "\n")
    # 600519 更新到新日历尾；000413 自身 end 停在 08-25（已退市/停牌）
    (prov / "instruments" / "all.txt").write_text(
        "SH600519\t2020-01-02\t2026-08-27\nSZ000413\t2020-01-02\t2026-08-25\n")
    _write_bin(prov / "features" / "sh600519" / "close.day.bin", 0, [1.0, 2.0, 3.0, 4.0])
    _write_bin(prov / "features" / "sz000413" / "close.day.bin", 0, [1.0, 2.0])   # 只有 2 行
    # 覆盖面**比例**是另一道守卫（真实数据集里落后票只占 ~5.6%）；这里样本只有 2 只，
    # 放宽比例阈值，专测"逐只抽样要容忍合法落后"这一条。
    monkeypatch.setattr(ud, "MIN_BIN_COVERAGE", 0.3)   # 1/3 对齐 ≥ 0.3
    ud.validate_staging(prov, "2026-08-27", 2, ["600519", "000413"])   # 不该抛


def test_validate_still_catches_active_instrument_shortfall(tmp_path):
    """活跃票（自身 end 就是日历尾）行数不足 → 照旧拦截，不能为了修误报放水。"""
    prov = make_provider(tmp_path)
    cal = ud.read_calendar(prov) + [pd.Timestamp("2026-08-26"), pd.Timestamp("2026-08-27")]
    (prov / "calendars" / "day.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in cal) + "\n")
    (prov / "instruments" / "all.txt").write_text(
        "SH600519\t2020-01-02\t2026-08-27\nSZ000001\t2020-01-02\t2026-08-27\n")
    _write_bin(prov / "features" / "sh600519" / "close.day.bin", 0, [1.0, 2.0, 3.0, 4.0])
    _write_bin(prov / "features" / "sz000001" / "close.day.bin", 0, [1.0])   # 说 08-27 却只 1 行
    with pytest.raises(ud.ValidateError, match="行数错位"):
        ud.validate_staging(prov, "2026-08-27", 2, ["600519", "000001"])
