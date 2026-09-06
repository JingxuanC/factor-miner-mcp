#!/usr/bin/env python3
"""update_data.py — qlib cn_data 每日增量更新管线（Slice 1, docs/DESIGN-QLIB-DATA-SERVICE.md）

流程:
  1. 交易日历: 拉上证指数日 K，得到 cn_data 日历之后的新交易日（周末/节假日自然为空）
  2. 抓取: 全市场 A 股日线 raw + qfq 双路（腾讯 ifzq 主用 → mootdx → 东财，熔断切换）
  3. 复权对齐: 以老 bin 尾部与 qfq 重叠日算 scale，保证 close/factor 在边界日连续
  4. 落地: 临时 CSV → DumpDataUpdate 写入 staging 副本（cn_data.__staging__）
  5. 校验: staging 日历最新日 == 指数最新交易日；instruments 不回退；抽样 bin 行数对齐
  6. 切换: os.rename 原子换目录（旧目录保留到校验过后再删）
  7. 重建: gen_data.gen_full 重生成 daily_pv_all.h5（临时文件 + os.replace）
  8. manifest: 写 <out_dir>/manifest.json（含 h5 sha256）

数据源全挂 / 大面积失败 / 校验不过 → 保留旧数据，退出码非 0。

用法:
  python3 -m factor_miner.update_data [--provider-uri ~/.qlib/qlib_data/cn_data]
      [--out-dir data/factor_mining] [--source auto|tencent|mootdx|eastmoney]
      [--limit N] [--skip-h5] [--min-interval 0.15]
"""
import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor_miner.qlib_dump_bin import DumpDataUpdate

log = logging.getLogger("update_data")

# 存量数据约定（实测推断 + 验证，见设计文档）:
#   $close/$open/$high/$low  = 前复权价（以快照生成日为基准）
#   $factor                  = 前复权价 / 原始价
#   $volume                  = 原始成交量(手) / factor
#   $amount/$vwap/$adjclose/$change 本管线不更新（停留在老日历末端，qlib 读取自动截尾）
DUMP_FIELDS = ["open", "high", "low", "close", "volume", "factor"]

A_SHARE_RE = re.compile(r"^(60[0-9]{4}|68[0-9]{4}|00[0-9]{4}|30[0-9]{4})$")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NO_SOURCE = 3     # 数据源全挂 / 大面积失败
EXIT_VALIDATE = 4      # 校验失败


class FetchError(Exception):
    """单源抓取失败（触发熔断计数）。"""


# ═══════════════ 数据源 ═══════════════

def market_of(code: str) -> str:
    return "sh" if code[0] in "69" else "sz"


def _bars_df(rows) -> pd.DataFrame:
    """统一内部 K 线格式: index=Timestamp(date), 列 open/high/low/close/volume(手, 原始)。"""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


class TencentKline:
    """腾讯 ifzq newfqkline 日 K。实测容器内 ~0.3s/req 稳定，与 internal/gateway
    实时行情同源。行格式: [date, open, close, high, low, volume(手), ...]（可取前 6 列）。
    注意: 2026-08 实测老端点 fqkline/get 会被 WAF 501，newfqkline 正常；
    返回条数按 count 给（可能多于 start..end 窗口），调用方按日期自行截取。"""

    name = "tencent"
    URL = "https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"

    def __init__(self, min_interval: float = 0.15):
        import requests  # noqa: PLC0415

        self._min_interval = min_interval
        self._last = 0.0
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})

    def _get(self, params) -> dict:
        wait = self._min_interval - (time.time() - self._last) + random.uniform(0.02, 0.08)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self._s.get(self.URL, params=params, timeout=15)
            return r.json()
        finally:
            self._last = time.time()

    def _klines(self, symbol: str, start: str, end: str, fq: str) -> pd.DataFrame:
        key = "qfqday" if fq == "qfq" else "day"
        d = self._get({"param": f"{symbol},day,{start},{end},320,{fq}"})
        node = (d.get("data") or {}).get(symbol) or {}
        rows = node.get(key) or node.get("day") or []
        if not rows:
            raise FetchError(f"tencent: {symbol} empty klines")
        return _bars_df([
            {"date": r[0], "open": r[1], "high": r[3], "low": r[4], "close": r[2], "volume": r[5]}
            for r in rows
        ])

    def fetch_stock(self, code: str, start: str, end: str):
        sym = market_of(code) + code
        return self._klines(sym, start, end, ""), self._klines(sym, start, end, "qfq")

    def fetch_index_days(self, start: str, end: str) -> list:
        df = self._klines("sh000001", start, end, "")
        return [d.strftime("%Y-%m-%d") for d in df.index]


class MootdxKline:
    """通达信 mootdx（复用 server.tdx_client 的服务器探测）。
    注意: 2026-08 实测当前网络 TCP 能连但协议层无数据返回，基本必熔断；
    保留作为国内网络环境下的首选（批量 TCP 抓取远快于 HTTP）。"""

    name = "mootdx"

    def __init__(self):
        from server import tdx_client  # noqa: PLC0415 — 延迟导入，单测/无 mootdx 环境可 import 本模块

        self._client = tdx_client()

    @staticmethod
    def _to_df(df) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise FetchError("mootdx: empty bars")
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "volume" not in df.columns and "vol" in df.columns:
            df["volume"] = df["vol"]
        idx = pd.to_datetime(df["datetime"]) if "datetime" in df.columns else pd.to_datetime(df.index)
        out = df[["open", "high", "low", "close", "volume"]].astype(float)
        out.index = idx
        return out.sort_index()

    def _klines(self, code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
        kwargs = {"adjust": adjust} if adjust else {}
        df = self._client.bars(symbol=code, frequency=9, start=0, offset=60, **kwargs)
        df = self._to_df(df)
        return df.loc[start:end]

    def fetch_stock(self, code: str, start: str, end: str):
        raw = self._klines(code, start, end, "")
        qfq = self._klines(code, start, end, "qfq")
        if qfq.empty:
            raise FetchError(f"mootdx: {code} empty qfq")
        return raw, qfq

    def fetch_index_days(self, start: str, end: str) -> list:
        df = self._klines("000001", start, end, "")  # 上证指数
        return [d.strftime("%Y-%m-%d") for d in df.index]


class EastmoneyKline:
    """东财 push2his 日 K 兜底（复用 server.em_get 的节流+重试，~1.3s/req）。
    全市场 2 请求/股约 4h，只在主源全挂时使用。IP 易被限流，切勿调低节流。"""

    name = "eastmoney"
    URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    def _klines(self, code: str, start: str, end: str, fqt: str) -> pd.DataFrame:
        from server import em_get  # noqa: PLC0415 — 延迟导入

        secid = ("1." if code[0] in "69" else "0.") + code
        r = em_get(self.URL, params={
            "secid": secid, "klt": "101", "fqt": fqt,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "beg": start.replace("-", ""), "end": end.replace("-", ""),
        }, timeout=15)
        data = (r.json() or {}).get("data")
        if not data or not data.get("klines"):
            raise FetchError(f"eastmoney: {secid} empty klines")
        # f51..f57: date,open,close,high,low,volume(手),amount(元)
        return _bars_df([
            dict(zip(["date", "open", "close", "high", "low", "volume"], k.split(",")[:6]))
            for k in data["klines"]
        ])

    def fetch_stock(self, code: str, start: str, end: str):
        return self._klines(code, start, end, "0"), self._klines(code, start, end, "1")

    def fetch_index_days(self, start: str, end: str) -> list:
        df = self._klines("000001", start, end, "0")
        return [d.strftime("%Y-%m-%d") for d in df.index]


class ChainFetcher:
    """多源链式抓取: 单源连续失败 _BREAK_AFTER 次即熔断本次运行；全源熔断抛 FetchError。"""

    _BREAK_AFTER = 3

    def __init__(self, sources: list):
        self._sources = sources
        self._fails = {s.name: 0 for s in sources}
        self._dead = set()
        self.last_source = ""

    def _alive(self):
        return [s for s in self._sources if s.name not in self._dead]

    def alive(self) -> bool:
        return bool(self._alive())

    def _call(self, method: str, *args):
        errs = []
        for s in self._alive():
            try:
                out = getattr(s, method)(*args)
                self._fails[s.name] = 0
                self.last_source = s.name
                return out
            except Exception as e:  # noqa: BLE001 — 任何单源失败都尝试下一源
                errs.append(f"{s.name}: {e}")
                self._fails[s.name] += 1
                if self._fails[s.name] >= self._BREAK_AFTER:
                    self._dead.add(s.name)
                    log.warning("数据源熔断: %s (%s)", s.name, e)
        raise FetchError("all sources failed: " + "; ".join(errs))

    def fetch_stock(self, code: str, start: str, end: str):
        return self._call("fetch_stock", code, start, end)

    def fetch_index_days(self, start: str, end: str) -> list:
        return self._call("fetch_index_days", start, end)


def build_fetcher(source: str, min_interval: float) -> ChainFetcher:
    """source=auto: mootdx(国内网络最快) → tencent → eastmoney(最稳但最慢)。"""
    def _make(name: str):
        if name == "tencent":
            return TencentKline(min_interval=min_interval)
        if name == "mootdx":
            return MootdxKline()
        if name == "eastmoney":
            return EastmoneyKline()
        raise ValueError(f"unknown source: {name}")

    names = ["mootdx", "tencent", "eastmoney"] if source == "auto" else [source]
    sources = []
    for n in names:
        try:
            sources.append(_make(n))
        except Exception as e:  # noqa: BLE001 — 构造失败(如 mootdx 无服务器)直接跳过
            log.warning("数据源不可用, 跳过: %s (%s)", n, e)
    if not sources:
        raise FetchError("no data source available")
    return ChainFetcher(sources)


def list_symbols(provider_dir: Path) -> list:
    """全市场 A 股代码。优先东财 clist 实时列表（含新上市，分页），失败回退
    instruments/all.txt（会漏掉快照之后的新 IPO，下次成功时补上）。
    BJ 股票腾讯/东财日 K 覆盖不全，v1 不含。"""
    codes = []
    try:
        from server import em_get  # noqa: PLC0415

        for mkt in ("m:1+t:2,m:1+t:23", "m:0+t:6,m:0+t:80"):  # 沪A + 深A（含科创/创业）
            pn, pz = 1, 500
            while True:
                r = em_get("https://82.push2.eastmoney.com/api/qt/clist/get", params={
                    "pn": str(pn), "pz": str(pz), "po": "1", "np": "1",
                    "fltt": "2", "invt": "2", "fs": mkt, "fields": "f12",
                }, timeout=15)
                diff = ((r.json() or {}).get("data") or {}).get("diff") or []
                codes += [str(d["f12"]) for d in diff if A_SHARE_RE.match(str(d.get("f12", "")))]
                if len(diff) < pz:
                    break
                pn += 1
    except Exception as e:  # noqa: BLE001
        log.warning("东财 clist 获取股票列表失败, 回退 instruments/all.txt: %s", e)
        codes = []
    if not codes:
        inst = provider_dir / "instruments" / "all.txt"
        for line in inst.read_text().splitlines():
            sym = line.split("\t")[0].strip()
            m = re.match(r"^(SH|SZ)(\d{6})$", sym)
            if m and A_SHARE_RE.match(m.group(2)):
                codes.append(m.group(2))
    return sorted(set(codes))


# ═══════════════ 老 bin 读取 / 复权对齐 ═══════════════

def read_calendar(provider_dir: Path) -> list:
    cal_path = provider_dir / "calendars" / "day.txt"
    return [pd.Timestamp(x) for x in cal_path.read_text().split()]


def read_bin_tail(provider_dir: Path, code: str, field: str, calendar: list,
                  end_date: pd.Timestamp, n: int = 40) -> dict:
    """读 features/<fname>/<field>.day.bin 尾部 n 个值 → {Timestamp: float}（跳过 NaN）。
    bin 格式: 首个 float32 为 start_idx（在 calendar 中的起始下标），其后逐日 float32。
    注意 bin 末行对应该 instrument 自己的 end_date（停牌/退市股票早于日历末端），
    不能按日历末端对齐。"""
    import bisect  # noqa: PLC0415

    fname = (market_of(code) + code).lower()
    bin_path = provider_dir / "features" / fname / f"{field}.day.bin"
    if not bin_path.exists():
        return {}
    arr = np.fromfile(bin_path, dtype="<f")
    if len(arr) < 2:
        return {}
    start_idx = int(arr[0])
    values = arr[1:][-n:]
    end_idx = bisect.bisect_right(calendar, end_date) - 1
    dates = calendar[end_idx - len(values) + 1: end_idx + 1]
    return {d: float(v) for d, v in zip(dates, values) if not np.isnan(v)}


def compute_scale(old_close: dict, qfq: pd.DataFrame, boundary: pd.Timestamp) -> float:
    """复权连续性: scale = 老close(dc) / 新qfq(dc)，dc 为 ≤boundary 的重叠日中最近者。
    老 close 是快照基准日前复权价，新 qfq 是今日基准日前复权价；中间发生过分红除权
    的股票两个基准不同，用 scale 把新数据折回老基准，保证 close/factor 在边界连续。
    无重叠日（长期停牌后复牌等）→ 1.0 并告警（除权会造成跳变，罕见，记录在案）。"""
    common = [d for d in qfq.index if d <= boundary and d in old_close]
    if not common:
        return 1.0
    dc = max(common)
    q = float(qfq.loc[dc, "close"])
    if not np.isfinite(q) or q <= 0:
        return 1.0
    return old_close[dc] / q


def build_rows(code: str, new_days: list, raw: pd.DataFrame, qfq: pd.DataFrame, scale: float) -> pd.DataFrame:
    """生成一只股票的增量行（index=new_days, 列=DUMP_FIELDS）。停牌日行留 NaN。"""
    idx = pd.DatetimeIndex(new_days)
    raw = raw.reindex(idx)
    qfq = qfq.reindex(idx)
    out = pd.DataFrame(index=idx, columns=DUMP_FIELDS, dtype=float)
    for f in ("open", "high", "low", "close"):
        out[f] = qfq[f] * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        out["factor"] = np.where(raw["close"] > 0, out["close"] / raw["close"], scale)
        out["volume"] = np.where(out["factor"] > 0, raw["volume"] / out["factor"], np.nan)
    out[out["close"].isna()] = np.nan  # 停牌日整行 NaN
    return out


# ═══════════════ staging / 校验 / 切换 ═══════════════

def copy_to_staging(provider_dir: Path, staging: Path):
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(provider_dir, staging)


def validate_staging(staging: Path, expected_last_day: str, min_instruments: int, sample_codes: list):
    """校验 staging 副本。任何一项不过抛 ValidateError。"""
    cal = read_calendar(staging)
    last = cal[-1].strftime("%Y-%m-%d")
    if last != expected_last_day:
        raise ValidateError(f"日历最新日 {last} != 指数最近交易日 {expected_last_day}")
    n_inst = sum(1 for _ in open(staging / "instruments" / "all.txt"))
    if n_inst < min_instruments:
        raise ValidateError(f"instruments 回退: {n_inst} < {min_instruments}")
    for code in sample_codes:
        fname = (market_of(code) + code).lower()
        bin_path = staging / "features" / fname / "close.day.bin"
        if not bin_path.exists():
            raise ValidateError(f"{code} close.day.bin 缺失")
        arr = np.fromfile(bin_path, dtype="<f")
        want = len(cal) - int(arr[0])
        if len(arr) - 1 != want:
            raise ValidateError(f"{code} close bin 行数错位: {len(arr) - 1} != calendar 对齐值 {want}")


class ValidateError(Exception):
    pass


def atomic_swap(provider_dir: Path, staging: Path, prev: Path):
    """同文件系统目录 rename 是原子的: 读取方（挖掘/回测）任一时刻看到完整旧树或新树。"""
    if prev.exists():
        shutil.rmtree(prev)
    os.rename(provider_dir, prev)
    try:
        os.rename(staging, provider_dir)
    except Exception:
        os.rename(prev, provider_dir)  # 回滚
        raise
    shutil.rmtree(prev)


# ═══════════════ h5 重建 + manifest ═══════════════

def regen_h5(provider_dir: Path, out_dir: Path) -> tuple:
    """复用 gen_data.gen_full 全量重建 daily_pv_all.h5（临时文件 + os.replace 原子发布）。
    返回 (h5_path, elapsed_sec, sha256)。"""
    from factor_miner import gen_data  # noqa: PLC0415 — 延迟导入（需要 pyqlib）

    t0 = time.time()
    D = gen_data._init_qlib(str(provider_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    # 临时目录必须落在 out_dir 同文件系统: 容器里 /tmp 与 /data 是不同挂载点,
    # 跨设备 os.replace 会报 EXDEV (Invalid cross-device link)
    with tempfile.TemporaryDirectory(prefix="h5_", dir=out_dir) as tmp:
        tmp_path = gen_data.gen_full(D, Path(tmp))
        target = out_dir / "daily_pv_all.h5"
        os.replace(tmp_path, target)
    elapsed = time.time() - t0
    # TODO(perf): 全量重建实测耗时若 >10min，应改增量追加（只重算新增交易日再 concat）
    h = hashlib.sha256()
    with open(target, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return target, elapsed, h.hexdigest()


def write_manifest(out_dir: Path, info: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".manifest.json.tmp"
    tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    os.replace(tmp, out_dir / "manifest.json")


# ═══════════════ 主流程 ═══════════════

def run(provider_uri: str, out_dir: str, fetcher=None, source: str = "auto",
        symbols: list = None, limit: int = None, skip_h5: bool = False,
        min_interval: float = 0.15, max_workers: int = 8, force: bool = False) -> int:
    t_run = time.time()
    provider_dir = Path(provider_uri).expanduser()
    out = Path(out_dir)
    staging = provider_dir.parent / (provider_dir.name + ".__staging__")
    prev = provider_dir.parent / (provider_dir.name + ".__prev__")

    calendar = read_calendar(provider_dir)
    d0 = calendar[-1]
    today = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d")
    start = (d0 - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    log.info("cn_data 当前最新: %s, 抓取窗口 %s → %s", d0.strftime("%Y-%m-%d"), start, today)

    if fetcher is None:
        try:
            fetcher = build_fetcher(source, min_interval)
        except FetchError as e:
            log.error("数据源全部不可用: %s", e)
            return EXIT_NO_SOURCE

    # 1. 新交易日历（指数日 K 为准；周末/节假日自然无新日期）
    try:
        index_days = fetcher.fetch_index_days(start, today)
    except FetchError as e:
        log.error("指数日历抓取失败（数据源全挂）: %s", e)
        return EXIT_NO_SOURCE
    new_days = [d for d in index_days if pd.Timestamp(d) > d0]
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    if new_days and new_days[-1] == now.strftime("%Y-%m-%d") and now.hour * 60 + now.minute < 15 * 60 + 15:
        # 盘中触发：当日 bar 未收盘（不完整），留给 15:30 调度/收盘后手动触发
        log.info("当日 %s 未收盘（%s），剔除当日 bar", new_days[-1], now.strftime("%H:%M"))
        new_days = new_days[:-1]
    if not new_days and not force:
        log.info("无新交易日（最新 %s），无需更新", d0.strftime("%Y-%m-%d"))
        return EXIT_OK
    log.info("新交易日: %s", new_days)

    # 2. 股票列表 + 逐股抓取
    if symbols is None:
        symbols = list_symbols(provider_dir)
    if limit:
        symbols = symbols[:limit]
    log.info("待更新股票: %d 只 (source=%s)", len(symbols), source)

    inst_path = provider_dir / "instruments" / "all.txt"
    old_end = {}  # FNAME(大写) -> 该 instrument 已落地的最后日期
    for line in inst_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            old_end[parts[0].strip().upper()] = pd.Timestamp(parts[2].strip())
    # 回补窗口: 日历尾部 + 新交易日。个股按自己的 instrument end 取增量日期，
    # 昨日抓取失败/漏更的股票今天自动补齐缺口（而不是永远留 NaN 洞）。
    backfill_days = calendar[-30:] + [pd.Timestamp(d) for d in new_days]

    csv_dir = Path(tempfile.mkdtemp(prefix="qlib_csv_"))
    updated, new_syms, failed, last_day_hits = [], [], 0, 0
    try:
        for i, code in enumerate(symbols):
            try:
                raw, qfq = fetcher.fetch_stock(code, start, today)
            except FetchError as e:
                failed += 1
                if not fetcher.alive():
                    log.error("数据源全挂，中止: %s", e)
                    return EXIT_NO_SOURCE
                continue
            fname = (market_of(code) + code).upper()
            is_new = fname not in old_end
            scale = 1.0
            if not is_new:
                old_close = read_bin_tail(provider_dir, code, "close", calendar, old_end[fname])
                overlap = [d for d in qfq.index if d <= d0 and d in old_close]
                if overlap:
                    scale = compute_scale(old_close, qfq, d0)
                else:
                    log.warning("%s 无重叠日，scale=1.0（若期间除权会有跳变）", code)
            target_days = new_days if is_new else [d for d in backfill_days if d > old_end[fname]]
            if qfq[qfq.index.isin(pd.DatetimeIndex(target_days))].empty:
                continue  # 增量窗口内未交易（停牌），保持老数据
            rows = build_rows(code, target_days, raw, qfq, scale)
            if not np.isnan(rows["close"].iloc[-1]):
                last_day_hits += 1
            if is_new:
                rows = rows.dropna(subset=["close"])  # 新股票只写真值行（start 由最小日期定）
            rows_out = rows.reset_index().rename(columns={"index": "date"})
            rows_out["date"] = rows_out["date"].dt.strftime("%Y-%m-%d")
            rows_out.to_csv(csv_dir / f"{market_of(code)}{code}.csv", index=False)
            (new_syms if is_new else updated).append(code)
            if (i + 1) % 500 == 0:
                log.info("抓取进度 %d/%d, 更新 %d, 失败 %d", i + 1, len(symbols), len(updated), failed)

        if not updated and not new_syms:
            log.error("没有任何股票更新成功（大面积失败）")
            return EXIT_NO_SOURCE
        if failed > len(symbols) * 0.5:
            log.error("失败率过高: %d/%d，保留旧数据", failed, len(symbols))
            return EXIT_NO_SOURCE
        total = len(updated) + len(new_syms)
        if last_day_hits < max(1, total // 2):
            # 指数有最近交易日但过半股票无当日数据 → 数据尚未发布，不能落库
            log.error("最近交易日 %s 覆盖率过低 (%d/%d)，数据未就绪，保留旧数据",
                      new_days[-1], last_day_hits, total)
            return EXIT_VALIDATE
        log.info("抓取完成: 存量更新 %d, 新上市 %d, 失败 %d, 耗时 %.0fs",
                 len(updated), len(new_syms), failed, time.time() - t_run)

        # 3. staging 副本 + dump_bin 增量
        log.info("复制 cn_data → staging ...")
        copy_to_staging(provider_dir, staging)
        dumper = DumpDataUpdate(
            str(csv_dir), str(staging), freq="day", max_workers=max_workers,
            exclude_fields="date,symbol",
        )
        dumper.dump()

        # 4. 校验
        sample = (updated[:3] + new_syms[:2]) or updated[:5]
        latest_day = new_days[-1] if new_days else calendar[-1].strftime("%Y-%m-%d")
        try:
            validate_staging(staging, latest_day, len(old_end), sample)
        except ValidateError as e:
            log.error("staging 校验失败，保留旧数据: %s", e)
            return EXIT_VALIDATE
        log.info("staging 校验通过")

        # 5. 原子切换
        atomic_swap(provider_dir, staging, prev)
        log.info("cn_data 已切换到新版本（最新交易日 %s）", latest_day)
    finally:
        shutil.rmtree(csv_dir, ignore_errors=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    # 6. 重建 h5（失败不回滚 cn_data——h5 可独立重跑）
    h5_sha, h5_elapsed = None, 0.0
    if not skip_h5:
        try:
            _, h5_elapsed, h5_sha = regen_h5(provider_dir, out)
            log.info("daily_pv_all.h5 重建完成, 耗时 %.0fs", h5_elapsed)
        except Exception as e:  # noqa: BLE001
            log.error("h5 重建失败（cn_data 已更新，可手动补跑 gen_data --full）: %s", e)
            return EXIT_FAIL

    # 7. manifest
    write_manifest(out, {
        "version": today,
        "last_trading_day": new_days[-1] if new_days else calendar[-1].strftime("%Y-%m-%d"),
        "instrument_count": len(old_end) + len(new_syms),
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "h5_sha256": h5_sha,
        "source": fetcher.last_source,
        "updated_symbols": len(updated),
        "new_symbols": len(new_syms),
        "fetch_failed": failed,
        "elapsed_sec": round(time.time() - t_run, 1),
        "h5_elapsed_sec": round(h5_elapsed, 1),
    })
    log.info("manifest 已写入 %s", out / "manifest.json")
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser(description="qlib cn_data 每日增量更新")
    ap.add_argument("--provider-uri", default="~/.qlib/qlib_data/cn_data")
    ap.add_argument("--out-dir", default="data/factor_mining")
    ap.add_argument("--source", default="auto", choices=["auto", "tencent", "mootdx", "eastmoney"])
    ap.add_argument("--limit", type=int, default=None, help="只更新前 N 只（联调/冒烟用）")
    ap.add_argument("--skip-h5", action="store_true", help="跳过 daily_pv_all.h5 重建")
    ap.add_argument("--force", action="store_true", help="无新交易日也强制回补缺口(limit污染后修复用)")
    ap.add_argument("--min-interval", type=float, default=0.15, help="腾讯源请求间隔秒")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return run(args.provider_uri, args.out_dir, source=args.source, limit=args.limit,
               skip_h5=args.skip_h5, min_interval=args.min_interval, max_workers=args.max_workers,
               force=args.force)


if __name__ == "__main__":
    sys.exit(main())
