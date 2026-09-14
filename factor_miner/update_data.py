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
并发提交（第二个进程同时跑同一 provider）→ 文件锁立即拒绝（EXIT_LOCKED=5），
避免两个流程同时 rmtree staging + atomic_swap 同一 provider_dir 造成数据损坏。

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

try:
    import fcntl
except ImportError:  # Windows 无 fcntl → UpdateLock 退化为无锁（只记 warning）
    fcntl = None

from factor_miner.qlib_dump_bin import DumpDataUpdate

log = logging.getLogger("update_data")

# 存量数据约定（实测推断 + 验证，见设计文档）:
#   $close/$open/$high/$low  = 前复权价（以快照生成日为基准）
#   $factor                  = 前复权价 / 原始价
#   $volume                  = 原始成交量(手) / factor
#   $amount/$vwap/$adjclose/$change 本管线不更新（停留在老日历末端，qlib 读取自动截尾）
DUMP_FIELDS = ["open", "high", "low", "close", "volume", "factor"]

A_SHARE_RE = re.compile(r"^(60[0-9]{4}|68[0-9]{4}|00[0-9]{4}|30[0-9]{4})$")

# A 股全市场量级下限：低于此值说明股票列表接口退化（截断/限流），
# 绝不能拿它当 universe 去更新，否则日历会全局前移而只有极少数票拿到新 bar。
MIN_UNIVERSE = int(os.environ.get("FACTOR_MINER_MIN_UNIVERSE", "2000") or 0)
# staging 校验：close.day.bin 与日历对齐的 instrument 占比下限
MIN_BIN_COVERAGE = float(os.environ.get("FACTOR_MIN_BIN_COVERAGE", "0.75") or 0)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NO_SOURCE = 3     # 数据源全挂 / 大面积失败
EXIT_VALIDATE = 4      # 校验失败
EXIT_LOCKED = 5        # 已有更新任务在跑（并发被单飞锁拒绝）


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
        from .fetchers import tdx_client  # noqa: PLC0415 — 延迟导入，单测/无 mootdx 环境可 import 本模块

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
        from .fetchers import em_get  # noqa: PLC0415 — 延迟导入

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


def all_txt_codes(provider_dir: Path) -> list:
    """instruments/all.txt 里的 A 股代码（本地已落库的存量universe）。"""
    inst = provider_dir / "instruments" / "all.txt"
    out = []
    try:
        for line in inst.read_text().splitlines():
            sym = line.split("\t")[0].strip()
            m = re.match(r"^(SH|SZ)(\d{6})$", sym)
            if m and A_SHARE_RE.match(m.group(2)):
                out.append(m.group(2))
    except OSError as e:
        log.warning("读取 %s 失败: %s", inst, e)
    return out


def list_symbols(provider_dir: Path) -> list:
    """全市场 A 股代码。优先东财 clist 实时列表（含新上市），失败/退化回退
    instruments/all.txt。BJ 股票腾讯/东财日 K 覆盖不全，v1 不含。

    东财 clist 会**静默截断分页**（请求 pz=500 也只回 100 行，接口异常时甚至
    只回 2 行），所以这里有三道防线，缺一不可：
      1) 终止条件用响应里的 total，不再用 `len(diff) < pz`——一个被截断的短页
         会被旧逻辑当成"列表到底了"，于是"全市场"退化成 100 只甚至 2 只；
      2) 结果与 all.txt **求并集**，接口退化时不会丢掉存量票；
      3) 少于 MIN_UNIVERSE 一律判为接口异常，直接以 all.txt 为准。
    """
    codes: list = []
    total_seen = 0
    degraded = False
    try:
        from .fetchers import em_get  # noqa: PLC0415

        for mkt in ("m:1+t:2,m:1+t:23", "m:0+t:6,m:0+t:80"):  # 沪A + 深A（含科创/创业）
            pn, pz = 1, 100  # 东财实际单页上限 100
            while True:
                r = em_get("https://82.push2.eastmoney.com/api/qt/clist/get", params={
                    "pn": str(pn), "pz": str(pz), "po": "1", "np": "1",
                    "fltt": "2", "invt": "2", "fs": mkt, "fields": "f12",
                }, timeout=15)
                data = (r.json() or {}).get("data") or {}
                diff = data.get("diff") or []
                total_seen = max(total_seen, int(data.get("total") or 0))
                codes += [str(d["f12"]) for d in diff if A_SHARE_RE.match(str(d.get("f12", "")))]
                if len(diff) < pz or (total_seen and pn * pz >= total_seen):
                    break
                pn += 1
                if pn > 200:  # 硬上限：防 total 异常导致死循环
                    log.warning("clist 分页超过 200 页，强制结束")
                    break
    except Exception as e:  # noqa: BLE001
        log.warning("东财 clist 获取股票列表失败, 回退 instruments/all.txt: %s", e)
        degraded = True

    fetched = sorted(set(codes))
    if not fetched:
        degraded = True
    elif total_seen and len(fetched) < total_seen * 0.8:
        log.warning("clist 只取到 %d 只 / total=%d，判定分页截断", len(fetched), total_seen)
        degraded = True
    if fetched and len(fetched) < MIN_UNIVERSE:
        log.error("clist 只返回 %d 只 (< %d)，判定接口异常", len(fetched), MIN_UNIVERSE)
        degraded = True

    old = all_txt_codes(provider_dir)
    if degraded or not fetched:
        out = old
    else:
        out = sorted(set(fetched) | set(old))  # 并集：绝不多丢存量
    if len(out) < MIN_UNIVERSE:
        log.error("股票列表仅 %d 只 (< %d)，all.txt 也可能不完整", len(out), MIN_UNIVERSE)
    return out


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
    # 全量覆盖面：日历前移了、但只有极少数票真的写进新 bar —— 旧逻辑看不见这种
    # "局部落库"（all.txt 行数由 copytree 继承、sample 又只抽刚更新的那几只），
    # 结果就是日历全局前进而 99% 的票停在上一日。这里按 bin 与日历的对齐比例兜底。
    aligned, checked = bin_alignment(staging, len(cal))
    ratio = aligned / checked if checked else 0.0
    need = max(1, int(min_instruments * MIN_BIN_COVERAGE))
    if checked and (ratio < MIN_BIN_COVERAGE or aligned < need):
        raise ValidateError(
            f"新交易日覆盖面过低: {aligned}/{checked} ({ratio:.1%}) 只票的 bin 对齐日历尾"
            f"（需要 ≥{need} 只），疑似只更新了部分标的")


def bin_alignment(staging: Path, cal_len: int) -> tuple:
    """统计 features/*/close.day.bin 中"行数与全局日历对齐"的只数。

    只读每个 bin 的头 4 字节（起始索引）+ 文件大小，~6000 只票毫秒级。
    退市/长期停牌的票天然对不齐，所以调用方用比例阈值而不是要求全对齐。
    """
    root = staging / "features"
    aligned = checked = 0
    try:
        for d in root.iterdir():
            p = d / "close.day.bin"
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size < 4:
                continue
            checked += 1
            start = int(np.fromfile(p, dtype="<f", count=1)[0])
            if (size - 4) // 4 == cal_len - start:
                aligned += 1
    except OSError as e:
        log.warning("覆盖面统计失败: %s", e)
    return aligned, checked


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


# ═══════════════ 并发单飞锁 ═══════════════

LOCK_ENV = "FACTOR_MINER_UPDATE_LOCK"


def update_lock_path(provider_uri: str) -> Path:
    """固定锁文件路径：env FACTOR_MINER_UPDATE_LOCK 覆盖，否则 /tmp 下按 provider 命名。"""
    override = os.environ.get(LOCK_ENV)
    if override:
        return Path(override).expanduser()
    key = hashlib.sha1(str(Path(provider_uri).expanduser()).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"factor_miner_update_{key}.lock"


class UpdateLock:
    """update_data 进程级单飞锁（fcntl.flock 独占 + 非阻塞）。

    - 第二个并发提交立刻得到 False → 调用方返回 EXIT_LOCKED，**零副作用**
      （不会 rmtree staging、不会 atomic_swap，避免两进程互相踩坏 provider_dir）
    - 进程退出/崩溃由内核自动释放，不留死锁
    - 无 fcntl 的平台（Windows）退化为无锁并 warning，不阻断服务
    - flock 绑定 open file description，同进程第二次 open 也会被拒
    """

    def __init__(self, path):
        self.path = Path(path)
        self._fd = None

    def acquire(self) -> bool:
        if fcntl is None:
            log.warning("fcntl 不可用，update_data 并发锁降级为无锁")
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as e:
            log.warning("锁文件不可用 (%s)，降级为无锁: %s", self.path, e)
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        try:  # 记录持锁 pid，便于并发方排查
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        except OSError:
            pass
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        finally:
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def holder_pid(path) -> str:
    """锁文件里记录的持有者（尽力而为，读不到返回空串）。"""
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


# ═══════════════ 主流程 ═══════════════

def run(provider_uri: str, out_dir: str, fetcher=None, source: str = "auto",
        symbols: list = None, limit: int = None, skip_h5: bool = False,
        min_interval: float = 0.15, max_workers: int = 8, force: bool = False) -> int:
    """带并发互斥的入口：已有更新在跑 → 立刻返回 EXIT_LOCKED（零副作用）。"""
    lock = UpdateLock(update_lock_path(provider_uri))
    if not lock.acquire():
        log.error("已有 update_data 任务在运行（lock=%s holder=%s），拒绝并发提交",
                  lock.path, holder_pid(lock.path))
        return EXIT_LOCKED
    try:
        return _run_impl(provider_uri, out_dir, fetcher=fetcher, source=source,
                         symbols=symbols, limit=limit, skip_h5=skip_h5,
                         min_interval=min_interval, max_workers=max_workers,
                         force=force)
    finally:
        lock.release()


def _run_impl(provider_uri: str, out_dir: str, fetcher=None, source: str = "auto",
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
    auto_universe = symbols is None
    if symbols is None:
        symbols = list_symbols(provider_dir)
    if limit:
        symbols = symbols[:limit]
    if auto_universe and len(symbols) < MIN_UNIVERSE:
        # 接口退化时拿 100/2 只当"全市场"去跑，会推进全局日历却只写极少数票——
        # 这正是 2026-09-08 起 4 个交易日只剩 SZ000001/SZ000002 的成因。
        # 只拦自动发现的 universe；调用方显式指定 symbols 视为有意为之。
        log.error("股票列表仅 %d 只 (< %d)，判定接口截断，保留旧数据", len(symbols), MIN_UNIVERSE)
        return EXIT_NO_SOURCE
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

    # 断点续抓：FACTOR_MINER_CSV_DIR 指定沿用上一轮的抓取目录（默认新建临时目录），
    # FACTOR_MINER_CSV_RESUME=1 时已存在的 CSV 直接复用、只补缺失的票。
    # 抓 5000+ 只票走的是免费源（约 1 只/秒，一跑 1.5 小时），中断后从头再来
    # 代价极高——而中断（进程被杀/容器重启）并不会损坏 provider 目录，
    # 因为落库是最后一步的原子 swap。这个开关让那一小时的抓取不白费。
    csv_dir = Path(os.environ.get("FACTOR_MINER_CSV_DIR") or tempfile.mkdtemp(prefix="qlib_csv_"))
    csv_dir.mkdir(parents=True, exist_ok=True)
    resume = os.environ.get("FACTOR_MINER_CSV_RESUME") == "1"
    reused = 0
    updated, new_syms, failed, last_day_hits = [], [], 0, 0
    try:
        for i, code in enumerate(symbols):
            # 进度日志放在循环顶部：复用分支会 continue，放在底部就看不到复用阶段的进度
            if (i + 1) % 500 == 0:
                log.info("抓取进度 %d/%d, 更新 %d, 复用 %d, 失败 %d",
                         i + 1, len(symbols), len(updated) + len(new_syms), reused, failed)
            fname = (market_of(code) + code).upper()
            is_new = fname not in old_end
            csv_path = csv_dir / f"{market_of(code)}{code}.csv"
            if resume and csv_path.exists():
                # 复用上一轮已抓好的 CSV，且按与抓取路径相同的口径计入 last_day_hits，
                # 否则覆盖率守卫会被"复用的票不算命中"错误触发。
                reused += 1
                try:
                    tail = pd.read_csv(csv_path, usecols=["close"]).tail(1)
                    if not tail.empty and not np.isnan(float(tail["close"].iloc[-1])):
                        last_day_hits += 1
                except (OSError, ValueError, KeyError) as e:
                    log.warning("%s 复用 CSV 读取失败，改为重新抓取: %s", code, e)
                    reused -= 1
                    csv_path.unlink(missing_ok=True)
                else:
                    (new_syms if is_new else updated).append(code)
                    continue
            try:
                raw, qfq = fetcher.fetch_stock(code, start, today)
            except FetchError as e:
                failed += 1
                if not fetcher.alive():
                    log.error("数据源全挂，中止: %s", e)
                    return EXIT_NO_SOURCE
                continue
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
        log.info("抓取完成: 存量更新 %d, 新上市 %d, 失败 %d, 复用 %d, 耗时 %.0fs",
                 len(updated), len(new_syms), failed, reused, time.time() - t_run)

        # 3. staging 副本 + dump_bin 增量
        log.info("复制 cn_data → staging ...")
        copy_to_staging(provider_dir, staging)
        dumper = DumpDataUpdate(
            str(csv_dir), str(staging), freq="day", max_workers=max_workers,
            exclude_fields="date,symbol",
        )
        dumper.dump()

        # 4. 校验
        # 抽样要覆盖"刚更新的"和"存量里随机的"两面：只抽刚更新的几只时，
        # 一旦只有少数票更新成功（局部落库），校验反而会因为抽到这几只而通过。
        # old_end 的键是 SH600000 形式，这里剥掉市场前缀还原成 validate 要的 6 位码。
        pool = sorted({k[2:] for k in old_end} | set(updated) | set(new_syms))
        sample = random.sample(pool, min(30, len(pool))) if pool else []
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
