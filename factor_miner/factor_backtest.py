#!/usr/bin/env python3
"""factor_backtest.py — qlib 全量回测驱动（DESIGN-FACTOR-MINER.md §6）

流程（每轮实验一个 workspace）：
  1. 进程池重执行 SOTA 库全部 factor.py + 新因子（全量数据 daily_pv_all.h5）
     - 逐因子复用 exec_cache：key = md5(code + 数据版本)，命中则跳过重算；
       调用方（Go 侧经 sidecar API）接 DB 表 factor_exec_cache，写回时扫描
       work_dir/factors/<name>/result.h5 并用同一 key（见 cache_key()）
     - SOTA 因子重算失败 → quarantine：剔除出本轮拼接、记入 result.sota_broken，
       不阻塞循环；新因子失败 → ok=False + traceback，回内环重写
  2. 去重闸门：新因子 vs 任一 SOTA 日均截面 Pearson IC >= 0.99 → 丢弃
     （ok=False + dedup_dropped=True，对应 RD-Agent FactorEmptyError 语义）
  3. 写 combined_factors_df.h5（qlib 0.9.6 load_dataset 不认 parquet）+ Alpha20 基线特征
  4. Jinja2 预渲染 conf.yaml → qrun（timeout 1800s，超时记失败进 trace）
     → 移植 read_exp_res.py 逻辑解析最新 recorder → metrics + 净值序列

指标口径（反馈统一 with_cost，修正 RD-Agent trace 模板 without_cost 的内部不一致）：
  IC / Rank IC / ICIR / 1day.excess_return_with_cost.annualized_return / max_drawdown

smoke profile（CI/联调）：30 股 × 2019 一年 × num_threads 2 × leaves 31。

移植自 microsoft/RD-Agent（MIT）：
  - scenarios/qlib/developer/factor_runner.py（拼接/去重/parquet 写入）
  - scenarios/qlib/experiment/factor_template/read_exp_res.py（recorder 解析）
  - utils/qlib.py ALPHA20（基线特征表达式）
qlib/mlflow 延迟导入，未装 pyqlib 的环境可 import 本模块；但 **jinja2 是硬依赖**
—— _render_conf 用它预渲染 conf 模板，与 qlib 无关，所以它在 requirements.txt 里
常装（曾经只在 Dockerfile 的 qlib 分支装，aarch64 跳过 qlib 时 factor_oos_check
就死在 ModuleNotFoundError 上）。

用法:
  python -m factor_miner.factor_backtest --new mom20 factor.py \
      [--sota old1 s1.py ...] --work-dir /tmp/bt --data-h5 data/factor_mining/daily_pv_all.h5 \
      [--profile smoke]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback as tb_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import sandbox

# mlflow >= 3.12 把文件系统后端（./mlruns，qlib 默认）列入维护模式，不设此
# 环境变量直接抛 MlflowException。模块级 setdefault 会传播给 qrun 子进程。
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

DEDUP_IC = 0.99
DEFAULT_TIMEOUT = 1800  # qrun 墙壁时钟（spec §6 评审修订）
DEFAULT_PROVIDER_URI = "~/.qlib/qlib_data/cn_data"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONF_TEMPLATE = PACKAGE_DIR / "conf_combined_factors.yaml"

# Alpha20 基线特征表达式：照抄 rdagent/utils/qlib.py ALPHA20（MIT）
ALPHA20 = {
    "RESI5": "Resi($close, 5)/$close",
    "WVMA5": "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
    "RSQR5": "Rsquare($close, 5)",
    "KLEN": "($high-$low)/$open",
    "RSQR10": "Rsquare($close, 10)",
    "CORR5": "Corr($close, Log($volume+1), 5)",
    "CORD5": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "CORR10": "Corr($close, Log($volume+1), 10)",
    "ROC60": "Ref($close, 60)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "VSTD5": "Std($volume, 5)/($volume+1e-12)",
    "RSQR60": "Rsquare($close, 60)",
    "CORR60": "Corr($close, Log($volume+1), 60)",
    "WVMA60": "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",
    "STD5": "Std($close, 5)/$close",
    "RSQR20": "Rsquare($close, 20)",
    "CORD60": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",
    "CORD10": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "KLOW": "(Less($open, $close)-$low)/$open",
}

# smoke profile：30 股 × 2019 一年 × num_threads 2 × leaves 31（spec §6）
SMOKE_UNIVERSE = [
    "SH600000", "SH600004", "SH600006", "SH600008", "SH600009",
    "SH600010", "SH600011", "SH600012", "SH600015", "SH600016",
    "SH600018", "SH600019", "SH600020", "SH600021", "SH600023",
    "SH600025", "SH600026", "SH600027", "SH600028", "SH600029",
    "SH600030", "SH600031", "SH600033", "SH600036", "SH600037",
    "SH600038", "SH600039", "SH600048", "SH600050", "SH600061",
]
PROFILE_OVERRIDES = {
    "full": {},
    "smoke": {
        "market": "[" + ", ".join(SMOKE_UNIVERSE) + "]",
        "train_start": "2019-01-01",
        "train_end": "2019-06-30",
        "valid_start": "2019-07-01",
        "valid_end": "2019-09-30",
        "test_start": "2019-10-01",
        "test_end": "2019-12-31",
        "num_threads": 2,
        "num_leaves": 31,
    },
}


@dataclass
class FactorSrc:
    name: str
    code: str


@dataclass
class BacktestResult:
    ok: bool
    dedup_dropped: bool = False
    sota_broken: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    net_values: list[float] = field(default_factory=list)
    # 带日期的净值曲线 [{date, i, value}]（qlib 官方 value 列）。与 net_values 同源，
    # 但 net_values 是"只有数值"的既有形状，下游（执行器 result.json / Go 侧）依赖它，
    # 所以新增字段而不是改掉它。
    net_curve: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)  # 买卖点标记 [{symbol,date,action}]
    error: str = ""
    traceback: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "dedup_dropped": self.dedup_dropped,
            "sota_broken": self.sota_broken,
            "metrics": self.metrics,
            "net_values": self.net_values,
            "net_curve": self.net_curve,
            "trades": self.trades,
            "error": self.error,
            "traceback": self.traceback,
        }


MANIFEST_NAME = "manifest.json"


def _manifest_version(data_h5: Path) -> str:
    """manifest.json 的 h5_sha256（update_data 落地）→ 版本串；不可信时返回 ""。

    可信条件：manifest 存在且 **mtime >= h5 mtime**。若 h5 在 manifest 之后被
    重建（如直接跑 gen_data），manifest 里的 sha 已过期，退回 mtime+size。
    """
    try:
        data_h5 = Path(data_h5)
        mf = data_h5.parent / MANIFEST_NAME
        if not mf.exists():
            return ""
        if mf.stat().st_mtime < data_h5.stat().st_mtime:
            return ""
        info = json.loads(mf.read_text())
        sha = str(info.get("h5_sha256") or "")
        return f"sha256:{sha}" if len(sha) >= 32 else ""
    except Exception:  # noqa: BLE001 — manifest 坏/不可读按无 manifest 处理
        return ""


def data_version(data_h5: Path) -> str:
    """exec_cache 的数据版本：manifest sha256 优先，退化 mtime+size。

    旧实现对整份 h5 做 md5（全量 h5 每次回测全文件 hash，且同一回测算两次）；
    新实现 O(1) 只读元数据，且 h5 同内容重生成时版本不变（sha256 相同）→ 缓存
    不会因"每日重建但数据没变"而整体失效。docs: cache_key() 的调用方契约不变。
    """
    data_h5 = Path(data_h5)
    v = _manifest_version(data_h5)
    if v:
        return v
    st = data_h5.stat()
    return f"mtime:{st.st_mtime_ns}:size:{st.st_size}"


def cache_key(code: str, version: str) -> str:
    """exec_cache 键：md5(code + 数据版本)。调用方写回 DB 缓存必须用同一函数。"""
    return hashlib.md5((code + "\n" + version).encode()).hexdigest()


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """归一化为 (datetime, instrument) 两级 MultiIndex 并排序（移植 process_factor_data 语义）。"""
    if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names:
        dt = df.index.get_level_values("datetime")
        inst = (
            df.index.get_level_values("instrument")
            if "instrument" in df.index.names
            else df.index.get_level_values(1 - df.index.names.index("datetime"))
        )
        df = df.copy()
        df.index = pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])
    return df.sort_index()


def _net_curve(report) -> list:
    """组合净值 → ``[{date, value}]``，取 qlib 报告自身的日期索引与 ``value`` 列。

    为什么用 ``value`` 而不是从 return/cost 累乘重建：``value`` 是 qlib 逐日记录的
    账户净值本身（分母/初始资金口径由 qlib 决定），累乘重建会引入与官方口径的
    细微偏差，而这个序列是要画给使用者看的净值曲线 —— 用官方值更对得上
    metrics 里的年化/回撤（那两项也是从同一份 recorder 读的）。

    索引确实是 datetime（生产实测：``index.name='datetime'``、
    ``dtype=datetime64[us]``，2021-01-04 → 2026-08-26 共 1369 行），但索引形态不是
    本函数能保证的，而**数值索引会被 pandas 静默当成 epoch**——``pd.Timestamp(0)``
    返回 1970-01-01 且不抛异常，于是一条 RangeIndex 报告会变成"1970 年的净值曲线"，
    比没有日期更糟。所以数值索引显式退回下标 ``i``；其余情况尝试解析，解析不了
    同样退回下标 —— 宁可 x 轴是"第 N 个交易日"，也不要整条曲线消失或标注错误年份。
    只丢**开头**的 0：生产实测 7 份 report **每一份**的 ``value`` 首值都是 ``0.0``
    （qlib 首日占位），而紧随其后的才是真实账户权益（约 8.7e7）。留着它，曲线会从 0
    起步，读起来像"亏光了本金"，而且会把整个 y 轴压扁。``i`` 在丢弃后保持原值，
    所以时间轴不会因此错位。序列中间的 0 是真实净值，不能动；整条序列全为 0 时原样
    返回，免得用"没有曲线"代替"曲线是平的"。
    """
    if report is None:
        return []
    try:
        values = report["value"]
    except Exception:  # noqa: BLE001 — 没有 value 列（报告结构变化）
        return []

    out: list = []
    # 数值索引 → 不解析日期（pandas 会静默给出 1970 epoch）
    numeric_index = pd.api.types.is_numeric_dtype(report.index)
    for i, (idx, raw) in enumerate(values.items()):
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        date = None
        if not numeric_index:
            try:
                ts = pd.Timestamp(idx)
                date = None if pd.isna(ts) else ts.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                date = None
        out.append({"date": date, "i": i, "value": v})
    # 丢掉开头的 0（qlib 首日占位）。只丢开头 —— 中间的 0 是真实净值。
    first_nonzero = None
    for k, p in enumerate(out):
        if p["value"] != 0:
            first_nonzero = k
            break
    if first_nonzero:
        out = out[first_nonzero:]
    return out


def _read_result_h5(job_dir: Path) -> pd.DataFrame:
    try:
        df = pd.read_hdf(job_dir / "result.h5", key="data")
    except (KeyError, ValueError):
        df = pd.read_hdf(job_dir / "result.h5")
    if isinstance(df, pd.Series):
        df = df.to_frame("factor")
    return _normalize_index(df)



def stage_h5_for_sandbox(data_h5: Path, job_dir: Path, window_days: int) -> Path:
    """把 h5 按窗口截取后放进 job 目录，返回沙箱要读的路径。

    window_days <= 0 → symlink 全量（联调/需要全历史时显式选择）。
    窗口是"最近 N 个交易日"，索引层级名沿用原文件的 ``datetime`` / ``instrument``，
    HDF key 保持 ``data``（因子代码惯用 ``pd.read_hdf('daily_pv.h5')`` 单 key 读取）。

    **本模块是这份逻辑的唯一实现**。历史上 factor_worker 里还有一份窗口化实现，
    而 _exec_factor_worker 自己写了一句 `symlink_to(全量)` —— 两条路径行为不一致：
    走回测的那条永远喂全量，于是撞破沙箱地址空间上限被 SIGKILL，且错误信息里
    看不出任何"数据太大"的线索。重复实现是根因，所以只留一份。
    """
    dst = job_dir / "daily_pv.h5"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if window_days <= 0:
        dst.symlink_to(Path(data_h5).resolve())
        return dst

    df = pd.read_hdf(data_h5, key="data")
    try:
        dts = df.index.get_level_values("datetime").unique().sort_values()
    except (KeyError, AttributeError):
        # 索引层级名不是预期结构：原样落盘，不做窗口（安全兜底）
        df.to_hdf(dst, key="data", mode="w")
        return dst
    if len(dts) > window_days:
        keep = set(dts[-window_days:])
        df = df[df.index.get_level_values("datetime").isin(keep)]
    df.to_hdf(dst, key="data", mode="w")
    return dst


def _exec_factor_worker(args: tuple) -> tuple[str, Optional[pd.DataFrame], str]:
    """进程池 worker（必须模块顶层以便 pickle）：沙箱执行 factor.py 并读回 result.h5。

    返回 (name, df_or_None, error_or_empty)。
    """
    name, code, data_h5, job_dir, timeout, window_days = args
    job = Path(job_dir)
    try:
        job.mkdir(parents=True, exist_ok=True)
        # factor.py 契约：读同目录 daily_pv.h5。
        # **必须走 stage_h5_for_sandbox**（窗口截取）—— 这里曾经直接 symlink 全量，
        # 于是沙箱读 1500 万行、顶破 RLIMIT_AS 被 SIGKILL（exit -9）。
        stage_h5_for_sandbox(data_h5, job, window_days)
        res = sandbox.run_factor_source(code, str(job), timeout=timeout)
        if res.returncode != 0 or not (job / "result.h5").exists():
            err = res.violation or (res.stderr or res.stdout or f"exit {res.returncode}")
            return name, None, err
        return name, _read_result_h5(job), ""
    except Exception:  # noqa: BLE001 — 失败语义由调用方区分 SOTA/新因子
        return name, None, tb_module.format_exc()


def _daily_ic(a: pd.Series, b: pd.Series) -> float:
    """日均截面 Pearson IC（移植 factor_runner.deduplicate_new_factors 的 groupby(datetime).corr().mean()）。"""
    pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if pair.empty:
        return 0.0
    ics = pair.groupby(level="datetime").apply(lambda x: x["a"].corr(x["b"])).dropna()
    return float(ics.mean()) if len(ics) else 0.0


def _render_conf(template_path: Path, out_path: Path, profile: str,
                 provider_uri: str | None = None) -> None:
    """Jinja2 预渲染 conf 模板（不用 qrun env-var 机制，spec §6 评审修订）。"""
    from jinja2 import Template  # noqa: PLC0415 — 与 qlib 一样保持模块可独立 import

    ctx = {
        # Alpha20 基线特征：str(list) 即 YAML flow 序列（沿用上游写法）
        "feature_expressions": str(list(ALPHA20.values())),
        "feature_names": str(list(ALPHA20.keys())),
    }
    ctx.update(PROFILE_OVERRIDES[profile])
    # provider_uri 可注入：默认沿用模板自带值（本地行为不变），拆出执行器后由
    # --provider-uri 指向执行器容器里的挂载路径。**必须两边同源**，否则回测会读
    # 到另一份数据而静默偏掉 —— 执行器侧另有面板版本校验兜底。
    if provider_uri:
        ctx["provider_uri"] = provider_uri
    rendered = Template(template_path.read_text()).render(**ctx)
    out_path.write_text(rendered)


def _pick(metrics: dict, *keys: str) -> Optional[float]:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


# 买卖点导出上限（TopkDropout 每日调仓，多年全市场回防 JSON 膨胀）
# 沙箱输入窗口（交易日）。与 factor_worker.WINDOW_DAYS 同一个 env、同一个默认值 ——
# 但定义在这里，因为**沙箱暂存发生在本模块**（stage_h5_for_sandbox）。
#
# 为什么必须是窗口而不是全量：daily_pv_all.h5 约 1500 万行，而沙箱上的是
# RLIMIT_AS **硬**上限（sandbox.MAX_AS_BYTES，默认 4GiB，虚拟地址空间含 mmap）。
# 全量读 + pandas 中间结果会顶破它 → 子进程直接 SIGKILL（`exit -9`），
# 表现为 "new factor 'x' failed"，真正的原因完全看不出来。
WINDOW_DAYS = int(os.environ.get("FACTOR_MINER_WINDOW_DAYS", "400") or 0)

MAX_TRADES = 5000


def _positions_to_trades(positions: dict) -> list[dict]:
    """逐日持仓 → 买卖点标记（集合差分，不依赖 qlib 类，方便单测）：

    今日持有而昨日未持有 = buy（入场日），昨日持有今日消失 = sell（出场日）。
    TopkDropout 策略下同一只股票可能多次进出，每次进出各产生一对标记。
    """
    trades: list[dict] = []
    prev: set[str] = set()
    for day in sorted(positions):
        pos = positions[day]
        get_list = getattr(pos, "get_stock_list", None)
        if callable(get_list):
            held = set(get_list())
        else:  # 兼容裸 dict 持仓（amount > 0 视为持有）
            held = {
                k for k, v in dict(pos).items()
                if isinstance(v, dict) and v.get("amount", 0) > 0
            }
        date_str = str(pd.Timestamp(day).date())
        for sym in sorted(held - prev):
            trades.append({"symbol": sym, "date": date_str, "action": "buy"})
        for sym in sorted(prev - held):
            trades.append({"symbol": sym, "date": date_str, "action": "sell"})
        prev = held
        if len(trades) >= MAX_TRADES:
            break
    return trades


def read_exp_res(work_dir: Path, provider_uri: str = DEFAULT_PROVIDER_URI) -> tuple[dict, list[float], list[dict]]:
    """解析 qrun 产物（移植 read_exp_res.py：最新 recorder → metrics + 组合净值
    + 买卖点标记）。

    差异：上游子进程 cwd=workspace 靠默认 mlflow uri 找到 ./mlruns；
    本函数进程内调用，显式把 exp_manager uri 指到 work_dir/mlruns。
    买卖点从 positions_normal_1day.pkl 逐日持仓差分推导；该文件缺失时
    降级为空列表（不影响 metrics/净值）。
    """
    import qlib  # noqa: PLC0415 — 延迟导入，未装 pyqlib 可 import 本模块

    qlib.init(
        provider_uri=provider_uri,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "file:" + str((work_dir / "mlruns").resolve()),
                "default_exp_name": "Experiment",
            },
        },
    )
    from qlib.workflow import R  # noqa: PLC0415

    latest_recorder = None
    for experiment in R.list_experiments():
        for recorder_id in R.list_recorders(experiment_name=experiment):
            if recorder_id is None:
                continue
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            try:
                end_time = recorder.info["end_time"]
            except Exception:  # noqa: BLE001 — 与上游一致，坏 recorder 跳过
                continue
            if end_time is None:
                continue
            if latest_recorder is None or end_time > latest_recorder.info["end_time"]:
                latest_recorder = recorder
    if latest_recorder is None:
        raise RuntimeError("no recorders found in qrun output")

    raw = dict(latest_recorder.list_metrics())
    metrics = {
        "IC": _pick(raw, "IC"),
        "RankIC": _pick(raw, "Rank IC"),
        "ICIR": _pick(raw, "ICIR"),
        "annualized_return_with_cost": _pick(
            raw, "1day.excess_return_with_cost.annualized_return"
        ),
        "max_drawdown": _pick(raw, "1day.excess_return_with_cost.max_drawdown"),
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}

    report = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    net = ((report["return"] - report["cost"] + 1).cumprod()).tolist()  # with_cost 净值
    # 带日期的净值曲线（qlib 官方 value 列）。net_values 保持原样不动 —— 它在
    # 下游有既有消费者（执行器 result.json、Go 侧），改形状会连带打断它们。
    net_curve = _net_curve(report)
    trades: list[dict] = []
    try:
        positions = latest_recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        trades = _positions_to_trades(dict(positions))
    except Exception:  # noqa: BLE001 — 持仓产物缺失/损坏不阻塞主结果
        pass
    return metrics, [float(v) for v in net], trades, net_curve


def run_backtest(
    sota_factors: list[FactorSrc],
    new_factor: FactorSrc,
    work_dir: Path | str,
    data_h5: Path | str,
    conf_template: Path | str = DEFAULT_CONF_TEMPLATE,
    exec_cache: Callable[[str], Optional[Path]] = lambda key: None,
    timeout: int = DEFAULT_TIMEOUT,
    profile: str = "full",
    factor_timeout: int = sandbox.DEFAULT_TIMEOUT_SECONDS,
    n_proc: int = 4,
    version: str | None = None,
    execute: Callable[[Path, Path, int, str | None], dict] | None = None,
    provider_uri: str | None = None,
    window_days: int | None = None,
) -> BacktestResult:
    """跑一轮 qlib 回测。失败语义见模块 docstring。

    version：exec_cache 的数据版本；调用方（factor_worker）已算过就传进来，
    避免同一回测把数据版本算两次（旧版这里会重复 hash 整份 h5）。

    execute：第 4 步「执行」的注入点。None = 本地起 qrun（默认，与历史行为一致）；
    传入可调用对象则交给它 —— factor_executor 就是用它把 qrun 搬到独立进程/机器上。
    签名 `(work_dir, conf_path, timeout, version) -> dict`，返回统一 outcome
    （成功 `{ok: True, metrics, net_values, trades}`，失败 `{ok: False, error, traceback}`）。

    window_days：喂给沙箱的行情窗口（交易日）。None = WINDOW_DAYS（400）。
    0 = symlink 全量，仅联调/确需全历史时用 —— 全量会顶破沙箱 RLIMIT_AS 被 SIGKILL。

    provider_uri：渲染 conf 时写进 qlib_init 的数据路径。None = 沿用模板自带值。
    拆出执行器后必须传执行器自己的挂载路径 —— 否则远端 qrun 会去找一个不存在的
    路径，或在更糟的情况下读到另一份数据。
    """
    work_dir = Path(work_dir)
    data_h5 = Path(data_h5)
    factors_dir = work_dir / "factors"
    work_dir.mkdir(parents=True, exist_ok=True)
    # 沙箱输入窗口：调用方可覆盖，默认取 WINDOW_DAYS（400 交易日）
    window_days = WINDOW_DAYS if window_days is None else window_days

    # ── Step 1: 进程池重执行（逐因子 exec_cache 命中则跳过）──
    version = version or data_version(data_h5)
    all_factors = [("sota", f) for f in sota_factors] + [("new", new_factor)]
    dfs: dict[str, pd.DataFrame] = {}
    misses: list[tuple[str, FactorSrc]] = []
    for kind, f in all_factors:
        cached = exec_cache(cache_key(f.code, version))
        if cached is not None:
            try:
                dfs[f.name] = _read_result_h5(Path(cached))
                continue
            except Exception:  # noqa: BLE001 — 缓存损坏则重算
                pass
        misses.append((kind, f))

    if misses:
        jobs = [
            (f.name, f.code, str(data_h5), str(factors_dir / f.name), factor_timeout,
             window_days)
            for _, f in misses
        ]
        results: dict[str, tuple[Optional[pd.DataFrame], str]] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_proc) as pool:
            for name, df, err in pool.map(_exec_factor_worker, jobs):
                results[name] = (df, err)
        sota_broken: list[str] = []
        for kind, f in misses:
            df, err = results[f.name]
            if df is not None:
                dfs[f.name] = df
            elif kind == "new":
                return BacktestResult(ok=False, error=f"new factor '{f.name}' failed", traceback=err)
            else:
                sota_broken.append(f.name)  # quarantine：剔除但不阻塞
    else:
        sota_broken = []

    sota_dfs = [dfs[f.name] for f in sota_factors if f.name in dfs]
    new_df = dfs[new_factor.name]

    # ── Step 2: 去重闸门 ──
    new_series = new_df.iloc[:, 0]
    for f in sota_factors:
        if f.name not in dfs:
            continue
        ic = _daily_ic(new_series, dfs[f.name].iloc[:, 0])
        if abs(ic) >= DEDUP_IC:
            return BacktestResult(
                ok=False,
                dedup_dropped=True,
                sota_broken=sota_broken,
                error=f"new factor '{new_factor.name}' dropped: daily IC {ic:.4f} vs SOTA '{f.name}' >= {DEDUP_IC}",
            )

    # ── Step 3: combined_factors_df.h5（单层列名；"feature" 层级由 conf
    #    的 StaticDataLoader dict 分支 {feature: ...} 在加载时拼接。
    #    qlib 0.9.6 load_dataset 只认 .h5/.pkl/.csv，parquet 不支持）──
    combined = pd.concat(
        [df.rename(columns={df.columns[0]: name}) for name, df in
         [(f.name, dfs[f.name]) for f in sota_factors if f.name in dfs]
         + [(new_factor.name, new_df)]],
        axis=1,
    ).dropna()
    combined = combined.sort_index()
    combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
    combined.to_hdf(work_dir / "combined_factors_df.h5", key="data")

    # ── Step 4: 渲染 conf → 执行 → 解析 ──
    #
    # 「执行」这一步可注入：默认在本地起 qrun，也可以交给远程执行器（见
    # factor_executor/）。拆出去的理由是资源隔离，不是逻辑差异 —— 因此两边必须
    # 产出同一套 (metrics, net_values, trades)，否则同一个因子在本地与远程会给出
    # 不同指标，而这种分叉几乎不可能被发现。
    conf_path = work_dir / "conf.yaml"
    _render_conf(Path(conf_template), conf_path, profile, provider_uri)
    if execute is None:
        outcome = execute_local(work_dir, timeout)
    else:
        outcome = execute(work_dir, conf_path, timeout, version)
    if not outcome.get("ok"):
        return BacktestResult(
            ok=False,
            sota_broken=sota_broken,
            error=str(outcome.get("error") or "backtest failed"),
            traceback=str(outcome.get("traceback") or ""),
        )
    metrics = outcome.get("metrics") or {}
    net_values = outcome.get("net_values") or []
    trades = outcome.get("trades") or []
    # 远程路径由执行器带回 net_curve；本地路径由 execute_local 填充。都会走这里。
    net_curve = outcome.get("net_curve") or []
    return BacktestResult(
        ok=True, sota_broken=sota_broken, metrics=metrics, net_values=net_values,
        net_curve=net_curve, trades=trades,
    )


def execute_local(work_dir: Path, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """在**本进程所在容器**里执行 qrun 并解析 —— 注入点的默认实现。

    返回统一的 outcome dict：成功 `{ok: True, metrics, net_values, trades}`，
    失败 `{ok: False, error, traceback}`。远程执行器返回同一个形状，调用方无分支。
    """
    conf_path = work_dir / "conf.yaml"
    try:
        proc = subprocess.run(
            ["qrun", str(conf_path)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,   # 自己判 returncode 并翻译成 outcome，不需要 run() 抛异常
            # qrun 子进程同样上 rlimit（仅 RLIMIT_FSIZE，见 sandbox.qrun_preexec：
            # RLIMIT_AS/CPU 会杀掉多线程全量回测，故不复用 _apply_limits）
            preexec_fn=sandbox.qrun_preexec(),
        )
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": f"qrun timeout {timeout}s",
            "traceback": (e.stderr or "") if isinstance(e.stderr, str) else "",
        }
    except FileNotFoundError:
        return {"ok": False, "error": "qrun not found on PATH"}
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"qrun exit {proc.returncode}",
            "traceback": proc.stderr[-4000:] if proc.stderr else proc.stdout[-4000:],
        }
    try:
        metrics, net_values, trades, net_curve = read_exp_res(work_dir)
    except Exception:  # noqa: BLE001
        return {
            "ok": False,
            "error": "failed to parse qrun output",
            "traceback": tb_module.format_exc(),
        }
    return {"ok": True, "metrics": metrics, "net_values": net_values, "trades": trades,
            "net_curve": net_curve}


def _load_factor_src(name: str, path: str) -> FactorSrc:
    return FactorSrc(name=name, code=Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description="FactorMiner qlib 回测驱动")
    ap.add_argument("--new", nargs=2, metavar=("NAME", "FACTOR_PY"), required=True)
    ap.add_argument("--sota", nargs=2, metavar=("NAME", "FACTOR_PY"), action="append", default=[])
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--data-h5", required=True)
    ap.add_argument("--conf", default=str(DEFAULT_CONF_TEMPLATE))
    ap.add_argument("--profile", choices=["full", "smoke"], default="full")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    sota = [_load_factor_src(n, p) for n, p in args.sota]
    new = _load_factor_src(*args.new)
    res = run_backtest(
        sota_factors=sota,
        new_factor=new,
        work_dir=args.work_dir,
        data_h5=args.data_h5,
        conf_template=args.conf,
        timeout=args.timeout,
        profile=args.profile,
    )
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
