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
qlib/mlflow/jinja2 均延迟导入，未装 pyqlib 的环境可 import 本模块。

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
            "trades": self.trades,
            "error": self.error,
            "traceback": self.traceback,
        }


def data_version(data_h5: Path) -> str:
    """数据版本 = 文件内容 md5（流式，全量 h5 也不占内存）。"""
    h = hashlib.md5()
    with open(data_h5, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _read_result_h5(job_dir: Path) -> pd.DataFrame:
    try:
        df = pd.read_hdf(job_dir / "result.h5", key="data")
    except (KeyError, ValueError):
        df = pd.read_hdf(job_dir / "result.h5")
    if isinstance(df, pd.Series):
        df = df.to_frame("factor")
    return _normalize_index(df)


def _exec_factor_worker(args: tuple) -> tuple[str, Optional[pd.DataFrame], str]:
    """进程池 worker（必须模块顶层以便 pickle）：沙箱执行 factor.py 并读回 result.h5。

    返回 (name, df_or_None, error_or_empty)。
    """
    name, code, data_h5, job_dir, timeout = args
    job = Path(job_dir)
    try:
        job.mkdir(parents=True, exist_ok=True)
        link = job / "daily_pv.h5"  # factor.py 契约：读同目录 daily_pv.h5
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path(data_h5).resolve())
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


def _render_conf(template_path: Path, out_path: Path, profile: str) -> None:
    """Jinja2 预渲染 conf 模板（不用 qrun env-var 机制，spec §6 评审修订）。"""
    from jinja2 import Template  # noqa: PLC0415 — 与 qlib 一样保持模块可独立 import

    ctx = {
        # Alpha20 基线特征：str(list) 即 YAML flow 序列（沿用上游写法）
        "feature_expressions": str(list(ALPHA20.values())),
        "feature_names": str(list(ALPHA20.keys())),
    }
    ctx.update(PROFILE_OVERRIDES[profile])
    rendered = Template(template_path.read_text()).render(**ctx)
    out_path.write_text(rendered)


def _pick(metrics: dict, *keys: str) -> Optional[float]:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


# 买卖点导出上限（TopkDropout 每日调仓，多年全市场回防 JSON 膨胀）
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
    trades: list[dict] = []
    try:
        positions = latest_recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        trades = _positions_to_trades(dict(positions))
    except Exception:  # noqa: BLE001 — 持仓产物缺失/损坏不阻塞主结果
        pass
    return metrics, [float(v) for v in net], trades


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
) -> BacktestResult:
    """跑一轮 qlib 回测。失败语义见模块 docstring。"""
    work_dir = Path(work_dir)
    data_h5 = Path(data_h5)
    factors_dir = work_dir / "factors"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 进程池重执行（逐因子 exec_cache 命中则跳过）──
    version = data_version(data_h5)
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
            (f.name, f.code, str(data_h5), str(factors_dir / f.name), factor_timeout)
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

    # ── Step 4: 渲染 conf → qrun → 解析 ──
    conf_path = work_dir / "conf.yaml"
    _render_conf(Path(conf_template), conf_path, profile)
    try:
        proc = subprocess.run(
            ["qrun", str(conf_path)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return BacktestResult(
            ok=False,
            sota_broken=sota_broken,
            error=f"qrun timeout {timeout}s",
            traceback=(e.stderr or "") if isinstance(e.stderr, str) else "",
        )
    except FileNotFoundError:
        return BacktestResult(ok=False, sota_broken=sota_broken, error="qrun not found on PATH")
    if proc.returncode != 0:
        return BacktestResult(
            ok=False,
            sota_broken=sota_broken,
            error=f"qrun exit {proc.returncode}",
            traceback=proc.stderr[-4000:] if proc.stderr else proc.stdout[-4000:],
        )
    try:
        metrics, net_values, trades = read_exp_res(work_dir)
    except Exception:  # noqa: BLE001
        return BacktestResult(
            ok=False,
            sota_broken=sota_broken,
            error="failed to parse qrun output",
            traceback=tb_module.format_exc(),
        )
    return BacktestResult(
        ok=True, sota_broken=sota_broken, metrics=metrics, net_values=net_values, trades=trades
    )


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
