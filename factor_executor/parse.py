"""解析 qrun 产物 —— 从 factor_miner/factor_backtest.py 移植，供执行器独立使用。

**为什么要移植而不是 import 原模块**：执行器镜像里没有沙箱、没有 panel 归一化、
也不需要 `FactorSrc` 那些概念；它只需要"跑完 qrun 之后把 mlruns 读成指标"。让执行器
依赖 miner 的业务模块会把两边重新耦合起来 —— 而这次拆分的全部意义就是切断这个耦合。

与原实现保持一致的两处刻意差异都记在各自函数上；数值口径必须完全一致，否则同一个
因子在"本地回测"和"远程回测"下会给出不同的指标，而那种不一致极难被发现。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

logger = logging.getLogger("factor-executor.parse")

# 买卖点导出上限（TopkDropout 每日调仓，多年全市场回测会让 JSON 膨胀）
MAX_TRADES = 5000


def _net_curve(report) -> list:
    """组合净值 → ``[{date, value}]``。与 factor_backtest._net_curve 同口径。

    取 qlib 报告自身的日期索引与 ``value`` 列（qlib 记录的账户净值本身），而不是
    从 return/cost 累乘重建 —— 后者会与官方口径有细微偏差，而这个序列要跟同一份
    recorder 读出的年化/回撤对得上。

    **数值索引显式退回下标 ``i``**：``pd.Timestamp(0)`` 会静默返回 1970-01-01
    （不抛异常），于是 RangeIndex 报告会变成"1970 年的净值曲线"，比没有日期更糟。
    其余情况尝试解析日期，解析不了同样退回下标 —— 宁可 x 轴是"第 N 个交易日"，
    也不要整条曲线消失或标注错误年份。
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
    except Exception:  # noqa: BLE001 — 报告结构变化
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


def _pick(metrics: dict, *keys: str) -> float | None:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def positions_to_trades(positions: dict) -> list[dict]:
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


def read_exp_res(work_dir: Path, provider_uri: str) -> tuple[dict, list[float], list[dict]]:
    """解析 qrun 产物：最新 recorder → metrics + 组合净值 + 买卖点标记。

    与 factor_backtest.read_exp_res 同口径，唯一差异是 **provider_uri 必须显式传入**
    （拆出去之后没有 `DEFAULT_PROVIDER_URI` 这种"家目录默认值"可言，执行器挂到什么
    路径就是什么路径，写死一个默认值只会掩盖挂载错误）。
    """
    import qlib

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
    from qlib.workflow import R

    latest_recorder = None
    for experiment in R.list_experiments():
        for recorder_id in R.list_recorders(experiment_name=experiment):
            if recorder_id is None:
                continue
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
            try:
                end_time = recorder.info["end_time"]
            except Exception as exc:  # noqa: BLE001 — 与上游一致，坏 recorder 跳过
                logger.debug("recorder %s unreadable, skipped: %s", recorder_id, exc)
                continue
            if end_time is None:
                logger.debug("recorder %s has no end_time, skipped", recorder_id)
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
    # 与本地实现一致：None 一律剔除，而不是留着 null 让调用方自己判断
    metrics = {k: v for k, v in metrics.items() if v is not None}

    # 含成本净值：**必须**是 (return - cost + 1).cumprod()。
    # 这里踩过一次：写成 (1+return).cumprod() 就变成不含成本的曲线，数值不同却
    # 看不出来 —— 本地回测与远程回测的净值会静默分叉。
    report = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    net = ((report["return"] - report["cost"] + 1).cumprod()).tolist()

    trades: list[dict] = []
    try:
        positions = latest_recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        trades = positions_to_trades(dict(positions))
    except Exception as exc:  # noqa: BLE001 — 持仓产物缺失/损坏不阻塞主结果
        logger.debug("positions unavailable, trades left empty: %s", exc)
    return metrics, [float(v) for v in net], trades, _net_curve(report)
