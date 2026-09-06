#!/usr/bin/env python3
"""factor_eval.py — debug 评估 battery（DESIGN-FACTOR-MINER.md §5.3）

沙箱跑通 debug 数据后，对 result.h5 做纯 pandas 结构化检查：
  single_column    结果必须单列（factor.py 契约）
  no_inf           无 ±inf
  datetime_index   索引含 datetime 层且可解析为时间戳
  daily_frequency  无分钟级（盘中）间隔，日频契约
无 ground truth：跑通 + 全部检查通过即过；不过则回内环重写，
LLM 反馈附具体失败检查项名（EvalResult.feedback()）。

检查逻辑移植自 microsoft/RD-Agent
rdagent/components/coder/factor_coder/eva_utils.py（MIT）：
FactorSingleColumnEvaluator / FactorInfEvaluator / FactorDatetimeDailyEvaluator。
偏差：RD-Agent 仅检测恰好 1 分钟的 diff，本模块泛化为任何 <1 天的正间隔。

用法:
  python -m factor_miner.factor_eval result.h5
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd

FactorOutput = Union[pd.DataFrame, pd.Series, str, Path]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class EvalResult:
    ok: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        """失败检查项名列表——LLM 重写反馈需要具体到哪一项。"""
        return [c.name for c in self.checks if not c.ok]

    def feedback(self) -> str:
        """拼给 LLM 内环重写的反馈文本。"""
        if self.ok:
            return "All checks passed."
        lines = ["The following checks FAILED:"]
        for c in self.checks:
            mark = "OK  " if c.ok else "FAIL"
            lines.append(f"[{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _load(source: FactorOutput) -> pd.DataFrame:
    """接受 DataFrame/Series/result.h5 路径，统一为 DataFrame。"""
    if isinstance(source, pd.Series):
        return source.to_frame("factor")
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    try:
        return pd.read_hdf(path, key="data")  # factor.py 契约 key="data"
    except (KeyError, ValueError):
        return pd.read_hdf(path)  # 单 group 文件可不指定 key


def check_single_column(df: pd.DataFrame) -> CheckResult:
    """单列契约（移植 FactorSingleColumnEvaluator）。"""
    n = len(df.columns)
    if n == 1:
        return CheckResult("single_column", True, "the dataframe has only one column")
    return CheckResult(
        "single_column",
        False,
        f"the dataframe has {n} columns; the factor contract requires exactly one",
    )


def check_no_inf(df: pd.DataFrame) -> CheckResult:
    """无 ±inf（移植 FactorInfEvaluator）。"""
    n = int(df.isin([float("inf"), float("-inf")]).sum().sum())
    if n == 0:
        return CheckResult("no_inf", True, "no infinite values")
    return CheckResult("no_inf", False, f"the dataframe has {n} infinite values")


def _datetime_values(df: pd.DataFrame) -> pd.Index | None:
    """取 datetime 层值；无 datetime 层返回 None。"""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names:
        return df.index.get_level_values("datetime")
    return None


def check_datetime_index(df: pd.DataFrame) -> CheckResult:
    """索引含可解析的 datetime 层（移植 FactorDatetimeDailyEvaluator 前半）。"""
    values = _datetime_values(df)
    if values is None:
        return CheckResult(
            "datetime_index",
            False,
            f"index has no datetime level (names={list(df.index.names)})",
        )
    try:
        pd.to_datetime(values)
    except Exception as e:  # noqa: BLE001 — 任何解析失败都视为契约违反
        return CheckResult("datetime_index", False, f"datetime level not parseable: {e}")
    return CheckResult("datetime_index", True, "datetime level present and parseable")


def check_daily_frequency(df: pd.DataFrame) -> CheckResult:
    """无分钟级间隔（移植 FactorDatetimeDailyEvaluator 后半，泛化 <1 天）。"""
    values = _datetime_values(df)
    if values is None:
        return CheckResult("daily_frequency", False, "no datetime level to check")
    ts = pd.to_datetime(values).to_series().sort_values()
    diffs = ts.diff().dropna().unique()
    intraday = [d for d in diffs if pd.Timedelta(0) < d < pd.Timedelta(days=1)]
    if intraday:
        return CheckResult(
            "daily_frequency",
            False,
            f"intraday interval detected (min positive diff {min(intraday)}); factor must be daily",
        )
    return CheckResult("daily_frequency", True, "the dataframe is daily")


CHECKS = [check_single_column, check_no_inf, check_datetime_index, check_daily_frequency]


def evaluate(source: FactorOutput) -> EvalResult:
    """跑完整 battery。加载失败视为整体失败（单项 load 检查）。"""
    try:
        df = _load(source)
    except Exception as e:  # noqa: BLE001 — 文件缺失/损坏都回重写循环
        return EvalResult(ok=False, checks=[CheckResult("load", False, f"failed to load result: {e}")])
    checks = [fn(df) for fn in CHECKS]
    return EvalResult(ok=all(c.ok for c in checks), checks=checks)


def main() -> int:
    ap = argparse.ArgumentParser(description="FactorMiner debug 评估 battery")
    ap.add_argument("result", help="result.h5 路径")
    args = ap.parse_args()

    res = evaluate(args.result)
    print(res.feedback())
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
