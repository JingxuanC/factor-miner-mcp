"""把 RD-Agent 的 qlib 回测接到 factor-miner-mcp（零 fork、零 Docker、零 conda）。

为什么接在 **runner** 而不是 env/workspace：
    RD-Agent 的 `QlibFactorRunner.develop()` 才是"因子代码 → 合并 → 回测 → 指标"
    这一步；而 factor-miner 的 `factor_backtest(sota, new_factors, profile)` 恰好
    做的是同一件事（而且它的 `conf_combined_factors.yaml` 就是从 RD-Agent
    `scenarios/qlib/experiment/factor_template/` 移植的，指标口径一致：
    统一 `with_cost`、`FactorEmptyError` 语义）。
    → 所以把整段交给 miner，比替换 Env（只换 qrun 的物理去处）更彻底：
      连"在哪个 python 环境里跑因子代码"都不再需要（宿主机不需要 qlib/conda）。

下游契约（必须精确满足，否则 feedback 层直接 KeyError）：
    `rdagent/scenarios/qlib/developer/feedback.py`：
        IMPORTANT_METRICS = ["IC",
                             "1day.excess_return_with_cost.annualized_return",
                             "1day.excess_return_with_cost.max_drawdown"]
        process_results() 会 `combined_df.loc[IMPORTANT_METRICS]`
    `QlibFBWorkspace.execute()` 的原始实现还要求工作区里同时存在
    `qlib_res.csv` 与 `ret.pkl`，否则返回 `(None, log)` → 上层抛 FactorEmptyError。
    → 本 runner 自己写这两个文件并把结果塞进 `exp.result`，语义与上游一致。

启用方式（二选一，都不改 RD-Agent 源码）：
    QLIB_FACTOR_RUNNER=bridge.runner.FactorMinerRunner   # 环境变量
    或在自己的 FactorBasePropSetting 子类里把 runner 指过来
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from rdagent.core.exception import FactorEmptyError

try:  # 上游路径（v0.8.x）
    from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
except ImportError:  # pragma: no cover - 兼容改名
    from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner  # type: ignore

from .miner_client import BacktestOutcome, MinerClient, MinerError

# miner 的 metrics key → RD-Agent feedback 层硬编码的 key
METRIC_KEY_MAP: dict[str, str] = {
    "IC": "IC",
    "annualized_return_with_cost": "1day.excess_return_with_cost.annualized_return",
    "max_drawdown": "1day.excess_return_with_cost.max_drawdown",
}
# 额外保留（feedback 只取上面三个，多出来的无害且对人有信息量）
EXTRA_METRIC_KEY_MAP: dict[str, str] = {
    "RankIC": "RankIC",
    "ICIR": "ICIR",
}


def _factor_code_of(workspace: Any) -> dict[str, str] | None:
    """从一个 FactorFBWorkspace 里取出 {name, code}。"""
    files = getattr(workspace, "file_dict", None) or {}
    code = files.get("factor.py")
    if not code:
        return None
    task = getattr(workspace, "target_task", None)
    name = getattr(task, "factor_name", None) or getattr(task, "name", None)
    if not name:
        return None
    return {"name": str(name), "code": code}


def collect_factors(experiments: Iterable[Any]) -> list[dict[str, str]]:
    """按 RD-Agent 的数据结构收集 [{name, code}]（同名后者覆盖前者）。"""
    out: dict[str, str] = {}
    for exp in experiments or []:
        for workspace in getattr(exp, "sub_workspace_list", None) or []:
            item = _factor_code_of(workspace)
            if item:
                out[item["name"]] = item["code"]
    return [{"name": n, "code": c} for n, c in out.items()]


def metrics_to_series(metrics: dict[str, Any]) -> pd.Series:
    """把 miner 的 metrics 映射成 feedback 层要的 Series（三个 key 是硬要求）。

    **Series 必须有名字（字符串 "0"）**：上游 `feedback.process_results` 做的是
        pd.DataFrame(current_result).rename(columns={"0": "Current Result"})
    —— 无名 Series 建成 DataFrame 后列名是**整数 0**，rename（字符串 key）匹配不上，
    后面取 `row["Current Result"]` 就 `KeyError`。上游自己的 `exp.result` 来自
    `pd.read_csv(..., index_col=0).iloc[:, 0]`，列名正是字符串 "0"。这里对齐它。
    """
    data: dict[str, float] = {}
    for src, dst in {**METRIC_KEY_MAP, **EXTRA_METRIC_KEY_MAP}.items():
        value = metrics.get(src)
        if value is None:
            continue
        try:
            data[dst] = float(value)
        except (TypeError, ValueError):
            continue
    missing = [dst for dst in METRIC_KEY_MAP.values() if dst not in data]
    if missing:
        # 不静默：缺 key 一定会在 feedback 层炸成 KeyError，这里提前说清楚
        raise FactorEmptyError(
            "miner 返回的 metrics 缺少 feedback 层必需的 key："
            f"{missing}（miner metrics={list(metrics)}）"
        )
    series = pd.Series(data, dtype="float64")
    series.name = "0"  # ← 见 docstring：上游按字符串 "0" 重命名列
    return series


class FactorMinerRunner(QlibFactorRunner):
    """用 factor-miner-mcp 替代 qrun/Docker/conda 的 QlibFactorRunner。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = MinerClient(
            url=os.environ.get("FACTOR_MINER_MCP_URL"),
            poll_interval=float(os.environ.get("FACTOR_MINER_POLL_SEC", "10")),
        )
        self.profile = os.environ.get("FACTOR_MINER_PROFILE", "full")

    # ── 参数 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _windows() -> dict[str, str] | None:
        """把 RD-Agent 的训练/验证/测试窗口透传给 miner（不设则由 miner 用模板默认）。"""
        try:
            from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting

            fbps = FactorBasePropSetting()
            fields = ("train_start", "train_end", "valid_start", "valid_end", "test_start")
            windows = {f: getattr(fbps, f) for f in fields if getattr(fbps, f, None)}
            if getattr(fbps, "test_end", None):
                windows["test_end"] = fbps.test_end
            return windows or None
        except Exception:  # noqa: BLE001 - 拿不到就用 miner 默认，不因此失败
            return None

    # ── 主流程 ───────────────────────────────────────────────────────────
    def develop(self, exp: Any) -> Any:
        # 上游 develop() 的第一件事：确保 baseline 实验已执行。
        # feedback 层会读 `exp.based_experiments[-1].result` 当 SOTA 对比，
        # 少了这一步第一轮就会在 feedback 里炸 `KeyError: 'SOTA Result'`
        # （踩过：我第一版 override 里漏了这段，循环走到 feedback 才崩）。
        based = getattr(exp, "based_experiments", None) or []
        if based and getattr(based[-1], "result", None) is None:
            self._ensure_baseline(based[-1])

        new_factors = collect_factors([exp])
        if not new_factors:
            raise FactorEmptyError("本轮没有任何 factor.py 产出")

        sota = collect_factors(based)

        try:
            outcome: BacktestOutcome = self.client.backtest(
                sota=sota,
                new_factors=new_factors,
                profile=self.profile,
                windows=self._windows(),
            )
        except MinerError as exc:
            # 基础设施问题（槽位被占/不可达）也要走 RD-Agent 的跳过语义，
            # 否则一个卡住的槽位会把整条循环打死。
            raise FactorEmptyError(f"miner 调用失败：{exc}") from exc

        if not outcome.ok:
            raise FactorEmptyError(
                f"miner 回测未通过：{str(outcome.error)[:300]}"
                f"（dedup_dropped={outcome.raw.get('dedup_dropped')}）"
            )

        series = metrics_to_series(outcome.metrics)
        self._write_back(exp, series, outcome)

        # 与上游完全一致的取法：从 qlib_res.csv 读回（列名即字符串 "0"），
        # 这样 exp.result 的形状/名字与 RD-Agent 自己的 execute() 产物逐字节等价。
        workspace = getattr(exp, "experiment_workspace", None)
        ws_path = Path(getattr(workspace, "workspace_path", "")) if workspace else None
        result_series = series
        if ws_path and (ws_path / "qlib_res.csv").exists():
            try:
                result_series = pd.read_csv(ws_path / "qlib_res.csv", index_col=0).iloc[:, 0]
            except Exception:  # noqa: BLE001 - 读回失败就用内存里的
                result_series = series
        exp.result = result_series
        exp.stdout = (
            f"[factor-miner] profile={self.profile} "
            f"elapsed={outcome.elapsed_sec}s factors={[f['name'] for f in new_factors]}\n"
            + "\n".join(f"{k}: {v}" for k, v in series.items())
        )
        return exp

    # ── baseline（上游 develop() 的前置步骤）───────────────────────────────
    def _ensure_baseline(self, baseline: Any) -> None:
        """执行 baseline 实验 —— 上游这一步跑的是纯 Alpha20 基线回测。

        为什么必须有：`feedback.py:72` 读 `sota_result = exp.based_experiments[-1].result`
        当作 SOTA 对比；baseline 没跑 → 它是 None → 上游 `process_results` 里
        `pd.DataFrame(None)` 没有 "SOTA Result" 列 → `KeyError: 'SOTA Result'`
        （这个报错完全指不到病因，我第一版桥就卡在这里）。

        miner 侧为此加了 `baseline=True` 参数（纯 Alpha20，无因子），
        **并沿用与因子轮相同的 profile/windows** —— 否则基线（csi300/2008-2018）
        与因子轮（如 smoke 的 30 票/2019）不可比，"是否优于 SOTA" 就失去意义。
        """
        try:
            outcome: BacktestOutcome = self.client.backtest(
                sota=[],
                new_factors=[],
                profile=self.profile,
                windows=self._windows(),
                baseline=True,
            )
        except MinerError as exc:
            raise FactorEmptyError(f"baseline 回测无法提交：{exc}") from exc

        if not outcome.ok:
            raise FactorEmptyError(
                f"baseline 回测失败：{str(outcome.error)[:300]}"
            )

        series = metrics_to_series(outcome.metrics)
        self._write_back(baseline, series, outcome)
        baseline.result = series

    # ── 产物写回（对齐 QlibFBWorkspace.execute 的产物契约）───────────────
    def _write_back(self, exp: Any, series: pd.Series, outcome: BacktestOutcome) -> None:
        workspace = getattr(exp, "experiment_workspace", None)
        ws_path: Path | None = Path(getattr(workspace, "workspace_path", "."))
        if workspace is None or ws_path is None:
            return
        ws_path.mkdir(parents=True, exist_ok=True)

        # 1) qlib_res.csv —— 上游用 pd.read_csv(..., index_col=0).iloc[:, 0]
        series.to_csv(ws_path / "qlib_res.csv")

        # 2) ret.pkl —— 上游用它画净值曲线（缺它 execute() 直接返回 None）
        curve = outcome.net_curve or []
        if curve:
            frame = pd.DataFrame(curve)
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"])
                frame = frame.set_index("date")
            frame.to_pickle(ws_path / "ret.pkl")

        # 3) 顺手把原始返回留档，便于事后核对指标来源
        try:
            import json

            (ws_path / "miner_result.json").write_text(
                json.dumps(outcome.raw, ensure_ascii=False, default=str)[:2_000_000]
            )
        except Exception:  # noqa: BLE001
            pass
