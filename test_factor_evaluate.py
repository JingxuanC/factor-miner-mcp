"""factor_evaluate 的契约测试 —— 不需要 qlib、不需要 h5、不需要沙箱。

为什么这个工具存在（2026-09 实测的断链）：

    factor_execute   → 只回契约检查 {eval_ok, eval_detail}，**不回因子值**
    factor_tearsheet → 要调用方自带 factor_values + klines（自己造不出来）
    ⇒ 没有任何工具能把「一段因子代码」变成「专业指标」，因子看板卡在这一步

factor_evaluate 把仓库里已有的两块接起来：`_run_factor_window`（沙箱跑代码，
因子值与收盘价取自**同一次窗口切片**）+ `analytics.factor_tearsheet`。

这里钉住三条契约：
  1. 输出是**专业口径**：tearsheet 原样透传 + 单调性/半衰期/t 值摘要；
  2. 输出**永不抛异常**（沙箱失败、数据缺失、样本不足都走 JSON error）；
  3. 假设缺失只是**标记**，不影响拿到指标（必填是入库闸门，不是评估闸门）。

跑法：
    python3 test_factor_evaluate.py
或 pytest 下自动收集。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import factor_worker as fw  # noqa: E402


# ── 合成「因子值 + 收盘价」：次日收益由前一日因子驱动，所以 IC 为正 ──────────

def _synthetic(n_dates: int = 80, n_inst: int = 12, seed: int = 0):
    """返回 (factor, close) 两个 (datetime, instrument) 索引的 Series。"""
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    insts = [f"s{i:02d}" for i in range(n_inst)]
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    factor = pd.Series(rng.normal(size=n_dates * n_inst), index=idx, name="factor")

    wide = factor.unstack()  # date × instrument
    # 次日收益 = 0.002 × 前一日因子 + 极小噪声 ⇒ 因子对收益有正向预测力
    ret = 0.002 * wide.shift(1) + 1e-4 * rng.normal(size=wide.shape)
    close = (1.0 + ret.fillna(0.0)).cumprod().stack()
    close.index.names = ["datetime", "instrument"]
    return factor, close.rename("close")


class _Env:
    """把 DATA_DIR 指到临时目录并伪造沙箱执行结果。"""

    def __init__(self, tmp: Path, fake=_synthetic, raise_exc: Exception | None = None):
        self.tmp = tmp
        self.fake = fake
        self.raise_exc = raise_exc
        self._old_data_dir = fw.DATA_DIR
        self._old_runner = fw._run_factor_window

    def __enter__(self):
        fw.DATA_DIR = self.tmp

        def _runner(code, data_h5, window_days=None):
            if self.raise_exc is not None:
                raise self.raise_exc
            return self.fake()

        fw._run_factor_window = _runner
        (self.tmp / "daily_pv_all.h5").write_bytes(b"fake")
        (self.tmp / "daily_pv_debug.h5").write_bytes(b"fake")
        return self

    def __exit__(self, *exc):
        fw.DATA_DIR = self._old_data_dir
        fw._run_factor_window = self._old_runner
        return False


# ═══════════════ 1. 纯函数：单调性 / 半衰期 / t 值 ═══════════════

def test_quantile_monotonicity_is_strict_not_almost():
    """rho 高 ≠ 单调：严格单调为 True，0.9 的“几乎单调”必须为 False。"""
    inc = fw.quantile_monotonicity(
        {"q1": 0.001, "q2": 0.002, "q3": 0.003, "q4": 0.004, "q5": 0.005}, 5)
    assert inc["monotonic"] is True and inc["direction"] == "increasing"
    assert inc["rho"] > 0.999 and inc["values"] == [0.001, 0.002, 0.003, 0.004, 0.005]

    dec = fw.quantile_monotonicity({f"q{q}": 0.01 - q * 0.001 for q in range(1, 6)}, 5)
    assert dec["monotonic"] is True and dec["direction"] == "decreasing"
    assert dec["rho"] < -0.999

    # 前四档都递增、最后一档回撤 → rho 仍高达 0.7 但**不是**单调
    nearly = fw.quantile_monotonicity(
        {"q1": 0.001, "q2": 0.002, "q3": 0.003, "q4": 0.004, "q5": 0.0035}, 5)
    assert nearly["monotonic"] is False and nearly["direction"] == "non_monotonic"
    assert nearly["rho"] is not None and nearly["rho"] > 0.5

    # 4/5 递增但末档掉到最低 → rho 直接掉到 0：只报 rho 会看不出「哪里不对」
    broken = fw.quantile_monotonicity(
        {"q1": 0.001, "q2": 0.002, "q3": 0.003, "q4": 0.004, "q5": 0.0005}, 5)
    assert broken["monotonic"] is False and broken["rho"] < 0.5


def test_quantile_monotonicity_refuses_to_guess():
    """任一档位缺失就报 unknown，不拿剩下的档位硬算。"""
    for bad in ({"q1": 0.1, "q2": None, "q3": 0.3, "q4": 0.4, "q5": 0.5}, {}, None):
        out = fw.quantile_monotonicity(bad, 5)
        assert out["monotonic"] is None and out["rho"] is None
        assert out["direction"] == "unknown" and out["values"] is None


def test_ic_half_life_interpolates_and_refuses_noise():
    """半衰期决定持有期：线性插值；IC 不降反升时返回 None（噪声不是衰减）。"""
    # 0.1 → 0.0：穿过一半（0.05）落在 1d..5d 的 3/4 处 = 3.0
    assert fw.ic_half_life({"1d": 0.1, "5d": 0.0}) == 3.0
    # 恰好在节点上踩到一半 → 就报那个节点（5d），不硬插值到节点之外
    assert fw.ic_half_life({"1d": 0.064, "5d": 0.032}) == 5.0
    # 更常见的形状：1d=0.064, 5d=0.041, 10d=0.022 → 在 5d..10d 之间穿过 0.032
    hl = fw.ic_half_life({"1d": 0.064, "5d": 0.041, "10d": 0.022})
    assert 5.0 < hl < 10.0
    # 10d 才跌到一半以内、5d 还在半衰线之上 → 必须落在 5..10 之间而不是报 10
    assert fw.ic_half_life({"1d": 0.064, "5d": 0.05, "10d": 0.02}) < 10.0
    # 衰减不动 / 反而升高 / 只剩一个周期 / 基准为 0 → 一律 None
    assert fw.ic_half_life({"1d": 0.02, "5d": 0.03, "10d": 0.04}) is None
    assert fw.ic_half_life({"1d": 0.05}) is None
    assert fw.ic_half_life({"1d": 0.0, "5d": 0.0}) is None
    assert fw.ic_half_life({}) is None
    assert fw.ic_half_life(None) is None


def test_ic_t_stat():
    """t = mean/std × sqrt(N)；样本不足或 std<=0 时 None，不编数字。"""
    assert fw.ic_t_stat(0.05, 0.1, 100) == 0.5 * 10
    assert fw.ic_t_stat(0.05, 0.1, 1) is None
    assert fw.ic_t_stat(0.05, 0.0, 100) is None
    assert fw.ic_t_stat(None, 0.1, 100) is None


def test_eval_periods_normalized():
    """持有期入参：去重、升序、剔除非正/非数/超限值，空则默认 [1,5,10]。"""
    assert fw._eval_periods(None) == [1, 5, 10]
    assert fw._eval_periods([10, 1, 5, 1]) == [1, 5, 10]
    assert fw._eval_periods(["5", 0, -3, "x", 999]) == [5]
    assert fw._eval_periods([]) == [1, 5, 10]


# ═══════════════ 2. 端到端：代码 → 专业指标 ═══════════════

def test_factor_evaluate_returns_professional_metrics(tmp: Path):
    with _Env(tmp):
        out = json.loads(fw.factor_evaluate("factor.py 内容", hypothesis="动量：强者恒强"))

    assert out["ok"] is True, out
    assert out["error"] == ""
    assert out["hypothesis"] == "动量：强者恒强" and out["hypothesis_missing"] is False
    assert out["dataset"] == "full" and out["window_days"] == fw.WINDOW_DAYS
    assert out["n_dates"] == 80 and out["n_symbols"] == 12
    assert set(out["eval_window"]) == {"start", "end"}

    # tearsheet 原样透传（界面四张图直接吃这个）
    tear = out["tearsheet"]
    for key in ("quantile_returns", "long_short", "ic", "turnover"):
        assert key in tear, (key, tear)
    assert tear["ic"]["mean"] is not None

    # 专业摘要层：单调性 / 半衰期 / t 值
    assert set(out["monotonicity"]) == {"1d", "5d", "10d"}
    assert out["monotonicity"]["1d"]["direction"] == "increasing"
    assert out["monotonicity"]["1d"]["monotonic"] is True
    assert out["ic_t_stat"] is not None and out["ic_t_stat"] > 0


def test_factor_evaluate_flags_missing_hypothesis_but_still_measures(tmp: Path):
    """假设缺失只标记、不拦评估：必填是入库闸门，不是评估闸门。"""
    with _Env(tmp):
        out = json.loads(fw.factor_evaluate("factor.py 内容", hypothesis="  "))

    assert out["ok"] is True
    assert out["hypothesis_missing"] is True
    # 原样透传，不替调用方改写（只标记缺失）
    assert out["hypothesis"] == "  "
    assert out["tearsheet"]["ic"]["mean"] is not None


def test_factor_evaluate_debug_dataset_uses_full_history(tmp: Path):
    """debug 集很小，窗口必须为 0（全量）——否则调试集被截到没几天。"""
    with _Env(tmp):
        out = json.loads(fw.factor_evaluate("factor.py 内容", dataset="debug"))
    assert out["ok"] is True and out["dataset"] == "debug" and out["window_days"] == 0


# ═══════════════ 3. 失败路径：永不抛异常 ═══════════════

def test_factor_evaluate_missing_data_file(tmp: Path):
    with _Env(tmp):
        (tmp / "daily_pv_all.h5").unlink()
        out = json.loads(fw.factor_evaluate("factor.py 内容"))
    assert out["ok"] is False and "data file missing" in out["error"]
    assert out["tearsheet"] is None


def test_factor_evaluate_sandbox_failure_is_reported_not_raised(tmp: Path):
    """沙箱失败（白名单违规/超时/非零退出）必须连原因一起回，不能吞。"""
    with _Env(tmp, raise_exc=RuntimeError("whitelist violation: os")):
        out = json.loads(fw.factor_evaluate("import os"))
    assert out["ok"] is False
    assert "whitelist violation: os" in out["error"]
    assert out["traceback"] and "RuntimeError" in out["traceback"]


def test_factor_evaluate_passes_through_tearsheet_error(tmp: Path):
    """样本不足时 tearsheet 走 error 分支 —— 原因要透传，不能变成“评估失败”。"""
    with _Env(tmp, fake=lambda: _synthetic(n_dates=3, n_inst=12)):
        out = json.loads(fw.factor_evaluate("factor.py 内容"))
    assert out["ok"] is False
    assert "样本不足" in out["error"] or "不足" in out["error"]
    assert out["error"] != "factor_evaluate worker exception"


def test_factor_evaluate_worker_exception_is_json(tmp: Path):
    """连内部异常也要落成 JSON（tool 通道不接受抛异常）。"""
    garbage = lambda: (object(), object())  # noqa: E731 — 沙箱"成功"但返回值不是 Series
    with _Env(tmp, fake=garbage):
        out = json.loads(fw.factor_evaluate("factor.py 内容"))
    assert out["ok"] is False and out["error"] == "factor_evaluate worker exception"
    assert "AttributeError" in out["traceback"]


# ═══════════════ 4. 注册与传输层 ═══════════════

def test_tool_registered_with_schema():
    import tools

    assert "factor_evaluate" in tools.TOOLS and "factor_evaluate" in tools.HANDLERS
    schema = tools.TOOLS["factor_evaluate"].to_dict()
    assert schema["inputSchema"]["required"] == ["code"]
    assert "hypothesis" in schema["inputSchema"]["properties"]
    # 重负载：必须走异步队列，否则一次评估占住 HTTP 连接
    import server

    assert "factor_evaluate" in server.ASYNC_TOOLS
    # 网关参数校验读的是函数注解（本仓库用了 from __future__ import annotations，
    # 注解是字符串，由 mcp_common._ann_type 还原）—— 必须能还原成基础类型
    # 才会被强转；还原不出来就等于网关不校验这个参数。
    import inspect

    from mcp_common import _ann_type

    params = inspect.signature(tools.HANDLERS["factor_evaluate"]).parameters
    assert _ann_type(params["code"].annotation) is str
    assert _ann_type(params["quantiles"].annotation) is int
    assert _ann_type(params["dataset"].annotation) is str
    assert _ann_type(params["window_days"].annotation) is int


if __name__ == "__main__":
    # 自带 runner（镜像里可能没有 pytest）
    import tempfile
    import traceback

    plain = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in plain:
        with tempfile.TemporaryDirectory() as td:
            try:
                if fn.__code__.co_argcount:
                    fn(Path(td))
                else:
                    fn()
                print(f"PASS  {fn.__name__}", flush=True)
            except Exception:
                failures += 1
                print(f"FAIL  {fn.__name__}", flush=True)
                traceback.print_exc()
    print("FAILURES:", failures)
    raise SystemExit(1 if failures else 0)
