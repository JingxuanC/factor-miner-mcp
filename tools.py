"""tools.py — 因子挖掘 MCP 的工具注册表（7 个工具）。

从 Athena py-sidecar server.py 摘出 factor 域子集，独立成项目后不再
依赖 66+ 工具的全量注册表。handler 实现在 factor_worker.py /
factors.py / models.py，qlib 相关重依赖全部惰性导入（未装 pyqlib
也可启动服务、调用 compute_factors 等纯 pandas 工具）。
"""

from __future__ import annotations

import json
from typing import Optional

from factors import FactorEngine
from models import ModelEngine


# ── Toolkit interface ──
class ToolDef:
    def __init__(self, name: str, description: str, inputSchema: dict):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

    def to_dict(self):
        return {"name": self.name, "description": self.description, "inputSchema": self.inputSchema}


TOOLS: dict[str, ToolDef] = {}
HANDLERS: dict[str, callable] = {}


def tool(name: str, description: str, properties: dict, required: Optional[list] = None):
    """Decorator to register a tool."""
    def deco(fn):
        TOOLS[name] = ToolDef(name, description, {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
        })
        HANDLERS[name] = fn
        return fn
    return deco


# ═══════════════════════════════════════════════════════════════
# FactorMiner 五件套：沙箱执行 → 评估 → 全量回测 → OOS 准入 → 日频/衰减巡检
# ═══════════════════════════════════════════════════════════════

@tool("factor_execute", "Run a factor.py in the sandbox (whitelist imports, rlimit, 120s timeout) "
      "against daily_pv debug/full h5, then run the eval battery. "
      "Returns JSON string: {stdout, eval_ok, eval_detail}.",
      {"code": {"type": "string", "description": "factor.py source code (reads daily_pv.h5, writes result.h5)"},
       "debug": {"type": "boolean", "description": "true=debug dataset (100股x2年), false=full", "default": True}},
      required=["code"])
def factor_execute(code: str, debug: bool = True) -> str:
    from factor_worker import factor_execute as _impl
    return _impl(code, debug)


@tool("factor_backtest", "Full qlib backtest: re-run SOTA factors + new factors (sandbox, exec cache), "
      "concat with Alpha20 baseline, qrun (1800s timeout), parse recorder metrics. "
      "Returns JSON string: {ok, dedup_dropped, sota_broken, metrics, correlations, net_values, error, traceback}.",
      {"sota": {"type": "array", "description": "SOTA factors: [{name, code}, ...]"},
       "new_factors": {"type": "array", "description": "New factors to evaluate: [{name, code}, ...]"},
       "profile": {"type": "string", "enum": ["full", "smoke"], "default": "full"},
       "windows": {"type": "object", "description": "Optional backtest windows: train_start/train_end/valid_start/valid_end/test_start/test_end"}},
      required=["sota", "new_factors"])
def factor_backtest(sota: list, new_factors: list, profile: str = "full", windows: Optional[dict] = None) -> str:
    from factor_worker import factor_backtest as _impl
    return _impl(sota, new_factors, profile, windows)


@tool("factor_oos_check", "Production-admission OOS check: run the factor twice — "
      "mining window (test 2017→now) and pure out-of-sample window (test 2021-01→now) — "
      "report IC/annualized/max_drawdown for both plus relative decay. "
      "Returns JSON string: {oos: {ic, annualized_return, max_drawdown}, mining: {...}, decay: float|null, ok, error}.",
      {"code": {"type": "string", "description": "factor.py source code"},
       "name": {"type": "string", "description": "factor name"}},
      required=["code", "name"])
def factor_oos_check(code: str, name: str) -> str:
    from factor_worker import factor_oos_check as _impl
    return _impl(code, name)


@tool("factor_daily_compute", "Daily post-close compute of live factors: refresh "
      "daily_pv_all.h5 if stale (in-process gen_data), run each factor.py via sandbox, take the "
      "latest datetime row per instrument, write Redis dfactor:{symbol} (TTL 48h). "
      "Returns JSON string: {written, symbols, errors}.",
      {"factors": {"type": "array", "description": "Live factors: [{name, code}, ...]"}},
      required=["factors"])
def factor_daily_compute(factors: list) -> str:
    from factor_worker import factor_daily_compute as _impl
    return _impl(factors)


@tool("factor_recent_ic", "Weekly decay probe: trailing N-trading-day mean "
      "cross-sectional Pearson IC (factor vs next-day return), pure pandas, no qlib. "
      "Returns JSON string: {ok, ic, days, error}.",
      {"code": {"type": "string", "description": "factor.py source code"},
       "name": {"type": "string", "description": "factor name"},
       "lookback_days": {"type": "integer", "description": "Trailing trading days (default 60)", "default": 60}},
      required=["code", "name"])
def factor_recent_ic(code: str, name: str, lookback_days: int = 60) -> str:
    from factor_worker import factor_recent_ic as _impl
    return _impl(code, name, lookback_days)


# ═══════════════════════════════════════════════════════════════
# 纯 pandas 扩展工具（无 qlib 依赖）
# ═══════════════════════════════════════════════════════════════

factor_engine = FactorEngine()
model_engine = ModelEngine()


def _compute_factors_handler(symbol: str = "", klines: list = None) -> str:
    return json.dumps(factor_engine.compute({"symbol": symbol, "klines": klines or []}))


def _predict_handler(symbol: str = "", factors: dict = None) -> str:
    return json.dumps(model_engine.predict({"symbol": symbol, "factors": factors or {}}))


HANDLERS.setdefault("compute_factors", _compute_factors_handler)
HANDLERS.setdefault("predict", _predict_handler)

# compute_factors / predict 无 @tool schema，由 server 端硬编码补（见 server.py）
EXTRA_SCHEMAS = {
    "compute_factors": {"name": "compute_factors", "description": "Compute Alpha158 factors from OHLCV data",
                        "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "klines": {"type": "array"}}}},
    "predict": {"name": "predict", "description": "ML model prediction from factor values",
                "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "factors": {"type": "object"}}}},
}
