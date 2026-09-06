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


# ═══════════════════════════════════════════════════════════════
# ML 滚动训练（RollingTrainer：Alpha158 因子 + 次日收益标签 → LGBM）
# lightgbm 缺失时 trainer 内部优雅降级（status=error），服务照常启动。
# ═══════════════════════════════════════════════════════════════

@tool("ml_train_rolling", "Rolling LGBM training on expanding window: computes Alpha158-style "
      "factors + next-day-return labels from raw OHLCV klines, trains LGBMRegressor, "
      "evaluates IC/rank_ic/sharpe on the validation tail, saves model to disk. "
      "Returns JSON string: {status, ic, rank_ic, sharpe, n_samples, n_features, model_path}.",
      {"klines_list": {"type": "array", "description": "[{symbol, klines: [{date,open,high,low,close,volume}, ...]}, ...] (>=30 klines per symbol)"},
       "validation_days": {"type": "integer", "description": "Tail samples for validation (default 20)", "default": 20},
       "early_stopping_rounds": {"type": "integer", "description": "LGBM patience (default 20)", "default": 20}},
      required=["klines_list"])
def ml_train_rolling(klines_list: list, validation_days: int = 20, early_stopping_rounds: int = 20) -> str:
    from trainer import get_trainer
    return json.dumps(get_trainer().train(klines_list, validation_days, early_stopping_rounds))


@tool("ml_predict", "Next-day return predictions from the trained rolling model. "
      "Prefers Redis factor:{symbol} snapshots, falls back to on-the-fly factor compute. "
      "Returns JSON string: {status, n_predicted, predictions: {symbol: score}}.",
      {"klines_list": {"type": "array", "description": "[{symbol, klines: [...]}, ...] (klines used when no Redis snapshot)"}},
      required=["klines_list"])
def ml_predict(klines_list: list) -> str:
    from trainer import get_trainer
    return json.dumps(get_trainer().predict(klines_list))


@tool("ml_metrics", "Rolling trainer status: last training date, per-day metrics "
      "(ic/rank_ic/sharpe/top10_return), model existence, feature names.",
      {})
def ml_metrics() -> str:
    from trainer import get_trainer
    return json.dumps(get_trainer().get_metrics())

# ═══════════════════════════════════════════════════════════════
# 因子评估与组合分析五件套（analytics.py）
# 纯 numpy/pandas/scipy 内置实现开箱可用；pypfopt/hmmlearn/ruptures/arch
# 惰性导入做可选增强，输出 method 字段标注实际实现。全部轻负载同步。
# ═══════════════════════════════════════════════════════════════

@tool("factor_tearsheet", "Alphalens-style factor evaluation (hand-written pandas, no alphalens "
      "dependency): quantile returns per forward period, long-short spread, daily cross-sectional "
      "Spearman IC (mean/IR/decay), top/bottom bucket turnover. "
      "Returns JSON string: {quantile_returns, long_short, ic: {mean, ir, series_summary, decay}, turnover, method}.",
      {"factor_values": {"type": "array", "description": "[{date, symbol, value}, ...]"},
       "klines": {"type": "object", "description": "{symbol: [{date, close}, ...]}"},
       "quantiles": {"type": "integer", "description": "Number of quantile buckets (default 5)", "default": 5},
       "periods": {"type": "array", "description": "Forward return periods in days (default [1,5,10])", "default": [1, 5, 10]}},
      required=["factor_values", "klines"])
def factor_tearsheet(factor_values: list, klines: dict, quantiles: int = 5,
                     periods: Optional[list] = None) -> str:
    import analytics
    return analytics.factor_tearsheet(factor_values, klines, quantiles, periods)


@tool("portfolio_optimize", "Portfolio optimization: hrp (hierarchical risk parity, "
      "López de Prado 2016) / equal / min_variance (numpy analytic, long-only) built in; "
      "mean_variance via pypfopt when installed (lazy import). "
      "Returns JSON string: {weights, expected_return, volatility, sharpe, method}.",
      {"symbols": {"type": "array", "description": "Symbols to allocate"},
       "klines": {"type": "object", "description": "{symbol: [{date, close}, ...]}"},
       "method": {"type": "string", "enum": ["hrp", "equal", "min_variance", "mean_variance"], "default": "hrp"},
       "lookback": {"type": "integer", "description": "Estimation window in days (default 120)", "default": 120}},
      required=["symbols", "klines"])
def portfolio_optimize(symbols: list, klines: dict, method: str = "hrp",
                       lookback: int = 120) -> str:
    import analytics
    return analytics.portfolio_optimize(symbols, klines, method, lookback)


@tool("regime_detect", "Market regime detection (bull/bear/range): built-in rule engine "
      "(20d momentum + realized vol thresholds); GaussianHMM (ret+vol features) via hmmlearn "
      "when installed (lazy import). Returns JSON string: "
      "{current_regime, regime_history, regime_stats, method}.",
      {"klines": {"type": "array", "description": "[{date, close, volume?}, ...] (index or single stock)"},
       "n_regimes": {"type": "integer", "description": "2 (bull/bear) or 3 (+range), default 3", "default": 3}},
      required=["klines"])
def regime_detect(klines: list, n_regimes: int = 3) -> str:
    import analytics
    return analytics.regime_detect(klines, n_regimes)


@tool("change_point", "Structural change-point detection: built-in CUSUM (Page 1954) + binary "
      "segmentation on mean shifts; PELT (Killick 2012) via ruptures when installed (lazy import). "
      "Returns JSON string: {change_points: [{date, index, significance}], method}.",
      {"series": {"type": "array", "description": "[{date, value}, ...]"},
       "method": {"type": "string", "enum": ["auto", "cusum_binseg", "pelt"], "default": "auto"},
       "max_bkps": {"type": "integer", "description": "Max breakpoints (default 5)", "default": 5}},
      required=["series"])
def change_point(series: list, method: str = "auto", max_bkps: int = 5) -> str:
    import analytics
    return analytics.change_point(series, method, max_bkps)


@tool("vol_forecast", "Volatility forecast: built-in EWMA (RiskMetrics 1996, lambda=0.94, flat "
      "multi-day extrapolation) + Parkinson high/low reference when OHLC given; GARCH(1,1) via "
      "arch when installed (lazy import). Returns JSON string: "
      "{forecast: [{day, vol}], current_vol, method} (daily vol as decimal).",
      {"klines": {"type": "array", "description": "[{date, close, high?, low?}, ...]"},
       "horizon": {"type": "integer", "description": "Forecast days ahead (default 5)", "default": 5},
       "method": {"type": "string", "enum": ["auto", "ewma", "garch"], "default": "auto"}},
      required=["klines"])
def vol_forecast(klines: list, horizon: int = 5, method: str = "auto") -> str:
    import analytics
    return analytics.vol_forecast(klines, horizon, method)


# compute_factors / predict 无 @tool schema，由 server 端硬编码补（见 server.py）
EXTRA_SCHEMAS = {
    "compute_factors": {"name": "compute_factors", "description": "Compute Alpha158 factors from OHLCV data",
                        "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "klines": {"type": "array"}}}},
    "predict": {"name": "predict", "description": "ML model prediction from factor values",
                "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "factors": {"type": "object"}}}},
}
