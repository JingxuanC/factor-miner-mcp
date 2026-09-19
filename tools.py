"""tools.py — 因子挖掘 MCP 的工具注册表（7 个工具）。

从 Athena py-sidecar server.py 摘出 factor 域子集，独立成项目后不再
依赖 66+ 工具的全量注册表。handler 实现在 factor_worker.py /
factors.py / models.py，qlib 相关重依赖全部惰性导入（未装 pyqlib
也可启动服务、调用 compute_factors 等纯 pandas 工具）。
"""

from __future__ import annotations

import json
from typing import Any, Optional

import panel
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
            "required": required or [],
        })
        HANDLERS[name] = fn
        return fn
    return deco


# ═══════════════════════════════════════════════════════════════
# 形状容差参数（统一 panel 契约，见 panel.py）
#
# 注解必须是 Any：网关的参数校验基于**函数的类型注解**（mcp_common._ann_type
# 读 p.annotation），写成 list/dict 会在进入工具函数之前就把另一种形状拒掉 ——
# 函数体内的 panel 归一化根本没机会执行（2026-09-15 实测踩到：
# "参数 'klines' 类型错误：期望 array，实际收到 {...}"）。
# JSON Schema 用 anyOf 如实 advertise 两种形状。
# ═══════════════════════════════════════════════════════════════

_PANEL_SCHEMA = {
    "anyOf": [{"type": "array"}, {"type": "object"}],
    "description": "接受多种等价形状（panel.py 契约）：扁平 bar 数组 "
                   "[{date,open,high,low,close},...] / {symbol: [bars]} / "
                   "{columns:[...], rows:[[...]]} 列式面板 / 宽表 {symbol:{date:close}} / "
                   "长表 [{date,symbol,close}]。单序列工具收到多序列会明确报错。",
}

_SERIES_LIST_SCHEMA = {
    "anyOf": [{"type": "array"}, {"type": "object"}],
    "description": "多序列（panel.py 契约）：原生 [{symbol, klines:[...]}, ...]，"
                   "也接受 {symbol: [bars]} 或 [[bars], ...]。",
}


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
      "Returns JSON string: {oos: {ic, annualized_return, max_drawdown}, mining: {...}, "
      "decay: float|null, mining_net_curve: [{date, i, value}], oos_net_curve: [...], ok, error}. "
      "The two net-value curves come from the report qlib already produced for each window "
      "(downsampled to <=400 points); drawing them together is the visual read of `decay`.",
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
      "Returns JSON string: {ok, ic, days, series, error}. `series` is the per-day "
      "IC breakdown [{date, ic, n}] (ascending by date) for decay curves — it comes "
      "from the same groupby that already produced `ic`, so it costs nothing extra.",
      {"code": {"type": "string", "description": "factor.py source code"},
       "name": {"type": "string", "description": "factor name"},
       "lookback_days": {"type": "integer", "description": "Trailing trading days (default 60)", "default": 60}},
      required=["code", "name"])
def factor_recent_ic(code: str, name: str, lookback_days: int = 60) -> str:
    from factor_worker import factor_recent_ic as _impl
    return _impl(code, name, lookback_days)


@tool("factor_evaluate", "One-call professional factor evaluation: run factor.py in the sandbox "
      "and return the alphalens-style tearsheet for it — no need to supply factor_values/klines "
      "yourself (this is the difference from factor_tearsheet, and why it closes the code→metrics "
      "gap that factor_execute leaves open with its contract check only). Factor values and close "
      "prices come from the same window slice, so they are exactly aligned. Adds a professional "
      "summary: per-period quantile monotonicity (strict, plus Spearman rho), IC decay half-life in "
      "trading days (decides holding period), IC t-stat, and a hypothesis passthrough that flags a "
      "missing economic hypothesis. "
      "Returns JSON string: {ok, error, tearsheet, monotonicity, ic_half_life_days, ic_t_stat, "
      "hypothesis, hypothesis_missing, dataset, window_days, n_dates, n_symbols, eval_window}.",
      {"code": {"type": "string", "description": "factor.py source code (sandbox: import whitelist, rlimit, 120s timeout)"},
       "hypothesis": {"type": "string", "description": "Economic hypothesis (the professional gate); "
                                                       "absent → hypothesis_missing=true, metrics still returned"},
       "quantiles": {"type": "integer", "description": "Number of quantile buckets (default 5)", "default": 5},
       "periods": {"type": "array", "description": "Forward return periods in trading days (default [1,5,10], max 120)",
                   "default": [1, 5, 10]},
       "dataset": {"type": "string", "enum": ["full", "debug"],
                   "description": "full = daily_pv_all.h5 windowed to FACTOR_MINER_WINDOW_DAYS; debug = small static set",
                   "default": "full"},
       "window_days": {"type": "integer", "description": "Override the trailing trading-day window (0 = full history)"}},
      required=["code"])
def factor_evaluate(code: str, hypothesis: str = "", quantiles: int = 5,
                    periods: Optional[list] = None, dataset: str = "full",
                    window_days: int = None) -> str:
    from factor_worker import factor_evaluate as _impl
    return _impl(code, hypothesis, quantiles, periods, dataset, window_days)


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
# ML 滚动训练（RollingTrainer：Alpha158 因子 + 次日收益标签 → LGBM 或 MASTER）
# lightgbm/torch 缺失时 trainer 内部优雅降级（status=error），服务照常启动。
# ═══════════════════════════════════════════════════════════════

@tool("ml_train_rolling", "Rolling training on expanding window: point-in-time Alpha158-style "
      "factors (recomputed per day on the history prefix, no look-ahead) + next-day-return labels "
      "from raw OHLCV klines. model='lgbm' (default) trains LGBMRegressor; model='master' trains "
      "MASTER (AAAI 2024 股票专用 Transformer，截面 batch + market-guided gating，torch CPU). "
      "Evaluates IC/rank_ic/sharpe on the validation tail, saves model to disk. "
      "Returns JSON string: {status, model_type, ic, rank_ic, sharpe, n_samples, n_features, model_path, ...}.",
      {"klines_list": {**_SERIES_LIST_SCHEMA,
       "description": ">=62 klines per symbol; master 为截面模型，建议 >=10 只股票"},
       "model": {"type": "string", "enum": ["lgbm", "master"], "description": "训练后端（默认 lgbm，向后兼容）", "default": "lgbm"},
       "validation_days": {"type": "integer", "description": "Tail samples for validation (default 20)", "default": 20},
       "early_stopping_rounds": {"type": "integer", "description": "LGBM patience (default 20, 仅 lgbm)", "default": 20},
       "min_history": {"type": "integer", "description": "Min history days before a row is used (default 60, needed by ma_60)", "default": 60},
       "step": {"type": "integer", "description": "Day sampling stride, >1 trades sample count for speed (default 1)", "default": 1},
       "seq_len": {"type": "integer", "description": "MASTER lookback 序列长度（默认 8，仅 master）", "default": 8},
       "epochs": {"type": "integer", "description": "MASTER 训练轮数（默认 3，CPU 保护；仅 master）", "default": 3},
       "d_model": {"type": "integer", "description": "MASTER 隐层维度（默认 64；仅 master）", "default": 64},
       "lr": {"type": "number", "description": "MASTER Adam 学习率（默认 3e-4；仅 master）", "default": 0.0003},
       "max_symbols": {"type": "integer", "description": "MASTER 股票数上限校验（默认 50，CPU 保护；仅 master）", "default": 50}},
      required=["klines_list"])
def ml_train_rolling(klines_list: Any, model: str = "lgbm", validation_days: int = 20,
                     early_stopping_rounds: int = 20, min_history: int = 60, step: int = 1,
                     seq_len: int = 8, epochs: int = 3, d_model: int = 64,
                     lr: float = 3e-4, max_symbols: int = 50) -> str:
    try:
        klines_list = panel.as_series_list(klines_list)
    except panel.PanelError as e:
        return json.dumps({"error": "klines_list 无法解析: %s" % e})
    from trainer import get_trainer
    trainer = get_trainer()
    if model == "master":
        return json.dumps(trainer.train_master(
            klines_list, validation_days=validation_days, min_history=min_history,
            step=step, seq_len=seq_len, epochs=epochs, d_model=d_model, lr=lr,
            max_symbols=max_symbols))
    return json.dumps(trainer.train(klines_list, validation_days, early_stopping_rounds,
                                    min_history=min_history, step=step))


@tool("ml_predict", "Next-day return predictions from the trained rolling model. "
      "model='lgbm' (default) prefers Redis factor:{symbol} snapshots, falls back to on-the-fly "
      "factor compute; model='master' 用最近 seq_len 天特征序列出分（需先 model='master' 训练）。 "
      "Returns JSON string: {status, n_predicted, predictions: {symbol: score}}.",
      {"klines_list": {**_SERIES_LIST_SCHEMA,
       "description": "klines used when no Redis snapshot; master 至少需 seq_len 根"},
       "model": {"type": "string", "enum": ["lgbm", "master"], "description": "预测后端（默认 lgbm，向后兼容）", "default": "lgbm"}},
      required=["klines_list"])
def ml_predict(klines_list: Any, model: str = "lgbm") -> str:
    try:
        klines_list = panel.as_series_list(klines_list)
    except panel.PanelError as e:
        return json.dumps({"error": "klines_list 无法解析: %s" % e})
    from trainer import get_trainer
    trainer = get_trainer()
    if model == "master":
        return json.dumps(trainer.predict_master(klines_list))
    return json.dumps(trainer.predict(klines_list))


@tool("ml_metrics", "Rolling trainer status: last training date, per-day metrics "
      "(ic/rank_ic/sharpe/top10_return), model existence, feature names; "
      "model_type 标注模型类型（lgbm/master），master 块单独报告深度学习模型状态。",
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
      {"factor_values": {**_PANEL_SCHEMA,
                         "description": "因子值面板，接受多种形状（panel.py 契约）；原生 [{date, symbol, value}, ...]"},
       "klines": _PANEL_SCHEMA,
       "quantiles": {"type": "integer", "description": "Number of quantile buckets (default 5)", "default": 5},
       "periods": {"type": "array", "description": "Forward return periods in days (default [1,5,10])", "default": [1, 5, 10]}},
      required=["factor_values", "klines"])
def factor_tearsheet(factor_values: Any, klines: Any, quantiles: int = 5,
                     periods: Optional[list] = None) -> str:
    import analytics
    return analytics.factor_tearsheet(factor_values, klines, quantiles, periods)


@tool("portfolio_optimize", "Portfolio optimization: hrp (hierarchical risk parity, "
      "López de Prado 2016) / equal / min_variance (numpy analytic, long-only) built in; "
      "mean_variance via pypfopt when installed (lazy import). "
      "Returns JSON string: {weights, expected_return, volatility, sharpe, method}.",
      {"symbols": {"type": "array", "description": "Symbols to allocate"},
       "klines": _PANEL_SCHEMA,
       "method": {"type": "string", "enum": ["hrp", "equal", "min_variance", "mean_variance"], "default": "hrp"},
       "lookback": {"type": "integer", "description": "Estimation window in days (default 120)", "default": 120}},
      required=["symbols", "klines"])
def portfolio_optimize(symbols: list, klines: Any, method: str = "hrp",
                       lookback: int = 120) -> str:
    import analytics
    return analytics.portfolio_optimize(symbols, klines, method, lookback)


@tool("regime_detect", "Market regime detection (bull/bear/range): built-in rule engine "
      "(20d momentum + realized vol thresholds); GaussianHMM (ret+vol features) via hmmlearn "
      "when installed (lazy import). Returns JSON string: "
      "{current_regime, regime_history, regime_stats, method}.",
      {"klines": _PANEL_SCHEMA,
       "n_regimes": {"type": "integer", "description": "2 (bull/bear) or 3 (+range), default 3", "default": 3}},
      required=["klines"])
def regime_detect(klines: Any, n_regimes: int = 3) -> str:
    import analytics
    return analytics.regime_detect(klines, n_regimes)


@tool("change_point", "Structural change-point detection: built-in CUSUM (Page 1954) + binary "
      "segmentation on mean shifts; PELT (Killick 2012) via ruptures when installed (lazy import). "
      "Returns JSON string: {change_points: [{date, index, significance}], method}.",
      {"series": _PANEL_SCHEMA,
       "method": {"type": "string", "enum": ["auto", "cusum_binseg", "pelt"], "default": "auto"},
       "max_bkps": {"type": "integer", "description": "Max breakpoints (default 5)", "default": 5}},
      required=["series"])
def change_point(series: Any, method: str = "auto", max_bkps: int = 5) -> str:
    import analytics
    return analytics.change_point(series, method, max_bkps)


@tool("vol_forecast", "Volatility forecast with a real term structure: mean-reverting EWMA "
      "(EWMA current variance + AR(1) mean reversion estimated from rolling realized variance, "
      "phi & long-run vol reported, pure numpy) is the default; pure RiskMetrics EWMA "
      "(lambda=0.94) is available explicitly and is FLAT by construction (no mean reversion, "
      "therefore no term structure); GARCH(1,1) via arch when installed (lazy import); "
      "Parkinson high/low reference when OHLC given. Returns JSON string: "
      "{forecast: [{day, vol}], current_vol, method, term_structure, horizon_ratio, phi?, "
      "long_run_vol?, half_life_days?} (daily vol as decimal).",
      {"klines": _PANEL_SCHEMA,
       "horizon": {"type": "integer", "description": "Forecast days ahead (default 5)", "default": 5},
       "method": {"type": "string", "enum": ["auto", "ewma", "ewma_mr", "garch"], "default": "auto",
                  "description": "auto=有 arch 走 GARCH，否则走均值回复 ewma_mr；"
                                 "ewma=纯 RiskMetrics（多期天然平坦，无期限结构）；"
                                 "ewma_mr=EWMA 当前方差 + AR(1) 均值回复，有期限结构"}},
      required=["klines"])
def vol_forecast(klines: Any, horizon: int = 5, method: str = "auto") -> str:
    import analytics
    return analytics.vol_forecast(klines, horizon, method)


@tool("update_data", "Incremental update of the qlib cn_data daily bars (trading-calendar aligned, "
      "raw+qfq dual path, tencent/mootdx/eastmoney circuit-breaker chain, staging + atomic swap), "
      "then rebuilds the daily_pv_all.h5 factor dataset. Keeps old data on failure. "
      "Runs as an async job — poll with the job_status tool. Requires pyqlib + a mounted cn_data baseline. "
      "Returns JSON string: {ok, exit_code, provider_uri, out_dir}.",
      {"source": {"type": "string", "enum": ["auto", "tencent", "mootdx", "eastmoney"],
                  "description": "Data source chain (default auto: mootdx→tencent→eastmoney)", "default": "auto"},
       "limit": {"type": "integer", "description": "Only update first N symbols (smoke test)", "default": 0},
       "skip_h5": {"type": "boolean", "description": "Skip daily_pv_all.h5 rebuild", "default": False},
       "force": {"type": "boolean", "description": "Force backfill even with no new trading day", "default": False}},
      required=[])
def update_data(source: str = "auto", limit: int = 0, skip_h5: bool = False, force: bool = False) -> str:
    import os
    from factor_miner import update_data as _ud
    provider_uri = os.environ.get("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data")
    out_dir = os.environ.get("FACTOR_MINER_DATA_DIR", "data/factor_mining")
    rc = _ud.run(provider_uri, out_dir, source=source, limit=limit or None,
                 skip_h5=skip_h5, force=force)
    payload = {"ok": rc == 0, "exit_code": rc,
               "provider_uri": provider_uri, "out_dir": out_dir}
    if rc != 0:
        # 显式 error 字段：JobQueue 据此把 job 标成 status="error"（不再一律 done）
        if rc == _ud.EXIT_LOCKED:
            payload["error"] = "已有 update_data 任务在运行，本次提交被并发单飞锁拒绝"
            payload["locked"] = True
        else:
            payload["error"] = f"update_data failed with exit_code={rc}"
    return json.dumps(payload, ensure_ascii=False)


# compute_factors / predict 无 @tool schema，由 server 端硬编码补（见 server.py）
EXTRA_SCHEMAS = {
    "compute_factors": {"name": "compute_factors", "description": "Compute Alpha158 factors from OHLCV data",
                        "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "klines": {"type": "array"}},
                                        "required": []}},
    # predict 是 Phase-0 占位引擎（models.ModelEngine），**不是**训练模型推理。
    # 2026-09-15 实测：缺输入时它恒返回 signal=0.0/confidence=0.5，会被上层误当
    # 成真实模型输出。现在缺输入返回 status=insufficient_input 且 signal/confidence
    # 为 null；真推理请用 ml_predict（需 klines_list）。
    "predict": {"name": "predict",
                "description": "Phase-0 PLACEHOLDER engine (NOT a trained model). "
                               "Requires factors {roc_5, volume_ratio, ...}; returns "
                               "status=insufficient_input with null signal/confidence when "
                               "inputs are missing, and always sets is_mock=true. "
                               "For real inference use ml_predict (needs klines_list).",
                "inputSchema": {"type": "object",
                                "properties": {"symbol": {"type": "string"},
                                               "factors": {"type": "object"}},
                                "required": []}},
}


# 形状容差参数清单：(工具名, 参数名) —— 供测试钉住网关层的注解放宽。
# 这些参数的注解必须是 Any，见 panel.py 与 mcp_common._ann_type。
SHAPE_TOLERANT_ARGS = [
    ("factor_tearsheet", "klines"), ("factor_tearsheet", "factor_values"),
    ("portfolio_optimize", "klines"), ("regime_detect", "klines"),
    ("change_point", "series"), ("vol_forecast", "klines"),
    ("ml_predict", "klines_list"), ("ml_train_rolling", "klines_list"),
]
