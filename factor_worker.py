#!/usr/bin/env python3
"""factor_worker.py — FactorMiner sidecar tool handlers（DESIGN-FACTOR-MINER.md §2/§4/§6/§12）

P1 集成胶水：把 factor_miner 三个核心模块（sandbox / factor_eval / factor_backtest）
桥接到 server.py 的 /call-tool 通道。约定：
- 所有 handler 都返回 JSON STRING（/call-tool 把返回值 stringify 进 {"result": ...}），
  失败也返回 JSON、绝不抛异常——Go 侧 miner.SidecarExecutor / SidecarBacktester 解析。
- 本模块只被 server.py import（不 import server.py，避免循环依赖）。
- qlib/mlflow/jinja2 保持惰性导入（在 factor_backtest.py / 本文件函数内），
  未装 pyqlib 的环境可 import 本模块、可跑 factor_execute。

P3 生产链路（§6 OOS 段 / §9 生产准入 / §12）：
- factor_oos_check：promoted→live 人工卡点的 2021-今 纯样本外检验
- factor_daily_compute：live 因子每日盘后全量计算 → Redis dfactor:{symbol}（TTL 48h）
- factor_recent_ic：周日衰减巡检（近 60 交易日截面 IC，纯 pandas，不走 qlib）
"""
from __future__ import annotations

import json
import os
import shutil
import traceback as tb_module
import uuid
from pathlib import Path

import pandas as pd

from factor_miner import factor_backtest as fb
from factor_miner import factor_eval, sandbox

# 数据目录可用 env 覆盖（联调/CI 指向合成数据）
DATA_DIR = Path(os.environ.get("FACTOR_MINER_DATA_DIR", "data/factor_mining"))
JOBS_ROOT = Path(os.environ.get("FACTOR_MINER_JOBS_DIR", "/tmp/factor_jobs"))
BACKTEST_ROOT = Path(os.environ.get("FACTOR_MINER_BACKTEST_DIR", "/tmp/factor_backtests"))
# 文件级执行缓存：md5(code + 数据版本) → result.h5 拷贝（§6 逐因子缓存）
EXEC_CACHE_DIR = DATA_DIR / "exec_cache"

EXEC_TIMEOUT = 120  # 沙箱墙壁时钟上限（§4）

# ── P3 生产链路常量（§12）──
OOS_TEST_START = "2021-01-01"      # 生产准入 OOS 窗口起点（§6 评审修订）
MINING_TEST_START = "2017-01-01"   # 挖掘回路 test 起点（对齐 RD-Agent 口径）
DFACTOR_PREFIX = "dfactor:"        # live 因子日频 Redis 前缀（区别于盘中 factor: 300s）
DFACTOR_TTL = 48 * 3600            # 日频 cadence：48h
RECENT_IC_LOOKBACK = 60            # 周日衰减巡检默认回看交易日数

# Redis 客户端（模块内自持，不 import server.py；测试可 monkeypatch _get_redis）
_REDIS = None
_REDIS_FAILED_AT = 0.0
_REDIS_RETRY_SEC = 60


def _get_redis():
    """惰性连接 Redis（env REDIS_URL）。失败进入 60s 冷却，返回 None 不抛异常。"""
    global _REDIS, _REDIS_FAILED_AT
    import time as _time

    if _REDIS is not None and _REDIS is not False:
        return _REDIS
    if _REDIS is False and _time.time() - _REDIS_FAILED_AT < _REDIS_RETRY_SEC:
        return None
    try:
        import redis as _r  # noqa: PLC0415 — 惰性导入

        client = _r.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                             socket_timeout=3, socket_connect_timeout=3)
        client.ping()
        _REDIS = client
        return _REDIS
    except Exception:  # noqa: BLE001 — redis 不可用只降级，不杀工具
        _REDIS = False
        _REDIS_FAILED_AT = _time.time()
        return None


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def factor_execute(code: str, debug: bool = True) -> str:
    """沙箱执行 factor.py + debug 评估 battery（§4/§5.3）。

    返回 JSON: {"stdout":..., "eval_ok": bool, "eval_detail":...}
    eval_detail 失败时含具体检查项名 + 反馈（EvalResult.feedback()），
    执行失败时含 stderr/traceback，白名单违规时含违规信息。
    """
    data_h5 = DATA_DIR / ("daily_pv_debug.h5" if debug else "daily_pv_all.h5")
    if not data_h5.exists():
        return _json({"stdout": "", "eval_ok": False,
                      "eval_detail": f"data file missing: {data_h5}"})
    job_dir = JOBS_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    # factor.py 契约：读同目录 daily_pv.h5，写 result.h5
    (job_dir / "daily_pv.h5").symlink_to(data_h5.resolve())

    res = sandbox.run_factor_source(code, str(job_dir), timeout=EXEC_TIMEOUT)
    if res.violation:
        return _json({"stdout": "", "eval_ok": False,
                      "eval_detail": f"whitelist violation: {res.violation}"})
    result_h5 = job_dir / "result.h5"
    if res.returncode != 0 or not result_h5.exists():
        detail = res.stderr or res.stdout or f"factor.py exit {res.returncode}"
        if res.timed_out:
            detail = f"[sandbox timeout {EXEC_TIMEOUT}s]\n" + detail
        return _json({"stdout": res.stdout, "eval_ok": False, "eval_detail": detail})

    eval_res = factor_eval.evaluate(result_h5)
    return _json({
        "stdout": res.stdout,
        "eval_ok": eval_res.ok,
        "eval_detail": eval_res.feedback(),
    })


def _write_back_cache(factors: list, version: str, work_dir: Path) -> None:
    """回测后把 work_dir/factors/<name>/result.h5 写回文件级 exec_cache
    （key 与 factor_backtest.cache_key 一致；factor_backtest.py docstring 的调用方契约）。"""
    EXEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for f in factors:
        result = work_dir / "factors" / f.name / "result.h5"
        dst = EXEC_CACHE_DIR / f"{fb.cache_key(f.code, version)}.h5"
        if result.exists() and not dst.exists():
            try:
                shutil.copy2(result, dst)
            except OSError:
                pass  # 缓存写失败不杀回测


def _load_factor_df(fsrc: "fb.FactorSrc", version: str, work_dir: Path):
    """从 exec_cache / 本次 work_dir 加载因子值（去重闸门相关性输入）。"""
    cached = EXEC_CACHE_DIR / f"{fb.cache_key(fsrc.code, version)}.h5"
    for path in (cached, work_dir / "factors" / fsrc.name / "result.h5"):
        if not path.exists():
            continue
        try:
            try:
                df = pd.read_hdf(path, key="data")  # factor.py 契约 key="data"
            except (KeyError, ValueError):
                df = pd.read_hdf(path)
            if isinstance(df, pd.Series):
                df = df.to_frame("factor")
            # 归一化 (datetime, instrument) 索引，复用 factor_backtest 内部函数
            return fb._normalize_index(df)
        except Exception:  # noqa: BLE001 — 缓存损坏按缺失处理
            continue
    return None


def _correlations(new_src: "fb.FactorSrc", sota_srcs: list, version: str,
                  work_dir: Path) -> list:
    """新因子 vs 每个 SOTA 因子的日均截面 Pearson IC（Go 侧去重闸门输入，§6.2）。"""
    new_df = _load_factor_df(new_src, version, work_dir)
    if new_df is None:
        return []
    out = []
    for s in sota_srcs:
        sdf = _load_factor_df(s, version, work_dir)
        if sdf is None:
            out.append(0.0)
        else:
            out.append(fb._daily_ic(new_df.iloc[:, 0], sdf.iloc[:, 0]))
    return out


def _render_windows_conf(work_dir: Path, profile: str, windows: dict) -> Path:
    """把 Go cfg 的回测窗口注入 conf 模板。

    预渲染后的 yaml 作为 conf_template 传给 run_backtest——其内部 _render_conf
    对无 Jinja 占位的文本是 no-op，profile 覆盖已在此时合并。
    """
    from jinja2 import Template  # noqa: PLC0415 — 与 qlib 一样保持惰性导入

    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = {
        "feature_expressions": str(list(fb.ALPHA20.values())),
        "feature_names": str(list(fb.ALPHA20.keys())),
    }
    ctx.update(fb.PROFILE_OVERRIDES.get(profile, {}))
    ctx.update({k: v for k, v in windows.items() if v})
    out = work_dir / "conf_windows.yaml"
    out.write_text(Template(fb.DEFAULT_CONF_TEMPLATE.read_text()).render(**ctx))
    return out


def _run_backtest(sota: list, new_factors: list, profile: str, windows: dict | None) -> dict:
    data_h5 = DATA_DIR / "daily_pv_all.h5"  # 回测永远跑全量数据（§6）
    if not data_h5.exists():
        raise FileNotFoundError(f"data file missing: {data_h5}")
    version = fb.data_version(data_h5)

    sota_srcs = [fb.FactorSrc(name=f["name"], code=f["code"]) for f in sota]

    def exec_cache(key: str):
        p = EXEC_CACHE_DIR / f"{key}.h5"
        return p if p.exists() else None

    # factor_backtest.run_backtest 单轮只接一个新因子（RD-Agent 实验语义），
    # 多新因子在此逐一轮训；SOTA 因子经 exec_cache 只重算一次。
    correlations: dict = {}
    sota_broken: set = set()
    metrics: dict = {}
    net_values: list = []
    dropped: list = []
    errors: dict = {}
    last_tb = ""
    any_ok = False

    for nf in new_factors:
        new_src = fb.FactorSrc(name=nf["name"], code=nf["code"])
        work_dir = BACKTEST_ROOT / uuid.uuid4().hex
        conf = (_render_windows_conf(work_dir, profile, windows)
                if windows else fb.DEFAULT_CONF_TEMPLATE)
        res = fb.run_backtest(
            sota_factors=sota_srcs,
            new_factor=new_src,
            work_dir=work_dir,
            data_h5=data_h5,
            conf_template=conf,
            exec_cache=exec_cache,
            profile=profile,
        )
        _write_back_cache(sota_srcs + [new_src], version, work_dir)
        correlations[new_src.name] = _correlations(new_src, sota_srcs, version, work_dir)
        sota_broken.update(res.sota_broken)
        if res.ok:
            any_ok = True
            if not metrics:
                # 组合指标取首个成功新因子的回测（SOTA+该因子拼接口径）
                metrics, net_values = res.metrics, res.net_values
        elif res.dedup_dropped:
            dropped.append(new_src.name)
        else:
            errors[new_src.name] = res.error
            last_tb = res.traceback or last_tb

    all_dropped = len(dropped) == len(new_factors) and not errors
    error = ""
    if errors:
        error = "; ".join(f"{n}: {e}" for n, e in errors.items())
    return {
        "ok": any_ok,
        # 全部被去重闸门丢弃 = RD-Agent FactorEmptyError 语义（Go 侧记 decision=false）
        "dedup_dropped": all_dropped,
        "sota_broken": sorted(sota_broken),
        "metrics": metrics,
        "correlations": correlations,
        "net_values": net_values,
        "error": error,
        "traceback": last_tb,
    }


def factor_backtest(sota: list, new_factors: list, profile: str = "full",
                    windows: dict | None = None) -> str:
    """qlib 全量回测（§6）。sota/new_factors 元素为 {"name":..., "code":...}。

    返回 JSON: BacktestResult 字段 + correlations（新因子名 → 对每个 SOTA 的 IC 列表）。
    任何异常都兜底成 ok=False 的 JSON，不抛出（/call-tool 通道约定）。
    """
    try:
        return _json(_run_backtest(sota or [], new_factors or [], profile, windows))
    except Exception:  # noqa: BLE001 — tool 通道永不抛异常
        return _json({
            "ok": False, "dedup_dropped": False, "sota_broken": [],
            "metrics": {}, "correlations": {}, "net_values": [],
            "error": "factor_backtest worker exception",
            "traceback": tb_module.format_exc(),
        })


# ═══════════════════════════════════════════════════════════════
# P3 生产链路（§6 OOS 段 / §9 生产准入 / §12）
# ═══════════════════════════════════════════════════════════════


def _run_factor_df(code: str, data_h5: Path) -> pd.DataFrame:
    """沙箱执行单个 factor.py 并读回 (datetime, instrument) 归一化 DataFrame。"""
    job_dir = JOBS_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "daily_pv.h5").symlink_to(data_h5.resolve())
    res = sandbox.run_factor_source(code, str(job_dir), timeout=EXEC_TIMEOUT)
    if res.violation:
        raise RuntimeError(f"whitelist violation: {res.violation}")
    if res.returncode != 0 or not (job_dir / "result.h5").exists():
        detail = res.stderr or res.stdout or f"factor.py exit {res.returncode}"
        if res.timed_out:
            detail = f"[sandbox timeout {EXEC_TIMEOUT}s]\n" + detail
        raise RuntimeError(detail)
    return fb._read_result_h5(job_dir)


def latest_row_per_instrument(df: pd.DataFrame) -> dict:
    """取最新 datetime 截面 → {instrument: value}（纯函数，单独可测）。"""
    if df is None or df.empty:
        return {}
    dt = df.index.get_level_values("datetime").max()
    cross = df.xs(dt, level="datetime").iloc[:, 0].dropna()
    return {str(inst): float(v) for inst, v in cross.items()}


def write_daily_factors_to_redis(values: dict, ttl: int = DFACTOR_TTL,
                                 redis_client=None) -> int:
    """合并后的 live 因子值写 Redis：dfactor:{symbol} → {name: value}，TTL 48h。

    redis_client 为 None 时走 _get_redis()；测试可传 fake client 或
    monkeypatch _get_redis。返回成功写入的 key 数；redis 不可用返回 0。
    """
    r = redis_client if redis_client is not None else _get_redis()
    if r is None:
        return 0
    written = 0
    for symbol, factors in values.items():
        if not factors:
            continue
        try:
            r.setex(DFACTOR_PREFIX + symbol, ttl, json.dumps(factors))
            written += 1
        except Exception:  # noqa: BLE001 — 单 symbol 写失败不阻塞其余
            continue
    return written


def _last_weekday(today: "pd.Timestamp") -> "pd.Timestamp":
    """最近的交易日近似：周末回退到周五（不查交易日历，够用于新鲜度判断）。"""
    wd = today.weekday()
    if wd >= 5:
        return today - pd.Timedelta(days=wd - 4)
    return today


def _ensure_data_fresh(data_h5: Path, errors: list) -> None:
    """daily_pv_all.h5 最新日期落后于最近交易日 → 进程内跑 gen_data 全量刷新。
    刷新失败只记 errors，不阻塞（用旧数据继续算）。"""
    try:
        df = pd.read_hdf(data_h5, key="data")
        idx = df.index
        max_dt = (idx.get_level_values("datetime") if isinstance(idx, pd.MultiIndex)
                  else idx).max()
        if pd.Timestamp(max_dt).normalize() >= _last_weekday(pd.Timestamp.today().normalize()):
            return
    except Exception as e:  # noqa: BLE001 — 读不出日期按需要刷新处理
        errors.append(f"read h5 freshness failed: {e}")
    try:
        from factor_miner import gen_data  # noqa: PLC0415 — qlib 惰性导入

        D = gen_data._init_qlib(os.environ.get(
            "QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data"))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        gen_data.gen_full(D, DATA_DIR)
    except Exception as e:  # noqa: BLE001 — 刷新失败降级用旧数据
        errors.append(f"gen_data refresh failed (using stale h5): {e}")


def factor_daily_compute(factors: list) -> str:
    """live 因子每日盘后全量计算 → Redis dfactor:{symbol}（§12.1）。

    factors 元素 {"name":..., "code":...}。h5 过期先进程内 gen_data 刷新；
    单因子失败记 errors 不阻塞其余。返回 JSON: {written, symbols, errors}。
    """
    errors: list = []
    data_h5 = DATA_DIR / "daily_pv_all.h5"
    if not data_h5.exists():
        return _json({"written": 0, "symbols": 0,
                      "errors": [f"data file missing: {data_h5}"]})
    _ensure_data_fresh(data_h5, errors)

    values: dict = {}
    for f in factors or []:
        name, code = f.get("name", ""), f.get("code", "")
        if not name or not code:
            errors.append(f"factor missing name/code: {f!r}")
            continue
        try:
            latest = latest_row_per_instrument(_run_factor_df(code, data_h5))
        except Exception as e:  # noqa: BLE001 — 单因子失败不阻塞其余
            errors.append(f"{name}: {e}")
            continue
        for symbol, v in latest.items():
            values.setdefault(symbol, {})[name] = v

    written = write_daily_factors_to_redis(values)
    return _json({"written": written, "symbols": len(values), "errors": errors})


def _pick_metrics(metrics: dict) -> dict:
    """从回测 metrics 提取 OOS 报告三元组（键名对齐 read_exp_res 口径）。"""
    return {
        "ic": metrics.get("IC"),
        "annualized_return": metrics.get("annualized_return_with_cost"),
        "max_drawdown": metrics.get("max_drawdown"),
    }


def _oos_backtest(code: str, name: str, test_start: str) -> dict:
    """单因子 qlib 回测（无 SOTA 拼接），test_start 可覆盖——OOS 与挖掘窗口共用
    train/valid 默认值（模板 2008-2014 / 2015-2016），仅 test 窗口不同（§6）。"""
    return _run_backtest([], [{"name": name, "code": code}], "full",
                         {"test_start": test_start})


def factor_oos_check(code: str, name: str) -> str:
    """生产准入纯样本外检验（§6 OOS 段 / §9 人工卡点）：同一因子分别跑
    挖掘窗口（test 2017→今）与 OOS 窗口（test 2021-01→今）两次 qlib 回测，
    报告 IC/年化/回撤对比与相对衰减 decay = 1 - oos.ic/mining.ic
    （mining.ic==0 → decay null，由 Go 侧按不可证伪拒绝）。

    返回 JSON: {oos: {ic, annualized_return, max_drawdown}, mining: {...},
                decay: float|null, ok, error}。永不抛异常。
    """
    try:
        mining = _oos_backtest(code, name, MINING_TEST_START)
        if not mining.get("ok"):
            return _json({"oos": {}, "mining": {}, "decay": None, "ok": False,
                          "error": f"mining-window backtest failed: {mining.get('error', '')}"})
        oos = _oos_backtest(code, name, OOS_TEST_START)
        if not oos.get("ok"):
            return _json({"oos": {}, "mining": _pick_metrics(mining["metrics"]),
                          "decay": None, "ok": False,
                          "error": f"oos-window backtest failed: {oos.get('error', '')}"})
        m_ic = (mining["metrics"] or {}).get("IC")
        o_ic = (oos["metrics"] or {}).get("IC")
        decay = None
        if m_ic and o_ic is not None:
            decay = 1.0 - float(o_ic) / float(m_ic)
        return _json({
            "oos": _pick_metrics(oos["metrics"]),
            "mining": _pick_metrics(mining["metrics"]),
            "decay": decay,
            "ok": True,
            "error": "",
        })
    except Exception:  # noqa: BLE001 — tool 通道永不抛异常
        return _json({"oos": {}, "mining": {}, "decay": None, "ok": False,
                      "error": "factor_oos_check worker exception",
                      "traceback": tb_module.format_exc()})


def factor_recent_ic(code: str, name: str,
                     lookback_days: int = RECENT_IC_LOOKBACK) -> str:
    """周日衰减巡检（§12.3）：近 lookback_days 个交易日的日均截面 Pearson IC
    （因子值 vs 次日收益），纯 pandas 不走 qlib——周频跑全库成本必须低。

    返回 JSON: {ok, ic, days, error}。数据不足（<10 个交易日）记 error。
    """
    try:
        data_h5 = DATA_DIR / "daily_pv_all.h5"
        if not data_h5.exists():
            return _json({"ok": False, "ic": None, "days": 0,
                          "error": f"data file missing: {data_h5}"})
        fac = _run_factor_df(code, data_h5).iloc[:, 0].rename("factor")
        pv = pd.read_hdf(data_h5, key="data")
        close = pv["$close"].rename("close")
        # 次日收益（T+1 开盘不可得的近似：close→close），与挖掘 label 口径同族
        ret1 = close.groupby(level="instrument").pct_change().groupby(
            level="instrument").shift(-1).rename("ret")
        pair = pd.concat([fac, ret1], axis=1).dropna()
        if pair.empty:
            return _json({"ok": False, "ic": None, "days": 0,
                          "error": "no overlapping factor/return data"})
        dts = pair.index.get_level_values("datetime").unique().sort_values()
        tail = dts[-max(int(lookback_days), 1):]
        pair = pair.loc[pair.index.get_level_values("datetime").isin(tail)]
        ics = pair.groupby(level="datetime").apply(
            lambda x: x["factor"].corr(x["ret"])).dropna()
        if len(ics) < 10:
            return _json({"ok": False, "ic": None, "days": len(ics),
                          "error": f"insufficient recent days: {len(ics)}"})
        return _json({"ok": True, "ic": float(ics.mean()), "days": len(ics),
                      "error": ""})
    except Exception:  # noqa: BLE001 — tool 通道永不抛异常
        return _json({"ok": False, "ic": None, "days": 0,
                      "error": "factor_recent_ic worker exception",
                      "traceback": tb_module.format_exc()})
