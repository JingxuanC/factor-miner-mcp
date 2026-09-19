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
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import traceback as tb_module
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from factor_miner import factor_backtest as fb
from factor_miner import factor_eval, sandbox

logger = logging.getLogger("factor-worker")

# 数据目录可用 env 覆盖（联调/CI 指向合成数据）
DATA_DIR = Path(os.environ.get("FACTOR_MINER_DATA_DIR", "data/factor_mining"))
JOBS_ROOT = Path(os.environ.get("FACTOR_MINER_JOBS_DIR", "/tmp/factor_jobs"))
BACKTEST_ROOT = Path(os.environ.get("FACTOR_MINER_BACKTEST_DIR", "/tmp/factor_backtests"))
# 文件级执行缓存：md5(code + 数据版本) → result.h5 拷贝（§6 逐因子缓存）
EXEC_CACHE_DIR = DATA_DIR / "exec_cache"

# ── 磁盘清理（原实现 JOBS_ROOT/BACKTEST_ROOT/EXEC_CACHE 永不清理 → /tmp 积压）──
CLEANUP_TTL_SEC = int(os.environ.get("FACTOR_MINER_CLEANUP_TTL_SEC", 24 * 3600))
CLEANUP_INTERVAL_SEC = int(os.environ.get("FACTOR_MINER_CLEANUP_INTERVAL_SEC", 3600))
EXEC_CACHE_MAX_ENTRIES = int(os.environ.get("FACTOR_MINER_EXEC_CACHE_MAX", 500))
EXEC_CACHE_MAX_BYTES = int(os.environ.get("FACTOR_MINER_EXEC_CACHE_MAX_BYTES", 20 * 1024 ** 3))
# 本项目在 /tmp 与数据目录里的中间产物前缀（中断残留；TTL 保护进行中的任务）
_TMP_STALE_PREFIXES = ("qlib_csv_",)
_DATA_STALE_PREFIXES = ("h5_",)
_CLEANUP_STARTED = False

EXEC_TIMEOUT = 120  # 沙箱墙壁时钟上限（§4）

# ── P3 生产链路常量（§12）──
OOS_TEST_START = "2021-01-01"      # 生产准入 OOS 窗口起点（§6 评审修订）
MINING_TEST_START = "2017-01-01"   # 挖掘回路 test 起点（对齐 RD-Agent 口径）
DFACTOR_PREFIX = "dfactor:"        # live 因子日频 Redis 前缀（区别于盘中 factor: 300s）
DFACTOR_TTL = 48 * 3600            # 日频 cadence：48h
RECENT_IC_LOOKBACK = 60            # 周日衰减巡检默认回看交易日数

# 沙箱输入窗口：只把最近 N 个交易日喂进沙箱（0 = 全量，保持旧行为）。
#
# 为什么必须截：沙箱 RLIMIT_AS=2GB（sandbox.MAX_AS_BYTES），而全量
# daily_pv_all.h5 是 6000+ 标的 × 4300+ 交易日（实测 15,095,115 行 × 6 列）。
# 因子代码 `pd.read_hdf('daily_pv.h5')` 把它读成 numpy 时虚拟地址空间直接爆掉：
#   numpy._core._exceptions._ArrayMemoryError: Unable to allocate 345. MiB
# 而两个调用方其实都用不到全量历史 —— factor_recent_ic 只取尾部 lookback_days 天、
# factor_daily_compute 只取最新截面 —— 截窗不改变结果，却把沙箱内存降一个数量级，
# 顺带把单次执行从"算 4300 天"降到"算 400 天"。
#
# 400 个交易日（约 1.6 年）覆盖绝大多数公开因子（20 日反转、60/250 日动量、
# 1 年波动率等）。需要更长历史的因子把 FACTOR_MINER_WINDOW_DAYS 调大，
# 必要时同时调大 FACTOR_MINER_MAX_AS_BYTES。
WINDOW_DAYS = int(os.environ.get("FACTOR_MINER_WINDOW_DAYS", "400") or 0)

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


# ═══════════════ 磁盘清理（TTL + exec_cache LRU） ═══════════════

def cleanup_stale_dirs(root, ttl_sec: int = CLEANUP_TTL_SEC, now: float = None,
                       prefixes: tuple = None) -> list:
    """删掉 root 下 mtime 超过 ttl 的条目（目录/文件），返回被删路径。

    prefixes 非空时只处理名字以上述前缀开头的条目（用于清理共享 /tmp 里的本项目残留）。
    只按 mtime 判断、不递归；删除失败（权限/占用）跳过不抛。
    """
    root = Path(root)
    if not root.exists():
        return []
    cutoff = (time.time() if now is None else now) - max(int(ttl_sec), 0)
    removed: list = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    for p in entries:
        if prefixes and not p.name.startswith(prefixes):
            continue
        try:
            if p.stat().st_mtime >= cutoff:
                continue
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
            removed.append(str(p))
        except OSError:
            continue
    return removed


def cleanup_exec_cache(max_entries: int = EXEC_CACHE_MAX_ENTRIES,
                       max_bytes: int = EXEC_CACHE_MAX_BYTES) -> list:
    """exec_cache LRU：按 mtime 保留最新 max_entries 个，再把总量压到 max_bytes 内。"""
    if not EXEC_CACHE_DIR.exists():
        return []
    entries = []
    try:
        for p in EXEC_CACHE_DIR.iterdir():
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
    except OSError:
        return []
    entries.sort(key=lambda x: x[0], reverse=True)  # 新 → 旧
    keep_bytes, drop = 0, []
    for i, (_mt, size, p) in enumerate(entries):
        if i < max(int(max_entries), 0) and (max_bytes <= 0 or keep_bytes + size <= max_bytes):
            keep_bytes += size
        else:
            drop.append(p)
    removed: list = []
    for p in drop:
        try:
            p.unlink()
            removed.append(str(p))
        except OSError:
            continue
    return removed


def _cleanup_stale_staging(now: float = None) -> list:
    """update_data 崩溃残留的 <provider>.__staging__ / .__prev__（TTL 保护进行中任务）。"""
    uri = os.environ.get("QLIB_PROVIDER_URI")
    if not uri:
        return []
    prov = Path(uri).expanduser()
    removed: list = []
    for name in (prov.name + ".__staging__", prov.name + ".__prev__"):
        if (prov.parent / name).exists():
            removed += cleanup_stale_dirs(prov.parent, now=now, prefixes=(name,))
    return removed


def cleanup_disk(now: float = None) -> dict:
    """一次全量清理：job/回测目录（TTL）+ exec_cache（LRU）+ 中断残留。"""
    out = {
        "jobs": cleanup_stale_dirs(JOBS_ROOT, now=now),
        "backtests": cleanup_stale_dirs(BACKTEST_ROOT, now=now),
        "exec_cache": cleanup_exec_cache(),
        "tmp": cleanup_stale_dirs(tempfile.gettempdir(), now=now,
                                  prefixes=_TMP_STALE_PREFIXES),
        "h5_tmp": cleanup_stale_dirs(DATA_DIR, now=now, prefixes=_DATA_STALE_PREFIXES),
        "staging": _cleanup_stale_staging(now=now),
    }
    total = sum(len(v) for v in out.values())
    if total:
        logger.info("disk cleanup removed %d entries: %s", total,
                    {k: len(v) for k, v in out.items() if v})
    return out


def start_cleanup_thread(interval_sec: int = CLEANUP_INTERVAL_SEC):
    """起一个后台清理线程（幂等）。server.py 启动时调一次，之后每 interval 跑一次。"""
    global _CLEANUP_STARTED
    if _CLEANUP_STARTED:
        return None
    _CLEANUP_STARTED = True
    interval = max(int(interval_sec), 60)

    def _loop():
        while True:
            time.sleep(interval)
            try:
                cleanup_disk()
            except Exception as e:  # noqa: BLE001 — 清理失败不影响服务
                logger.warning("disk cleanup failed: %s", e)

    t = threading.Thread(target=_loop, daemon=True, name="factor-cleanup")
    t.start()
    return t


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
    # debug 数据集很小（100 股 × 2 年），保持 symlink；全量则按窗口截取，
    # 否则 15M 行会撑爆沙箱 RLIMIT_AS（见 WINDOW_DAYS 说明）。
    stage_h5_for_sandbox(data_h5, job_dir, 0 if debug else WINDOW_DAYS)

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
        # 明示评估窗口：全量执行时只喂了最近 WINDOW_DAYS 个交易日（不静默改语义）
        "window_days": 0 if debug else WINDOW_DAYS,
        "dataset": "debug" if debug else "full",
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


def _render_windows_conf(work_dir: Path, profile: str, windows: dict,
                         provider_uri: str | None = None) -> Path:
    """把 Go cfg 的回测窗口注入 conf 模板。

    预渲染后的 yaml 作为 conf_template 传给 run_backtest——其内部 _render_conf
    对无 Jinja 占位的文本是 no-op，profile 覆盖已在此时合并。

    **provider_uri 必须在这里就注入**：因为上面那句 no-op 是双向的 —— 一旦这里渲染
    过一次、占位被替换掉，后面再传 provider_uri 也回天无力。漏掉它的后果实测过：
    OOS 路径（唯一走 windows 的路径）把 provider_uri 落回模板默认的
    `~/.qlib/qlib_data/cn_data`，在执行器容器里那个路径是空的，qrun 报
    `instrument: {...} does not contain data for day` —— 而同一因子走非 windows
    路径时 provider_uri 是对的。
    """
    from jinja2 import Template  # noqa: PLC0415 — 与 qlib 一样保持惰性导入

    work_dir.mkdir(parents=True, exist_ok=True)
    ctx = {
        "feature_expressions": str(list(fb.ALPHA20.values())),
        "feature_names": str(list(fb.ALPHA20.keys())),
    }
    ctx.update(fb.PROFILE_OVERRIDES.get(profile, {}))
    ctx.update({k: v for k, v in windows.items() if v})
    if provider_uri:
        ctx["provider_uri"] = provider_uri
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
    net_curve: list = []
    dropped: list = []
    errors: dict = {}
    last_tb = ""
    any_ok = False

    # 远程模式下 conf 的 qlib_init.provider_uri 必须指向**执行器**的挂载路径
    remote_uri = (os.environ.get("FACTOR_EXECUTOR_PROVIDER_URI") or None
                  if os.environ.get("FACTOR_EXECUTOR_URL", "").strip() else None)
    for nf in new_factors:
        new_src = fb.FactorSrc(name=nf["name"], code=nf["code"])
        work_dir = BACKTEST_ROOT / uuid.uuid4().hex
        conf = (_render_windows_conf(work_dir, profile, windows, remote_uri)
                if windows else fb.DEFAULT_CONF_TEMPLATE)
        res = fb.run_backtest(
            sota_factors=sota_srcs,
            new_factor=new_src,
            work_dir=work_dir,
            data_h5=data_h5,
            conf_template=conf,
            exec_cache=exec_cache,
            profile=profile,
            version=version,  # 复用本函数已算的数据版本，避免重复计算
            execute=_executor_dispatch(work_dir, version),
            # 远程模式下 conf 的 qlib_init.provider_uri 必须指向**执行器**的挂载路径。
            # 本地模式不传，沿用模板默认值（行为不变）。
            provider_uri=remote_uri,
        )
        _write_back_cache(sota_srcs + [new_src], version, work_dir)
        correlations[new_src.name] = _correlations(new_src, sota_srcs, version, work_dir)
        sota_broken.update(res.sota_broken)
        if res.ok:
            any_ok = True
            if not metrics:
                # 组合指标取首个成功新因子的回测（SOTA+该因子拼接口径）
                metrics, net_values = res.metrics, res.net_values
                # 带日期的净值曲线与 net_values 同源同口径，取同一个成功因子那一轮
                net_curve = list(getattr(res, "net_curve", None) or [])
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
        # 带日期的净值曲线（已抽稀）。net_values 保持原样不动，见 _downsample_curve。
        "net_curve": _downsample_curve(net_curve),
        "error": error,
        "traceback": last_tb,
    }


def _executor_dispatch(work_dir: Path, version: str | None):
    """把 qrun 交给远程执行器跑；未配置执行器时返回 None（走本地执行）。

    为什么要拆出去：qrun 是全市场 LGBM 训练，内存是 GB 级，而本容器被限制在
    1536 MiB。实测后果是内核 memcg 直接 OOM 掉**整个 miner 容器**，并且失败的
    qrun 会变孤儿进程继续占内存、把容器卡死到只能重启。

    只搬「执行」这一步：Step 1–3（沙箱跑因子、去重闸门、拼 combined_factors）
    留在本地，它们本来就是轻的，而且依赖面板归一化与 exec_cache。

    conf 由本地渲染后传给执行器，其中 `qlib_init.provider_uri` 来自
    `FACTOR_EXECUTOR_PROVIDER_URI`（执行器容器内的挂载路径）—— 两边必须指向
    同一份数据。执行器侧的**面板版本校验**是兜底：不一致就拒绝执行，而不是
    读另一份数据静默跑出偏掉的结果。
    """
    url = os.environ.get("FACTOR_EXECUTOR_URL", "").strip()
    if not url:
        return None

    def execute(work: Path, conf_path: Path, timeout: int, ver: str | None) -> dict:
        from factor_executor import client as executor_client

        try:
            return executor_client.run_backtest(work, data_version=ver,
                                                deadline=time.time() + timeout + 120)
        except executor_client.ExecutorError as exc:
            # 执行器侧的问题不该伪装成"因子有问题"：以回测失败返回，但把原因写清
            return {"ok": False, "error": f"executor error: {exc}"}

    return execute


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


def stage_h5_for_sandbox(data_h5: Path, job_dir: Path, window_days: int) -> Path:
    """把 h5 按窗口截取后放进 job 目录 —— 委托给 fb.stage_h5_for_sandbox。

    这里刻意**不再保留第二份实现**：历史上有两份（本文件一份、_exec_factor_worker
    自己 inline 的 symlink 一份），而后者漏了窗口，导致回测路径永远喂全量、撞破
    沙箱地址空间上限被 SIGKILL。单一实现是这次修复的核心。
    """
    return fb.stage_h5_for_sandbox(data_h5, job_dir, window_days)


def _run_factor_window(code: str, data_h5: Path,
                       window_days: int | None = None) -> tuple[pd.Series, pd.Series]:
    """在**同一个**窗口切片上跑 factor.py，并把该切片的收盘价一并交回。

    返回 ``(factor_series, close_series)``，两者同为 (datetime, instrument) 索引。

    存在的理由是一次全量读取代两次。``daily_pv_all.h5`` 是 pandas 的 Fixed
    格式（PyTables 里是 ``block0_values`` 数组、没有 table），所以**无法按列或
    按行部分读**——实测 ``columns=['$close']`` 直接 TypeError，只能整表读入，
    一次 ~710MiB / ~27s。

    ``factor_recent_ic`` 需要两样东西：因子值（来自跑完的 result.h5）和收盘价
    （算次日收益）。原先它先让 ``_run_factor_df`` 整读一次全量去暂存窗口，再
    自己整读第二次去取收盘价——峰值因此叠到 ~1375MiB，在 1536MiB 的容器上限
    下把整个服务 OOM 掉（内核 memcg 击杀，容器重启）。

    切片的收盘价与"整读全量再取 ``$close``"**逐值相同**：切片就是该窗口的行
    子集，这里再按索引排序，与全量读的顺序一致。所以既省掉一次 710MiB 的整
    读，又不改变数值口径。

    job_dir 不返回：它留在 JOBS_ROOT 由 ``cleanup_stale_dirs`` 按 TTL 回收，
    与改动前一致。
    """
    job_dir = JOBS_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    stage_h5_for_sandbox(data_h5, job_dir,
                         WINDOW_DAYS if window_days is None else window_days)
    res = sandbox.run_factor_source(code, str(job_dir), timeout=EXEC_TIMEOUT)
    if res.violation:
        raise RuntimeError(f"whitelist violation: {res.violation}")
    if res.returncode != 0 or not (job_dir / "result.h5").exists():
        detail = res.stderr or res.stdout or f"factor.py exit {res.returncode}"
        if res.timed_out:
            detail = f"[sandbox timeout {EXEC_TIMEOUT}s]\n" + detail
        raise RuntimeError(detail)

    factor = fb._read_result_h5(job_dir).iloc[:, 0].rename("factor")
    staged = pd.read_hdf(job_dir / "daily_pv.h5", key="data").sort_index()
    close = staged["$close"].rename("close")
    del staged
    return factor, close


def _run_factor_df(code: str, data_h5: Path,
                   window_days: int | None = None) -> pd.DataFrame:
    """沙箱执行单个 factor.py 并读回 (datetime, instrument) 归一化 DataFrame。

    ``window_days`` 缺省用 ``WINDOW_DAYS``（默认 400 交易日）；见该常量的说明 ——
    全量 h5 会撑爆沙箱 RLIMIT_AS。
    """
    factor, close = _run_factor_window(code, data_h5, window_days)
    del close  # 调用方只要因子值；少留一份切片在内存里
    return factor.to_frame("factor")


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

    symbol 一律**转小写**再拼键：写入侧的 symbol 来自 qlib instrument 名（`SH600340`
    大写），而读取侧 athena 的 `factor.Store.GetDaily` 拼的是 `dkeyPrefix + q.Symbol`，
    其 symbol 约定是小写（`sh600340`）。Redis 键大小写敏感，不归一就会出现
    "写进去了但 MGet 全是 nil"的静默失联（2026-09-14 实际发生）。

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
            r.setex(DFACTOR_PREFIX + str(symbol).lower(), ttl, json.dumps(factors))
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


# 净值曲线上行点数上限。挖掘窗口（test 2017→今）约 2200 个交易日；原样塞进 MCP
# 响应既胖又没必要 —— 画一条曲线不需要每个交易日一个点。
NET_CURVE_MAX_POINTS = 400


def _downsample_curve(curve: list | None, max_points: int = NET_CURVE_MAX_POINTS) -> list:
    """按**等距下标**抽稀净值曲线，首尾必取。

    为什么是下标抽稀而不是按日期聚合：净值曲线要保留形态（回撤的深度与位置），
    按时间聚合会改变局部极值。等距抽稀在这点上是中性的——它只丢分辨率。

    刻意不做"保留极值点"的聪明抽稀：那会让抽稀后的曲线与 metrics 里的
    max_drawdown 对不上（图上一个更深的谷），而这两个数字本应互相印证。
    """
    if not curve:
        return []
    try:
        n = int(max_points)
    except (TypeError, ValueError):
        n = NET_CURVE_MAX_POINTS
    if n <= 0 or len(curve) <= n:
        return list(curve)
    if n == 1:
        return [curve[-1]]

    # 等距取 n 个下标，首尾必取。用 (len-1) 而不是 len 做分母，末点才恰好命中。
    span = len(curve) - 1
    idxs: list[int] = []
    for k in range(n):
        i = round(k * span / (n - 1))
        if not idxs or i != idxs[-1]:
            idxs.append(i)
    return [curve[i] for i in idxs]


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
            # 透传 traceback：因子失败的真实原因（如沙箱内存不足）之前被吞掉，
            # 只剩一句 "new factor 'x' failed"，完全无法自查（2026-09-15 实测）。
            return _json({"oos": {}, "mining": {}, "decay": None, "ok": False,
                          "mining_net_curve": [], "oos_net_curve": [],
                          "error": f"mining-window backtest failed: {mining.get('error', '')}",
                          "traceback": mining.get("traceback", "")})
        oos = _oos_backtest(code, name, OOS_TEST_START)
        if not oos.get("ok"):
            return _json({"oos": {}, "mining": _pick_metrics(mining["metrics"]),
                          "decay": None, "ok": False,
                          # 挖掘窗口成功了，曲线照样给出去 —— 有半张图总好过没有
                          "mining_net_curve": _downsample_curve(mining.get("net_curve")),
                          "oos_net_curve": [],
                          "error": f"oos-window backtest failed: {oos.get('error', '')}",
                          "traceback": oos.get("traceback", "")})
        m_ic = (mining["metrics"] or {}).get("IC")
        o_ic = (oos["metrics"] or {}).get("IC")
        decay = None
        if m_ic and o_ic is not None:
            decay = 1.0 - float(o_ic) / float(m_ic)
        return _json({
            "oos": _pick_metrics(oos["metrics"]),
            "mining": _pick_metrics(mining["metrics"]),
            "decay": decay,
            # 两条净值曲线（已抽稀）。qlib 本来就跑出了 report，此前只取了三元组
            # metrics，曲线被丢掉。两个窗口画在同一张图上就是 decay 的可视化 ——
            # 这也是把净值放在这里而不是别处的原因：它天然带着"两个窗口"的对比。
            "mining_net_curve": _downsample_curve(mining.get("net_curve")),
            "oos_net_curve": _downsample_curve(oos.get("net_curve")),
            "ok": True,
            "error": "",
        })
    except Exception:  # noqa: BLE001 — tool 通道永不抛异常
        return _json({"oos": {}, "mining": {}, "decay": None, "ok": False,
                      "mining_net_curve": [], "oos_net_curve": [],
                      "error": "factor_oos_check worker exception",
                      "traceback": tb_module.format_exc()})


def _ic_series_payload(ics, counts=None) -> list[dict]:
    """逐日 IC 序列 → 可直接 JSON 的 ``[{date, ic, n}]``。

    ``ics`` 是 ``groupby(level="datetime")`` 的产物，本来就被算出来用于取均值 ——
    这里只是不再把它丢掉。序列是**零额外计算成本**的：截面相关在标量均值算出之前
    就已经逐个交易日算过了。

    ``date`` 统一成 ``YYYY-MM-DD``：调用方要把它画到时间轴上、还要与行情日期对齐，
    ISO 字符串比时间戳少一层歧义。``n`` 是该交易日的有效样本数（因子值与次日收益
    同时存在的标的数）——样本太少时那条 IC 本身不可信，带上它比让前端只看一条
    孤零零的线有用。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for dt in ics.index:
        val = ics.loc[dt]
        if isinstance(val, pd.Series):  # 重复日期会让 .loc 返回 Series
            val = val.iloc[0]
        if not np.isfinite(val):
            continue
        try:
            date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        # 日期唯一化：时间轴上有两个同日期点会让渲染器画出回折，且任何按日期
        # 合并的调用方都会拿到重复键。真实 groupby 不会重复，这里是防御性的。
        if date_str in seen:
            continue
        seen.add(date_str)
        point = {"date": date_str, "ic": float(val)}
        if counts is not None:
            cnt = counts.get(dt)
            point["n"] = None if cnt is None else int(cnt)
        out.append(point)
    return out


def factor_recent_ic(code: str, name: str,
                     lookback_days: int = RECENT_IC_LOOKBACK) -> str:
    """周日衰减巡检（§12.3）：近 lookback_days 个交易日的日均截面 Pearson IC
    （因子值 vs 次日收益），纯 pandas 不走 qlib——周频跑全库成本必须低。

    返回 JSON: {ok, ic, days, series, error}。数据不足（<10 个交易日）记 error。
    ``series`` 是逐日 IC（``[{date, ic, n}]``，按日期升序），供看板画衰减曲线；
    它取自原本就算出来用于取均值的那个 groupby，不增加计算与内存。

    内存口径见 ``_run_factor_window``：整读一次全量（Fixed 格式无法部分读），
    因子值和收盘价都取自同一次暂存，不再整读第二遍。此前两次整读把峰值叠到
    ~1375MiB，在 1536MiB 的容器上限下会触发内核 memcg OOM，把整个服务打死。
    """
    try:
        data_h5 = DATA_DIR / "daily_pv_all.h5"
        if not data_h5.exists():
            return _json({"ok": False, "ic": None, "days": 0, "series": [],
                          "error": f"data file missing: {data_h5}"})
        fac, close = _run_factor_window(code, data_h5)
        # 次日收益（T+1 开盘不可得的近似：close→close），与挖掘 label 口径同族
        ret1 = close.groupby(level="instrument").pct_change().groupby(
            level="instrument").shift(-1).rename("ret")
        del close
        pair = pd.concat([fac, ret1], axis=1).dropna()
        del fac, ret1
        if pair.empty:
            return _json({"ok": False, "ic": None, "days": 0, "series": [],
                          "error": "no overlapping factor/return data"})
        dts = pair.index.get_level_values("datetime").unique().sort_values()
        tail = dts[-max(int(lookback_days), 1):]
        pair = pair.loc[pair.index.get_level_values("datetime").isin(tail)]
        grouped = pair.groupby(level="datetime")
        ics = grouped.apply(lambda x: x["factor"].corr(x["ret"])).dropna()
        # 逐日有效样本数，与 IC 同一批分组，同样零额外成本
        counts = grouped.size()
        if len(ics) < 10:
            return _json({"ok": False, "ic": None, "days": len(ics), "series": [],
                          "error": f"insufficient recent days: {len(ics)}"})
        return _json({"ok": True, "ic": float(ics.mean()), "days": len(ics),
                      "series": _ic_series_payload(ics, counts),
                      "error": ""})
    except Exception:  # noqa: BLE001 — tool 通道永不抛异常
        return _json({"ok": False, "ic": None, "days": 0, "series": [],
                      "error": "factor_recent_ic worker exception",
                      "traceback": tb_module.format_exc()})


# ═══════════════ 因子专业评估：代码 → 专业指标（一条链） ═══════════════
#
# 为什么需要它：`factor_execute` 只回契约检查 {eval_ok, eval_detail}，
# `factor_tearsheet` 需要调用方自带 factor_values + klines。两者之间**没有
# 任何工具能把一段因子代码变成专业指标** —— 因子看板因此卡住。
# factor_evaluate 就是把仓库里已有的两块接起来：沙箱跑代码（
# `_run_factor_window`，因子值与收盘价取自**同一次窗口切片**）+ 同仓库的
# `analytics.factor_tearsheet`（alphalens 口径）。不是新造能力。

EVAL_PERIOD_MAX = 120  # 持有期上限（交易日）；超过它 IC/分层都不再有统计意义


def _period_to_days(key) -> int | None:
    """"5d" → 5；无法解析返回 None（不抛，外部字典可能带任意键）。"""
    try:
        return int(str(key).rstrip("ds").strip())
    except (TypeError, ValueError):
        return None


def quantile_monotonicity(q_mean: dict, quantiles: int) -> dict:
    """分层收益单调性（q1..qQ 的均值序列）。

    专业口径里"单调"是**严格**判断，rho 只表示趋势强度，两者分开报 ——
    把 rho=0.9 的"几乎单调"说成单调，正是因子评估最常见的自欺。

    返回 ``{rho, direction, monotonic, values}``；任一档位缺失（None）时
    三项均为 None/unknown，不猜。
    """
    empty = {"rho": None, "direction": "unknown", "monotonic": None, "values": None}
    if not isinstance(q_mean, dict) or quantiles < 2:
        return empty
    vals = [q_mean.get(f"q{q}") for q in range(1, int(quantiles) + 1)]
    if any(v is None for v in vals):
        return empty
    vals = [float(v) for v in vals]
    rho = None
    try:
        from scipy import stats as _stats  # noqa: PLC0415 — 与 analytics 同款惰性导入

        rho = float(_stats.spearmanr(range(1, len(vals) + 1), vals).statistic)
        if not math.isfinite(rho):
            rho = None
    except Exception:  # noqa: BLE001 — rho 只是强度指标，算不出不影响主结论
        rho = None
    inc = all(a < b for a, b in zip(vals, vals[1:]))
    dec = all(a > b for a, b in zip(vals, vals[1:]))
    direction = "increasing" if inc else ("decreasing" if dec else "non_monotonic")
    return {"rho": rho, "direction": direction,
            "monotonic": bool(inc or dec), "values": vals}


def ic_half_life(decay: dict) -> float | None:
    """IC 衰减半衰期（交易日）：|IC| 从最短持有期跌到其一半所需期数，线性插值。

    输入是 tearsheet 的 ``ic.decay``，如 ``{"1d": 0.064, "5d": 0.041, "10d": 0.022}``。
    半衰期决定**持有期**，比 IC 均值更能说明因子能不能用（§7 四张图的第 2 张）。
    未跌到一半（长周期 IC 反而更高）返回 None —— 那是噪声不是衰减，不插值。
    """
    items = []
    for k, v in (decay or {}).items():
        p = _period_to_days(k)
        if p is None or v is None or p <= 0:
            continue
        items.append((p, abs(float(v))))
    if len(items) < 2:
        return None
    items.sort()
    p0, v0 = items[0]
    if v0 <= 0:
        return None
    half = v0 / 2.0
    for i in range(1, len(items)):
        p_prev, v_prev = items[i - 1]
        p, v = items[i]
        if v <= half:
            span = v_prev - v
            if span <= 0:
                return float(p)
            return float(p_prev + (p - p_prev) * (v_prev - half) / span)
    return None


def ic_t_stat(ic_mean, ic_std, days) -> float | None:
    """IC 的 t 统计量 = mean/std × sqrt(N)（N = 截面 IC 的交易日数）。

    没有它就只能看 IC 均值大小 —— 机构评审先看显著性再看大小（§7 的「t 2.41 ✅」）。
    """
    if ic_mean is None or ic_std is None or days is None:
        return None
    ic_std, days = float(ic_std), int(days)
    if ic_std <= 0 or days < 2:
        return None
    return float(ic_mean) / ic_std * math.sqrt(days)


def _eval_periods(periods: list | None) -> list[int]:
    """持有期入参归一化：正整数、去重、升序、上限 EVAL_PERIOD_MAX；空则 [1,5,10]。"""
    out = []
    for p in (periods or [1, 5, 10]):
        try:
            days = int(p)
        except (TypeError, ValueError):
            continue
        if 1 <= days <= EVAL_PERIOD_MAX and days not in out:
            out.append(days)
    return sorted(out) or [1, 5, 10]


def factor_evaluate(code: str, hypothesis: str = "", quantiles: int = 5,
                    periods: list | None = None, dataset: str = "full",
                    window_days: int | None = None) -> str:
    """沙箱跑一段因子代码 → **直接**返回 alphalens 口径的专业评估（一条链）。

    与 ``factor_execute`` 的区别：那个只回契约检查（eval_ok/eval_detail），
    这个回**专业指标**；与 ``factor_tearsheet`` 的区别：那个要调用方自带
    factor_values + klines，这个自带数据。

    关键实现点：因子值与收盘价取自 ``_run_factor_window`` 的**同一次窗口切片**，
    因此两者天然对齐（不需要事后 intersect，也不会出现日期轴错位）；且只整读
    一次 h5（Fixed 格式无法部分读，两次整读会把容器 OOM 掉，见该函数说明）。

    ``hypothesis``（经济假设）只做**透传 + 缺失标记**，不硬失败：专业看板要求
    假设必填（§4），但"必填"是入库闸门，不该让一次评估拿不到指标。

    返回 JSON 字符串，**永不抛异常**：
    ``{ok, error, tearsheet, monotonicity, ic_half_life_days, ic_t_stat,
       hypothesis, hypothesis_missing, dataset, window_days, n_dates, n_symbols,
       eval_window, quantiles, periods}``
    """
    periods = _eval_periods(periods)
    try:
        quantiles = max(2, int(quantiles))
    except (TypeError, ValueError):
        quantiles = 5
    debug = str(dataset).lower() == "debug"
    data_h5 = DATA_DIR / ("daily_pv_debug.h5" if debug else "daily_pv_all.h5")
    window = 0 if debug else (WINDOW_DAYS if window_days is None else int(window_days))
    window = max(0, int(window))

    ctx = {
        "hypothesis": hypothesis or "",
        "hypothesis_missing": not (hypothesis or "").strip(),
        "dataset": "debug" if debug else "full",
        "window_days": window,
        "quantiles": quantiles,
        "periods": periods,
    }

    if not data_h5.exists():
        return _json({**ctx, "ok": False, "error": f"data file missing: {data_h5}",
                      "tearsheet": None, "monotonicity": {}, "ic_half_life_days": None,
                      "ic_t_stat": None, "n_dates": 0, "n_symbols": 0, "eval_window": {}})

    # ── 1. 沙箱执行（复用 factor_execute 的执行路径，含白名单 + rlimit + 超时）──
    try:
        fac, close = _run_factor_window(code, data_h5, window)
    except Exception as e:  # noqa: BLE001 — 沙箱失败要连 stderr/traceback 一起回
        return _json({**ctx, "ok": False, "error": f"factor execution failed: {e}",
                      "traceback": tb_module.format_exc(), "tearsheet": None,
                      "monotonicity": {}, "ic_half_life_days": None, "ic_t_stat": None,
                      "n_dates": 0, "n_symbols": 0, "eval_window": {}})

    try:
        # panel 契约把 close/c/price/**value** 都当价格列；result.h5 的列名是
        # "factor"，不在别名里，改名成 close 才能被 as_dataframe 接受。
        fac_df = fac.rename("close").reset_index()
        px_df = close.rename("close").reset_index()
        del fac, close
        fac_df = fac_df.dropna(subset=["close"])
        px_df = px_df.dropna(subset=["close"])
        n_dates = int(fac_df["datetime"].nunique())
        n_symbols = int(fac_df["instrument"].nunique())
        dts = sorted(fac_df["datetime"].unique())
        ctx["n_dates"], ctx["n_symbols"] = n_dates, n_symbols
        ctx["eval_window"] = ({
            "start": pd.Timestamp(dts[0]).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(dts[-1]).strftime("%Y-%m-%d"),
        } if dts else {})

        # ── 2. 专业评估（同仓库既有实现，不重写因子分析）──
        # 直接传 DataFrame（panel 契约接受）：全量窗口 ~2.4M 行，转 dict-records
        # 的对象开销会 OOM，而这里根本不需要走 JSON。
        import analytics  # noqa: PLC0415 — scipy/numpy 惰性导入，未装也能 import 本模块

        raw = analytics.factor_tearsheet(fac_df, px_df, quantiles, periods)
        del fac_df, px_df
    except Exception:  # noqa: BLE001 — tool 通道永不抛异常
        return _json({**ctx, "ok": False, "error": "factor_evaluate worker exception",
                      "traceback": tb_module.format_exc(), "tearsheet": None,
                      "monotonicity": {}, "ic_half_life_days": None, "ic_t_stat": None})

    tear = json.loads(raw)
    if "error" in tear:
        # tearsheet 自己走 error 分支（样本不足/形状无法解析）—— 透传原因，
        # 否则上层只能看到"评估失败的评估"，与 §7 说的静默失败同源。
        return _json({**ctx, "ok": False, "error": tear["error"], "tearsheet": None,
                      "monotonicity": {}, "ic_half_life_days": None, "ic_t_stat": None})

    # ── 3. 专业摘要层（单调性 / 半衰期 / t 值）──
    monotonicity = {
        p: quantile_monotonicity(
            ((tear.get("quantile_returns") or {}).get(p) or {}).get("mean_period_return"),
            quantiles)
        for p in tear.get("quantile_returns") or {}
    }
    ic = tear.get("ic") or {}
    return _json({
        **ctx,
        "ok": True,
        "error": "",
        "tearsheet": tear,
        "monotonicity": monotonicity,
        "ic_half_life_days": ic_half_life(ic.get("decay") or {}),
        "ic_t_stat": ic_t_stat(
            ic.get("mean"), ic.get("std"), (ic.get("series_summary") or {}).get("days")),
    })
