"""量化回测 run 台账（SQLite）。

## 为什么需要它

审计（`docs/design/architecture-quant-link-audit.md`）实测出的问题：
miner 侧**唯一的持久化写点**是 `factor_executor/server.py` 的 `result.json`，
而产物目录有 **TTL 24h**（`factor_worker.CLEANUP_TTL_SEC`）。
后果：**今天跑的回测，明天连指标都查不到** —— 说不清"这行 metrics 是哪次跑、什么参数、
哪份数据版本、哪个 mlruns run"。

所以这里把"run 级事实"独立落库：**产物文件可以被 TTL 删，指标与归因信息不能**。

## 设计取舍

- **best-effort**：所有函数吞掉异常并返回安全值 —— 台账写不进去绝不能影响作业终态
  （作业结果的权威来源仍是 job 队列；台账是"历史与归因"）。
- **SQLite 而非 PG**：miner 是被 1536MiB 限制的容器，台账写入必须便宜且无网络依赖；
  库文件落在**已挂载的宿主目录** `/app/usage/`（`/opt/mcp-suite/usage/factor-miner`），
  该目录不参与 backtests 的 TTL 清理。
- **run_id = job_id**：与 `job_status` 用同一个 id，调用方拿到的 job_id 直接就是 run_id，
  不需要第二套编号（少一个会漂移的概念）。
- **`mlruns_run_id` 与 `artifacts_dir` 都落库**：将来按 run 显式取指标（而不是
  "取最新 recorder"）时，凭这两个字段就能定位到确切产物。

字段与 DDL 见 `_DDL`；语义见 audit 文档 §4。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("factor-mcp.ledger")

_LEDGER_PATH_ENV = "FACTOR_MINER_LEDGER_PATH"
_DEFAULT_CANDIDATES = (
    "/app/usage/quant_runs.db",  # 容器内：挂载在宿主 /opt/mcp-suite/usage/factor-miner
    str(Path.home() / ".factor_miner" / "quant_runs.db"),  # 本地开发兜底
)
# net_curve 已由 worker 抽稀到 <=400 点；再限一次总量，避免异常 payload 撑爆一列
_MAX_CURVE_POINTS = 500

_DDL = """
CREATE TABLE IF NOT EXISTS quant_runs (
    run_id        TEXT PRIMARY KEY,
    tool          TEXT NOT NULL,
    tenant        TEXT,
    status        TEXT NOT NULL,
    ok            INTEGER,
    engine        TEXT,
    spec          TEXT,
    data_version  TEXT,
    code_hashes   TEXT,
    mlruns_run_id TEXT,
    artifacts_dir TEXT,
    metrics       TEXT,
    net_curve     TEXT,
    elapsed_sec   REAL,
    error_code    TEXT,
    error_msg     TEXT,
    created_at    REAL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_quant_runs_tool_time ON quant_runs (tool, finished_at DESC);
CREATE INDEX IF NOT EXISTS idx_quant_runs_status   ON quant_runs (status, finished_at DESC);
CREATE INDEX IF NOT EXISTS idx_quant_runs_tenant   ON quant_runs (tenant, finished_at DESC);
"""

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_path_used: Optional[str] = None


def _resolve_path() -> str:
    env = os.environ.get(_LEDGER_PATH_ENV)
    if env:
        return env
    for cand in _DEFAULT_CANDIDATES:
        try:
            Path(cand).parent.mkdir(parents=True, exist_ok=True)
            if os.access(Path(cand).parent, os.W_OK):
                return cand
        except Exception:  # noqa: BLE001
            continue
    return _DEFAULT_CANDIDATES[-1]


def _get_conn() -> sqlite3.Connection:
    global _conn, _path_used
    if _conn is not None:
        return _conn
    path = _resolve_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_DDL)
    conn.commit()
    _conn, _path_used = conn, path
    logger.info("run ledger ready: %s", path)
    return conn


def ledger_path() -> Optional[str]:
    """台账文件路径（未初始化时返回将要使用的路径）。"""
    try:
        _get_conn()
        return _path_used
    except Exception:  # noqa: BLE001
        return None


def _sanitize(obj: Any) -> Any:
    """把 NaN/Inf 换成 None。

    原因（实测踩到）：miner 的指标里 `RankIC` 在 smoke 档常是 NaN，而
    `json.dumps` 默认会写成裸 `NaN` —— 那是**非法 JSON**，Python 能读，但
    LangAlpha 是 TS/JS，`JSON.parse` 会直接抛。台账是要跨语言消费的，所以在这里收口。
    """
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _extract(result_text: str | None) -> dict[str, Any]:
    """从工具返回的 JSON 字符串里抽出可归因字段（缺就留空，不猜）。"""
    out: dict[str, Any] = {}
    if not result_text:
        return out
    try:
        payload = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return out
    if not isinstance(payload, dict):
        return out

    if isinstance(payload.get("metrics"), dict):
        out["metrics"] = _sanitize(payload["metrics"])
    curve = payload.get("net_curve")
    if isinstance(curve, list) and curve:
        out["net_curve"] = _sanitize(curve[:_MAX_CURVE_POINTS])
    for src, dst in (("data_version", "data_version"),
                     ("artifacts_dir", "artifacts_dir"),
                     ("mlruns_run_id", "mlruns_run_id"),
                     ("engine", "engine"),
                     ("spec", "spec"),
                     ("code_hashes", "code_hashes")):
        val = payload.get(src)
        if val not in (None, "", [], {}):
            out[dst] = val
    return out


def record(
    *,
    run_id: str,
    tool: str,
    status: str,
    tenant: str = "",
    ok: Optional[bool] = None,
    result_text: str | None = None,
    error_text: str = "",
    elapsed_sec: Optional[float] = None,
    created_at: Optional[float] = None,
    finished_at: Optional[float] = None,
) -> bool:
    """写一行 run 台账（幂等：同 run_id 覆盖）。**永不抛异常。**"""
    try:
        extra = _extract(result_text)
        row = {
            "run_id": run_id,
            "tool": tool,
            "tenant": tenant or None,
            "status": status,
            "ok": None if ok is None else int(bool(ok)),
            "engine": extra.get("engine"),
            "spec": json.dumps(extra["spec"], ensure_ascii=False)
            if isinstance(extra.get("spec"), (dict, list)) else extra.get("spec"),
            "data_version": extra.get("data_version"),
            "code_hashes": json.dumps(extra["code_hashes"], ensure_ascii=False)
            if isinstance(extra.get("code_hashes"), (dict, list)) else extra.get("code_hashes"),
            "mlruns_run_id": extra.get("mlruns_run_id"),
            "artifacts_dir": extra.get("artifacts_dir"),
            "metrics": json.dumps(extra.get("metrics") or {}, ensure_ascii=False),
            "net_curve": json.dumps(extra.get("net_curve") or [], ensure_ascii=False),
            "elapsed_sec": elapsed_sec,
            "error_code": None,
            "error_msg": (error_text or None),
            "created_at": created_at or time.time(),
            "finished_at": finished_at or time.time(),
        }
        with _lock:
            conn = _get_conn()
            conn.execute(
                """
                INSERT INTO quant_runs (
                    run_id, tool, tenant, status, ok, engine, spec, data_version,
                    code_hashes, mlruns_run_id, artifacts_dir, metrics, net_curve,
                    elapsed_sec, error_code, error_msg, created_at, finished_at
                ) VALUES (
                    :run_id, :tool, :tenant, :status, :ok, :engine, :spec, :data_version,
                    :code_hashes, :mlruns_run_id, :artifacts_dir, :metrics, :net_curve,
                    :elapsed_sec, :error_code, :error_msg, :created_at, :finished_at
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status, ok=excluded.ok, engine=excluded.engine,
                    spec=COALESCE(excluded.spec, quant_runs.spec),
                    data_version=COALESCE(excluded.data_version, quant_runs.data_version),
                    code_hashes=COALESCE(excluded.code_hashes, quant_runs.code_hashes),
                    mlruns_run_id=COALESCE(excluded.mlruns_run_id, quant_runs.mlruns_run_id),
                    artifacts_dir=COALESCE(excluded.artifacts_dir, quant_runs.artifacts_dir),
                    metrics=excluded.metrics, net_curve=excluded.net_curve,
                    elapsed_sec=excluded.elapsed_sec, error_msg=excluded.error_msg,
                    finished_at=excluded.finished_at
                """,
                row,
            )
            conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001 — 台账失败绝不影响作业
        logger.warning("run ledger write failed (%s): %s", run_id, exc)
        return False


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for col in ("metrics", "net_curve"):
        raw = out.get(col)
        if isinstance(raw, str):
            try:
                # 读侧也消毒：修复前写入的行里可能留着裸 NaN（非法 JSON，
                # JS 客户端 JSON.parse 会抛），不能只靠写入侧收口。
                out[col] = _sanitize(json.loads(raw))
            except json.JSONDecodeError:
                pass
    if out.get("ok") is not None:
        out["ok"] = bool(out["ok"])
    return out


def get(run_id: str) -> Optional[dict[str, Any]]:
    try:
        with _lock:
            conn = _get_conn()
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM quant_runs WHERE run_id = ?", (run_id,)
            )
            row = cur.fetchone()
        return _row_to_dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("run ledger read failed: %s", exc)
        return None


def list_runs(
    *, limit: int = 20, tool: str = "", status: str = "", tenant: str = ""
) -> list[dict[str, Any]]:
    """按时间倒序列出 run（不含 net_curve，避免列表接口变得又大又慢）。"""
    try:
        clauses, params = [], []
        if tool:
            clauses.append("tool = ?")
            params.append(tool)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if tenant:
            clauses.append("tenant = ?")
            params.append(tenant)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(int(limit or 20), 200)))
        sql = (
            "SELECT run_id, tool, tenant, status, ok, engine, spec, data_version,"
            " code_hashes, mlruns_run_id, artifacts_dir, metrics, elapsed_sec,"
            " error_msg, created_at, finished_at "
            f"FROM quant_runs {where} ORDER BY finished_at DESC LIMIT ?"
        )
        with _lock:
            conn = _get_conn()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        out = []
        for row in rows:
            item = _row_to_dict(row)
            if isinstance(item.get("metrics"), str):
                try:
                    item["metrics"] = _sanitize(json.loads(item["metrics"]))
                except json.JSONDecodeError:
                    pass
            out.append(item)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("run ledger list failed: %s", exc)
        return []


def stats() -> dict[str, Any]:
    try:
        with _lock:
            conn = _get_conn()
            total = conn.execute("SELECT COUNT(*) FROM quant_runs").fetchone()[0]
            by_status = dict(
                conn.execute(
                    "SELECT status, COUNT(*) FROM quant_runs GROUP BY status"
                ).fetchall()
            )
            last = conn.execute(
                "SELECT MAX(finished_at) FROM quant_runs"
            ).fetchone()[0]
        return {
            "ledger_path": _path_used,
            "total": total,
            "by_status": by_status,
            "last_finished_at": last,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
