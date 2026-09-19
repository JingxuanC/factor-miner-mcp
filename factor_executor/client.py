"""执行器的 HTTP 客户端 —— miner 侧唯一的出口。

零依赖（标准库 urllib），且**不 import qlib**：这个模块会在 miner 容器里被加载，
而 miner 不该再为"回测要用到的东西"付出任何代价。

异步语义是刻意的：`/run` 立刻返回 job_id，客户端轮询到终态。因为 qrun 是分钟级
到小时级的任务，同步等待会占着一条连接，也正是本地那版把 qrun 和调用方同生共死
的根源。
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOB_POLL_INTERVAL = 5.0
# 轮询读超时；qrun 跑在别人的进程里，这里的读只是拿状态，短超时足够
POLL_TIMEOUT = 60.0


class ExecutorError(RuntimeError):
    """执行器不可用或明确拒绝了这个 job。"""


def executor_url() -> str:
    """执行器地址；空 = 未配置，调用方应继续本地执行。"""
    import os

    return os.environ.get("FACTOR_EXECUTOR_URL", "").strip().rstrip("/")


def _post(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"},
                                method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {"error": str(e)}
    except Exception as exc:
        raise ExecutorError(f"executor unreachable at {url}: "
                            f"{type(exc).__name__}: {exc}") from exc


def _get(url: str, timeout: float) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": str(e)}
    except Exception as exc:
        raise ExecutorError(f"executor unreachable at {url}: "
                            f"{type(exc).__name__}: {exc}") from exc


def health() -> dict:
    url = executor_url()
    if not url:
        return {"configured": False}
    try:
        _, body = _get(f"{url}/health", 10.0)
        body["configured"] = True
        return body
    except ExecutorError as exc:
        return {"configured": True, "reachable": False, "error": str(exc)}


def run_backtest(job_dir: Path | str, data_version: str | None = None,
                 deadline: float | None = None) -> dict[str, Any]:
    """把 job 目录交给执行器并等到终态。

    返回执行器的 result dict（`{ok, metrics, net_values, trades, error, ...}`），
    或抛 `ExecutorError`（不可达 / 被拒绝）。**失败的回测不抛异常** —— 它以
    `ok=False` 返回，和本地实现的失败语义一致，调用方不需要分支两套错误处理。
    """
    url = executor_url()
    if not url:
        raise ExecutorError("FACTOR_EXECUTOR_URL is not configured")

    status, body = _post(f"{url}/run",
                         {"job_dir": str(job_dir), "data_version": data_version}, 60.0)
    if status != 200 or "job_id" not in body:
        raise ExecutorError(f"executor refused the job ({status}): "
                            f"{body.get('error') or body}")

    job_id = body["job_id"]
    logger.info("backtest submitted to executor as %s", job_id)
    while True:
        if deadline is not None and time.time() > deadline:
            raise ExecutorError(f"executor job {job_id} did not finish before the deadline")
        _, job = _get(f"{url}/jobs/{job_id}", POLL_TIMEOUT)
        state = job.get("status")
        if state == "done":
            return job.get("result") or {"ok": False, "error": "executor returned no result"}
        if state not in ("running", "queued"):
            raise ExecutorError(f"executor job {job_id} ended in unexpected state {state!r}")
        time.sleep(JOB_POLL_INTERVAL)
