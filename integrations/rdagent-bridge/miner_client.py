"""factor-miner-mcp 的 MCP 客户端 —— RD-Agent 的回测后端。

为什么走 miner MCP（:50053）而不是执行器（:50054）：
    实测执行器的 50054 **没有发布到宿主机**（`docker ps` 只显示容器内 50053/tcp，
    宿主 `curl 127.0.0.1:50054` → HTTP 000），只有同 docker 网络的容器能用
    `http://factor-executor:50054`。而 miner MCP 宿主可达，并且它顺带提供了
    异步作业机制、`/work` 同挂校验、面板与 qlib 数据同源校验。

协议要点（MCP streamable HTTP）：
    initialize → 拿 `Mcp-Session-Id` 响应头 → 后续请求带上该头；
    SSE 响应里正文是 `data: {...}`；鉴权用 `X-License-Key` 请求头（miner 侧 auth=on）。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_URL = "http://127.0.0.1:50053/mcp"
DEFAULT_LICENSE_ENV = "MCP_LICENSE_KEY"

# 终态集合：mcp_gateway.JobQueue 里作业的 status 取值为 queued/running/done/error
# （**没有 "ok"** —— "ok" 是 *tool call* 的计数标签，不是 job 状态。踩过这个坑：
#  按 {"ok","error"} 判断会永远等不到终态，作业其实早跑完了。）
TERMINAL = {"done", "error", "not_found"}
SUCCESS = {"done"}


class MinerError(RuntimeError):
    """miner 返回了 JSON-RPC error（鉴权失败、槽位被占、参数非法……）。"""


class MinerBusy(MinerError):
    """miner 的重任务槽位被别人占着（-32029）。"""


@dataclass
class BacktestOutcome:
    ok: bool
    metrics: dict[str, Any]
    net_curve: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    correlations: dict[str, Any]
    error: str | None
    elapsed_sec: float | None
    raw: dict[str, Any]


class MinerClient:
    def __init__(
        self,
        url: str | None = None,
        license_key: str | None = None,
        *,
        poll_interval: float = 10.0,
        http_timeout: float = 240.0,
    ) -> None:
        self.url = url or os.environ.get("FACTOR_MINER_MCP_URL", DEFAULT_URL)
        self.license_key = license_key or os.environ.get(DEFAULT_LICENSE_ENV, "")
        self.poll_interval = poll_interval
        self.http_timeout = http_timeout
        self._sid: str | None = None

    # ── transport ────────────────────────────────────────────────────────
    def _post(self, payload: dict[str, Any], timeout: float | None = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.license_key:
            headers["X-License-Key"] = self.license_key
        if self._sid:
            headers["Mcp-Session-Id"] = self._sid
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.http_timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self._sid = sid
                return resp.read().decode()
        except urllib.error.HTTPError as exc:  # 4xx/5xx 也要把正文带出来
            raise MinerError(f"HTTP {exc.code}: {exc.read().decode()[:300]}") from exc
        except urllib.error.URLError as exc:
            raise MinerError(f"miner 不可达（{self.url}）：{exc.reason}") from exc

    @staticmethod
    def _unwrap(out: str) -> dict[str, Any]:
        """把 SSE 包装剥成一层的 payload。"""
        match = re.search(r"data:\s*(\{.*\})", out, re.S)
        payload = json.loads(match.group(1) if match else out)
        if "error" in payload:
            err = payload["error"]
            msg = f"{err.get('code')}: {err.get('message')}"
            raise MinerBusy(msg) if err.get("code") == -32029 else MinerError(msg)
        text = payload.get("result", {}).get("content", [{}])[0].get("text", "")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"__text__": text}

    def connect(self) -> "MinerClient":
        if self._sid:
            return self
        self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "rdagent-factor-miner-bridge", "version": "1"},
                },
            }
        )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    # ── tools ────────────────────────────────────────────────────────────
    def call_tool(self, name: str, arguments: dict[str, Any], *, req_id: int = 2) -> dict[str, Any]:
        self.connect()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        return self._unwrap(self._post(payload))

    def job_status(self, job_id: str, *, req_id: int = 99) -> dict[str, Any]:
        return self.call_tool("job_status", {"job_id": job_id}, req_id=req_id)

    def wait(self, job_id: str, *, deadline_sec: float | None = None, progress: bool = True) -> dict[str, Any]:
        """轮询到终态。deadline_sec 缺省取 miner 侧的 EXECUTOR_TIMEOUT + 余量。"""
        budget = deadline_sec if deadline_sec is not None else 1800.0 + 300.0
        started = time.monotonic()
        while time.monotonic() - started < budget:
            time.sleep(self.poll_interval)
            state = self.job_status(job_id)
            status = str(state.get("status"))
            if progress:
                print(
                    f"    [miner {job_id}] status={status} "
                    f"elapsed={time.monotonic() - started:.0f}s",
                    flush=True,
                )
            if status in TERMINAL:
                return state
        raise MinerError(f"等待作业 {job_id} 超时（{budget:.0f}s）")

    # ── 业务封装 ──────────────────────────────────────────────────────────
    def health(self, timeout: float = 8.0) -> bool:
        try:
            self.connect()
            self.call_tool("job_status", {"job_id": "__probe__"}, req_id=5)
            return True
        except MinerError:
            return False

    def backtest(
        self,
        sota: Iterable[dict[str, str]],
        new_factors: Iterable[dict[str, str]],
        *,
        profile: str = "full",
        windows: dict[str, str] | None = None,
        baseline: bool = False,
        deadline_sec: float | None = None,
    ) -> BacktestOutcome:
        """提交 factor_backtest 并等到终态。

        baseline=True → 纯 Alpha20 基线回测（new_factors 必须为空）。RD-Agent 集成需要它：
        上游 `QlibFactorRunner.develop()` 会先跑 baseline，feedback 层再读
        `exp.based_experiments[-1].result` 当 SOTA 对比。

        注意：miner 的重任务槽位是 per-license **串行单槽**，被占时会抛 MinerBusy(-32029)，
        而拿不到 job_id 并不等于没提交成功（实测过），所以 busy 时要核对 `/metrics` 台账。
        """
        self.connect()
        args: dict[str, Any] = {
            "sota": list(sota or []),
            "new_factors": list(new_factors or []),
            "profile": profile,
        }
        if windows:
            args["windows"] = windows
        if baseline:
            args["baseline"] = True
        try:
            submitted = self._unwrap(
                self._post(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "factor_backtest", "arguments": args},
                    }
                )
            )
        except MinerBusy as busy:
            raise MinerBusy(
                f"{busy}（miner 的串行槽位被占；若确认无作业在跑，重启 mcp-factor-miner 可清槽）"
            ) from busy

        job_id = submitted.get("job_id")
        if not job_id:
            raise MinerError(f"提交未返回 job_id：{submitted}")

        started = time.monotonic()
        state = self.wait(job_id, deadline_sec=deadline_sec)
        wall_sec = time.monotonic() - started
        result = state.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"ok": False, "error": result}
        if not isinstance(result, dict):
            result = {
                "ok": False,
                "error": f"job {job_id} 终态={state.get('status')} 无 result："
                f"{state.get('error') or '(空)'}",
            }

        return BacktestOutcome(
            ok=bool(result.get("ok")) and str(state.get("status")) in SUCCESS,
            metrics=result.get("metrics") or {},
            net_curve=result.get("net_curve") or [],
            trades=result.get("trades") or [],
            correlations=result.get("correlations") or {},
            error=result.get("error") or state.get("error"),
            # miner 的 BacktestResult 不一定带 elapsed_sec（实测为 None）→ 缺就用客户端墙钟
            elapsed_sec=result.get("elapsed_sec") or round(wall_sec, 1),
            raw=result,
        )
