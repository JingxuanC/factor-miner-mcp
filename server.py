#!/usr/bin/env python3
"""Factor Miner MCP Server — 因子挖掘工具集的独立 MCP 服务。

用法:
    python3 server.py --port 50053

端点:
    GET  /health        健康检查
    GET  /tools         工具列表（JSON schema）
    POST /mcp           MCP JSON-RPC（initialize / tools/list / tools/call）
    POST /mcp/factor    兼容路径
    GET  /jobs/<id>     异步任务状态/结果（重负载工具走队列）
    GET  /quota         当前 license key 的额度余量（鉴权模式）
    GET  /queue-stats   队列概况
    GET  /metrics       Prometheus 指标（文本格式，无需鉴权）

鉴权与额度（mcp_gateway.py）：
    环境变量 MCP_LICENSE_FILE 指向 license JSON 时强制鉴权
    （请求头 X-License-Key）；未配置 = 开放模式（本地/内网）。
    重负载工具（factor_execute/factor_backtest/factor_oos_check/
    factor_daily_compute）提交即入队返回 job_id，客户端轮询
    /jobs/<id> 拿结果。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools import EXTRA_SCHEMAS, HANDLERS, TOOLS  # noqa: F401 — 副作用：注册全部工具

from mcp_gateway import METRICS, JobQueue, LicenseStore, QueueFull, QuotaExceeded

logger = logging.getLogger("factor-mcp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

SERVER_NAME = "factor-miner-mcp"
VERSION = "1.0.0"

# 重负载工具：提交后入异步队列执行（返回 job_id 轮询），不占 HTTP 连接。
ASYNC_TOOLS = {"factor_execute", "factor_backtest", "factor_oos_check", "factor_daily_compute",
               "ml_train_rolling", "update_data"}

# 异步任务查询工具（不走 HANDLERS，在 _handle_mcp 里特殊处理；查状态不扣额度）
JOB_STATUS_SCHEMA = {
    "name": "job_status",
    "description": "查询异步任务状态/结果。传入提交重任务时返回的 job_id，"
                   "返回 status（queued/running/done/error）、result 或 error、elapsed_sec。"
                   "重任务提交后用它轮询，无需直接 HTTP 访问 GET /jobs/<id>。",
    "inputSchema": {
        "type": "object",
        "properties": {"job_id": {"type": "string", "description": "异步任务 ID"}},
        "required": ["job_id"],
    },
}


class FactorHandler(BaseHTTPRequestHandler):
    license_store: LicenseStore | None = None
    job_queue: JobQueue | None = None

    def log_message(self, fmt, *args):
        logger.debug("HTTP %s", fmt % args)

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _tool_schemas(self):
        schemas = [t.to_dict() for t in TOOLS.values()]
        schemas.extend(EXTRA_SCHEMAS.values())
        if self.job_queue:
            schemas.append(JOB_STATUS_SCHEMA)
        return schemas

    def _job_status(self, mid, tool_args):
        job_id = str(tool_args.get("job_id", "")).strip()
        job = (self.job_queue.get(job_id, key=self._license_key())
               if job_id and self.job_queue else None)
        if job is None:
            payload = {"job_id": job_id, "status": "not_found",
                       "note": "任务不存在/结果已过期（默认保留 1h），或不属于当前 license key"}
        else:
            payload = job
        METRICS.inc_call("job_status", "ok")
        self._send(200, {"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": False}})

    def _license_key(self) -> str:
        return self.headers.get("X-License-Key", "")

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "version": VERSION, "mode": "factor-mcp",
                             "auth": bool(self.license_store and self.license_store.enabled)})
        elif self.path == "/tools":
            self._send(200, {"tools": self._tool_schemas()})
        elif self.path == "/quota":
            if not (self.license_store and self.license_store.enabled):
                self._send(200, {"mode": "open"})
                return
            ok, info = self.license_store.check(self._license_key())
            if not ok:
                self._send(401, {"error": info})
                return
            self._send(200, self.license_store.quota_of(self._license_key()))
        elif self.path == "/metrics":
            # Prometheus 抓取端点，不要求鉴权（只含工具名级聚合，不泄露 key）
            body = METRICS.render(self.job_queue).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/queue-stats":
            self._send(200, self.job_queue.stats() if self.job_queue else {})
        elif self.path.startswith("/jobs/"):
            if not self.job_queue:
                self._send(404, {"error": "queue disabled"})
                return
            job = self.job_queue.get(self.path[len("/jobs/"):], key=self._license_key())
            if job is None:
                self._send(404, {"error": "job not found"})
                return
            self._send(200, job)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": f"invalid JSON: {e}"})
            return
        if self.path in ("/mcp", "/mcp/factor"):
            self._handle_mcp(data)
        else:
            self._send(404, {"error": "not found"})

    def _handle_mcp(self, data):
        mid = data.get("id")
        method = data.get("method", "")
        params = data.get("params") or {}

        if method == "initialize":
            import uuid as _uuid
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Mcp-Session-Id", str(_uuid.uuid4()))
            self.end_headers()
            self.wfile.write(json.dumps({
                "jsonrpc": "2.0", "id": mid,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": VERSION},
                },
            }).encode())
            return

        if method == "notifications/initialized":
            self._send(200, {"jsonrpc": "2.0", "id": mid, "result": {}})
            return

        if method == "tools/list":
            self._send(200, {"jsonrpc": "2.0", "id": mid,
                             "result": {"tools": self._tool_schemas()}})
            return

        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            # 鉴权：license 模式强制校验 key（initialize/tools/list 保持开放便于发现）
            store = self.license_store
            key = self._license_key()
            if store and store.enabled:
                ok, info = store.check(key)
                METRICS.inc_license_check("ok" if ok else "invalid")
                if not ok:
                    METRICS.inc_call(tool_name, "rejected_license")
                    self._send(200, {"jsonrpc": "2.0", "id": mid,
                                     "error": {"code": -32001, "message": info}})
                    return
            if tool_name == "job_status":
                self._job_status(mid, tool_args)
                return
            if tool_name not in HANDLERS:
                self._send(200, {"jsonrpc": "2.0", "id": mid,
                                 "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}})
                return
            is_async = tool_name in ASYNC_TOOLS and self.job_queue
            # 额度：先扣再跑（异步任务失败不退还——成本已发生）
            if store and store.enabled:
                try:
                    store.consume(key, heavy=bool(is_async))
                except QuotaExceeded as e:
                    METRICS.inc_call(tool_name, "rejected_quota")
                    self._send(200, {"jsonrpc": "2.0", "id": mid,
                                     "error": {"code": -32029, "message": str(e)}})
                    return
            # 重负载 → 入队异步执行，返回 job_id 供轮询
            if is_async:
                try:
                    job_id = self.job_queue.submit(tool_name, tool_args, key=key)
                except QueueFull as e:
                    self._send(200, {"jsonrpc": "2.0", "id": mid,
                                     "error": {"code": -32029, "message": str(e)}})
                    return
                METRICS.inc_call(tool_name, "queued")
                self._send(200, {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": json.dumps({
                        "job_id": job_id, "status": "queued",
                        "poll": f"/jobs/{job_id}",
                        "note": "重任务已入队，调用 job_status 工具传入 job_id 轮询拿结果",
                    }, ensure_ascii=False)}], "isError": False}})
                return
            # 同步执行：记 ok/error + 延迟（异步任务在 JobQueue._worker 完成时记）
            started = time.time()
            try:
                result = HANDLERS[tool_name](**tool_args)
                METRICS.inc_call(tool_name, "ok")
                self._send(200, {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": str(result)}], "isError": False}})
            except Exception as e:  # noqa: BLE001
                logger.error("tool call error %s: %s", tool_name, e)
                METRICS.inc_call(tool_name, "error")
                self._send(200, {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}})
            finally:
                METRICS.observe_latency(tool_name, time.time() - started)
            return

        self._send(200, {"jsonrpc": "2.0", "id": mid,
                         "error": {"code": -32601, "message": f"Unknown method: {method}"}})


def main():
    ap = argparse.ArgumentParser(description="因子挖掘 MCP 服务")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, default=50053, help="监听端口（默认 50053）")
    ap.add_argument("--license-file", default=os.environ.get("MCP_LICENSE_FILE", ""),
                    help="license key JSON 路径（env MCP_LICENSE_FILE）；不配置=开放模式")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MCP_WORKERS", "2")),
                    help="异步任务 worker 数（env MCP_WORKERS，默认 2）")
    ap.add_argument("--queue-size", type=int, default=int(os.environ.get("MCP_QUEUE_SIZE", "50")),
                    help="异步队列上限（env MCP_QUEUE_SIZE，默认 50）")
    args = ap.parse_args()

    FactorHandler.license_store = LicenseStore(args.license_file, domain="factor")
    FactorHandler.job_queue = JobQueue(HANDLERS, workers=args.workers, maxsize=args.queue_size)

    server = ThreadingHTTPServer((args.host, args.port), FactorHandler)
    logger.info("factor MCP listening on %s:%d (tools=%d, auth=%s, workers=%d)",
                args.host, args.port, len(TOOLS) + len(EXTRA_SCHEMAS),
                "on" if FactorHandler.license_store.enabled else "open", args.workers)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        FactorHandler.job_queue.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
