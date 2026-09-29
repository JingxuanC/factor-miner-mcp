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
    GET  /reports/<f>   回测/OOS HTML 报告静态文件（FACTOR_MINER_REPORT_DIR；
                        由 factor_backtest/factor_oos_check 传 html_report=true 生成，
                        响应里带 report_url；鉴权模式下要求 X-License-Key）
    GET  /quota         当前 license key 的额度余量（鉴权模式）
    GET  /queue-stats   队列概况
    GET  /metrics       Prometheus 指标（文本格式，无需鉴权）

鉴权与额度（mcp_gateway.py）：
    环境变量 MCP_LICENSE_FILE 指向 license JSON 时强制鉴权
    （请求头 X-License-Key）；未配置 = 开放模式（本地/内网）。
    重负载工具（factor_execute/factor_backtest/factor_oos_check/
    factor_daily_compute/factor_evaluate）提交即入队返回 job_id，客户端轮询
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
from mcp_common import coerce_args

logger = logging.getLogger("factor-mcp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

SERVER_NAME = "factor-miner-mcp"
VERSION = "1.0.0"

# 重负载工具：提交后入异步队列执行（返回 job_id 轮询），不占 HTTP 连接。
ASYNC_TOOLS = {"factor_execute", "factor_backtest", "factor_oos_check", "factor_daily_compute",
               "factor_evaluate", "ml_train_rolling", "update_data"}

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
    reports_dir: str = ""  # FACTOR_MINER_REPORT_DIR（server.py main 注入）

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
        # 注意顺序：/reports 与 /reports/（索引页）必须先于 /reports/ 前缀判——
        # "/reports/" 同时是索引路径和文件路由的前缀，startswith 先命中会把
        # 索引请求当成空文件名打到 _serve_report 上（404）。
        elif self.path in ("/reports", "/reports/"):
            self._serve_report_index()
        elif self.path.startswith("/reports/"):
            self._serve_report()
        else:
            self._send(404, {"error": "not found"})

    def _serve_report_index(self):
        """报告目录页：GET /reports → 按 mtime 倒序列出全部报告（链接 + 生成时间）。

        与 _serve_report 同一鉴权口径；未配置目录时与单文件路由一样 404。
        """
        store = self.license_store
        if store and store.enabled:
            ok, info = store.check(self._license_key())
            if not ok:
                self._send(401, {"error": info})
                return
        if not self.reports_dir:
            self._send(404, {"error": "reports dir not configured"})
            return
        import pathlib  # noqa: PLC0415

        root = pathlib.Path(self.reports_dir)
        files = sorted((p for p in root.glob("*.html") if p.is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            import datetime  # noqa: PLC0415
            import html as _html  # noqa: PLC0415 — 文件名转义（见下）

            def _row(p):
                # 文件名来自 glob，而 Unix 文件名可以含 <>&" —— 报告目录通常只有
                # save_report 写入（已消毒），但人工/其他进程放进来的文件不消毒，
                # 索引页又是登录后可看的 HTML：不转义就是一个存储型 XSS 入口。
                name = _html.escape(p.name, quote=True)
                mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
                return (f'<li><a href="/reports/{name}">{name}</a>'
                        f'<span class="meta">  {mtime:%Y-%m-%d %H:%M}  '
                        f'{(p.stat().st_size / 1024):.0f} KB</span></li>')

            items = "".join(_row(p) for p in files)
            body = f"<h1>回测报告归档</h1><ul>{items}</ul>"
        else:
            body = ("<h1>回测报告归档</h1><p>暂无报告（调用 factor_backtest / "
                    "factor_oos_check 时传 html_report=true 生成）</p>")
        page = ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>报告归档</title><style>body{font-family:-apple-system,'PingFang SC',"
                "sans-serif;margin:32px auto;max-width:760px;color:#1c2128}"
                "li{margin:8px 0}.meta{color:#8c959f;font-size:12px}"
                "a{color:#0969da;text-decoration:none}</style></head><body>"
                + body + "</body></html>").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def _serve_report(self):
        """报告静态文件：GET /reports/<file>.html → FACTOR_MINER_REPORT_DIR。

        安全边界：
          - 未配置报告目录 → 404（不泄露路径存在性）
          - 文件名规范化后必须仍落在报告目录内（防 ../ 路径穿越）
          - 鉴权模式下要求有效 license key（报告含策略细节，与 /quota 同级管控）
        """
        store = self.license_store
        if store and store.enabled:
            ok, info = store.check(self._license_key())
            if not ok:
                self._send(401, {"error": info})
                return
        if not self.reports_dir:
            self._send(404, {"error": "reports dir not configured"})
            return
        from urllib.parse import unquote  # noqa: PLC0415

        name = unquote(self.path[len("/reports/"):]).lstrip("/")
        # 只服务 .html；名字里带路径分隔符/空白的直接拒（规范化前就拦）
        if not name.endswith(".html") or "/" in name or "\\" in name or ".." in name:
            self._send(404, {"error": "not found"})
            return
        import pathlib  # noqa: PLC0415

        root = pathlib.Path(self.reports_dir).resolve()
        fpath = (root / name).resolve()
        if fpath.parent != root or not fpath.is_file():
            self._send(404, {"error": "not found"})
            return
        body = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # 报告是静态资产，允许代理短缓存；改动靠新文件名（时间戳 slug）
        self.send_header("Cache-Control", "public, max-age=300")
        self.end_headers()
        self.wfile.write(body)

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
            # 参数类型强制（mcp-common）：LLM 常把数字传成字符串（max_lag="2"）或显式
            # 传 null。放在派发之前 —— 坏参数不消耗配额、不进队列，并给出点名参数与
            # 期望类型的可操作错误（原先要等 handler 抛 TypeError，客户端只看到
            # 一句 Python 异常栈，无法据此纠正）。
            tool_args, _arg_err = coerce_args(HANDLERS, tool_name, tool_args)
            if _arg_err is not None:
                METRICS.inc_call(tool_name, "rejected_args")
                self._send(200, {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": _arg_err}], "isError": True}})
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
    ap.add_argument("--job-timeout", type=int,
                    default=int(os.environ.get("MCP_JOB_TIMEOUT_SEC", "3600")),
                    help="异步任务 handler 级超时秒（env MCP_JOB_TIMEOUT_SEC，默认 3600；"
                         "0=不超时；单工具可再经 MCP_JOB_TIMEOUT_<TOOL> 覆盖）")
    args = ap.parse_args()

    FactorHandler.license_store = LicenseStore(args.license_file, domain="factor")
    FactorHandler.job_queue = JobQueue(HANDLERS, workers=args.workers, maxsize=args.queue_size,
                                       handler_timeout_sec=args.job_timeout)
    FactorHandler.reports_dir = os.environ.get("FACTOR_MINER_REPORT_DIR", "")

    # 磁盘清理：启动时跑一次，之后每小时（job/回测目录 TTL + exec_cache LRU）
    try:
        import factor_worker as _fw
        _summary = _fw.cleanup_disk()
        _fw.start_cleanup_thread()
        logger.info("disk cleanup done at startup: ttl=%ss removed=%s",
                    _fw.CLEANUP_TTL_SEC,
                    {k: len(v) for k, v in _summary.items() if v})
    except Exception as e:  # noqa: BLE001 — 清理失败绝不影响服务启动
        logger.warning("disk cleanup init failed: %s", e)

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
