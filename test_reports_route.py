#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_reports_route.py — GET /reports/<file> 静态报告路由测试

起真实 ThreadingHTTPServer（随机端口），验证：
  - 正常返回 text/html
  - 未配置目录 → 404
  - 路径穿越（../、子路径、非 .html）→ 404
"""
import http.client
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report  # noqa: E402
import server  # noqa: E402 — 导入即注册全部工具（server.py 约定）
from server import FactorHandler  # noqa: E402


def _start(tmp_path, reports_dir):
    FactorHandler.license_store = None
    FactorHandler.job_queue = None
    FactorHandler.reports_dir = str(reports_dir) if reports_dir else ""
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FactorHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, httpd.server_address[1]


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), body


def test_serve_html(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    fname = report.save_report("<html><body>hi</body></html>", reports, "demo-20260929")
    httpd, port = _start(tmp_path, reports)
    try:
        status, headers, body = _get(port, f"/reports/{fname}")
        assert status == 200
        assert "text/html" in headers.get("Content-Type", "")
        assert body.decode() == "<html><body>hi</body></html>"
    finally:
        httpd.shutdown()


def test_not_configured_404(tmp_path):
    httpd, port = _start(tmp_path, None)
    try:
        status, _, _ = _get(port, "/reports/x.html")
        assert status == 404
    finally:
        httpd.shutdown()


def test_traversal_blocked(tmp_path):
    (tmp_path / "secret.html").write_text("top-secret")
    reports = tmp_path / "reports"
    reports.mkdir()
    httpd, port = _start(tmp_path, reports)
    try:
        for path in ("/reports/../secret.html", "/reports/%2e%2e/secret.html",
                     "/reports/sub/x.html", "/reports/x.txt",
                     "/reports/a%5cb.html", "/reports/..%2fsecret.html"):
            status, _, _ = _get(port, path)
            assert status == 404, f"{path} should be 404, got {status}"
        # secret.html 本身没被泄露
        status, _, body = _get(port, "/reports/secret.html")
        assert status == 404
    finally:
        httpd.shutdown()


def test_worker_attach_report(tmp_path, monkeypatch=None):
    """_attach_report：配置 REPORT_DIR 时落盘并给 URL，未配置时只内联。"""
    import factor_worker as fw

    payload = {"ok": True}
    # 未配置：只有内联
    old = fw.REPORT_DIR
    try:
        fw.REPORT_DIR = None
        out = fw._attach_report(dict(payload), "<html>x</html>", "t1")
        assert out["html_report"] == "<html>x</html>"
        assert "report_url" not in out

        reports = tmp_path / "r"
        reports.mkdir()
        fw.REPORT_DIR = fw.Path(str(reports))
        out = fw._attach_report(dict(payload), "<html>x</html>", "t2-20260929")
        assert out["report_file"] == "t2-20260929.html"
        assert out["report_url"] == "/reports/t2-20260929.html"
        assert (reports / "t2-20260929.html").read_text() == "<html>x</html>"
    finally:
        fw.REPORT_DIR = old


def test_report_index_lists_files(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    report.save_report("<html>a</html>", reports, "r1")
    report.save_report("<html>b</html>", reports, "r2")
    httpd, port = _start(tmp_path, reports)
    try:
        status, headers, body = _get(port, "/reports/")
        text = body.decode()
        assert status == 200
        assert "text/html" in headers.get("Content-Type", "")
        assert "r1.html" in text and "r2.html" in text
        assert "/reports/r1.html" in text
        # 不配置目录 → 404
        httpd2, port2 = _start(tmp_path, None)
        try:
            assert _get(port2, "/reports/")[0] == 404
        finally:
            httpd2.shutdown()
    finally:
        httpd.shutdown()


def test_report_index_escapes_filename(tmp_path):
    import pathlib

    reports = tmp_path / "reports"
    reports.mkdir()
    # Unix 文件名允许 <>&"：索引页必须转义（存储型 XSS 防护）
    pathlib.Path(reports / "x<script>.html").write_text("x")
    httpd, port = _start(tmp_path, reports)
    try:
        _, _, body = _get(port, "/reports/")
        text = body.decode()
        assert "<script>" not in text
        assert "&lt;script&gt;" in text
    finally:
        httpd.shutdown()
