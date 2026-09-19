"""用量落盘必须扛得住并发写。

`LicenseStore.consume()` 在 HTTP 请求线程里被调用，而网关是
`ThreadingHTTPServer`，所以两个请求同时扣额度、同时落盘是常态。

2026-09-17 实测在生产日志里出现过一次：

    WARNING mcp-gateway usage save failed: [Errno 2] No such file or directory:
    '/app/usage/.usage-factor.tmp' -> '/app/usage/.usage-factor.json'

原因是临时文件路径**由目标路径推导**（`.with_suffix(".tmp")`），所有并发写共享
同一个 `.tmp`：A 先 rename 走，B 的 rename 源文件已经不在，于是 ENOENT。而锁只
护住了"裁剪快照"那一段，没护住写盘。

这里用一道屏障让两个线程真正同时到达 rename 点 —— 不依赖调度的运气。
"""

import json
import os
import threading

import mcp_gateway
from mcp_gateway import LicenseStore

KEY = "ak_test_concurrent"
DOMAIN = "factor"


def _store(tmp_path):
    """一个启用了 license 的 store（未配置文件时是开放模式，不会落盘）。"""
    license_file = tmp_path / "licenses.json"
    license_file.write_text(json.dumps({"keys": {KEY: {"name": "t", "daily_quota": 0}}}))
    return LicenseStore(str(license_file), domain=DOMAIN, usage_dir=str(tmp_path))


def test_usage_survives_two_concurrent_saves(tmp_path, monkeypatch, caplog):
    """两个线程同时落盘时，两次扣额度都必须被记下来。

    要同时防住两个失败：**文件系统层**的固定 `.tmp` 会让其中一个 rename 报
    ENOENT（实测过的那条日志），以及**语义层**的丢更新 —— `os.replace` 是
    "最后写者赢"，若写盘不串行，先完成的那次扣减会被后写的那份快照覆盖掉。
    """
    store = _store(tmp_path)
    real_replace = os.replace
    gate = threading.Event()
    replacers = []
    replacers_mu = threading.Lock()

    def gated_replace(src, dst):
        if not str(src).endswith(".tmp"):
            return real_replace(src, dst)
        with replacers_mu:
            replacers.append(threading.current_thread().name)
            n = len(replacers)
        if n == 1:
            # 第一个到达的线程卡在 rename 前，等对手也进来。
            # 只等第一个：若等待的线程不在写盘临界区外、而是持锁等待，
            # 对手永远进不来，这里就会靠 timeout 兜底（测试变慢但不会假绿）。
            gate.wait(timeout=2)
        else:
            gate.set()
        return real_replace(src, dst)

    monkeypatch.setattr(mcp_gateway.os, "replace", gated_replace)

    errors = []
    with caplog.at_level("WARNING"):
        threads = [threading.Thread(target=lambda: store.consume(KEY, heavy=False)) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    failed = [r.getMessage() for r in caplog.records if "usage save failed" in r.getMessage()]
    assert failed == [], f"并发落盘丢了写入：{failed}"
    assert errors == []

    # 两次扣额度：落盘结果里必须是 2，不是被覆盖后的 1。
    saved = json.loads((tmp_path / f".usage-{DOMAIN}.json").read_text())
    today = next(iter(saved))
    assert saved[today][KEY]["calls"] == 2, f"扣减被覆盖（丢更新）：{saved}"


def test_save_leaves_no_temp_file_behind(tmp_path):
    """落盘后不留 `.tmp`：它是中介物，不是产物。"""
    store = _store(tmp_path)
    store.consume(KEY, heavy=False)
    store.consume(KEY, heavy=False)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"残留临时文件：{leftovers}"
    assert (tmp_path / f".usage-{DOMAIN}.json").exists()
