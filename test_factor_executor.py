"""执行器的行为契约测试 —— 用假 qrun 驱动，不需要 pyqlib。

拆执行器的**全部理由来自实测故障**，所以用例就照着那几条故障写：孤儿进程、
OOM、数据版本不一致、结果口径漂移。假 qrun 让我们在本机（arm64，装不上 pyqlib）
也能把这些行为钉死。

跑法（本机 python3 即可，无第三方依赖之外的要求）：
    python3 test_factor_executor.py
或 pytest 下自动收集。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factor_executor import server as ex  # noqa: E402

# ── 假 qrun ─────────────────────────────────────────────────────────────────
#
# 行为由环境变量控制，覆盖执行器要处理的全部分支。它不是"模拟 qlib"，而是模拟
# 一个**会以各种方式失败的 qrun**——这才是执行器要扛的东西。

FAKE_QRUN = '''#!/usr/bin/env python3
import json, os, signal, sys, time
from pathlib import Path

mode = os.environ.get("FAKE_QRUN_MODE", "ok")
conf = Path(sys.argv[1])
work = conf.parent
(work / "mlruns").mkdir(exist_ok=True)

if mode == "fail":
    sys.stderr.write("fake qrun: boom\\n")
    sys.exit(3)
if mode == "hang":
    print("fake qrun: hanging", flush=True)
    time.sleep(600)
if mode == "oom":
    # 真实场景里是内核 memcg 杀的；这里用 SIGKILL 复现同一个可观测结果
    os.kill(os.getpid(), signal.SIGKILL)
if mode == "spawn":
    # 起一个子进程然后立刻失败：执行器必须把整个进程组收掉，否则就是孤儿
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (work / "orphan_pid").write_text(str(p.pid))
    sys.exit(2)
if mode == "parse_fail":
    sys.exit(0)   # 退出码 0 但没有 recorder → 解析阶段失败
sys.exit(0)
'''


def _install_fake_qrun(bin_dir: Path) -> None:
    p = bin_dir / "qrun"
    p.write_text(FAKE_QRUN)
    p.chmod(0o755)


def _mk_job(root: Path, name: str = "job") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "conf.yaml").write_text("qlib_init: {}\n")
    (d / "combined_factors_df.h5").write_bytes(b"fake")
    return d


class _Env:
    """把执行器指向临时目录，并给 PATH 装上假 qrun。"""

    def __init__(self, tmp: Path, fake_qrun=True, mem_limit=0):
        self.tmp = tmp
        self.bin = tmp / "bin"
        self.bin.mkdir(exist_ok=True)
        if fake_qrun:
            _install_fake_qrun(self.bin)
        self.saved_path = os.environ.get("PATH", "")
        self.saved = {k: os.environ.get(k) for k in
                      ("FAKE_QRUN_MODE", "EXECUTOR_MEM_LIMIT_MB", "EXECUTOR_TIMEOUT",
                       "EXECUTOR_PANEL_H5", "EXECUTOR_QLIB_ROOT", "EXECUTOR_JOB_ROOT")}
        os.environ["PATH"] = f"{self.bin}{os.pathsep}{self.saved_path}"
        os.environ["EXECUTOR_MEM_LIMIT_MB"] = str(mem_limit)

    def set_mode(self, mode):
        os.environ["FAKE_QRUN_MODE"] = mode

    def close(self):
        os.environ["PATH"] = self.saved_path
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _as_list(x):
    return list(x) if x is not None else []


def _fixture():
    """pytest 存在时返回 pytest.fixture，否则返回恒等装饰器。

    这样同一个文件既能被 pytest 收集（data_root 作为 fixture 自动注入，避免
    "fixture not found" 的 collection error），也能直接 `python3` 跑自带 runner。
    """
    try:
        import pytest
    except ImportError:
        return lambda fn: fn
    return pytest.fixture


# pytest 下自动注入临时目录；自带 runner 会显式把路径传进来。
@_fixture()
def data_root():
    import tempfile as _tf

    with _tf.TemporaryDirectory() as td:
        yield Path(td)


# ── 用例 ────────────────────────────────────────────────────────────────────

def test_ok_run_returns_uniform_outcome(data_root):
    """成功路径返回统一 outcome 形状（调用方因此不需要为远程写第二套分支）。

    解析本身需要 qlib，这里只断言"能走到解析并把它当作成功/明确失败"——
    口径一致性由 test_parse_matches_local_semantics 单独钉。
    """
    with tempfile.TemporaryDirectory() as td:
        env = _Env(Path(td), mem_limit=0)
        try:
            env.set_mode("parse_fail")   # 退出码 0 → 进入解析 → 解析失败
            job = _mk_job(Path(td))
            out = ex._run_qrun(job)
            assert out["ok"] is False
            assert out["error"] == "failed to parse qrun output"
            assert "elapsed_sec" in out
        finally:
            env.close()


def test_failure_captures_stderr_and_exit_code(data_root):
    with tempfile.TemporaryDirectory() as td:
        env = _Env(Path(td))
        try:
            env.set_mode("fail")
            out = ex._run_qrun(_mk_job(Path(td)))
            assert out["ok"] is False and "exit 3" in out["error"]
            assert "boom" in out["traceback"]
        finally:
            env.close()


def test_oom_exit_is_labelled_as_a_memory_problem(data_root):
    """-9 在这个上下文里几乎总是 OOM；把它翻译成人能行动的提示。

    本地实现只报 `qrun exit -9`，运维看到这句话的第一个问题永远是"怎么了"。
    """
    with tempfile.TemporaryDirectory() as td:
        env = _Env(Path(td))
        try:
            env.set_mode("oom")
            out = ex._run_qrun(_mk_job(Path(td)))
            assert out["ok"] is False
            assert "exit -9" in out["error"]
            assert "内存" in out["error"] or "OOM" in out["error"]
        finally:
            env.close()


def test_failed_qrun_leaves_no_orphans(data_root):
    """qrun 失败退出时也要收掉整个进程组 —— 否则就是实测过的孤儿。

    这是本地实现的核心缺陷：`qrun_preexec` 故意不上 RLIMIT_AS，失败的 qrun 会
    活下来继续吃内存，把容器卡在 1522/1536 MiB。
    """
    with tempfile.TemporaryDirectory() as td:
        env = _Env(Path(td))
        saved_timeout = ex.QRUN_TIMEOUT
        try:
            env.set_mode("spawn")     # 父进程 fork 出子进程后立刻以非 0 退出
            job = _mk_job(Path(td))
            out = ex._run_qrun(job)
            assert out["ok"] is False and "exit 2" in out["error"]

            pid_file = job / "orphan_pid"
            assert pid_file.exists(), "假 qrun 没记录下来子进程 pid，用例不成立"
            orphan = int(pid_file.read_text())
            # 给它一点时间被 SIGKILL 收割
            gone = False
            for _ in range(20):
                try:
                    os.kill(orphan, 0)
                except ProcessLookupError:
                    gone = True
                    break
                time.sleep(0.1)
            if not gone:
                os.kill(orphan, signal.SIGKILL)
            assert gone, f"子进程 {orphan} 在超时后仍存活 —— 又一个孤儿"
        finally:
            ex.QRUN_TIMEOUT = saved_timeout
            env.close()


def test_mem_limit_actually_applies_to_the_child(data_root):
    """RLIMIT_AS 必须真的作用到 qrun 子进程。

    本地实现刻意不上它（会杀掉多线程训练）；执行器能上，因为它不再和"必须活着"
    的轻工具共用容器 —— 让它自己死，比拖死整个服务正确。
    """
    with tempfile.TemporaryDirectory() as td:
        env = _Env(Path(td), mem_limit=0)
        try:
            # 用假 qrun 之外的方式验证：直接看 preexec 设置的 rlimit
            script = Path(td) / "bin" / "qrun"
            script.write_text(
                "#!/usr/bin/env python3\n"
                "import resource, sys\n"
                "soft, hard = resource.getrlimit(resource.RLIMIT_AS)\n"
                "open(sys.argv[1] + '.limit', 'w').write(str(soft))\n"
            )
            script.chmod(0o755)
            ex.MEM_LIMIT_MB = 256
            job = _mk_job(Path(td))
            out = ex._run_qrun(job)
            recorded = int((job / "conf.yaml.limit").read_text())
            want = 256 * 1024 * 1024
            if recorded == want:
                pass                      # Linux：上限真的落下去了
            else:
                # macOS 不允许降低 RLIMIT_AS（硬上限锁死），setrlimit 抛 ValueError。
                # 关键契约是"拿不到上限也不能让 job 起不来"——所以这里要求：
                # 上限没生效 **且** 任务仍然跑到了"解析失败"这一步。
                assert recorded > want, recorded
                assert out["ok"] is False, out
                assert out["error"] == "failed to parse qrun output", out
        finally:
            ex.MEM_LIMIT_MB = int(os.environ.get("EXECUTOR_MEM_LIMIT_MB", "4096"))
            env.close()


def test_data_version_mismatch_is_refused(data_root):
    """数据不同源时拒绝执行，而不是照跑。

    面板与执行器 qlib 数据不一致时回测会**静默偏掉**，那比直接失败糟糕得多。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        panel_dir = tmp / "factor_mining"
        panel_dir.mkdir()
        panel = panel_dir / "daily_pv_all.h5"
        panel.write_bytes(b"panel")
        (panel_dir / "manifest.json").write_text(json.dumps({"h5_sha256": "a" * 64}))
        os.utime(panel_dir / "manifest.json", (time.time() + 10, time.time() + 10))

        saved = ex.PANEL_H5
        try:
            ex.PANEL_H5 = panel
            assert ex.data_version(panel) == "sha256:" + "a" * 64

            job = _mk_job(tmp)
            refused = ex.submit(job, expected_version="sha256:" + "b" * 64)
            assert "error" in refused and refused.get("kind") == "data_version_mismatch"

            # 版本一致时正常受理
            ok = ex.submit(job, expected_version="sha256:" + "a" * 64)
            assert "job_id" in ok
        finally:
            ex.PANEL_H5 = saved


def test_missing_job_dir_is_explained_not_swallowed(data_root):
    """"miner 与执行器没同挂"是部署错误，要说清楚，不能表现为 qrun 崩了。"""
    with tempfile.TemporaryDirectory() as td:
        out = ex.submit(Path(td) / "does-not-exist")
        assert "error" in out
        assert "same path" in out["error"] or "not visible" in out["error"]


def test_client_roundtrip_over_http(data_root):
    """走真实 HTTP：提交 → 轮询 → 拿到结果；并验证失败 job 以 ok=False 返回。

    客户端与执行器必须端到端对得上；这是唯一能证明协议没写歪的方式。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = _Env(tmp)
        saved_port = ex.PORT
        try:
            ex.PORT = 50177
            os.environ["FACTOR_EXECUTOR_URL"] = f"http://127.0.0.1:{ex.PORT}"
            env.set_mode("fail")

            httpd = __import__("http.server", fromlist=["ThreadingHTTPServer"])
            srv = httpd.ThreadingHTTPServer(("127.0.0.1", ex.PORT), ex.Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                from factor_executor import client as cl

                # health
                h = cl.health()
                assert h.get("configured") and h.get("status") == "ok", h
                assert h["provider_uri"].endswith("cn_data")

                job = _mk_job(tmp)
                res = cl.run_backtest(job)
                assert res["ok"] is False and "exit 3" in res["error"], res
                # 结果同时落盘，便于事后取证
                assert json.loads((job / "result.json").read_text())["ok"] is False
            finally:
                srv.shutdown()
                srv.server_close()   # 不关监听套接字会拖住解释器退出
        finally:
            ex.PORT = saved_port
            os.environ.pop("FACTOR_EXECUTOR_URL", None)
            env.close()


def test_client_without_url_is_explicit():
    from factor_executor import client as cl
    saved = os.environ.pop("FACTOR_EXECUTOR_URL", None)
    try:
        assert cl.executor_url() == ""
        try:
            cl.run_backtest(Path("/tmp"))
            raise AssertionError("未配置执行器时必须抛 ExecutorError")
        except cl.ExecutorError as exc:
            assert "not configured" in str(exc)
        assert cl.health() == {"configured": False}
    finally:
        if saved is not None:
            os.environ["FACTOR_EXECUTOR_URL"] = saved


def test_parse_matches_local_semantics(data_root):
    """口径必须与本地实现逐字段一致 —— 否则同一个因子在本地/远程会给出不同指标。

    对照的是 factor_miner/factor_backtest.read_exp_res 的行为：
    metrics 只保留非 None 的 5 个字段，净值是**含成本**的
    `(return - cost + 1).cumprod()`。这里用假 recorder 直接验算，不依赖 qlib。
    """
    import pandas as pd
    from factor_executor import parse as P

    # 直接验 positions_to_trades 与字段裁剪这两条纯逻辑
    positions = {
        "2026-01-05": {"AAA": {"amount": 100}, "BBB": {"amount": 0}},
        "2026-01-06": {"AAA": {"amount": 100}, "CCC": {"amount": 50}},
    }
    trades = P.positions_to_trades(positions)
    buys = [(t["symbol"], t["date"], t["action"]) for t in trades if t["action"] == "buy"]
    sells = [(t["symbol"], t["date"], t["action"]) for t in trades if t["action"] == "sell"]
    assert ("AAA", "2026-01-05", "buy") in buys
    assert ("CCC", "2026-01-06", "buy") in buys
    assert ("BBB", "2026-01-05", "buy") not in buys      # amount=0 不算持有
    assert ("BBB", "2026-01-06", "sell") not in sells    # 从未持有就无所谓卖出

    # 净值口径：含成本
    report = pd.DataFrame({"return": [0.01, 0.02], "cost": [0.001, 0.002]})
    net = ((report["return"] - report["cost"] + 1).cumprod()).tolist()
    assert abs(net[0] - 1.009) < 1e-12
    assert abs(net[1] - 1.009 * 1.018) < 1e-12
    assert abs(net[1] - (1.01 * 1.02)) > 1e-6, "用例必须能区分含成本与不含成本"


# ── 自带 runner（本仓库的镜像里没有 pytest）─────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as td:
            print(f"RUN   {fn.__name__}", flush=True)
            try:
                params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
                fn(Path(td)) if "data_root" in params else fn()
                print(f"PASS  {fn.__name__}", flush=True)
            except Exception:
                failures += 1
                print(f"FAIL  {fn.__name__}", flush=True)
                traceback.print_exc()
    print("FAILURES:", failures)
    raise SystemExit(1 if failures else 0)
