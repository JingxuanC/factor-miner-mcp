"""Factor backtest executor — 把 qrun 从 miner 容器里搬出来的那一侧。

## 为什么单独跑一个服务

`qrun` 是全市场 LGBM 训练 + 组合回测，内存需求是 GB 级；而 miner 容器被限制在
1536 MiB（该上限是 2026-09-16 单容器拖垮整机事故之后加的护栏）。两者塞在同一个
cgroup 里的后果实测过：qrun 撞顶 → **内核 memcg OOM 杀掉整个 miner 容器** →
`unless-stopped` 重启 → 客户端看到 `RemoteProtocolError`。而且失败的 qrun 会变成
**孤儿进程继续占内存**，把容器卡在 1522/1536 MiB，此后任何请求都必然 OOM。

搬出来之后：qrun 有自己的 cgroup、自己的上限，失败只影响这一个 job。

## 契约

执行器只认两样输入（都在 job 目录里）：

    <job_dir>/combined_factors_df.h5   因子面板，(datetime, instrument) 索引，一层列名
    <job_dir>/conf.yaml                qlib 工作流配置；provider_uri 指向执行器自己的挂载

输出写回同一目录，并保持与本地回测一致：

    <job_dir>/mlruns/                  qrun 的实验产物（解析口径见 parse.py）
    <job_dir>/result.json              {ok, metrics, net_values, trades, error, ...}

**要求 miner 与执行器把宿主机上的 backtest 根目录挂到同一个容器路径**，因为请求里
传的是路径而不是文件内容（几十 MB 的面板不适合走 HTTP 请求体）。这是刻意的取舍：
同机部署下共享文件系统比搬字节简单得多，代价是把"必须同挂"变成一个显式前提 ——
所以启动时会检查它，而不是等第一个 job 失败。

## 失败语义

- 每个 job 一个子进程组，超时 **killpg**：不会再留孤儿。
- `RLIMIT_AS` 按 `EXECUTOR_MEM_LIMIT_MB` 上（默认 4096 MiB）。这是本地实现缺的那一环
  （`qrun_preexec` 故意只上 `RLIMIT_FSIZE`，因为 RLIMIT_AS 会杀掉多线程训练）——
  在这里能上，是因为执行器不再和"必须活着"的轻工具共用容器：让它自己死掉，
  比拖死整个服务正确。
- 数据版本不一致时**拒绝执行**，而不是照跑：本地 h5 与执行器 qlib 数据不同源时，
  回测结果会静默偏掉，那比失败更糟。
"""

from __future__ import annotations

import json
import logging
import os
import resource
import signal
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from factor_executor.parse import read_exp_res

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("factor-executor")

# job 目录的**容器内**根路径。它必须是 miner 与执行器共同挂载的宿主机目录，
# 否则请求里带的路径在执行器侧不存在。
JOB_ROOT = Path(os.environ.get("EXECUTOR_JOB_ROOT", "/work"))
# qlib 二进制数据根（挂 qlib_data 这一层，不是 cn_data —— 见下方 provider_uri 说明）
QLIB_DATA_ROOT = Path(os.environ.get("EXECUTOR_QLIB_ROOT", "/qlib_data"))
# 数据版本校验用的 panel；只读 manifest.json。留空则跳过校验（并在响应里标注）
PANEL_H5 = Path(os.environ.get("EXECUTOR_PANEL_H5", "/data/factor_mining/daily_pv_all.h5"))
# 单个 qrun 的墙壁时钟
QRUN_TIMEOUT = float(os.environ.get("EXECUTOR_TIMEOUT", "1800"))
# qrun 子进程地址空间上限。0 = 不限
MEM_LIMIT_MB = int(os.environ.get("EXECUTOR_MEM_LIMIT_MB", "4096"))
# 结果保留时长；执行器只服务当前这次调用，过期条目没有读取者
JOB_TTL = float(os.environ.get("EXECUTOR_JOB_TTL", "7200"))
PORT = int(os.environ.get("EXECUTOR_PORT", "50054"))

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


# ── 数据版本 ────────────────────────────────────────────────────────────────
#
# 与 factor_backtest.data_version 同一套口径（manifest 的 h5_sha256 优先，退化
# mtime+size）。刻意复制而不是 import：执行器不该依赖 miner 的业务模块。

def _manifest_version(h5: Path) -> str:
    try:
        if not h5.exists():
            return ""
        mf = h5.parent / "manifest.json"
        if not mf.exists():
            return ""
        if mf.stat().st_mtime < h5.stat().st_mtime:
            return ""
        sha = str(json.loads(mf.read_text()).get("h5_sha256") or "")
        return f"sha256:{sha}" if len(sha) >= 32 else ""
    except Exception:  # noqa: BLE001
        return ""


def data_version(h5: Path) -> str:
    v = _manifest_version(h5)
    if v:
        return v
    if not h5.exists():
        return ""
    st = h5.stat()
    return f"mtime:{st.st_mtime_ns}:size:{st.st_size}"


def provider_uri() -> str:
    """qlib 的 provider_uri（执行器视角）。

    `cn_data` 由日更的 atomic_swap 整体 rename 替换，所以**必须挂 qlib_data 这一层
    的祖父目录**；写死 cn_data 的更深路径会在一次日更之后指向已被删除的 inode。
    """
    return str(QLIB_DATA_ROOT / "cn_data")


# ── 执行 ────────────────────────────────────────────────────────────────────

@contextmanager
def _address_space_limit():
    """在**当前进程**临时设 RLIMIT_AS，让随后 spawn 的子进程继承它。

    为什么不用 `preexec_fn`：它在多线程进程里是不安全的（fork 后只可能调用
    async-signal-safe 的东西，而这个服务是 ThreadingHTTPServer + 后台执行线程，
    并发 spawn 时会踩到）。`Popen(start_new_session=True)` 走的是 posix_spawn/
    fork+exec 的安全路径，用来拿进程组；内存上限改用"设好再 spawn、spawn 完恢复"
    来传递 —— 两者都不依赖 preexec_fn。

    平台可能拒绝这个上限（macOS 不允许降低 RLIMIT_AS）。那种情况下记一条 warning
    照跑：拿不到上限不该让回测起不来。
    """
    if MEM_LIMIT_MB <= 0:
        yield
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        want = MEM_LIMIT_MB * 1024 * 1024
        # 平台硬上限可能低于我们想要的值；取小者，否则 setrlimit 直接抛
        if hard != resource.RLIM_INFINITY:
            want = min(want, hard)
        resource.setrlimit(resource.RLIMIT_AS, (want, hard))
    except (ValueError, OSError) as exc:  # pragma: no cover — 取决于平台
        logger.warning("could not apply RLIMIT_AS=%sMiB: %s", MEM_LIMIT_MB, exc)
        yield
        return
    try:
        yield
    finally:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except (ValueError, OSError):  # pragma: no cover
            pass


def _run_qrun(job_dir: Path) -> dict[str, Any]:
    conf = job_dir / "conf.yaml"
    if not conf.exists():
        return {"ok": False, "error": f"conf.yaml missing in {job_dir}"}

    started = time.time()
    # 输出**落文件**，不用管道。
    #
    # 管道写端会被 qrun fork 出的子进程继承：父进程退出后写端仍开着，`communicate()`
    # 就等到 EOF（即等到那个子进程也结束）。实测在 spawn 场景下直接吃满超时预算，
    # 日志里只留下一个 elapsed=None。落文件既没有这个陷阱，又让失败现场事后可查
    # （qrun 的输出本来就该留着）。
    log_path = job_dir / "qrun.log"
    # 用 Popen 而不是 subprocess.run：run() 在 TimeoutExpired 之前已经 wait 过子进程，
    # 此时它的 pid 已被回收，os.getpgid(pid) 直接 ProcessLookupError —— 于是"超时杀
    # 进程组"这句话是假的，qrun 拉起来的子进程照旧变孤儿。
    try:
        log_fh = log_path.open("w")
    except OSError as exc:
        return {"ok": False, "error": f"cannot open {log_path}: {exc}"}
    try:
        try:
            with _address_space_limit():
                proc = subprocess.Popen(
                    ["qrun", str(conf)],
                    cwd=str(job_dir),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    # 独立会话/进程组：超时或失败时可按组收割，且不依赖 preexec_fn
                    start_new_session=True,
                )
        except FileNotFoundError:
            return {"ok": False, "error": "qrun not found on PATH"}
        # pgid 必须**现在就记**：_preexec 里 os.setsid() 让子进程成为会话/进程组
        # 组长，所以组号 == 子进程 pid。但父进程一旦被 reap，os.getpgid(pid) 就报
        # ESRCH —— 实测 `_kill_group(proc.pid)` 因此从未真正杀到任何东西，孤儿照旧
        # 活着（orphan 的 pgid 正是那个已消失的 pid）。这就是本项目那个"孤儿 qrun
        # 卡死容器"的问题换了个位置复现，靠记录组号修掉。
        pgid = proc.pid
        try:
            proc.wait(timeout=QRUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            _kill_group(pgid)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover — SIGKILL 之后不该发生
                pass
            return {
                "ok": False,
                "error": f"qrun timeout {QRUN_TIMEOUT:.0f}s",
                "traceback": _tail(log_path),
                "elapsed_sec": round(time.time() - started, 1),
            }
    finally:
        log_fh.close()

    elapsed = round(time.time() - started, 1)
    if proc.returncode != 0:
        # 父进程非 0 退出时也要收进程组：qrun 可能已经 fork 出 worker 才失败，
        # 那些 worker 不会随父进程消失 —— 本地那版把容器卡在 1522/1536 MiB 的
        # 孤儿 qrun 就是这个形状。只在失败路径收，正常退出不动（避免误杀仍在
        # 收尾的后代）。用记下来的 pgid，不用 proc.pid（已被 reap，查不到组号）。
        _kill_group(pgid)
        # 负值 = 被信号杀死；-9 在这个上下文里基本就是 memcg/OOM
        hint = "（很可能是内存不足被 OOM kill：调大 EXECUTOR_MEM_LIMIT_MB 或减少并行回测）" \
            if proc.returncode == -9 else ""
        return {
            "ok": False,
            "error": f"qrun exit {proc.returncode}{hint}",
            "traceback": _tail(log_path),
            "elapsed_sec": elapsed,
        }

    try:
        metrics, net_values, trades = read_exp_res(job_dir, provider_uri())
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": "failed to parse qrun output",
            "traceback": f"{type(exc).__name__}: {exc}\n{_tail(log_path)}",
            "elapsed_sec": elapsed,
        }
    return {
        "ok": True,
        "metrics": metrics,
        "net_values": net_values,
        "trades": trades,
        "elapsed_sec": elapsed,
    }


def _tail(path: Path, limit: int = 4000) -> str:
    """日志尾部 —— 失败现场最相关的那一段。文件缺失/不可读时返回空串。"""
    try:
        return path.read_text(errors="replace")[-limit:]
    except OSError:
        return ""


def _kill_group(pgid: int) -> None:
    """按进程组收割 qrun 及其后代。**pgid 由调用方在 Popen 后立刻取得**。

    必须按**组**杀：qrun 是多线程/多进程的（LightGBM + joblib），只杀直接子进程会
    留下一堆孤儿 —— 这正是本地那版把 miner 容器卡在 1522/1536 MiB 的机制。

    不要在这里调用 os.getpgid(pid)：父进程被 reap 之后它会抛 ESRCH，于是"杀进程组"
    变成一句空话。签名收 pgid 而不是 pid 就是为了让这件事在类型上不可能写错。
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(1.0)


def _execute(job_id: str, job_dir: Path) -> None:
    try:
        result = _run_qrun(job_dir)
    except Exception as exc:
        logger.exception("job %s crashed", job_id)
        result = {"ok": False, "error": f"executor error: {type(exc).__name__}: {exc}"}
    result["job_id"] = job_id
    try:
        (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False))
    except OSError as exc:
        logger.warning("job %s could not persist result: %s", job_id, exc)
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(status="done", result=result, finished_at=time.time())
    logger.info("job %s done ok=%s elapsed=%ss", job_id, result.get("ok"),
                result.get("elapsed_sec"))


def _prune() -> None:
    cutoff = time.time() - JOB_TTL
    with _lock:
        for jid in [k for k, v in _jobs.items() if v.get("finished_at", 0) and v["finished_at"] < cutoff]:
            _jobs.pop(jid, None)


def submit(job_dir: Path, expected_version: str | None = None) -> dict[str, Any]:
    """校验并启动一个回测 job。返回 {job_id, status} 或 {error}。"""
    _prune()
    if not job_dir.is_dir():
        return {"error": f"job_dir not visible to the executor: {job_dir}. "
                         "miner 与执行器必须把宿主机 backtest 根目录挂到同一路径。"}

    if expected_version:
        actual = data_version(PANEL_H5) if str(PANEL_H5) else ""
        if actual and actual != expected_version:
            # 宁可拒绝也不照跑：面板不同源时回测会静默偏掉
            return {"error": "data version mismatch: "
                             f"miner={expected_version} executor={actual}",
                    "kind": "data_version_mismatch"}

    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"status": "running", "job_dir": str(job_dir),
                         "started_at": time.time(), "result": None, "finished_at": 0}
    threading.Thread(target=_execute, args=(job_id, job_dir), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


# ── HTTP ────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # 请求处理线程必须是 daemon：ThreadingHTTPServer 默认把它们建成非守护线程，
    # 于是"服务停了但还有请求线程在等"会**阻塞整个进程退出** —— 关容器时要等
    # 最久的那条请求结束。执行器只服务 job 提交/轮询，没有需要优雅收尾的连接。
    daemon_threads = True

    def log_message(self, fmt, *args):
        logger.debug("%s " + fmt, self.address_string(), *args)

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "mode": "factor-executor",
                "job_root": str(JOB_ROOT),
                "job_root_exists": JOB_ROOT.is_dir(),
                "provider_uri": provider_uri(),
                "provider_exists": (QLIB_DATA_ROOT / "cn_data").is_dir(),
                "panel": str(PANEL_H5),
                "data_version": data_version(PANEL_H5) if str(PANEL_H5) else "",
                "mem_limit_mb": MEM_LIMIT_MB,
                "timeout_sec": QRUN_TIMEOUT,
            })
            return
        if self.path.startswith("/jobs/"):
            jid = self.path[len("/jobs/"):]
            with _lock:
                job = _jobs.get(jid)
            if job is None:
                self._send(404, {"error": "unknown job"})
                return
            self._send(200, {"job_id": jid, "status": job["status"],
                             "result": job["result"]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/run":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "invalid JSON body"})
            return
        job_dir = body.get("job_dir")
        if not job_dir:
            self._send(400, {"error": "job_dir is required"})
            return
        out = submit(Path(job_dir), body.get("data_version"))
        self._send(200 if "job_id" in out else 409, out)


def main() -> int:
    logger.info("job_root=%s (exists=%s)", JOB_ROOT, JOB_ROOT.is_dir())
    logger.info("provider_uri=%s (exists=%s)", provider_uri(),
                (QLIB_DATA_ROOT / "cn_data").is_dir())
    logger.info("panel=%s version=%s", PANEL_H5,
                data_version(PANEL_H5) if str(PANEL_H5) else "(skipped)")
    logger.info("mem_limit=%sMiB timeout=%ss port=%s", MEM_LIMIT_MB, QRUN_TIMEOUT, PORT)
    if not (QLIB_DATA_ROOT / "cn_data").is_dir():
        # 启动就说清楚，而不是等第一个 job 失败
        logger.warning("qlib data not found at %s — qrun 会在 init 阶段失败",
                       provider_uri())
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
