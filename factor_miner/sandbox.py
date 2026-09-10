"""sandbox.py — factor.py 沙箱执行（DESIGN-FACTOR-MINER.md §4）

每个因子一个独立 job 目录 + subprocess：
- 降权执行：有 sudo 时经 `sudo -u nobody` 运行（uid 65534），AST 白名单挡不住
  的 open()/pd.read_* 读文件路径被文件权限兜死——licenses.json 等密钥
  （root:10001 640）nobody 读不到；无 sudo 的环境（开发机）降级为当前用户
- 资源硬限制（setrlimit）：地址空间 2GB / CPU 300s —— sidecar 与 engine 同容器，
  "合法但爆炸"的 pandas 计算不能连累交易主循环
- 墙壁时钟 timeout 120s（RD-Agent 3600s 过宽）
- import 白名单 AST 静态检查：pandas/numpy/scipy/statsmodels/sklearn/标准库
- 环境变量最小化：不注入任何凭证

P0 仅提供执行原语；job 编排（factor_worker.py）在 P1。
"""
from __future__ import annotations

import ast
import os
import resource
import subprocess
import sys
from dataclasses import dataclass, field

MAX_AS_BYTES = 2 * 1024 * 1024 * 1024  # 2GB 地址空间（宿主机 3.7G 内存，留余量）
MAX_CPU_SECONDS = 300
DEFAULT_TIMEOUT_SECONDS = 120

# import 白名单：顶级包名 → 允许；标准库全允许
ALLOWED_TOP_LEVEL = {
    "pandas", "numpy", "scipy", "statsmodels", "sklearn",
    "math", "cmath", "statistics", "datetime", "time", "calendar",
    "collections", "itertools", "functools", "operator", "copy",
    "json", "re", "string", "random", "decimal", "fractions",
    "pathlib", "typing", "dataclasses", "abc", "warnings", "io",
}
# 显式拒绝（即使在标准库内）：逃逸/网络/进程类
DENIED_TOP_LEVEL = {
    "os", "sys", "subprocess", "socket", "shutil", "signal", "ctypes",
    "multiprocessing", "threading", "asyncio", "http", "urllib",
    "requests", "ftplib", "smtplib", "telnetlib", "pickle", "shelve",
    "importlib", "builtins", "code", "codeop", "pty", "mmap",
}


class SandboxViolation(Exception):
    """AST 白名单拦截。"""


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    violation: str = ""


def check_imports(source: str) -> None:
    """AST 静态检查 import 白名单；违规抛 SandboxViolation。"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for top in names:
            if top in DENIED_TOP_LEVEL:
                raise SandboxViolation(f"denied import: {top}")
            if top not in ALLOWED_TOP_LEVEL:
                raise SandboxViolation(f"import not in whitelist: {top}")
        # __import__ / eval / exec / open 写盘等动态逃逸
        if isinstance(node, ast.Name) and node.id in {"__import__", "eval", "exec", "compile"}:
            raise SandboxViolation(f"denied builtin: {node.id}")


def _apply_limits() -> None:  # preexec_fn，子进程 fork 后 exec 前
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS_BYTES, MAX_AS_BYTES))
    except (ValueError, OSError):
        pass  # macOS 开发机不允许调低 RLIMIT_AS（生产 Linux 容器正常生效）
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))


SANDBOX_UID = 65534  # nobody；sudoers 规则见 Dockerfile（factor-sandbox）

_SUDO_OK: "bool | None" = None  # sudo 降权可用性探测缓存（None=未探测）


def _sudo_drop_allowed(sudo: str) -> bool:
    """探测 sudoers 是否配了 appuser→nobody 规则（Dockerfile 里装），结果缓存。

    没配规则的机器（开发机）直接降级为当前用户执行，不让 sudo 报错污染每次调用。
    """
    global _SUDO_OK
    if _SUDO_OK is None:
        try:
            r = subprocess.run([sudo, "-n", "-u", f"#{SANDBOX_UID}",
                                sys.executable, "-c", "pass"],
                               capture_output=True, timeout=10)
            _SUDO_OK = r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _SUDO_OK = False
    return _SUDO_OK


def _drop_cmd(cmd: list[str], env: dict, cwd: str) -> list[str]:
    """sudo 降权可用时，包一层 `sudo -u nobody` 执行。

    job 目录 chmod 0777 / 脚本 0644，让 nobody 能进目录、读脚本、写 result.h5；
    密钥文件（640 root:10001）对 nobody 不可读，AST 白名单的文件读取盲区
    由文件权限兜底。sudo 不可用（开发机）原样返回，降级为当前用户执行。
    """
    import shutil
    sudo = shutil.which("sudo")
    if not sudo or os.geteuid() in (0, SANDBOX_UID) or not _sudo_drop_allowed(sudo):
        return cmd
    try:
        os.chmod(cwd, 0o777)
        target = cmd[1] if os.path.isabs(cmd[1]) else os.path.join(cwd, cmd[1])
        os.chmod(target, 0o644)
    except OSError:
        pass
    env_args = [f"{k}={v}" for k, v in env.items()]
    return [sudo, "-n", "-u", f"#{SANDBOX_UID}", *env_args, *cmd]


def run_python(
    script: str,
    cwd: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    args: list[str] | None = None,
) -> SandboxResult:
    """在 job 目录内以受限子进程执行 python 脚本。"""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "/tmp",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "LANG": "C.UTF-8",
        # 不注入任何凭证/代理/数据库环境变量
    }
    cmd = _drop_cmd([sys.executable, script, *(args or [])], env, cwd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_apply_limits,
        )
    except subprocess.TimeoutExpired as e:
        return SandboxResult(
            returncode=-1,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + f"\n[sandbox] timeout {timeout}s",
            timed_out=True,
        )
    return SandboxResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def run_factor_source(
    source: str,
    job_dir: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> SandboxResult:
    """白名单检查 → 写 factor.py → 受限执行。违规不进子进程直接返回。"""
    try:
        check_imports(source)
    except SandboxViolation as e:
        return SandboxResult(returncode=-1, stdout="", stderr="", violation=str(e))
    path = os.path.join(job_dir, "factor.py")
    with open(path, "w") as f:
        f.write(source)
    return run_python("factor.py", cwd=job_dir, timeout=timeout)
