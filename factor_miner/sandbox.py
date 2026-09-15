"""sandbox.py — factor.py 沙箱执行（DESIGN-FACTOR-MINER.md §4）

每个因子一个独立 job 目录 + subprocess：
- 降权执行：有 sudo 时经 `sudo -u nobody` 运行（uid 65534），AST 白名单挡不住
  的 open()/pd.read_* 读文件路径被文件权限兜死——licenses.json 等密钥
  （root:10001 640）nobody 读不到；无 sudo 的环境（开发机）降级为当前用户
- 资源硬限制（setrlimit）：地址空间 2GB / CPU 300s —— sidecar 与 engine 同容器，
  "合法但爆炸"的 pandas 计算不能连累交易主循环
- 墙壁时钟 timeout 120s（RD-Agent 3600s 过宽）
- import 白名单 AST 静态检查：pandas/numpy/scipy/statsmodels/sklearn/标准库
- 环境变量最小化：不注入任何凭证、**不透传 PYTHONPATH**

文件系统隔离（分层防御，见 _BOOTSTRAP_TEMPLATE 与 check_imports）：
1. 静态 AST：拒绝 open()/io.open() 的写模式、拒绝写文本/建链等变更方法、
   拒绝任何文件类调用里的绝对路径 / ~ / ".." 字面量（含 Path(...)）
2. 运行时 audit hook（sys.addaudithook）：拒绝 job 目录外的**写**、
   拒绝 /etc/passwd /etc/shadow /etc/hosts /etc/sudoers /root/** 等敏感**读**，
   并把违规写入 stderr（宿主侧填 SandboxResult.violation）
3. TMPDIR/HOME 重定向进 job 目录，配合 2 使 tempfile/pandas 临时写不越界
4. RLIMIT_AS / RLIMIT_CPU / RLIMIT_FSIZE（见 _apply_limits）
真隔离（chroot/mount namespace/seccomp）需容器权限，本模块在无 root 的环境下
提供上述分层防御 + violation 记录，不宣称等价于容器隔离。

P0 仅提供执行原语；job 编排（factor_worker.py）在 P1。
"""
from __future__ import annotations

import ast
import os
import resource
import subprocess
import sys
from dataclasses import dataclass, field

# RLIMIT_AS：地址空间硬上限。注意这是**虚拟地址空间**（含 mmap），不是 RSS ——
# pandas/pytables 读一张宽表时 VA 远高于实际驻留内存。
#
# 默认从 2GB 提到 4GB：2026-09-15 实测，2GB 连**读完** daily_pv_all.h5 都不够
# （15,126,361 行 × 6 列 float32；报错 Unable to allocate 346 MiB for
# shape (6, 15126361)）。解释器 + mmap 423MB 文件 + 346MiB 数组 + concat 临时副本
# 叠加后 VA 超 2GB。关键背景：daily_pv*.h5 是 **Fixed 格式** HDF store，
# `columns=` 选列不被允许（TypeError: cannot pass a column specification when
# reading a Fixed format store），所以「只读需要的列」这条路走不通，全量读不可避。
#
# _exec_factor_worker（factor_backtest / factor_oos_check 走这条）是**无条件
# symlink 全量数据**的，因此默认值必须能容纳全量读，否则那两个工具开箱即死。
# 实测：4GB 通过，6GB 同样通过；按窗口截取的路径（factor_worker.WINDOW_DAYS，
# 供 factor_recent_ic 用）峰值仅约 1.2GB，不受影响。
MAX_AS_BYTES = int(os.environ.get("FACTOR_MINER_MAX_AS_BYTES", 4 * 1024 * 1024 * 1024))
MAX_CPU_SECONDS = 300
DEFAULT_TIMEOUT_SECONDS = 120
# RLIMIT_FSIZE：单文件最大字节。4GB 远高于 result.h5 实际体量，只为挡"写爆磁盘"
MAX_FILE_BYTES = int(os.environ.get("FACTOR_MINER_MAX_FILE_BYTES", 4 * 1024 * 1024 * 1024))

# 运行时违规标记：bootstrap 的 audit hook 写 stderr，宿主据此填 violation 字段
SANDBOX_VIOLATION_MARK = "[sandbox:violation]"
# 运行时只读敏感名单（读隔离兜底；字面量路径已由静态 AST 拦下）
SENSITIVE_READ_FILES = ("/etc/passwd", "/etc/shadow", "/etc/hosts", "/etc/sudoers",
                        "/etc/gshadow", "/etc/group")
SENSITIVE_READ_PREFIXES = ("/root/", "/etc/ssh/")

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
# 动态逃逸用的 builtin 名（__builtins__/getattr 可绕过静态 open 检查）
DENIED_BUILTINS = {
    "__import__", "eval", "exec", "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input",
    "__builtins__", "__loader__", "__spec__",
}
# 文件系统变更方法：factor.py 只需 df.to_hdf("result.h5")，静态直接拒绝
DENIED_FILE_METHODS = {
    "write_text", "write_bytes", "symlink_to", "hardlink_to", "unlink",
    "rmdir", "rmtree", "mkdtemp", "chmod", "lchmod", "replace", "rename",
}
# DENIED_FILE_METHODS 里与 pandas/DataFrame 方法同名的项：这些名字在因子代码里
# 绝大多数时候是**数据处理**而不是文件系统操作，因此只在带越界路径字面量时拒绝。
# 见 _check_call 的说明（2026-09-15 修的真实误拒）。
_PANDAS_METHOD_NAMES = {"rename", "replace"}
# 会落盘/读盘的调用名：其**字面量路径参数**不容许越界（绝对/~ /".."）
FILE_CALL_NAMES = {
    "open", "Path", "read_csv", "read_table", "read_hdf", "read_parquet",
    "read_pickle", "read_excel", "read_feather", "read_json", "to_csv",
    "to_hdf", "to_parquet", "to_pickle", "to_excel", "to_feather",
    "save", "savez", "savez_compressed", "load", "loadtxt", "savetxt",
    "imread", "imsave", "read_text", "read_bytes",
}
_OPEN_WRITE_CHARS = set("wax+")
# 路径类关键字参数名（pd.read_* / to_* 的路径可能走 kwargs）
_PATH_KWARGS = {"path", "path_or_buf", "filepath_or_buffer", "file", "filename",
                "fname", "name", "dir", "to"}


class SandboxViolation(Exception):
    """AST 白名单拦截。"""


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    violation: str = ""


def _str_const(node) -> "str | None":
    """ast 字符串字面量 → 值（py3.8+ 统一为 Constant）。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _unsafe_literal_path(path: str) -> bool:
    """字面量路径是否越界：绝对路径 / ~ / 含 ".." / Windows 盘符。"""
    t = path.strip()
    if not t or t == "-":
        return False
    if os.path.isabs(t) or t.startswith("~"):
        return True
    if len(t) > 1 and t[1] == ":":
        return True
    return ".." in t.replace("\\", "/").split("/")


def _open_write_mode(node: ast.Call) -> "str | None":
    """open()/Path.open() 的写模式返回值：非只读模式或非字面量模式 → 违规字符串。"""
    args = list(node.args)
    mode_node = args[1] if len(args) >= 2 else None
    for kw in node.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
    if mode_node is None:
        return None  # 缺省模式 = 只读
    mode = _str_const(mode_node)
    if mode is None:
        return "<dynamic mode>"  # 动态模式无法证明是只读 → 拒绝
    return mode if any(c in mode for c in _OPEN_WRITE_CHARS) else None


def _check_call(node: ast.Call) -> None:
    """静态拦截文件系统逃逸：写模式 open / 变更方法 / 越界字面量路径。"""
    name = _call_name(node)
    if name == "open":
        bad_mode = _open_write_mode(node)
        if bad_mode:
            raise SandboxViolation(f"denied: write mode open (mode={bad_mode!r})")
    # 与 pandas 同名的变更方法（rename/replace）只在**带越界字面量路径**时拒绝。
    # 2026-09-15 实测：无条件按方法名拒绝会把 df.rename("factor") /
    # df.replace([inf], nan) 这类完全合法的 pandas 惯用法拒掉，标准因子模板
    # 连静态检查都过不去。真正的逃逸风险来自**路径字面量**，而不是方法名 ——
    # 相对路径的变更都发生在 job 目录内，本来就被允许。
    if name in DENIED_FILE_METHODS and name not in _PANDAS_METHOD_NAMES:
        raise SandboxViolation(f"denied: filesystem-mutating call: {name}()")
    if name in DENIED_FILE_METHODS or name in FILE_CALL_NAMES:
        candidates = list(node.args) + [k.value for k in node.keywords
                                        if k.arg in _PATH_KWARGS]
        for arg in candidates:
            lit = _str_const(arg)
            if lit is not None and _unsafe_literal_path(lit):
                raise SandboxViolation(
                    f"denied: out-of-job path literal {lit!r} in {name}()")


def check_imports(source: str) -> None:
    """AST 静态检查：import 白名单 + 动态逃逸 builtin + 文件系统写/越界路径。

    违规抛 SandboxViolation（不进子进程）。函数名保持向后兼容。
    """
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
        if isinstance(node, ast.Name) and node.id in DENIED_BUILTINS:
            raise SandboxViolation(f"denied builtin: {node.id}")
        if isinstance(node, ast.Call):
            _check_call(node)


def _apply_limits() -> None:  # preexec_fn，子进程 fork 后 exec 前
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS_BYTES, MAX_AS_BYTES))
    except (ValueError, OSError):
        pass  # macOS 开发机不允许调低 RLIMIT_AS（生产 Linux 容器正常生效）
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    _set_fsize_limit()


def _set_fsize_limit() -> None:
    """RLIMIT_FSIZE：单文件写入上限（超限 SIGXFSZ），防沙箱内写爆磁盘。"""
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    except (ValueError, OSError, AttributeError):
        pass  # macOS 上 RLIMIT_FSIZE 可设；不支持则跳过


def qrun_preexec():
    """qrun 子进程的 preexec_fn：只上 RLIMIT_FSIZE。

    刻意**不**复用 _apply_limits：qrun（qlib + lightgbm 多线程全量回测）在 2GB
    RLIMIT_AS 与 300s RLIMIT_CPU 下必被杀，会破坏生产回测。CPU 上限可选，
    经 env FACTOR_MINER_QRUN_CPU_LIMIT（秒，0=不设）显式开启。
    """
    def _apply() -> None:
        _set_fsize_limit()
        cpu = int(os.environ.get("FACTOR_MINER_QRUN_CPU_LIMIT", "0") or 0)
        if cpu > 0:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            except (ValueError, OSError):
                pass
    return _apply


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


_BOOTSTRAP_NAME = "_sandbox_bootstrap.py"

# 运行时 bootstrap：装 audit hook（拒 job 外写 + 敏感读）+ RLIMIT_FSIZE +
# TMPDIR/HOME 重定向，然后 exec job 目录里的 factor.py。占位符由 _render_bootstrap 填。
_BOOTSTRAP_TEMPLATE = '''\
"""sandbox 运行时 bootstrap（由 factor_miner.sandbox 生成，勿手工编辑）。"""
import os
import resource
import sys

MARK = "__MARK__"
JOB = os.path.dirname(os.path.abspath(__file__))
JOB_REAL = os.path.realpath(JOB)
SENSITIVE_FILES = __SENSITIVE_FILES__
SENSITIVE_PREFIXES = __SENSITIVE_PREFIXES__
EXTRA_DENY = [p for p in os.environ.get("FACTOR_MINER_SANDBOX_DENY_READ", "").split(",") if p]
WRITE_BITS = (getattr(os, "O_WRONLY", 1) | getattr(os, "O_RDWR", 2) |
              getattr(os, "O_CREAT", 64) | getattr(os, "O_APPEND", 1024) |
              getattr(os, "O_TRUNC", 512))

try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (__FSIZE__, __FSIZE__))
except Exception:
    pass

# 库级临时/家目录写重定向进 job 目录
for _key, _sub in (("TMPDIR", "tmp"), ("TEMP", "tmp"), ("TMP", "tmp"), ("HOME", "home")):
    _p = os.path.join(JOB, _sub)
    try:
        os.makedirs(_p, mode=0o777, exist_ok=True)
        os.chmod(_p, 0o777)
        os.environ[_key] = _p
    except Exception:
        pass
sys.dont_write_bytecode = True  # 禁 .pyc 写（否则 import 会被下面的 hook 拦）


def _inside(path, root):
    for candidate in (os.path.abspath(path), os.path.realpath(path)):
        for r in (JOB, JOB_REAL):
            try:
                if os.path.commonpath([candidate, r]) == r:
                    return True
            except Exception:
                pass
    return False


def _is_write(mode, flags):
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    if isinstance(flags, int):
        return bool(flags & WRITE_BITS)
    return False


def _hook(event, args):
    if event != "open" or not args:
        return
    path, mode, flags = (tuple(args) + (None, None, None))[:3]
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    p = os.fsdecode(path)
    if not p:
        return
    reason = ""
    if _is_write(mode, flags):
        if not _inside(p, JOB):
            reason = "write outside job dir"
    else:
        abs_p = os.path.abspath(p)
        real = os.path.realpath(p)
        if (abs_p in SENSITIVE_FILES or real in SENSITIVE_FILES
                or abs_p.startswith(SENSITIVE_PREFIXES)
                or real.startswith(SENSITIVE_PREFIXES)
                or any(abs_p.startswith(x) or real.startswith(x) for x in EXTRA_DENY)):
            reason = "read of sensitive path"
    if reason:
        msg = MARK + " " + reason + ": " + p
        sys.stderr.write(msg + "\\n")
        sys.stderr.flush()
        raise PermissionError(msg)


sys.addaudithook(_hook)

_target = os.path.join(JOB, "factor.py")
with open(_target, "rb") as _f:
    _src = _f.read()
exec(compile(_src, "factor.py", "exec"),
     {"__name__": "__main__", "__file__": _target, "__builtins__": __builtins__})
'''


def _render_bootstrap(job_dir: str) -> str:
    """填占位符生成 bootstrap 源码（job_dir 参数保留给调用方语义，路径由 __file__ 推导）。"""
    return (_BOOTSTRAP_TEMPLATE
            .replace("__MARK__", SANDBOX_VIOLATION_MARK)
            .replace("__FSIZE__", str(MAX_FILE_BYTES))
            .replace("__SENSITIVE_FILES__", repr(tuple(SENSITIVE_READ_FILES)))
            .replace("__SENSITIVE_PREFIXES__", repr(tuple(SENSITIVE_READ_PREFIXES))))


def _sandbox_env(cwd: str) -> dict:
    """子进程最小环境：PATH/LANG + job 内的 HOME；**不透传 PYTHONPATH**。

    HOME 指到 job 目录内，配合 runtime audit hook（只放行 job 目录内的写），
    让 ~/.cache、~/.pytablesrc 之类的库级写入也不越界。
    """
    home = os.path.join(cwd, "home")
    try:
        os.makedirs(home, mode=0o777, exist_ok=True)
        os.chmod(home, 0o777)
    except OSError:
        pass
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": home,
        "LANG": "C.UTF-8",
        # 不注入任何凭证/代理/数据库环境变量；不传 PYTHONPATH（防宿主模块注入）
    }


def _extract_violation(stderr: str) -> str:
    """从子进程 stderr 里提取运行时 audit hook 记录的违规行。"""
    if not stderr or SANDBOX_VIOLATION_MARK not in stderr:
        return ""
    for line in stderr.splitlines():
        if SANDBOX_VIOLATION_MARK in line:
            return line.replace(SANDBOX_VIOLATION_MARK, "").strip()
    return ""


def run_python(
    script: str,
    cwd: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    args: list[str] | None = None,
) -> SandboxResult:
    """在 job 目录内以受限子进程执行 python 脚本。"""
    env = _sandbox_env(cwd)
    # 库级临时文件也落在 job 目录内（audit hook 只放行 job 目录内的写）
    tmpdir = os.path.join(cwd, "tmp")
    try:
        os.makedirs(tmpdir, mode=0o777, exist_ok=True)
        os.chmod(tmpdir, 0o777)
    except OSError:
        pass
    env["TMPDIR"] = tmpdir
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
        stderr = e.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        return SandboxResult(
            returncode=-1,
            stdout=stdout,
            stderr=stderr + f"\n[sandbox] timeout {timeout}s",
            timed_out=True,
            violation=_extract_violation(stderr),
        )
    return SandboxResult(returncode=proc.returncode, stdout=proc.stdout,
                         stderr=proc.stderr,
                         violation=_extract_violation(proc.stderr))


def run_factor_source(
    source: str,
    job_dir: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> SandboxResult:
    """静态检查 → 写 factor.py + 运行时 bootstrap → 受限执行。

    静态违规（写模式 open / 越界路径 / 越界 import）不进子进程直接返回；
    动态逃逸由 bootstrap 的 audit hook 拦截并写 stderr，宿主回填 violation。
    """
    try:
        check_imports(source)
    except SandboxViolation as e:
        return SandboxResult(returncode=-1, stdout="", stderr="", violation=str(e))
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "factor.py")
    with open(path, "w") as f:
        f.write(source)
    bootstrap = os.path.join(job_dir, _BOOTSTRAP_NAME)
    with open(bootstrap, "w") as f:
        f.write(_render_bootstrap(job_dir))
    return run_python(_BOOTSTRAP_NAME, cwd=job_dir, timeout=timeout)
