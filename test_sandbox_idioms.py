"""沙箱静态检查的回归测试。

2026-09-15 实测：`check_imports` 按**方法名**拒绝 DENIED_FILE_METHODS，
把 `df.rename("factor")` / `df.replace([inf], nan)` 这类完全合法的 pandas
惯用法误判成文件系统变更调用 —— 标准因子模板连静态检查都过不去。
真逃逸风险来自**路径字面量**，不是方法名。
"""

import pytest

from factor_miner.sandbox import SandboxViolation, check_imports


def test_pandas_idioms_are_allowed():
    """与文件方法同名的 pandas 调用必须放行。"""
    check_imports(
        "import pandas as pd\n"
        "out = df.rename('factor')\n"
        "out = out.replace([1, 2], [3, 4])\n"
        "out = out.replace(float('inf'), float('nan'))\n"
    )


def test_path_escape_still_denied():
    """带越界字面量路径的变更调用必须继续被拒（放宽不能丢掉防护）。"""
    for bad in (
        "p.rename('/etc/passwd')",
        "p.replace('../outside.h5')",
        "Path('/tmp/x').rename('/tmp/y')",
        "p.rename('~/secret')",
    ):
        with pytest.raises(SandboxViolation):
            check_imports(bad)


def test_unambiguous_mutators_still_denied_by_name():
    """没有 pandas 同名冲突的变更方法仍按名字无条件拒绝。"""
    for bad in (
        "p.unlink()",
        "p.rmtree()",
        "p.write_text('x')",
        "p.write_bytes(b'x')",
        "p.chmod(0o777)",
        "p.symlink_to('x')",
        "p.hardlink_to('x')",
        "p.mkdtemp()",
    ):
        with pytest.raises(SandboxViolation):
            check_imports(bad)


def test_escape_hatches_still_denied():
    """动态逃逸 builtin 与越界 import 必须继续被拒。"""
    for bad in ("globals()", "eval('1')", "exec('1')", "__import__('os')",
                "getattr(x, 'y')"):
        with pytest.raises(SandboxViolation):
            check_imports(bad)
    for bad in ("import os", "import subprocess", "from shutil import rmtree",
                "import requests"):
        with pytest.raises(SandboxViolation):
            check_imports(bad)


def test_write_mode_open_denied():
    with pytest.raises(SandboxViolation):
        check_imports("open('x', 'w')")
    check_imports("open('x', 'r')")  # 只读放行


def test_template_passes_static_check():
    """派发出去的模板必须能过静态检查（否则作者会先撞在检查器上）。"""
    import pathlib
    tpl = pathlib.Path(__file__).resolve().parent / "templates" / "factor_template.py"
    if not tpl.exists():
        pytest.skip("templates/factor_template.py 不存在")
    src = tpl.read_text()
    # 模板里含模块 docstring，check_imports 只做 import/call 检查，注释无影响
    check_imports(src)
