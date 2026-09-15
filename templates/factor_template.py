"""标准因子脚本模板 —— 满足沙箱的 result.h5 契约。

## 契约（由 factor_miner/sandbox.py + factor_backtest._read_result_h5 定义）

**读**：同目录 ``daily_pv.h5``，HDF key 固定为 ``"data"``
       —— MultiIndex ``(datetime, instrument)``，列名带 qlib 的 ``$`` 前缀：
       ``$open $high $low $close $volume``。
       注意：这是 **Fixed 格式** HDF store，**不能按列读**
       （``columns=`` 会报 ``TypeError: cannot pass a column specification
       when reading a Fixed format store``），只能整体读入。

**写**：同目录 ``result.h5``，HDF key 同样固定为 ``"data"``
       —— MultiIndex ``(datetime, instrument)``，单列因子值
       （列名随意，读取侧取 ``.iloc[:, 0]``）。

2026-09-15 之前这个契约只散落在 factor_backtest 的实现里，没有模板：
脚本没写 result.h5 时，报错是 ``RuntimeError: factor.py exit 0``
（退出码 0 但读不到结果），极难自查。

## 内存策略（踩过的坑）

全量 ``daily_pv_all.h5`` 是 15,126,361 行 × 6 列 float32：

- 整体读入峰值约 **0.9GB RSS / 更高 VA**；
- 再 ``unstack("instrument")`` 成宽表（4304 × 6082）峰值约 **2.0GB**，
  会顶到沙箱 2.15GB 的 ``RLIMIT_AS``（**虚拟地址空间**，含 mmap）而 MemoryError；
- 所以：**不要 unstack**，在长表上 ``groupby(level="instrument")`` 算因子，
  并尽早 ``del`` 掉不再需要的整表。这条路径峰值约 1.2GB，留足余量。

沙箱默认 ``MAX_AS_BYTES`` 已由 2GB 提到 4GB（全量读 + pytables 临时副本需要），
需要更大可设环境变量 ``FACTOR_MINER_MAX_AS_BYTES``。

## 沙箱限制（factor_miner/sandbox.py）

- **import 白名单**：``pandas numpy scipy statsmodels sklearn`` +
  大部分标准库（math/statistics/datetime/collections/itertools/functools/
  operator/copy/json/re/string/random/decimal/fractions/pathlib/typing/
  dataclasses/abc/warnings/io）
- **禁用顶级包**：``os sys subprocess socket shutil signal ctypes
  multiprocessing threading asyncio http urllib requests pickle shelve
  importlib builtins``
- **禁用 builtin**：``eval exec compile globals locals vars getattr setattr
  delattr __import__ breakpoint input``
  → 这是为什么不能用 ``globals()`` 探测命名空间
- **禁用文件方法**：``write_text write_bytes unlink rmdir rmtree rename
  replace chmod …``（``to_hdf`` 可用）
  → ``rename`` / ``replace`` 与 pandas 方法同名，2026-09-15 前会被**误拒**；
  现已改为只在带越界路径字面量（绝对路径 / ``~`` / ``..``）时才拒。
- **路径字面量不得越界**：不能用绝对路径 / ``~`` / ``..``
- 因子脚本在**独立子进程**里跑（rlimit + audit hook），超时 120s（回测路径更长）

调试入口：``factor_execute(code=<本文件内容>, debug=True)``
→ 返回 ``{"stdout", "eval_ok", "eval_detail"}``，``eval_ok=true`` 即契约通过。
"""

import pandas as pd

# ── 1. 整体读入（Fixed 格式不能选列），随后只留一列并释放整表 ──────────
df = pd.read_hdf("daily_pv.h5", key="data")

s = df["$close"].sort_index()
del df  # 其余列不再需要，尽早释放（全量表 0.9GB）

# ── 2. 在长表上算因子（不要 unstack 成宽表：会顶到沙箱地址空间上限） ──
# 这里换成你的因子表达式。示例：20 日反转（超卖反弹）
factor = -s.groupby(level="instrument").pct_change(20)
factor.name = "factor"

# ── 3. 写 result.h5（key 必须也是 "data"） ───────────────────────────
out = factor.to_frame()
out.to_hdf("result.h5", key="data")
