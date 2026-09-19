"""因子回测执行器 —— 把 qrun 从 miner 容器里搬出去的那一侧。

- `server.py`：执行器本体（HTTP + 子进程隔离 + killpg + 结果解析）
- `client.py`：miner 侧的客户端（零依赖，不 import qlib）
- `parse.py`：qrun 产物解析（与本地口径逐字段一致）

拆分动机、契约与失败语义见 server.py 模块 docstring 与测试。
"""
