"""rdagent ↔ factor-miner-mcp 桥。

两个文件：
    miner_client.py  MCP 客户端（走 :50053，不走未发布的执行器 :50054）
    runner.py        FactorMinerRunner —— 用 miner 替代 qrun/Docker/conda

启用：
    export QLIB_FACTOR_RUNNER=bridge.runner.FactorMinerRunner
    export FACTOR_MINER_MCP_URL=http://127.0.0.1:50053/mcp   # 跨机时用 SSH 隧道
    export MCP_LICENSE_KEY=<miner 的 license key>
"""

__all__ = ["miner_client", "runner"]
