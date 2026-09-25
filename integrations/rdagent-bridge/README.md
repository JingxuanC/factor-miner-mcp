# RD-Agent ⇄ factor-miner 桥

把 RD-Agent 的 qlib 循环回测接到本服务（**零 fork、零 Docker、零 conda**）。

## 用法
把本目录内容放进你的 RD-Agent checkout 里（例如 `RD-Agent/bridge/`），然后：

```bash
export QLIB_FACTOR_RUNNER=bridge.runner.FactorMinerRunner   # 点分路径注入，不改 RD-Agent 源码
export FACTOR_MINER_MCP_URL=http://127.0.0.1:50053/mcp      # 跨机时走 SSH 隧道
export MCP_LICENSE_KEY=<miner 的 license key>
export FACTOR_MINER_PROFILE=smoke                           # 首跑控成本
export FACTOR_COSTEER_PYTHON_BIN=$(which python3)           # 本机常没有 `python`
# baseline 与因子轮必须同 profile（否则 SOTA 对比无意义）
.venv/bin/rdagent fin_factor --loop-n 1
```

## 文件
| 文件 | 作用 |
|---|---|
| `miner_client.py` | MCP 客户端（异步作业 + 轮询；成功终态是 `done` 不是 `ok`） |
| `runner.py` | `FactorMinerRunner`：覆盖 `QlibFactorRunner.develop()`，把「因子代码→回测→指标」交给 miner；含 baseline 前置与产物写回（`qlib_res.csv` + `ret.pkl`） |
| `embed_shim.py` | 本地 OpenAI 兼容 embedding 服务（sentence-transformers 后端）—— 智谱 429/ollama 崩时的兜底 |
| `selftest.py` | 不跑 LLM 的自检：真回测 + 三个必需 metric key + 产物契约 |

## 契约（踩过的坑，别重蹈）
1. `feedback` 层硬编码三个 metric key（`IC` / `1day.excess_return_with_cost.annualized_return` / `...max_drawdown`），缺一个就 `KeyError`；
2. metrics Series **必须 named `"0"`**（上游按字符串重命名列）；
3. `job_status` 成功终态是 **`done`**；`X-License-Key` 必需；
4. 重任务是 per-license 串行单槽，`-32029` 时不要假定"没提交成功"；
5. `--loop-n` 才是跑完整一轮（`--step-n` 是单步）；必须 `dotenv run --` 或走 CLI（否则 `.env` 不加载）。
