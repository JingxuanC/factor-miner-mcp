# Factor Miner MCP

A 股量化因子挖掘工具集的独立 MCP（Model Context Protocol）服务。从
[Athena](https://github.com/JingxuanC/Athena) 的 py-sidecar 中抽取
factor 域，让任何 MCP 客户端（Claude Desktop、Kimi Code、Cursor、自研
Agent）都能直接驱动完整的「因子挖掘 → 评估 → 回测 → 上线巡检」流水线。

## 工具清单（16 个）

**因子挖掘**

| 工具 | 说明 | 负载 |
|------|------|------|
| `factor_execute` | 沙箱执行 factor.py（import 白名单 + rlimit + 120s 超时），跑评估电池 | 重（异步） |
| `factor_backtest` | 全量 qlib 回测：SOTA 因子 + 新因子对齐 Alpha20 baseline，qrun 出指标 | 重（异步） |
| `factor_oos_check` | 生产准入 OOS 检查：挖掘窗口 vs 纯样本外窗口，报告 IC/年化/回撤 + 衰减 | 重（异步） |
| `factor_daily_compute` | 每日收盘后计算在线因子，写 Redis `dfactor:{symbol}`（TTL 48h） | 重（异步） |
| `update_data` | qlib cn_data 每日增量更新（三源熔断 + 原子切换）+ 重建 h5 数据集 | 重（异步） |
| `factor_recent_ic` | 衰减巡检：近 N 交易日截面 IC（纯 pandas，无需 qlib） | 轻 |
| `compute_factors` | 从 OHLCV K线计算 Alpha158 风格因子（纯 pandas） | 轻 |
| `predict` | 因子值 → ML 信号预测 | 轻 |

**ML**

| 工具 | 说明 | 负载 |
|------|------|------|
| `ml_train_rolling` | 滚动 LGBM 训练：K线 → Alpha158 因子 + 次日收益标签 → 扩张窗训练，输出 IC/RankIC/Sharpe | 重（异步） |
| `ml_predict` | 用滚动模型出次日收益预测（优先 Redis 因子快照，回退实时计算） | 轻 |
| `ml_metrics` | 训练器状态：最近训练日、逐日指标、模型是否存在、特征清单 | 轻 |

**因子评估与组合**

| 工具 | 说明 | 负载 |
|------|------|------|
| `factor_tearsheet` | Alphalens 式因子完整评估：分位数组收益、多空价差、IC 序列（均值/IR/衰减）、换手率（手写 pandas） | 轻 |
| `portfolio_optimize` | 组合优化：HRP / 等权 / 最小方差内置；装 pypfopt 后支持 `mean_variance`（max Sharpe） | 轻 |
| `regime_detect` | 牛/熊/震荡识别：规则状态机（动量 + 已实现波动率阈值）内置；装 hmmlearn 走 GaussianHMM | 轻 |
| `change_point` | 结构突变检测：CUSUM + 二分分割（水平 + 漂移两路）内置；装 ruptures 走 PELT | 轻 |
| `vol_forecast` | 波动率预测：EWMA（λ=0.94）+ Parkinson 高低价参考内置；装 arch 走 GARCH(1,1) | 轻 |

重负载工具提交即入队返回 `job_id`，轮询 `GET /jobs/<id>` 拿结果，
不占 HTTP 连接。

### 实现说明与算法出处

- `factor_tearsheet` **手写 pandas 而非依赖 alphalens 本体**：alphalens
  已半停维护（上游多年无实质更新），其依赖链与 pandas>=2 冲突频发；
  分位数收益 / IC / 换手率逻辑本身很短，手写可控、可测、零额外依赖。
- HRP — López de Prado (2016), *Building Diversified Portfolios that
  Outperform Out-of-Sample*（相关距离 → 层次聚类 → 拟对角化 → 递归二分）。
- EWMA — RiskMetrics (1996), J.P. Morgan Technical Document，λ=0.94，
  多期预测平坦外推。
- PELT — Killick et al. (2012), *Optimal Detection of Changepoints With
  a Linear Computational Cost*, JASA（可选增强，内置为 CUSUM（Page 1954）
  + 二分分割）。
- HMM — GaussianHMM（收益率 + 滚动波动率两特征，可选增强，内置为规则
  状态机）。
- 全部 5 个工具：**纯 numpy/pandas/scipy 路径开箱可用**，重库
  （pypfopt / hmmlearn / ruptures / arch）惰性导入做可选增强，输出
  `method` 字段标注实际实现。

## 快速开始

```bash
pip install -r requirements.txt
python3 server.py --port 50053
```

验证：

```bash
curl http://127.0.0.1:50053/health
curl http://127.0.0.1:50053/tools
```

接入 MCP 客户端（以 Claude Desktop / Kimi Code 为例）：

```yaml
# mcp 配置
factor:
  url: http://127.0.0.1:50053/mcp
```

## Docker 部署

无需本地 Python 环境，一条命令起服务：

```bash
docker compose up -d        # 构建镜像 + 启动容器（首次构建约 3-5 分钟）
docker compose ps           # 查看状态
docker compose logs -f      # 跟踪日志
```

验证：

```bash
curl http://127.0.0.1:50053/health
curl http://127.0.0.1:50053/tools   # 应返回 16 个工具
```

说明与限制：

- `pyqlib` 按构建架构自动处理（Dockerfile `WITH_QLIB` build-arg，默认 `auto`）：
  - **amd64**（云服务器常见架构）：自动安装官方 manylinux wheel，
    `factor_backtest` / `gen_data` / `update_data` 开箱即用；
  - **aarch64**（Apple Silicon / ARM 服务器）：PyPI 无 linux/aarch64 wheel，
    默认跳过，回测/数据工具返回明确错误，其余 12 个工具不受影响。
    需要时用源码编译构建：`docker build --build-arg WITH_QLIB=1`
    （慢，约 10-20 分钟，构建期需能访问 GitHub；拉不动可换镜像：
    `--build-arg QLIB_GIT_URL=https://gitee.com/mirrors/qlib.git`）；
    或直接用 `docker run --platform linux/amd64` 跑 amd64 镜像
    （QEMU 模拟，慢但能用）。
  - 完全禁用：`--build-arg WITH_QLIB=0`。
  国内构建加速：`--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。
- h5 数据集可通过 volume 挂载：
  `- ./data:/app/data` 并设 `FACTOR_MINER_DATA_DIR=/app/data/factor_mining`。
- `lightgbm` 已随镜像安装（镜像内含 libgomp1 OpenMP 运行时），
  `ml_train_rolling` / `ml_predict` 可用。训练出的模型默认落在容器
  `/tmp/athena_models/`（重建即丢），生产部署请挂卷并设
  `FACTOR_MINER_MODEL_DIR=/app/models`。
- Redis 缓存（`dfactor:*` 写入）可选：设置 `REDIS_URL` 指向可达的
  Redis，缺失时自动降级跳过缓存写入。

license 鉴权（可选）：在 `docker-compose.yml` 中取消注释，把宿主机
`licenses.json` 挂进容器并设置 `MCP_LICENSE_FILE`：

```yaml
environment:
  MCP_LICENSE_FILE: /app/licenses/licenses.json
volumes:
  - ./licenses.json:/app/licenses/licenses.json:ro
```

## 数据准备（回测类工具需要）

`factor_execute` / `factor_backtest` / `factor_oos_check` 依赖 qlib
cn_data 导出的日频量价 h5 数据集：

```bash
pip install pyqlib  # arm64 Linux 需从 GitHub 源码编译，见 requirements.txt 注释
python3 -m factor_miner.update_data          # qlib cn_data 增量更新
python3 -m factor_miner.gen_data --debug     # 调试集（100 股 × 2 年）
python3 -m factor_miner.gen_data --full      # 全量（回测用）
```

`factor_recent_ic` / `compute_factors` / `predict` 纯 pandas 实现，
不依赖 qlib，开箱即用。

## 每日数据同步（update_data）

`update_data` 已封装为 MCP 工具（异步 job）：交易日历对齐 → 三源熔断抓取
（mootdx → 腾讯 → 东财）→ raw+qfq 复权对齐 → staging 校验 → 原子切换 →
自动重建 `daily_pv_all.h5`。任一环节失败保留旧数据，退出码非 0。

部署要点：

- **持久化**：qlib cn_data 必须挂卷（compose 里 `./qlib_data:/app/qlib_data` +
  `QLIB_PROVIDER_URI=/app/qlib_data/cn_data`），否则容器重建数据就没了；
  h5 数据集挂 `./data:/app/data`。
- **冷启动**：增量更新从既有 cn_data 日历尾部续抓，首次需要一份基础数据
  （从现有 Athena 部署拷贝 `~/.qlib/qlib_data/cn_data`，或 qlib 社区 dump），
  之后每日只增量。
- **定时调度**：宿主机 crontab 每个交易日收盘后调一次（非交易日自动空转）：

```cron
# 周一到周五 15:40 / 18:10 各跑一轮（收盘后数据落定有延迟，双轮兜底）
40 15 * * 1-5 curl -s -X POST http://127.0.0.1:50053/mcp -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"update_data","arguments":{}}}'
10 18 * * 1-5 curl -s -X POST http://127.0.0.1:50053/mcp -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"update_data","arguments":{}}}'
```

返回 `job_id`，轮询 `GET /jobs/<id>` 拿结果；也可用 CLI：
`docker exec factor-miner-mcp python3 -m factor_miner.update_data --limit 10`（冒烟）。

## 鉴权与额度（可选）

默认开放模式（本地/内网）。设置环境变量后强制 license key 鉴权：

```bash
export MCP_LICENSE_FILE=/path/to/licenses.json
python3 server.py --port 50053
# 客户端请求头：X-License-Key: <key>
```

license JSON 格式与额度语义见 `mcp_gateway.py`  docstring。`GET /quota`
查余量，`GET /queue-stats` 看队列。

## 端点一览

```
GET  /health        健康检查
GET  /tools         工具 JSON schema 列表
POST /mcp           MCP JSON-RPC（initialize / tools/list / tools/call）
GET  /jobs/<id>     异步任务状态/结果
GET  /quota         license 额度余量（鉴权模式）
GET  /queue-stats   队列概况
GET  /metrics       Prometheus 指标（无需鉴权）
```

## 可观察性 / Observability

`GET /metrics` 输出 Prometheus text exposition 格式（无需鉴权，仅工具名级聚合）：

- `mcp_tool_calls_total{tool,status}` — 调用计数，status ∈ ok/error/rejected_license/rejected_quota/queued
- `mcp_tool_latency_seconds_sum{tool}` / `mcp_tool_latency_seconds_count{tool}` — 延迟累计/次数（异步任务从入队到完成）
- `mcp_license_check_total{result}` — license 校验计数（ok/invalid）
- `mcp_uptime_seconds` — 进程启动至今秒数
- `mcp_queue_depth` — 当前排队任务数（gauge）
- `mcp_queue_jobs_total{status}` — 异步任务完成计数（done/error）

Prometheus 抓取配置示例：

```yaml
scrape_configs:
  - job_name: factor-miner-mcp
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:50053"]
```

## 沙箱安全模型

`factor_execute` 等执行用户提交的 factor.py 时在子进程沙箱中运行：
AST import 白名单（仅 pandas/numpy 等）、rlimit 资源限制、120s 超时、
隔离工作目录。实现见 `factor_miner/sandbox.py`。

## 致谢

- `factor_miner/gen_data.py` 移植自 microsoft/RD-Agent（MIT）
- `factor_miner/qlib_dump_bin.py` 裁剪自 microsoft/qlib v0.9.6（MIT）
- 本项目主体来自 [Athena](https://github.com/JingxuanC/Athena)

## License

MIT
