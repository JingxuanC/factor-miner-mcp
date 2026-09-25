"""桥的自检 —— 不跑 LLM 循环，只验证「RD-Agent ⇄ miner」这一段是真的通。

跑法（在 RD-Agent 仓库根目录）：
    MCP_LICENSE_KEY=... FACTOR_MINER_MCP_URL=http://127.0.0.1:50053/mcp \
        .venv/bin/python -m bridge.selftest            # 默认 smoke，约 45s
    PROFILE=full .venv/bin/python -m bridge.selftest   # full，约 3.5~5 分钟

它验证三件事：
    1. 上游硬编码契约：`IMPORTANT_METRICS` 的三个 key 都能被我们的映射填满；
    2. 真回测：miner 返回 ok + 指标（走真实 MCP、真实 qrun 执行器）；
    3. 产物契约：`qlib_res.csv` + `ret.pkl` 能写进工作区（上游 execute() 缺这两个就返回 None）。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

FACTOR_CODE = (
    "import pandas as pd\n"
    "df = pd.read_hdf('daily_pv.h5', key='data')\n"
    "s = df['$close'].sort_index()\n"
    "f = s.groupby(level='instrument').pct_change(2)\n"
    "f.name = 'factor'\n"
    "f.to_frame().to_hdf('result.h5', key='data')\n"
)


def main() -> int:
    from bridge.miner_client import MinerClient
    from bridge.runner import collect_factors, metrics_to_series

    # 0) 上游契约（从 RD-Agent 自己的模块里读，避免我抄错）
    from rdagent.scenarios.qlib.developer.feedback import IMPORTANT_METRICS

    print(f"[0] 上游 IMPORTANT_METRICS = {IMPORTANT_METRICS}")

    # 1) 纯函数：从 RD-Agent 的数据结构里取因子代码
    task = SimpleNamespace(factor_name="bridge_probe_mom2")
    ws = SimpleNamespace(file_dict={"factor.py": FACTOR_CODE}, target_task=task)
    exp = SimpleNamespace(sub_workspace_list=[ws], based_experiments=[])
    factors = collect_factors([exp])
    assert factors == [{"name": "bridge_probe_mom2", "code": FACTOR_CODE}], factors
    print(f"[1] collect_factors ok → {[f['name'] for f in factors]}")

    # 2) 真回测
    profile = os.environ.get("PROFILE", "smoke")
    client = MinerClient()
    print(f"[2] 向 miner 提交 factor_backtest(profile={profile}) …（URL={client.url}）")
    outcome = client.backtest([], factors, profile=profile)
    if not outcome.ok:
        print(f"    ✗ miner 返回失败：{outcome.error}")
        return 1
    print(f"    ✓ ok，elapsed={outcome.elapsed_sec}s metrics={outcome.metrics}")
    print(f"      净值点数={len(outcome.net_curve)} 成交={len(outcome.trades)}")

    # 3) 映射 + 缺 key 会当场炸（而不是等到 feedback 层 KeyError）
    series = metrics_to_series(outcome.metrics)
    missing = [k for k in IMPORTANT_METRICS if k not in series.index]
    if missing:
        print(f"    ✗ 缺 feedback 必需 key：{missing}")
        return 1
    print("[3] metrics 映射 ok（feedback 层三个硬 cod 的 key 齐了）")
    print(series.to_string())

    # 4) 产物写回契约
    from bridge.runner import FactorMinerRunner

    ws_dir = Path(tempfile.mkdtemp(prefix="bridge_ws_"))
    fake_exp = SimpleNamespace(experiment_workspace=SimpleNamespace(workspace_path=ws_dir))
    runner = FactorMinerRunner.__new__(FactorMinerRunner)  # 不跑 __init__（不依赖 rdagent 的 settings）
    runner._write_back(fake_exp, series, outcome)

    res_csv, ret_pkl = ws_dir / "qlib_res.csv", ws_dir / "ret.pkl"
    assert res_csv.exists(), "qlib_res.csv 没写出来"
    assert ret_pkl.exists(), "ret.pkl 没写出来（上游 execute() 缺它直接返回 None）"

    import pandas as pd

    back = pd.read_csv(res_csv, index_col=0).iloc[:, 0]
    for key in IMPORTANT_METRICS:
        assert key in back.index, f"{key} 在写回的 csv 里丢了"
    print(f"[4] 产物 ok → {ws_dir}")
    print(f"    qlib_res.csv:\n{back.to_string()}")

    print("\n✅ 桥自检通过：RD-Agent 的因子代码已经能在生产 miner 上跑出指标，且满足上游产物契约。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
