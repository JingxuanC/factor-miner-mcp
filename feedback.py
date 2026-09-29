#!/usr/bin/env python3
"""feedback.py — rdagent 式 R&D 闭环的"反馈层"（纯标准库，零依赖）

在 rdagent 的 factor scenario 里，每轮循环是：
  hypothesis(假设) → implement(写因子代码) → 实验(回测) → feedback(反馈) → 下一轮假设

本项目已有完整的"实验层"（factor_evaluate / factor_backtest / factor_oos_check），
但实验结果到下一轮假设之间是断的——agent 只能自己对着 JSON 总结。本模块把
"总结"规则化：同一种失败永远得到同一种诊断和同一种下一步建议，避免 LLM 每轮
自由发挥漏掉检查项（rdagent 的 feedback 也是模板+数据的确定性生成，创意部分
留给 researcher 模型）。

输入容忍三种工具的结果形状（自动识别，调用方不用指定来源）：
  factor_evaluate  → {"tearsheet": {...}, "eval_ok": ...}
  factor_backtest  → {"ok", "dedup_dropped", "sota_broken", "metrics", "correlations"}
  factor_oos_check → {"oos": {...}, "mining": {...}, "decay"}

输出 {"verdict", "feedback", "metrics"}：
  verdict ∈ success / fixable / rejected —— Go 侧可据此记 decision
  feedback  —— Markdown 文本，原样喂回 researcher prompt 即可
  metrics   —— 从结果里提取出的关键值（供聚合看板/追踪进化曲线）
"""
from __future__ import annotations

import json
import math

# 判定阈值（与 worker/Go 侧卡点口径一致：decay > 60% 拒绝见 factor_oos_check）
IC_WEAK = 0.02          # 日均截面 IC 低于此值 = 信号弱
IC_STRONG = 0.05        # 高于此值且 OOS 不衰减 = 可进组合层
DECAY_BAD = 0.6
DEDUP_GATE = 0.99


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _normalize(result) -> dict:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return {"__unparseable__": True}
    return result if isinstance(result, dict) else {"__unparseable__": True}


def _detect_shape(r: dict) -> str:
    if r.get("__unparseable__"):
        return "unknown"
    if "decay" in r or ("mining" in r and "oos" in r):
        return "oos"
    if "dedup_dropped" in r or "correlations" in r or "sota_broken" in r:
        return "backtest"
    if "tearsheet" in r or "eval_ok" in r:
        return "evaluate"
    return "unknown"


def _extract(r: dict, shape: str) -> dict:
    """从各形状里提取统一口径的关键指标（取不到就是 None，不猜）。"""
    out = {"ic": None, "annualized": None, "max_drawdown": None,
           "decay": None, "rank_ic": None, "icir": None}
    if shape == "evaluate":
        t = r.get("tearsheet") or {}
        ic = t.get("ic") or {}
        out["ic"] = _f(ic.get("mean"))
        out["icir"] = _f(ic.get("ir"))
        # long_short 在 tearsheet 顶层（{period: {annualized: ...}}），
        # 不在 quantile_returns 里；monotonicity 在 result 顶层（worker 摘要层算的）
        ls = (t.get("long_short") or {}).get("1d") or {}
        out["annualized"] = _f(ls.get("annualized"))
        mono = r.get("monotonicity") or {}
        out["monotonic"] = any(mono.values()) if isinstance(mono, dict) else None
    elif shape == "backtest":
        m = r.get("metrics") or {}
        out["ic"] = _f(m.get("IC"))
        out["rank_ic"] = _f(m.get("Rank IC"))
        out["icir"] = _f(m.get("ICIR"))
        out["annualized"] = _f(m.get("1day.excess_return_with_cost.annualized_return"))
        out["max_drawdown"] = _f(m.get("1day.excess_return_with_cost.max_drawdown"))
    elif shape == "oos":
        out["ic"] = _f((r.get("mining") or {}).get("ic"))
        out["annualized"] = _f((r.get("mining") or {}).get("annualized_return"))
        out["max_drawdown"] = _f((r.get("mining") or {}).get("max_drawdown"))
        out["oos_ic"] = _f((r.get("oos") or {}).get("ic"))
        out["decay"] = _f(r.get("decay"))
    return out


def build_feedback(hypothesis: str, result, prev: dict | None = None) -> dict:
    """核心入口。prev 可选：上一轮提取出的 metrics（本函数返回值里的 metrics
    字段原样传回即可），用于"相对上一轮有没有进步"的进化追踪。"""
    r = _normalize(result)
    shape = _detect_shape(r)
    m = _extract(r, shape)
    prev = prev or {}

    # ── 诊断（按优先级：硬失败 > 形状未知 > 闸门 > 衰减 > 弱信号 > 通过）──
    lines: list[str] = []
    verdict = "rejected"
    # 硬失败最先判：形状未知的错误 dict（如 oos 失败返回 {error, traceback}）也要
    # 落到这条分支，而不是被"无法识别"覆盖——错误信息本身就是最有价值的反馈
    if r.get("error") and not r.get("ok"):
        verdict = "fixable"
        tb = (r.get("traceback") or "").strip().splitlines()
        tail = tb[-1] if tb else ""
        lines.append(f"**诊断**：实验执行失败（{shape}）。最后一行异常：`{tail}`。"
                     "先修代码再谈假设——异常因子不产生任何信息。")
        if "MemoryError" in (r.get("traceback") or "") or "MemoryError" in str(r.get("error")):
            lines.append("- 内存爆点：因子代码把全量 h5 读成了 numpy。用 rolling/ewm 等向量化"
                         "算子替代 Python 循环，或分块计算。")
        if "whitelist" in str(r.get("error", "")):
            lines.append("- 沙箱白名单违规：只用 pandas/numpy，禁 os/network/文件写。")
    elif shape == "unknown":
        verdict = "fixable"
        lines.append("**诊断**：结果 JSON 无法识别（不是 evaluate/backtest/oos 任一形状），"
                     "先把工具返回原文贴回再分析。")
    elif shape == "backtest" and r.get("dedup_dropped"):
        verdict = "rejected"
        corr = r.get("correlations") or {}
        worst = max((abs(_f(v) or 0) for ics in corr.values()
                     for v in (ics if isinstance(ics, list) else [ics])), default=0)
        lines.append(f"**诊断**：去重闸门丢弃（与 SOTA 日均截面 IC 最高 |{worst:.3f}| ≥ "
                     f"{DEDUP_GATE}）。这不是新因子，是旧因子的重参数化。")
        lines.append("- 下一步：改构造逻辑（不同的数据域/不同的时间尺度/横截面标准化），"
                     "而不是继续调窗口长度。")
    elif shape == "backtest" and r.get("sota_broken"):
        lines.append(f"**注意**：SOTA 因子 {r['sota_broken']} 本轮重算失败被隔离，"
                     "去重对比可能不完整——修复 SOTA 后再下定论。")
    elif m["ic"] is not None and m["ic"] <= -IC_STRONG:
        # 方向反了是性价比最高的失败：信号强、构造对、只差一个负号。
        # 必须放在衰减/弱信号之前判——负 IC 过 decay 分支会得出荒谬结论
        # （"挖掘窗 IC -0.06 是假象"），放过它就漏掉了最便宜的修复路径。
        verdict = "fixable"
        lines.append(f"**诊断**：因子方向反了。IC {m['ic']:.4f}（绝对值 ≥ 强信号线 "
                     f"{IC_STRONG}，但为负）——信号真实存在，只差一个负号。")
        lines.append(f"- 下一步：代码里对因子值取负（`-factor` / `1/factor` 视构造而定），"
                     f"取负后预期 IC ≈ {-m['ic']:.4f}。先重测再谈其他改进。")
    elif m["decay"] is not None and m["decay"] > DECAY_BAD:
        verdict = "rejected"
        lines.append(f"**诊断**：过拟合挖掘窗口。OOS 相对衰减 {m['decay']:.1%}"
                     f"（> {DECAY_BAD:.0%}），挖掘窗 IC {m['ic']} 是假象。")
        lines.append("- 下一步：给因子加经济约束（可解释的中间量）、换验证切片做嵌套检验，"
                     "或降低表达式自由度。")
    elif m["ic"] is not None and m["ic"] < IC_WEAK:
        verdict = "rejected"
        lines.append(f"**诊断**：信号弱。日均截面 IC {m['ic']:.4f} < {IC_WEAK}，"
                     "接近纯噪声（|IC| ~ 0.003）。")
        lines.append("- 下一步：换假设方向比调参数有用——检查因子与经济逻辑的对应关系。")
    elif m["ic"] is not None and m["ic"] >= IC_STRONG \
            and (m["decay"] is None or m["decay"] <= 0.3):
        verdict = "success"
        decay_note = (f"OOS 衰减 {m['decay']:.1%}" if m["decay"] is not None
                      else "无 OOS 验证（建议补 factor_oos_check）")
        lines.append(f"**诊断**：通过。IC {m['ic']:.4f}，{decay_note}。"
                     "可进入组合层（portfolio_optimize）或作为新 SOTA 参与下一轮挖掘。")
    else:
        verdict = "fixable"
        ic_s = f"{m['ic']:.4f}" if m["ic"] is not None else "—"
        lines.append(f"**诊断**：有信号但未达准入线。IC {ic_s}"
                     f"（强信号线 {IC_STRONG}），{'还有衰减空间' if m['decay'] else '缺 OOS 验证'}。"
                     "值得沿当前方向迭代一轮。")

    # ── 与上一轮的进化对比 ──
    if prev.get("ic") is not None and m["ic"] is not None:
        delta = m["ic"] - prev["ic"]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        lines.append(f"\n**进化追踪**：IC {prev['ic']:.4f} → {m['ic']:.4f}（{arrow} {abs(delta):.4f}）"
                     + ("，方向正确" if delta > 0 else "，方向错误——考虑回退上一轮假设"))

    hyp = (hypothesis or "").strip() or "（未提供假设）"
    feedback = (f"## 本轮实验反馈\n\n**假设**：{hyp}\n\n"
                + "\n\n".join(lines)
                + "\n\n**指标快照**：" + json.dumps(
                    {k: v for k, v in m.items() if v is not None}, ensure_ascii=False))
    return {"verdict": verdict, "feedback": feedback, "metrics": m, "shape": shape}


# ═══════════════════════════════════════════════════════════════
# 闭环记忆：每轮 (假设, verdict, metrics) 落 SQLite，跨轮趋势可读
#
# 选 SQLite 而不是 JSONL：服务端已经是状态服务（job/额度/缓存），闭环历史是
# 同类状态——要按假设查询、要防并发写坏、要能攒几千轮后仍然秒查，JSONL 全会
# 撞墙。sqlite3 是标准库，零新增依赖，与 server.py 的 stdlib 哲学一致。
# ═══════════════════════════════════════════════════════════════

import sqlite3 as _sqlite3  # noqa: E402

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_rounds (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    hypothesis TEXT NOT NULL DEFAULT '',
    verdict    TEXT NOT NULL DEFAULT '',
    ic         REAL,
    annualized REAL,
    max_drawdown REAL,
    decay      REAL,
    shape      TEXT
);
"""


def _open(db_path) -> "_sqlite3.Connection":
    conn = _sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")   # 并发读写不互堵（清理线程同时在跑）
    conn.executescript(_SCHEMA)
    return conn


def log_round(db_path, hypothesis: str, verdict: str, metrics: dict) -> dict:
    """追加一轮记录，返回 {"round": id}。写失败不抛——记忆是锦上添花。"""
    import time as _time  # noqa: PLC0415

    try:
        with _open(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO feedback_rounds(ts, hypothesis, verdict, ic, annualized,"
                " max_drawdown, decay, shape) VALUES (?,?,?,?,?,?,?,?)",
                (_time.strftime("%Y-%m-%d %H:%M:%S"), hypothesis or "", verdict,
                 _f(metrics.get("ic")), _f(metrics.get("annualized")),
                 _f(metrics.get("max_drawdown")), _f(metrics.get("decay")),
                 metrics.get("shape") or ""))
            return {"round": cur.lastrowid}
    except _sqlite3.Error:
        return {"round": None}


def history_trend(db_path, n: int = 10) -> dict | None:
    """最近 n 轮的 IC 趋势。库不存在/无有效 IC → None（调用方不输出趋势段）。"""
    import pathlib  # noqa: PLC0415

    if not pathlib.Path(str(db_path)).exists():
        return None
    try:
        with _open(db_path) as conn:
            rows = conn.execute(
                "SELECT ic FROM feedback_rounds WHERE ic IS NOT NULL"
                " ORDER BY id DESC LIMIT ?", (int(n),)).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM feedback_rounds WHERE ic IS NOT NULL"
            ).fetchone()[0]
    except _sqlite3.Error:
        return None
    ics = [r[0] for r in rows][::-1]  # 按时间升序
    if not ics:
        return None
    return {"n": total, "ic_first": ics[0], "ic_last": ics[-1],
            "ic_mean": sum(ics) / len(ics), "window": len(ics)}
