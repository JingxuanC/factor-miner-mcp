#!/usr/bin/env python3
"""report.py — 回测结果 → 自包含 HTML 报告渲染（纯标准库）

设计约束（与项目"纯 pandas 手写 tearsheet"同一哲学）：
  - **零第三方依赖**：不引 plotly/matplotlib，图表用内联 SVG 手写折线/柱状，
    离线单文件可直接双击打开、可归档、可贴进任何 wiki。
  - **零外部资源**：不引 CDN 字体/JS/CSS，全部 inline；报告在隔离网段也能看。
  - **永不抛异常**：渲染是"锦上添花"层，任何畸形输入都降级为占位文本，
    不能反过来弄挂 factor_backtest 工具通道（/call-tool 约定：工具永不抛）。

消费的数据形状 = factor_worker._run_backtest 的返回 dict：
  ok / dedup_dropped / sota_broken / metrics / correlations /
  net_values / net_curve / error / traceback
另有 meta（profile、新因子名、SOTA 名、数据版本、耗时）由调用方补充，
缺省时按占位渲染。

图表实现要点：
  - 折线用 <polyline>，先等距抽稀到 <= MAX_POINTS 个点（回测净值 1369 行
    没必要全画），再线性映射到 viewBox 坐标。
  - 回撤子图从净值曲线现算：running_max 之后 (v/peak - 1)，
    与 metrics 里的 max_drawdown 同源同口径。
  - 所有用户可控字符串（因子名、error、traceback）过 html.escape。
"""
from __future__ import annotations

import html
import math
import time

MAX_POINTS = 600          # 单条折线最大点数（抽稀后）
_WIDTH = 860              # SVG viewBox 宽
_HEIGHT = 260             # 主图高
_DD_HEIGHT = 120          # 回撤子图高
_MARGIN = {"l": 56, "r": 16, "t": 14, "b": 26}  # 坐标轴留白

_METRIC_LABELS = {
    "IC": ("IC", ""),
    "Rank IC": ("Rank IC", ""),
    "ICIR": ("ICIR", ""),
    "1day.excess_return_with_cost.annualized_return": ("年化超额（扣费）", ""),
    "1day.excess_return_with_cost.max_drawdown": ("最大回撤（扣费）", ""),
    "1day.excess_return_with_cost.information_ratio": ("信息比率（扣费）", ""),
}


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt(v) -> str:
    """数字格式化：None/非有限值 → '—'，其余保留 4 位有效小数。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:.4f}"


def _downsample(points: list, max_points: int = MAX_POINTS) -> list:
    """等距抽稀，保持首尾。点数已达标时原样返回。"""
    if len(points) <= max_points:
        return list(points)
    step = (len(points) - 1) / (max_points - 1)
    return [points[int(i * step)] for i in range(max_points)]


def _curve_xy(curve: list) -> tuple[list, list]:
    """net_curve [{date,i,value}] → (xs, ys)，只取有限正值（qlib 净值恒正）。"""
    xs, ys = [], []
    for p in curve or []:
        try:
            v = float(p.get("value"))
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(v) and v > 0:
            xs.append(p.get("date") or p.get("i"))
            ys.append(v)
    return xs, ys


def _svg_line(ys: list, width: int, height: int, color: str,
              y_fmt: str = "{:.2f}") -> str:
    """单条折线 SVG（无 x 轴标签，只有 y 轴 min/max 刻度）。"""
    if len(ys) < 2:
        return f'<div class="empty">数据点不足（{len(ys)}），无法绘制</div>'
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1e-12
    iw = width - _MARGIN["l"] - _MARGIN["r"]
    ih = height - _MARGIN["t"] - _MARGIN["b"]
    pts = " ".join(
        f"{_MARGIN['l'] + i / (len(ys) - 1) * iw:.1f},"
        f"{_MARGIN['t'] + (1 - (v - lo) / span) * ih:.1f}"
        for i, v in enumerate(ys)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="line chart">'
        f'<line x1="{_MARGIN["l"]}" y1="{_MARGIN["t"]}" x2="{_MARGIN["l"]}" '
        f'y2="{height - _MARGIN["b"]}" class="axis"/>'
        f'<line x1="{_MARGIN["l"]}" y1="{height - _MARGIN["b"]}" x2="{width - _MARGIN["r"]}" '
        f'y2="{height - _MARGIN["b"]}" class="axis"/>'
        f'<text x="{_MARGIN["l"] - 6}" y="{_MARGIN["t"] + 8}" class="tick" text-anchor="end">'
        f'{_esc(y_fmt.format(hi))}</text>'
        f'<text x="{_MARGIN["l"] - 6}" y="{height - _MARGIN["b"]}" class="tick" text-anchor="end">'
        f'{_esc(y_fmt.format(lo))}</text>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>'
        f"</svg>"
    )


def _drawdown(ys: list) -> list:
    """净值序列 → 回撤序列（负值，0 为顶部）。"""
    out, peak = [], None
    for v in ys:
        peak = v if peak is None else max(peak, v)
        out.append(v / peak - 1 if peak else 0.0)
    return out


def _svg_two_lines(series: dict, width: int, height: int,
                   colors: dict, y_fmt: str = "{:.2f}") -> str:
    """多条折线叠一张图。series = {label: [values]}，带图例。"""
    cleaned = {}
    for label, ys in series.items():
        vals = []
        for v in ys or []:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and f > 0:
                vals.append(f)
        if len(vals) >= 2:
            cleaned[label] = vals
    if not cleaned:
        return '<div class="empty">净值曲线数据不足（两个窗口各需 ≥2 个有效点）</div>'
    all_v = [v for vals in cleaned.values() for v in vals]
    lo, hi = min(all_v), max(all_v)
    span = (hi - lo) or 1e-12
    iw = width - _MARGIN["l"] - _MARGIN["r"]
    ih = height - _MARGIN["t"] - _MARGIN["b"]
    legend_y = _MARGIN["t"] + 4
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="net value comparison">']
    parts.append(f'<line x1="{_MARGIN["l"]}" y1="{_MARGIN["t"]}" x2="{_MARGIN["l"]}" y2="{height - _MARGIN["b"]}" class="axis"/>')
    parts.append(f'<line x1="{_MARGIN["l"]}" y1="{height - _MARGIN["b"]}" x2="{width - _MARGIN["r"]}" y2="{height - _MARGIN["b"]}" class="axis"/>')
    parts.append(f'<text x="{_MARGIN["l"] - 6}" y="{_MARGIN["t"] + 8}" class="tick" text-anchor="end">{_esc(y_fmt.format(hi))}</text>')
    parts.append(f'<text x="{_MARGIN["l"] - 6}" y="{height - _MARGIN["b"]}" class="tick" text-anchor="end">{_esc(y_fmt.format(lo))}</text>')
    for k, (label, vals) in enumerate(cleaned.items()):
        color = colors.get(label, "#0969da")
        pts = " ".join(
            f"{_MARGIN['l'] + i / (len(vals) - 1) * iw:.1f},"
            f"{_MARGIN['t'] + (1 - (v - lo) / span) * ih:.1f}"
            for i, v in enumerate(vals)
        )
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        lx = _MARGIN["l"] + 12 + k * 150
        parts.append(f'<rect x="{lx}" y="{legend_y - 8}" width="10" height="3" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{legend_y - 4}" class="tick">{_esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_corr_bars(corr: dict, sota_names: list) -> str:
    """correlations {new: [ic_per_sota]} → 水平柱状图，标出去重闸门线 0.99。"""
    if not corr:
        return '<div class="empty">无相关性数据</div>'
    rows = []
    for new_name, ics in corr.items():
        ic_list = ics if isinstance(ics, list) else [ics]
        for i, ic in enumerate(ic_list):
            sota = sota_names[i] if i < len(sota_names) else f"SOTA-{i}"
            try:
                v = float(ic)
            except (TypeError, ValueError):
                continue
            rows.append((_esc(sota), _esc(new_name), v))
    if not rows:
        return '<div class="empty">相关性数据为空</div>'
    lo = min(0.0, min(r[2] for r in rows))
    hi = max(1.0, max(r[2] for r in rows))  # 0.99 闸门线要看得见
    span = (hi - lo) or 1e-12
    bar_h, gap = 16, 8
    width = 860
    label_w, axis_w = 150, 16
    iw = width - label_w - axis_w
    height = (bar_h + gap) * len(rows) + 24
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="correlation bars">']
    # 0.99 去重闸门竖线
    x99 = label_w + (0.99 - lo) / span * iw
    parts.append(f'<line x1="{x99:.1f}" y1="0" x2="{x99:.1f}" y2="{height - 20}" class="gate"/>')
    parts.append(f'<text x="{x99:.1f}" y="{height - 6}" class="tick" text-anchor="middle">dedup gate 0.99</text>')
    for k, (sota, new, v) in enumerate(rows):
        y = k * (bar_h + gap) + 4
        w = abs(v) / span * iw
        x = label_w + (v - lo) / span * iw - w if v < 0 else label_w + (0 - lo) / span * iw
        cls = "bar bad" if abs(v) >= 0.99 else "bar"
        parts.append(f'<text x="{label_w - 6}" y="{y + bar_h - 4}" class="tick" text-anchor="end">{sota} × {new}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" class="{cls}"/>')
        parts.append(f'<text x="{x + w + 4:.1f}" y="{y + bar_h - 4}" class="tick">{v:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


_CSS = """
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:24px auto;max-width:920px;color:#1c2128;background:#fff}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:28px 0 8px;border-bottom:1px solid #d8dee4;padding-bottom:4px}
.meta{color:#59636e;font-size:12px;margin-bottom:16px}
.banner{padding:10px 14px;border-radius:6px;font-weight:600;margin:12px 0}
.ok{background:#dafbe1;border:1px solid #2da44e}.bad{background:#ffebe9;border:1px solid #cf222e}.warn{background:#fff8c5;border:1px solid #9a6700}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d8dee4;padding:6px 10px;text-align:left}th{background:#f6f8fa}
.axis{stroke:#8c959f;stroke-width:1}.tick{fill:#59636e;font-size:10px}
polyline{} .gate{stroke:#cf222e;stroke-dasharray:4 3;stroke-width:1}
.bar{fill:#0969da}.bar.bad{fill:#cf222e}
.empty{color:#8c959f;font-size:13px;padding:12px 0}
pre{background:#f6f8fa;border:1px solid #d8dee4;border-radius:6px;padding:10px;font-size:11px;overflow-x:auto;white-space:pre-wrap}
.foot{color:#8c959f;font-size:11px;margin-top:32px}
"""


def render_backtest_report(result: dict, meta: dict | None = None) -> str:
    """BacktestResult dict → 自包含 HTML 字符串。永不抛异常。"""
    meta = meta or {}
    result = result or {}
    try:
        return _render(result, meta)
    except Exception as e:  # noqa: BLE001 — 渲染层永不弄挂工具通道
        return (
            "<!doctype html><html><head><meta charset='utf-8'><title>report error</title></head>"
            f"<body><h1>报告渲染失败</h1><pre>{_esc(e)}</pre></body></html>"
        )


def _render(result: dict, meta: dict) -> str:
    ok = bool(result.get("ok"))
    dedup = bool(result.get("dedup_dropped"))
    status_cls = "ok" if ok else ("warn" if dedup else "bad")
    if ok:
        status = "✅ 回测通过"
    elif dedup:
        status = "⚠️ 全部被去重闸门丢弃（RD-Agent FactorEmptyError 语义）"
    else:
        status = "❌ 回测失败"

    xs, ys = _curve_xy(result.get("net_curve"))
    # net_curve 为空时退回 net_values（无日期版本）
    if len(ys) < 2 and result.get("net_values"):
        ys = [float(v) for v in result["net_values"]
              if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
    curve_svg = _svg_line(_downsample(ys), _WIDTH, _HEIGHT, "#0969da")
    dd = _drawdown(ys)
    dd_svg = _svg_line(_downsample(dd), _WIDTH, _DD_HEIGHT, "#cf222e", y_fmt="{:.1%}") if len(dd) >= 2 else ""

    metrics = result.get("metrics") or {}
    m_rows = "".join(
        f"<tr><th>{_esc(label)}</th><td>{_fmt(metrics.get(key))}</td></tr>"
        for key, (label, _) in _METRIC_LABELS.items()
    )
    # 未列在标签表里的指标也带上（qlib 版本差异出现新键时不丢信息）
    known = set(_METRIC_LABELS)
    extra = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_fmt(v)}</td></tr>"
        for k, v in metrics.items() if k not in known
    )

    sota_broken = result.get("sota_broken") or []
    broken_html = ""
    if sota_broken:
        items = "".join(f"<li>{_esc(n)}</li>" for n in sota_broken)
        broken_html = f'<div class="banner warn">SOTA 因子重算失败（已隔离，不阻塞本轮）：<ul>{items}</ul></div>'

    err_html = ""
    if result.get("error"):
        err_html = f'<h2>错误</h2><div class="banner bad">{_esc(result["error"])}</div>'
    tb_html = ""
    if result.get("traceback"):
        tb_html = f"<h2>Traceback</h2><pre>{_esc(result['traceback'])}</pre>"

    trades = result.get("trades") or []
    if trades:
        t_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                _esc(t.get("symbol", "")), _esc(t.get("date", "")), _esc(t.get("action", "")))
            for t in trades
        )
        trades_html = f"<table><tr><th>标的</th><th>日期</th><th>动作</th></tr>{t_rows}</table>"
    else:
        trades_html = '<div class="empty">无买卖点数据（当前回测口径下 trades 未采集）</div>'

    new_names = [f["name"] for f in meta.get("new_factors", []) if isinstance(f, dict)]
    sota_names = [f["name"] for f in meta.get("sota", []) if isinstance(f, dict)]
    corr_svg = _svg_corr_bars(result.get("correlations") or {}, sota_names)
    names_line = " / ".join(_esc(n) for n in new_names) or "—"
    sota_line = ", ".join(_esc(n) for n in sota_names) or "—"

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>回测报告 · {_esc(names_line)}</title>
<style>{_CSS}</style></head><body>
<h1>因子回测报告</h1>
<div class="meta">
新因子：<b>{names_line}</b> ｜ SOTA 基线：{sota_line} ｜ profile：{_esc(meta.get("profile", "—"))}
｜ 数据版本：{_esc(meta.get("data_version", "—"))} ｜ 生成时间：{_esc(time.strftime("%Y-%m-%d %H:%M:%S"))}
</div>
<div class="banner {status_cls}">{status}</div>
{broken_html}
<h2>组合指标（SOTA + 新因子，扣费口径）</h2>
<table>{m_rows}{extra}</table>
<h2>净值曲线</h2>
{curve_svg}
<h2>回撤</h2>
{dd_svg or '<div class="empty">净值曲线不足，无法计算回撤</div>'}
<h2>新因子 × SOTA 日均 IC（去重闸门：|IC| ≥ 0.99 丢弃）</h2>
{corr_svg}
<h2>买卖点</h2>
{trades_html}
{err_html}
{tb_html}
<div class="foot">factor-miner-mcp · report.py（自包含单文件，无外部依赖）</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════
# OOS 准入报告：挖掘窗口 vs 纯样本外窗口
# ═══════════════════════════════════════════════════════════════

_OOS_METRIC_KEYS = {"ic": "IC", "annualized_return": "年化超额（扣费）",
                    "max_drawdown": "最大回撤（扣费）"}


def render_oos_report(result: dict, meta: dict | None = None) -> str:
    """factor_oos_check 返回 dict → 自包含 HTML。永不抛异常。"""
    meta = meta or {}
    result = result or {}
    try:
        return _render_oos(result, meta)
    except Exception as e:  # noqa: BLE001 — 渲染层永不弄挂工具通道
        return (
            "<!doctype html><html><head><meta charset='utf-8'><title>report error</title></head>"
            f"<body><h1>报告渲染失败</h1><pre>{_esc(e)}</pre></body></html>"
        )


def _render_oos(result: dict, meta: dict) -> str:
    ok = bool(result.get("ok"))
    decay = result.get("decay")
    try:
        decay_f = float(decay) if decay is not None else None
        if not math.isfinite(decay_f):
            decay_f = None
    except (TypeError, ValueError):
        decay_f = None

    # 衰减分级：生产准入的可视化读法（阈值与 Go 侧卡点保持一致的量级观感）
    if not ok:
        status_cls, status = "bad", "❌ OOS 检验失败（准入拒绝）"
    elif decay_f is None:
        status_cls, status = "warn", "⚠️ 衰减不可计算（mining IC 为 0 → 不可证伪，拒绝）"
    elif decay_f <= 0.3:
        status_cls, status = "ok", f"✅ 通过 · 相对衰减 {decay_f:.1%}（≤30%）"
    elif decay_f <= 0.6:
        status_cls, status = "warn", f"⚠️ 衰减偏大 {decay_f:.1%}（30%~60%，人工复核）"
    else:
        status_cls, status = "bad", f"❌ 严重衰减 {decay_f:.1%}（>60%，拒绝）"

    def _ys(key):
        return [p.get("value") for p in (result.get(key) or [])]

    compare_svg = _svg_two_lines(
        {"挖掘窗口": _downsample(_ys("mining_net_curve")),
         "样本外窗口": _downsample(_ys("oos_net_curve"))},
        _WIDTH, _HEIGHT, {"挖掘窗口": "#0969da", "样本外窗口": "#cf222e"})

    mining = result.get("mining") or {}
    oos = result.get("oos") or {}
    m_rows = "".join(
        f"<tr><th>{label}</th><td>{_fmt(mining.get(k))}</td><td>{_fmt(oos.get(k))}</td></tr>"
        for k, label in _OOS_METRIC_KEYS.items()
    )

    err_html = ""
    if result.get("error"):
        err_html = f'<h2>错误</h2><div class="banner bad">{_esc(result["error"])}</div>'
    tb_html = ""
    if result.get("traceback"):
        tb_html = f"<h2>Traceback</h2><pre>{_esc(result['traceback'])}</pre>"

    name = _esc(meta.get("name", "—"))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OOS 准入报告 · {name}</title>
<style>{_CSS}</style></head><body>
<h1>因子 OOS 准入报告</h1>
<div class="meta">
因子：<b>{name}</b> ｜ 挖掘窗口 test：{_esc(meta.get("mining_test_start", "—"))} 起 ｜
样本外 test：{_esc(meta.get("oos_test_start", "—"))} ｜ 生成时间：{_esc(time.strftime("%Y-%m-%d %H:%M:%S"))}
</div>
<div class="banner {status_cls}">{status}</div>
<h2>指标对比（纯样本外 vs 挖掘窗口）</h2>
<table><tr><th></th><th>挖掘窗口</th><th>样本外窗口</th></tr>{m_rows}</table>
<h2>净值曲线对比（decay 的可视化）</h2>
{compare_svg}
{err_html}
{tb_html}
<div class="foot">factor-miner-mcp · report.py（自包含单文件，无外部依赖）</div>
</body></html>"""


def _safe_slug(slug: str) -> str:
    """文件名消毒：只留 [A-Za-z0-9._-]，其余换 "_"（防路径注入）。"""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(slug))[:120]


def save_report(html_str: str, reports_dir, slug: str) -> str:
    """报告落盘 → 返回文件名（不含目录）。reports_dir 不存在时抛 IOError，
    由调用方决定降级（调用方契约：未配置 FACTOR_REPORT_DIR 就不落盘）。"""
    import pathlib  # noqa: PLC0415 — 顶层已可用，此处显式无妨

    d = pathlib.Path(reports_dir)
    if not d.is_dir():
        raise IOError(f"reports dir not found: {d}")
    fname = _safe_slug(slug)
    if not fname.endswith(".html"):
        fname += ".html"
    (d / fname).write_text(html_str, encoding="utf-8")
    return fname
