"""analytics.py — 因子评估与组合分析五件套（纯 numpy/pandas/scipy 内置实现）。

工具:
    factor_tearsheet   Alphalens 式因子完整评估（手写 pandas，不用 alphalens 本体——
                       alphalens 已半停维护，且其依赖链与 pandas>=2 冲突频发；
                       分位数收益/IC/换手率逻辑本身很短，手写可控可测）
    portfolio_optimize HRP / 等权 / 最小方差（+ pypfopt 可选 mean_variance）
    regime_detect      规则版牛/熊/震荡状态机（+ hmmlearn 可选 GaussianHMM）
    change_point       CUSUM + 二分分割（+ ruptures 可选 PELT）
    vol_forecast       EWMA/RiskMetrics（+ arch 可选 GARCH(1,1)）

设计铁律：纯 numpy/pandas/scipy 路径开箱可用；重库全部惰性导入做可选增强，
输出 method 字段标注实际实现。

算法出处:
    HRP    — López de Prado (2016), "Building Diversified Portfolios that
             Outperform Out-of-Sample"
    EWMA   — RiskMetrics (1996), J.P. Morgan Technical Document, lambda=0.94
    PELT   — Killick et al. (2012), "Optimal Detection of Changepoints With
             a Linear Computational Cost", JASA
    CUSUM  — Page (1954), "Continuous Inspection Schemes", Biometrika
    HMM    — Rabiner (1989), "A Tutorial on Hidden Markov Models"
    Parkinson — Parkinson (1980), 高低价极值波动率估计量
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════
# 公共工具
# ═══════════════════════════════════════════════════════════════

def _klines_to_close_frame(klines: dict) -> pd.DataFrame:
    """{symbol: [{date, close}, ...]} → DataFrame(index=date, columns=symbol)。"""
    frames = {}
    for symbol, rows in (klines or {}).items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "date" not in df or "close" not in df:
            continue
        frames[symbol] = df.set_index("date")["close"].astype(float)
    if not frames:
        return pd.DataFrame()
    px = pd.DataFrame(frames).sort_index()
    return px.ffill()


def _klines_to_series(klines: list, key: str = "close") -> pd.Series:
    df = pd.DataFrame(klines or [])
    if df.empty or "date" not in df or key not in df:
        return pd.Series(dtype=float)
    return df.set_index("date")[key].astype(float).sort_index()


def _nan_safe(obj):
    """JSON 序列化前清理 NaN/inf 与 numpy 标量。"""
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


# ═══════════════════════════════════════════════════════════════
# 1. factor_tearsheet — Alphalens 式因子评估（手写 pandas）
# ═══════════════════════════════════════════════════════════════

def factor_tearsheet(factor_values: list, klines: dict,
                     quantiles: int = 5, periods: list = None) -> str:
    periods = [int(p) for p in (periods or [1, 5, 10])]
    quantiles = max(2, int(quantiles))

    fdf = pd.DataFrame(factor_values or [])
    if fdf.empty or not {"date", "symbol", "value"} <= set(fdf.columns):
        return json.dumps({"error": "factor_values 需含 date/symbol/value"})
    fac = fdf.pivot_table(index="date", columns="symbol", values="value").sort_index()

    px = _klines_to_close_frame(klines)
    if px.empty:
        return json.dumps({"error": "klines 为空或缺 date/close"})
    # 对齐：只保留因子与价格都有的日期/标的
    idx = fac.index.intersection(px.index)
    cols = fac.columns.intersection(px.columns)
    fac, px = fac.loc[idx, cols], px.loc[idx, cols]
    if len(idx) < max(periods) + 2:
        return json.dumps({"error": "对齐后样本不足"})

    quantile_returns = {}
    long_short = {}
    ic_decay = {}
    for p in periods:
        fwd = px.shift(-p) / px - 1.0  # forward p 日收益
        valid_rows = fwd.index[: len(fwd) - p]  # 尾部 p 行无未来数据
        qret = {q: [] for q in range(1, quantiles + 1)}
        for dt in valid_rows:
            f = fac.loc[dt]
            r = fwd.loc[dt]
            mask = f.notna() & r.notna()
            if mask.sum() < quantiles * 2:
                continue
            labels = pd.qcut(f[mask].rank(method="first"), quantiles,
                             labels=False) + 1
            grp = r[mask].groupby(labels).mean()
            for q, v in grp.items():
                qret[int(q)].append(v)
        q_mean = {f"q{q}": (float(np.mean(v)) if v else None)
                  for q, v in qret.items()}
        quantile_returns[f"{p}d"] = {
            "mean_period_return": q_mean,
            "annualized": {k: (None if v is None else v * TRADING_DAYS / p)
                           for k, v in q_mean.items()},
        }
        top, bot = q_mean[f"q{quantiles}"], q_mean["q1"]
        ls = None if (top is None or bot is None) else top - bot
        long_short[f"{p}d"] = {
            "mean_period_spread": ls,
            "annualized": None if ls is None else ls * TRADING_DAYS / p,
        }

        # 逐日截面 Spearman IC（forward p 日收益）
        ics = []
        for dt in valid_rows:
            f, r = fac.loc[dt], fwd.loc[dt]
            mask = f.notna() & r.notna()
            if mask.sum() >= 5:
                ic = stats.spearmanr(f[mask], r[mask]).statistic
                if np.isfinite(ic):
                    ics.append(float(ic))
        ic_decay[f"{p}d"] = float(np.mean(ics)) if ics else None
        if p == periods[0]:
            ic_arr = np.array(ics)
            ic_std = float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else None
            ic_out = {
                "period": f"{p}d",
                "mean": float(ic_arr.mean()) if len(ic_arr) else None,
                "std": ic_std,
                "ir": (float(ic_arr.mean() / ic_std)
                       if ic_std and ic_std > 0 else None),
                "series_summary": {
                    "days": int(len(ic_arr)),
                    "positive_ratio": (float((ic_arr > 0).mean())
                                       if len(ic_arr) else None),
                },
                "decay": ic_decay,  # 填完后统一（下方覆盖）
            }

    ic_out["decay"] = ic_decay

    # 换手率：最高/最低分位组合逐日成分变动比例均值
    turnover = {}
    for label, pick in (("top", quantiles), ("bottom", 1)):
        changes = []
        prev = None
        for dt in fac.index[: len(fac) - min(periods)]:
            f = fac.loc[dt].dropna()
            if len(f) < quantiles * 2:
                continue
            q = pd.qcut(f.rank(method="first"), quantiles, labels=False) + 1
            members = set(q[q == pick].index)
            if prev is not None and members:
                changes.append(1 - len(members & prev) / len(members))
            prev = members
        turnover[label] = float(np.mean(changes)) if changes else None

    return json.dumps(_nan_safe({
        "quantile_returns": quantile_returns,
        "long_short": long_short,
        "ic": ic_out,
        "turnover": turnover,
        "n_days": int(len(idx)), "n_symbols": int(len(cols)),
        "method": "pandas",
    }), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 2. portfolio_optimize — HRP / 等权 / 最小方差 / (可选) mean_variance
# ═══════════════════════════════════════════════════════════════

def _hrp_weights(cov: pd.DataFrame) -> pd.Series:
    """HRP（López de Prado 2016）：相关距离 → 层次聚类 → 拟对角化 → 递归二分。"""
    corr = cov.corr().fillna(0.0).clip(-1, 1)
    dist = np.sqrt(0.5 * (1 - corr))  # 相关距离矩阵
    np.fill_diagonal(dist.values, 0.0)
    link = linkage(squareform(dist.values, checks=False), method="single")
    order = leaves_list(link)  # 拟对角化排序
    sorted_cov = cov.iloc[order, order]

    def _cluster_var(c: pd.DataFrame) -> float:
        ivp = 1.0 / np.diag(c.values)
        ivp /= ivp.sum()
        return float(ivp @ c.values @ ivp)

    weights = pd.Series(1.0, index=sorted_cov.index)
    clusters = [list(sorted_cov.index)]
    while clusters:  # 递归二分
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            lv = _cluster_var(sorted_cov.loc[left, left])
            rv = _cluster_var(sorted_cov.loc[right, right])
            alpha = 1 - lv / (lv + rv) if (lv + rv) > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= (1 - alpha)
            nxt.extend([left, right])
        clusters = nxt
    return weights.sort_index()


def portfolio_optimize(symbols: list, klines: dict, method: str = "hrp",
                       lookback: int = 120) -> str:
    method = (method or "hrp").lower()
    px = _klines_to_close_frame({s: (klines or {}).get(s, []) for s in symbols})
    if px.empty:
        px = _klines_to_close_frame(klines)
    px = px.dropna(axis=1, how="all").tail(int(lookback) + 1)
    rets = px.pct_change().dropna(how="any")
    if rets.shape[0] < 20 or rets.shape[1] < 2:
        return json.dumps({"error": "有效收益样本不足（需 >=2 标的、>=20 日）"})

    mu = rets.mean() * TRADING_DAYS           # 年化期望收益
    cov = rets.cov() * TRADING_DAYS           # 年化协方差
    symbols = list(rets.columns)

    if method == "equal":
        w = pd.Series(1.0 / len(symbols), index=symbols)
        used = "equal"
    elif method == "min_variance":
        c = cov.values
        inv = np.linalg.pinv(c)
        ones = np.ones(len(symbols))
        denom = float(ones @ inv @ ones)
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            # 协方差奇异（如完全负相关）→ 逆波动率加权
            ivol = 1.0 / np.sqrt(np.diag(c))
            raw = ivol / ivol.sum()
        else:
            raw = inv @ ones / denom  # 解析解（允许做空）
        raw = np.clip(raw, 0, None)             # 收敛为 long-only
        raw = raw / raw.sum() if raw.sum() > 0 else np.full(len(symbols), 1 / len(symbols))
        w = pd.Series(raw, index=symbols)
        used = "min_variance"
    elif method == "mean_variance":
        try:
            from pypfopt import EfficientFrontier  # 惰性导入可选增强
            ef = EfficientFrontier(mu, cov)
            ef.max_sharpe(risk_free_rate=0.0)
            w = pd.Series(ef.clean_weights())
            w = w.reindex(symbols).fillna(0.0)
            used = "mean_variance(pypfopt)"
        except ImportError:
            return json.dumps({"error": "mean_variance 需安装 PyPortfolioOpt；"
                                        "可改用 hrp/equal/min_variance"})
    else:  # 默认 hrp
        w = _hrp_weights(cov).reindex(symbols).fillna(0.0)
        used = "hrp"

    w = w / w.sum()
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
    return json.dumps(_nan_safe({
        "weights": {s: float(w[s]) for s in symbols},
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe": (port_ret / port_vol) if port_vol > 0 else None,
        "lookback_days": int(rets.shape[0]),
        "method": used,
    }), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 3. regime_detect — 牛/熊/震荡识别（规则状态机 / 可选 HMM）
# ═══════════════════════════════════════════════════════════════

def _label_states(order_stats: dict, n_regimes: int) -> dict:
    """按状态平均收益排序映射标签：最高=bull，最低=bear，其余=range。"""
    ranked = sorted(order_stats, key=lambda s: order_stats[s]["mean_ret"])
    labels = {}
    for i, st in enumerate(ranked):
        if i == 0:
            labels[st] = "bear" if n_regimes >= 2 else "range"
        elif i == len(ranked) - 1:
            labels[st] = "bull"
        else:
            labels[st] = "range"
    return labels


def regime_detect(klines: list, n_regimes: int = 3) -> str:
    n_regimes = max(2, min(3, int(n_regimes)))
    close = _klines_to_series(klines, "close")
    if len(close) < 40:
        return json.dumps({"error": "K线不足（需 >=40 根）"})
    ret = close.pct_change()
    win = 20

    method = "rule"
    states = None
    if n_regimes >= 2:
        try:
            from hmmlearn.hmm import GaussianHMM  # 惰性导入可选增强
            feat = pd.DataFrame({
                "ret": ret,
                "vol": ret.rolling(5).std(),
            }).dropna()
            if len(feat) >= 40:
                X = (feat - feat.mean()) / feat.std()
                hmm = GaussianHMM(n_components=n_regimes,
                                  covariance_type="full",
                                  n_iter=200, random_state=42)
                hmm.fit(X.values)
                states = pd.Series(hmm.predict(X.values), index=feat.index)
                method = "hmm"
        except ImportError:
            pass
        except Exception:  # HMM 不收敛等 → 回退规则版
            states = None
            method = "rule"

    if states is None:
        # 规则版：20 日动量 + 已实现波动率阈值状态机
        # （波动阈值取全样本中位数：回看窗口分析允许用全样本分布定标）
        mom = close / close.shift(win) - 1
        vol = ret.rolling(win).std() * np.sqrt(TRADING_DAYS)
        vol_hi = vol > vol.median()
        raw = pd.Series("range", index=close.index)
        raw[(mom > 0) & ~vol_hi] = "bull"
        raw[(mom < 0)] = "bear" if n_regimes == 2 else "bear"
        if n_regimes == 3:
            raw[(mom < 0) & ~vol_hi] = "range"  # 低波动阴跌归为震荡
        raw[mom.isna()] = None
        states = raw.dropna()
        labeled = states
    else:
        stats_map = {}
        for st in states.unique():
            r = ret.reindex(states.index)[states == st]
            stats_map[st] = {"mean_ret": float(r.mean()) if len(r) else 0.0}
        labels = _label_states(stats_map, n_regimes)
        labeled = states.map(labels)

    regime_stats = {}
    for rg in sorted(set(labeled)):
        r = ret.reindex(labeled.index)[labeled == rg].dropna()
        regime_stats[rg] = {
            "mean_ret": float(r.mean()) if len(r) else None,
            "vol": float(r.std() * np.sqrt(TRADING_DAYS)) if len(r) > 1 else None,
            "days": int(len(r)),
        }

    history = [{"date": str(dt), "regime": rg} for dt, rg in labeled.items()]
    return json.dumps(_nan_safe({
        "current_regime": labeled.iloc[-1],
        "regime_history": history,
        "regime_stats": regime_stats,
        "method": method,
    }), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 4. change_point — CUSUM + 二分分割 / (可选) PELT
# ═══════════════════════════════════════════════════════════════

def _cusum_binseg(x: np.ndarray, max_bkps: int, min_len: int = 10):
    """均值漂移 CUSUM（Page 1954）+ 二分分割，返回 [(index, significance)]。

    原始序列扫描水平突变（level shift），一阶差分扫描漂移/斜率突变
    （trend break）——趋势序列的水平扫描会被段内趋势干扰，差分后均值
    漂移清晰。两路候选按显著度排序、按最小间距去重后截断。
    """
    def _scan(x, lo, hi, out):
        if hi - lo < 2 * min_len:
            return
        seg = x[lo:hi]
        cusum = np.cumsum(seg - seg.mean())
        i = int(np.argmax(np.abs(cusum)))
        # 显著性：分段均值差相对合并标准误的 t 统计量
        a, b = seg[: i + 1], seg[i + 1:]
        if len(a) < min_len or len(b) < min_len:
            return
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = abs(a.mean() - b.mean()) / se if se > 0 else 0.0
        if t < 2.0:  # 不显著则停止细分
            return
        out.append((lo + i + 1, float(t)))
        _scan(x, lo, lo + i + 1, out)
        _scan(x, lo + i + 1, hi, out)

    raw_cands, diff_cands = [], []
    _scan(x, 0, len(x), raw_cands)
    if len(x) > 2 * min_len:
        _scan(np.diff(x), 0, len(x) - 1, diff_cands)  # 漂移突变（diff 索引≈原索引）
    # 两路检测器轮流取最强候选（raw 与 diff 的 t 尺度不可比，避免高波动
    # 段内的水平候选把漂移突变挤出榜单），按最小间距去重后截断
    raw_cands.sort(key=lambda s: -s[1])
    diff_cands.sort(key=lambda s: -s[1])
    picked = []
    pools = [raw_cands, diff_cands]
    while len(picked) < max_bkps and any(pools):
        for pool in pools:
            if not pool:
                continue
            idx, t = pool.pop(0)
            if all(abs(idx - p[0]) >= min_len for p in picked):
                picked.append((idx, t))
            if len(picked) >= max_bkps:
                break
    picked.sort()
    return picked


def change_point(series: list, method: str = "auto", max_bkps: int = 5) -> str:
    df = pd.DataFrame(series or [])
    if df.empty or not {"date", "value"} <= set(df.columns):
        return json.dumps({"error": "series 需含 date/value"})
    df = df.sort_values("date").reset_index(drop=True)
    x = df["value"].astype(float).values
    if len(x) < 30:
        return json.dumps({"error": "序列太短（需 >=30 点）"})
    max_bkps = max(1, int(max_bkps))
    method = (method or "auto").lower()

    points = []
    used = "cusum_binseg"
    if method in ("auto", "pelt"):
        try:
            import ruptures as rpt  # 惰性导入可选增强
            bkps = rpt.Pelt(model="l2").fit(x).predict(pen=np.log(len(x)) * 2)
            bkps = [b for b in bkps if b < len(x)]
            # 超限则保留均值漂移最显著的前 max_bkps 个
            scored = []
            for b in bkps:
                lo = scored[-1][0] if scored else 0
                a, b_seg = x[lo:b], x[b:]
                if len(a) and len(b_seg):
                    se = np.sqrt(a.var() / len(a) + b_seg.var() / len(b_seg))
                    t = abs(a.mean() - b_seg.mean()) / se if se > 0 else 0.0
                    scored.append((b, float(t)))
            scored.sort(key=lambda s: -s[1])
            points = sorted(scored[:max_bkps])
            used = "pelt"
        except ImportError:
            if method == "pelt":
                return json.dumps({"error": "pelt 需安装 ruptures；或用 method=auto"})
            points = _cusum_binseg(x, max_bkps)
            used = "cusum_binseg"
    if not points and method in ("cusum_binseg", "cusum"):
        points = _cusum_binseg(x, max_bkps)
        used = "cusum_binseg"

    return json.dumps(_nan_safe({
        "change_points": [{"date": str(df["date"].iloc[i]), "index": int(i),
                           "significance": sig} for i, sig in points],
        "method": used,
    }), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# 5. vol_forecast — EWMA(RiskMetrics) / (可选) GARCH(1,1)
# ═══════════════════════════════════════════════════════════════

def vol_forecast(klines: list, horizon: int = 5, method: str = "auto") -> str:
    horizon = max(1, int(horizon))
    close = _klines_to_series(klines, "close")
    if len(close) < 30:
        return json.dumps({"error": "K线不足（需 >=30 根）"})
    ret = close.pct_change().dropna()
    method = (method or "auto").lower()

    forecast, used = None, "ewma"
    if method in ("auto", "garch"):
        try:
            from arch import arch_model  # 惰性导入可选增强
            am = arch_model(ret * 100, vol="Garch", p=1, q=1, rescale=False)
            fit = am.fit(disp="off", show_warning=False)
            f = fit.forecast(horizon=horizon, reindex=False)
            var = f.variance.values[-1] / 1e4  # %² → 小数²
            forecast = [float(np.sqrt(v)) for v in var]
            used = "garch"
        except ImportError:
            if method == "garch":
                return json.dumps({"error": "garch 需安装 arch；或用 method=auto"})
        except Exception:
            forecast, used = None, "ewma"  # 拟合失败回退 EWMA

    if forecast is None:
        lam = 0.94  # RiskMetrics (1996)
        var = float((ret ** 2).iloc[:20].mean())
        for r in ret.values:
            var = lam * var + (1 - lam) * r * r
        # EWMA 多期预测为平坦外推：E[σ²_{t+h}] = σ²_{t+1}
        forecast = [float(np.sqrt(var))] * horizon
        used = "ewma"

    out = {
        "forecast": [{"day": i + 1, "vol": v} for i, v in enumerate(forecast)],
        "current_vol": forecast[0],
        "method": used,
    }
    # 输入含 high/low 时附 Parkinson (1980) 当前波动率参考
    df = pd.DataFrame(klines)
    if {"high", "low"} <= set(df.columns):
        hl = np.log(df["high"].astype(float) / df["low"].astype(float)) ** 2
        out["parkinson_vol"] = float(np.sqrt(hl.tail(20).mean()
                                             / (4 * np.log(2))
                                             * TRADING_DAYS))
    return json.dumps(_nan_safe(out), ensure_ascii=False)
