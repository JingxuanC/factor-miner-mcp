"""
FactorEngine — Phase 1: Alpha158-style factor computation.

Computes 50+ factors from OHLCV data across 6 categories:
  Trend, Volatility, Volume, Price, Momentum, Overnight.

Real Alpha158 (158 factors) → Phase 4 with Polars acceleration.
"""

import math
from typing import Optional


class FactorEngine:
    """Computes Alpha158-style factors from OHLCV time series."""

    def compute(self, data: dict) -> dict:
        symbol = data["symbol"]
        klines = data["klines"]
        if not klines:
            return {"symbol": symbol, "factors": {}}

        closes = [k["close"] for k in klines]
        opens  = [k["open"] for k in klines]
        highs  = [k["high"] for k in klines]
        lows   = [k["low"] for k in klines]
        vols   = [k["volume"] for k in klines]
        n = len(closes)
        close = closes[-1]
        vol = vols[-1]

        factors = {}

        # ═══════════════════════════════════════════
        # 1. TREND (14 factors)
        # ═══════════════════════════════════════════

        # Moving averages
        for period in (5, 10, 20, 30, 60):
            ma = sma(closes, period)
            factors[f"ma_{period}"] = ma or close
            if ma:
                factors[f"close_over_ma_{period}"] = close / ma

        # Exponential moving averages
        for period in (5, 12, 26):
            factors[f"ema_{period}"] = ema(closes, period) or close

        # MACD (12, 26, 9)
        ema12 = factors.get("ema_12", close)
        ema26 = factors.get("ema_26", close)
        dif = ema12 - ema26
        factors["macd_dif"] = dif
        factors["macd_signal"] = ema(_collect_macd(closes, "dif"), 9) or dif
        factors["macd_hist"] = dif - factors["macd_signal"]

        # Rate of change
        for period in (5, 10, 20):
            factors[f"roc_{period}"] = roc(closes, period)

        # RSI (14)
        factors["rsi_14"] = rsi(closes, 14)

        # ADX (14)
        factors["adx_14"] = adx(highs, lows, closes, 14)

        # CCI (20)
        factors["cci_20"] = cci(highs, lows, closes, 20)

        # ═══════════════════════════════════════════
        # 2. VOLATILITY (8 factors)
        # ═══════════════════════════════════════════

        # Standard deviation of returns
        for period in (5, 10, 20):
            factors[f"std_{period}"] = rolling_std(returns(closes), period)

        # Average True Range
        factors["atr_14"] = atr(highs, lows, closes, 14)

        # Bollinger Bands
        ma20 = factors.get("ma_20", close)
        std20 = factors.get("std_20", 0)
        factors["bb_upper"] = ma20 + 2 * std20
        factors["bb_lower"] = ma20 - 2 * std20
        factors["bb_width"] = (factors["bb_upper"] - factors["bb_lower"]) / (ma20 + 1e-10)
        factors["bb_position"] = (close - factors["bb_lower"]) / (factors["bb_upper"] - factors["bb_lower"] + 1e-10)

        # High-Low ratio
        factors["hl_ratio"] = (highs[-1] - lows[-1]) / (lows[-1] + 1e-10)

        # ═══════════════════════════════════════════
        # 3. VOLUME (10 factors)
        # ═══════════════════════════════════════════

        for period in (5, 10, 20):
            ma_vol = sma(vols, period)
            factors[f"volume_ma_{period}"] = ma_vol or vol
            if ma_vol:
                factors[f"volume_ratio_{period}"] = vol / (ma_vol + 1e-10)

        # VWAP (simple approximation)
        vwap_num = 0.0
        vwap_den = 0.0
        for i in range(n):
            tp = (highs[i] + lows[i] + closes[i]) / 3
            vwap_num += tp * vols[i]
            vwap_den += vols[i]
        factors["vwap"] = vwap_num / (vwap_den + 1e-10) if vwap_den > 0 else close

        # OBV (On-Balance Volume) — last 5 days trend
        obv_vals = obv(closes, vols)
        factors["obv"] = obv_vals[-1] if obv_vals else 0
        if len(obv_vals) >= 5:
            factors["obv_trend"] = (obv_vals[-1] - obv_vals[-5]) / (abs(obv_vals[-5]) + 1e-10)

        # MFI (14)
        factors["mfi_14"] = mfi(highs, lows, closes, vols, 14)

        # Volume Price Trend
        factors["vpt"] = _vpt_last(closes, vols)

        # ═══════════════════════════════════════════
        # 4. PRICE (8 factors)
        # ═══════════════════════════════════════════

        factors["typical_price"] = (highs[-1] + lows[-1] + close) / 3
        factors["weighted_close"] = (highs[-1] + lows[-1] + 2 * close) / 4
        factors["open_close_ratio"] = close / (opens[-1] + 1e-10)
        factors["high_close_ratio"] = close / (highs[-1] + 1e-10)

        # Price position within day
        day_range = highs[-1] - lows[-1]
        if day_range > 0:
            factors["price_position"] = (close - lows[-1]) / day_range
        else:
            factors["price_position"] = 0.5

        # Gap from yesterday
        if n >= 2:
            factors["gap"] = opens[-1] / (closes[-2] + 1e-10) - 1

        # Upper/lower shadow ratios
        body = abs(close - opens[-1])
        upper_shadow = highs[-1] - max(close, opens[-1])
        lower_shadow = min(close, opens[-1]) - lows[-1]
        factors["upper_shadow_pct"] = upper_shadow / (day_range + 1e-10)
        factors["lower_shadow_pct"] = lower_shadow / (day_range + 1e-10)

        # ═══════════════════════════════════════════
        # 5. MOMENTUM (8 factors)
        # ═══════════════════════════════════════════

        # Price vs MA distance (normalized)
        for period in (5, 10, 20):
            ma = factors.get(f"ma_{period}", close)
            factors[f"price_vs_ma_{period}_pct"] = (close - ma) / (ma + 1e-10)

        # Consecutive up/down days
        up_days = 0
        down_days = 0
        for i in range(n - 2, max(n - 12, -1), -1):
            if closes[i + 1] > closes[i]:
                up_days += 1
                down_days = 0
            else:
                down_days += 1
                up_days = 0
        factors["consecutive_up"] = up_days
        factors["consecutive_down"] = down_days

        # New high/low within N days
        factors["new_high_20d"] = 1.0 if close >= max(highs[-20:]) else 0.0
        factors["new_low_20d"] = 1.0 if close <= min(lows[-20:]) else 0.0

        # Williams %R (14)
        factors["willr_14"] = willr(highs, lows, closes, 14)

        # ═══════════════════════════════════════════
        # 6. OVERNIGHT / AUCTION (5 factors)
        # ═══════════════════════════════════════════

        # Overnight return
        if n >= 2:
            factors["overnight_return"] = opens[-1] / (closes[-2] + 1e-10) - 1

        # Intraday return
        factors["intraday_return"] = close / (opens[-1] + 1e-10) - 1

        # Intraday amplitude
        factors["intraday_amplitude"] = day_range / (opens[-1] + 1e-10)

        # Turnover (volume / float shares proxy — use volume directly)
        factors["turnover_proxy"] = vol

        # Money flow volume
        tp = factors["typical_price"]
        prev_tp = (highs[-2] + lows[-2] + closes[-2]) / 3 if n >= 2 else tp
        if tp > prev_tp:
            factors["mfv"] = vol * tp
            factors["mfv_neg"] = 0.0
        else:
            factors["mfv"] = 0.0
            factors["mfv_neg"] = vol * tp

        return {"symbol": symbol, "factors": factors}


# ═══════════════════════════════════════════════
# Factor computation helpers
# ═══════════════════════════════════════════════

def sma(series, period):
    if len(series) < period:
        return None
    return sum(series[-period:]) / period

def ema(series, period):
    if len(series) < 2:
        return None
    k = 2 / (period + 1)
    result = series[0]
    for val in series[1:]:
        result = val * k + result * (1 - k)
    return result

def returns(closes):
    rets = []
    for i in range(1, len(closes)):
        rets.append(closes[i] / closes[i-1] - 1)
    return rets

def rolling_std(series, period):
    if len(series) < period:
        return 0.0
    window = series[-period:]
    mean = sum(window) / period
    var = sum((x - mean) ** 2 for x in window) / period
    return math.sqrt(var)

def roc(closes, period):
    if len(closes) <= period:
        return 0.0
    return (closes[-1] - closes[-(period+1)]) / closes[-(period+1)]

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = closes[i+1] - closes[i]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def adx(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0.0
    tr_sum, plus_dm_sum, minus_dm_sum = 0.0, 0.0, 0.0
    for i in range(-period, 0):
        tr = max(highs[i+1] - lows[i+1],
                 abs(highs[i+1] - closes[i]),
                 abs(lows[i+1] - closes[i]))
        tr_sum += tr
        up = highs[i+1] - highs[i]
        down = lows[i] - lows[i+1]
        if up > down and up > 0:
            plus_dm_sum += up
        if down > up and down > 0:
            minus_dm_sum += down
    plus_di = (plus_dm_sum / (tr_sum + 1e-10)) * 100
    minus_di = (minus_dm_sum / (tr_sum + 1e-10)) * 100
    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    return dx

def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(-period + 1, 1):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return sum(trs) / len(trs)

def cci(highs, lows, closes, period=20):
    if len(closes) < period:
        return 0.0
    tps = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(-period, 0)]
    tp_avg = sum(tps) / period
    md = sum(abs(tp - tp_avg) for tp in tps) / period
    if md == 0:
        return 0.0
    return (tps[-1] - tp_avg) / (0.015 * md)

def obv(closes, vols):
    obv_vals = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv_vals.append(obv_vals[-1] + vols[i])
        elif closes[i] < closes[i-1]:
            obv_vals.append(obv_vals[-1] - vols[i])
        else:
            obv_vals.append(obv_vals[-1])
    return obv_vals

def mfi(highs, lows, closes, vols, period=14):
    if len(closes) < period + 1:
        return 50.0
    pos_flow, neg_flow = 0.0, 0.0
    for i in range(-period, 0):
        tp = (highs[i+1] + lows[i+1] + closes[i+1]) / 3
        prev_tp = (highs[i] + lows[i] + closes[i]) / 3
        mf = tp * vols[i+1]
        if tp > prev_tp:
            pos_flow += mf
        else:
            neg_flow += mf
    if neg_flow == 0:
        return 100.0
    mr = pos_flow / neg_flow
    return 100 - (100 / (1 + mr))

def willr(highs, lows, closes, period=14):
    if len(closes) < period:
        return -50.0
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    if highest == lowest:
        return -50.0
    return (highest - closes[-1]) / (highest - lowest) * -100

def _vpt_last(closes, vols):
    if len(closes) < 2:
        return 0.0
    vpt = 0.0
    for i in range(1, len(closes)):
        pct = (closes[i] - closes[i-1]) / (closes[i-1] + 1e-10)
        vpt += pct * vols[i]
    return vpt

def _collect_macd(closes, field):
    """Build MACD DIF series for signal line computation."""
    vals = []
    for i in range(len(closes)):
        window = closes[:i+1]
        e12 = ema(window, 12)
        e26 = ema(window, 26)
        if e12 and e26:
            vals.append(e12 - e26)
        else:
            vals.append(0.0)
    return vals


# ═══════════════════════════════════════════════
# Intraday FactorEngine — minute-bar factors for Monitor
# ═══════════════════════════════════════════════

class IntradayFactorEngine:
    """Computes factors from minute-level (price, volume) bars.
    No OHLC needed — VWAP, intraday momentum, volume profile, MA deviation.
    Designed for 30-second Monitor scan cycles."""

    def compute(self, data: dict) -> dict:
        symbol = data["symbol"]
        bars = data.get("bars", [])
        if not bars or len(bars) < 5:
            return {"symbol": symbol, "factors": {}}

        prices = [b["p"] for b in bars]
        vols = [b.get("v", 0) for b in bars]
        n = len(prices)
        price = prices[-1]
        vol = vols[-1]

        f = {}

        # ── VWAP & deviation ──
        vwap_num = sum(prices[i] * vols[i] for i in range(n))
        vwap_den = sum(vols)
        vwap = vwap_num / vwap_den if vwap_den > 0 else price
        f["vwap"] = vwap
        f["vwap_deviation"] = (price - vwap) / (vwap + 1e-10)

        # ── Intraday moving averages ──
        for period in (5, 10, 20):
            ma = sma(prices, period)
            if ma:
                f[f"ma_{period}"] = ma
                f[f"price_vs_ma_{period}"] = (price - ma) / (ma + 1e-10)

        # ── Intraday momentum ──
        for period in (5, 10):
            f[f"roc_{period}"] = roc(prices, period)

        f["rsi_14"] = rsi(prices, 14)

        # ── Volume profile ──
        for period in (5, 10):
            ma_vol = sma(vols, period)
            if ma_vol:
                f[f"volume_ma_{period}"] = ma_vol
                f[f"volume_ratio_{period}"] = vol / (ma_vol + 1e-10)

        # ── Day position ──
        if n >= 2:
            day_high = max(prices)
            day_low = min(prices)
            day_range = day_high - day_low
            if day_range > 0:
                f["day_position"] = (price - day_low) / day_range
            f["day_return"] = (price - prices[0]) / (prices[0] + 1e-10)

        # ── Trend strength (price-based ADX proxy) ──
        # Compares short-term vs long-term directional movement
        if n >= 20:
            up_moves = sum(max(prices[i] - prices[i-5], 0) for i in range(5, n))
            down_moves = sum(max(prices[i-5] - prices[i], 0) for i in range(5, n))
            total = up_moves + down_moves
            if total > 0:
                f["trend_strength"] = abs(up_moves - down_moves) / total * 100  # 0-100, like ADX

        return {"symbol": symbol, "factors": f}


# ═══════════════════════════════════════════════════════════════════════
# Alpha360Engine — extended factor set: ~200 additional factors
# on top of FactorEngine (~50) for a total of ~250 dimensions.
# Categories: volatility distribution, volume microstructure, price
# patterns, acceleration, mean reversion, liquidity.
# ═══════════════════════════════════════════════════════════════════════

class Alpha360Engine:
    """Adds ~200 factors on top of FactorEngine base set."""

    def compute(self, base_factors: dict, klines: list) -> dict:
        """Compute Alpha360 factors given base Alpha158 factors + raw klines.

        Returns dict of additional factors (does NOT include base factors).
        Caller should merge with FactorEngine output.
        """
        if len(klines) < 5:
            return {}
        closes = [float(k.get("close", 0)) for k in klines]
        opens = [float(k.get("open", 0)) for k in klines]
        highs = [float(k.get("high", 0)) for k in klines]
        lows = [float(k.get("low", 0)) for k in klines]
        volumes = [float(k.get("volume", 0)) for k in klines]
        n = len(closes)
        f = {}

        # ── Volatility distribution (15 factors) ──
        returns = [(closes[i] - closes[i-1]) / max(closes[i-1], 1e-10) for i in range(1, n)]

        def _mean(arr): return sum(arr) / len(arr) if arr else 0.0
        def _std(arr): return (sum((x - _mean(arr))**2 for x in arr) / max(len(arr)-1, 1))**0.5 if len(arr) > 1 else 0.0
        mean_ret = _mean(returns)
        std_ret = _std(returns)

        for w in [5, 10, 20]:
            if n > w:
                r = returns[-w:]
                f[f"vol_skew_{w}d"] = (sum((x - _mean(r))**3 for x in r) / max(len(r), 2)) / max(std_ret**3, 1e-10) if std_ret > 0 else 0
                f[f"vol_kurt_{w}d"] = (sum((x - _mean(r))**4 for x in r) / max(len(r), 2)) / max(std_ret**4, 1e-10) - 3 if std_ret > 0 else 0
                f[f"vol_amplitude_{w}d"] = (_mean(highs[-w:]) - _mean(lows[-w:])) / max(_mean(closes[-w:]), 1e-10)
                f[f"vol_up_ratio_{w}d"] = len([x for x in r if x > 0]) / max(len(r), 1)

        # ── Volume microstructure (25 factors) ──
        avg_vol_5 = _mean(volumes[-5:]) if n >= 5 else volumes[-1]
        avg_vol_20 = _mean(volumes[-20:]) if n >= 20 else avg_vol_5
        avg_vol_60 = _mean(volumes[-60:]) if n >= 60 else avg_vol_20

        for w in [5, 10, 20, 60]:
            if n >= w:
                v = volumes[-w:]
                f[f"vol_ratio_{w}d"] = _mean(v) / max(avg_vol_60, 1)
                f[f"vol_std_{w}d"] = _std(v) / max(_mean(v), 1)
                f[f"vol_cv_{w}d"] = _std(v) / max(_mean(v), 1e-10)  # coefficient of variation
                if n >= 2*w:
                    f[f"vol_mom_{w}d"] = _mean(v) / max(_mean(volumes[-2*w:-w]), 1) - 1

        # Turnover acceleration
        if n >= 10:
            vol_ma5 = [_mean(volumes[i-5:i]) for i in range(5, n)]
            vol_ma20 = [_mean(volumes[i-20:i]) if i >= 20 else vol_ma5[i-5] for i in range(5, n)]
            f["vol_ma5_accel"] = (vol_ma5[-1] - vol_ma5[-2]) / max(abs(vol_ma5[-2]), 1) if len(vol_ma5) >= 2 else 0
            f["vol_ma5_ma20"] = vol_ma5[-1] / max(vol_ma20[-1], 1) - 1 if vol_ma20 else 0

        # Dollar volume (price × volume) for liquidity
        for w in [5, 20]:
            if n >= w:
                dv = [closes[i] * volumes[i] for i in range(n-w, n)]
                f[f"dollar_vol_{w}d"] = _mean(dv)

        # ── Price patterns (10 factors) ──
        for i in range(max(0, n-5), n):
            idx = i
            o, h, l, c = opens[idx], highs[idx], lows[idx], closes[idx]
            body = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            total_range = max(h - l, 1e-10)

            f[f"body_ratio_{n-idx}d"] = body / total_range
            f[f"upper_shadow_{n-idx}d"] = upper_shadow / max(body, 1e-10) if body > 0 else 0
            f[f"lower_shadow_{n-idx}d"] = lower_shadow / max(body, 1e-10) if body > 0 else 0

        # ── Acceleration factors (15 factors) ──
        for w in [5, 10, 20]:
            if n > w:
                ma = _mean(closes[-w:])
                ma_prev = _mean(closes[-2*w:-w])
                f[f"ma_{w}d_accel"] = (ma - ma_prev) / max(abs(ma_prev), 1e-10)
                # Second derivative of price
                if n > 3:
                    d1 = [(closes[i] - closes[i-1]) / max(closes[i-1], 1e-10) for i in range(n-2, n)]
                    d2 = d1[-1] - d1[0] if len(d1) >= 2 else 0
                    f[f"accel_{w}d"] = d2

        # ── Mean reversion factors (15 factors) ──
        for w in [5, 10, 20]:
            if n >= w:
                ma = _mean(closes[-w:])
                std_w = _std(closes[-w:])
                if std_w > 0:
                    f[f"zscore_{w}d"] = (closes[-1] - ma) / std_w
                    # Distance from Bollinger bands
                    f[f"bb_position_{w}d"] = (closes[-1] - (ma - 2*std_w)) / max(4*std_w, 1e-10)
                # Deviation from w-day high
                f[f"high_dev_{w}d"] = (closes[-1] - max(highs[-w:])) / max(abs(max(highs[-w:])), 1e-10)
                f[f"low_dev_{w}d"] = (closes[-1] - min(lows[-w:])) / max(abs(min(lows[-w:])), 1e-10)

        # ── Overnight / gap factors (10 factors) ──
        gaps = []
        for i in range(1, n):
            if closes[i-1] > 0:
                gaps.append((opens[i] - closes[i-1]) / closes[i-1])
        if gaps:
            for w in [5, 10, 20]:
                if len(gaps) >= w:
                    g = gaps[-w:]
                    f[f"gap_mean_{w}d"] = _mean(g)
                    f[f"gap_std_{w}d"] = _std(g)
                    f[f"gap_pos_ratio_{w}d"] = len([x for x in g if x > 0]) / max(len(g), 1)

        # ── High-low range factors (10 factors) ──
        for w in [5, 10, 20]:
            if n >= w:
                ranges = [(highs[i] - lows[i]) / max(closes[i-1], 1e-10) for i in range(n-w, n)]
                f[f"hl_range_mean_{w}d"] = _mean(ranges)
                f[f"hl_range_max_{w}d"] = max(ranges)
                f[f"hl_range_trend_{w}d"] = ranges[-1] - _mean(ranges[:-1]) if len(ranges) >= 2 else 0

        # ── Up/down day statistics (10 factors) ──
        for w in [5, 10, 20, 60]:
            if n >= w:
                up_days = sum(1 for i in range(n-w, n) if closes[i] > closes[i-1])
                down_days = w - up_days
                f[f"up_ratio_{w}d"] = up_days / w
                f[f"up_down_ratio_{w}d"] = up_days / max(down_days, 1)

        # ── Consecutive up/down streak (5 factors) ──
        streak_up = 0
        streak_down = 0
        for i in range(n-1, max(0, n-15)-1, -1):
            if closes[i] > closes[i-1]:
                if streak_down > 0:
                    break
                streak_up += 1
            else:
                if streak_up > 0:
                    break
                streak_down += 1
        f["streak_up"] = streak_up
        f["streak_down"] = streak_down

        # ── Relative price position (5 factors) ──
        for w in [20, 60, 120]:
            if n >= w:
                high_w = max(highs[-w:])
                low_w = min(lows[-w:])
                rng = max(high_w - low_w, 1e-10)
                f[f"price_position_{w}d"] = (closes[-1] - low_w) / rng

        # ── Volume-price correlation (5 factors) ──
        for w in [5, 10, 20]:
            if n >= w:
                p = closes[-w:]
                v = volumes[-w:]
                if len(p) > 2 and _std(p) > 0 and _std(v) > 0:
                    cov = sum((p[i]-_mean(p))*(v[i]-_mean(v)) for i in range(len(p))) / len(p)
                    f[f"vol_price_corr_{w}d"] = cov / max(_std(p)*_std(v), 1e-10)

        return f

    def add_cross_sectional(self, symbol_factors: dict) -> dict:
        """Compute cross-sectional rank factors across all symbols.

        Args:
            symbol_factors: {symbol: {factor_name: value, ...}}

        Returns:
            {symbol: {factor_name_rank: rank_pct, ...}}
        """
        if len(symbol_factors) < 2:
            return {}
        # Collect all factor values per symbol
        factor_keys = set()
        for factors in symbol_factors.values():
            factor_keys.update(factors.keys())
        factor_keys = sorted(factor_keys)
        # Rank each factor across symbols (0-1 percentile)
        ranked = {}
        for fk in factor_keys:
            vals = []
            for sym, factors in symbol_factors.items():
                v = factors.get(fk)
                if isinstance(v, (int, float)):
                    vals.append((sym, v))
            if len(vals) < 2:
                continue
            sorted_vals = sorted(vals, key=lambda x: x[1])
            n_vals = len(sorted_vals)
            for rank_pos, (sym, _) in enumerate(sorted_vals):
                if sym not in ranked:
                    ranked[sym] = {}
                ranked[sym][f"{fk}_rank"] = rank_pos / max(n_vals - 1, 1)
        return ranked
