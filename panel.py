"""统一 panel 契约 —— 所有量化工具入参的归一化层。

## 为什么需要它

2026-09-15 全链路实跑暴露的问题：同一个 `klines` 参数在不同工具里要求 4 种形状
（`factor_tearsheet` 要 `{symbol:[bars]}`、`vol_forecast` 要 `[bars]`、`learn_graph`
要 `{columns,rows}`、`dml_cate` 要 `{name:[...]}`），接一个 agent 就要重写一遍转换。
本模块把这些形状统一收口：**任何工具都应接受下面任一形状**，内部再归一化。

## 接受的形状（全部等价）

1. **flat bars**      ``[{date, open, high, low, close, volume?}, ...]``
   单序列。`symbol` 可缺省。
2. **keyed bars**     ``{symbol: [bars, ...]}``
   多序列。这是 `factor_tearsheet` / `portfolio_optimize` 的原生形状。
3. **columnar panel** ``{columns: [...], rows: [[...], ...]}``
   列式矩阵，`columns` 里需含 `date`（或 `datetime`）以及 `close` 或 `symbol`。
   这是 `learn_graph` / `pcmci_discover` 的原生形状。
4. **wide close**     ``{date: {symbol: close}}`` 或 ``{symbol: {date: close}}``
   宽表（谁在外层自动识别）。
5. **long frame**     ``[{date, symbol, close|value}, ...]``
   已含 symbol 的长表，因子面板与价格面板都用它。

## 归一化目标

- :func:`as_dataframe`  → long 形式 ``DataFrame[date, symbol, open, high, low, close, volume]``
- :func:`as_bar_dict`   → ``{symbol: [bar, ...]}``
- :func:`as_close_frame`→ ``DataFrame(index=date, columns=symbol)``
- :func:`as_bars`       → ``[bar, ...]``（单序列；多序列时报错而非静默取第一个）
- :func:`as_series`     → ``[{date, value}, ...]``
- :func:`as_matrix`     → ``(columns, rows)`` 全数值矩阵
- :func:`as_features`   → ``{name: [float, ...]}``

所有函数对无法识别的输入抛 :class:`PanelError`，错误信息里带上实际收到的形状，
避免出现「因子算出来是空的但没人知道为什么」。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PanelError",
    "as_dataframe",
    "as_bar_dict",
    "as_close_frame",
    "as_bars",
    "as_series",
    "as_matrix",
    "as_matrix_frame",
    "as_features",
    "as_series_list",
    "detect_shape",
    "present_fields",
    "is_flat_bars",
]

# 字段别名 —— 实盘/回测数据来自多个上游，列名不统一
_DATE_KEYS = ("date", "datetime", "time", "timestamp", "trade_date", "day")
_SYMBOL_KEYS = ("symbol", "instrument", "code", "ticker", "asset")
_FIELD_ALIASES = {
    "open": ("open", "o", "open_price"),
    "high": ("high", "h", "high_price"),
    "low": ("low", "l", "low_price"),
    "close": ("close", "c", "close_price", "px", "price", "value"),
    "volume": ("volume", "vol", "v", "turnover"),
    "amount": ("amount", "amt", "turnover_value"),
}


class PanelError(ValueError):
    """入参无法归一化。消息里包含实际形状，便于调用方自查。"""


def _shape_of(obj: Any) -> str:
    if isinstance(obj, pd.DataFrame):
        return "DataFrame(shape=%s, columns=%s)" % (obj.shape, list(obj.columns)[:8])
    if isinstance(obj, dict):
        keys = list(obj)[:8]
        inner = ""
        if keys:
            v = obj[keys[0]]
            if isinstance(v, (list, tuple)):
                inner = " 值类型=list(len=%d)" % len(v)
                if v and isinstance(v[0], dict):
                    inner += " 元素键=%s" % list(v[0])[:8]
            elif isinstance(v, dict):
                inner = " 值类型=dict(键=%s)" % list(v)[:5]
        return "dict(键=%s%s)" % (keys, inner)
    if isinstance(obj, (list, tuple)):
        head = obj[0] if obj else None
        if isinstance(head, dict):
            return "list(len=%d) 元素键=%s" % (len(obj), list(head)[:8])
        return "list(len=%d) 首元素=%r" % (len(obj), head)
    if isinstance(obj, pd.Series):
        return "Series(len=%d, name=%r)" % (len(obj), obj.name)
    return type(obj).__name__


def detect_shape(obj: Any) -> str:
    """给入参打标签，仅用于报错/日志，不参与归一化决策。"""
    if isinstance(obj, pd.DataFrame):
        return "long_frame"
    if isinstance(obj, pd.Series):
        return "series"
    if isinstance(obj, dict):
        if "columns" in obj and "rows" in obj:
            return "columnar_panel"
        if not obj:
            return "empty_dict"
        first = next(iter(obj.values()))
        if isinstance(first, (list, tuple)):
            return "keyed_bars"
        if isinstance(first, dict):
            return "wide_close"
        if isinstance(first, (int, float, np.floating)):
            return "series_dict"  # {date: value}
        return "unknown_dict"
    if isinstance(obj, (list, tuple)):
        if not obj:
            return "empty_list"
        first = obj[0]
        if isinstance(first, dict):
            keys = set(first)
            if keys & set(_SYMBOL_KEYS) and any(
                k in first for k in ("close", "c", "value")
            ):
                return "long_frame"
            if "close" in first or "c" in first or "open" in first:
                return "flat_bars"
            if "value" in first:
                return "series"
            return "unknown_rows"
        if isinstance(first, (int, float, np.floating)):
            return "value_list"
        if isinstance(first, (list, tuple)):
            return "matrix_rows"
        return "unknown_list"
    return "unknown"


def _pick(d: dict, aliases: Sequence[str], default=None):
    for a in aliases:
        if a in d and d[a] is not None:
            return d[a]
    return default


def _value_list_to_bars(values: Iterable[Any], dates: Sequence[Any] | None, symbol: str):
    vals = list(values)
    if dates is None:
        dates = list(range(len(vals)))
    n = min(len(vals), len(dates))
    return [
        {"date": dates[i], "symbol": symbol, "close": float(vals[i])}
        for i in range(n)
        if vals[i] is not None and not (isinstance(vals[i], float) and np.isnan(vals[i]))
    ]


# ---------------------------------------------------------------- 主归一化

def as_dataframe(obj: Any, default_symbol: str = "asset") -> pd.DataFrame:
    """任意受支持形状 → long ``DataFrame[date, symbol, open, high, low, close, volume]``。

    ``open/high/low`` 缺失时用 ``close`` 补齐（多数工具只需要 close）。
    """
    if obj is None:
        raise PanelError("入参为空（None）")
    if isinstance(obj, pd.DataFrame):
        return _normalize_long(obj, default_symbol)
    if isinstance(obj, pd.Series):
        df = obj.reset_index()
        df.columns = ["date", "close"][: len(df.columns)]
        df["symbol"] = default_symbol
        return _normalize_long(df, default_symbol)

    kind = detect_shape(obj)

    if kind == "columnar_panel":
        cols = list(obj.get("columns") or [])
        rows = list(obj.get("rows") or [])
        if not cols:
            raise PanelError("columnar panel 缺少 columns")
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        if rows and len(rows[0]) != len(cols):
            raise PanelError(
                "columnar panel 的 rows 行宽 %d 与 columns 数 %d 不一致"
                % (len(rows[0]), len(cols))
            )
        return _normalize_long(pd.DataFrame(rows, columns=cols), default_symbol)

    if kind == "keyed_bars":
        frames = []
        for sym, rows in obj.items():
            if not rows:
                continue
            sub = as_dataframe(rows, default_symbol=str(sym))
            if "symbol" not in sub or sub["symbol"].isna().all():
                sub["symbol"] = str(sym)
            else:
                sub["symbol"] = sub["symbol"].fillna(str(sym)).replace("", str(sym))
            frames.append(sub)
        if not frames:
            raise PanelError("keyed bars 里没有任何非空序列：" + _shape_of(obj))
        return pd.concat(frames, ignore_index=True)

    if kind == "wide_close":
        frames = []
        for outer, inner in obj.items():
            if not isinstance(inner, dict):
                raise PanelError("wide_close 内层必须是 dict：" + _shape_of(obj))
            # 判断哪一层是 symbol：内层键若是可解析日期则外键是 symbol，反之亦然
            is_date_inner = any(_looks_like_date(k) for k in list(inner)[:3])
            for k, v in inner.items():
                if is_date_inner:
                    sym, date = str(outer), k
                else:
                    sym, date = str(k), outer
                frames.append({"date": date, "symbol": sym, "close": v})
        if not frames:
            raise PanelError("wide_close 为空：" + _shape_of(obj))
        return _normalize_long(pd.DataFrame(frames), default_symbol)

    if kind == "series_dict":
        # {date: value} 只在键真的像日期时才接受 —— 否则 {"weird": 123} 会静默
        # 变成一条日期为 "weird" 的单点序列，正是要消灭的那类静默错误
        if not all(_looks_like_date(k) for k in obj):
            bad = [k for k in obj if not _looks_like_date(k)][:3]
            raise PanelError(
                "series_dict 的键不像日期（%s…）；若是 {名称: 数值} 请改用 "
                "{name: [values]} 或 {columns, rows}" % bad
            )
        return _normalize_long(
            pd.DataFrame(
                [{"date": k, "symbol": default_symbol, "close": v} for k, v in obj.items()]
            ),
            default_symbol,
        )

    if kind in ("flat_bars", "long_frame", "series", "matrix_rows", "value_list"):
        if kind == "matrix_rows":
            raise PanelError(
                "二维数组缺少列名，请用 {columns:[...], rows:[[...]]} 或 [{date,close},...]"
            )
        if kind == "value_list":
            return _normalize_long(
                pd.DataFrame(_value_list_to_bars(obj, None, default_symbol)), default_symbol
            )
        return _normalize_long(pd.DataFrame(list(obj)), default_symbol)

    raise PanelError("无法识别的入参形状 %s：%s" % (kind, _shape_of(obj)))


def is_flat_bars(obj: Any) -> bool:
    """该输入是否**已经是**扁平 bar 数组（单序列、无 symbol 列）。

    给调用方做「快速路径」判定用。关键点是**带 symbol 列的 list of dicts 不算**
    —— 那可能是多条序列，直接当 bars 处理会把多个标的静默合并成一条
    （2026-09-15 kronos `_parse_klines` 实测到的坑）。
    """
    if not (isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict)):
        return False
    keys = set(obj[0])
    if keys & set(_SYMBOL_KEYS):
        return False
    has_close = any(a in keys for a in _FIELD_ALIASES["close"])
    has_open = any(a in keys for a in _FIELD_ALIASES["open"])
    return has_close and has_open


def present_fields(obj: Any) -> set:
    """返回**原始输入里真实出现过**的标准字段集合。

    为什么需要它：归一化会用 close 补齐缺失的 open/high/low，于是「输入到底有
    没有 high/low」这件事在归一化之后就丢了。要决定是否计算依赖高低价的指标
    （Parkinson 波动率、ATR、振幅类因子）时必须回头问**原始输入**，
    否则会把补出来的 close 当成真实高低价、算出 0 波动率这种静默错误。
    """
    if obj is None:
        return set()
    if isinstance(obj, pd.DataFrame):
        keys = set(obj.columns)
    elif isinstance(obj, pd.Series):
        keys = {obj.name or "close"}
    elif isinstance(obj, dict):
        kind = detect_shape(obj)
        if kind == "columnar_panel":
            keys = {str(c) for c in (obj.get("columns") or [])}
        elif kind in ("wide_close", "series_dict"):
            return {"close"}  # 宽表/标量字典只承载数值，没有 OHLC
        elif kind == "keyed_bars":
            keys = set()
            for rows in obj.values():
                if rows and isinstance(rows[0], dict):
                    keys |= set(rows[0])
        else:
            first = next(iter(obj.values()), None)
            keys = set(first) if isinstance(first, dict) else set()
    elif isinstance(obj, (list, tuple)):
        if not obj:
            keys = set()
        elif isinstance(obj[0], dict):
            keys = set(obj[0])
        else:
            return {"close"}  # 裸数值序列
    else:
        keys = set()

    std = set()
    for k in keys:
        for name, aliases in _FIELD_ALIASES.items():
            if k in aliases:
                std.add(name)
        if k in _DATE_KEYS:
            std.add("date")
        if k in _SYMBOL_KEYS:
            std.add("symbol")
    return std


def _looks_like_date(v: Any) -> bool:
    if isinstance(v, (pd.Timestamp,)):
        return True
    s = str(v)
    return len(s) >= 8 and s[:4].isdigit() and ("-" in s or "/" in s or s.isdigit())


def _normalize_long(df: pd.DataFrame, default_symbol: str) -> pd.DataFrame:
    if df.empty:
        raise PanelError("归一化后为空")
    df = df.copy()
    # 列名别名 → 标准名
    date_col = None
    for a in _DATE_KEYS:
        if a in df.columns:
            date_col = a
            break
    if date_col is None:
        if "symbol" in df.columns and len(df.columns) == 2:
            date_col = [c for c in df.columns if c != "symbol"][0]
        else:
            df = df.reset_index()
            for a in _DATE_KEYS + ("index",):
                if a in df.columns:
                    date_col = a
                    break
    if date_col is None:
        raise PanelError("找不到日期列，已尝试 %s；实际列=%s" % (list(_DATE_KEYS), list(df.columns)))

    sym_col = next((a for a in _SYMBOL_KEYS if a in df.columns), None)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["symbol"] = (
        df[sym_col].astype(str) if sym_col else default_symbol
    )
    for std, aliases in _FIELD_ALIASES.items():
        raw = _pick(df, aliases)
        if raw is None:
            continue
        # 别名可能命中同一列（如 close 同时匹配 price 与 value），只取一次
        if std == "close" and "open" not in out:
            pass
        out[std] = pd.to_numeric(raw, errors="coerce")
    if "close" not in out.columns:
        raise PanelError("找不到价格列（close/c/price/value）")
    for f in ("open", "high", "low"):
        if f not in out.columns:
            out[f] = out["close"]
    out = out.dropna(subset=["date", "symbol"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------- 派生形状

def as_bar_dict(obj: Any, default_symbol: str = "asset") -> dict:
    """→ ``{symbol: [{date, open, high, low, close, volume}, ...]}``"""
    df = as_dataframe(obj, default_symbol)
    keep = not df.empty
    if not keep:
        raise PanelError("归一化后为空")
    cols = [c for c in ("date", "open", "high", "low", "close", "volume", "amount") if c in df]
    out: dict[str, list] = {}
    for sym, sub in df.groupby("symbol", sort=False):
        rows = sub[cols].to_dict("records")
        for r in rows:
            r["date"] = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
        out[str(sym)] = rows
    return out


def as_close_frame(obj: Any, default_symbol: str = "asset") -> pd.DataFrame:
    """→ ``DataFrame(index=date, columns=symbol)``，前向填充。"""
    df = as_dataframe(obj, default_symbol)
    px = df.pivot_table(index="date", columns="symbol", values="close")
    # 补齐日期轴：pivot_table 会把全 NaN 的交易日整行丢掉，那样停牌缺口就
    # 消失了、ffill 也就无从生效。日期轴必须是输入里出现过的所有日期。
    all_dates = pd.Index(sorted(df["date"].unique()), name="date")
    return px.reindex(all_dates).sort_index().ffill()


def as_bars(obj: Any, default_symbol: str = "asset") -> list:
    """→ 单个序列的 bar 列表。

    多序列时**报错**而不是静默取第一个 —— 静默取第一个正是 2026-09-15 那批
    形状错配难以定位的原因。
    """
    bd = as_bar_dict(obj, default_symbol)
    if len(bd) > 1:
        raise PanelError(
            "该工具只接受单序列，但收到 %d 个序列：%s。请指定其中一个（例如 {sym: bars}）"
            % (len(bd), list(bd)[:6])
        )
    return next(iter(bd.values()))


def as_series(obj: Any, key: str = "close", default_symbol: str = "asset") -> list:
    """→ ``[{date, value}, ...]``（change_point / granger 等标量序列工具用）。"""
    df = as_dataframe(obj, default_symbol)
    if key not in df.columns:
        raise PanelError("序列里没有 %r 列；有 %s" % (key, list(df.columns)))
    syms = list(df["symbol"].unique())
    if len(syms) > 1:
        raise PanelError(
            "该工具只接受单序列，但收到 %d 个：%s" % (len(syms), syms[:6])
        )
    out = df[["date", key]].rename(columns={key: "value"}).dropna()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict("records")


def _columnar_to_wide(obj: Any) -> pd.DataFrame:
    """``{columns, rows}`` 直接当宽表：日期列作 index，其余按数值列保留。

    用于「日期列 + 以标的命名的数值列」这种矩阵 —— 它没有 close 列，
    过不了 :func:`as_dataframe` 的价格列要求，但对因果图工具是合法输入。
    """
    if not (isinstance(obj, dict) and "columns" in obj and "rows" in obj):
        return pd.DataFrame()
    cols = [str(c) for c in (obj.get("columns") or [])]
    rows = list(obj.get("rows") or [])
    if not cols:
        return pd.DataFrame()
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [rows]
    df = pd.DataFrame(rows, columns=cols)
    date_col = next((c for c in cols if c in _DATE_KEYS), None)
    if date_col:
        df = df.set_index(pd.to_datetime(df[date_col], errors="coerce"))
        df = df.drop(columns=[date_col])
    df = df.drop(columns=[c for c in df.columns if c in _SYMBOL_KEYS], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    if date_col:
        df = df.sort_index()
    return df.dropna(axis=1, how="all")


def as_matrix_frame(obj: Any, value: str = "close") -> pd.DataFrame:
    """→ 数值宽表 ``DataFrame(index=date, columns=symbol)``（因果图工具用）。

    先按 panel 常规路径归一化；若输入是「日期列 + 数值列」的列式矩阵（没有
    close 列），退回 :func:`_columnar_to_wide` 直接当宽表。
    """
    wide = pd.DataFrame()
    try:
        df = as_dataframe(obj)
    except PanelError:
        df = None
    if df is not None and value in df.columns:
        wide = (df.pivot_table(index="date", columns="symbol", values=value)
                  .sort_index().ffill().dropna(axis=1, how="any"))
    if wide.empty or wide.shape[0] < 2:
        wide = _columnar_to_wide(obj)
    if wide.empty or wide.shape[0] < 2 or wide.shape[1] < 1:
        raise PanelError("矩阵化后样本不足：shape=%s" % (getattr(wide, "shape", None),))
    return wide


def as_matrix(obj: Any, value: str = "close", with_dates: bool = False):
    """→ ``(columns, rows)`` 全数值矩阵；``with_dates=True`` 时附日期。

    **列里不含日期** —— 因果图结构学习（learn_graph / pcmci_discover）会把
    rows 直接 ``np.asarray(..., dtype=float)``，日期列会让它崩在
    ``could not convert string to float``（2026-09-15 实测）。日期走独立的
    ``dates`` 字段，与原生 schema 一致。
    """
    wide = as_matrix_frame(obj, value)
    cols = [str(c) for c in wide.columns]
    rows = [[float(v) for v in r] for r in wide.to_numpy()]
    if with_dates:
        dates = [pd.Timestamp(i).strftime("%Y-%m-%d") for i in wide.index]
        return cols, rows, dates
    return cols, rows


def as_series_list(obj: Any, symbol_key: str = "symbol",
                   klines_key: str = "klines") -> list:
    """→ 原生多序列形态 ``[{symbol, klines: [bars]}, ...]``。

    这是 ``ml_train_rolling`` / ``ml_predict`` / kronos ``forecast_batch``
    的原生入参。接受三种等价写法：

    - 原生 ``[{symbol?, klines: [bars]}, ...]``
    - ``{symbol: [bars]}``（keyed bars，最常见的写法）
    - ``[[bars], ...]``（无标的名的纯列表）

    每个序列内部都会过 :func:`as_bars`，因此序列本身也可以是多形态的。
    空输入返回 ``[]``（``ml_predict`` 允许空列表走 Redis 因子里程碑）。
    """
    items: list = []
    if obj is None:
        return []
    if isinstance(obj, dict):
        items = [{symbol_key: str(k), klines_key: v} for k, v in obj.items()]
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, dict) and isinstance(v.get(klines_key), (list, tuple, dict)):
                items.append({
                    symbol_key: str(v.get(symbol_key) or "series-%d" % i),
                    klines_key: v[klines_key],
                })
            else:
                items.append({symbol_key: "series-%d" % i, klines_key: v})
    else:
        raise PanelError("多序列入参必须是 list 或 dict：%s" % _shape_of(obj))

    out = []
    for it in items:
        sym = it[symbol_key]
        try:
            bars = as_bars(it[klines_key], default_symbol=sym)
        except PanelError as e:
            raise PanelError("%s 的序列无法解析: %s" % (sym, e))
        out.append({symbol_key: sym, klines_key: bars})
    return out


def as_features(obj: Any, default_symbol: str = "asset") -> dict:
    """→ ``{name: [float, ...]}``（dml_cate 的 features）。

    接受：``{name: [values]}``（原生） / ``{columns,rows}`` / bars（展开成 per-symbol 序列）。
    """
    if isinstance(obj, dict) and "columns" in obj and "rows" in obj:
        cols = list(obj.get("columns") or [])
        rows = list(obj.get("rows") or [])
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        skip = set(_DATE_KEYS) | set(_SYMBOL_KEYS)
        out = {}
        for i, c in enumerate(cols):
            if str(c) in skip:
                continue  # 日期/标号列不是特征
            try:
                out[str(c)] = [
                    float(r[i]) if r[i] is not None and r[i] != "" else float("nan")
                    for r in rows
                ]
            except (TypeError, ValueError):
                continue  # 非数值列（如标的名）直接跳过，不抛错
        if not out:
            raise PanelError(
                "columnar panel 里没有可用的数值特征列（已跳过 %s）：columns=%s"
                % (sorted(skip), cols)
            )
        return out
    if isinstance(obj, dict):
        if obj and all(isinstance(v, (list, tuple)) for v in obj.values()):
            # {name: [标量]} 是原生 features；{symbol: [bar dict]} 不是 ——
            # 后者要走 as_dataframe 展开，否则 float(bar) 会炸
            scalar_only = all(
                all(not isinstance(x, (dict, list, tuple)) for x in v)
                for v in obj.values()
                if v
            )
            if scalar_only:
                return {str(k): [float(x) for x in v] for k, v in obj.items()}
        if obj and all(isinstance(v, (int, float)) for v in obj.values()):
            return {str(k): [float(v)] for k, v in obj.items()}
    df = as_dataframe(obj, default_symbol)
    wide = df.pivot_table(index="date", columns="symbol", values="close").sort_index().ffill()
    return {str(c): [float(v) for v in wide[c].to_numpy()] for c in wide.columns}
