"""数据健康：面板/qlib 日历的新鲜度与覆盖度（治 G7）。

## 为什么要有它

实测出来的最隐蔽风险：**回测会静默用旧数据**。当时的证据链是
`update_data` 四次 3600s 超时（东财 502 / mootdx bestip / 覆盖率 155-5204），
而面板版本只出现在执行器的启动日志里 —— 调用方（尤其是 agent）无从知道
"我这次回测读的面板是几天前的"。

于是把这件事变成**可查询 + 可拦截**：
- `data_status` 工具：面板版本 / 最后交易日 / 距今几个自然日 / qlib 日历尾 / 覆盖度；
- 回测前置闸门：`stale_days > threshold` 且未显式 `allow_stale` → 直接 `DATA_STALE`，
  而不是给一个看起来正常的旧数据结果。

阈值默认 5 个自然日（覆盖周末），可用 `FACTOR_MINER_STALE_DAYS` 覆盖；
`allow_stale=True` 只用于排查历史数据，正常调用不该用它。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = "/app/data/factor_mining"
DEFAULT_PROVIDER = "~/.qlib/qlib_data/cn_data"
DEFAULT_STALE_DAYS = 5


def _data_dir() -> Path:
    return Path(os.environ.get("FACTOR_MINER_DATA_DIR") or DEFAULT_DATA_DIR)


def _provider_dir() -> Path:
    """qlib 数据目录：容器内是 /app/.qlib/qlib_data/cn_data，本地是 ~/.qlib/...。

    两个都探测（存在优先），避免在容器里把日历读成 None 而误报"新鲜"。
    """
    env = os.environ.get("QLIB_PROVIDER_URI")
    if env:
        return Path(env).expanduser()
    for cand in ("/app/.qlib/qlib_data/cn_data", DEFAULT_PROVIDER):
        p = Path(cand).expanduser()
        if p.exists():
            return p
    return Path(DEFAULT_PROVIDER).expanduser()


def _read_calendar_tail(provider: Path) -> str | None:
    """qlib 日历最后一行 = 数据可回测的最后交易日（.bin 的位置语义由它决定）。"""
    cal = provider / "calendars" / "day.txt"
    try:
        with cal.open("rb") as fh:
            fh.seek(max(0, cal.stat().st_size - 4096))
            tail = fh.read().decode(errors="ignore").strip().splitlines()
        return tail[-1].strip() if tail else None
    except Exception:  # noqa: BLE001
        return None


def _parse_day(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    text = str(value)[:10].replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def status(provider_uri: str | None = None, data_dir: str | None = None) -> dict[str, Any]:
    """面板/日历的新鲜度与覆盖度。**永不抛异常**（健康查询不该自身出错）。"""
    try:
        panel_dir = Path(data_dir) if data_dir else _data_dir()
        provider = Path(provider_uri).expanduser() if provider_uri else _provider_dir()

        manifest: dict[str, Any] = {}
        mpath = panel_dir / "manifest.json"
        if mpath.exists():
            try:
                manifest = json.loads(mpath.read_text())
            except Exception:  # noqa: BLE001
                manifest = {}

        h5 = panel_dir / "daily_pv_all.h5"
        cal_tail = _read_calendar_tail(provider)
        # 面板日 = max(manifest 记录, h5 文件 mtime)
        # 为什么要有 mtime：`gen_data --full` 重建 h5 **不更新 manifest**（实测），
        # 只信 manifest 会把刚重建好的面板判成旧的。
        mtime_day = (
            datetime.fromtimestamp(h5.stat().st_mtime).date() if h5.exists() else None
        )
        manifest_day = _parse_day(manifest.get("last_trading_day"))
        panel_day = max([d for d in (manifest_day, mtime_day) if d], default=None)
        cal_day = _parse_day(cal_tail)
        # 有效数据日 = min(面板, 日历)：回测同时吃这两份，只有取较旧者才是保守正确的
        last_day = min([d for d in (panel_day, cal_day) if d], default=None)
        stale_days = (date.today() - last_day).days if last_day else None
        threshold = int(os.environ.get("FACTOR_MINER_STALE_DAYS", DEFAULT_STALE_DAYS))

        return {
            "ok": True,
            "panel_dir": str(panel_dir),
            "panel_exists": h5.exists(),
            "panel_bytes": h5.stat().st_size if h5.exists() else None,
            "panel_mtime": (
                datetime.fromtimestamp(h5.stat().st_mtime).isoformat(timespec="seconds")
                if h5.exists() else None
            ),
            "manifest_version": manifest.get("version"),
            "manifest_last_trading_day": manifest.get("last_trading_day"),
            "panel_last_trading_day": panel_day.isoformat() if panel_day else None,
            "qlib_calendar_tail": cal_tail,
            "calendar_last_trading_day": cal_day.isoformat() if cal_day else None,
            "provider_uri": str(provider),
            "last_trading_day": last_day.isoformat() if last_day else None,
            "stale_days": stale_days,
            "stale_threshold_days": threshold,
            "stale": (stale_days is not None and stale_days > threshold),
            "note": (
                "stale=true 时回测默认被拒（error_code=DATA_STALE）；"
                "排查历史数据可用 factor_backtest(..., allow_stale=true)"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def staleness(provider_uri: str | None = None) -> tuple[bool, int | None, str]:
    """给回测闸门用的轻量查询 → (是否过期, stale_days, 人类可读说明)。"""
    st = status(provider_uri=provider_uri)
    if not st.get("ok"):
        return False, None, ""  # 查不出来就不拦（宁可放行也不误杀）
    if not st.get("stale"):
        return False, st.get("stale_days"), ""
    return True, st.get("stale_days"), (
        f"面板最后交易日 {st.get('last_trading_day')}，距今 {st.get('stale_days')} 天 "
        f"(>{st.get('stale_threshold_days')}) —— 回测默认拒绝以免静默用旧数据"
    )
