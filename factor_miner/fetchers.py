"""fetchers.py — 行情抓取共享助手（东财节流 session / mootdx 探测客户端）。

移植自 Athena py-sidecar server.py（em_get / tdx_client），独立成包内模块，
使 update_data 不再依赖外部 server.py。所有重依赖（requests/mootdx）惰性导入。
"""
from __future__ import annotations

# ── Eastmoney anti-blocking: global throttle + session reuse ──
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_EM_SESSION = None
_EM_MIN_INTERVAL = 1.0
_em_last_call = [0.0]


def _get_em_session():
    global _EM_SESSION
    if _EM_SESSION is None:
        import requests as _r
        _EM_SESSION = _r.Session()
        _EM_SESSION.headers.update({"User-Agent": _UA})
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            _adapter = HTTPAdapter(max_retries=Retry(
                total=3, connect=3, backoff_factor=0.6,
                status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]))
            _EM_SESSION.mount("https://", _adapter)
            _EM_SESSION.mount("http://", _adapter)
        except Exception:
            pass
    return _EM_SESSION


def em_get(url: str, params: dict = None, headers: dict = None, timeout: int = 15,
           method: str = "GET", **kwargs):
    """Eastmoney unified request: auto throttle + session reuse + default UA.
    All eastmoney.com APIs must go through this to avoid IP ban."""
    import random as _random
    import time as _time
    wait = _EM_MIN_INTERVAL - (_time.time() - _em_last_call[0])
    if wait > 0:
        _time.sleep(wait + _random.uniform(0.1, 0.5))
    try:
        if method.upper() == "POST":
            return _get_em_session().post(url, params=params, headers=headers,
                                          timeout=timeout, **kwargs)
        return _get_em_session().get(url, params=params, headers=headers, timeout=timeout, **kwargs)
    finally:
        _em_last_call[0] = _time.time()


# ── mootdx TCP probing client ──
_TDX_SERVERS = [
    ('119.97.185.59', 7709), ('124.70.133.119', 7709), ('116.205.183.150', 7709),
    ('123.60.73.44', 7709),  ('116.205.163.254', 7709), ('121.36.225.169', 7709),
    ('123.60.70.228', 7709), ('124.71.9.153', 7709),    ('110.41.147.114', 7709),
    ('124.71.187.122', 7709),
]

_mootdx_quotes = None
_mootdx_quotes_loaded = False


def _get_mootdx():
    global _mootdx_quotes, _mootdx_quotes_loaded
    if not _mootdx_quotes_loaded:
        try:
            from mootdx.quotes import Quotes
            _mootdx_quotes = Quotes
        except ImportError:
            _mootdx_quotes = None
        _mootdx_quotes_loaded = True
    return _mootdx_quotes


def _probe(ip, port, timeout=2.0):
    import socket as _socket
    try:
        with _socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False


def tdx_client(market='std'):
    """Create mootdx client with TCP probing + 3-level fallback to avoid 0.11.x BESTIP bug."""
    Quotes = _get_mootdx()
    if Quotes is None:
        raise RuntimeError("mootdx not installed. Run: pip install mootdx")
    for ip, port in _TDX_SERVERS:
        if _probe(ip, port):
            return Quotes.factory(market=market, server=(ip, port))
    try:
        return Quotes.factory(market=market, bestip=True)
    except Exception:
        pass
    try:
        return Quotes.factory(market=market)
    except Exception as e:
        raise RuntimeError(
            "All mootdx servers unreachable. Overseas IPs usually timeout on TCP 7709. "
            "Use domestic proxy or update _TDX_SERVERS list. Original error: %s" % e
        )
