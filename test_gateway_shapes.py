"""网关层形状容差验收。

网关的参数校验基于**函数的类型注解**（mcp_common._ann_type 读 p.annotation）。
2026-09-15 实测：函数体内接了 panel 归一化，但注解仍写 `klines: list`，
于是 dict 形状在**进入工具函数之前**就被拒：
    {"error": "参数 'klines' 类型错误：期望 array，实际收到 {...}"}
本测试钉住这个拦截点。
"""

import mcp_common
import tools

BARS = [{"date": "2026-01-%02d" % i, "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0 + i * 0.1, "volume": 100} for i in range(1, 31)]
KEYED = {"AAA": BARS, "BBB": BARS}
COLUMNAR = {"columns": ["date", "open", "high", "low", "close"],
            "rows": [[b["date"], b["open"], b["high"], b["low"], b["close"]] for b in BARS]}


def _coerce(tool, arg, value):
    args, err = mcp_common.coerce_args(tools.HANDLERS, tool, {arg: value})
    return args, err


def test_gateway_accepts_object_shapes():
    for tool, arg in tools.SHAPE_TOLERANT_ARGS:
        for value in (KEYED, COLUMNAR):
            args, err = _coerce(tool, arg, value)
            assert err is None, "%s.%s 被网关拒了: %s" % (tool, arg, err)
            assert isinstance(args[arg], dict)


def test_gateway_still_accepts_array_shapes():
    """放宽不能把原生形状弄坏（回归保护）。"""
    for tool, arg in tools.SHAPE_TOLERANT_ARGS:
        args, err = _coerce(tool, arg, BARS)
        assert err is None, (tool, arg, err)
        assert isinstance(args[arg], list)


def test_validator_is_actually_active():
    """对照：注解仍是 int 的参数必须照旧被拒。

    这一条证明上面两个测试不是因为「校验整体失效」才通过的 ——
    放宽是外科手术式的，只作用于形状容差参数。
    """
    args, err = mcp_common.coerce_args(tools.HANDLERS, "vol_forecast", {"horizon": "abc"})
    assert err is not None, "校验器没有生效 —— 上两个测试的通过没有意义"
    assert "integer" in err
