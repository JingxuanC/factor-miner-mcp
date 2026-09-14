#!/usr/bin/env python3
"""analytics 组合优化单测（纯 numpy/pandas/scipy，不联网）。

锁住 2026-09-14 的回归：`np.sqrt(DataFrame).values` 是只读视图，
`np.fill_diagonal(dist.values, 0)` 会抛 "underlying array is read-only" ——
HRP 作为默认方法因此从未跑通过。
"""

import json
import random

import pytest

analytics = pytest.importorskip("analytics")


def _bars(n=130, seed=0, drift=0.0):
    rnd = random.Random(seed)
    px, out = 10.0, []
    for i in range(n):
        px *= (1 + rnd.uniform(-0.02, 0.02) + drift)
        out.append({"date": "2026-01-%02d" % (1 + i % 28), "open": px, "high": px * 1.01,
                    "low": px * 0.99, "close": px, "volume": 1e6})
    return out


def _payload(k=5):
    syms = ["sh6005%02d" % i for i in range(k)]
    return syms, {s: _bars(seed=i) for i, s in enumerate(syms)}


def test_hrp_runs_and_weights_sum_to_one():
    syms, kl = _payload(5)
    out = json.loads(analytics.portfolio_optimize(syms, kl, "hrp", 120))
    assert "error" not in out, out
    w = out.get("weights") or {}
    assert set(w) == set(syms)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in w.values()), "long-only"


def test_equal_and_min_variance_also_run():
    syms, kl = _payload(4)
    for m in ("equal", "min_variance"):
        out = json.loads(analytics.portfolio_optimize(syms, kl, m, 120))
        assert "error" not in out, (m, out)
        assert abs(sum((out.get("weights") or {}).values()) - 1.0) < 1e-6


def test_insufficient_samples_reports_error():
    syms, kl = _payload(3)
    tiny = {s: kl[s][:10] for s in syms}          # 只有 10 天 → 收益样本不足
    out = json.loads(analytics.portfolio_optimize(syms, tiny, "hrp", 120))
    assert "error" in out
