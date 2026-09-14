#!/usr/bin/env python3
"""factor_worker 的 Redis 写入单测（不联网、不需要真 Redis）。"""

import json

import factor_worker as fw


class FakeRedis:
    def __init__(self):
        self.store = {}

    def setex(self, key, ttl, val):
        self.store[key] = (ttl, val)


def test_write_daily_factors_lowercases_symbol():
    """写入侧必须把 symbol 归一成小写：athena 用 `dfactor:sh600340` 查，
    而写入侧拿到的 qlib instrument 名是大写 `SH600340`。不归一就是静默失联。"""
    r = FakeRedis()
    n = fw.write_daily_factors_to_redis({"SH600340": {"reversal20": 1.0}}, redis_client=r)
    assert n == 1
    assert "dfactor:sh600340" in r.store
    assert "dfactor:SH600340" not in r.store, "大写键会让 athena 的 MGet 全 miss"
    ttl, val = r.store["dfactor:sh600340"]
    assert ttl == fw.DFACTOR_TTL
    assert json.loads(val)["reversal20"] == 1.0


def test_write_daily_factors_skips_empty_values():
    r = FakeRedis()
    n = fw.write_daily_factors_to_redis(
        {"sh600000": {"f": 1.0}, "sz000001": {}, "SZ000002": None}, redis_client=r)
    assert n == 1
    assert set(r.store) == {"dfactor:sh600000"}


def test_write_daily_factors_returns_zero_without_redis(monkeypatch):
    monkeypatch.setattr(fw, "_get_redis", lambda: None)
    assert fw.write_daily_factors_to_redis({"sh600000": {"f": 1.0}}) == 0


def test_write_daily_factors_survives_one_bad_symbol():
    """单只写失败不阻塞其余（否则一只脏数据就让全市场因子落不了地）。"""
    class Flaky(FakeRedis):
        def setex(self, key, ttl, val):
            if key.endswith("sz000002"):
                raise RuntimeError("boom")
            super().setex(key, ttl, val)

    r = Flaky()
    n = fw.write_daily_factors_to_redis(
        {"sh600000": {"f": 1.0}, "SZ000002": {"f": 2.0}}, redis_client=r)
    assert n == 1 and set(r.store) == {"dfactor:sh600000"}
