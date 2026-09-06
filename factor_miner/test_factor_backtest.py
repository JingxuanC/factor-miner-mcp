"""_positions_to_trades 单测：逐日持仓集合差分 → 买卖点标记（不依赖 qlib）。"""

from types import SimpleNamespace

import pandas as pd

from factor_miner.factor_backtest import _positions_to_trades


def _pos(stocks):
    return SimpleNamespace(get_stock_list=lambda: list(stocks))


def test_positions_to_trades_buy_sell_pairs():
    positions = {
        pd.Timestamp("2019-10-08"): _pos({"SH600519", "SZ000001"}),
        pd.Timestamp("2019-10-09"): _pos({"SH600519"}),          # SZ000001 卖出
        pd.Timestamp("2019-10-10"): _pos({"SH600519", "SZ000001"}),  # SZ000001 再次买入
        pd.Timestamp("2019-10-11"): _pos(set()),                  # 全部清仓
    }
    trades = _positions_to_trades(positions)
    assert trades == [
        {"symbol": "SH600519", "date": "2019-10-08", "action": "buy"},
        {"symbol": "SZ000001", "date": "2019-10-08", "action": "buy"},
        {"symbol": "SZ000001", "date": "2019-10-09", "action": "sell"},
        {"symbol": "SZ000001", "date": "2019-10-10", "action": "buy"},
        {"symbol": "SH600519", "date": "2019-10-11", "action": "sell"},
        {"symbol": "SZ000001", "date": "2019-10-11", "action": "sell"},
    ]


def test_positions_to_trades_empty():
    assert _positions_to_trades({}) == []


def test_positions_to_trades_dict_fallback():
    """裸 dict 持仓（amount>0 视为持有）也支持。"""
    positions = {
        pd.Timestamp("2019-10-08"): {"SH600519": {"amount": 100}, "SZ000001": {"amount": 0}},
    }
    trades = _positions_to_trades(positions)
    assert trades == [{"symbol": "SH600519", "date": "2019-10-08", "action": "buy"}]
