#!/usr/bin/env python3
"""gen_data.py — FactorMiner 数据层生成（DESIGN-FACTOR-MINER.md §3）

从 qlib cn_data 二进制库导出挖掘用日频量价 h5：
  --debug  daily_pv_debug.h5  100 股 × 2018-01-01→2019-12-31（沙箱 debug 用，静态）
  --full   daily_pv_all.h5    全部股票 × 2008-12-29→今（qlib 回测用，run 前增量重生成）

列契约：$open/$close/$high/$low/$volume/$factor，MultiIndex(datetime, instrument)。
移植自 microsoft/RD-Agent scenarios/qlib/experiment/factor_data_template/generate.py（MIT）。

用法:
  python -m factor_miner.gen_data --debug [--out-dir data/factor_mining]
  python -m factor_miner.gen_data --full
"""
import argparse
import sys
from pathlib import Path

FIELDS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
FULL_START = "2008-12-29"
DEBUG_START, DEBUG_END, DEBUG_N = "2018-01-01", "2019-12-31", 100


def _init_qlib(provider_uri: str):
    import qlib  # noqa: PLC0415 — 延迟导入，未装 pyqlib 的环境可 import 本模块

    qlib.init(provider_uri=provider_uri)
    from qlib.data import D  # noqa: PLC0415

    return D


def gen_full(D, out: Path) -> Path:
    data = D.features(D.instruments(), FIELDS, freq="day").swaplevel().sort_index()
    data = data.loc[FULL_START:].sort_index()
    path = out / "daily_pv_all.h5"
    data.to_hdf(path, key="data")
    return path


def gen_debug(D, out: Path) -> Path:
    data = (
        D.features(D.instruments(), FIELDS, start_time=DEBUG_START, end_time=DEBUG_END, freq="day")
        .swaplevel()
        .sort_index()
    )
    instruments = data.reset_index()["instrument"].unique()[:DEBUG_N]
    data = data.swaplevel().loc[instruments].swaplevel().sort_index()
    path = out / "daily_pv_debug.h5"
    data.to_hdf(path, key="data")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="FactorMiner qlib 数据导出")
    ap.add_argument("--debug", action="store_true", help="生成 100 股 debug 集")
    ap.add_argument("--full", action="store_true", help="生成全量数据集")
    ap.add_argument("--provider-uri", default="~/.qlib/qlib_data/cn_data")
    ap.add_argument("--out-dir", default="data/factor_mining")
    args = ap.parse_args()

    if not args.debug and not args.full:
        ap.error("至少指定 --debug 或 --full")

    D = _init_qlib(args.provider_uri)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print(f"[gen_data] debug -> {gen_debug(D, out)}")
    if args.full:
        print(f"[gen_data] full  -> {gen_full(D, out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
