"""qlib_dump_bin.py — 从 microsoft/qlib v0.9.6 scripts/dump_bin.py 裁剪移植（MIT License）。

原文件: https://github.com/microsoft/qlib/blob/v0.9.6/scripts/dump_bin.py
裁剪说明:
  - 仅保留 DumpDataBase + DumpDataUpdate（增量更新是本管线唯一用到的模式）
  - 去掉 fire / tqdm / loguru 依赖（容器内未安装 fire；进度改用标准 logging）
  - 逻辑逐行保持 upstream 语义，未改任何对齐/追加行为

CSV 契约（供调用方）:
  - 目录下每个文件名为 instrument 小写（sh600519.csv），或 CSV 内含 symbol 列
  - 必须含 date 列；调用方务必传 exclude_fields="date,symbol"，
    否则 symbol 列会被当作行情字段写成 symbol.day.bin 而报错
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Iterable, List, Union

import logging

import numpy as np
import pandas as pd

try:
    from qlib.utils import code_to_fname, fname_to_code
except ImportError:  # 本地单测环境无 pyqlib；cn 数据命名下与 upstream 等价
    def fname_to_code(fname: str):
        prefix = "_qlib_"
        if fname.startswith(prefix):
            fname = fname.lstrip(prefix)
        return fname

    def code_to_fname(code: str):
        return code


logger = logging.getLogger("dump_bin")


class DumpDataBase:
    INSTRUMENTS_START_FIELD = "start_datetime"
    INSTRUMENTS_END_FIELD = "end_datetime"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    DAILY_FORMAT = "%Y-%m-%d"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        csv_path = Path(csv_path).expanduser()
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        self.symbol_field_name = symbol_field_name
        self.csv_files = sorted(csv_path.glob(f"*{self.file_suffix}") if csv_path.is_dir() else [csv_path])
        if limit_nums is not None:
            self.csv_files = self.csv_files[: int(limit_nums)]
        self.qlib_dir = Path(qlib_dir).expanduser()

        self.freq = freq
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT

        self.works = max_workers
        self.date_field_name = date_field_name

        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        self._calendars_list = []

        self._mode = self.ALL_MODE
        self._kwargs = {}

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self, file_or_df: [Path, pd.DataFrame], *, is_begin_end: bool = False, as_set: bool = False
    ) -> Iterable[pd.Timestamp]:
        if not isinstance(file_or_df, pd.DataFrame):
            df = self._get_source_data(file_or_df)
        else:
            df = file_or_df
        if df.empty or self.date_field_name not in df.columns.tolist():
            _calendars = pd.Series(dtype=np.float32)
        else:
            _calendars = df[self.date_field_name]

        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            return _calendars.min(), _calendars.max()
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(str(file_path.resolve()), low_memory=False)
        df[self.date_field_name] = df[self.date_field_name].astype(str).astype("datetime64[ns]")
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        return (
            self._include_fields
            if self._include_fields
            else set(df_columns) - set(self._exclude_fields) if self._exclude_fields else df_columns
        )

    @staticmethod
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ],
        )

        return df

    def save_calendars(self, calendars_data: list):
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        calendars_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        result_calendars_list = [self._format_datetime(x) for x in calendars_data]
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve())
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [self.symbol_field_name, self.INSTRUMENTS_START_FIELD, self.INSTRUMENTS_END_FIELD]
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.symbol_field_name] = instruments_data[self.symbol_field_name].apply(
                lambda x: fname_to_code(x.lower()).upper()
            )
            instruments_data.to_csv(instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False)
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        # calendars
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype("datetime64[ns]")
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        # align index
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        r_df = df.reindex(cal_df.index)
        return r_df

    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        return calendar_list.index(df.index.min())

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        if df.empty:
            logger.warning("%s data is None or empty", features_dir.name)
            return
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        # align index
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning("%s data is not in calendars", features_dir.name)
            return
        # used when creating a bin file
        date_index = self.get_datetime_index(_df, calendar_list)
        for field in self.get_dump_fields(_df.columns):
            bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
            if field not in _df.columns:
                continue
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                # update
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            else:
                # append; self._mode == self.ALL_MODE or not bin_path.exists()
                np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def _dump_bin(self, file_or_data: [Path, pd.DataFrame], calendar_list: List[pd.Timestamp]):
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(str(file_or_data.iloc[0][self.symbol_field_name]).lower())
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        else:
            raise ValueError(f"not support {type(file_or_data)}")
        if df is None or df.empty:
            logger.warning("%s data is None or empty", code)
            return

        # try to remove dup rows or it will cause exception when reindex.
        df = df.drop_duplicates(self.date_field_name)

        # features save dir
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        self._data_to_bin(df, calendar_list, features_dir)

    def dump(self):
        raise NotImplementedError("dump not implemented!")

    def __call__(self, *args, **kwargs):
        self.dump()


class DumpDataUpdate(DumpDataBase):
    """增量更新：日历并集扩展 + 存量股票 append + 新股票新建 bin。

    调用方必须保证每只存量股票的 CSV 覆盖其 instrument end 之后的全部新日历日
    （停牌日补 NaN 行），否则 append 行数与新日历长度错位。本管线由
    update_data.py 负责该不变量。"""

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        super().__init__(
            csv_path,
            qlib_dir,
            freq,
            max_workers,
            date_field_name,
            file_suffix,
            symbol_field_name,
            exclude_fields,
            include_fields,
        )
        self._mode = self.UPDATE_MODE
        self._old_calendar_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        # NOTE: all.txt only exists once for each stock
        self._update_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )  # type: dict

        # load all csv files
        self._all_data = self._load_all_source_data()  # type: pd.DataFrame
        self._new_calendar_list = self._old_calendar_list + sorted(
            filter(lambda x: x > self._old_calendar_list[-1], self._all_data[self.date_field_name].unique())
        )

    def _load_all_source_data(self):
        # NOTE: Need more memory
        logger.info("start load all source data (%d files)....", len(self.csv_files))
        all_df = []

        def _read_csv(file_path: Path):
            _df = pd.read_csv(file_path, parse_dates=[self.date_field_name])
            if self.symbol_field_name not in _df.columns:
                _df[self.symbol_field_name] = self.get_symbol_from_file(file_path)
            return _df

        with ThreadPoolExecutor(max_workers=self.works) as executor:
            done = 0
            for df in executor.map(_read_csv, self.csv_files):
                if not df.empty:
                    all_df.append(df)
                done += 1
                if done % 1000 == 0:
                    logger.info("load source data %d/%d", done, len(self.csv_files))

        logger.info("end of load all data.")
        return pd.concat(all_df, sort=False)

    def _dump_features(self):
        logger.info("start dump features......")
        error_code = {}
        with ProcessPoolExecutor(max_workers=self.works) as executor:
            futures = {}
            for _code, _df in self._all_data.groupby(self.symbol_field_name):
                _code = fname_to_code(str(_code).lower()).upper()
                _start, _end = self._get_date(_df, is_begin_end=True)
                if not (isinstance(_start, pd.Timestamp) and isinstance(_end, pd.Timestamp)):
                    continue
                if _code in self._update_instruments:
                    # exists stock, will append data
                    _update_calendars = (
                        _df[_df[self.date_field_name] > self._update_instruments[_code][self.INSTRUMENTS_END_FIELD]][
                            self.date_field_name
                        ]
                        .sort_values()
                        .to_list()
                    )
                    if _update_calendars:
                        self._update_instruments[_code][self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                        futures[executor.submit(self._dump_bin, _df, _update_calendars)] = _code
                else:
                    # new stock
                    _dt_range = self._update_instruments.setdefault(_code, dict())
                    _dt_range[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_start)
                    _dt_range[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                    futures[executor.submit(self._dump_bin, _df, self._new_calendar_list)] = _code

            done = 0
            for _future in as_completed(futures):
                try:
                    _future.result()
                except Exception:
                    error_code[futures[_future]] = traceback.format_exc()
                done += 1
                if done % 1000 == 0:
                    logger.info("dump features %d/%d", done, len(futures))
            if error_code:
                logger.warning("dump bin errors: %s", error_code)

        logger.info("end of features dump.")

    def dump(self):
        self.save_calendars(self._new_calendar_list)
        self._dump_features()
        df = pd.DataFrame.from_dict(self._update_instruments, orient="index")
        df.index.names = [self.symbol_field_name]
        self.save_instruments(df.reset_index())
