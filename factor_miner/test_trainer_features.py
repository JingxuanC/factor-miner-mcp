#!/usr/bin/env python3
"""特征名解析单测（不需要 lightgbm：用假模型对象）。

锁住 2026-09-14 的事故：`_feature_names` 只在内存里，进程一换就 None →
predict 里每只票都 continue → 返回 {"status":"ok","n_predicted":0}，
**一只都没算却报成功**。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trainer as T  # noqa: E402


class FakeModel:
    def __init__(self, n):
        self.n_features_in_ = n


def _trainer(tmp_path, names=None):
    t = object.__new__(T.RollingTrainer)          # 绕开重构造（不加载引擎/模型）
    t.model_path = str(tmp_path / "lgbm_rolling.pkl")
    t._feature_names = names
    return t


def test_derive_from_engine_keys_when_nothing_persisted(tmp_path):
    """顺序不是猜的：train() 用的就是 sorted(feats.keys())，这里必须逐字一致。"""
    t = _trainer(tmp_path)
    factors = {"b": 1.0, "a": 2.0, "c": 3.0}
    names, err = t._resolve_features(FakeModel(3), factors)
    assert err is None and names == ["a", "b", "c"]


def test_length_mismatch_is_explicit_error(tmp_path):
    t = _trainer(tmp_path)
    names, err = t._resolve_features(FakeModel(64), {"a": 1.0, "b": 2.0})
    assert names is None and "特征数与模型不匹配" in err


def test_sidecar_roundtrip(tmp_path):
    t = _trainer(tmp_path, names=["f1", "f2"])
    t._save_feature_names(["f1", "f2"])
    assert os.path.exists(t._feature_names_path)
    t2 = _trainer(tmp_path)
    assert t2._load_feature_names() == ["f1", "f2"]
    names, err = t2._resolve_features(FakeModel(2), {"f1": 1.0, "f2": 2.0})
    assert names == ["f1", "f2"] and err is None


def test_missing_sidecar_returns_none(tmp_path):
    t = _trainer(tmp_path)
    assert t._load_feature_names() is None
    names, err = t._resolve_features(FakeModel(2), {})
    assert names is None and "无法确定特征名" in err
