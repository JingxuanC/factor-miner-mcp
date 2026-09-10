"""
RollingTrainer: expanding-window LGBM training for daily stock prediction.

Qlib-inspired workflow:
  1. Receive raw OHLCV klines from Go side (via /train/rolling endpoint)
  2. Point-in-time feature extraction: for each day i, recompute Alpha158
     factors on klines[:i+1] — the exact same snapshot predict() would see
     at close of day i; label is the next-day return
  3. Each training cycle: train on expanding window, validate on tail
  4. Save model to disk, produce next-day predictions → Redis pred:{sym}

No historical factor storage needed — compute on the fly like Qlib.
"""

import json, time, os, pickle, math
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    # OSError: libomp 缺失时 dlopen 失败不是 ImportError（macOS 常见）
    HAS_LGB = False

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

# Import factor engine
try:
    from factors import FactorEngine
except ImportError:
    FactorEngine = None


# ── label computation ───────────────────────────────────────────────────

def compute_labels(klines: list) -> list:
    """Compute next-day return label for each kline.

    label[i] = close[i+1] / close[i] - 1
    Last row has no label (no T+1), returns NaN.
    """
    labels = []
    for i in range(len(klines) - 1):
        today = float(klines[i].get("close", 0))
        tomorrow = float(klines[i + 1].get("close", 0))
        if today <= 0:
            labels.append(float("nan"))
        else:
            labels.append((tomorrow / today) - 1.0)
    labels.append(float("nan"))  # last day
    return labels


# ── rolling trainer ──────────────────────────────────────────────────────

class RollingTrainer:
    """Expanding-window LGBM training orchestrated from Go side.

    Data flow:
      Go sends klines → Python computes factors + labels → LGBM train →
      model.pkl → predict → Redis pred:{sym}
    """

    def __init__(self, redis_client=None, model_dir="/tmp/athena_models"):
        self.redis = redis_client
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "lgbm_rolling.pkl")
        self.master_path = os.path.join(model_dir, "master.pt")
        self.last_train_at: Optional[str] = None
        self.metrics: Dict[str, dict] = {}
        self.master_last_train_at: Optional[str] = None
        self.master_metrics: Dict[str, dict] = {}
        self._factor_engine = FactorEngine() if FactorEngine else None
        self._feature_names: Optional[List[str]] = None  # cached for predict

    # ── feature + label extraction ────────────────────────────────────

    def extract_features(
        self, klines: list, labels: list, min_history: int = 60, step: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Point-in-time extraction: one feature row per day.

        For each day i (i >= min_history-1), factors are recomputed on the
        prefix klines[:i+1] — identical semantics to what predict() sees at
        the close of day i (no look-ahead). Costs O(n²) per stock instead of
        O(n); step > 1 subsamples days to trade sample count for speed.
        """
        if self._factor_engine is None:
            return np.array([]), np.array([]), [], []

        norm = [
            {"date": k.get("date", ""),
             "open": float(k.get("open", 0)),
             "high": float(k.get("high", 0)),
             "low": float(k.get("low", 0)),
             "close": float(k.get("close", 0)),
             "volume": float(k.get("volume", 0))}
            for k in klines
        ]

        feat_names = self._feature_names
        X_rows, y_rows, date_rows = [], [], []
        for i in range(max(min_history - 1, 1), len(klines) - 1, max(step, 1)):
            lbl = labels[i]
            if math.isnan(lbl) or math.isinf(lbl):
                continue
            feats = self._factor_engine.compute(
                {"symbol": "", "klines": norm[:i + 1]}
            ).get("factors", {})
            if not feats:
                continue
            if feat_names is None:
                feat_names = sorted(feats.keys())
            X_rows.append([feats.get(k, 0.0) for k in feat_names])
            y_rows.append(lbl)
            date_rows.append(klines[i].get("date", ""))

        if self._feature_names is None and feat_names:
            self._feature_names = feat_names
        if not X_rows:
            return np.array([]), np.array([]), [], []
        return (
            np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.float32),
            date_rows,
            [""] * len(date_rows),
        )

    # ── training ──────────────────────────────────────────────────────

    def train(
        self,
        klines_list: List[dict],  # [{symbol, klines: [...]}, ...]
        validation_days: int = 20,
        early_stopping_rounds: int = 20,
        min_history: int = 60,
        step: int = 1,
    ) -> dict:
        """Run one expanding-window training cycle.

        Args:
            klines_list: [{symbol: "600519", klines: [{date, open, high, low, close, volume}, ...]}, ...]
            validation_days: tail samples for validation
            early_stopping_rounds: LGBM patience
            min_history: 首日所需最短历史（ma_60 需要 60；更小则前期因子失真）
            step: 按日抽样步长（>1 时减少样本换速度）

        Returns:
            {status, ic, rank_ic, sharpe, n_samples, n_features, trained_at}
        """
        if not HAS_LGB:
            return {"status": "error", "message": "lightgbm not installed"}

        t0 = time.time()
        today = time.strftime("%Y-%m-%d")

        # Collect all training rows across all symbols, split per-stock by time
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        for item in klines_list:
            sym = item.get("symbol", "")
            klines = item.get("klines", [])
            if len(klines) < max(30, min_history + 2):
                continue
            labels = compute_labels(klines)
            X, y, dates, _ = self.extract_features(
                klines, labels, min_history=min_history, step=step)
            if len(X) == 0:
                continue
            # Per-stock split: last validation_days rows → val, rest → train
            vd = min(validation_days, max(5, len(y) // 5))
            if len(y) > vd:
                all_X_train.append(X[:-vd])
                all_y_train.append(y[:-vd])
                all_X_val.append(X[-vd:])
                all_y_val.append(y[-vd:])

        if not all_X_train:
            return {"status": "error", "message": "no training data extracted"}

        X = np.vstack(all_X_train).astype(np.float32)
        y = np.concatenate(all_y_train).astype(np.float32)
        X_val = np.vstack(all_X_val).astype(np.float32) if all_X_val else X[-10:]
        y_val = np.concatenate(all_y_val).astype(np.float32) if all_y_val else y[-10:]

        if len(y) < 100:
            return {"status": "error", "message": f"insufficient data: {len(y)} samples"}

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        callbacks = [lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=callbacks,
        )

        # Evaluate
        pred_val = model.predict(X_val)
        ic = float(np.corrcoef(pred_val, y_val)[0, 1]) if len(pred_val) > 1 else 0
        rank_ic = float(np.corrcoef(np.argsort(pred_val), np.argsort(y_val))[0, 1]) if len(pred_val) > 1 else 0
        top10_idx = np.argsort(pred_val)[-10:]
        top10_ret = float(np.mean(y_val[top10_idx])) if len(top10_idx) > 0 else 0
        sharpe = (top10_ret / (np.std(y_val[top10_idx]) + 1e-8)) * np.sqrt(252) if len(top10_idx) > 1 else 0

        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)

        top_feats = []
        if self._feature_names:
            imp = sorted(zip(self._feature_names, model.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
            top_feats = [{"feature": n, "importance": int(v)} for n, v in imp[:20]]

        self.last_train_at = today
        self.metrics[today] = {
            "ic": round(ic, 4),
            "rank_ic": round(rank_ic, 4),
            "sharpe": round(sharpe, 4),
            "top10_return": round(top10_ret, 6),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "feature_importance_top20": top_feats,
        }

        return {
            "status": "ok",
            "trained_at": today,
            "model_path": self.model_path,
            "ic": round(ic, 4),
            "rank_ic": round(rank_ic, 4),
            "sharpe": round(sharpe, 4),
            "top10_return": round(top10_ret, 6),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "feature_importance_top20": top_feats,
            "duration_ms": int((time.time() - t0) * 1000),
        }

    # ── prediction ─────────────────────────────────────────────────────

    @staticmethod
    def _norm_klines(klines: list) -> list:
        return [
            {"date": k.get("date", ""),
             "open": float(k.get("open", 0)),
             "high": float(k.get("high", 0)),
             "low": float(k.get("low", 0)),
             "close": float(k.get("close", 0)),
             "volume": float(k.get("volume", 0))}
            for k in klines
        ]

    def predict(self, klines_list: List[dict]) -> dict:
        """Generate next-day predictions from latest klines.

        Uses Redis factor:{sym} (pre-computed by batch_compute) to avoid
        recomputing factors here. Falls back to on-the-fly factor computation.
        """
        if not HAS_LGB:
            return {"status": "error", "message": "lightgbm not installed"}
        if not os.path.exists(self.model_path):
            return {"status": "error", "message": "no trained model found"}

        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        results = {}
        for item in klines_list:
            sym = item.get("symbol", "")
            # Prefer Redis factor snapshot (latest, written by batch_compute)
            factors = None
            if self.redis:
                raw = self.redis.get(f"factor:{sym}")
                if raw:
                    factors = json.loads(raw)

            # Fallback: compute on the fly
            if not factors:
                klines = item.get("klines", [])
                if not klines or self._factor_engine is None:
                    continue
                result = self._factor_engine.compute({
                    "symbol": sym,
                    "klines": [
                        {"date": k.get("date", ""),
                         "open": float(k.get("open", 0)),
                         "high": float(k.get("high", 0)),
                         "low": float(k.get("low", 0)),
                         "close": float(k.get("close", 0)),
                         "volume": float(k.get("volume", 0))}
                        for k in klines
                    ],
                })
                factors = result.get("factors", {})

            if not factors or self._feature_names is None:
                continue

            X_row = np.array([[factors.get(k, 0.0) for k in self._feature_names]], dtype=np.float32)
            pred = float(model.predict(X_row)[0])

            if self.redis:
                self.redis.setex(
                    f"pred:{sym}", 86400,
                    json.dumps({"score": pred, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                )
            results[sym] = round(pred, 6)

        return {"status": "ok", "n_predicted": len(results), "predictions": results}

    # ── MASTER（深度学习后端，截面 Transformer）─────────────────────────
    #
    # 与 LGBM 路径的差异：MASTER 是截面模型——每个 batch 是「同一交易日的
    # 全部股票」，SAttention 在股票间做注意力，label 在截面上 zscore。
    # 市场上下文（gate 输入）取自当日截面特征的 mean/std（2*F 维），
    # 替代原实现的 63 维 qlib 市场信息特征。

    @staticmethod
    def _build_days(feats_by_sym: dict, mean: np.ndarray, std: np.ndarray,
                    seq_len: int, split: str) -> dict:
        """date -> [(seq (seq_len,F) float32, label, symbol), ...]"""
        days: Dict[str, list] = {}
        for sym, (X, y, dates, vd) in feats_by_sym.items():
            n = len(y)
            lo, hi = (0, n - vd) if split == "train" else (n - vd, n)
            lo = max(lo, seq_len - 1)
            if hi - lo < 1:
                continue
            Xn = ((X - mean) / std).astype(np.float32)
            for j in range(lo, hi):
                days.setdefault(dates[j], []).append(
                    (Xn[j - seq_len + 1:j + 1], float(y[j]), sym))
        return days

    @staticmethod
    def _day_batch(items: list, n_feat: int, seq_len: int):
        """一个交易日的截面 batch: (N, T, 3F)；gate 段只填最后一行。"""
        n = len(items)
        xb = np.zeros((n, seq_len, 3 * n_feat), dtype=np.float32)
        yb = np.zeros(n, dtype=np.float32)
        last = np.stack([it[0][-1] for it in items])
        ctx = np.concatenate([last.mean(axis=0), last.std(axis=0)]).astype(np.float32)
        for k, (seq, lbl, _sym) in enumerate(items):
            xb[k, :, :n_feat] = seq
            xb[k, -1, n_feat:] = ctx
            yb[k] = lbl
        return xb, yb

    def train_master(
        self,
        klines_list: List[dict],
        validation_days: int = 20,
        min_history: int = 60,
        step: int = 1,
        seq_len: int = 8,
        epochs: int = 3,
        d_model: int = 64,
        lr: float = 3e-4,
        dropout: float = 0.5,
        t_nhead: int = 4,
        s_nhead: int = 2,
        beta: float = 5.0,
        max_symbols: int = 50,
        seed: int = 42,
    ) -> dict:
        """Train MASTER on the same point-in-time features as train().

        返回形状与 train() 兼容，另加 model_type/seq_len/epochs/d_model 等。
        """
        if not HAS_TORCH:
            return {"status": "error",
                    "message": "torch not installed（MASTER 后端需要 torch CPU 版，见 Dockerfile）"}
        if len(klines_list) > max_symbols:
            return {"status": "error",
                    "message": f"too many symbols: {len(klines_list)} > max_symbols={max_symbols}（CPU 保护上限）"}
        if self._factor_engine is None:
            return {"status": "error", "message": "factor engine unavailable"}
        from factor_miner.master_nn import MASTER

        t0 = time.time()
        today = time.strftime("%Y-%m-%d")
        torch.set_num_threads(2)  # 2 核服务器保护
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1) point-in-time 特征（复用 LGBM 路径的 extract_features）
        feats_by_sym = {}
        for item in klines_list:
            sym = item.get("symbol", "")
            klines = item.get("klines", [])
            if len(klines) < max(30, min_history + seq_len + 2):
                continue
            labels = compute_labels(klines)
            X, y, dates, _ = self.extract_features(
                klines, labels, min_history=min_history, step=step)
            if len(X) < seq_len + 10:
                continue
            vd = min(validation_days, max(5, len(y) // 5))
            if len(y) <= vd:
                continue
            feats_by_sym[sym] = (X, y, dates, vd)

        if len(feats_by_sym) < 4:
            return {"status": "error",
                    "message": f"MASTER 是截面模型，至少需 4 只有效股票（实际 {len(feats_by_sym)}）"}

        feat_names = list(self._feature_names or [])
        n_feat = len(feat_names)

        # 2) 特征归一化：统计量只用训练段（不用验证段，避免前视）
        train_rows = np.vstack([X[:-vd] for X, _y, _d, vd in feats_by_sym.values()])
        mean = train_rows.mean(axis=0).astype(np.float32)
        std = train_rows.std(axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0

        # 3) 按交易日组织截面序列
        train_days = self._build_days(feats_by_sym, mean, std, seq_len, "train")
        val_days = self._build_days(feats_by_sym, mean, std, seq_len, "valid")
        day_keys = [d for d in sorted(train_days) if len(train_days[d]) >= 4]
        if not day_keys:
            return {"status": "error", "message": "no cross-sectional day with >= 4 stocks"}
        n_train_samples = sum(len(train_days[d]) for d in day_keys)

        # 4) 训练（batch = 一个交易日的截面）
        model = MASTER(d_feat=n_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead,
                       T_dropout_rate=dropout, S_dropout_rate=dropout,
                       gate_input_start_index=n_feat, gate_input_end_index=3 * n_feat,
                       beta=beta)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        last_losses: List[float] = []
        for _epoch in range(epochs):
            model.train()
            last_losses = []
            for k in np.random.permutation(len(day_keys)):
                xb, yb = self._day_batch(train_days[day_keys[k]], n_feat, seq_len)
                feature = torch.from_numpy(xb)
                label = torch.from_numpy(yb)
                n = label.shape[0]
                if n >= 40:
                    p = int(0.025 * n)
                    idx = label.sort().indices[p:n - p]  # 截面双侧各去 2.5% 极端值
                else:
                    idx = torch.arange(n)
                lbl = label[idx]
                lbl = (lbl - lbl.mean()) / (lbl.std() + 1e-8)  # 截面 zscore
                pred = model(feature[idx])
                loss = torch.mean((pred - lbl) ** 2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                opt.step()
                last_losses.append(loss.item())

        # 5) 验证尾段（与 LGBM 路径相同的 pooled 指标口径）
        model.eval()
        preds, labels_v = [], []
        with torch.no_grad():
            for d in sorted(val_days):
                xb, yb = self._day_batch(val_days[d], n_feat, seq_len)
                preds.append(model(torch.from_numpy(xb)).numpy())
                labels_v.append(yb)
        pred_val = np.concatenate(preds)
        y_val = np.concatenate(labels_v)
        ic = float(np.corrcoef(pred_val, y_val)[0, 1]) if len(pred_val) > 1 else 0
        rank_ic = float(np.corrcoef(np.argsort(pred_val), np.argsort(y_val))[0, 1]) if len(pred_val) > 1 else 0
        top10_idx = np.argsort(pred_val)[-10:]
        top10_ret = float(np.mean(y_val[top10_idx])) if len(top10_idx) > 0 else 0
        sharpe = (top10_ret / (np.std(y_val[top10_idx]) + 1e-8)) * np.sqrt(252) if len(top10_idx) > 1 else 0

        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "feature_names": feat_names,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "seq_len": seq_len,
                "d_feat": n_feat,
                "d_model": d_model,
                "t_nhead": t_nhead,
                "s_nhead": s_nhead,
                "dropout": dropout,
                "beta": beta,
            },
        }, self.master_path)

        metrics = {
            "ic": round(ic, 4),
            "rank_ic": round(rank_ic, 4),
            "sharpe": round(sharpe, 4),
            "top10_return": round(top10_ret, 6),
            "n_samples": n_train_samples,
            "n_features": n_feat,
            "model_type": "master",
            "seq_len": seq_len,
            "epochs": epochs,
            "d_model": d_model,
            "n_symbols": len(feats_by_sym),
            "n_train_days": len(day_keys),
            "train_loss_last": round(last_losses[-1], 6) if last_losses else None,
        }
        self.master_last_train_at = today
        self.master_metrics[today] = metrics

        return {
            "status": "ok",
            "trained_at": today,
            "model_path": self.master_path,
            "model_type": "master",
            "ic": round(ic, 4),
            "rank_ic": round(rank_ic, 4),
            "sharpe": round(sharpe, 4),
            "top10_return": round(top10_ret, 6),
            "n_samples": n_train_samples,
            "n_features": n_feat,
            "feature_importance_top20": [],  # MASTER 无特征重要性，占位保持形状兼容
            "seq_len": seq_len,
            "epochs": epochs,
            "d_model": d_model,
            "n_symbols": len(feats_by_sym),
            "n_train_days": len(day_keys),
            "train_loss_last": metrics["train_loss_last"],
            "duration_ms": int((time.time() - t0) * 1000),
        }

    def predict_master(self, klines_list: List[dict]) -> dict:
        """Score with the trained MASTER: 每只股票取最近 seq_len 天特征序列。"""
        if not HAS_TORCH:
            return {"status": "error",
                    "message": "torch not installed（MASTER 后端需要 torch CPU 版，见 Dockerfile）"}
        if not os.path.exists(self.master_path):
            return {"status": "error", "message": "no trained master model found"}
        if self._factor_engine is None:
            return {"status": "error", "message": "factor engine unavailable"}
        from factor_miner.master_nn import MASTER

        torch.set_num_threads(2)
        try:
            ckpt = torch.load(self.master_path, map_location="cpu", weights_only=True)
        except TypeError:  # 老版本 torch 无 weights_only 参数
            ckpt = torch.load(self.master_path, map_location="cpu")
        cfg = ckpt["config"]
        feat_names = cfg["feature_names"]
        n_feat = cfg["d_feat"]
        seq_len = cfg["seq_len"]
        mean = np.array(cfg["mean"], dtype=np.float32)
        std = np.array(cfg["std"], dtype=np.float32)

        model = MASTER(d_feat=n_feat, d_model=cfg["d_model"],
                       t_nhead=cfg["t_nhead"], s_nhead=cfg["s_nhead"],
                       T_dropout_rate=cfg["dropout"], S_dropout_rate=cfg["dropout"],
                       gate_input_start_index=n_feat, gate_input_end_index=3 * n_feat,
                       beta=cfg["beta"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        seqs, syms = [], []
        for item in klines_list:
            sym = item.get("symbol", "")
            klines = item.get("klines", [])
            if len(klines) < seq_len:
                continue
            norm = self._norm_klines(klines)
            rows = []
            for i in range(len(klines) - seq_len, len(klines)):
                feats = self._factor_engine.compute(
                    {"symbol": sym, "klines": norm[:i + 1]}).get("factors", {})
                if not feats:
                    break
                rows.append([feats.get(k, 0.0) for k in feat_names])
            if len(rows) < seq_len:
                continue
            seqs.append(((np.array(rows, dtype=np.float32) - mean) / std).astype(np.float32))
            syms.append(sym)

        if not seqs:
            return {"status": "error", "message": "no symbol has enough history for seq_len"}

        n = len(seqs)
        last = np.stack([s[-1] for s in seqs])
        ctx = np.concatenate([last.mean(axis=0), last.std(axis=0)]).astype(np.float32)
        xb = np.zeros((n, seq_len, 3 * n_feat), dtype=np.float32)
        for k, s in enumerate(seqs):
            xb[k, :, :n_feat] = s
            xb[k, -1, n_feat:] = ctx
        with torch.no_grad():
            pred = model(torch.from_numpy(xb)).numpy()

        results = {}
        for sym, p in zip(syms, pred):
            score = round(float(p), 6)
            if self.redis:
                self.redis.setex(
                    f"pred:{sym}", 86400,
                    json.dumps({"score": score, "model": "master",
                                "at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                )
            results[sym] = score

        return {"status": "ok", "model_type": "master",
                "n_predicted": len(results), "predictions": results}

    def get_metrics(self) -> dict:
        return {
            "last_train_at": self.last_train_at,
            "metrics": self.metrics,
            "model_exists": os.path.exists(self.model_path),
            "feature_names": self._feature_names,
            "model_type": "lgbm",
            "master": {
                "model_exists": os.path.exists(self.master_path),
                "last_train_at": self.master_last_train_at,
                "metrics": self.master_metrics,
                "model_type": "master",
            },
        }


# ── singleton ────────────────────────────────────────────────────────────

_trainer: Optional[RollingTrainer] = None

def get_trainer(redis_client=None) -> RollingTrainer:
    global _trainer
    if _trainer is None:
        _trainer = RollingTrainer(
            redis_client=redis_client,
            model_dir=os.environ.get("FACTOR_MINER_MODEL_DIR", "/tmp/athena_models"),
        )
    elif redis_client is not None and _trainer.redis is None:
        _trainer.redis = redis_client
    return _trainer
