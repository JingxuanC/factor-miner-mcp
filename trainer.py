"""
RollingTrainer: expanding-window LGBM training for daily stock prediction.

Qlib-inspired workflow:
  1. Receive raw OHLCV klines from Go side (via /train/rolling endpoint)
  2. Internally compute Alpha158/360 factors + label (next-day return) from klines
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
        self.last_train_at: Optional[str] = None
        self.metrics: Dict[str, dict] = {}
        self._factor_engine = FactorEngine() if FactorEngine else None
        self._feature_names: Optional[List[str]] = None  # cached for predict

    # ── feature + label extraction ────────────────────────────────────

    def extract_features(
        self, klines: list, labels: list
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Compute factors ONCE on full window, reuse for all historical rows.

        O(n) instead of O(n²): FactorEngine.compute() called once per stock,
        then all rows share the same feature vector (only labels differ).

        This is the point-in-time approximation — factors are slow-moving enough
        that a single snapshot works for past labels in daily-frequency training.
        """
        if self._factor_engine is None:
            return np.array([]), np.array([]), [], []

        # Compute factors once on the full klines window
        result = self._factor_engine.compute({
            "symbol": "",
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
        feats = result.get("factors", {})
        if not feats:
            return np.array([]), np.array([]), [], []

        if self._feature_names is None:
            self._feature_names = sorted(feats.keys())
        feat_row = [feats.get(k, 0.0) for k in self._feature_names]

        # One row per day with valid label (all share same factor vector)
        X_rows, y_rows, date_rows = [], [], []
        for i in range(len(klines) - 1):  # last row has no label
            lbl = labels[i]
            if math.isnan(lbl) or math.isinf(lbl):
                continue
            X_rows.append(feat_row)
            y_rows.append(lbl)
            date_rows.append(klines[i].get("date", ""))

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
    ) -> dict:
        """Run one expanding-window training cycle.

        Args:
            klines_list: [{symbol: "600519", klines: [{date, open, high, low, close, volume}, ...]}, ...]
            validation_days: tail samples for validation
            early_stopping_rounds: LGBM patience

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
            if len(klines) < 30:
                continue
            labels = compute_labels(klines)
            X, y, dates, _ = self.extract_features(klines, labels)
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

        self.last_train_at = today
        self.metrics[today] = {
            "ic": round(ic, 4),
            "rank_ic": round(rank_ic, 4),
            "sharpe": round(sharpe, 4),
            "top10_return": round(top10_ret, 6),
            "n_samples": len(y),
            "n_features": X.shape[1],
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
            "duration_ms": int((time.time() - t0) * 1000),
        }

    # ── prediction ─────────────────────────────────────────────────────

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

    def get_metrics(self) -> dict:
        return {
            "last_train_at": self.last_train_at,
            "metrics": self.metrics,
            "model_exists": os.path.exists(self.model_path),
            "feature_names": self._feature_names,
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
