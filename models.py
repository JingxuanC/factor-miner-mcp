"""
ModelEngine — Phase 0 mock implementation.

Loads trained models and runs inference. In Phase 0 we return a
mock signal and confidence. Real model loading (ONNX / sklearn)
comes in Phase 4.
"""


class ModelEngine:
    """Loads trained models and runs inference."""

    def predict(self, factors: dict) -> dict:
        """Run ML prediction on computed factors.

        Args:
            factors: {
                "symbol": "600519",
                "factors": {"roc_5": 0.02, "ma_20": 1850.5, ...}
            }

        Returns:
            {
                "symbol": "600519",
                "signal": 0.023,
                "confidence": 0.65
            }
        """
        symbol = factors.get("symbol", "")
        factor_values = factors.get("factors", {})

        # Phase 0: Simple heuristic-based mock prediction.
        # In later phases this will be replaced with an ONNX model.
        signal = self._mock_signal(factor_values)
        confidence = self._mock_confidence(factor_values)

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
        }

    def _mock_signal(self, factor_values: dict) -> float:
        """Derive a mock signal from factor values."""
        roc = factor_values.get("roc_5", 0.0)
        vol_ratio = factor_values.get("volume_ratio", 1.0)
        typical_price = factor_values.get("typical_price", 0.0)

        # Simple mock: signal is a weighted combination of factors
        # Normalised to [-0.1, 0.1] range as a placeholder
        raw = (roc * 0.5) + ((vol_ratio - 1.0) * 0.05)
        return max(-0.1, min(0.1, round(raw, 4)))

    def _mock_confidence(self, factor_values: dict) -> float:
        """Derive a mock confidence score."""
        std = factor_values.get("std_20", 0.0)
        mfv = factor_values.get("mfv", 0.0)
        vol_ratio = factor_values.get("volume_ratio", 1.0)

        # Lower volatility + positive money flow = higher confidence
        vol_penalty = min(std * 5, 0.3)
        flow_bonus = 0.1 if mfv > 0 else 0.0
        vol_bonus = 0.1 if vol_ratio > 1.2 else 0.0

        confidence = 0.5 + flow_bonus + vol_bonus - vol_penalty
        return max(0.0, min(1.0, round(confidence, 2)))
