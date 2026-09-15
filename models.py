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

        2026-09-15 修订：本引擎是 **Phase 0 占位实现**，只认 roc_5 / volume_ratio /
        std_20 / mfv 四个入参。此前缺输入时会恒返回 ``signal=0.0``、
        ``confidence=0.5``，而 MCP 工具 ``predict`` 把它原样透出 ——
        调用方无从分辨那是一个真实模型输出还是"什么都没算"（实测就是这个现象）。
        现在：
        - 缺必需输入 → ``status="insufficient_input"``，signal/confidence 为 ``None``
        - 任何情况下都带 ``is_mock=True`` / ``method="phase0_mock"``
        - 真实推理用 ``ml_predict``（需 klines_list）
        """
        symbol = factors.get("symbol", "")
        factor_values = factors.get("factors", {}) or {}

        required = ("roc_5", "volume_ratio")
        missing = [k for k in required if k not in factor_values]
        if missing:
            return {
                "symbol": symbol,
                "signal": None,
                "confidence": None,
                "status": "insufficient_input",
                "is_mock": True,
                "method": "phase0_mock",
                "missing": missing,
                "error": "Phase-0 占位引擎缺少必需输入 %s；它只有 roc_5/volume_ratio/"
                         "std_20/mfv 四个入参，真实推理请用 ml_predict（需 klines_list）"
                         % missing,
            }

        # Phase 0: Simple heuristic-based mock prediction.
        # In later phases this will be replaced with an ONNX model.
        signal = self._mock_signal(factor_values)
        confidence = self._mock_confidence(factor_values)

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "status": "ok",
            "is_mock": True,
            "method": "phase0_mock",
            "note": "Phase-0 占位实现，非训练模型输出；真实推理请用 ml_predict",
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
