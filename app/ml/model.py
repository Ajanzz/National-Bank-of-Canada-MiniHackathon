"""
XGBoost-based bias classifier wrapper.

Loads the trained model from saved_models/bias_xgb.joblib.
Falls back to heuristic probabilities if no model file exists yet.

Based on the BiasLens QHacks 2026 ML pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_PATH = MODEL_DIR / "bias_xgb.joblib"

# 4 bias classes the model is trained on
BIAS_CLASSES: list[str] = ["calm", "loss_aversion", "overtrading", "revenge_trading"]


class BiasMLClassifier:
    """XGBoost 4-class bias classifier.

    Detects: calm, loss_aversion, overtrading, revenge_trading.
    Falls back to heuristic probabilities when no saved model exists.
    """

    def __init__(self) -> None:
        self.model = None
        self.classes: list[str] = list(BIAS_CLASSES)

        try:
            import joblib  # type: ignore[import]

            if MODEL_PATH.exists():
                self.model = joblib.load(MODEL_PATH)
                # Restore class order saved during training; coerce to plain Python str
                if hasattr(self.model, "bias_classes_"):
                    self.classes = [str(c) for c in self.model.bias_classes_]
        except Exception:
            pass  # Model unavailable — heuristic fallback will be used

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    def predict_bias_probabilities(self, feature_vector: np.ndarray) -> dict[str, float]:
        """
        Given an 18-element feature vector, return per-class probabilities.

        Returns a dict like:
            {"calm": 0.12, "overtrading": 0.65, "loss_aversion": 0.15, "revenge_trading": 0.08}

        When no trained model exists, uses lightweight heuristic formulas.
        """
        if self.model is not None:
            x = np.asarray(feature_vector, dtype=float).reshape(1, -1)
            probs = self.model.predict_proba(x)[0]
            return {label: float(p) for label, p in zip(self.classes, probs)}

        # ── Heuristic fallback ──────────────────────────────────────────
        # Normalize first 18 features into [0, 1] range using rough maxima
        fv = np.asarray(feature_vector, dtype=float)
        safe = np.where(np.isfinite(fv), fv, 0.0)

        # Feature indices (must match FEATURE_NAMES order in features.py)
        # 0=trades_per_hour, 3=burst_count_60s, 11=loss_hold_ratio,
        # 12=avg_size_after_loss_ratio, 13=reentry_after_loss_mean_sec, 14=max_streak
        tph = float(np.clip(safe[0] / 20.0, 0, 1)) if len(safe) > 0 else 0.3
        burst = float(np.clip(safe[3] / 10.0, 0, 1)) if len(safe) > 3 else 0.3
        hold_ratio = float(np.clip((safe[11] - 1.0) / 3.0, 0, 1)) if len(safe) > 11 else 0.3
        size_ratio = float(np.clip((safe[12] - 1.0) / 2.0, 0, 1)) if len(safe) > 12 else 0.3
        reentry = float(np.clip(1.0 - safe[13] / 600.0, 0, 1)) if len(safe) > 13 else 0.3
        streak = float(np.clip(safe[14] / 5.0, 0, 1)) if len(safe) > 14 else 0.3

        overtrading = float(min(1.0, 0.3 + tph * 0.5 + burst * 0.3))
        loss_aversion = float(min(1.0, 0.2 + hold_ratio * 0.8))
        revenge = float(min(1.0, 0.2 + size_ratio * 0.4 + reentry * 0.3 + streak * 0.2))
        calm = float(max(0.0, 1.0 - overtrading - loss_aversion - revenge))

        # Re-normalise so they sum to 1
        total = overtrading + loss_aversion + revenge + calm
        if total > 0:
            overtrading /= total
            loss_aversion /= total
            revenge /= total
            calm /= total

        return {
            "calm": calm,
            "loss_aversion": loss_aversion,
            "overtrading": overtrading,
            "revenge_trading": revenge,
        }
