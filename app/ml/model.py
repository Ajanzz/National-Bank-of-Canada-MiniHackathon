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

# Module-level singleton — loaded once at first use, never reloaded per request
_CLASSIFIER_INSTANCE: "BiasMLClassifier | None" = None


def get_classifier() -> "BiasMLClassifier":
    """Return the module-level cached classifier, loading from disk only once."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = BiasMLClassifier()
    return _CLASSIFIER_INSTANCE


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
        """Given an 18-element feature vector, return per-class probabilities."""
        return self.predict_batch([feature_vector])[0]

    def predict_batch(self, feature_vectors: list[np.ndarray]) -> list[dict[str, float]]:
        """
        Batch prediction: run predict_proba once on the full matrix.
        Dramatically faster than calling predict_bias_probabilities per window.
        """
        if not feature_vectors:
            return []

        X = np.array(feature_vectors, dtype=float)  # (n_windows, 18)

        if self.model is not None:
            probs = self.model.predict_proba(X)  # (n_windows, 4)
            return [
                {label: float(p) for label, p in zip(self.classes, row)}
                for row in probs
            ]

        # ── Heuristic fallback — vectorised ────────────────────────────
        safe = np.where(np.isfinite(X), X, 0.0)
        ncols = X.shape[1]
        tph         = np.clip(safe[:, 0] / 20.0, 0, 1)
        burst       = np.clip(safe[:, 3] / 10.0, 0, 1)       if ncols > 3  else np.full(len(X), 0.3)
        hold_ratio  = np.clip((safe[:, 11] - 1.0) / 3.0, 0, 1) if ncols > 11 else np.full(len(X), 0.3)
        size_ratio  = np.clip((safe[:, 12] - 1.0) / 2.0, 0, 1) if ncols > 12 else np.full(len(X), 0.3)
        reentry     = np.clip(1.0 - safe[:, 13] / 600.0, 0, 1) if ncols > 13 else np.full(len(X), 0.3)
        streak      = np.clip(safe[:, 14] / 5.0, 0, 1)         if ncols > 14 else np.full(len(X), 0.3)

        overtrading   = np.minimum(1.0, 0.3 + tph * 0.5 + burst * 0.3)
        loss_aversion = np.minimum(1.0, 0.2 + hold_ratio * 0.8)
        revenge       = np.minimum(1.0, 0.2 + size_ratio * 0.4 + reentry * 0.3 + streak * 0.2)
        calm          = np.maximum(0.0, 1.0 - overtrading - loss_aversion - revenge)
        total         = overtrading + loss_aversion + revenge + calm
        total         = np.where(total > 0, total, 1.0)
        overtrading  /= total; loss_aversion /= total
        revenge      /= total; calm          /= total

        return [
            {"calm": float(calm[i]), "loss_aversion": float(loss_aversion[i]),
             "overtrading": float(overtrading[i]), "revenge_trading": float(revenge[i])}
            for i in range(len(X))
        ]
