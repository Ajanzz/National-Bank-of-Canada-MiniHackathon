"""
Train an XGBoost multi-class classifier for trading bias detection.

Reads the four synthetic trader CSVs from training_data/, engineers
features over sliding windows, trains with 5-fold cross-validation,
and saves the model to saved_models/bias_xgb.joblib.

Usage:
    python -m app.ml.train_xgboost
    python -m app.ml.train_xgboost --window-size 50 --stride 25 --folds 5

Run generate_datasets.py first if training_data/ is empty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    _DEPS_OK = True
except ImportError as _err:
    _DEPS_OK = False
    _DEPS_ERR = str(_err)

from app.ml.features import extract_windowed_features, FEATURE_NAMES

DATA_DIR = Path(__file__).resolve().parent / "training_data"
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_PATH = MODEL_DIR / "bias_xgb.joblib"
REPORT_PATH = MODEL_DIR / "training_report.json"

LABEL_MAP: dict[str, str] = {
    "calm_trader.csv": "calm",
    "overtrader.csv": "overtrading",
    "loss_averse_trader.csv": "loss_aversion",
    "revenge_trader.csv": "revenge_trading",
}


def load_and_featurize(
    window_size: int = 50,
    stride: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the 4 training CSVs, extract windowed features, return (X, y)."""
    all_features: list[np.ndarray] = []
    all_labels: list[str] = []

    for filename, label in LABEL_MAP.items():
        csv_path = DATA_DIR / filename
        if not csv_path.exists():
            print(f"  WARNING: {filename} not found — skipping. Run generate_datasets.py first.")
            continue

        df = pd.read_csv(csv_path)
        print(f"  {filename}: {len(df)} trades → label='{label}'")

        windows = extract_windowed_features(df, window_size=window_size, stride=stride)
        all_features.extend(windows)
        all_labels.extend([label] * len(windows))
        print(f"    → {len(windows)} windows")

    if not all_features:
        raise RuntimeError("No training data found. Run: python -m app.ml.generate_datasets")

    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"\n  Total: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Class distribution: { {lbl: int((y == lbl).sum()) for lbl in sorted(set(y))} }")
    return X, y


def train(window_size: int = 50, stride: int = 25, n_folds: int = 5) -> None:
    """Train XGBoost, evaluate with CV, save joblib model."""
    if not _DEPS_OK:
        print(f"ERROR: Missing dependencies: {_DEPS_ERR}")
        print("Run: pip install xgboost scikit-learn joblib")
        return

    print("Loading and featurizing training data...")
    X, y = load_and_featurize(window_size=window_size, stride=stride)

    # Encode labels as integers
    classes = sorted(set(y))
    label_to_int = {lbl: i for i, lbl in enumerate(classes)}
    y_encoded = np.array([label_to_int[lbl] for lbl in y])

    print(f"\nClasses: {classes}")

    # ── Model ──────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # ── Cross-validation ───────────────────────────────────────────────
    print(f"\nRunning {n_folds}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    # ── Full fit ───────────────────────────────────────────────────────
    print("\nFitting on full dataset...")
    model.fit(X, y_encoded)

    # Attach class labels to model for easy decoding at inference time
    model.bias_classes_ = classes  # type: ignore[attr-defined]

    # ── Save ───────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # ── Classification report ──────────────────────────────────────────
    y_pred = model.predict(X)
    report = classification_report(y_encoded, y_pred, target_names=classes, output_dict=True)
    print("\nIn-sample classification report:")
    print(classification_report(y_encoded, y_pred, target_names=classes))

    # Save training report
    training_report = {
        "window_size": window_size,
        "stride": stride,
        "n_folds": n_folds,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "classification_report": report,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(training_report, f, indent=2)
    print(f"Training report saved → {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost bias classifier")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    train(window_size=args.window_size, stride=args.stride, n_folds=args.folds)
