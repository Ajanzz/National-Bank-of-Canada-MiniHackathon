"""
Train an XGBoost multi-class classifier for trading bias detection.

Improvements over v1:
  - Multi-seed data generation: each class is generated with 3 seeds and pooled,
    preventing the model from memorising seed-specific artefacts.
  - Smaller stride (10) for far more training windows per CSV.
  - Early stopping on a 10% validation split of the training set so n_estimators
    is chosen by the data, not hardcoded.
  - Stronger regularisation: reg_alpha, reg_lambda, gamma, lower max_depth.
  - Optional --tune flag: uses Optuna to find the best hyperparameters before
    final training (requires: pip install optuna).

Usage:
    python -m app.ml.train_xgboost
    python -m app.ml.train_xgboost --window-size 50 --stride 10 --folds 5
    python -m app.ml.train_xgboost --tune --tune-trials 50
    python -m app.ml.train_xgboost --zip path/to/trading_datasets.zip
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    _DEPS_OK = True
except ImportError as _err:
    _DEPS_OK = False
    _DEPS_ERR = str(_err)

from app.ml.features import extract_windowed_features, FEATURE_NAMES
from app.ml.generate_datasets import GENERATORS, _DEFAULT_SEEDS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # NBH-backend/
DEFAULT_ZIP = REPO_ROOT / "trading_datasets.zip"
MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_PATH = MODEL_DIR / "bias_xgb.joblib"
REPORT_PATH = MODEL_DIR / "training_report.json"

LABEL_MAP: dict[str, str] = {
    "calm_trader.csv": "calm",
    "overtrader.csv": "overtrading",
    "loss_averse_trader.csv": "loss_aversion",
    "revenge_trader.csv": "revenge_trading",
}

# Best params found via Optuna (updated by --tune).  Used as the default when
# --tune is not passed so you get good results without the search overhead.
_BEST_PARAMS: dict = {
    "n_estimators":    600,
    "max_depth":       4,
    "learning_rate":   0.03,
    "subsample":       0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha":       0.1,
    "reg_lambda":      1.5,
    "gamma":           0.05,
}


def _load_zip(zip_path: Path, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Load features from a zip of 4 labelled CSVs (single-seed legacy path)."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        available = {Path(n).name: n for n in zf.namelist() if n.endswith(".csv")}
        print(f"  Found in zip: {list(available.keys())}")

        for filename, label in LABEL_MAP.items():
            if filename not in available:
                print(f"  WARNING: {filename} not in zip — skipping.")
                continue
            with zf.open(available[filename]) as f:
                df = pd.read_csv(io.BytesIO(f.read()))
            print(f"  {filename}: {len(df)} rows → label='{label}'")
            windows = extract_windowed_features(df, window_size=window_size, stride=stride)
            all_features.extend(windows)
            all_labels.extend([label] * len(windows))
            print(f"    → {len(windows)} windows")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    return X, y


def load_and_featurize(
    zip_path: Path,
    window_size: int = 50,
    stride: int = 10,
    n_trades: int = 5000,
    seeds: dict[str, list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate each class with multiple seeds, extract windowed features, and
    return (X, y).  Multi-seed generation prevents the model from memorising
    seed-specific price/timing artefacts.

    Falls back to the zip CSVs for any class whose generator is unavailable.
    """
    if seeds is None:
        seeds = _DEFAULT_SEEDS

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []

    for filename, label in LABEL_MAP.items():
        fn = GENERATORS.get(filename)
        if fn is None:
            print(f"  WARNING: no generator for {filename} — skipping.")
            continue

        class_seeds = seeds.get(filename, [42])
        frames: list[pd.DataFrame] = []
        for seed in class_seeds:
            frames.append(fn(n_trades=n_trades, seed=seed))
        df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        print(f"  {filename}: {len(df)} rows ({len(class_seeds)} seeds × {n_trades}) → label='{label}'")
        windows = extract_windowed_features(df, window_size=window_size, stride=stride)
        all_features.extend(windows)
        all_labels.extend([label] * len(windows))
        print(f"    → {len(windows)} windows")

    if not all_features:
        print("  No generators available — falling back to zip.")
        return _load_zip(zip_path, window_size, stride)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    print(f"\n  Total: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Class distribution: { {lbl: int((y == lbl).sum()) for lbl in sorted(set(y))} }")
    return X, y


def _make_model(classes: list[str], params: dict | None = None, early_stopping_rounds: int | None = None) -> "XGBClassifier":
    p = {**_BEST_PARAMS, **(params or {})}
    kwargs: dict = dict(
        **p,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**kwargs)


def _tune(
    X: np.ndarray,
    y_encoded: np.ndarray,
    classes: list[str],
    n_trials: int = 50,
    n_folds: int = 3,
) -> dict:
    """Run Optuna TPE search and return the best hyperparameter dict."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed — skipping HPO. Run: pip install optuna")
        return {}

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
        }
        scores = cross_val_score(
            _make_model(classes, params), X, y_encoded,
            cv=cv, scoring="accuracy", n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best CV accuracy: {study.best_value:.3%}")
    print(f"  Best params: {best}")
    return best


def train(
    zip_path: Path = DEFAULT_ZIP,
    window_size: int = 50,
    stride: int = 25,
    n_folds: int = 5,
    n_trades: int = 5000,
    tune: bool = False,
    tune_trials: int = 50,
) -> None:
    """
    1. Generate data from all 4 classes using multiple seeds each.
    2. Stratified 80/20 split.
    3. Train on 80% with early stopping on a 10% validation split → honest test report.
    4. Optional Optuna HPO on full data.
    5. Retrain on 100% with the chosen params → save production model.
    """
    if not _DEPS_OK:
        print(f"ERROR: Missing dependencies: {_DEPS_ERR}")
        print("Run: pip install xgboost scikit-learn joblib")
        return

    print(f"Generating and featurizing (n_trades={n_trades}/class/seed, stride={stride})...")
    X, y = load_and_featurize(zip_path, window_size=window_size, stride=stride, n_trades=n_trades)

    # Encode labels as integers
    classes = sorted(set(y))
    label_to_int = {lbl: i for i, lbl in enumerate(classes)}
    y_encoded = np.array([label_to_int[lbl] for lbl in y])
    print(f"\nClasses: {classes}")

    # ── Optional Optuna HPO ────────────────────────────────────────────
    hpo_params: dict = {}
    if tune:
        print(f"\nRunning Optuna HPO ({tune_trials} trials, {n_folds}-fold CV)...")
        hpo_params = _tune(X, y_encoded, classes, n_trials=tune_trials, n_folds=n_folds)

    best_params = {**_BEST_PARAMS, **hpo_params}

    # ── 80/20 stratified split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
    )
    print(f"\n80/20 split → train: {len(X_train)}, test: {len(X_test)}")

    # Inner 10% of train as early-stopping validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=42
    )

    # ── Train on 80% with early stopping ──────────────────────────────
    print("Training on 80% (with early stopping on 10% val split)...")
    model_eval = _make_model(classes, {**best_params, "n_estimators": 1000}, early_stopping_rounds=30)
    model_eval.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_n = model_eval.best_iteration + 1
    print(f"  Early stopping chose n_estimators={best_n}")

    # ── Evaluate on held-out 20% ───────────────────────────────────────
    y_pred = model_eval.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    print("\nHeld-out test set classification report (20%):")
    print(classification_report(y_test, y_pred, target_names=classes))

    # ── Cross-validation on full data ──────────────────────────────────
    print(f"Running {n_folds}-fold CV on full dataset (n_estimators={best_n})...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        _make_model(classes, {**best_params, "n_estimators": best_n}),
        X, y_encoded, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    print(f"  CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    # ── Final model: retrain on 100% of data ──────────────────────────
    print("\nRetraining on 100% of data for production model...")
    model_final = _make_model(classes, {**best_params, "n_estimators": best_n})
    model_final.fit(X, y_encoded)
    model_final.bias_classes_ = classes  # type: ignore[attr-defined]

    # ── Save ───────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_final, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    training_report = {
        "trained_from": "multi-seed generators",
        "window_size": window_size,
        "stride": stride,
        "n_trades_per_class": n_trades,
        "n_estimators_final": best_n,
        "n_folds": n_folds,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "total_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "hyperparameters": {**best_params, "n_estimators": best_n},
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "classification_report": report,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(training_report, f, indent=2)
    print(f"Training report saved → {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost bias classifier")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="Path to trading_datasets.zip (legacy fallback)")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25, help="Feature extraction stride (default: 25)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trades", type=int, default=5000, help="Trades per class per seed (default: 5000)")
    parser.add_argument("--tune", action="store_true", help="Run Optuna HPO before final training (requires: pip install optuna)")
    parser.add_argument("--tune-trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()

    train(
        zip_path=args.zip,
        window_size=args.window_size,
        stride=args.stride,
        n_folds=args.folds,
        n_trades=args.trades,
        tune=args.tune,
        tune_trials=args.tune_trials,
    )
