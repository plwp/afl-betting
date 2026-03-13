"""Model training: logistic regression + LightGBM with calibration."""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.frozen import FrozenEstimator
import lightgbm as lgb

from config import (
    FEATURE_PATH, MODEL_DIR, FEATURE_COLS,
    TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)


def temporal_split(df: pd.DataFrame):
    """Split data by year into train/val/test."""
    train = df[df["year"] <= TRAIN_END].copy()
    val = df[(df["year"] >= VAL_START) & (df["year"] <= VAL_END)].copy()
    test = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END)].copy()
    return train, val, test


def evaluate(y_true, y_prob, label: str, market_prob=None):
    """Print evaluation metrics."""
    ll = log_loss(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    print(f"\n--- {label} ---")
    print(f"  Log Loss:    {ll:.4f}")
    print(f"  Brier Score: {bs:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    if market_prob is not None:
        m_ll = log_loss(y_true, market_prob)
        m_bs = brier_score_loss(y_true, market_prob)
        m_acc = accuracy_score(y_true, (market_prob >= 0.5).astype(int))
        print(f"  Market Log Loss:    {m_ll:.4f}")
        print(f"  Market Brier Score: {m_bs:.4f}")
        print(f"  Market Accuracy:    {m_acc:.4f}")
    return {"log_loss": ll, "brier_score": bs, "accuracy": acc}


def plot_calibration(y_true, y_prob, label: str, path: str):
    """Save calibration curve plot."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, "s-", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration: {label}")
    ax.legend()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration plot saved: {path}")


def train_models(df: pd.DataFrame = None):
    """Train logistic regression and LightGBM, calibrate, evaluate."""
    if df is None:
        df = pd.read_parquet(FEATURE_PATH)

    os.makedirs(MODEL_DIR, exist_ok=True)
    train, val, test = temporal_split(df)

    X_train = train[FEATURE_COLS]
    y_train = train["home_win"].values
    X_val = val[FEATURE_COLS]
    y_val = val["home_win"].values
    X_test = test[FEATURE_COLS]
    y_test = test["home_win"].values

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_sc, y_train)

    lr_val_prob = lr.predict_proba(X_val_sc)[:, 1]
    lr_test_prob = lr.predict_proba(X_test_sc)[:, 1]

    evaluate(y_val, lr_val_prob, "LogReg (Val)", val["market_prob_home"].values)
    evaluate(y_test, lr_test_prob, "LogReg (Test)", test["market_prob_home"].values)

    # --- Ensemble Model ---
    # We use a simple average of calibrated LogReg and calibrated LightGBM
    from sklearn.ensemble import VotingClassifier

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        random_state=42,
        verbose=-1,
    )

    # Use CalibratedClassifierCV with CV (default 5-fold)
    # This is more stable as it uses more data for calibration
    cal_lr = CalibratedClassifierCV(lr, method="sigmoid", cv=5)
    cal_lgb = CalibratedClassifierCV(lgb_model, method="sigmoid", cv=5)

    # Combine X_train and X_val for a larger calibration/training set
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Scale for LogReg
    X_train_full_sc = scaler.fit_transform(X_train_full)
    X_test_sc = scaler.transform(X_test)

    print("Training Calibrated LogReg...")
    cal_lr.fit(X_train_full_sc, y_train_full)
    print("Training Calibrated LightGBM...")
    cal_lgb.fit(X_train_full, y_train_full)

    lr_prob = cal_lr.predict_proba(X_test_sc)[:, 1]
    lgb_prob = cal_lgb.predict_proba(X_test)[:, 1]
    
    # Ensemble: 70% LogReg, 30% LightGBM (LogReg is more robust here)
    ensemble_prob = 0.7 * lr_prob + 0.3 * lgb_prob
    
    evaluate(y_test, lr_prob, "Calibrated LogReg (Test)", test["market_prob_home"].values)
    evaluate(y_test, lgb_prob, "Calibrated LightGBM (Test)", test["market_prob_home"].values)
    evaluate(y_test, ensemble_prob, "Ensemble (Test)", test["market_prob_home"].values)

    # Calibration plots
    plot_calibration(y_test, ensemble_prob, "Ensemble",
                     os.path.join(MODEL_DIR, "calibration_ensemble.png"))

    # --- Save ---
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(cal_lr, os.path.join(MODEL_DIR, "logreg_cal.pkl"))
    joblib.dump(cal_lgb, os.path.join(MODEL_DIR, "lgb_cal.pkl"))
    print(f"\nModels saved to {MODEL_DIR}/")

    return {
        "scaler": scaler,
        "logreg_cal": cal_lr,
        "lgb_cal": cal_lgb,
        "ensemble_prob": ensemble_prob
    }


if __name__ == "__main__":
    train_models()
