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

    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(0)],
    )

    lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]

    evaluate(y_val, lgb_val_prob, "LightGBM (Val)", val["market_prob_home"].values)
    evaluate(y_test, lgb_test_prob, "LightGBM (Test)", test["market_prob_home"].values)

    # --- Calibration (isotonic on validation set) ---
    # Calibrate LightGBM
    cal_lgb = CalibratedClassifierCV(FrozenEstimator(lgb_model), method="isotonic")
    cal_lgb.fit(X_val, y_val)

    cal_test_prob = cal_lgb.predict_proba(X_test)[:, 1]
    evaluate(y_test, cal_test_prob, "Calibrated LightGBM (Test)",
             test["market_prob_home"].values)

    # Calibration plots
    plot_calibration(y_test, lgb_test_prob, "LightGBM Raw",
                     os.path.join(MODEL_DIR, "calibration_lgb_raw.png"))
    plot_calibration(y_test, cal_test_prob, "LightGBM Calibrated",
                     os.path.join(MODEL_DIR, "calibration_lgb_cal.png"))

    # --- Feature Importance ---
    importance = pd.Series(
        lgb_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nFeature Importance (LightGBM):")
    print(importance.to_string())

    # --- Save ---
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(lr, os.path.join(MODEL_DIR, "logreg.pkl"))
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgb_raw.pkl"))
    joblib.dump(cal_lgb, os.path.join(MODEL_DIR, "lgb_calibrated.pkl"))
    print(f"\nModels saved to {MODEL_DIR}/")

    return {
        "scaler": scaler,
        "logreg": lr,
        "lgb_raw": lgb_model,
        "lgb_calibrated": cal_lgb,
    }


if __name__ == "__main__":
    train_models()
