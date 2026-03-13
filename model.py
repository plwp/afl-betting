"""Model training: regularized logistic regression + LightGBM + calibrated stack."""

import os
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from config import (
    FEATURE_COLS,
    FEATURE_PATH,
    MODEL_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)


def _clip_probs(values) -> np.ndarray:
    """Keep probabilities away from 0/1 to stabilize log loss and logit math."""
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def _logit(values) -> np.ndarray:
    """Stable logit transform."""
    probs = _clip_probs(values)
    return np.log(probs / (1 - probs))


def temporal_split(df: pd.DataFrame):
    """Split data by year into train/val/test."""
    train = df[df["year"] <= TRAIN_END].copy()
    val = df[(df["year"] >= VAL_START) & (df["year"] <= VAL_END)].copy()
    test = df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END)].copy()
    return train, val, test


def evaluate(y_true, y_prob, label: str, market_prob=None):
    """Print evaluation metrics."""
    y_prob = _clip_probs(y_prob)
    ll = log_loss(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    print(f"\n--- {label} ---")
    print(f"  Log Loss:    {ll:.4f}")
    print(f"  Brier Score: {bs:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    if market_prob is not None:
        market_prob = _clip_probs(market_prob)
        m_ll = log_loss(y_true, market_prob)
        m_bs = brier_score_loss(y_true, market_prob)
        m_acc = accuracy_score(y_true, (market_prob >= 0.5).astype(int))
        print(f"  Market Log Loss:    {m_ll:.4f}")
        print(f"  Market Brier Score: {m_bs:.4f}")
        print(f"  Market Accuracy:    {m_acc:.4f}")
        print(f"  Delta vs Market LL: {m_ll - ll:+.4f}")
    return {"log_loss": ll, "brier_score": bs, "accuracy": acc}


def plot_calibration(y_true, y_prob, label: str, path: str):
    """Save calibration curve plot."""
    frac_pos, mean_pred = calibration_curve(y_true, _clip_probs(y_prob), n_bins=10)
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


@dataclass
class EnsemblePredictor:
    """Final prediction bundle used by both backtest and live scanning."""

    feature_cols: list[str]
    scaler: StandardScaler
    logreg: LogisticRegression
    lgb_model: lgb.LGBMClassifier
    stacker: LogisticRegression

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.feature_cols].copy()
        return pd.DataFrame(X, columns=self.feature_cols)

    def _base_probs(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = self._frame(X)
        lr_prob = self.logreg.predict_proba(self.scaler.transform(frame))[:, 1]
        lgb_prob = self.lgb_model.predict_proba(frame)[:, 1]
        market_prob = frame["market_prob_home"].to_numpy(dtype=float)
        return _clip_probs(lr_prob), _clip_probs(lgb_prob), _clip_probs(market_prob)

    def _stack_features(self, X) -> np.ndarray:
        lr_prob, lgb_prob, market_prob = self._base_probs(X)
        return np.column_stack(
            [
                _logit(lr_prob),
                _logit(lgb_prob),
                _logit(market_prob),
                lgb_prob - market_prob,
                lr_prob - market_prob,
            ]
        )

    def predict_proba(self, X) -> np.ndarray:
        prob = _clip_probs(self.stacker.predict_proba(self._stack_features(X))[:, 1])
        return np.column_stack([1 - prob, prob])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _tune_logreg(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[StandardScaler, LogisticRegression, dict]:
    """Pick logistic regression regularization by validation log loss."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    candidates = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    best_score = float("inf")
    best_model = None
    best_meta = {}

    for c_value in candidates:
        model = LogisticRegression(max_iter=5000, C=c_value, solver="lbfgs")
        model.fit(X_train_sc, y_train)
        val_prob = model.predict_proba(X_val_sc)[:, 1]
        score = log_loss(y_val, _clip_probs(val_prob))
        if score < best_score:
            best_score = score
            best_model = model
            best_meta = {"C": c_value, "val_log_loss": score}

    return scaler, best_model, best_meta


def _tune_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[lgb.LGBMClassifier, dict]:
    """Pick a conservative LightGBM configuration by validation log loss."""
    candidates = [
        {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.2,
            "reg_lambda": 0.5,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.025,
            "max_depth": 5,
            "num_leaves": 23,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.3,
            "reg_lambda": 1.5,
        },
    ]

    best_score = float("inf")
    best_model = None
    best_meta = {}

    for params in candidates:
        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=42,
            verbose=-1,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, _clip_probs(val_prob))
        if score < best_score:
            best_score = score
            best_model = model
            best_meta = dict(params)
            best_meta["val_log_loss"] = score
            best_meta["best_iteration_"] = getattr(model, "best_iteration_", None)

    return best_model, best_meta


def _fit_stacker(
    lr_val_prob: np.ndarray,
    lgb_val_prob: np.ndarray,
    market_val_prob: np.ndarray,
    y_val: np.ndarray,
) -> LogisticRegression:
    """Blend model signals with the market using a calibrated logistic stacker."""
    stack_X = np.column_stack(
        [
            _logit(lr_val_prob),
            _logit(lgb_val_prob),
            _logit(market_val_prob),
            lgb_val_prob - market_val_prob,
            lr_val_prob - market_val_prob,
        ]
    )

    best_score = float("inf")
    best_model = None
    for c_value in [0.05, 0.1, 0.25, 0.5, 1.0]:
        model = LogisticRegression(max_iter=5000, C=c_value, solver="lbfgs")
        model.fit(stack_X, y_val)
        score = log_loss(y_val, _clip_probs(model.predict_proba(stack_X)[:, 1]))
        if score < best_score:
            best_score = score
            best_model = model

    return best_model


def fit_model_bundle(
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> tuple[EnsemblePredictor, dict]:
    """Train base models on one period and fit the stacker on a later calibration fold."""
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["home_win"].to_numpy()
    X_cal = calibration_df[FEATURE_COLS]
    y_cal = calibration_df["home_win"].to_numpy()

    scaler, logreg, logreg_meta = _tune_logreg(X_train, y_train, X_cal, y_cal)
    lgb_model, lgb_meta = _tune_lgbm(X_train, y_train, X_cal, y_cal)

    lr_cal_prob = logreg.predict_proba(scaler.transform(X_cal))[:, 1]
    lgb_cal_prob = lgb_model.predict_proba(X_cal)[:, 1]
    market_cal_prob = calibration_df["market_prob_home"].to_numpy()
    stacker = _fit_stacker(lr_cal_prob, lgb_cal_prob, market_cal_prob, y_cal)

    predictor = EnsemblePredictor(
        feature_cols=list(FEATURE_COLS),
        scaler=scaler,
        logreg=logreg,
        lgb_model=lgb_model,
        stacker=stacker,
    )
    meta = {"logreg": logreg_meta, "lgb": lgb_meta}
    return predictor, meta


def train_models(df: pd.DataFrame = None):
    """Train the model stack, evaluate it, and save artifacts."""
    if df is None:
        df = pd.read_parquet(FEATURE_PATH)

    os.makedirs(MODEL_DIR, exist_ok=True)
    train, val, test = temporal_split(df)
    if train.empty or val.empty or test.empty:
        raise ValueError("Temporal split produced an empty train/val/test segment")

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    predictor, meta = fit_model_bundle(train, val)
    print(f"Selected logistic params: {meta['logreg']}")
    print(f"Selected LightGBM params: {meta['lgb']}")

    X_val = val[FEATURE_COLS]
    X_test = test[FEATURE_COLS]
    y_val = val["home_win"].to_numpy()
    y_test = test["home_win"].to_numpy()
    market_val = val["market_prob_home"].to_numpy()
    market_test = test["market_prob_home"].to_numpy()

    lr_val_prob, lgb_val_prob, _ = predictor._base_probs(X_val)
    lr_test_prob, lgb_test_prob, _ = predictor._base_probs(X_test)
    ens_val_prob = predictor.predict_proba(X_val)[:, 1]
    ens_test_prob = predictor.predict_proba(X_test)[:, 1]

    market_val_metrics = evaluate(y_val, market_val, "Market (Val)")
    market_test_metrics = evaluate(y_test, market_test, "Market (Test)")
    logreg_val_metrics = evaluate(y_val, lr_val_prob, "LogReg (Val)", market_val)
    logreg_test_metrics = evaluate(y_test, lr_test_prob, "LogReg (Test)", market_test)
    lgb_val_metrics = evaluate(y_val, lgb_val_prob, "LightGBM (Val)", market_val)
    lgb_test_metrics = evaluate(y_test, lgb_test_prob, "LightGBM (Test)", market_test)
    ensemble_val_metrics = evaluate(y_val, ens_val_prob, "Ensemble (Val)", market_val)
    ensemble_test_metrics = evaluate(y_test, ens_test_prob, "Ensemble (Test)", market_test)

    plot_calibration(
        y_test,
        market_test,
        "Market",
        os.path.join(MODEL_DIR, "calibration_market.png"),
    )
    plot_calibration(
        y_test,
        lgb_test_prob,
        "LightGBM Raw",
        os.path.join(MODEL_DIR, "calibration_lgb_raw.png"),
    )
    plot_calibration(
        y_test,
        ens_test_prob,
        "Ensemble",
        os.path.join(MODEL_DIR, "calibration_ensemble.png"),
    )

    importance = pd.Series(
        predictor.lgb_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nFeature Importance (LightGBM):")
    print(importance.to_string())

    joblib.dump(predictor.scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(predictor.logreg, os.path.join(MODEL_DIR, "logreg.pkl"))
    joblib.dump(predictor.lgb_model, os.path.join(MODEL_DIR, "lgb_raw.pkl"))
    joblib.dump(predictor, os.path.join(MODEL_DIR, "lgb_calibrated.pkl"))
    joblib.dump(predictor, os.path.join(MODEL_DIR, "model_bundle.pkl"))
    print(f"\nModels saved to {MODEL_DIR}/")

    return {
        "predictor": predictor,
        "metrics": {
            "market_val": market_val_metrics,
            "market_test": market_test_metrics,
            "logreg_val": logreg_val_metrics,
            "logreg_test": logreg_test_metrics,
            "lgb_val": lgb_val_metrics,
            "lgb_test": lgb_test_metrics,
            "ensemble_val": ensemble_val_metrics,
            "ensemble_test": ensemble_test_metrics,
        },
    }


if __name__ == "__main__":
    train_models()
