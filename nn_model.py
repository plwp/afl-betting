"""Neural net training in an isolated process (avoids OpenMP clash with LightGBM).

This module is designed to be run as a subprocess via `python3 nn_model.py`.
It reads training data from a temp file, trains the MLP with team embeddings,
and writes predictions back to a temp file.

Can also be imported directly in a process that does NOT import lightgbm.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from config import ALL_TEAMS, FEATURE_COLS, TEAM_EMBED_DIM


def _clip_probs(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def _get_team_indices(df: pd.DataFrame):
    team_to_idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    home_idx = df["home_team"].map(team_to_idx).fillna(0).astype(int).values
    away_idx = df["away_team"].map(team_to_idx).fillna(0).astype(int).values
    return home_idx, away_idx


class NeuralNetPredictor:
    """MLP with team embeddings. sklearn-compatible predict_proba()."""

    def __init__(self, n_features: int, n_teams: int, embed_dim: int = TEAM_EMBED_DIM,
                 lr: float = 1e-3, epochs: int = 200, batch_size: int = 256):
        self.n_features = n_features
        self.n_teams = n_teams
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()

        input_dim = n_features + 2 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )
        self.home_embed = nn.Embedding(n_teams, embed_dim)
        self.away_embed = nn.Embedding(n_teams, embed_dim)

        all_params = (
            list(self.mlp.parameters())
            + list(self.home_embed.parameters())
            + list(self.away_embed.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=1e-4)

    def _forward(self, X_t, home_t, away_t):
        h_emb = self.home_embed(home_t)
        a_emb = self.away_embed(away_t)
        return self.mlp(torch.cat([X_t, h_emb, a_emb], dim=1))

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        X_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        home_idx, away_idx = _get_team_indices(train_df)

        X_t = torch.from_numpy(X_scaled)
        y_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
        home_t = torch.from_numpy(home_idx.astype(np.int64))
        away_t = torch.from_numpy(away_idx.astype(np.int64))

        loss_fn = nn.BCELoss()
        best_val_loss = float("inf")
        patience_counter = 0

        self.mlp.train()
        self.home_embed.train()
        self.away_embed.train()

        n = len(X_t)
        bs = self.batch_size
        for epoch in range(self.epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                pred = self._forward(X_t[idx], home_t[idx], away_t[idx])
                loss = loss_fn(pred, y_t[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if val_df is not None and (epoch + 1) % 5 == 0:
                val_prob = self._predict_raw(val_df[FEATURE_COLS], val_df)
                val_loss = log_loss(val_df["home_win"].values, _clip_probs(val_prob))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 4:
                        break

    def _predict_raw(self, X: pd.DataFrame, full_df: pd.DataFrame) -> np.ndarray:
        self.mlp.eval()
        self.home_embed.eval()
        self.away_embed.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X).astype(np.float32)
            home_idx, away_idx = _get_team_indices(full_df)
            X_t = torch.from_numpy(X_scaled)
            home_t = torch.from_numpy(home_idx.astype(np.int64))
            away_t = torch.from_numpy(away_idx.astype(np.int64))
            probs = self._forward(X_t, home_t, away_t).numpy().flatten()
        self.mlp.train()
        self.home_embed.train()
        self.away_embed.train()
        return probs

    def predict_proba(self, X: pd.DataFrame, full_df: pd.DataFrame) -> np.ndarray:
        probs = _clip_probs(self._predict_raw(X, full_df))
        return np.column_stack([1 - probs, probs])


def _subprocess_main():
    """Entry point when run as subprocess from model.py.

    Protocol:
      stdin args: train_path val_path output_path predict_paths...
      - Reads train/val parquet, trains NN
      - Writes calibration probs to output_path
      - For each predict_path pair (input, output), writes predictions
    """
    args = sys.argv[1:]
    train_path = args[0]
    val_path = args[1]
    output_path = args[2]
    predict_pairs = []
    i = 3
    while i < len(args):
        predict_pairs.append((args[i], args[i + 1]))
        i += 2

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["home_win"].to_numpy()
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["home_win"].to_numpy()

    nn_pred = NeuralNetPredictor(n_features=len(FEATURE_COLS), n_teams=len(ALL_TEAMS))
    nn_pred.fit(X_train, y_train, train_df, val_df=val_df)

    # Calibration probabilities
    cal_probs = nn_pred.predict_proba(X_val, val_df)[:, 1]
    val_ll = log_loss(y_val, _clip_probs(cal_probs))

    # Save calibration probs + meta
    result = {
        "cal_probs": cal_probs.tolist(),
        "val_log_loss": val_ll,
    }

    # Predict on additional datasets
    for input_path, _ in predict_pairs:
        pred_df = pd.read_parquet(input_path)
        probs = nn_pred.predict_proba(pred_df[FEATURE_COLS], pred_df)[:, 1]
        result[input_path] = probs.tolist()

    with open(output_path, "w") as f:
        json.dump(result, f)

    print(f"NN val_log_loss={val_ll:.4f}", file=sys.stderr)


if __name__ == "__main__":
    _subprocess_main()
