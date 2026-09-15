"""
forecaster.py
=============
Stage 3: predicts next-timestep per-slice traffic as a Gaussian (mean, std),
not a point estimate -- see notebooks/00_overview.ipynb, Section 5 and
notebooks/03_forecaster.ipynb for the full design rationale.

No dependency on digital_twin.py -- this module only needs a plain float
sequence (the demand history for one slice) and is agnostic to where that
sequence came from, so it's testable in isolation.

Public API used by notebooks/03_forecaster.ipynb:
    train_forecaster(series, window_len, hidden_size, epochs, lr, seed) -> TrainResult
    predict(result, window) -> {"mean": float, "std": float}
    evaluate_calibration(result, series, window_len, z) -> float (coverage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProbabilisticLSTM(nn.Module):
    """An LSTM that reads a window of past (normalised) values and outputs
    the predictive distribution (mean, log-variance) of the next value, in
    the same normalised space.

    log-variance (not variance directly) is predicted for numerical
    stability -- it can be any real number, so no activation/clipping is
    needed to keep the variance positive; variance is recovered as
    exp(log_var) wherever it's actually used.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 24, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)  # -> [mean, log_var]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, window_len, input_size) -> mean, log_var, each (batch,)"""
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        mean_logvar = self.head(last_hidden)
        return mean_logvar[:, 0], mean_logvar[:, 1]


def gaussian_nll_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood -- see notebooks/00_overview.ipynb,
    Section 5. Minimising this (rather than plain MSE) lets the model learn
    to predict a *wider* variance for genuinely harder-to-predict moments
    (e.g. right before a spike) instead of always being equally confident.
    """
    precision = torch.exp(-log_var)
    return (0.5 * log_var + 0.5 * precision * (target - mean) ** 2).mean()


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def make_windows(series: List[float], window_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a 1-D series into sliding-window (X, y) pairs.
    X[i] = series[i : i+window_len], y[i] = series[i+window_len].

    Raises ValueError if the series is too short to produce even one
    window -- silently returning empty arrays would let a training call
    fail confusingly deep inside the LSTM instead of here.
    """
    if len(series) <= window_len:
        raise ValueError(
            f"series has {len(series)} points, need more than window_len={window_len} to build a single window"
        )
    xs, ys = [], []
    for i in range(len(series) - window_len):
        xs.append(series[i: i + window_len])
        ys.append(series[i + window_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training result + normalization bundle
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    model: ProbabilisticLSTM
    window_len: int
    series_mean: float
    series_std: float
    train_losses: List[float] = field(default_factory=list)


def train_forecaster(
    series: List[float],
    window_len: int = 10,
    hidden_size: int = 24,
    epochs: int = 250,
    lr: float = 0.01,
    seed: int = 42,
) -> TrainResult:
    """Normalises `series` (zero mean, unit std) internally, trains a
    ProbabilisticLSTM by minimising Gaussian NLL, and returns a TrainResult
    bundling the trained model with the normalisation stats needed to map
    predictions back to real units in `predict()`.

    A full-batch training loop is used -- the synthetic series here are
    short enough that mini-batching isn't necessary.
    """
    torch.manual_seed(seed)

    series_arr = np.array(series, dtype=np.float32)
    series_mean = float(series_arr.mean())
    series_std = float(series_arr.std()) or 1.0  # guard against a constant series
    normalized = ((series_arr - series_mean) / series_std).tolist()

    X, y = make_windows(normalized, window_len)
    X_t = torch.tensor(X).unsqueeze(-1)  # (N, window_len, 1)
    y_t = torch.tensor(y)

    model = ProbabilisticLSTM(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        mean, log_var = model(X_t)
        loss = gaussian_nll_loss(mean, log_var, y_t)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

    return TrainResult(
        model=model,
        window_len=window_len,
        series_mean=series_mean,
        series_std=series_std,
        train_losses=train_losses,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(result: TrainResult, window: List[float]) -> dict:
    """Run one forward pass on `window` (raw units, length == result.window_len)
    and return {"mean": float, "std": float} back in the *original* (un-
    normalised) units -- this is what gets handed to
    schemas.SliceState.forecast_mean_mbps / forecast_std_mbps.
    """
    if len(window) != result.window_len:
        raise ValueError(f"window must have length {result.window_len}, got {len(window)}")

    result.model.eval()
    normalized_window = [(v - result.series_mean) / result.series_std for v in window]
    x = torch.tensor(normalized_window, dtype=torch.float32).view(1, -1, 1)
    mean_n, log_var_n = result.model(x)

    mean = float(mean_n.item()) * result.series_std + result.series_mean
    std = float(torch.exp(0.5 * log_var_n).item()) * result.series_std
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Evaluation: calibration, not just point accuracy
# ---------------------------------------------------------------------------

def evaluate_calibration(result: TrainResult, series: List[float], window_len: int = 10, z: float = 1.0) -> float:
    """Rolls `predict()` forward one step at a time over `series` and
    returns the fraction of true next-values landing within
    mean +/- z*std -- i.e. empirical coverage at the given z.

    A point-accuracy metric alone says nothing about whether the predicted
    *uncertainty* is trustworthy; for a well-calibrated model, coverage at
    z=1.0 should land near the Gaussian-implied ~68%, and z=2.0 near ~95%.
    """
    covered = 0
    total = 0
    for i in range(len(series) - window_len):
        window = series[i: i + window_len]
        actual = series[i + window_len]
        p = predict(result, window)
        if abs(actual - p["mean"]) <= z * p["std"]:
            covered += 1
        total += 1

    if total == 0:
        raise ValueError(f"series has {len(series)} points, need more than window_len={window_len}")
    return covered / total
