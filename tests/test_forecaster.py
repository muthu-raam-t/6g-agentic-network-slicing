import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from forecaster import (
    ProbabilisticLSTM,
    make_windows,
    gaussian_nll_loss,
    train_forecaster,
    predict,
    evaluate_calibration,
)


def _periodic_series(n=200):
    return [10.0 + 5.0 * math.sin(i / 5.0) for i in range(n)]


def test_model_forward_shape():
    model = ProbabilisticLSTM(hidden_size=8)
    x = torch.randn(4, 10, 1)  # batch=4, window_len=10
    mean, log_var = model(x)
    assert mean.shape == (4,)
    assert log_var.shape == (4,)


def test_gaussian_nll_orders_confident_correct_below_confident_wrong():
    mean_correct = torch.tensor([5.0])
    target = torch.tensor([5.0])
    log_var = torch.tensor([-2.0])  # same (small) variance both times
    mean_wrong = torch.tensor([50.0])

    good_loss = gaussian_nll_loss(mean_correct, log_var, target)
    bad_loss = gaussian_nll_loss(mean_wrong, log_var, target)
    assert good_loss.item() < bad_loss.item()


def test_make_windows_shapes():
    series = list(range(30))
    X, y = make_windows(series, window_len=5)
    assert X.shape == (25, 5)
    assert y.shape == (25,)
    assert X[0].tolist() == [0, 1, 2, 3, 4]
    assert y[0] == 5


def test_make_windows_raises_on_too_short_series():
    try:
        make_windows([1.0, 2.0, 3.0], window_len=10)
        assert False, "expected ValueError for too-short series"
    except ValueError:
        pass


def test_train_forecaster_loss_decreases_on_a_simple_periodic_series():
    series = _periodic_series(150)
    result = train_forecaster(series, window_len=15, hidden_size=16, epochs=80, lr=1e-2, seed=1)
    assert result.train_losses[-1] < result.train_losses[0]
    assert result.train_losses[-1] < result.train_losses[0] * 0.7  # meaningfully better, not just noise


def test_predict_returns_sane_mean_and_positive_std():
    series = _periodic_series(150)
    result = train_forecaster(series, window_len=15, hidden_size=16, epochs=80, lr=1e-2, seed=1)
    window = series[-15:]
    p = predict(result, window)
    assert "mean" in p and "std" in p
    assert p["std"] > 0
    # a reasonably trained model on an easy periodic series shouldn't be wildly off
    assert abs(p["mean"] - series[-1]) < 20.0


def test_evaluate_calibration_coverage_in_sane_range():
    series = _periodic_series(200)
    result = train_forecaster(series, window_len=15, hidden_size=16, epochs=100, lr=1e-2, seed=1)
    coverage = evaluate_calibration(result, series, window_len=15, z=1.0)
    assert 0.0 <= coverage <= 1.0
    coverage_2std = evaluate_calibration(result, series, window_len=15, z=2.0)
    assert coverage_2std >= coverage  # wider band must never cover less
