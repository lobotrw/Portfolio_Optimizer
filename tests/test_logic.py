"""Offline unit tests for the metric / portfolio helpers in functions.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions import portfolio_optimization as po  # noqa: E402


def prices():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame({"BIG": [50000.0, 55000, 60000, 50000], "SMALL": [1.0, 2, 3, 4]}, index=idx)


def test_portfolio_value_is_capital_weighted_not_price_weighted():
    p = prices()
    v = po.portfolio_value(p, {"BIG": 0.5, "SMALL": 0.5})
    assert v.iloc[0] == pytest.approx(1.0)
    # half the capital in each: 0.5 * 50000/50000 + 0.5 * 4/1 = 2.5
    assert v.iloc[-1] == pytest.approx(2.5)
    # the old app method (sum of weight x price level) would be ~flat, dominated by BIG
    old = (p * pd.Series({"BIG": 0.5, "SMALL": 0.5})).sum(axis=1)
    assert old.iloc[-1] / old.iloc[0] == pytest.approx(25002 / 25000.5)


def test_infer_frequency():
    assert po.infer_frequency(pd.date_range("2024-01-01", periods=10, freq="D")) == 365
    assert po.infer_frequency(pd.bdate_range("2024-01-01", periods=10)) == 252


def test_train_test_split_is_chronological_and_non_overlapping():
    p = pd.DataFrame({"A": np.arange(1.0, 101)}, index=pd.date_range("2024-01-01", periods=100))
    train, test = po.train_test_split(p, 0.7)
    assert len(train) == 70 and train.index[-1] == test.index[0]
    assert len(test.pct_change().dropna()) == 30          # 30 new returns, none shared with train
    assert po.train_test_split(p, 1.0)[1] is None


def test_sortino_uses_downside_deviation_over_all_periods():
    s = pd.Series([100.0, 110, 99, 108.9, 98.01])
    r = s.pct_change().dropna()
    dd = np.sqrt((r.clip(upper=0) ** 2).mean())
    assert po.sortino(s, freq=365) == pytest.approx(r.mean() / dd * np.sqrt(365))


def test_sharpe_uses_given_frequency():
    s = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0.001, 0.02, 300))))
    assert po.sharpe(s, 252) / po.sharpe(s, 365) == pytest.approx(np.sqrt(252 / 365))


def test_random_weights_respect_bounds():
    w = po.random_weights(5, 2000, (0.05, 0.65))
    assert len(w) == 2000
    assert (w >= 0.05 - 1e-9).all() and (w <= 0.65 + 1e-9).all()
    assert np.allclose(w.sum(axis=1), 1)
