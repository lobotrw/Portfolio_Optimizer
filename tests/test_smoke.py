"""Smoke tests: upload a price CSV (via a patched uploader) and run every optimization method.
Uses the bundled offline fixture tests/fixtures/prices.csv (no network)."""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
HARNESS = str(HERE / "app_with_upload.py")


def run(method=None, bounds=None):
    at = AppTest.from_file(HARNESS, default_timeout=300)
    at.run()
    if method:
        at.radio(key="visibility").set_value(method).run()
        if bounds:
            at.slider[0].set_value(bounds).run()
        at.button[0].click().run()
    return at


def test_upload_page_runs():
    at = run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("method", ["Sharpe", "Min Volatility", "Omega"])
def test_each_method_runs(method):
    at = run(method)
    assert not at.exception, [e.message for e in at.exception]
    weights = at.dataframe[0].value
    assert abs(float(weights.sum().iloc[0]) - 1) < 1e-3


def test_infeasible_bounds_show_error_not_crash():
    at = run("Sharpe", (0.3, 0.65))          # 5 assets x 0.30 min = 1.5 > 1
    assert not at.exception
    assert any("infeasible" in e.value for e in at.error)


def test_max_sharpe_with_only_losing_assets_falls_back_to_min_vol(monkeypatch):
    # every asset has a negative mean return in this fixture, so max-Sharpe is impossible
    # (PyPortfolioOpt: "at least one of the assets must have an expected return exceeding the risk-free rate")
    monkeypatch.setenv("PO_FIXTURE", "prices_all_negative.csv")
    at = run("Sharpe")
    assert not at.exception, [e.message for e in at.exception]
    assert any("Falling back" in w.value for w in at.warning)
    assert abs(float(at.dataframe[0].value.sum().iloc[0]) - 1) < 1e-3


def test_performance_table_is_out_of_sample_by_default():
    at = run("Min Volatility")
    assert not at.exception
    assert any("out-of-sample" in c.value for c in at.caption)
    cols = list(at.table[0].value.columns)
    assert cols[0].startswith("Optimised (test") and "Equal weight (same period)" in cols


def test_weights_do_not_depend_on_test_period_prices(monkeypatch, tmp_path):
    """No lookahead: scrambling the prices after the training cut must not change the weights."""
    import pandas as pd
    src = pd.read_csv(HERE / "fixtures" / "prices.csv")
    cut = int(len(src) * 0.70)
    alt = src.copy()
    alt.iloc[cut:, 1:] = alt.iloc[cut:, 1:].to_numpy()[::-1] * 3.0   # wildly different "future"
    path = tmp_path / "alt.csv"
    alt.to_csv(path, index=False)
    w_orig = run("Min Volatility").dataframe[0].value
    monkeypatch.setenv("PO_FIXTURE", str(path))
    w_alt = run("Min Volatility").dataframe[0].value
    pd.testing.assert_frame_equal(w_orig, w_alt)
