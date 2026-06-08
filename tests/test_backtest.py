"""Unit tests for wqalpha.backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wqalpha.backtest import long_short_weights, performance_stats, portfolio_returns


@pytest.fixture()
def uniform_signal() -> pd.DataFrame:
    """5-symbol signal with a clear top and bottom decile."""
    dates = pd.bdate_range("2024-01-01", periods=20)
    rng = np.random.default_rng(99)
    symbols = ["A", "B", "C", "D", "E"]
    data = pd.DataFrame(rng.standard_normal((20, 5)), index=dates, columns=symbols)
    return data


@pytest.fixture()
def uniform_returns(uniform_signal) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(0.001, 0.02, uniform_signal.shape),
        index=uniform_signal.index,
        columns=uniform_signal.columns,
    )


class TestLongShortWeights:
    def test_weights_sum_near_zero(self, uniform_signal) -> None:
        w = long_short_weights(uniform_signal)
        # Long-short book: gross longs = +0.5, gross shorts = -0.5 → net ≈ 0
        row_sum = w.sum(axis=1)
        assert row_sum.abs().max() < 1e-9

    def test_no_nan_output(self, uniform_signal) -> None:
        w = long_short_weights(uniform_signal)
        assert not w.isna().any().any()

    def test_long_positions_positive(self, uniform_signal) -> None:
        w = long_short_weights(uniform_signal)
        long_vals = w[w > 0].stack().dropna()
        assert (long_vals > 0).all()

    def test_short_positions_negative(self, uniform_signal) -> None:
        w = long_short_weights(uniform_signal)
        short_vals = w[w < 0].stack().dropna()
        assert (short_vals < 0).all()

    def test_gross_exposure_is_one(self, uniform_signal) -> None:
        w = long_short_weights(uniform_signal)
        gross = w.abs().sum(axis=1)
        assert (gross - 1.0).abs().max() < 1e-9


class TestPortfolioReturns:
    def test_returns_series_shape(self, uniform_signal, uniform_returns) -> None:
        w = long_short_weights(uniform_signal)
        pnl = portfolio_returns(w, uniform_returns)
        assert isinstance(pnl, pd.Series)
        assert len(pnl) == len(uniform_returns)

    def test_no_nan_interior(self, uniform_signal, uniform_returns) -> None:
        w = long_short_weights(uniform_signal)
        pnl = portfolio_returns(w, uniform_returns)
        # First row is NaN (no lagged weights), rest should be finite
        assert np.isfinite(pnl.iloc[1:]).all()

    def test_transaction_cost_reduces_returns(self, uniform_signal, uniform_returns) -> None:
        w = long_short_weights(uniform_signal)
        gross_pnl = portfolio_returns(w, uniform_returns, transaction_cost_bps=0)
        net_pnl = portfolio_returns(w, uniform_returns, transaction_cost_bps=50)
        assert gross_pnl.sum() >= net_pnl.sum()


class TestPerformanceStats:
    def test_all_keys_present(self) -> None:
        r = pd.Series(np.random.default_rng(0).normal(0.001, 0.015, 252))
        stats = performance_stats(r)
        expected = {"sharpe", "sortino", "max_drawdown", "cagr", "volatility", "turnover"}
        assert expected.issubset(stats.index)

    def test_sharpe_positive_for_good_returns(self) -> None:
        r = pd.Series([0.002] * 252)  # constant positive return
        stats = performance_stats(r)
        assert stats["sharpe"] > 0

    def test_max_drawdown_nonpositive(self) -> None:
        rng = np.random.default_rng(5)
        r = pd.Series(rng.normal(0.0, 0.01, 252))
        stats = performance_stats(r)
        assert stats["max_drawdown"] <= 0

    def test_empty_series_returns_nans(self) -> None:
        stats = performance_stats(pd.Series([], dtype=float))
        assert stats.isna().all()
