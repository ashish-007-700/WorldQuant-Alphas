from __future__ import annotations

import numpy as np
import pandas as pd

from wqalpha.backtest import long_short_weights, performance_stats, portfolio_returns
from wqalpha.data import wide
from wqalpha.metrics import alpha_decay, forward_returns, hit_ratio, icir, information_coefficient
from wqalpha.risk import capm, sector_neutralize


def test_metrics_backtest_and_risk(sample_panel) -> None:
    returns = wide(sample_panel, "returns")
    signal = wide(sample_panel, "close").rank(axis=1, pct=True)
    ic = information_coefficient(signal, forward_returns(returns, 1))
    assert np.isfinite(icir(ic)) or np.isnan(icir(ic))
    assert 0 <= hit_ratio(ic) <= 1
    assert len(alpha_decay(signal, returns, 5)) == 5

    weights = long_short_weights(signal)
    strategy = portfolio_returns(weights, returns, transaction_cost_bps=5)
    stats = performance_stats(strategy)
    assert {"sharpe", "sortino", "max_drawdown", "cagr", "volatility", "turnover"}.issubset(stats.index)

    market = returns.mean(axis=1).rename("market")
    exposures = capm(strategy, market)
    assert "beta_mkt" in exposures.index or exposures.empty

    sector_map = pd.Series(
        {"RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Financials", "INFY": "IT", "ITC": "Consumer"}
    )
    neutral = sector_neutralize(signal, sector_map)
    assert neutral.shape == signal.shape
