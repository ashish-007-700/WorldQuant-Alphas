from __future__ import annotations

import numpy as np
import pandas as pd

from wqalpha.metrics import turnover


def long_short_weights(
    signal: pd.DataFrame,
    long_quantile: float = 0.90,
    short_quantile: float = 0.10,
) -> pd.DataFrame:
    long_cut = signal.quantile(long_quantile, axis=1)
    short_cut = signal.quantile(short_quantile, axis=1)
    longs = signal.ge(long_cut, axis=0)
    shorts = signal.le(short_cut, axis=0)

    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    weights = weights.mask(longs, 1.0)
    weights = weights.mask(shorts, -1.0)
    long_count = longs.sum(axis=1).replace(0, np.nan)
    short_count = shorts.sum(axis=1).replace(0, np.nan)
    weights = weights.where(~longs, weights.div(long_count, axis=0) * 0.5)
    weights = weights.where(~shorts, weights.div(short_count, axis=0) * 0.5)
    return weights.fillna(0.0)


def portfolio_returns(
    weights: pd.DataFrame, returns: pd.DataFrame, transaction_cost_bps: float = 0.0
) -> pd.Series:
    gross = (weights.shift(1).reindex_like(returns).fillna(0.0) * returns).sum(axis=1)
    costs = turnover(weights).reindex(gross.index).fillna(0.0) * transaction_cost_bps / 10_000.0
    return (gross - costs).rename("portfolio_return")


def performance_stats(returns: pd.Series, annualization: int = 252) -> pd.Series:
    clean = returns.dropna()
    downside = clean[clean < 0]
    cumulative = (1.0 + clean).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    years = len(clean) / annualization
    cagr = cumulative.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and len(cumulative) else np.nan
    vol = clean.std(ddof=1) * np.sqrt(annualization)
    return pd.Series(
        {
            "sharpe": clean.mean() / clean.std(ddof=1) * np.sqrt(annualization)
            if clean.std(ddof=1) != 0
            else np.nan,
            "sortino": clean.mean() / downside.std(ddof=1) * np.sqrt(annualization)
            if len(downside) > 1 and downside.std(ddof=1) != 0
            else np.nan,
            "max_drawdown": drawdown.min() if len(drawdown) else np.nan,
            "cagr": cagr,
            "volatility": vol,
            "turnover": np.nan,
        }
    )
