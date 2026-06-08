from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def forward_returns(returns: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    return returns.shift(-horizon)


def information_coefficient(
    signal: pd.DataFrame, fwd_returns: pd.DataFrame, method: str = "spearman"
) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date in signal.index.intersection(fwd_returns.index):
        x = signal.loc[date]
        y = fwd_returns.loc[date]
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            values[date] = np.nan
        elif method == "spearman":
            values[date] = float(spearmanr(x[valid], y[valid]).correlation)
        else:
            values[date] = float(x[valid].corr(y[valid]))
    return pd.Series(values, name="ic")


def icir(ic: pd.Series, annualization: int = 252) -> float:
    clean = ic.dropna()
    return float(clean.mean() / clean.std(ddof=1) * np.sqrt(annualization)) if len(clean) > 1 else np.nan


def hit_ratio(ic: pd.Series) -> float:
    clean = ic.dropna()
    return float((clean > 0).mean()) if len(clean) else np.nan


def alpha_decay(signal: pd.DataFrame, returns: pd.DataFrame, max_lag: int = 20) -> pd.Series:
    decay = {}
    for lag in range(1, max_lag + 1):
        decay[lag] = information_coefficient(signal, forward_returns(returns, lag)).mean()
    return pd.Series(decay, name="alpha_decay")


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
