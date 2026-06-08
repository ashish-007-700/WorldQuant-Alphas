from __future__ import annotations

import numpy as np
import pandas as pd


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank by date."""

    return x.rank(axis=1, pct=True)


def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling percentile rank of the most recent observation."""

    def last_rank(values: np.ndarray) -> float:
        series = pd.Series(values)
        return float(series.rank(pct=True).iloc[-1])

    return x.rolling(window, min_periods=window).apply(last_rank, raw=True)


def delay(x: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return x.shift(periods)


def delta(x: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return x.diff(periods)


def correlation(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).cov(y)


def decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return x.rolling(window, min_periods=window).apply(lambda v: float(np.dot(v, weights)), raw=True)


def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).max()


def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).min()


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).apply(lambda v: float(np.argmax(v) + 1), raw=True)


def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).apply(lambda v: float(np.argmin(v) + 1), raw=True)


def scale(x: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    denom = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(denom, axis=0) * k


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def stddev(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).std()


def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).sum()


def product(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).apply(np.prod, raw=True)


def signed_power(x: pd.DataFrame, exponent: float) -> pd.DataFrame:
    return np.sign(x) * np.power(np.abs(x), exponent)


def neutralize_by_group(signal: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Demean each date's signal within sector/industry groups."""

    aligned_groups = groups.reindex(signal.columns)
    out = signal.copy()
    for group in aligned_groups.dropna().unique():
        cols = aligned_groups[aligned_groups == group].index
        out[cols] = signal[cols].sub(signal[cols].mean(axis=1), axis=0)
    return out


def log(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise natural log, clipped to avoid -inf on non-positive values."""
    return np.log(x.clip(lower=1e-12))


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling (time-series) mean."""
    return x.rolling(window, min_periods=window).mean()


def abs_val(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise absolute value."""
    return x.abs()


def indneutralize(signal: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Industry/sector neutralization — demean signal within groups each date."""
    return neutralize_by_group(signal, groups)


def max_(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise maximum of two same-shaped DataFrames."""
    return x.where(x >= y, y)


def min_(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise minimum of two same-shaped DataFrames."""
    return x.where(x <= y, y)
