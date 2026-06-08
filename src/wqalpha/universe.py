"""Universe management — liquidity and price filters for tradeable stocks.

Provides functions to apply minimum Average Daily Volume (ADV) and price
filters to alpha signals before portfolio construction.  This prevents the
framework from trading illiquid micro-caps that cannot be accessed at scale.

All monetary values are in Indian Rupees (INR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ADV Filter
# ---------------------------------------------------------------------------

def compute_adv_inr(
    panel: pd.DataFrame,
    adv_days: int = 20,
) -> pd.DataFrame:
    """Compute N-day average daily turnover in INR (close × volume).

    Parameters
    ----------
    panel:
        Long-format equity panel with columns: date, symbol, close, volume.
    adv_days:
        Rolling window (trading days) for the ADV calculation.

    Returns
    -------
    Wide DataFrame (date × symbol) of ADV in INR.
    """
    if isinstance(panel.index, pd.MultiIndex):
        flat = panel.reset_index()
    else:
        flat = panel.copy()

    flat["turnover_inr"] = flat["close"] * flat["volume"]
    tv = flat.pivot(index="date", columns="symbol", values="turnover_inr")
    tv.index = pd.to_datetime(tv.index)
    return tv.rolling(adv_days, min_periods=max(1, adv_days // 2)).mean()


def apply_adv_filter(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    min_adv_cr: float = 50.0,
    adv_days: int = 20,
) -> pd.DataFrame:
    """Zero-out signal for stocks below minimum average daily volume.

    Parameters
    ----------
    signal:
        Wide (date × symbol) signal matrix.
    panel:
        Long-format equity panel.
    min_adv_cr:
        Minimum ADV in INR crore (1 crore = 10^7 INR).
        Default: 50 crore (~$6M USD) — typical institutional liquidity threshold.
    adv_days:
        Rolling window for ADV computation.

    Returns
    -------
    Signal matrix with illiquid positions set to NaN.
    """
    min_adv_inr = min_adv_cr * 1e7
    adv = compute_adv_inr(panel, adv_days=adv_days)
    adv = adv.reindex_like(signal)
    liquid_mask = adv >= min_adv_inr
    return signal.where(liquid_mask)


def apply_price_filter(
    signal: pd.DataFrame,
    panel: pd.DataFrame,
    min_price: float = 10.0,
) -> pd.DataFrame:
    """Zero-out signal for stocks below minimum price (penny stock filter).

    Parameters
    ----------
    min_price:
        Minimum closing price in INR (default: ₹10).
    """
    if isinstance(panel.index, pd.MultiIndex):
        flat = panel.reset_index()
    else:
        flat = panel.copy()

    close = flat.pivot(index="date", columns="symbol", values="close")
    close.index = pd.to_datetime(close.index)
    close = close.reindex_like(signal)
    return signal.where(close >= min_price)


def build_liquid_universe(
    panel: pd.DataFrame,
    adv_days: int = 20,
    min_adv_cr: float = 50.0,
    min_price: float = 10.0,
) -> pd.DataFrame:
    """Build a Boolean mask of tradeable (liquid) stocks.

    Parameters
    ----------
    panel:
        Long-format equity panel.
    adv_days:
        Rolling ADV window.
    min_adv_cr:
        Minimum ADV in INR crore.
    min_price:
        Minimum price filter in INR.

    Returns
    -------
    Wide Boolean DataFrame (date × symbol). True = stock is tradeable that day.
    """
    if isinstance(panel.index, pd.MultiIndex):
        flat = panel.reset_index()
    else:
        flat = panel.copy()

    min_adv_inr = min_adv_cr * 1e7
    adv = compute_adv_inr(panel, adv_days=adv_days)
    close = flat.pivot(index="date", columns="symbol", values="close")
    close.index = pd.to_datetime(close.index)

    liquid = (adv >= min_adv_inr) & (close >= min_price)
    return liquid


def coverage_report(panel: pd.DataFrame, min_adv_cr: float = 50.0) -> pd.DataFrame:
    """Return a per-symbol liquidity coverage report.

    Shows what fraction of trading days each stock passes the ADV filter.
    """
    if isinstance(panel.index, pd.MultiIndex):
        flat = panel.reset_index()
    else:
        flat = panel.copy()

    universe = build_liquid_universe(panel, min_adv_cr=min_adv_cr)
    report = pd.DataFrame({
        "total_days":   universe.count(),
        "liquid_days":  universe.sum(),
        "liquid_pct":   (universe.sum() / universe.count() * 100).round(1),
        "avg_adv_cr":   (compute_adv_inr(panel).mean() / 1e7).round(1),
    })
    return report.sort_values("liquid_pct", ascending=False)
