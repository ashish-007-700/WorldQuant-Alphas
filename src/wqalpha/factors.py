"""Indian equity factor construction (Fama-French style).

Constructs four daily return factors from real NSE data:

  mkt_rf : Nifty 50 excess return over the RBI risk-free rate
  smb    : Small-Minus-Big (bottom market-cap tercile minus top tercile,
           constructed from the Nifty 50 universe by market-cap rank)
  hml    : High-Minus-Low (top B/M proxy minus bottom, approximated by
           inverse of trailing 12m return as a value signal)
  mom    : 12-1 month momentum factor (top quintile minus bottom quintile
           from the Nifty 50 universe)

The RBI repo rate is used as the risk-free proxy. The rate is hand-coded from
RBI policy announcements (updated to June 2026).  For real production use,
source this from the CCIL/FBIL overnight MIBOR or a Bloomberg feed.

Loading
-------
    from wqalpha.factors import build_factors, load_factors

    # Build from raw index CSV + equity panel
    factors = build_factors(panel, raw_index_path="data/india_factors_raw.csv")
    factors.to_csv("data/india_factors.csv")

    # Load pre-built file
    factors = load_factors("data/india_factors.csv")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RBI Repo Rate history (annualised %)
# Source: RBI monetary policy announcements
# ---------------------------------------------------------------------------

_RBI_REPO_RATE = pd.Series(
    {
        "2019-01-01": 6.50,
        "2019-04-04": 6.25,
        "2019-06-06": 5.75,
        "2019-08-07": 5.40,
        "2019-10-04": 5.15,
        "2020-03-27": 4.40,
        "2020-05-22": 4.00,
        "2022-05-04": 4.40,
        "2022-06-08": 4.90,
        "2022-08-05": 5.40,
        "2022-09-30": 5.90,
        "2022-12-07": 6.25,
        "2023-02-08": 6.50,
        "2025-02-07": 6.25,
        "2025-04-09": 6.00,
        "2025-06-06": 5.75,
    },
    name="repo_rate",
)
_RBI_REPO_RATE.index = pd.to_datetime(_RBI_REPO_RATE.index)


def _daily_rf(dates: pd.DatetimeIndex) -> pd.Series:
    """Return a daily risk-free rate series aligned to *dates*."""
    rate = _RBI_REPO_RATE.reindex(dates, method="ffill").ffill().bfill()
    return (rate / 100.0 / 252.0).rename("rf")


# ---------------------------------------------------------------------------
# Core factor construction
# ---------------------------------------------------------------------------

def build_factors(
    panel: pd.DataFrame,
    raw_index_path: str | Path = "data/india_factors_raw.csv",
) -> pd.DataFrame:
    """Construct daily Indian Fama-French 4-factor returns.

    Parameters
    ----------
    panel:
        Long-format equity panel (MultiIndex date×symbol or flat with date/symbol cols).
        Must contain: close, returns, and optionally market_cap.
    raw_index_path:
        Path to the CSV saved by ``scripts/fetch_factor_data.py``.

    Returns
    -------
    DataFrame indexed by date with columns: ``mkt_rf``, ``smb``, ``hml``, ``mom``, ``rf``.
    """
    # ------------------------------------------------------------------
    # 1. Load index data
    # ------------------------------------------------------------------
    idx = pd.read_csv(raw_index_path, index_col="date", parse_dates=True)

    # ------------------------------------------------------------------
    # 2. Prepare equity panel in wide form
    # ------------------------------------------------------------------
    if isinstance(panel.index, pd.MultiIndex):
        flat = panel.reset_index()
    else:
        flat = panel.copy()

    # Pivot to wide (date × symbol)
    close_wide  = flat.pivot(index="date", columns="symbol", values="close")
    ret_wide    = flat.pivot(index="date", columns="symbol", values="returns")
    close_wide.index = pd.to_datetime(close_wide.index)
    ret_wide.index   = pd.to_datetime(ret_wide.index)

    # Market cap proxy: use close price (proportional to market cap within this universe)
    mcap_wide = close_wide.copy()
    if "market_cap" in flat.columns:
        mc_tmp = flat.pivot(index="date", columns="symbol", values="market_cap")
        mc_tmp.index = pd.to_datetime(mc_tmp.index)
        mcap_wide = mc_tmp

    # ------------------------------------------------------------------
    # 3. Market factor (Mkt-RF)
    # ------------------------------------------------------------------
    nifty50_ret = idx["nifty50"] if "nifty50" in idx.columns else ret_wide.mean(axis=1)
    all_dates = nifty50_ret.index
    rf_series = _daily_rf(all_dates)
    mkt_rf = (nifty50_ret - rf_series).rename("mkt_rf")

    # ------------------------------------------------------------------
    # 4. SMB — Small Minus Big
    # Rank stocks by market cap each day; top tercile = Big, bottom = Small
    # ------------------------------------------------------------------
    def _smb_day(date: pd.Timestamp) -> float:
        if date not in mcap_wide.index or date not in ret_wide.index:
            return np.nan
        mc = mcap_wide.loc[date].dropna()
        r  = ret_wide.loc[date].reindex(mc.index).dropna()
        mc, r = mc.align(r, join="inner")
        if len(mc) < 6:
            return np.nan
        tercile = len(mc) // 3
        sorted_mc = mc.sort_values()
        small = r[sorted_mc.index[:tercile]].mean()
        big   = r[sorted_mc.index[-tercile:]].mean()
        return small - big

    smb = pd.Series(
        {d: _smb_day(d) for d in all_dates},
        name="smb",
    )

    # ------------------------------------------------------------------
    # 5. HML — High Minus Low (value factor)
    # Proxy: low 12m return stocks (cheap/distressed) minus high 12m return stocks (expensive)
    # This is the inverse momentum approach, approximating book-to-market
    # ------------------------------------------------------------------
    mom_12m = ret_wide.rolling(252, min_periods=126).sum()   # 12-month cumulative

    def _hml_day(date: pd.Timestamp) -> float:
        if date not in mom_12m.index or date not in ret_wide.index:
            return np.nan
        val = mom_12m.loc[date].dropna()
        r   = ret_wide.loc[date].reindex(val.index).dropna()
        val, r = val.align(r, join="inner")
        if len(val) < 6:
            return np.nan
        q = len(val) // 3
        sorted_val = val.sort_values()
        high_bm = r[sorted_val.index[:q]].mean()   # low past return = high B/M (value)
        low_bm  = r[sorted_val.index[-q:]].mean()  # high past return = low B/M (growth)
        return high_bm - low_bm

    hml = pd.Series(
        {d: _hml_day(d) for d in all_dates},
        name="hml",
    )

    # ------------------------------------------------------------------
    # 6. MOM — 12-1 Month Momentum
    # Top quintile (past winners) minus bottom quintile (past losers)
    # Standard 12-1 formation: use 11-month return with 1-month skip
    # ------------------------------------------------------------------
    mom_formation = ret_wide.shift(21).rolling(231, min_periods=100).sum()  # ~11m skip 1m

    def _mom_day(date: pd.Timestamp) -> float:
        if date not in mom_formation.index or date not in ret_wide.index:
            return np.nan
        m = mom_formation.loc[date].dropna()
        r = ret_wide.loc[date].reindex(m.index).dropna()
        m, r = m.align(r, join="inner")
        if len(m) < 5:
            return np.nan
        q = max(1, len(m) // 5)
        sorted_m = m.sort_values()
        winners = r[sorted_m.index[-q:]].mean()
        losers  = r[sorted_m.index[:q]].mean()
        return winners - losers

    mom = pd.Series(
        {d: _mom_day(d) for d in all_dates},
        name="mom",
    )

    # ------------------------------------------------------------------
    # 7. Assemble
    # ------------------------------------------------------------------
    factors = pd.concat([mkt_rf, smb, hml, mom, rf_series], axis=1)
    factors.index.name = "date"
    factors = factors.dropna(subset=["mkt_rf"])

    return factors


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_factors(path: str | Path = "data/india_factors.csv") -> pd.DataFrame:
    """Load a pre-built factor CSV."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.index.name = "date"
    return df


def describe_factors(factors: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for factor returns."""
    ann = 252
    stats = pd.DataFrame(
        {
            "mean_ann":   factors.mean() * ann,
            "vol_ann":    factors.std() * (ann ** 0.5),
            "sharpe":     factors.mean() / factors.std() * (ann ** 0.5),
            "skew":       factors.skew(),
            "min":        factors.min(),
            "max":        factors.max(),
        }
    )
    return stats.round(4)
