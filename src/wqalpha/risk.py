"""Risk attribution, factor exposure, and neutralization utilities.

All factor regression functions work with the real Indian Fama-French factors
from ``wqalpha.factors``.  Pass a pre-built factors DataFrame or load from
``data/india_factors.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Factor regression (core)
# ---------------------------------------------------------------------------

def factor_regression(strategy_returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """OLS factor regression.

    Parameters
    ----------
    strategy_returns:
        Daily portfolio return series.
    factors:
        DataFrame of factor returns (one column per factor).

    Returns
    -------
    Series with beta_{factor}, alpha, r_squared.
    """
    joined = pd.concat([strategy_returns.rename("strategy"), factors], axis=1).dropna()
    if len(joined) < len(factors.columns) + 2:
        return pd.Series(dtype=float)
    x = sm.add_constant(joined[factors.columns])
    model = sm.OLS(joined["strategy"], x).fit()
    out = model.params.add_prefix("beta_")
    out["alpha"]     = model.params["const"]
    out["alpha_ann"] = model.params["const"] * 252
    out["r_squared"] = model.rsquared
    out["t_alpha"]   = model.tvalues["const"]
    out["p_alpha"]   = model.pvalues["const"]
    return out


# ---------------------------------------------------------------------------
# Single-factor CAPM
# ---------------------------------------------------------------------------

def capm(strategy_returns: pd.Series, market_returns: pd.Series) -> pd.Series:
    """CAPM single-factor regression against a market return series."""
    return factor_regression(strategy_returns, market_returns.rename("mkt").to_frame())


# ---------------------------------------------------------------------------
# Fama-French models
# ---------------------------------------------------------------------------

def fama_french_3(strategy_returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Fama-French 3-factor regression (mkt_rf, smb, hml).

    Parameters
    ----------
    factors:
        Must contain columns: ``mkt_rf``, ``smb``, ``hml``.
        Use ``wqalpha.factors.load_factors()`` to load the Indian factor file.
    """
    cols = [c for c in ["mkt_rf", "smb", "hml"] if c in factors.columns]
    return factor_regression(strategy_returns, factors[cols])


def fama_french_4(strategy_returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Fama-French 4-factor regression (mkt_rf, smb, hml, mom)."""
    cols = [c for c in ["mkt_rf", "smb", "hml", "mom"] if c in factors.columns]
    return factor_regression(strategy_returns, factors[cols])


def fama_french_5(strategy_returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Fama-French 5-factor regression (mkt_rf, smb, hml, rmw, cma)."""
    cols = [c for c in ["mkt_rf", "smb", "hml", "rmw", "cma"] if c in factors.columns]
    return factor_regression(strategy_returns, factors[cols])


# ---------------------------------------------------------------------------
# Alpha attribution table
# ---------------------------------------------------------------------------

def alpha_attribution(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Run CAPM, FF3, and FF4 regressions and return a comparison table.

    Parameters
    ----------
    strategy_returns:
        Daily portfolio returns.
    factors:
        Indian factor DataFrame from ``wqalpha.factors.load_factors()``.
        Must contain: ``mkt_rf``, ``smb``, ``hml``, ``mom``.
    models:
        Subset of [``"capm"``, ``"ff3"``, ``"ff4"``] to compute (default: all three).

    Returns
    -------
    DataFrame with one column per model and rows: alpha, alpha_ann, beta_mkt_rf,
    beta_smb, beta_hml, beta_mom, r_squared.
    """
    if models is None:
        models = ["capm", "ff3", "ff4"]

    results = {}
    if "capm" in models and "mkt_rf" in factors.columns:
        results["capm"] = capm(strategy_returns, factors["mkt_rf"])
    if "ff3" in models:
        results["ff3"] = fama_french_3(strategy_returns, factors)
    if "ff4" in models:
        results["ff4"] = fama_french_4(strategy_returns, factors)

    tbl = pd.DataFrame(results)
    display_rows = ["alpha", "alpha_ann", "t_alpha", "p_alpha", "r_squared",
                    "beta_mkt_rf", "beta_smb", "beta_hml", "beta_mom"]
    existing = [r for r in display_rows if r in tbl.index]
    return tbl.loc[existing].round(4)


# ---------------------------------------------------------------------------
# Sector neutralization
# ---------------------------------------------------------------------------

def sector_neutralize(signal: pd.DataFrame, sector_map: pd.Series) -> pd.DataFrame:
    """Demean each alpha signal within sectors on each date.

    Parameters
    ----------
    signal:
        Wide (date × symbol) signal matrix.
    sector_map:
        Series mapping symbol → sector string.
    """
    out = signal.copy()
    for sector, symbols in sector_map.groupby(sector_map).groups.items():
        cols = [c for c in symbols if c in signal.columns]
        if cols:
            out[cols] = signal[cols].sub(signal[cols].mean(axis=1), axis=0)
    return out


# ---------------------------------------------------------------------------
# Sector-stratified IC
# ---------------------------------------------------------------------------

def sector_ic(
    signal: pd.DataFrame,
    forward_ret: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """Compute Information Coefficient per sector per day.

    Parameters
    ----------
    signal:
        Wide (date × symbol) signal matrix.
    forward_ret:
        Wide (date × symbol) 1-day forward returns matrix.
    sector_map:
        Series mapping symbol → sector string.

    Returns
    -------
    DataFrame with one column per sector and DatetimeIndex.
    Row value = cross-sectional rank IC between signal and forward return
    within that sector on that date.
    """
    from scipy.stats import spearmanr

    sectors = sector_map.unique()
    ic_dict: dict[str, pd.Series] = {}

    for sector in sorted(sectors):
        syms = [s for s in sector_map[sector_map == sector].index if s in signal.columns]
        if not syms:
            continue
        sig_sec  = signal[syms]
        fwd_sec  = forward_ret.reindex(columns=syms)
        daily_ic = []
        for date in sig_sec.index:
            s = sig_sec.loc[date].dropna()
            f = fwd_sec.loc[date].reindex(s.index).dropna()
            s, f = s.align(f, join="inner")
            if len(s) < 3:
                daily_ic.append(np.nan)
            else:
                rho, _ = spearmanr(s, f)
                daily_ic.append(rho)
        ic_dict[sector] = pd.Series(daily_ic, index=sig_sec.index, name=sector)

    return pd.DataFrame(ic_dict)


# ---------------------------------------------------------------------------
# Rolling beta
# ---------------------------------------------------------------------------

def rolling_beta(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = 252,
) -> pd.DataFrame:
    """Rolling CAPM beta for each stock.

    Parameters
    ----------
    returns:
        Wide (date × symbol) daily return matrix.
    market_returns:
        Market return series (e.g. Nifty 50).
    lookback:
        Rolling window in trading days (default: 252 = 1 year).
    """
    cov = returns.rolling(lookback, min_periods=lookback // 2).cov(market_returns)
    var = market_returns.rolling(lookback, min_periods=lookback // 2).var()
    return cov.div(var, axis=0)


# ---------------------------------------------------------------------------
# Beta neutralization
# ---------------------------------------------------------------------------

def beta_neutralize(signal: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
    """Remove market beta exposure from a cross-sectional signal.

    For each date, regresses the signal on cross-sectional beta and returns
    the residuals (market-beta-neutral signal).

    Parameters
    ----------
    signal:
        Wide (date × symbol) signal matrix.
    betas:
        Wide (date × symbol) rolling beta matrix from ``rolling_beta()``.
    """
    residuals = pd.DataFrame(index=signal.index, columns=signal.columns, dtype=float)
    for date in signal.index:
        y = signal.loc[date]
        x = betas.loc[date] if date in betas.index else pd.Series(index=signal.columns, dtype=float)
        valid = y.notna() & x.notna()
        if valid.sum() < 3:
            continue
        x_valid = sm.add_constant(x[valid])
        model = sm.OLS(y[valid], x_valid).fit()
        residuals.loc[date, valid] = model.resid
    return residuals.replace([np.inf, -np.inf], np.nan)
