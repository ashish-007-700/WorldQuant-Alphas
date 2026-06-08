"""Generate a realistic synthetic NSE/BSE daily equity panel for development and testing.

Output: data/india_equities_sample.csv
  - 50 NSE large-cap tickers
  - 3 years of business days (~756 rows per symbol, ~37,800 total)
  - Columns: date, symbol, open, high, low, close, volume, vwap, sector, market_cap, returns
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Universe: 50 representative NSE large-cap tickers with sector and base price
# ---------------------------------------------------------------------------

UNIVERSE = [
    ("RELIANCE",    "Energy",         2500, 100e9),
    ("HDFCBANK",    "Financials",     1650,  90e9),
    ("INFY",        "IT",             1500,  80e9),
    ("TCS",         "IT",             3500, 120e9),
    ("ICICIBANK",   "Financials",     950,   70e9),
    ("HINDUNILVR",  "Consumer",       2400,  55e9),
    ("ITC",         "Consumer",       440,   45e9),
    ("SBIN",        "Financials",     570,   50e9),
    ("BHARTIARTL",  "Telecom",        900,   48e9),
    ("KOTAKBANK",   "Financials",     1800,  60e9),
    ("LT",          "Industrials",    2800,  40e9),
    ("AXISBANK",    "Financials",     1000,  30e9),
    ("BAJFINANCE",  "Financials",     7000,  45e9),
    ("HCLTECH",     "IT",             1200,  35e9),
    ("WIPRO",       "IT",             450,   25e9),
    ("ASIANPAINT",  "Consumer",       3100,  30e9),
    ("MARUTI",      "Auto",           9500,  28e9),
    ("SUNPHARMA",   "Healthcare",     1100,  26e9),
    ("NTPC",        "Utilities",      240,   22e9),
    ("POWERGRID",   "Utilities",      230,   21e9),
    ("ULTRACEMCO",  "Materials",      8000,  23e9),
    ("TITAN",       "Consumer",       2900,  26e9),
    ("TECHM",       "IT",             1100,  20e9),
    ("NESTLEIND",   "Consumer",       22000, 21e9),
    ("BAJAJ-AUTO",  "Auto",           7500,  22e9),
    ("DRREDDY",     "Healthcare",     5200,  22e9),
    ("INDUSINDBK",  "Financials",     1200,  18e9),
    ("GRASIM",      "Materials",      2000,  15e9),
    ("ADANIENT",    "Industrials",    2400,  25e9),
    ("COALINDIA",   "Energy",         400,   25e9),
    ("CIPLA",       "Healthcare",     1200,  19e9),
    ("DIVISLAB",    "Healthcare",     4000,  21e9),
    ("EICHERMOT",   "Auto",           3500,  14e9),
    ("BRITANNIA",   "Consumer",       4800,  12e9),
    ("HDFCLIFE",    "Financials",     650,   13e9),
    ("SBILIFE",     "Financials",     1400,  14e9),
    ("HEROMOTOCO",  "Auto",           3000,  12e9),
    ("ONGC",        "Energy",         200,   25e9),
    ("BPCL",        "Energy",         450,   10e9),
    ("APOLLOHOSP",  "Healthcare",     5500,  16e9),
    ("TATACONSUM",  "Consumer",       830,   8e9),
    ("TATASTEEL",   "Materials",      130,   15e9),
    ("JSWSTEEL",    "Materials",      810,   20e9),
    ("HINDALCO",    "Materials",      470,   10e9),
    ("ADANIPORTS",  "Industrials",    800,   17e9),
    ("SHREECEM",    "Materials",      27000, 10e9),
    ("SBICARD",     "Financials",     800,   8e9),
    ("M&M",         "Auto",           1600,  14e9),
    ("VEDL",        "Materials",      330,   12e9),
    ("TATAPOWER",   "Utilities",      380,   12e9),
]


def generate_panel(seed: int = 42, years: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=years * 252)
    n_dates = len(dates)

    rows: list[dict] = []
    for symbol, sector, base_price, market_cap in UNIVERSE:
        price = float(base_price)
        # Symbol-specific drift and vol parameters
        annual_drift = rng.uniform(0.05, 0.20)
        annual_vol   = rng.uniform(0.18, 0.45)
        daily_drift  = annual_drift / 252
        daily_vol    = annual_vol / np.sqrt(252)

        # Sector co-movement factor (shared shock)
        sector_betas = {
            "Energy": 0.9, "Financials": 1.1, "IT": 0.8, "Consumer": 0.6,
            "Telecom": 0.7, "Industrials": 1.0, "Utilities": 0.5,
            "Healthcare": 0.6, "Auto": 1.0, "Materials": 1.1,
        }
        beta = sector_betas.get(sector, 1.0)

        for i, date in enumerate(dates):
            # Intraday patterns + market + idiosyncratic shocks
            market_shock = rng.normal(0, 0.01)
            idio_shock   = rng.normal(daily_drift, daily_vol)
            ret = beta * market_shock + idio_shock

            open_ = price * (1 + rng.normal(0, 0.005))
            close = price * (1 + ret)
            intra_range = abs(rng.normal(0, daily_vol * 0.6))
            high = max(open_, close) * (1 + intra_range)
            low  = min(open_, close) * (1 - intra_range)

            # Volume: log-normal with some mean-reversion
            base_vol = rng.lognormal(mean=np.log(5_000_000), sigma=0.5)
            vol_int = int(max(100_000, base_vol))

            vwap = (high + low + close) / 3

            # Market cap grows with price
            mcap = market_cap * (close / base_price)

            rows.append({
                "date":       date,
                "symbol":     symbol,
                "open":       round(open_, 2),
                "high":       round(high, 2),
                "low":        round(low, 2),
                "close":      round(close, 2),
                "volume":     vol_int,
                "vwap":       round(vwap, 2),
                "sector":     sector,
                "market_cap": round(mcap, 0),
            })
            price = close

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Add returns (per symbol)
    df = df.set_index(["date", "symbol"]).sort_index()
    df["returns"] = df.groupby(level="symbol")["close"].pct_change()
    df = df.reset_index()

    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent.parent / "data" / "india_equities_sample.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel = generate_panel()
    panel.to_csv(out_path, index=False)
    print(f"Saved {len(panel):,} rows x {len(panel.columns)} columns -> {out_path}")
    print(panel.dtypes)
    print(panel.head(3))
