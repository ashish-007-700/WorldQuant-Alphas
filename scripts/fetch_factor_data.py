"""Fetch real Indian equity factor index data from Yahoo Finance.

Downloads the following NSE/BSE indices used to construct an Indian
Fama-French-style 4-factor model:

  - Nifty 50         (^NSEI)            — broad market
  - Nifty 100        (^CNX100)          — large-cap
  - Nifty Smallcap 100 (^CNXSC)        — small-cap
  - Nifty 500 Value 50 (^NIFTY500VL50) — value proxy
  - Nifty 500 Growth   (^NIFTY500GR50) — growth proxy
  - Nifty Midcap 100 (^CNXMC)          — mid-cap

Saves to data/india_factors_raw.csv (wide format, one column per index).

Usage       
-----
    python scripts/fetch_factor_data.py
    python scripts/fetch_factor_data.py --years 5
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


FACTOR_INDICES = {
    "nifty50":      "^NSEI",       # Nifty 50 — broad market
    "nifty100":     "^CNX100",     # Nifty 100 — large-cap
    "niftyit":      "^CNXIT",      # Nifty IT sector
    "niftybank":    "^NSEBANK",    # Nifty Bank sector
    "niftyenergy":  "^CNXENERGY",  # Nifty Energy sector (proxy for value/cyclicals)
}


def fetch_index(name: str, ticker: str, start: str, end: str, retries: int = 3) -> pd.Series | None:
    """Download a single index from Yahoo Finance and return daily returns."""
    import yfinance as yf

    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False, timeout=30)
            if df.empty:
                print(f"  [WARN] {name} ({ticker}): no data returned")
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            close = df["close"].dropna()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close.index.name = "date"
            ret = close.pct_change().rename(name)
            print(f"  [OK] {name:15s} ({ticker:20s}) : {len(ret)} rows  "
                  f"{ret.index.min().date()} -> {ret.index.max().date()}")
            return ret
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [WARN] {name} attempt {attempt+1}/{retries}: {exc}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  [ERROR] {name} ({ticker}): all attempts failed — skipping.")
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch Indian factor index data from Yahoo Finance.")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="data/india_factors_raw.csv")
    args = p.parse_args(argv)

    end_date = args.end or date.today().strftime("%Y-%m-%d")
    start_date = args.start or (date.today() - timedelta(days=args.years * 365)).strftime("%Y-%m-%d")

    print(f"=== Indian Factor Index Fetcher ===")
    print(f"  Period : {start_date} -> {end_date}")
    print(f"  Indices: {len(FACTOR_INDICES)}")
    print()

    series: list[pd.Series] = []
    for name, ticker in FACTOR_INDICES.items():
        ret = fetch_index(name, ticker, start=start_date, end=end_date)
        if ret is not None:
            series.append(ret)
        time.sleep(0.5)

    if not series:
        print("[ERROR] No data downloaded.")
        return 1

    factors_raw = pd.concat(series, axis=1)
    factors_raw.index.name = "date"
    factors_raw = factors_raw.sort_index()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    factors_raw.to_csv(out_path)

    print()
    print(f"=== Done ===")
    print(f"  Date range : {factors_raw.index.min().date()} to {factors_raw.index.max().date()}")
    print(f"  Shape      : {factors_raw.shape}")
    print(f"  Saved to   : {out_path.resolve()}")
    print()
    print(factors_raw.tail(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
