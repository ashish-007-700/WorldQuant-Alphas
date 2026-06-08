"""Fetch real NSE equity data from Yahoo Finance (yfinance).

Downloads 5 years of daily OHLCV data for the Nifty 50 constituents and saves
a production-ready panel CSV to data/india_equities.csv.

Usage
-----
    python scripts/fetch_nse_data.py              # full Nifty 50, 5 years
    python scripts/fetch_nse_data.py --years 3    # 3 years
    python scripts/fetch_nse_data.py --symbols RELIANCE TCS INFY  # custom tickers

Notes
-----
* Yahoo Finance uses the ".NS" suffix for NSE-listed stocks.
* VWAP is approximated as (High + Low + Close) / 3 (intraday VWAP unavailable
  from end-of-day feeds — this is the standard approximation).
* Market cap = Close × Shares Outstanding (fetched from yfinance .info).
  Falls back to NaN if info is unavailable.
* Sector mapping is hard-coded for speed and reliability (avoids per-ticker
  .info API calls which are slow and rate-limited).
* Adjust prices for splits and dividends (auto_adjust=True, default in yfinance).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Nifty 50 constituents as of 2024 — Yahoo Finance ".NS" tickers
# ---------------------------------------------------------------------------

NIFTY50 = {
    # symbol (no .NS)  : sector
    "RELIANCE":    "Energy",
    "HDFCBANK":    "Financials",
    "ICICIBANK":   "Financials",
    "INFY":        "IT",
    "TCS":         "IT",
    "BHARTIARTL":  "Telecom",
    "SBIN":        "Financials",
    "HINDUNILVR":  "Consumer",
    "ITC":         "Consumer",
    "KOTAKBANK":   "Financials",
    "LT":          "Industrials",
    "HCLTECH":     "IT",
    "BAJFINANCE":  "Financials",
    "AXISBANK":    "Financials",
    "ASIANPAINT":  "Consumer",
    "MARUTI":      "Auto",
    "NTPC":        "Utilities",
    "SUNPHARMA":   "Healthcare",
    "TITAN":       "Consumer",
    "ULTRACEMCO":  "Materials",
    "WIPRO":       "IT",
    "POWERGRID":   "Utilities",
    "TECHM":       "IT",
    "NESTLEIND":   "Consumer",
    "DRREDDY":     "Healthcare",
    "BAJAJ-AUTO":  "Auto",
    "INDUSINDBK":  "Financials",
    "ADANIENT":    "Industrials",
    "GRASIM":      "Materials",
    "COALINDIA":   "Energy",
    "CIPLA":       "Healthcare",
    "DIVISLAB":    "Healthcare",
    "EICHERMOT":   "Auto",
    "BRITANNIA":   "Consumer",
    "HDFCLIFE":    "Financials",
    "SBILIFE":     "Financials",
    "HEROMOTOCO":  "Auto",
    "ONGC":        "Energy",
    "BPCL":        "Energy",
    "APOLLOHOSP":  "Healthcare",
    "TATACONSUM":  "Consumer",
    "TATASTEEL":   "Materials",
    "JSWSTEEL":    "Materials",
    "HINDALCO":    "Materials",
    "ADANIPORTS":  "Industrials",
    "SHREECEM":    "Materials",
    "M&M":         "Auto",
    "VEDL":        "Materials",
    "TATAPOWER":   "Utilities",
    "BAJAJFINSV":  "Financials",
}


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_symbol(symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    """Download OHLCV from Yahoo Finance for a single NSE symbol.

    Parameters
    ----------
    symbol:
        NSE ticker WITHOUT .NS suffix (e.g. "RELIANCE").
    start, end:
        Date strings in "YYYY-MM-DD" format.
    retries:
        Number of download attempts before giving up.

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume] indexed by date,
    or None if all attempts fail.
    """
    import yfinance as yf

    ticker = f"{symbol}.NS"
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,   # adjusts for splits & dividends
                progress=False,
                timeout=30,
            )
            if df.empty:
                print(f"  [WARN] {symbol}: no data returned from Yahoo Finance")
                return None

            # yfinance returns MultiIndex columns when downloading single ticker
            # in some versions — flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = df.columns.str.lower()
            # Keep only OHLCV
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            return df

        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [WARN] {symbol} attempt {attempt+1}/{retries} failed: {exc}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  [ERROR] {symbol}: all {retries} attempts failed — skipping.")
    return None


def build_panel(
    symbols: dict[str, str],
    start: str,
    end: str,
    delay_between_tickers: float = 0.5,
) -> pd.DataFrame:
    """Download and assemble the full panel for all symbols.

    Parameters
    ----------
    symbols:
        Dict mapping NSE ticker (no suffix) → sector string.
    start, end:
        Date strings.
    delay_between_tickers:
        Seconds to sleep between each ticker download to avoid rate-limiting.

    Returns
    -------
    Long-format DataFrame with columns:
        date, symbol, open, high, low, close, volume, vwap, sector, returns
    """
    all_frames: list[pd.DataFrame] = []
    n = len(symbols)

    for i, (symbol, sector) in enumerate(symbols.items(), 1):
        print(f"  [{i:02d}/{n}] Downloading {symbol}.NS ...")
        df = fetch_symbol(symbol, start=start, end=end)
        if df is None or df.empty:
            continue

        df["symbol"] = symbol
        df["sector"] = sector

        # VWAP approximation: (H + L + C) / 3
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

        all_frames.append(df.reset_index())

        if i < n:
            time.sleep(delay_between_tickers)

    if not all_frames:
        raise RuntimeError("No data was downloaded — check your internet connection.")

    panel = pd.concat(all_frames, ignore_index=True)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Daily returns per symbol (using adjusted close)
    panel = panel.set_index(["date", "symbol"]).sort_index()
    panel["returns"] = panel.groupby(level="symbol")["close"].pct_change()
    panel = panel.reset_index()

    # Column order
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "vwap",
            "sector", "returns"]
    panel = panel[cols]

    return panel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch real NSE data from Yahoo Finance and save to CSV."
    )
    p.add_argument(
        "--years", type=int, default=5,
        help="Number of years of history to fetch (default: 5).",
    )
    p.add_argument(
        "--start", default=None,
        help="Override start date (YYYY-MM-DD). Overrides --years.",
    )
    p.add_argument(
        "--end", default=None,
        help="Override end date (YYYY-MM-DD, default: today).",
    )
    p.add_argument(
        "--symbols", nargs="+", default=None,
        help="Custom list of NSE tickers (no .NS suffix). Defaults to Nifty 50.",
    )
    p.add_argument(
        "--out", default="data/india_equities.csv",
        help="Output CSV path (default: data/india_equities.csv).",
    )
    p.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between ticker downloads to avoid rate limiting (default: 0.3).",
    )
    p.add_argument(
        "--incremental", action="store_true",
        help="Only download dates after the last row in the existing CSV.",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="Run data quality checks after download and print report.",
    )
    p.add_argument(
        "--quality-only", action="store_true",
        help="Skip download; only run data quality checks on the existing CSV.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_path = Path(args.out)

    # --- Quality-only mode ---
    if args.quality_only:
        if not out_path.exists():
            print(f"[ERROR] File not found: {out_path}")
            return 1
        import sys; sys.path.insert(0, 'src')
        from wqalpha.dataquality import print_quality_report
        panel = pd.read_csv(out_path, parse_dates=['date'])
        print_quality_report(panel)
        return 0

    # --- Incremental mode: find last date in existing CSV ---
    end_date = args.end or date.today().strftime("%Y-%m-%d")
    if args.incremental and out_path.exists():
        existing = pd.read_csv(out_path, usecols=["date"], parse_dates=["date"])
        last_date = existing["date"].max()
        # Start from the day after the last row
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[Incremental] Last date in existing file: {last_date.date()}")
        print(f"[Incremental] Fetching from: {start_date}")
    elif args.start:
        start_date = args.start
    else:
        start_date = (date.today() - timedelta(days=args.years * 365)).strftime("%Y-%m-%d")

    # Build symbol→sector dict
    if args.symbols:
        symbols = {s.upper(): NIFTY50.get(s.upper(), "Unknown") for s in args.symbols}
    else:
        symbols = NIFTY50

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== NSE Data Fetcher ===")
    print(f"  Tickers  : {len(symbols)} symbols")
    print(f"  Period   : {start_date} -> {end_date}")
    print(f"  Output   : {out_path}")
    print(f"  Mode     : {'incremental' if args.incremental else 'full'}")
    print(f"  Source   : Yahoo Finance (auto-adjusted prices)")
    print()

    new_panel = build_panel(symbols, start=start_date, end=end_date,
                            delay_between_tickers=args.delay)

    # --- Merge with existing data if incremental ---
    if args.incremental and out_path.exists() and not new_panel.empty:
        existing = pd.read_csv(out_path, parse_dates=["date"])
        panel = pd.concat([existing, new_panel], ignore_index=True)
        panel = panel.drop_duplicates(subset=["date", "symbol"]).sort_values(["date", "symbol"])
        print(f"  New rows added : {len(new_panel):,}")
    else:
        panel = new_panel

    # Summary stats
    n_sym  = panel["symbol"].nunique()
    n_days = panel["date"].nunique()
    date_min = pd.to_datetime(panel["date"]).min().strftime("%Y-%m-%d")
    date_max = pd.to_datetime(panel["date"]).max().strftime("%Y-%m-%d")

    panel.to_csv(out_path, index=False)

    print()
    print(f"=== Done ===")
    print(f"  Symbols downloaded : {n_sym} / {len(symbols)}")
    print(f"  Date range         : {date_min} to {date_max} ({n_days} trading days)")
    print(f"  Total rows         : {len(panel):,}")
    print(f"  Saved to           : {out_path.resolve()}")
    print()
    print(panel.dtypes)
    print()
    print(panel.tail(5).to_string())

    # --- Optional data quality check ---
    if args.validate:
        print()
        import sys; sys.path.insert(0, 'src')
        from wqalpha.dataquality import print_quality_report
        print_quality_report(panel)

    return 0


if __name__ == "__main__":
    sys.exit(main())
