from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=60)
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
    rows = []
    rng = np.random.default_rng(7)
    for symbol_idx, symbol in enumerate(symbols):
        price = 100 + symbol_idx * 10 + rng.normal(0, 1)
        for date in dates:
            ret = rng.normal(0.0005, 0.02)
            open_ = price * (1 + rng.normal(0, 0.003))
            close = price * (1 + ret)
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
            volume = rng.integers(100_000, 1_000_000)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "vwap": (high + low + close) / 3,
                    "sector": ["Energy", "IT", "Financials", "IT", "Consumer"][symbol_idx],
                }
            )
            price = close
    frame = pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
    frame["returns"] = frame.groupby(level="symbol")["close"].pct_change()
    return frame
