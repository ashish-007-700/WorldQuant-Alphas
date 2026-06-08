from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from wqalpha.types import Panel


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DataLoader:
    """Load and validate daily Indian equity data in long panel format."""

    path: Path
    date_column: str = "date"
    symbol_column: str = "symbol"

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: str) -> "DataLoader":
        return cls(Path(path), **kwargs)

    def load(self) -> Panel:
        frame = pd.read_csv(self.path)
        return normalize_panel(frame, self.date_column, self.symbol_column)


def normalize_panel(
    frame: pd.DataFrame,
    date_column: str = "date",
    symbol_column: str = "symbol",
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> Panel:
    missing = {date_column, symbol_column, *required_columns}.difference(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    out = frame.copy()
    out[date_column] = pd.to_datetime(out[date_column]).dt.tz_localize(None)
    out[symbol_column] = out[symbol_column].astype(str)
    out = out.sort_values([date_column, symbol_column]).set_index([date_column, symbol_column])
    out.index.names = ["date", "symbol"]

    numeric_columns = [c for c in out.columns if c not in {"sector", "industry"}]
    out[numeric_columns] = out[numeric_columns].apply(pd.to_numeric, errors="coerce")

    if "vwap" not in out:
        dollar_range = out["high"] + out["low"] + out["close"]
        out["vwap"] = dollar_range / 3.0

    if "returns" not in out:
        out["returns"] = out.groupby(level="symbol")["close"].pct_change()

    return out.sort_index()


def wide(panel: Panel, column: str) -> pd.DataFrame:
    """Convert a panel column to a date x symbol matrix."""

    if panel.index.names != ["date", "symbol"]:
        raise ValueError("Panel index must be MultiIndex(date, symbol).")
    return panel[column].unstack("symbol").sort_index()


def align_like(panel: Panel, matrix: pd.DataFrame, name: str) -> pd.Series:
    """Convert a date x symbol matrix back to a panel-aligned Series."""

    series = matrix.stack().rename(name)
    series.index.names = ["date", "symbol"]
    return series.reindex(panel.index)
