from __future__ import annotations

import pandas as pd

from wqalpha.data import wide
from wqalpha import operators as op
from wqalpha.types import Panel


class FeatureEngine:
    """Convenience accessor for common alpha inputs as date x symbol matrices.

    All properties return *wide* DataFrames indexed by date with symbol columns,
    which is the standard input format for all operator functions.
    """

    def __init__(self, panel: Panel):
        self.panel = panel

    # ------------------------------------------------------------------
    # Core matrix accessor
    # ------------------------------------------------------------------

    def matrix(self, column: str) -> pd.DataFrame:
        return wide(self.panel, column)

    # ------------------------------------------------------------------
    # OHLCV + derived basics
    # ------------------------------------------------------------------

    @property
    def open(self) -> pd.DataFrame:
        return self.matrix("open")

    @property
    def high(self) -> pd.DataFrame:
        return self.matrix("high")

    @property
    def low(self) -> pd.DataFrame:
        return self.matrix("low")

    @property
    def close(self) -> pd.DataFrame:
        return self.matrix("close")

    @property
    def volume(self) -> pd.DataFrame:
        return self.matrix("volume")

    @property
    def vwap(self) -> pd.DataFrame:
        return self.matrix("vwap")

    @property
    def returns(self) -> pd.DataFrame:
        return self.matrix("returns")

    @property
    def market_cap(self) -> pd.DataFrame | None:
        """Market capitalisation matrix, or None if not in panel."""
        if "market_cap" not in self.panel.columns:
            return None
        return self.matrix("market_cap")

    # ------------------------------------------------------------------
    # Average daily volume variants (adv{N})
    # ------------------------------------------------------------------

    def adv(self, n: int) -> pd.DataFrame:
        """Rolling N-day average daily volume."""
        return self.volume.rolling(n, min_periods=n).mean()

    @property
    def adv5(self) -> pd.DataFrame:
        return self.adv(5)

    @property
    def adv10(self) -> pd.DataFrame:
        return self.adv(10)

    @property
    def adv15(self) -> pd.DataFrame:
        return self.adv(15)

    @property
    def adv20(self) -> pd.DataFrame:
        return self.adv(20)

    @property
    def adv30(self) -> pd.DataFrame:
        return self.adv(30)

    @property
    def adv40(self) -> pd.DataFrame:
        return self.adv(40)

    @property
    def adv50(self) -> pd.DataFrame:
        return self.adv(50)

    @property
    def adv60(self) -> pd.DataFrame:
        return self.adv(60)

    @property
    def adv81(self) -> pd.DataFrame:
        return self.adv(81)

    @property
    def adv120(self) -> pd.DataFrame:
        return self.adv(120)

    @property
    def adv150(self) -> pd.DataFrame:
        return self.adv(150)

    @property
    def adv180(self) -> pd.DataFrame:
        return self.adv(180)

    # ------------------------------------------------------------------
    # Sector / industry helpers
    # ------------------------------------------------------------------

    @property
    def sector_map(self) -> pd.Series:
        """Symbol → sector mapping (most recent value per symbol).

        Returns an empty Series if the panel has no ``sector`` column.
        """
        if "sector" not in self.panel.columns:
            return pd.Series(dtype=str)
        return self.panel["sector"].groupby(level="symbol").last()

    def indneutralize(self, x: pd.DataFrame) -> pd.DataFrame:
        """Demean *x* (date × symbol matrix) within each sector each date.

        Falls through to identity when no sector information is available.
        """
        groups = self.sector_map
        if groups.empty:
            return x
        return op.neutralize_by_group(x, groups)
