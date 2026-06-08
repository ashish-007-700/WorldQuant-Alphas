"""Data quality checks for the NSE equity panel.

Detects common data issues before they corrupt alpha signals:
  - Missing values per symbol/column
  - Stale prices (N consecutive identical closes)
  - Return outliers (|ret| > K * rolling std)
  - Volume spikes (volume > N × 20d average)
  - Calendar gaps (missing trading days)
  - Zero-volume rows (no trading activity)

Example
-------
    from wqalpha.dataquality import check_panel, DataQualityReport

    panel = pd.read_csv("data/india_equities.csv", parse_dates=["date"])
    report = check_panel(panel)
    print(report.summary())
    print(report.bad_symbols())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class DataQualityReport:
    """Results of a data quality check run."""
    missing:     pd.DataFrame = field(default_factory=pd.DataFrame)  # symbol x column
    stale:       pd.DataFrame = field(default_factory=pd.DataFrame)  # date x symbol: stale streak
    outliers:    pd.DataFrame = field(default_factory=pd.DataFrame)  # flagged return outliers
    vol_spikes:  pd.DataFrame = field(default_factory=pd.DataFrame)  # flagged volume spikes
    zero_volume: pd.DataFrame = field(default_factory=pd.DataFrame)  # zero-volume rows
    gaps:        list[dict]   = field(default_factory=list)          # calendar gaps per symbol

    def summary(self) -> str:
        lines = ["=== Data Quality Report ==="]
        lines.append(f"  Missing values per column:")
        for col in self.missing.columns:
            total = self.missing[col].sum()
            if total > 0:
                lines.append(f"    {col:20s}: {total:,} cells ({total/self.missing[col].count()*100:.1f}%)")

        n_stale = (self.stale >= 3).sum().sum() if not self.stale.empty else 0
        lines.append(f"  Stale price streaks (>=3 days): {n_stale:,}")

        n_out = len(self.outliers) if not self.outliers.empty else 0
        lines.append(f"  Return outliers (>5 std-dev): {n_out:,}")

        n_spikes = len(self.vol_spikes) if not self.vol_spikes.empty else 0
        lines.append(f"  Volume spikes (>10x ADV): {n_spikes:,}")

        n_zero = len(self.zero_volume) if not self.zero_volume.empty else 0
        lines.append(f"  Zero-volume rows: {n_zero:,}")

        n_gaps = sum(len(g.get("gaps", [])) for g in self.gaps)
        lines.append(f"  Calendar gaps (missing trading days): {n_gaps:,}")

        return "\n".join(lines)

    def bad_symbols(self, threshold_pct: float = 5.0) -> list[str]:
        """Return symbols with >threshold_pct% missing values in any column."""
        if self.missing.empty:
            return []
        pct = self.missing / self.missing.sum() * 100
        return list(pct[(pct > threshold_pct).any(axis=1)].index)


# ---------------------------------------------------------------------------
# Core check function
# ---------------------------------------------------------------------------

def check_panel(
    panel: pd.DataFrame,
    stale_days: int = 3,
    outlier_sigma: float = 5.0,
    volume_spike_x: float = 10.0,
) -> DataQualityReport:
    """Run all data quality checks on the equity panel.

    Parameters
    ----------
    panel:
        Long-format equity panel with columns: date, symbol, open, high, low,
        close, volume, returns.
    stale_days:
        Number of consecutive identical closes to flag as stale.
    outlier_sigma:
        Number of rolling std-deviations beyond which a return is flagged as outlier.
    volume_spike_x:
        Multiple of 20d average volume to flag as a volume spike.

    Returns
    -------
    :class:`DataQualityReport`
    """
    if isinstance(panel.index, pd.MultiIndex):
        flat = panel.reset_index()
    else:
        flat = panel.copy()
    flat["date"] = pd.to_datetime(flat["date"])

    # ------------------------------------------------------------------
    # 1. Missing values
    # ------------------------------------------------------------------
    core_cols = ["open", "high", "low", "close", "volume", "returns", "vwap"]
    available = [c for c in core_cols if c in flat.columns]
    missing = flat.groupby("symbol")[available].apply(lambda g: g.isna().sum())

    # ------------------------------------------------------------------
    # 2. Stale prices (close unchanged for N+ days)
    # ------------------------------------------------------------------
    close_wide = flat.pivot(index="date", columns="symbol", values="close").sort_index()
    streak = pd.DataFrame(0, index=close_wide.index, columns=close_wide.columns)
    for col in close_wide.columns:
        s = close_wide[col]
        is_same = s == s.shift(1)
        # Count consecutive identical closes
        counter = np.zeros(len(s), dtype=int)
        for i in range(1, len(s)):
            counter[i] = counter[i-1] + 1 if is_same.iloc[i] else 0
        streak[col] = counter
    stale = streak

    # ------------------------------------------------------------------
    # 3. Return outliers
    # ------------------------------------------------------------------
    ret_wide = flat.pivot(index="date", columns="symbol", values="returns").sort_index()
    rolling_std = ret_wide.rolling(60, min_periods=20).std()
    abs_ret = ret_wide.abs()
    outlier_mask = abs_ret > (outlier_sigma * rolling_std)
    outlier_rows = []
    for col in ret_wide.columns:
        for date, flagged in outlier_mask[col].items():
            if flagged and not pd.isna(flagged):
                outlier_rows.append({
                    "date": date,
                    "symbol": col,
                    "return": ret_wide.loc[date, col],
                    "sigma_multiple": abs_ret.loc[date, col] / rolling_std.loc[date, col],
                })
    outliers = pd.DataFrame(outlier_rows)

    # ------------------------------------------------------------------
    # 4. Volume spikes
    # ------------------------------------------------------------------
    vol_wide = flat.pivot(index="date", columns="symbol", values="volume").sort_index()
    adv20 = vol_wide.rolling(20, min_periods=5).mean()
    spike_mask = vol_wide > (volume_spike_x * adv20)
    spike_rows = []
    for col in vol_wide.columns:
        for date, flagged in spike_mask[col].items():
            if flagged and not pd.isna(flagged):
                spike_rows.append({
                    "date": date,
                    "symbol": col,
                    "volume": vol_wide.loc[date, col],
                    "adv20": adv20.loc[date, col],
                    "multiple": vol_wide.loc[date, col] / adv20.loc[date, col],
                })
    vol_spikes = pd.DataFrame(spike_rows)

    # ------------------------------------------------------------------
    # 5. Zero-volume rows
    # ------------------------------------------------------------------
    zero_volume = flat[flat["volume"] == 0][["date", "symbol", "close", "volume"]].copy()

    # ------------------------------------------------------------------
    # 6. Calendar gaps (missing trading days per symbol)
    # ------------------------------------------------------------------
    all_dates = set(flat["date"].unique())
    gaps_info = []
    for sym, grp in flat.groupby("symbol"):
        sym_dates = set(grp["date"])
        missing_dates = sorted(all_dates - sym_dates)
        if missing_dates:
            gaps_info.append({"symbol": sym, "gaps": missing_dates, "n_gaps": len(missing_dates)})

    return DataQualityReport(
        missing=missing,
        stale=stale,
        outliers=outliers,
        vol_spikes=vol_spikes,
        zero_volume=zero_volume,
        gaps=gaps_info,
    )


def print_quality_report(panel: pd.DataFrame) -> DataQualityReport:
    """Convenience: run checks and print summary."""
    report = check_panel(panel)
    print(report.summary())
    if report.bad_symbols():
        print(f"\n  Symbols with >5% missing: {report.bad_symbols()}")
    return report
