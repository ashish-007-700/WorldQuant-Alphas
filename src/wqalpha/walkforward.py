"""Walk-forward validation engine for alpha strategies.

Provides expanding-window and rolling-window out-of-sample backtesting.
This is the standard approach for avoiding look-ahead bias and overfitting
in quantitative research.

Example
-------
    from wqalpha.walkforward import WalkForwardEngine
    from wqalpha.alphas import register_all_alphas
    from wqalpha.registry import AlphaRegistry

    registry = AlphaRegistry()
    register_all_alphas(registry)

    engine = WalkForwardEngine(
        panel=panel,
        registry=registry,
        alpha_name="alpha_001",
        n_train=504,   # 2 years
        n_test=63,     # 1 quarter OOS
        mode="expanding",
    )
    result = engine.run()
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wqalpha.backtest import long_short_weights, performance_stats, portfolio_returns
from wqalpha.data import wide
from wqalpha.metrics import forward_returns, icir, information_coefficient


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Stores per-fold and aggregate out-of-sample results."""
    alpha_name: str
    mode: str
    n_train: int
    n_test: int
    fold_returns: list[pd.Series] = field(default_factory=list)
    fold_ic: list[pd.Series] = field(default_factory=list)
    fold_dates: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)

    @property
    def oos_returns(self) -> pd.Series:
        """Concatenated out-of-sample daily portfolio returns."""
        if not self.fold_returns:
            return pd.Series(dtype=float)
        return pd.concat(self.fold_returns).sort_index().rename(f"{self.alpha_name}_oos")

    @property
    def oos_ic(self) -> pd.Series:
        """Concatenated out-of-sample daily IC values."""
        if not self.fold_ic:
            return pd.Series(dtype=float)
        return pd.concat(self.fold_ic).sort_index().rename(f"{self.alpha_name}_ic")

    def summary(self) -> pd.DataFrame:
        """Per-fold performance summary table."""
        rows = []
        for i, (fold_ret, fold_ic_s, (test_start, test_end)) in enumerate(
            zip(self.fold_returns, self.fold_ic, self.fold_dates), 1
        ):
            stats = performance_stats(fold_ret)
            ic_mean = fold_ic_s.dropna().mean() if len(fold_ic_s.dropna()) > 0 else np.nan
            ir = icir(fold_ic_s)
            rows.append({
                "fold": i,
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "sharpe": stats.get("sharpe", np.nan),
                "cagr": stats.get("cagr", np.nan),
                "max_drawdown": stats.get("max_drawdown", np.nan),
                "mean_ic": ic_mean,
                "icir": ir,
                "n_days": len(fold_ret.dropna()),
            })
        return pd.DataFrame(rows).set_index("fold")

    def aggregate_stats(self) -> pd.Series:
        """Full OOS performance statistics."""
        stats = performance_stats(self.oos_returns)
        ic = self.oos_ic
        stats["mean_ic"] = ic.dropna().mean()
        stats["icir"]    = icir(ic)
        stats["n_folds"] = len(self.fold_returns)
        stats["n_oos_days"] = len(self.oos_returns.dropna())
        return stats


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """Walk-forward validation for a single alpha.

    Parameters
    ----------
    panel:
        Long-format equity panel (date, symbol, OHLCV + returns).
    registry:
        :class:`~wqalpha.registry.AlphaRegistry` with the alpha registered.
    alpha_name:
        Name of the alpha to validate (e.g. ``"alpha_001"``).
    n_train:
        Number of training trading days per fold.
    n_test:
        Number of test (OOS) trading days per fold.
    mode:
        ``"expanding"`` (grows training window) or ``"rolling"`` (fixed window).
    transaction_cost_bps:
        Transaction cost in basis points (default: 10bps = 0.10%).
    long_quantile / short_quantile:
        Portfolio construction quantile thresholds.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        registry,
        alpha_name: str,
        n_train: int = 504,
        n_test: int = 63,
        mode: str = "expanding",
        transaction_cost_bps: float = 10.0,
        long_quantile: float = 0.90,
        short_quantile: float = 0.10,
    ):
        self.panel = panel
        self.registry = registry
        self.alpha_name = alpha_name
        self.n_train = n_train
        self.n_test = n_test
        self.mode = mode
        self.tc_bps = transaction_cost_bps
        self.long_q = long_quantile
        self.short_q = short_quantile

        # Get sorted unique dates
        if isinstance(panel.index, pd.MultiIndex):
            self.dates = sorted(panel.index.get_level_values("date").unique())
        else:
            self.dates = sorted(pd.to_datetime(panel["date"]).unique())

    def _get_panel_slice(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Extract a date-range slice of the panel."""
        start_date = self.dates[start_idx]
        end_date   = self.dates[end_idx - 1]
        if isinstance(self.panel.index, pd.MultiIndex):
            flat = self.panel.reset_index()
        else:
            flat = self.panel.copy()
        flat["date"] = pd.to_datetime(flat["date"])
        mask = (flat["date"] >= start_date) & (flat["date"] <= end_date)
        sub = flat[mask].set_index(["date", "symbol"]).sort_index()
        return sub

    def run(self) -> WalkForwardResult:
        """Execute all walk-forward folds.

        Returns
        -------
        :class:`WalkForwardResult` containing per-fold and aggregate statistics.
        """
        n_dates = len(self.dates)
        if n_dates < self.n_train + self.n_test:
            raise ValueError(
                f"Not enough dates ({n_dates}) for n_train={self.n_train} + n_test={self.n_test}"
            )

        result = WalkForwardResult(
            alpha_name=self.alpha_name,
            mode=self.mode,
            n_train=self.n_train,
            n_test=self.n_test,
        )

        fold = 1
        test_start_idx = self.n_train

        while test_start_idx + self.n_test <= n_dates:
            train_start_idx = 0 if self.mode == "expanding" else test_start_idx - self.n_train
            train_end_idx   = test_start_idx
            test_end_idx    = test_start_idx + self.n_test

            test_start_date = self.dates[test_start_idx]
            test_end_date   = self.dates[test_end_idx - 1]

            # --- Train: compute alpha on training data ---
            train_panel = self._get_panel_slice(train_start_idx, train_end_idx)
            try:
                signal_series = self.registry.compute(self.alpha_name, train_panel)
            except Exception as exc:
                print(f"  [WARN] Fold {fold}: signal computation failed: {exc}")
                test_start_idx += self.n_test
                fold += 1
                continue

            # --- Test: apply signal to OOS period ---
            test_panel  = self._get_panel_slice(test_start_idx, test_end_idx)
            ret_matrix  = wide(test_panel, "returns")

            # Use the last signal value from training as the OOS signal
            sig_matrix  = signal_series.unstack("symbol")
            last_signal = sig_matrix.iloc[[-1]].reindex(
                index=ret_matrix.index, method="ffill"
            ).reindex(columns=ret_matrix.columns)

            fwd1 = forward_returns(ret_matrix, horizon=1)
            ic   = information_coefficient(last_signal, fwd1)

            weights = long_short_weights(last_signal, self.long_q, self.short_q)
            pnl     = portfolio_returns(weights, ret_matrix, self.tc_bps)

            result.fold_returns.append(pnl)
            result.fold_ic.append(ic)
            result.fold_dates.append((test_start_date, test_end_date))

            test_start_idx += self.n_test
            fold += 1

        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def walk_forward_all(
    panel: pd.DataFrame,
    registry,
    alpha_names: list[str] | None = None,
    n_train: int = 504,
    n_test: int = 63,
    mode: str = "expanding",
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Run walk-forward validation for multiple alphas and return a comparison table.

    Returns
    -------
    DataFrame indexed by alpha name with columns:
    sharpe, cagr, max_drawdown, mean_ic, icir, n_folds.
    """
    names = alpha_names or registry.names()
    rows = []
    for i, name in enumerate(names, 1):
        print(f"  [{i:03d}/{len(names)}] Walk-forward: {name} ...", end="", flush=True)
        try:
            engine = WalkForwardEngine(panel, registry, name, n_train, n_test, mode,
                                       transaction_cost_bps)
            res = engine.run()
            agg = res.aggregate_stats()
            rows.append({"alpha": name, **agg.to_dict()})
            print(f" sharpe={agg.get('sharpe', float('nan')):.3f}")
        except Exception as exc:
            print(f" ERROR: {exc}")
            rows.append({"alpha": name})
    return pd.DataFrame(rows).set_index("alpha")
