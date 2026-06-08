"""ResearchDashboard — bundles all publication-quality charts into a single object."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from wqalpha.visualization import (
    plot_alpha_decay,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_drawdown,
    plot_factor_exposure,
    plot_ic_distribution,
    plot_ic_timeseries,
    plot_performance_table,
    plot_quantile_returns,
)


@dataclass
class ResearchDashboard:
    """Matplotlib dashboard bundle for publication-quality alpha review.

    Parameters
    ----------
    ic:
        Series of daily IC values.
    decay:
        Alpha decay series (index = forward horizon in days).
    returns:
        Strategy daily return series.
    exposures:
        Factor exposure Series from CAPM / Fama-French regression.
    alpha_matrix:
        Wide DataFrame of multiple alpha signals (date × alpha).
    signal:
        Optional signal matrix for quantile-return analysis.
    raw_returns:
        Optional returns matrix for quantile-return analysis.
    benchmark:
        Optional benchmark return series for cumulative return comparison.
    alpha_name:
        Optional alpha identifier for plot titles.
    """

    ic: pd.Series
    decay: pd.Series
    returns: pd.Series
    exposures: pd.Series
    alpha_matrix: pd.DataFrame
    signal: pd.DataFrame | None = field(default=None)
    raw_returns: pd.DataFrame | None = field(default=None)
    benchmark: pd.Series | None = field(default=None)
    alpha_name: str = ""
    performance_stats: pd.Series | None = field(default=None)

    def render(self) -> dict[str, object]:
        """Render all charts and return a dict of figure objects."""
        figs: dict[str, object] = {
            "ic_distribution": plot_ic_distribution(self.ic, self.alpha_name),
            "ic_timeseries":   plot_ic_timeseries(self.ic, alpha_name=self.alpha_name),
            "alpha_decay":     plot_alpha_decay(self.decay, self.alpha_name),
            "cumulative_returns": plot_cumulative_returns(
                self.returns, benchmark=self.benchmark, label=self.alpha_name or "Strategy"
            ),
            "drawdown": plot_drawdown(self.returns, label=self.alpha_name or "Strategy"),
            "factor_exposure": plot_factor_exposure(self.exposures),
            "correlation_heatmap": plot_correlation_heatmap(self.alpha_matrix),
        }
        if self.signal is not None and self.raw_returns is not None:
            figs["quantile_returns"] = plot_quantile_returns(self.signal, self.raw_returns)
        if self.performance_stats is not None:
            figs["performance_table"] = plot_performance_table(self.performance_stats)
        return figs
