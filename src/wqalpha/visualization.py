"""Publication-quality research charts for the WQ Alpha India framework.

All plot functions return a ``matplotlib.figure.Figure`` that can be saved,
displayed in a notebook, or embedded in a :class:`ResearchDashboard`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _libs():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent dark-themed styling to an axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_ic_distribution(ic: pd.Series, alpha_name: str = "") -> object:
    """Histogram + KDE of daily Information Coefficients.

    Parameters
    ----------
    ic:
        Series of per-date IC values.
    alpha_name:
        Optional label for the plot title.
    """
    plt, sns = _libs()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(ic.dropna(), kde=True, bins=25, color="#4C9BE8", ax=ax, alpha=0.75)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    mean_ic = ic.dropna().mean()
    ax.axvline(mean_ic, color="#E84C4C", linestyle="-", linewidth=1.5,
               label=f"Mean IC = {mean_ic:.4f}")
    ax.legend(fontsize=9)
    title = f"IC Distribution — {alpha_name}" if alpha_name else "IC Distribution"
    _style(ax, title, "Information Coefficient")
    fig.tight_layout()
    return fig


def plot_ic_timeseries(ic: pd.Series, rolling_window: int = 63, alpha_name: str = "") -> object:
    """Rolling-mean IC timeseries with confidence band.

    Parameters
    ----------
    ic:
        Series of per-date IC values.
    rolling_window:
        Window for rolling mean overlay (default: 63 days ≈ 1 quarter).
    """
    plt, _ = _libs()
    fig, ax = plt.subplots(figsize=(12, 4))
    clean = ic.dropna()
    ax.bar(clean.index, clean.values, color=np.where(clean >= 0, "#4C9BE8", "#E84C4C"),
           width=1, alpha=0.5, label="Daily IC")
    rolling_mean = clean.rolling(rolling_window, min_periods=min(rolling_window // 2, 10)).mean()
    ax.plot(rolling_mean.index, rolling_mean.values, color="white", linewidth=1.8,
            label=f"{rolling_window}d Rolling Mean")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    title = f"IC Timeseries — {alpha_name}" if alpha_name else "IC Timeseries"
    _style(ax, title, ylabel="IC")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_alpha_decay(decay: pd.Series, alpha_name: str = "") -> object:
    """Bar chart of mean IC at each forward horizon.

    Parameters
    ----------
    decay:
        Series indexed 1..N of mean IC values from :func:`~wqalpha.metrics.alpha_decay`.
    """
    plt, _ = _libs()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C9BE8" if v >= 0 else "#E84C4C" for v in decay.values]
    ax.bar(decay.index.astype(str), decay.values, color=colors, edgecolor="none")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    title = f"Alpha Decay — {alpha_name}" if alpha_name else "Alpha Decay"
    _style(ax, title, "Forward Horizon (days)", "Mean IC")
    fig.tight_layout()
    return fig


def plot_cumulative_returns(returns: pd.Series, benchmark: pd.Series | None = None,
                             label: str = "Strategy") -> object:
    """Cumulative growth-of-1 curve with optional benchmark.

    Parameters
    ----------
    returns:
        Daily return series.
    benchmark:
        Optional benchmark return series (e.g., equal-weight index).
    label:
        Legend label for the strategy.
    """
    plt, _ = _libs()
    fig, ax = plt.subplots(figsize=(12, 5))
    cum = (1 + returns.dropna()).cumprod()
    ax.plot(cum.index, cum.values, color="#4C9BE8", linewidth=2, label=label)
    if benchmark is not None:
        cum_bench = (1 + benchmark.dropna()).cumprod().reindex(cum.index)
        ax.plot(cum_bench.index, cum_bench.values, color="grey", linewidth=1.2,
                linestyle="--", label="Benchmark")
    ax.axhline(1, color="grey", linestyle=":", linewidth=0.8)
    ax.fill_between(cum.index, 1, cum.values,
                    where=(cum.values >= 1), alpha=0.15, color="#4C9BE8")
    ax.fill_between(cum.index, 1, cum.values,
                    where=(cum.values < 1), alpha=0.15, color="#E84C4C")
    _style(ax, "Cumulative Returns", ylabel="Growth of ₹1")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_drawdown(returns: pd.Series, label: str = "Strategy") -> object:
    """Underwater (drawdown) chart."""
    plt, _ = _libs()
    fig, ax = plt.subplots(figsize=(12, 4))
    cum = (1 + returns.dropna()).cumprod()
    dd = cum / cum.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color="#E84C4C", alpha=0.6, label=label)
    ax.axhline(0, color="grey", linewidth=0.8)
    max_dd = dd.min()
    ax.axhline(max_dd, color="#E84C4C", linestyle="--", linewidth=1,
               label=f"Max DD = {max_dd:.2%}")
    _style(ax, "Drawdown", ylabel="Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_quantile_returns(signal: pd.DataFrame, returns: pd.DataFrame,
                           n_quantiles: int = 5, periods: int = 1) -> object:
    """Mean forward return per signal quantile (quintile bar chart).

    Parameters
    ----------
    signal:
        Date × symbol signal matrix.
    returns:
        Date × symbol returns matrix.
    n_quantiles:
        Number of quantile buckets (default: 5).
    periods:
        Forward return horizon in days (default: 1).
    """
    plt, _ = _libs()
    from wqalpha.metrics import forward_returns

    fwd = forward_returns(returns, periods)
    quantile_rets: dict[int, list[float]] = {q: [] for q in range(1, n_quantiles + 1)}
    common_dates = signal.index.intersection(fwd.index)
    for date in common_dates:
        s = signal.loc[date].dropna()
        r = fwd.loc[date].reindex(s.index).dropna()
        if len(r) < n_quantiles * 2:
            continue
        s, r = s.align(r, join="inner")
        labels = pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
        for q in range(n_quantiles):
            mask = labels == q
            if mask.any():
                quantile_rets[q + 1].append(r[mask].mean())

    means = [np.nanmean(quantile_rets[q]) for q in range(1, n_quantiles + 1)]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E84C4C" if m < 0 else "#4C9BE8" for m in means]
    ax.bar([f"Q{q}" for q in range(1, n_quantiles + 1)], means, color=colors)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2%}"))
    _style(ax, f"Mean {periods}d Forward Return by Quantile", "Quantile", "Mean Return")
    fig.tight_layout()
    return fig


def plot_factor_exposure(exposures: pd.Series, title: str = "Factor Exposure") -> object:
    """Horizontal bar chart of factor loadings from regression."""
    plt, _ = _libs()
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_data = exposures.drop(labels=["alpha", "r_squared"], errors="ignore")
    colors = ["#4C9BE8" if v >= 0 else "#E84C4C" for v in plot_data.values]
    ax.barh(plot_data.index, plot_data.values, color=colors)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    _style(ax, title, "Beta")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(signals: pd.DataFrame, title: str = "Alpha Correlation Heatmap") -> object:
    """Correlation heatmap across alpha signals."""
    plt, sns = _libs()
    corr = signals.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) // 2), max(6, len(corr) // 2)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap="vlag", center=0,
        vmin=-1, vmax=1, linewidths=0.3,
        annot=len(corr) <= 15, fmt=".2f", annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def plot_performance_table(stats: pd.Series) -> object:
    """Render performance statistics as a formatted table figure."""
    plt, _ = _libs()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    formats = {
        "sharpe": ".2f", "sortino": ".2f", "max_drawdown": ".2%",
        "cagr": ".2%", "volatility": ".2%", "turnover": ".2f",
    }
    rows = []
    for key, fmt in formats.items():
        if key in stats.index:
            val = stats[key]
            rows.append([key.replace("_", " ").title(),
                         f"{val:{fmt}}" if np.isfinite(val) else "N/A"])
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        if row == 0:
            cell.set_facecolor("#2E2E3A")
            cell.set_text_props(color="white", fontweight="bold")
    fig.tight_layout()
    return fig
