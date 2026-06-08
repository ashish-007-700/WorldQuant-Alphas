# WQ Alpha India

A production-style Python research framework for implementing and validating the 101 Formulaic Alphas on Indian equity market data.

The framework assumes daily OHLCV data in long format with a `MultiIndex` of `date, symbol`.
Required columns are `open`, `high`, `low`, `close`, `volume`; optional columns include
`vwap`, `returns`, `sector`, `market_cap`, and risk-factor columns.

## Architecture

- `wqalpha.data`: Indian equity data loading, schema validation, return/VWAP enrichment.
- `wqalpha.features`: reusable market feature engineering.
- `wqalpha.operators`: vectorized cross-sectional and time-series alpha operators.
- `wqalpha.registry`: alpha metadata, registration, batch execution.
- `wqalpha.alphas`: 101 independent alpha callables.
- `wqalpha.metrics`: IC, ICIR, hit ratio, alpha decay, turnover.
- `wqalpha.backtest`: equal-weight long-short portfolio construction and performance.
- `wqalpha.risk`: CAPM, Fama-French 3/5, sector and beta neutralization.
- `wqalpha.visualization`: publication-quality research charts.

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

```python
from wqalpha.data import DataLoader
from wqalpha.alphas import register_all_alphas
from wqalpha.registry import AlphaRegistry

panel = DataLoader.from_csv("data/india_equities.csv").load()
registry = AlphaRegistry()
register_all_alphas(registry)
alpha_001 = registry.compute("alpha_001", panel)
```

## Data Contract

Input rows should contain:

```text
date,symbol,open,high,low,close,volume,vwap,sector,market_cap
```

Dates are converted to timezone-naive pandas timestamps and sorted as `(date, symbol)`.
