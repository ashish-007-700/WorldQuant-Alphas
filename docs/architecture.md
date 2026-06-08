# Architecture

## Research Flow

1. Load NSE/BSE daily OHLCV data with `DataLoader`.
2. Normalize to a `MultiIndex(date, symbol)` panel.
3. Convert required fields to date-by-symbol matrices through `FeatureEngine`.
4. Compute alpha signals through `AlphaRegistry`.
5. Validate predictive power with IC, ICIR, hit ratio, alpha decay, and turnover.
6. Construct an equal-weight top-decile/bottom-decile long-short book.
7. Measure Sharpe, Sortino, drawdown, CAGR, volatility, and turnover.
8. Regress returns against CAPM and Fama-French-style factor sets.
9. Apply sector or beta neutralization when needed.
10. Produce charts for research review and interview presentation.

## Production Notes

- Keep survivorship bias out of the security master.
- Use adjusted OHLCV data after dividends, splits, and symbol changes.
- Lag all signals before portfolio returns.
- Model Indian market frictions explicitly: STT, stamp duty, exchange fees, GST, slippage, and liquidity caps.
- Use sector mappings aligned to NSE/BSE/industry taxonomy.
- Treat the published formulas as research baselines, not deployable alpha by themselves.
