# Implementation Roadmap

## Phase 1: Research Baseline

- Load daily NSE/BSE panel data.
- Validate operators against deterministic unit tests.
- Compute all 101 signals and store outputs.
- Rank IC and ICIR by alpha and sector.

## Phase 2: Portfolio Research

- Add transaction cost and liquidity constraints.
- Evaluate top/bottom decile long-short portfolios.
- Report Sharpe, Sortino, maximum drawdown, CAGR, volatility, and turnover.
- Compare raw, sector-neutral, and beta-neutral variants.

## Phase 3: Risk Model

- Build Indian CAPM and local Fama-French factor files.
- Add rolling factor exposure diagnostics.
- Add residual alpha testing after risk neutralization.

## Phase 4: Production Hardening

- Add point-in-time security master and corporate action loader.
- Add experiment tracking and artifact storage.
- Add scheduled data-quality checks.
- Add walk-forward validation and capacity analysis.
