"""Command-line entry-point for wqalpha research.

Usage
-----
    python -m wqalpha [--config configs/research.yaml] [--data PATH] [--top N]

Loads the data file specified in the config, registers all 101 alphas, computes
each signal, and prints a ranked IC / ICIR / hit-ratio summary table to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wqalpha",
        description="WorldQuant 101 Formulaic Alphas — research summary.",
    )
    p.add_argument(
        "--config",
        default="configs/research.yaml",
        help="Path to research.yaml config file (default: configs/research.yaml)",
    )
    p.add_argument(
        "--data",
        default=None,
        help="Override the data input path from config.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top alphas to display (default: 20).",
    )
    p.add_argument(
        "--sort-by",
        choices=["icir", "mean_ic", "hit_ratio"],
        default="icir",
        help="Column to sort the output table by (default: icir).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from wqalpha.alphas import register_all_alphas
    from wqalpha.config import load_config
    from wqalpha.data import DataLoader, wide
    from wqalpha.logging import configure_logging
    from wqalpha.metrics import forward_returns, hit_ratio, icir, information_coefficient
    from wqalpha.registry import AlphaRegistry

    configure_logging()

    args = _build_parser().parse_args(argv)

    # ------------------------------------------------------------------
    # Load config and data
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    data_path = args.data or cfg.get("data", {}).get("input_path", "data/india_equities_sample.csv")

    print(f"[wqalpha] Loading data from: {data_path}")
    try:
        panel = DataLoader.from_csv(data_path).load()
    except FileNotFoundError:
        print(f"[ERROR] Data file not found: {data_path}", file=sys.stderr)
        return 1

    print(f"[wqalpha] Panel: {len(panel):,} rows, "
          f"{panel.index.get_level_values('symbol').nunique()} symbols, "
          f"{panel.index.get_level_values('date').nunique()} dates.")

    # ------------------------------------------------------------------
    # Register alphas and compute signals
    # ------------------------------------------------------------------
    registry = AlphaRegistry()
    register_all_alphas(registry)

    returns_matrix = wide(panel, "returns")
    fwd1 = forward_returns(returns_matrix, horizon=1)

    annualization = cfg.get("reporting", {}).get("annualization", 252)

    rows: list[dict] = []
    print(f"[wqalpha] Computing {len(registry.names())} alphas ...")
    for name in registry.names():
        try:
            signal_series = registry.compute(name, panel)
            signal_matrix = signal_series.unstack("symbol").reindex(returns_matrix.index)
            ic = information_coefficient(signal_matrix, fwd1)
            mean_ic = float(ic.dropna().mean()) if ic.dropna().size else np.nan
            ir = icir(ic, annualization)
            hr = hit_ratio(ic)
            rows.append({"alpha": name, "mean_ic": mean_ic, "icir": ir, "hit_ratio": hr})
        except Exception as exc:
            rows.append({"alpha": name, "mean_ic": np.nan, "icir": np.nan, "hit_ratio": np.nan})
            print(f"  [WARN] {name} failed: {exc}")

    table = pd.DataFrame(rows).set_index("alpha")
    table = table.sort_values(args.sort_by, ascending=False, na_position="last")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", args.top + 5)

    print(f"\n{'='*55}")
    print(f"  Top {args.top} alphas ranked by {args.sort_by}")
    print(f"{'='*55}")
    print(table.head(args.top).to_string())
    print(f"\n[wqalpha] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
