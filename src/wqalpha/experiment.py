"""Lightweight experiment tracking — no external dependencies required.

Logs alpha research runs to a JSON-lines file (experiments/runs.jsonl) and
provides loading, comparison, and filtering utilities.

Example
-------
    from wqalpha.experiment import Experiment, ExperimentLog

    # Record a run
    with Experiment("alpha_001_baseline") as exp:
        exp.log_params({"long_q": 0.90, "tc_bps": 10})
        exp.log_metrics({"sharpe": 1.23, "icir": 0.45, "max_drawdown": -0.12})
        exp.log_tags({"phase": "in-sample", "universe": "nifty50"})

    # Compare runs
    log = ExperimentLog()
    log.load().sort_values("sharpe", ascending=False).head(20)
"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Experiment run
# ---------------------------------------------------------------------------

class Experiment:
    """Context manager that records a single research experiment run.

    Parameters
    ----------
    name:
        Human-readable experiment name (e.g. "alpha_001_sector_neutral").
    log_dir:
        Directory where ``runs.jsonl`` is stored (default: ``experiments/``).
    """

    def __init__(self, name: str, log_dir: str | Path = "experiments"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.run_id = str(uuid.uuid4())[:8]
        self._params: dict = {}
        self._metrics: dict = {}
        self._tags: dict = {}
        self._start: datetime | None = None
        self._end: datetime | None = None

    def __enter__(self) -> "Experiment":
        self._start = datetime.now(tz=timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end = datetime.now(tz=timezone.utc)
        status = "failed" if exc_type else "completed"
        self._save(status)

    def log_params(self, params: dict) -> "Experiment":
        """Log hyperparameters (e.g. lookback window, quantile thresholds)."""
        self._params.update(params)
        return self

    def log_metrics(self, metrics: dict) -> "Experiment":
        """Log numeric performance metrics (Sharpe, IC, etc.)."""
        self._metrics.update(metrics)
        return self

    def log_tags(self, tags: dict) -> "Experiment":
        """Log metadata tags (e.g. universe, phase, alpha_name)."""
        self._tags.update(tags)
        return self

    def _save(self, status: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id":    self.run_id,
            "name":      self.name,
            "status":    status,
            "started":   self._start.isoformat() if self._start else None,
            "ended":     self._end.isoformat() if self._end else None,
            "duration_s": (
                (self._end - self._start).total_seconds()
                if self._start and self._end else None
            ),
            "params":    self._params,
            "metrics":   self._metrics,
            "tags":      self._tags,
        }
        path = self.log_dir / "runs.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Experiment log reader
# ---------------------------------------------------------------------------

class ExperimentLog:
    """Reads and analyses the experiment run log.

    Parameters
    ----------
    log_dir:
        Directory containing ``runs.jsonl``.
    """

    def __init__(self, log_dir: str | Path = "experiments"):
        self.log_dir = Path(log_dir)
        self._runs_path = self.log_dir / "runs.jsonl"

    def load(self) -> pd.DataFrame:
        """Load all runs into a flat DataFrame.

        Metrics and params are flattened into columns.
        """
        if not self._runs_path.exists():
            return pd.DataFrame()

        rows = []
        with open(self._runs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                flat = {
                    "run_id": record.get("run_id"),
                    "name":   record.get("name"),
                    "status": record.get("status"),
                    "started": record.get("started"),
                    "duration_s": record.get("duration_s"),
                }
                flat.update({f"p_{k}": v for k, v in record.get("params", {}).items()})
                flat.update(record.get("metrics", {}))
                flat.update({f"t_{k}": v for k, v in record.get("tags", {}).items()})
                rows.append(flat)

        df = pd.DataFrame(rows)
        if "started" in df.columns:
            df["started"] = pd.to_datetime(df["started"])
        return df

    def compare(self, run_ids: list[str]) -> pd.DataFrame:
        """Side-by-side comparison of specific runs by run_id."""
        df = self.load()
        return df[df["run_id"].isin(run_ids)].set_index("run_id").T

    def best(self, metric: str = "sharpe", n: int = 10) -> pd.DataFrame:
        """Return the top N runs sorted by a metric."""
        df = self.load()
        if metric not in df.columns:
            raise KeyError(f"Metric '{metric}' not found. Available: {df.columns.tolist()}")
        return df.sort_values(metric, ascending=False).head(n)

    def filter(self, **kwargs) -> pd.DataFrame:
        """Filter runs by tag or param value (e.g. filter(t_universe='nifty50'))."""
        df = self.load()
        for k, v in kwargs.items():
            if k in df.columns:
                df = df[df[k] == v]
        return df

    def delete_run(self, run_id: str) -> None:
        """Remove a specific run from the log (rewrites the file)."""
        if not self._runs_path.exists():
            return
        lines = []
        with open(self._runs_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get("run_id") != run_id:
                    lines.append(line)
        with open(self._runs_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def clear(self) -> None:
        """Delete all experiment logs (use with caution)."""
        if self._runs_path.exists():
            self._runs_path.unlink()
