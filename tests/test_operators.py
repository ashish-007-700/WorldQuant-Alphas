from __future__ import annotations

import numpy as np
import pandas as pd

from wqalpha import operators as op


def test_rank_cross_sectional() -> None:
    x = pd.DataFrame([[1, 3, 2], [4, 4, 1]], columns=list("abc"))
    ranked = op.rank(x)
    assert ranked.loc[0, "b"] == 1.0
    assert ranked.loc[1, "c"] == 1 / 3


def test_delay_delta_and_scale() -> None:
    x = pd.DataFrame({"a": [1.0, 3.0, 6.0], "b": [2.0, 4.0, 8.0]})
    assert op.delay(x, 1).iloc[1, 0] == 1.0
    assert op.delta(x, 1).iloc[2, 1] == 4.0
    assert np.isclose(op.scale(x).abs().sum(axis=1).iloc[-1], 1.0)


def test_ts_rank_and_decay_linear() -> None:
    x = pd.DataFrame({"a": [1, 2, 3, 2, 5]}, dtype=float)
    assert op.ts_rank(x, 3).iloc[2, 0] == 1.0
    assert np.isclose(op.decay_linear(x, 3).iloc[2, 0], (1 + 4 + 9) / 6)


def test_rolling_extrema_and_correlation() -> None:
    x = pd.DataFrame({"a": [1, 2, 5, 4, 3]}, dtype=float)
    y = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, dtype=float)
    assert op.ts_max(x, 3).iloc[2, 0] == 5
    assert op.ts_min(x, 3).iloc[2, 0] == 1
    assert op.ts_argmax(x, 3).iloc[2, 0] == 3
    assert op.ts_argmin(x, 3).iloc[2, 0] == 1
    assert np.isfinite(op.correlation(x, y, 3).iloc[2, 0])
    assert np.isfinite(op.covariance(x, y, 3).iloc[2, 0])
    assert np.isfinite(op.stddev(x, 3).iloc[2, 0])
