"""WorldQuant 101 Formulaic Alphas — Kakushadze (2016).

Each alpha function accepts a :class:`~wqalpha.features.FeatureEngine` and returns
a *date × symbol* ``pd.DataFrame`` of raw signal values.

Notes
-----
* ``IndNeutralize`` is approximated by sector-level neutralisation using
  the ``sector`` column when available; falls through to identity otherwise.
* Window parameters with decimals in the paper are rounded to integers.
* Alphas that require ``cap`` (market capitalisation) degrade gracefully
  when that column is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wqalpha import operators as op
from wqalpha.features import FeatureEngine
from wqalpha.registry import Alpha, AlphaRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(x: pd.DataFrame) -> pd.DataFrame:
    """Replace ±inf with NaN."""
    return x.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Alpha functions  (1 – 101)
# ---------------------------------------------------------------------------

def alpha_001(f: FeatureEngine) -> pd.DataFrame:
    """rank(Ts_ArgMax(SignedPower((returns<0 ? stddev(returns,20) : close), 2), 5)) - 0.5"""
    r, c = f.returns, f.close
    base = c.where(r >= 0, op.stddev(r, 20))
    return _clean(op.rank(op.ts_argmax(op.signed_power(base, 2.0), 5)) - 0.5)


def alpha_002(f: FeatureEngine) -> pd.DataFrame:
    """-1 * corr(rank(delta(log(volume), 2)), rank((close-open)/open), 6)"""
    v, c, o = f.volume, f.close, f.open
    return _clean(-1 * op.correlation(
        op.rank(op.delta(op.log(v), 2)),
        op.rank((c - o) / o),
        6,
    ))


def alpha_003(f: FeatureEngine) -> pd.DataFrame:
    """-1 * corr(rank(open), rank(volume), 10)"""
    return _clean(-1 * op.correlation(op.rank(f.open), op.rank(f.volume), 10))


def alpha_004(f: FeatureEngine) -> pd.DataFrame:
    """-1 * Ts_Rank(rank(low), 9)"""
    return _clean(-1 * op.ts_rank(op.rank(f.low), 9))


def alpha_005(f: FeatureEngine) -> pd.DataFrame:
    """rank((open - sum(vwap,10)/10)) * (-1 * abs(rank((close - vwap))))"""
    o, v, c = f.open, f.vwap, f.close
    return _clean(op.rank(o - op.ts_sum(v, 10) / 10) * (-1 * op.rank(c - v).abs()))


def alpha_006(f: FeatureEngine) -> pd.DataFrame:
    """-1 * corr(open, volume, 10)"""
    return _clean(-1 * op.correlation(f.open, f.volume, 10))


def alpha_007(f: FeatureEngine) -> pd.DataFrame:
    """(adv20 < volume) ? (-1 * ts_rank(abs(delta(close,7)),60) * sign(delta(close,7))) : -1"""
    c, vol, adv20 = f.close, f.volume, f.adv20
    d7 = op.delta(c, 7)
    active = (-1 * op.ts_rank(d7.abs(), 60)) * op.sign(d7)
    return _clean(active.where(adv20 < vol, -1.0))


def alpha_008(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank((sum(open,5)*sum(returns,5)) - delay(sum(open,5)*sum(returns,5), 10))"""
    o, r = f.open, f.returns
    x = op.ts_sum(o, 5) * op.ts_sum(r, 5)
    return _clean(-1 * op.rank(x - op.delay(x, 10)))


def alpha_009(f: FeatureEngine) -> pd.DataFrame:
    """ts_min(d,5)>0 ? d : ts_max(d,5)<0 ? d : -d  where d=delta(close,1)"""
    d = op.delta(f.close, 1)
    tmin5 = op.ts_min(d, 5)
    tmax5 = op.ts_max(d, 5)
    result = -d                         # default: tmax5 >= 0 and tmin5 <= 0
    result = result.where(tmax5 >= 0, d)  # ts_max < 0 → d
    result = result.where(tmin5 <= 0, d)  # ts_min > 0 → d (overwrites above)
    return _clean(result)


def alpha_010(f: FeatureEngine) -> pd.DataFrame:
    """rank(alpha_009-like with window 4)"""
    d = op.delta(f.close, 1)
    tmin4 = op.ts_min(d, 4)
    tmax4 = op.ts_max(d, 4)
    inner = -d
    inner = inner.where(tmax4 >= 0, d)
    inner = inner.where(tmin4 <= 0, d)
    return _clean(op.rank(inner))


def alpha_011(f: FeatureEngine) -> pd.DataFrame:
    """(rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))"""
    diff = f.vwap - f.close
    return _clean(
        (op.rank(op.ts_max(diff, 3)) + op.rank(op.ts_min(diff, 3)))
        * op.rank(op.delta(f.volume, 3))
    )


def alpha_012(f: FeatureEngine) -> pd.DataFrame:
    """sign(delta(volume,1)) * (-1 * delta(close,1))"""
    return _clean(op.sign(op.delta(f.volume, 1)) * (-1 * op.delta(f.close, 1)))


def alpha_013(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank(cov(rank(close), rank(volume), 5))"""
    return _clean(-1 * op.rank(op.covariance(op.rank(f.close), op.rank(f.volume), 5)))


def alpha_014(f: FeatureEngine) -> pd.DataFrame:
    """(-1 * rank(delta(returns,3))) * corr(open, volume, 10)"""
    return _clean(
        (-1 * op.rank(op.delta(f.returns, 3))) * op.correlation(f.open, f.volume, 10)
    )


def alpha_015(f: FeatureEngine) -> pd.DataFrame:
    """-1 * sum(rank(corr(rank(high), rank(volume), 3)), 3)"""
    return _clean(
        -1 * op.ts_sum(op.rank(op.correlation(op.rank(f.high), op.rank(f.volume), 3)), 3)
    )


def alpha_016(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank(cov(rank(high), rank(volume), 5))"""
    return _clean(-1 * op.rank(op.covariance(op.rank(f.high), op.rank(f.volume), 5)))


def alpha_017(f: FeatureEngine) -> pd.DataFrame:
    """(-1*rank(ts_rank(close,10))) * rank(delta(delta(close,1),1)) * rank(ts_rank(vol/adv20,5))"""
    c = f.close
    return _clean(
        (-1 * op.rank(op.ts_rank(c, 10)))
        * op.rank(op.delta(op.delta(c, 1), 1))
        * op.rank(op.ts_rank(f.volume / f.adv20, 5))
    )


def alpha_018(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank(stddev(|close-open|,5) + (close-open) + corr(close,open,10))"""
    c, o = f.close, f.open
    return _clean(-1 * op.rank(
        op.stddev((c - o).abs(), 5) + (c - o) + op.correlation(c, o, 10)
    ))


def alpha_019(f: FeatureEngine) -> pd.DataFrame:
    """(-1 * sign((close-delay(close,7)) + delta(close,7))) * (1+rank(1+sum(returns,250)))"""
    c, r = f.close, f.returns
    d7 = op.delta(c, 7)
    # Use min_periods=1 for the long-window sum so the signal is non-NaN even on
    # short panels (the ranking is still valid cross-sectionally).
    ret_sum = r.rolling(250, min_periods=1).sum()
    return _clean(
        (-1 * op.sign((c - op.delay(c, 7)) + d7))
        * (1 + op.rank(1 + ret_sum))
    )


def alpha_020(f: FeatureEngine) -> pd.DataFrame:
    """(-1*rank(open-delay(high,1))) * rank(open-delay(close,1)) * rank(open-delay(low,1))"""
    o, h, c, l = f.open, f.high, f.close, f.low
    return _clean(
        (-1 * op.rank(o - op.delay(h, 1)))
        * op.rank(o - op.delay(c, 1))
        * op.rank(o - op.delay(l, 1))
    )


def alpha_021(f: FeatureEngine) -> pd.DataFrame:
    """Multi-condition based on 8-period vs 2-period moving average + std, and volume/adv20."""
    c, v = f.close, f.volume
    adv20 = f.adv20
    sma8, std8, sma2 = op.ts_mean(c, 8), op.stddev(c, 8), op.ts_mean(c, 2)
    cond1 = (sma8 + std8) < sma2           # avg+std < avg2 → -1
    cond2 = sma2 < (sma8 - std8)           # avg2 < avg-std → +1
    cond3 = (v / adv20) >= 1.0             # high volume → +1
    result = pd.DataFrame(-1.0, index=c.index, columns=c.columns)
    result = result.where(~cond3, 1.0)     # volume rule
    result = result.where(~cond2, 1.0)     # ma spread rule
    result = result.where(~cond1, -1.0)   # ma+std rule (highest priority)
    return _clean(result)


def alpha_022(f: FeatureEngine) -> pd.DataFrame:
    """-1 * delta(corr(high, volume, 5), 5) * rank(stddev(close, 20))"""
    return _clean(
        -1 * op.delta(op.correlation(f.high, f.volume, 5), 5)
        * op.rank(op.stddev(f.close, 20))
    )


def alpha_023(f: FeatureEngine) -> pd.DataFrame:
    """(sum(high,20)/20 < high) ? -1*delta(high,2) : 0"""
    h = f.high
    return _clean((-1 * op.delta(h, 2)).where(op.ts_mean(h, 20) < h, 0.0))


def alpha_024(f: FeatureEngine) -> pd.DataFrame:
    """delta(sum(close,100)/100,100)/delay(close,100) <= 0.05 ? -1*(close-ts_min(close,100)) : -delta(close,3)"""
    c = f.close
    sma100 = op.ts_mean(c, 100)
    ratio = op.delta(sma100, 100) / op.delay(c, 100)
    return _clean(
        (-1 * (c - op.ts_min(c, 100))).where(ratio <= 0.05, -1 * op.delta(c, 3))
    )


def alpha_025(f: FeatureEngine) -> pd.DataFrame:
    """rank((-1*returns) * adv20 * vwap * (high - close))"""
    return _clean(op.rank((-1 * f.returns) * f.adv20 * f.vwap * (f.high - f.close)))


def alpha_026(f: FeatureEngine) -> pd.DataFrame:
    """-1 * ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)"""
    return _clean(
        -1 * op.ts_max(
            op.correlation(op.ts_rank(f.volume, 5), op.ts_rank(f.high, 5), 5), 3
        )
    )


def alpha_027(f: FeatureEngine) -> pd.DataFrame:
    """0.5 < rank(sum(corr(rank(volume), rank(vwap), 6), 2)/2) ? -1 : 1"""
    x = op.rank(op.ts_sum(op.correlation(op.rank(f.volume), op.rank(f.vwap), 6), 2) / 2.0)
    result = pd.DataFrame(1.0, index=x.index, columns=x.columns)
    return _clean(result.where(x <= 0.5, -1.0))


def alpha_028(f: FeatureEngine) -> pd.DataFrame:
    """scale((corr(adv20, low, 5) + (high+low)/2) - close)"""
    return _clean(
        op.scale(op.correlation(f.adv20, f.low, 5) + (f.high + f.low) / 2 - f.close)
    )


def alpha_029(f: FeatureEngine) -> pd.DataFrame:
    """Complex nested rank/scale/log expression + ts_rank(delay(-returns,6),5)"""
    c = f.close
    inner = -1 * op.rank(op.delta(c - 1, 5))
    step = op.rank(op.rank(inner))
    step = op.ts_min(step, 2)
    step = op.scale(op.log(step.abs() + 1e-12))
    step = op.rank(op.rank(step))
    part1 = op.ts_min(step, 5)
    part2 = op.ts_rank(op.delay(-1 * f.returns, 6), 5)
    return _clean(part1 + part2)


def alpha_030(f: FeatureEngine) -> pd.DataFrame:
    """(1 - rank(sign(close-d1) + sign(d1-d2) + sign(d2-d3))) * sum(vol,5) / sum(vol,20)"""
    c, v = f.close, f.volume
    d1, d2, d3 = op.delay(c, 1), op.delay(c, 2), op.delay(c, 3)
    s = op.sign(c - d1) + op.sign(d1 - d2) + op.sign(d2 - d3)
    return _clean((1.0 - op.rank(s)) * op.ts_sum(v, 5) / op.ts_sum(v, 20))


def alpha_031(f: FeatureEngine) -> pd.DataFrame:
    """rank(rank(rank(decay_linear(-rank(rank(delta(close,10))),10)))) + rank(-delta(close,3)) + sign(scale(corr(adv20,low,12)))"""
    c = f.close
    part1 = op.rank(op.rank(op.rank(
        op.decay_linear(-1 * op.rank(op.rank(op.delta(c, 10))), 10)
    )))
    part2 = op.rank(-1 * op.delta(c, 3))
    part3 = op.sign(op.scale(op.correlation(f.adv20, f.low, 12)))
    return _clean(part1 + part2 + part3)


def alpha_032(f: FeatureEngine) -> pd.DataFrame:
    """scale((sum(close,7)/7 - close)) + 20*scale(corr(vwap, delay(close,5), 230))"""
    c, v = f.close, f.vwap
    # Use min_periods=20 so the long-window correlation degrades gracefully on
    # short panels (otherwise the whole alpha is all-NaN).
    corr_part = v.rolling(230, min_periods=20).corr(op.delay(c, 5))
    return _clean(
        op.scale(op.ts_mean(c, 7) - c)
        + 20 * op.scale(corr_part)
    )


def alpha_033(f: FeatureEngine) -> pd.DataFrame:
    """rank(-1 * (1 - open/close))"""
    return _clean(op.rank(-1 * (1 - f.open / f.close)))


def alpha_034(f: FeatureEngine) -> pd.DataFrame:
    """rank((1 - rank(std(returns,2)/std(returns,5))) + (1 - rank(delta(close,1))))"""
    r, c = f.returns, f.close
    return _clean(op.rank(
        (1 - op.rank(op.stddev(r, 2) / op.stddev(r, 5)))
        + (1 - op.rank(op.delta(c, 1)))
    ))


def alpha_035(f: FeatureEngine) -> pd.DataFrame:
    """ts_rank(volume,32) * (1-ts_rank((close+high)-low,16)) * (1-ts_rank(returns,32))"""
    return _clean(
        op.ts_rank(f.volume, 32)
        * (1 - op.ts_rank((f.close + f.high) - f.low, 16))
        * (1 - op.ts_rank(f.returns, 32))
    )


def alpha_036(f: FeatureEngine) -> pd.DataFrame:
    """Weighted sum of 5 sub-expressions involving corr, rank, ts_rank, std."""
    c, o, v = f.close, f.open, f.volume
    vwap, adv20, r = f.vwap, f.adv20, f.returns
    return _clean(
        2.21 * op.rank(op.correlation(c - o, op.delay(v, 1), 15))
        + 0.7  * op.rank(o - c)
        + 0.73 * op.rank(op.ts_rank(op.delay(-1 * r, 6), 5))
        + op.rank(op.correlation(vwap, adv20, 6).abs())
        + 0.45 * op.rank((op.ts_mean(r, 20) - r) / op.stddev(r, 20))
    )


def alpha_037(f: FeatureEngine) -> pd.DataFrame:
    """rank(corr(delay(open-close,1), close, 200)) + rank(open-close)"""
    o, c = f.open, f.close
    # min_periods=20 so the long-window correlation works on short panels too
    corr_part = op.delay(o - c, 1).rolling(200, min_periods=20).corr(c)
    return _clean(op.rank(corr_part) + op.rank(o - c))


def alpha_038(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank(ts_rank(close,10)) * rank(close/open)"""
    c, o = f.close, f.open
    return _clean(-1 * op.rank(op.ts_rank(c, 10)) * op.rank(c / o))


def alpha_039(f: FeatureEngine) -> pd.DataFrame:
    """(-1*rank(delta(close,7)*(1-rank(decay_linear(vol/adv20,9))))) * (1+rank(sum(returns,250)))"""
    c, r = f.close, f.returns
    ret_sum = r.rolling(250, min_periods=1).sum()
    return _clean(
        (-1 * op.rank(op.delta(c, 7) * (1 - op.rank(op.decay_linear(f.volume / f.adv20, 9)))))
        * (1 + op.rank(ret_sum))
    )


def alpha_040(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank(std(high,10)) * corr(high, volume, 10)"""
    return _clean(-1 * op.rank(op.stddev(f.high, 10)) * op.correlation(f.high, f.volume, 10))


def alpha_041(f: FeatureEngine) -> pd.DataFrame:
    """(high*low)^0.5 - vwap"""
    return _clean(op.signed_power(f.high * f.low, 0.5) - f.vwap)


def alpha_042(f: FeatureEngine) -> pd.DataFrame:
    """rank(vwap-close) / rank(vwap+close)"""
    v, c = f.vwap, f.close
    return _clean(op.rank(v - c) / op.rank(v + c))


def alpha_043(f: FeatureEngine) -> pd.DataFrame:
    """ts_rank(vol/adv20, 20) * ts_rank(-1*delta(close,7), 8)"""
    return _clean(
        op.ts_rank(f.volume / f.adv20, 20)
        * op.ts_rank(-1 * op.delta(f.close, 7), 8)
    )


def alpha_044(f: FeatureEngine) -> pd.DataFrame:
    """-1 * corr(high, rank(volume), 5)"""
    return _clean(-1 * op.correlation(f.high, op.rank(f.volume), 5))


def alpha_045(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank(sum(delay(close,5),20)/20 * corr(close,vol,2) * rank(corr(sum(close,5),sum(close,20),2)))"""
    c, v = f.close, f.volume
    return _clean(-1 * (
        op.rank(op.ts_mean(op.delay(c, 5), 20))
        * op.correlation(c, v, 2)
        * op.rank(op.correlation(op.ts_sum(c, 5), op.ts_sum(c, 20), 2))
    ))


def alpha_046(f: FeatureEngine) -> pd.DataFrame:
    """Trend acceleration: if mid>0.25 → -1; mid<0 → 1; else -delta(close,1)"""
    c = f.close
    mid = (op.delay(c, 20) - op.delay(c, 10)) / 10 - (op.delay(c, 10) - c) / 10
    d1 = op.delta(c, 1)
    result = -1 * d1                                   # default (-0.1 <= mid <= 0.25)
    result = result.where(mid >= -0.1, 1.0)           # mid < -0.1 → 1 (approx mid < 0)
    result = result.where(mid <= 0.25, -1.0)          # mid > 0.25 → -1
    return _clean(result)


def alpha_047(f: FeatureEngine) -> pd.DataFrame:
    """(1/close)*vol/adv20 * rank((high-close)/(sum(high,5)/5)) - rank(vwap-delay(vwap,5))"""
    c, h, v = f.close, f.high, f.volume
    adv20, vwap = f.adv20, f.vwap
    return _clean(
        (1 / c) * op.rank((h - c) / (op.ts_mean(h, 5))) * (v / adv20)
        - op.rank(vwap - op.delay(vwap, 5))
    )


def alpha_048(f: FeatureEngine) -> pd.DataFrame:
    """corr(delta(close,1), delta(delay(close,1),1), 250)*delta(close,1)/close / sum((d/delay(close,1))^2, 20)"""
    c = f.close
    d1 = op.delta(c, 1)
    d1_lag = op.delta(op.delay(c, 1), 1)
    corr250 = op.correlation(d1, d1_lag, 250)
    numer = corr250 * d1 / c
    denom = op.ts_sum((d1 / op.delay(c, 1)) ** 2, 20)
    return _clean(numer / denom)


def alpha_049(f: FeatureEngine) -> pd.DataFrame:
    """mid >= -0.1 → 1; else -1*delta(close,1)  [trend deceleration filter]"""
    c = f.close
    mid = (op.delay(c, 20) - op.delay(c, 10)) / 10 - (op.delay(c, 10) - c) / 10
    d1 = op.delta(c, 1)
    result = pd.DataFrame(1.0, index=c.index, columns=c.columns)
    result = result.where(mid >= -0.1, -1 * d1)
    return _clean(result)


def alpha_050(f: FeatureEngine) -> pd.DataFrame:
    """-1 * ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)"""
    return _clean(
        -1 * op.ts_max(op.rank(op.correlation(op.rank(f.volume), op.rank(f.vwap), 5)), 5)
    )


def alpha_051(f: FeatureEngine) -> pd.DataFrame:
    """mid >= -0.05 → 1; else -1*delta(close,1)"""
    c = f.close
    mid = (op.delay(c, 20) - op.delay(c, 10)) / 10 - (op.delay(c, 10) - c) / 10
    d1 = op.delta(c, 1)
    result = pd.DataFrame(1.0, index=c.index, columns=c.columns)
    result = result.where(mid >= -0.05, -1 * d1)
    return _clean(result)


def alpha_052(f: FeatureEngine) -> pd.DataFrame:
    """-1*(ts_min(low,5)-delay(ts_min(low,5),5))*rank((sum(returns,240)-sum(returns,20))/220)*ts_rank(vol,5)"""
    l, v, r = f.low, f.volume, f.returns
    tmin5 = op.ts_min(l, 5)
    momentum = (op.ts_sum(r, 240) - op.ts_sum(r, 20)) / 220
    return _clean(
        -(tmin5 - op.delay(tmin5, 5))
        * op.rank(momentum)
        * op.ts_rank(v, 5)
    )


def alpha_053(f: FeatureEngine) -> pd.DataFrame:
    """-1 * delta(((close-low)-(high-close))/(close-low+1e-6), 9)"""
    c, h, l = f.close, f.high, f.low
    x = ((c - l) - (h - c)) / (c - l + 1e-6)
    return _clean(-1 * op.delta(x, 9))


def alpha_054(f: FeatureEngine) -> pd.DataFrame:
    """-(low-close)*(open^5) / ((low-high)*(close^5) + 1e-6)"""
    o, h, l, c = f.open, f.high, f.low, f.close
    return _clean(
        -(l - c) * op.signed_power(o, 5)
        / ((l - h) * op.signed_power(c, 5) + 1e-6)
    )


def alpha_055(f: FeatureEngine) -> pd.DataFrame:
    """-1 * corr(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))), rank(volume), 6)"""
    c, h, l, v = f.close, f.high, f.low, f.volume
    norm = (c - op.ts_min(l, 12)) / (op.ts_max(h, 12) - op.ts_min(l, 12) + 1e-6)
    return _clean(-1 * op.correlation(op.rank(norm), op.rank(v), 6))


def alpha_056(f: FeatureEngine) -> pd.DataFrame:
    """-(rank(sum(returns,10)/sum(sum(returns,2),3)) * rank(returns*(cap or 1)))"""
    r = f.returns
    cap = f.market_cap
    cap_proxy = cap if cap is not None else pd.DataFrame(1.0, index=r.index, columns=r.columns)
    return _clean(
        -(op.rank(op.ts_sum(r, 10) / op.ts_sum(op.ts_sum(r, 2), 3))
          * op.rank(r * cap_proxy))
    )


def alpha_057(f: FeatureEngine) -> pd.DataFrame:
    """-(close-vwap) / decay_linear(rank(ts_argmax(close,30)), 2)"""
    c, v = f.close, f.vwap
    return _clean(
        -((c - v) / op.decay_linear(op.rank(op.ts_argmax(c, 30)), 2))
    )


def alpha_058(f: FeatureEngine) -> pd.DataFrame:
    """-1 * ts_rank(decay_linear(corr(indneutralize(vwap), volume, 4), 7), 6)"""
    vwap_n = f.indneutralize(f.vwap)
    return _clean(
        -1 * op.ts_rank(op.decay_linear(op.correlation(vwap_n, f.volume, 4), 7), 6)
    )


def alpha_059(f: FeatureEngine) -> pd.DataFrame:
    """-1 * ts_rank(decay_linear(corr(indneutralize(vwap*0.728+vwap*0.272), volume, 4), 16), 9)"""
    vwap = f.vwap
    blended_n = f.indneutralize(vwap * 0.728317 + vwap * (1 - 0.728317))
    return _clean(
        -1 * op.ts_rank(op.decay_linear(op.correlation(blended_n, f.volume, 4), 16), 9)
    )


def alpha_060(f: FeatureEngine) -> pd.DataFrame:
    """0 - (2*scale(rank(((close-low)-(high-close))/(high-low+1e-6)*vol)) - scale(rank(ts_argmin(close,10))))"""
    c, h, l, v = f.close, f.high, f.low, f.volume
    intraday = ((c - l) - (h - c)) / (h - l + 1e-6) * v
    return _clean(
        -(2 * op.scale(op.rank(intraday)) - op.scale(op.rank(op.ts_argmin(c, 10))))
    )


def alpha_061(f: FeatureEngine) -> pd.DataFrame:
    """rank(vwap-ts_min(vwap,16)) < rank(corr(vwap, adv180, 18)) ? 1 : 0"""
    vwap, adv180 = f.vwap, f.adv180
    lhs = op.rank(vwap - op.ts_min(vwap, 16))
    rhs = op.rank(op.correlation(vwap, adv180, 18))
    return _clean((lhs < rhs).astype(float))


def alpha_062(f: FeatureEngine) -> pd.DataFrame:
    """rank(corr(vwap, sum(adv20,22), 10)) < rank(rank(open)+rank(open) < rank((high+low)/2)+rank(high)) ? -1 : 1"""
    vwap, o, h, l = f.vwap, f.open, f.high, f.low
    lhs = op.rank(op.correlation(vwap, op.ts_sum(f.adv20, 22), 10))
    rhs = op.rank(
        (op.rank(o) + op.rank(o)) < (op.rank((h + l) / 2) + op.rank(h))
    )
    result = pd.DataFrame(1.0, index=vwap.index, columns=vwap.columns)
    return _clean(result.where(lhs >= rhs, -1.0))


def alpha_063(f: FeatureEngine) -> pd.DataFrame:
    """-1*rank(decay_linear(indneutral(delta(close,2)),8)) - rank(decay_linear(-1*corr(indneutral(vwap),adv180,13),4))"""
    c_n = f.indneutralize(f.close)
    vwap_n = f.indneutralize(f.vwap)
    part1 = op.rank(op.decay_linear(op.delta(c_n, 2), 8))
    part2 = op.rank(op.decay_linear(-1 * op.correlation(vwap_n, f.adv180, 13), 4))
    return _clean(-1 * part1 - part2)


def alpha_064(f: FeatureEngine) -> pd.DataFrame:
    """abs(rank(corr(sum(open*0.178+low*0.822,13),sum(adv120,13),17))) < rank(delta((h+l)/2*0.178+vwap*0.822,3)) ? 1 : -1"""
    o, l, h, vwap = f.open, f.low, f.high, f.vwap
    adv120 = f.adv120
    blend = o * 0.178404 + l * (1 - 0.178404)
    blend2 = (h + l) / 2 * 0.178404 + vwap * (1 - 0.178404)
    lhs = op.rank(op.correlation(op.ts_sum(blend, 13), op.ts_sum(adv120, 13), 17)).abs()
    rhs = op.rank(op.delta(blend2, 3))
    result = pd.DataFrame(-1.0, index=o.index, columns=o.columns)
    return _clean(result.where(lhs >= rhs, 1.0))


def alpha_065(f: FeatureEngine) -> pd.DataFrame:
    """rank(corr(open*0.008+vwap*0.992, sum(adv60,9), 6)) < rank(open-ts_min(open,14)) ? -1 : 1"""
    o, vwap = f.open, f.vwap
    adv60 = f.adv60
    blend = o * 0.00817205 + vwap * (1 - 0.00817205)
    lhs = op.rank(op.correlation(blend, op.ts_sum(adv60, 9), 6))
    rhs = op.rank(o - op.ts_min(o, 14))
    result = pd.DataFrame(1.0, index=o.index, columns=o.columns)
    return _clean(result.where(lhs >= rhs, -1.0))


def alpha_066(f: FeatureEngine) -> pd.DataFrame:
    """-1*(rank(decay_linear(delta(vwap,4),7)) + ts_rank(decay_linear((low-vwap)/(open-(h+l)/2+1e-6),11),7))"""
    vwap, l, o, h = f.vwap, f.low, f.open, f.high
    part1 = op.rank(op.decay_linear(op.delta(vwap, 4), 7))
    ratio = (l * 0.96633 + l * (1 - 0.96633) - vwap) / (o - (h + l) / 2 + 1e-6)
    part2 = op.ts_rank(op.decay_linear(ratio, 11), 7)
    return _clean(-1 * (part1 + part2))


def alpha_067(f: FeatureEngine) -> pd.DataFrame:
    """rank(high-ts_min(high,2)) * rank(rank(corr(indneutral(vwap), indneutral(adv20), 8)))"""
    h = f.high
    vwap_n = f.indneutralize(f.vwap)
    adv20_n = f.indneutralize(f.adv20)
    return _clean(
        op.rank(h - op.ts_min(h, 2))
        * op.rank(op.rank(op.correlation(vwap_n, adv20_n, 8)))
    )


def alpha_068(f: FeatureEngine) -> pd.DataFrame:
    """ts_rank(corr(rank(high), rank(adv15), 9), 14) < rank(delta(close*0.518+low*0.482,1)) ? -1 : 1"""
    h, l, c = f.high, f.low, f.close
    adv15 = f.adv15
    lhs = op.ts_rank(op.correlation(op.rank(h), op.rank(adv15), 9), 14)
    rhs = op.rank(op.delta(c * 0.518371 + l * (1 - 0.518371), 1))
    result = pd.DataFrame(1.0, index=h.index, columns=h.columns)
    return _clean(result.where(lhs >= rhs, -1.0))


def alpha_069(f: FeatureEngine) -> pd.DataFrame:
    """-1*rank(ts_max(delta(indneutral(vwap),2),4)) * rank(corr(indneutral(close),adv20,8))"""
    vwap_n = f.indneutralize(f.vwap)
    close_n = f.indneutralize(f.close)
    return _clean(
        -1 * op.rank(op.ts_max(op.delta(vwap_n, 2), 4))
        * op.rank(op.correlation(close_n, f.adv20, 8))
    )


def alpha_070(f: FeatureEngine) -> pd.DataFrame:
    """rank(delta(vwap,1)) * rank(corr(indneutral(close), adv50, 18))"""
    close_n = f.indneutralize(f.close)
    return _clean(
        op.rank(op.delta(f.vwap, 1))
        * op.rank(op.correlation(close_n, f.adv50, 18))
    )


def alpha_071(f: FeatureEngine) -> pd.DataFrame:
    """max(ts_rank(decay_linear(corr(ts_rank(close,3),ts_rank(adv180,12),18),4),16),
           ts_rank(decay_linear(rank(low+open-vwap*2),16),4))"""
    c, l, o, vwap = f.close, f.low, f.open, f.vwap
    adv180 = f.adv180
    part1 = op.ts_rank(op.decay_linear(
        op.correlation(op.ts_rank(c, 3), op.ts_rank(adv180, 12), 18), 4), 16)
    part2 = op.ts_rank(op.decay_linear(op.rank(l + o - vwap * 2), 16), 4)
    return _clean(op.max_(part1, part2))


def alpha_072(f: FeatureEngine) -> pd.DataFrame:
    """rank(decay_linear(corr((h+l)/2, adv40, 9), 10)) / rank(decay_linear(corr(ts_rank(vwap,4),ts_rank(vol,19),7),3))"""
    h, l, vwap, v = f.high, f.low, f.vwap, f.volume
    adv40 = f.adv40
    numer = op.rank(op.decay_linear(op.correlation((h + l) / 2, adv40, 9), 10))
    denom = op.rank(op.decay_linear(op.correlation(op.ts_rank(vwap, 4), op.ts_rank(v, 19), 7), 3))
    return _clean(numer / denom)


def alpha_073(f: FeatureEngine) -> pd.DataFrame:
    """-1*max(rank(decay_linear(delta(vwap,5),3)),
              ts_rank(decay_linear(-1*delta(low*0.147+open*0.853,2)/(low*0.147+open*0.853),3),17))"""
    vwap, l, o = f.vwap, f.low, f.open
    blend = l * 0.147155 + o * (1 - 0.147155)
    part1 = op.rank(op.decay_linear(op.delta(vwap, 5), 3))
    part2 = op.ts_rank(op.decay_linear(-1 * op.delta(blend, 2) / blend, 3), 17)
    return _clean(-1 * op.max_(part1, part2))


def alpha_074(f: FeatureEngine) -> pd.DataFrame:
    """rank(corr(close, sum(adv30,37), 15)) < rank(corr(rank(high*0.026+vwap*0.974), rank(vol), 11)) ? -1 : 1"""
    c, h, vwap, v = f.close, f.high, f.vwap, f.volume
    adv30 = f.adv30
    lhs = op.rank(op.correlation(c, op.ts_sum(adv30, 37), 15))
    blend = h * 0.0261661 + vwap * (1 - 0.0261661)
    rhs = op.rank(op.correlation(op.rank(blend), op.rank(v), 11))
    result = pd.DataFrame(1.0, index=c.index, columns=c.columns)
    return _clean(result.where(lhs >= rhs, -1.0))


def alpha_075(f: FeatureEngine) -> pd.DataFrame:
    """rank(corr(vwap, vol, 4)) < rank(corr(rank(low), rank(adv50), 12)) ? 1 : 0"""
    vwap, v = f.vwap, f.volume
    adv50 = f.adv50
    lhs = op.rank(op.correlation(vwap, v, 4))
    rhs = op.rank(op.correlation(op.rank(f.low), op.rank(adv50), 12))
    return _clean((lhs < rhs).astype(float))


def alpha_076(f: FeatureEngine) -> pd.DataFrame:
    """-1*max(rank(decay_linear(delta(vwap,1),12)),
              ts_rank(decay_linear(ts_rank(corr(indneutral(low),adv81,8),20),17),19))"""
    vwap = f.vwap
    low_n = f.indneutralize(f.low)
    adv81 = f.adv81
    part1 = op.rank(op.decay_linear(op.delta(vwap, 1), 12))
    part2 = op.ts_rank(op.decay_linear(
        op.ts_rank(op.correlation(low_n, adv81, 8), 20), 17), 19)
    return _clean(-1 * op.max_(part1, part2))


def alpha_077(f: FeatureEngine) -> pd.DataFrame:
    """min(rank(decay_linear(((h+l)/2+h)-(vwap+h),20)),
           rank(decay_linear(corr((h+l)/2, adv40, 3), 6)))"""
    h, l, vwap = f.high, f.low, f.vwap
    adv40 = f.adv40
    part1 = op.rank(op.decay_linear(((h + l) / 2 + h) - (vwap + h), 20))
    part2 = op.rank(op.decay_linear(op.correlation((h + l) / 2, adv40, 3), 6))
    return _clean(op.min_(part1, part2))


def alpha_078(f: FeatureEngine) -> pd.DataFrame:
    """rank(corr(sum(low*0.352+vwap*0.648,20), sum(adv40,20), 7)) / rank(corr(rank(vwap),rank(vol),6))"""
    l, vwap, v = f.low, f.vwap, f.volume
    adv40 = f.adv40
    blend = l * 0.352233 + vwap * (1 - 0.352233)
    numer = op.rank(op.correlation(op.ts_sum(blend, 20), op.ts_sum(adv40, 20), 7))
    denom = op.rank(op.correlation(op.rank(vwap), op.rank(v), 6))
    return _clean(numer / denom)


def alpha_079(f: FeatureEngine) -> pd.DataFrame:
    """rank(delta(indneutral(close*0.607+open*0.393),1)) < rank(corr(ts_rank(vwap,4),ts_rank(adv150,8),15)) ? 1 : -1"""
    c, o, vwap = f.close, f.open, f.vwap
    adv150 = f.adv150
    blend_n = f.indneutralize(c * 0.60733 + o * (1 - 0.60733))
    lhs = op.rank(op.delta(blend_n, 1))
    rhs = op.rank(op.correlation(op.ts_rank(vwap, 4), op.ts_rank(adv150, 8), 15))
    result = pd.DataFrame(1.0, index=c.index, columns=c.columns)
    return _clean(result.where(lhs < rhs, -1.0))


def alpha_080(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank(sign(delta(indneutral(open*0.868+high*0.132),4)))^2"""
    o, h = f.open, f.high
    blend_n = f.indneutralize(o * 0.868128 + h * (1 - 0.868128))
    return _clean(-1 * op.rank(op.sign(op.delta(blend_n, 4))) ** 2)


def alpha_081(f: FeatureEngine) -> pd.DataFrame:
    """rank(log(product(rank(rank(corr(vwap,sum(adv10,49),8))^4),15))) - rank(corr(rank(vwap),rank(vol),5))"""
    vwap, v = f.vwap, f.volume
    adv10 = f.adv10
    inner = op.rank(op.rank(op.correlation(vwap, op.ts_sum(adv10, 49), 8)) ** 4)
    part1 = op.rank(op.log(op.product(inner, 15)))
    part2 = op.rank(op.correlation(op.rank(vwap), op.rank(v), 5))
    return _clean(part1 - part2)


def alpha_082(f: FeatureEngine) -> pd.DataFrame:
    """-1*min(rank(decay_linear(delta(open,1),15)),
              ts_rank(decay_linear(corr(indneutral(vol),open*0.634+open*0.366,17),7),13))"""
    o, v = f.open, f.volume
    vol_n = f.indneutralize(v)
    blend = o * 0.634196 + o * (1 - 0.634196)
    part1 = op.rank(op.decay_linear(op.delta(o, 1), 15))
    part2 = op.ts_rank(op.decay_linear(op.correlation(vol_n, blend, 17), 7), 13)
    return _clean(-1 * op.min_(part1, part2))


def alpha_083(f: FeatureEngine) -> pd.DataFrame:
    """rank(delay((high-low)/(sum(close,5)/5),2)) * rank(rank(vol)) / ((high-low)/(sum(close,5)/5+1e-6) / (vwap-close+1e-6))"""
    h, l, c, v, vwap = f.high, f.low, f.close, f.volume, f.vwap
    hl_ratio = (h - l) / (op.ts_mean(c, 5) + 1e-6)
    numer = op.rank(op.delay(hl_ratio, 2)) * op.rank(op.rank(v))
    denom = hl_ratio / (vwap - c + 1e-6)
    return _clean(numer / denom)


def alpha_084(f: FeatureEngine) -> pd.DataFrame:
    """SignedPower(ts_rank(vwap-ts_max(vwap,15),21), delta(close,5))"""
    vwap, c = f.vwap, f.close
    base = op.ts_rank(vwap - op.ts_max(vwap, 15), 21)
    return _clean(op.signed_power(base, op.delta(c, 5)))


def alpha_085(f: FeatureEngine) -> pd.DataFrame:
    """rank(corr(high*0.877+close*0.123, adv30, 10)) * rank(corr(ts_rank((h+l)/2,4),ts_rank(vol,10),7))"""
    h, l, c, v = f.high, f.low, f.close, f.volume
    adv30 = f.adv30
    blend = h * 0.876703 + c * (1 - 0.876703)
    return _clean(
        op.rank(op.correlation(blend, adv30, 10))
        * op.rank(op.correlation(op.ts_rank((h + l) / 2, 4), op.ts_rank(v, 10), 7))
    )


def alpha_086(f: FeatureEngine) -> pd.DataFrame:
    """-1*(ts_rank(corr(close, sum(adv20,15), 6), 20) < rank(5*rank(rank(corr(rank(vwap),rank(close),7)))))"""
    c, vwap, v = f.close, f.vwap, f.volume
    adv20 = f.adv20
    lhs = op.ts_rank(op.correlation(c, op.ts_sum(adv20, 15), 6), 20)
    rhs = op.rank(5 * op.rank(op.rank(op.correlation(op.rank(vwap), op.rank(c), 7))))
    result = pd.DataFrame(1.0, index=c.index, columns=c.columns)
    result = result.where(lhs >= rhs, -1.0)
    return _clean(-1 * result)


def alpha_087(f: FeatureEngine) -> pd.DataFrame:
    """-1*max(rank(decay_linear(delta(indneutral(close*0.370+open*0.630),2),3)),
              ts_rank(decay_linear(abs(corr(indneutral(adv81),close,13)),5),14))"""
    c, o = f.close, f.open
    adv81 = f.adv81
    blend_n = f.indneutralize(c * 0.369701 + o * (1 - 0.369701))
    adv81_n = f.indneutralize(adv81)
    part1 = op.rank(op.decay_linear(op.delta(blend_n, 2), 3))
    part2 = op.ts_rank(op.decay_linear(op.correlation(adv81_n, c, 13).abs(), 5), 14)
    return _clean(-1 * op.max_(part1, part2))


def alpha_088(f: FeatureEngine) -> pd.DataFrame:
    """min(rank(decay_linear((rank(open)+rank(low)-(rank(high)+rank(close))),8)),
           ts_rank(decay_linear(corr(ts_rank(close,8),ts_rank(adv60,21),8),7),3))"""
    o, l, h, c, v = f.open, f.low, f.high, f.close, f.volume
    adv60 = f.adv60
    inner = op.rank(o) + op.rank(l) - op.rank(h) - op.rank(c)
    part1 = op.rank(op.decay_linear(inner, 8))
    part2 = op.ts_rank(op.decay_linear(
        op.correlation(op.ts_rank(c, 8), op.ts_rank(adv60, 21), 8), 7), 3)
    return _clean(op.min_(part1, part2))


def alpha_089(f: FeatureEngine) -> pd.DataFrame:
    """ts_rank(decay_linear(corr(low*0.967+low*0.033, adv10, 7), 6), 4)
       - ts_rank(decay_linear(delta(indneutral(vwap),3),10),15)"""
    l, vwap = f.low, f.vwap
    adv10 = f.adv10
    blend = l * 0.967285 + l * (1 - 0.967285)
    vwap_n = f.indneutralize(vwap)
    part1 = op.ts_rank(op.decay_linear(op.correlation(blend, adv10, 7), 6), 4)
    part2 = op.ts_rank(op.decay_linear(op.delta(vwap_n, 3), 10), 15)
    return _clean(part1 - part2)


def alpha_090(f: FeatureEngine) -> pd.DataFrame:
    """-1*(rank(corr(rank(vwap), rank(adv5), 5)) < rank(corr(rank(open), rank(adv15), 10)) ? 1 : 0)"""
    vwap, o = f.vwap, f.open
    adv5, adv15 = f.adv5, f.adv15
    lhs = op.rank(op.correlation(op.rank(vwap), op.rank(adv5), 5))
    rhs = op.rank(op.correlation(op.rank(o), op.rank(adv15), 10))
    return _clean(-1 * (lhs < rhs).astype(float))


def alpha_091(f: FeatureEngine) -> pd.DataFrame:
    """-1*(ts_rank(decay_linear(decay_linear(corr(indneutral(close),vol,10),16),4),5)
          - rank(decay_linear(corr(vwap,adv30,4),3)))"""
    c, v, vwap = f.close, f.volume, f.vwap
    adv30 = f.adv30
    close_n = f.indneutralize(c)
    part1 = op.ts_rank(op.decay_linear(
        op.decay_linear(op.correlation(close_n, v, 10), 16), 4), 5)
    part2 = op.rank(op.decay_linear(op.correlation(vwap, adv30, 4), 3))
    return _clean(-1 * (part1 - part2))


def alpha_092(f: FeatureEngine) -> pd.DataFrame:
    """min(ts_rank(decay_linear(((h+l)/2+close)<(low+open),15),19),
           ts_rank(decay_linear(corr(rank(low),rank(adv30),8),6),7))"""
    h, l, c, o = f.high, f.low, f.close, f.open
    adv30 = f.adv30
    cond = (((h + l) / 2 + c) < (l + o)).astype(float)
    part1 = op.ts_rank(op.decay_linear(cond, 15), 19)
    part2 = op.ts_rank(op.decay_linear(
        op.correlation(op.rank(l), op.rank(adv30), 8), 6), 7)
    return _clean(op.min_(part1, part2))


def alpha_093(f: FeatureEngine) -> pd.DataFrame:
    """ts_rank(decay_linear(corr(indneutral(vwap),adv81,17),19),7)
       / rank(decay_linear(delta(close*0.524+vwap*0.476,3),16))"""
    vwap, c = f.vwap, f.close
    adv81 = f.adv81
    vwap_n = f.indneutralize(vwap)
    numer = op.ts_rank(op.decay_linear(op.correlation(vwap_n, adv81, 17), 19), 7)
    blend = c * 0.524434 + vwap * (1 - 0.524434)
    denom = op.rank(op.decay_linear(op.delta(blend, 3), 16))
    return _clean(numer / denom)


def alpha_094(f: FeatureEngine) -> pd.DataFrame:
    """rank(vwap-ts_min(vwap,11)) < ts_rank(corr(ts_rank(vwap,20),ts_rank(adv60,4),18),3) ? -1 : 1"""
    vwap = f.vwap
    adv60 = f.adv60
    lhs = op.rank(vwap - op.ts_min(vwap, 11))
    rhs = op.ts_rank(op.correlation(op.ts_rank(vwap, 20), op.ts_rank(adv60, 4), 18), 3)
    result = pd.DataFrame(1.0, index=vwap.index, columns=vwap.columns)
    return _clean(result.where(lhs >= rhs, -1.0))


def alpha_095(f: FeatureEngine) -> pd.DataFrame:
    """rank(open-ts_min(open,12)) < ts_rank(rank(corr(sum((h+l+c+o)/4,19),sum(adv40,19),13)),12) ? 1 : 0"""
    o, h, l, c = f.open, f.high, f.low, f.close
    adv40 = f.adv40
    lhs = op.rank(o - op.ts_min(o, 12))
    mid_price = (h + l + c + o) / 4
    rhs = op.ts_rank(op.rank(
        op.correlation(op.ts_sum(mid_price, 19), op.ts_sum(adv40, 19), 13)
    ), 12)
    return _clean((lhs < rhs).astype(float))


def alpha_096(f: FeatureEngine) -> pd.DataFrame:
    """-1*max(ts_rank(decay_linear(corr(rank(vwap),rank(vol),4),4),8),
              ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7),ts_rank(adv60,4),4),13),14),13))"""
    vwap, c, v = f.vwap, f.close, f.volume
    adv60 = f.adv60
    part1 = op.ts_rank(op.decay_linear(
        op.correlation(op.rank(vwap), op.rank(v), 4), 4), 8)
    inner = op.correlation(op.ts_rank(c, 7), op.ts_rank(adv60, 4), 4)
    part2 = op.ts_rank(op.decay_linear(op.ts_argmax(inner, 13), 14), 13)
    return _clean(-1 * op.max_(part1, part2))


def alpha_097(f: FeatureEngine) -> pd.DataFrame:
    """rank(decay_linear(delta(indneutral(low*0.721+vwap*0.279),3),8))
       + ts_rank(decay_linear(ts_rank(corr(ts_rank(low,8),ts_rank(adv60,17),5),19),17),19)"""
    l, vwap = f.low, f.vwap
    adv60 = f.adv60
    blend_n = f.indneutralize(l * 0.721001 + vwap * (1 - 0.721001))
    part1 = op.rank(op.decay_linear(op.delta(blend_n, 3), 8))
    part2 = op.ts_rank(op.decay_linear(
        op.ts_rank(op.correlation(op.ts_rank(l, 8), op.ts_rank(adv60, 17), 5), 19), 17), 19)
    return _clean(part1 + part2)


def alpha_098(f: FeatureEngine) -> pd.DataFrame:
    """rank(decay_linear(corr(vwap,sum(adv5,26),5),7))
       - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open),rank(adv15),21),9),7),8))"""
    vwap, o = f.vwap, f.open
    adv5, adv15 = f.adv5, f.adv15
    part1 = op.rank(op.decay_linear(op.correlation(vwap, op.ts_sum(adv5, 26), 5), 7))
    inner = op.ts_argmin(op.correlation(op.rank(o), op.rank(adv15), 21), 9)
    part2 = op.rank(op.decay_linear(op.ts_rank(inner, 7), 8))
    return _clean(part1 - part2)


def alpha_099(f: FeatureEngine) -> pd.DataFrame:
    """-1 * rank(cov(rank(close), rank(volume), 5))  [same structure as alpha13, different spirit]"""
    return _clean(-1 * op.rank(op.covariance(op.rank(f.close), op.rank(f.volume), 5)))


def alpha_100(f: FeatureEngine) -> pd.DataFrame:
    """scale(indneutral(scale(indneutral(rank(intraday_vol_signal)))) - scale(indneutral(corr-based))) * (vol/adv20)"""
    c, h, l, v = f.close, f.high, f.low, f.volume
    adv20 = f.adv20
    intraday = ((c - l) - (h - c)) / (h - l + 1e-6) * v
    part1 = op.scale(f.indneutralize(op.scale(f.indneutralize(op.rank(intraday)))))
    part2 = op.scale(f.indneutralize(
        op.correlation(c, op.rank(adv20), 5) - op.rank(op.ts_argmin(c, 30))
    ))
    return _clean(-(part1 - part2) * (v / adv20))


def alpha_101(f: FeatureEngine) -> pd.DataFrame:
    """(close - open) / (high - low + 0.001)"""
    return _clean((f.close - f.open) / (f.high - f.low + 0.001))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_ALPHA_NAMES = [f"alpha_{n:03d}" for n in range(1, 102)]
_ALPHA_FUNCS = {
    name: globals()[name]
    for name in _ALPHA_NAMES
    if name in globals()
}


def register_all_alphas(registry: AlphaRegistry) -> None:
    """Register all 101 alpha functions into *registry*."""
    for n in range(1, 102):
        name = f"alpha_{n:03d}"
        if name not in _ALPHA_FUNCS:
            continue
        registry.register(
            Alpha(
                name=name,
                function=_ALPHA_FUNCS[name],
                description=f"WorldQuant 101 Formulaic Alpha #{n:03d} (Kakushadze 2016)",
                horizon=1,
            )
        )


__all__ = ["register_all_alphas"] + _ALPHA_NAMES
