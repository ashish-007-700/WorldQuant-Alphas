"""Smoke test for all Phase 3 & 4 modules."""
import sys
sys.path.insert(0, 'src')
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
import pandas as pd, numpy as np

print("=== Testing Phase 3 & 4 modules ===")
print()

# 1. factors
from wqalpha.factors import load_factors, describe_factors
f = load_factors('data/india_factors.csv')
print(f"[factors.py] Loaded {f.shape} factor file OK")
print(describe_factors(f).to_string())
print()

# 2. universe
from wqalpha.universe import build_liquid_universe, coverage_report
panel = pd.read_csv('data/india_equities.csv', parse_dates=['date'])
liq = build_liquid_universe(panel, min_adv_cr=50)
pct = liq.mean().mean() * 100
print(f"[universe.py] Liquid universe: {liq.shape}, avg liquidity {pct:.1f}%")
cr = coverage_report(panel)
print(cr.head(5).to_string())
print()

# 3. dataquality
from wqalpha.dataquality import check_panel
dq = check_panel(panel)
summary = dq.summary()
# Replace any non-ascii for safe printing
summary = summary.encode('ascii', errors='replace').decode('ascii')
print("[dataquality.py]")
print(summary)
bad = dq.bad_symbols()
if bad:
    print(f"  Bad symbols (>5% missing): {bad}")
print()

# 4. risk
from wqalpha.risk import alpha_attribution, sector_ic, fama_french_4
print("[risk.py] Imports OK")
print()

# 5. experiment
from wqalpha.experiment import Experiment, ExperimentLog
with Experiment('smoke_test', log_dir='experiments') as exp:
    exp.log_params({'alpha': 'alpha_001', 'tc_bps': 10})
    exp.log_metrics({'sharpe': 1.23, 'icir': 0.45})
    exp.log_tags({'phase': 'smoke_test'})
log = ExperimentLog('experiments')
df = log.load()
print(f"[experiment.py] Logged run OK, {len(df)} entries in log")
print()

# 6. walkforward (import only)
from wqalpha.walkforward import WalkForwardEngine, WalkForwardResult
print("[walkforward.py] Imports OK")
print()

print("=== All Phase 3 & 4 modules PASSED ===")
