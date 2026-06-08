"""Build and save the Indian Fama-French factor file."""
import sys
sys.path.insert(0, 'src')
import pandas as pd
from wqalpha.factors import build_factors, describe_factors

print("Loading equity panel...")
panel = pd.read_csv('data/india_equities.csv', parse_dates=['date'])

print("Building factors (this takes ~30s for SMB/HML/MOM daily loops)...")
factors = build_factors(panel, raw_index_path='data/india_factors_raw.csv')

factors.to_csv('data/india_factors.csv')
print(f"\nSaved to data/india_factors.csv")
print(f"Shape: {factors.shape}")
print(f"Date range: {factors.index.min().date()} to {factors.index.max().date()}")
print(f"NaN counts:\n{factors.isna().sum()}")
print()
print("=== Factor Statistics (Annualised) ===")
print(describe_factors(factors).to_string())
