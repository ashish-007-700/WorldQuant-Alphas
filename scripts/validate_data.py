import pandas as pd
import numpy as np

df = pd.read_csv('data/india_equities.csv', parse_dates=['date'])

print("=== REAL NSE DATA SUMMARY ===")
print(f"Total rows        : {len(df):,}")
print(f"Unique symbols    : {df['symbol'].nunique()}")
print(f"Date range        : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Trading days      : {df['date'].nunique()}")
print(f"Approx years      : {df['date'].nunique() / 252:.1f}")
print()

coverage = df.groupby('symbol')['date'].count()
print(f"Min rows per stock : {coverage.min()} ({coverage.idxmin()})")
print(f"Max rows per stock : {coverage.max()} ({coverage.idxmax()})")
print(f"Full coverage      : {(coverage == coverage.max()).all()}")
print()

rets = df['returns'].dropna()
print("=== RETURN STATISTICS ===")
print(f"Mean daily return : {rets.mean():.4%}")
print(f"Std daily return  : {rets.std():.4%}")
print(f"Ann. Sharpe (est) : {rets.mean() / rets.std() * np.sqrt(252):.2f}")
print(f"Skewness          : {rets.skew():.3f}")
print()

print("=== PRICE SANITY ===")
print(f"Any negative close : {(df['close'] < 0).any()}")
print(f"Any NaN close      : {df['close'].isna().any()}")
print(f"Any NaN volume     : {df['volume'].isna().any()}")
print()

print("=== SECTOR BREAKDOWN ===")
print(df.groupby('sector')['symbol'].nunique().sort_values(ascending=False).to_string())
print()

print("=== SAMPLE ROWS (RELIANCE last 5) ===")
print(df[df['symbol']=='RELIANCE'].tail(5)[['date','open','high','low','close','volume','returns']].to_string(index=False))
