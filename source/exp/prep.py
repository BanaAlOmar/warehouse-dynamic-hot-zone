"""Stage 1: data prep — weekly demand, orders for replay, forecasts.
Pre-registered: weeks 1-12 train/calibrate, weeks 13-21 test. Internal moves excluded from demand.
"""
from pathlib import Path
import pandas as pd, numpy as np, json, os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'exp' / 'out'
OUT.mkdir(parents=True, exist_ok=True)

picks = pd.read_parquet(
    DATA / 'picks_clean.parquet',
    columns=['DT_START', 'SKU', 'ORDER_ID', 'ORDER_TYPE'],
    dtype_backend='pyarrow',
)
picks['DT'] = pd.to_datetime(picks['DT_START'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
picks = picks.dropna(subset=['DT'])

# demand = customer picks only (exclude internal moves)
dem = picks[picks['ORDER_TYPE'] != 'internal move'].copy()

# week index (Sunday-ending, same as audit)
week_ends = sorted(dem.set_index('DT').groupby(pd.Grouper(freq='W')).size().index)
week_map = {w: i + 1 for i, w in enumerate(week_ends)}  # 1-based
dem['week'] = dem['DT'].dt.to_period('W-SUN').dt.end_time.dt.normalize()
# align to Grouper convention
dem['week_idx'] = pd.to_datetime(dem['DT']).dt.to_period('W-SUN').apply(lambda p: p.end_time.date())
we_dates = {pd.Timestamp(w).date(): i for w, i in week_map.items()}
dem['w'] = dem['week_idx'].map(we_dates)
dem = dem.dropna(subset=['w'])
dem['w'] = dem['w'].astype(int)

n_weeks = dem['w'].max()
print("weeks:", n_weeks, "| lines:", len(dem))

# weekly SKU demand (pick lines - primary velocity measure, per plan)
wk = dem.groupby(['w', 'SKU']).size().rename('lines').reset_index()
wk.to_parquet(OUT / 'weekly_demand.parquet')

# orders for replay: order_id, week, start ts, sku list
orders = dem.groupby('ORDER_ID').agg(
    w=('w', 'min'), t0=('DT', 'min'), n_lines=('SKU', 'size')).reset_index()
lines = dem[['ORDER_ID', 'SKU', 'w', 'DT']].copy()
orders.to_parquet(OUT / 'orders.parquet')
lines.to_parquet(OUT / 'lines.parquet')

# SKU master with imputation flag
skus = pd.read_csv(DATA / 'SKUs_clean.csv', dtype=str)
for c in ['WGT1', 'HGT1', 'WID1', 'DPTH1', 'QTY1IN2']:
    skus[c] = pd.to_numeric(skus[c], errors='coerce')
skus['unit_vol'] = (skus['HGT1'].clip(lower=1) * skus['WID1'].clip(lower=1) * skus['DPTH1'].clip(lower=1))
all_skus = sorted(dem['SKU'].unique())
master = skus.set_index('SKU')['unit_vol'].to_dict()
med_vol = float(np.nanmedian(list(master.values())))
sku_df = pd.DataFrame({'SKU': all_skus})
sku_df['unit_vol'] = sku_df['SKU'].map(master)
sku_df['imputed'] = sku_df['unit_vol'].isna()
sku_df['unit_vol'] = sku_df['unit_vol'].fillna(med_vol)
sku_df.to_parquet(OUT / 'sku_master.parquet')

# ---- forecasts: seasonal-naive & exponential smoothing, chosen by MASE on weeks 5-8 (fitted on 1-4+) ----
piv = wk.pivot(index='w', columns='SKU', values='lines').fillna(0.0).sort_index()
piv.to_parquet(OUT / 'weekly_pivot.parquet')
X = piv.values  # weeks x skus

def es_forecast(X, alpha):
    F = np.zeros_like(X, dtype=float)
    F[0] = X[0]
    for t in range(1, X.shape[0]):
        F[t] = alpha * X[t - 1] + (1 - alpha) * F[t - 1]
    return F  # F[t] = forecast FOR week t using data through t-1

def mase(actual, fc, insample):
    denom = np.mean(np.abs(np.diff(insample, axis=0)), axis=0) + 1e-9
    return float(np.mean(np.mean(np.abs(actual - fc), axis=0) / denom))

val_w = slice(4, 8)  # weeks 5-8 (0-based rows 4..7)
naive_fc = np.vstack([X[0:1], X[:-1]])  # prev week
scores = {'naive': mase(X[val_w], naive_fc[val_w], X[:8])}
best_alpha, best_s = None, np.inf
for a in [0.2, 0.3, 0.5, 0.7]:
    F = es_forecast(X, a)
    s = mase(X[val_w], F[val_w], X[:8])
    scores[f'es_{a}'] = s
    if s < best_s: best_s, best_alpha = s, a
print("forecast MASE:", json.dumps(scores, indent=1))
chosen = f'es_{best_alpha}' if best_s < scores['naive'] else 'naive'
F = es_forecast(X, best_alpha) if chosen.startswith('es') else naive_fc
np.save(OUT / 'forecast.npy', F)
np.save(OUT / 'actual.npy', X)
json.dump({'chosen': chosen, 'scores': scores, 'skus': list(piv.columns),
           'n_weeks': int(piv.index.max())},
          open(OUT / 'forecast_meta.json', 'w'))
print("chosen forecaster:", chosen)
