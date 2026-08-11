"""Retail II confirmatory-dataset GATE computation (training weeks ONLY).
Applies the frozen Section-14.3 filtering in FIXED ORDER, defines the chronological
60/40 split before looking at outcomes, and computes Gate B + drift on TRAIN weeks only.
STOPS before any test-window replay. No recalibration, no filter hunting.
"""
from pathlib import Path
import numpy as np, pandas as pd, json, sys
ROOT = Path(__file__).resolve().parents[1]
OUT = str(ROOT / 'exp' / 'out')
RAW = str(ROOT / 'data' / 'retail2_raw.parquet')

df = pd.read_parquet(RAW)
# normalize column names (two-sheet concat)
df = df.rename(columns={'Invoice': 'InvoiceNo', 'Customer ID': 'CustomerID'})
n0 = len(df)
log = {'raw_rows': n0}

# ---- FILTER CLASS 1: drop cancellation invoices (InvoiceNo starting 'C') ----
inv = df['InvoiceNo'].astype(str)
mask_cancel = inv.str.upper().str.startswith('C')
df = df[~mask_cancel]
log['c1_after_cancel'] = len(df); log['c1_dropped'] = int(mask_cancel.sum())

# ---- FILTER CLASS 2: drop non-positive quantities / returns ----
mask_qty = df['Quantity'] > 0
n_before = len(df); df = df[mask_qty]
log['c2_after_posqty'] = len(df); log['c2_dropped'] = n_before - len(df)

# ---- FILTER CLASS 3: non-product stock codes + missing StockCode/Description ----
NONPRODUCT = {'POST', 'M', 'DOT', 'BANK CHARGES', 'BANKCHARGES', 'AMAZONFEE',
              'CRUK', 'S', 'D', 'B', 'PADS', 'C2', 'GIFT', 'TEST',
              'ADJUST', 'ADJUST2', 'ADJUSTMENT', 'SAMPLES', 'MANUAL',
              'DCGS0069', 'DCGS0070', 'gift_0001'}
sc = df['StockCode'].astype(str).str.strip().str.upper()
desc = df['Description']
# exact non-product codes OR codes that are purely non-numeric adjustment/fee markers
is_nonproduct = sc.isin({x.upper() for x in NONPRODUCT})
# gift-card / test / adjustment prefixes
is_giftlike = sc.str.startswith('GIFT') | sc.str.startswith('DCGS') | sc.str.contains('TEST', na=False)
missing = df['StockCode'].isna() | df['Description'].isna() | (desc.astype(str).str.strip() == '')
mask3 = is_nonproduct | is_giftlike | missing
n_before = len(df); df = df[~mask3]
log['c3_after_nonproduct'] = len(df); log['c3_dropped'] = n_before - len(df)
log['c3_nonproduct_codes_hit'] = sorted(sc[is_nonproduct].unique().tolist())

# ---- define ISO-week stream and chronological 60/40 split (BEFORE cutoff, per spec order:
#      cutoff is computed on TRAINING weeks, so the split must be known first) ----
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
iso = df['InvoiceDate'].dt.isocalendar()
df['yw'] = iso['year'].astype(int) * 100 + iso['week'].astype(int)
weeks = sorted(df['yw'].unique())
n_weeks = len(weeks)
n_train = int(round(0.60 * n_weeks))
train_weeks = set(weeks[:n_train]); test_weeks = set(weeks[n_train:])
log['n_iso_weeks'] = n_weeks
log['n_train_weeks'] = n_train
log['n_test_weeks'] = n_weeks - n_train
log['train_week_first'] = int(weeks[0]); log['train_week_last'] = int(weeks[n_train - 1])
log['test_week_first'] = int(weeks[n_train]); log['test_week_last'] = int(weeks[-1])
log['train_pct'] = round(100 * n_train / n_weeks, 1)

df['is_train'] = df['yw'].isin(train_weeks)

# ---- FILTER CLASS 4: 99.9th-pct quantity cutoff on TRAINING-WEEK lines ONLY,
#      carried forward as an absolute cutoff to test weeks ----
train_q = df.loc[df['is_train'], 'Quantity']
cutoff = float(np.percentile(train_q, 99.9))
log['c4_qty_cutoff_999_train'] = cutoff
n_before = len(df); df = df[df['Quantity'] <= cutoff]
log['c4_after_cutoff'] = len(df); log['c4_dropped'] = n_before - len(df)

df.to_parquet(f'{OUT}/retail2_filtered.parquet')
log['final_rows'] = len(df)

# =====================================================================
# GATES on TRAINING WEEKS ONLY
# =====================================================================
train = df[df['is_train']]
sku_lines_train = train.groupby('StockCode').size().sort_values(ascending=False)
n_sku_train = int(sku_lines_train.shape[0])

# --- Gate B: discriminative power ---
#   requirement: >= 1000 SKUs AND top-decile line share < 80%
top_decile_n = max(1, int(round(0.10 * n_sku_train)))
top_decile_share = float(sku_lines_train.iloc[:top_decile_n].sum() / sku_lines_train.sum())
gateB_nsku_pass = n_sku_train >= 1000
gateB_conc_pass = top_decile_share < 0.80
gateB_pass = gateB_nsku_pass and gateB_conc_pass

# --- Drift gate: train-only week-to-week rank stability ---
#   weekly SKU line-count vectors over the training weeks; Kendall tau + top-decile Jaccard
from scipy.stats import kendalltau
tw = sorted(train_weeks)
piv = train.pivot_table(index='StockCode', columns='yw', values='Quantity',
                        aggfunc='size', fill_value=0)
taus, jacs = [], []
for i in range(1, len(tw)):
    a = piv[tw[i - 1]]; b = piv[tw[i]]
    both = (a > 0) | (b > 0)
    if both.sum() > 2:
        t, _ = kendalltau(a[both], b[both]); taus.append(t)
    ka = set(a.sort_values(ascending=False).index[:max(1, int(0.10 * (a > 0).sum()))])
    kb = set(b.sort_values(ascending=False).index[:max(1, int(0.10 * (b > 0).sum()))])
    if ka | kb:
        jacs.append(len(ka & kb) / len(ka | kb))
drift = {'mean_kendall_tau': float(np.nanmean(taus)),
         'mean_topdecile_jaccard': float(np.nanmean(jacs)),
         'n_week_pairs': len(taus)}

gates = {
    'filter_log': log,
    'gateB': {
        'n_sku_train': n_sku_train,
        'requirement_nsku': '>=1000',
        'nsku_pass': bool(gateB_nsku_pass),
        'top_decile_line_share': round(top_decile_share, 4),
        'requirement_concentration': '<0.80',
        'concentration_pass': bool(gateB_conc_pass),
        'GATE_B_PASS': bool(gateB_pass),
    },
    'drift_gate_train_only': drift,
}
json.dump(gates, open(f'{OUT}/retail2_gates.json', 'w'), indent=2)
print(json.dumps(gates, indent=2))
