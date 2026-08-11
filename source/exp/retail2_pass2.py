"""Retail II CONFIRMATORY run (Pass 2) -- applies the FROZEN dataset-3 RULES to Retail II.
Stage A (train-only): build weekly demand, MASE-selected forecast, per-move-gain B*, K saturation,
   layout sized to 4,431 slots (4,253 train actives / 0.96). lambda=0 carried. NO test data touched.
Stage B (confirmatory, ONCE): replay test weeks under A1 vs B1 vs reABC, Wilcoxon+Holm via stats_family.
Run:  python exp/retail2_pass2.py calib     # Stage A, prints chosen B*/forecaster/K
      python exp/retail2_pass2.py confirm    # Stage B, the sealed run
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'exp'))
from core import Layout
import policies as P
from problem import ReslotProblem
from solvers import solve_a1
from runner import eval_week
OUT = str(ROOT / 'exp' / 'out')
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'calib'

# ---------- shared: load filtered stream, build weekly pivot & split ----------
df = pd.read_parquet(f'{OUT}/retail2_filtered.parquet')
g = json.load(open(f'{OUT}/retail2_gates.json'))
tw_last = g['filter_log']['train_week_last']
# weekly SKU demand = pick LINES (count of invoice-lines per SKU per week), matching prep.py
wk = df.groupby(['yw', 'StockCode']).size().rename('lines').reset_index()
piv = wk.pivot(index='yw', columns='StockCode', values='lines').fillna(0.0).sort_index()
weeks = list(piv.index)
train_mask = np.array([w <= tw_last for w in weeks])
n_train = int(train_mask.sum())
X_full = piv.values                # [week, sku] actual weekly lines, FULL universe
SKUS_full = list(piv.columns)
# ---- OPTION B (human ruling 2026-07-25): restrict universe to TRAIN-ACTIVE SKUs.
# Unseen test-only SKUs (628, 20.9% of test demand) are dropped from the replay and
# reported as a named open-universe limitation. Leakage-free, spec-unchanged.
train_active = X_full[:n_train].sum(0) > 0
X = X_full[:, train_active]
SKUS = [s for s, keep in zip(SKUS_full, train_active) if keep]
n_unseen = int((~train_active & (X_full[n_train:].sum(0) > 0)).sum())
unseen_demand_pct = 100.0 * X_full[n_train:][:, ~train_active].sum() / X_full[n_train:].sum()
TESTW = list(range(n_train, len(weeks)))   # 0-based test week rows

# ---------- forecaster: ES(alpha) vs naive by MASE, TRAIN weeks only ----------
def es_forecast(X, alpha):
    F = np.zeros_like(X, dtype=float); F[0] = X[0]
    for t in range(1, X.shape[0]):
        F[t] = alpha * X[t - 1] + (1 - alpha) * F[t - 1]
    return F

def mase(actual, fc, insample):
    denom = np.mean(np.abs(np.diff(insample, axis=0)), axis=0) + 1e-9
    return float(np.mean(np.mean(np.abs(actual - fc), axis=0) / denom))

# validation window = last 4 train weeks fitted on earlier train weeks (same relative design)
val = slice(n_train - 4, n_train)
naive_fc = np.vstack([X[0:1], X[:-1]])
scores = {'naive': mase(X[val], naive_fc[val], X[:n_train])}
best_a, best_s = None, np.inf
for a in [0.2, 0.3, 0.5, 0.7]:
    s = mase(X[val], es_forecast(X, a)[val], X[:n_train]); scores[f'es_{a}'] = s
    if s < best_s: best_s, best_a = s, a
chosen = f'es_{best_a}' if best_s < scores['naive'] else 'naive'
F = es_forecast(X, best_a) if chosen.startswith('es') else naive_fc

# ---------- layout sized on TRAIN actives (frozen rule) ----------
train_actives = int((X[:n_train].sum(0) > 0).sum())
target_slots = int(np.ceil(train_actives / 0.96))
# scale slots_per_side_block to reach >= target on the SAME base geometry (20 aisles x 2 blocks x 2 sides)
sps = int(np.ceil(target_slots / (20 * 2 * 2)))
lay = Layout(slots_per_side_block=sps)
assert lay.n_slots >= len(SKUS), f"layout {lay.n_slots} < {len(SKUS)} train-active skus"
print(f"[Option B] universe={len(SKUS)} train-active SKUs; dropped {n_unseen} unseen "
      f"({unseen_demand_pct:.1f}% of test demand); layout={lay.n_slots} slots", flush=True)

# ---------- B* by per-move-gain rule on TRAIN window ----------
def a1_traj_train(B, weeks_idx):
    init = P.b1_abc(X[:n_train].sum(0), lay)
    a = init.copy(); trav = []; mv = []
    for w in weeks_idx:
        nw = a.copy() if B == 0 else solve_a1(ReslotProblem(a, F[w], lay, B, 0.0, 0.1, K=1600))
        mv.append(int((nw != a).sum())); a = nw
        trav.append(eval_week(lay, a, LINES, ORDERS, w, do_des=False)['travel_total_km'])
    return np.array(trav), np.array(mv)

# ---------- order/line replay tables (needed by eval_week) ----------
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
sku_idx = {s: i for i, s in enumerate(SKUS)}   # train-active universe only
wk_of = {w: i for i, w in enumerate(weeks)}
dfx = df.copy(); dfx['w'] = dfx['yw'].map(wk_of); dfx['si'] = dfx['StockCode'].map(sku_idx)
dfx = dfx.dropna(subset=['si'])                 # drop unseen-SKU lines (Option B)
dfx['si'] = dfx['si'].astype(int)
ORDERS = dfx.groupby('InvoiceNo').agg(w=('w', 'min'), t0=('InvoiceDate', 'min'),
                                      n_lines=('si', 'size')).reset_index().rename(
                                          columns={'InvoiceNo': 'ORDER_ID'})
LINES = dfx[['InvoiceNo', 'si', 'w', 'InvoiceDate']].rename(columns={'InvoiceNo': 'ORDER_ID'})

if STAGE == 'calib':
    # B curve on a bounded slice of train weeks (last 6 train weeks) -- rule, not number
    cw = list(range(max(1, n_train - 6), n_train))
    bc = []
    for B in [0, 25, 50, 100, 200]:
        tr, mv = a1_traj_train(B, cw)
        bc.append(dict(B=B, travel=float(tr.mean()), moves=float(mv.mean())))
        print(bc[-1], flush=True)
    bcv = pd.DataFrame(bc)
    marg = -np.diff(bcv['travel'].values)
    thr = 0.10 * marg.max() if marg.max() > 0 else 0
    kidx = 0
    for i, m in enumerate(marg):
        if m >= thr: kidx = i + 1
    B_star = int(bcv['B'].iloc[kidx])
    calib = {'chosen_forecaster': chosen, 'mase_scores': scores,
             'n_train_weeks': n_train, 'n_test_weeks': len(TESTW),
             'train_actives': train_actives, 'target_slots': target_slots,
             'layout_slots': int(lay.n_slots), 'slots_per_side_block': sps,
             'B_star': B_star, 'B_rule': 'last step marginal gain >= 10% of max step gain',
             'lambda': 0.0, 'alpha_congestion': 0.1, 'K': 1600,
             'bcurve_train': bc}
    json.dump(calib, open(f'{OUT}/retail2_calib.json', 'w'), indent=2)
    print("\nRETAIL2 CALIB:", json.dumps(calib, indent=2))

elif STAGE == 'confirm':
    calib = json.load(open(f'{OUT}/retail2_calib.json'))
    B = calib['B_star']
    init = P.b1_abc(X[:n_train].sum(0), lay)
    def run(step):
        a = init.copy(); rows = []
        for w in TESTW:
            nw = step(a, w); mv = int((nw != a).sum()); a = nw
            m = eval_week(lay, a, LINES, ORDERS, w, do_des=True); m['moves'] = mv; m['w'] = w
            rows.append(m)
        return pd.DataFrame(rows)
    b1 = run(lambda a, w: init)
    a1 = run(lambda a, w: solve_a1(ReslotProblem(a, F[w], lay, B, 0.0, 0.1, K=1600)))
    # reABC: recompute ABC on trailing 12-week velocity then budget_repair to B moves
    def reabc_step(a, w):
        vel = X[max(0, w - 12):w].sum(0)
        fresh = P.b1_abc(vel, lay)
        pr = ReslotProblem(a, F[w], lay, B, 0.0, 0.1, K=1600)
        return pr.budget_repair(fresh)
    re = run(reabc_step)
    for d, nm in [(b1, 'B1_ABC_static'), (a1, 'A1_pool'), (re, 'reABC_budgeted')]:
        d['policy'] = nm
    allrows = pd.concat([b1, a1, re], ignore_index=True)
    allrows.to_csv(f'{OUT}/retail2_confirm.csv', index=False)
    from stats_family import compute_family
    pv = allrows.pivot_table(index='w', columns='policy', values='travel_total_km')
    st = compute_family(pv,
        primary=[('A1 vs B1', 'A1_pool', 'B1_ABC_static'),
                 ('A1 vs reABC', 'A1_pool', 'reABC_budgeted')],
        descriptive=[('reABC vs B1', 'reABC_budgeted', 'B1_ABC_static')],
        ref_col='B1_ABC_static')
    st.to_csv(f'{OUT}/retail2_stats.csv', index=False)
    pd.set_option('display.width', 220)
    print("B1 mean %.1f  A1 mean %.1f  reABC mean %.1f (km/wk over %d test weeks)" % (
        pv['B1_ABC_static'].mean(), pv['A1_pool'].mean(), pv['reABC_budgeted'].mean(), len(TESTW)))
    print(st[['comparison', 'family', 'pct', 'weeks_favorable', 'n_weeks',
              'cliffs_delta', 'p_wilcoxon', 'p_holm']].round(5).to_string(index=False))
