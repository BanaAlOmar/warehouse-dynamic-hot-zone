"""Pass-1 item 2: largest-gap routing + FCFS batching arms on the test window.
Runs B1(ABC) and A1(heuristic) trajectories under each (routing, batch_cap) cell.
Writes checkpoint CSVs per-arm so a crash never loses completed arms.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, json, os, time
from exp.core import Layout
from exp.runner import load_all
from exp import policies as P
from exp.problem import ReslotProblem
from exp.solvers import solve_a1
from exp.routing_ext import eval_week_routed

OUT = str(ROOT / 'exp' / 'out')
CKPT = f'{OUT}/routing_arms.csv'
TEST = list(range(13, 22))

meta, skus, F, X, lines, orders, unit_vol, imputed = load_all()
lay = Layout()

# calibration-chosen params (frozen from Pass-0)
chosen = json.load(open(f'{OUT}/calib_chosen.json'))
B, lam, alpha = chosen['B'], chosen['lambda'], chosen['alpha']
print(f"frozen params: B={B} lam={lam} alpha={alpha}")

# precompute static B1 and per-week A1 assignments (A1 depends on forecast+prev)
abc_train = P.b1_abc(X[:12].sum(0), lay)

def a1_traj():
    """Return final shared-pool A1 assignments, forecast-driven and budgeted."""
    assign = abc_train.copy()
    out = {}
    for w in TEST:
        fc = F[w - 1]
        assign = solve_a1(ReslotProblem(assign, fc, lay, B, lam, alpha, K=1600))
        out[w] = assign.copy()
    return out

A1 = a1_traj()

# arms: (routing, batch_cap). single-order sshape is the Pass-0 baseline (sanity re-check).
arms = [('sshape', 1), ('largestgap', 1), ('sshape', 2), ('sshape', 4), ('largestgap', 4)]

rows = []
if os.path.exists(CKPT):
    rows = pd.read_csv(CKPT).to_dict('records')
    done = {(r['policy'], r['routing'], r['batch_cap']) for r in rows}
else:
    done = set()

for routing, cap in arms:
    for pol, get_assign in [('B1_ABC', lambda w: abc_train), ('A1_pool', lambda w: A1[w])]:
        if (pol, routing, cap) in done:
            print(f"skip {pol} {routing} cap{cap} (checkpointed)"); continue
        t = time.time()
        for w in TEST:
            r = eval_week_routed(lay, get_assign(w), lines, orders, w,
                                 routing=routing, batch_cap=cap, do_des=True)
            r['policy'] = pol
            rows.append(r)
        pd.DataFrame(rows).to_csv(CKPT, index=False)  # checkpoint after each (pol,arm)
        print(f"done {pol:8s} {routing:11s} cap{cap} in {time.time()-t:.1f}s")

df = pd.DataFrame(rows)
print("\n=== ROUTING/BATCHING ARMS (mean over test weeks) ===")
g = df.groupby(['routing', 'batch_cap', 'policy']).agg(
    travel_km=('travel_total_km', 'mean'),
    block_h=('block_wait_total_h', 'mean'),
    gini=('aisle_gini', 'mean'),
    tput=('tput_mean_min', 'mean'),
    n_tours=('n_tours', 'mean')).round(2)
print(g)

# A1 vs B1 delta per cell + does congestion negative result survive?
print("\n=== A1 vs B1 travel delta by routing/batch (does dynamic win survive?) ===")
piv = df.groupby(['routing', 'batch_cap', 'policy'])['travel_total_km'].mean().unstack('policy')
piv['delta_pct'] = 100 * (piv['A1_pool'] / piv['B1_ABC'] - 1)
print(piv.round(2))

print("\n=== A1 vs B1 blocking delta (the congestion negative result) ===")
pivb = df.groupby(['routing', 'batch_cap', 'policy'])['block_wait_total_h'].mean().unstack('policy')
pivb['delta_pct'] = 100 * (pivb['A1_pool'] / pivb['B1_ABC'] - 1)
print(pivb.round(2))
