"""Layout sensitivity (Section 8): OFAT screening on test weeks, B1 vs A1.
Factors (base *): blocks {1,2*,4}; aisle geometry {30x29, 20x43*, 14x62} (slots ~const);
depot {corner*, center}; slot pitch {0.8, 1.0*, 1.25}; pickers {5,15,25*,40,50} (DES only).
Then: two largest travel main effects -> 3x3 factorial (A1 vs B1, DES on).
RL is NOT retrained per cell (zero-shot, declared deviation).
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp.runner import load_all, run_policy_trajectory, eval_week
from exp.core import Layout
from exp import policies as P
from exp.problem import ReslotProblem
from exp.solvers import solve_a1

meta, skus, F, X, lines, orders, unit_vol, imputed = load_all()
TEST = list(range(13, 22))
OUT = ROOT / 'exp' / 'out'
cal = json.load(open(OUT / 'calib_chosen.json'))
LAM, ALPHA, B = cal['lambda'], cal['alpha'], cal['B']
train_vel = X[:12].sum(0)

def final_a1(prev, w, lay):
    return solve_a1(ReslotProblem(prev, F[w - 1], lay, B, LAM, ALPHA, K=1600))

def cell(tag, factor, level, do_des=False, n_pickers=25, **laykw):
    lay = Layout(**laykw)
    assert lay.n_slots >= len(skus), f"{tag}: {lay.n_slots} slots < {len(skus)}"
    init = P.b1_abc(train_vel, lay)
    b1, _ = run_policy_trajectory('B1', lay, F, X, lines, orders, TEST, init,
                                  lambda p, w: p, n_pickers=n_pickers, do_des=do_des)
    a1, _ = run_policy_trajectory('A1', lay, F, X, lines, orders, TEST, init,
                                  lambda p, w: final_a1(p, w, lay),
                                  n_pickers=n_pickers, do_des=do_des)
    for df, pol in [(b1, 'B1'), (a1, 'A1')]:
        df['cell'], df['factor'], df['level'], df['policy'] = tag, factor, str(level), pol
    print(f"{tag}: B1 {b1['travel_total_km'].mean():.0f}km A1 {a1['travel_total_km'].mean():.0f}km",
          flush=True)
    return pd.concat([b1, a1])

rows = []
rows.append(cell('base', 'base', 'base', do_des=True))
rows.append(cell('blocks1', 'blocks', 1, n_blocks=1, slots_per_side_block=86))
rows.append(cell('blocks4', 'blocks', 4, n_blocks=4, slots_per_side_block=22))
rows.append(cell('aisleL-', 'aisle_len', '-50%', n_aisles=30, slots_per_side_block=29))
rows.append(cell('aisleL+', 'aisle_len', '+50%', n_aisles=14, slots_per_side_block=62))
rows.append(cell('depotC', 'depot', 'center', depot='center'))
rows.append(cell('pitch-', 'pitch', 0.8, slot_pitch=0.8))
rows.append(cell('pitch+', 'pitch', 1.25, slot_pitch=1.25))
sens = pd.concat(rows)
sens.to_csv(OUT / 'sens_ofat.csv', index=False)

# picker-count sensitivity: DES on base-layout stored trajectories (travel unchanged)
lay = Layout()
init = P.b1_abc(train_vel, lay)
# Slotting trajectories do not depend on staffing, so compute the final A1 path
# once and replay it under each picker count.
a1_assignments = {}
assign = init.copy()
for w in TEST:
    assign = final_a1(assign, w, lay)
    a1_assignments[w] = assign.copy()
prow = []
for np_ in [5, 15, 25, 40, 50]:
    for pol, assignment_of_week in [
        ('B1', lambda w: init),
        ('A1', lambda w: a1_assignments[w]),
    ]:
        metrics = []
        previous = init
        for w in TEST:
            current = assignment_of_week(w)
            m = eval_week(lay, current, lines, orders, w,
                          n_pickers=np_, do_des=True)
            m['moves'] = int((current != previous).sum())
            metrics.append(m)
            previous = current
        df = pd.DataFrame(metrics)
        df['n_pickers'], df['policy'] = np_, pol
        prow.append(df)
    print(f"pickers={np_} done", flush=True)
pd.concat(prow).to_csv(OUT / 'sens_pickers.csv', index=False)

# ---- factor selection: two largest |main effect| on A1 mean travel ----
a1s = sens[sens['policy'] == 'A1'].groupby(['factor', 'level'])['travel_total_km'].mean()
base_travel = a1s['base']['base']
effects = {}
for fac in ['blocks', 'aisle_len', 'depot', 'pitch']:
    vals = list(a1s[fac].values) + [base_travel]
    effects[fac] = max(vals) - min(vals)
print("main effects (km):", effects)
top2 = sorted(effects, key=effects.get, reverse=True)[:2]
print("factorial factors:", top2)
json.dump({'effects_km': effects, 'top2': top2},
          open(OUT / 'sens_effects.json', 'w'), indent=1)

# ---- 3x3 (or 2x3) factorial on top-2 factors, DES on ----
levels = {'blocks': [(dict(n_blocks=1, slots_per_side_block=86), '1'),
                     (dict(), '2'), (dict(n_blocks=4, slots_per_side_block=22), '4')],
          'aisle_len': [(dict(n_aisles=30, slots_per_side_block=29), '-50%'),
                        (dict(), 'base'), (dict(n_aisles=14, slots_per_side_block=62), '+50%')],
          'depot': [(dict(), 'corner'), (dict(depot='center'), 'center')],
          'pitch': [(dict(slot_pitch=0.8), '0.8'), (dict(), '1.0'), (dict(slot_pitch=1.25), '1.25')]}
frow = []
for kw1, l1 in levels[top2[0]]:
    for kw2, l2 in levels[top2[1]]:
        kw = {**kw1, **kw2}
        frow.append(cell(f'F_{l1}_{l2}', f'{top2[0]}x{top2[1]}', f'{l1}|{l2}',
                         do_des=True, **kw))
pd.concat(frow).to_csv(OUT / 'sens_factorial.csv', index=False)
print("sensitivity done.")
