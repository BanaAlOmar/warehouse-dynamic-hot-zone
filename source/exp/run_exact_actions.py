"""Swap-accounted exact local benchmark on the primary test window.

Runs B1, deployed A1, and an exact sparse assignment MIP. The exact model covers
all weekly ReslotProblem decision SKUs but not the full T candidate-slot pool: it
permutes their current slots over reciprocal top-partner arcs plus every A1
incumbent arc. A1 and exact receive the identical weekly starting state, forecast,
B=100 realized-relocation budget, and replay machinery. The exact rows are thus a
paired same-state solver diagnostic, not a separately evolving policy trajectory.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp import policies as P
from exp.core import Layout
from exp.problem import ReslotProblem
from exp.runner import eval_week, load_all
from exp.solvers import solve_a1, solve_exact_assignment

OUT = Path(__file__).resolve().parent / "out"
TEST = list(range(13, 22))
K = 1600

meta, skus, F, X, lines, orders, unit_vol, imputed = load_all()
lay = Layout()
cal = json.load(open(OUT / "calib_chosen.json"))
B, lam, alpha = int(cal["B"]), float(cal["lambda"]), float(cal["alpha"])
init = P.b1_abc(X[:12].sum(0), lay)


def run(name, step):
    assign = init.copy()
    rows = []
    diagnostics = []
    for w in TEST:
        t0 = time.time()
        new, diag = step(assign, w)
        elapsed = time.time() - t0
        moves = int((new != assign).sum())
        assert moves <= B and len(np.unique(new)) == len(new)
        assign = new
        metric = eval_week(lay, assign, lines, orders, w, do_des=True)
        metric.update(policy=name, moves=moves, opt_time_s=elapsed)
        rows.append(metric)
        diagnostics.append({"week": w, "policy": name, **diag})
        print(name, w, "travel", round(metric["travel_total_km"], 3),
              "moves", moves, "solve_s", round(elapsed, 2), flush=True)
    return rows, diagnostics


all_rows = []
all_diag = []

rows, diag = run("B1_ABC_static",
                 lambda a, w: (init.copy(), {"status": "fixed"}))
all_rows += rows
all_diag += diag

# Paired same-state diagnostic: both solvers see the deployed A1 trajectory's
# current assignment.  The exact result is therefore a weekly solver benchmark,
# not a separately evolving operational policy.
assign = init.copy()
a1_rows, exact_rows, paired_diag = [], [], []
for w in TEST:
    prob = ReslotProblem(assign, F[w - 1], lay, B, lam, alpha, K=K)
    t0 = time.time()
    a1 = solve_a1(prob)
    a1_s = time.time() - t0
    exact, diag = solve_exact_assignment(
        prob, warm_assign=a1, partners_per_sku=30, time_limit=300)
    for name, candidate, elapsed, extra, dest in [
        ("A1_pool", a1, a1_s,
         {"status": "heuristic",
          "objective_gain": prob.objective(assign) - prob.objective(a1)},
         a1_rows),
        ("Exact_assignment_same_state", exact, diag["solve_s"], diag,
         exact_rows),
    ]:
        moves = int((candidate != assign).sum())
        metric = eval_week(lay, candidate, lines, orders, w, do_des=True)
        metric.update(policy=name, moves=moves, opt_time_s=elapsed)
        dest.append(metric)
        paired_diag.append({"week": w, "policy": name, **extra})
        print(name, w, "travel", round(metric["travel_total_km"], 3),
              "moves", moves, "solve_s", round(elapsed, 2), flush=True)
    assign = a1

all_rows += a1_rows + exact_rows
all_diag += paired_diag

df = pd.DataFrame(all_rows)
dg = pd.DataFrame(all_diag)
df.to_csv(OUT / "exact_actions_weekly.csv", index=False)
dg.to_csv(OUT / "exact_actions_diagnostics.csv", index=False)

p = df.pivot(index="week", columns="policy", values="travel_total_km")
b1 = p["B1_ABC_static"]
summary = []
for policy in p:
    summary.append({
        "policy": policy,
        "mean_km": p[policy].mean(),
        "pct_vs_B1": 100 * (p[policy].sum() / b1.sum() - 1),
        "weeks_better_than_B1": int((p[policy] < b1).sum()),
    })
pd.DataFrame(summary).to_csv(OUT / "exact_actions_summary.csv", index=False)
cmp = p[["A1_pool", "Exact_assignment_same_state"]].copy()
cmp["exact_minus_a1_km"] = (
    cmp["Exact_assignment_same_state"] - cmp["A1_pool"]
)
cmp["exact_vs_a1_pct"] = 100 * (
    cmp["Exact_assignment_same_state"] / cmp["A1_pool"] - 1
)
cmp["exact_route_better"] = cmp["exact_minus_a1_km"] < 0
cmp.reset_index().to_csv(OUT / "exact_replay_comparison.csv", index=False)
print(pd.DataFrame(summary).round(4).to_string(index=False), flush=True)
