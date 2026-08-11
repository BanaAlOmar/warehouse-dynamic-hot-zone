"""Open-assortment confirmation on Online Retail II.

Unlike the preregistered Option-B run, this replay keeps test-only SKUs.  A
policy-independent rolling membership schedule admits every SKU required in the
current week and, once the 4,480-slot layout is full, evicts a dormant SKU with
the smallest trailing-12-week demand.  Mandatory new-SKU admissions are reported
separately and do not consume the B=100 discretionary re-slotting budget.
"""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Layout
import policies as P
from problem import ReslotProblem
from runner import eval_week
from solvers import solve_a1

OUT = Path(__file__).resolve().parent / "out"
df = pd.read_parquet(OUT / "retail2_filtered.parquet")
gates = json.load(open(OUT / "retail2_gates.json"))
cal = json.load(open(OUT / "retail2_calib.json"))

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
wk = df.groupby(["yw", "StockCode"]).size().rename("lines").reset_index()
piv = wk.pivot(index="yw", columns="StockCode",
               values="lines").fillna(0.0).sort_index()
weeks = list(piv.index)
skus = list(piv.columns)
X = piv.values.astype(float)
n_train = int(cal["n_train_weeks"])
testw = list(range(n_train, len(weeks)))
n_sku = len(skus)


def es_forecast(x, alpha):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t - 1] + (1 - alpha) * out[t - 1]
    return out


chosen = cal["chosen_forecaster"]
if chosen.startswith("es_"):
    F = es_forecast(X, float(chosen.split("_")[1]))
else:
    F = np.vstack([X[0:1], X[:-1]])

lay = Layout(slots_per_side_block=int(cal["slots_per_side_block"]))
B = int(cal["B_star"])
K = int(cal["K"])
assert lay.n_slots == int(cal["layout_slots"])

sku_ix = {s: i for i, s in enumerate(skus)}
week_ix = {w: i for i, w in enumerate(weeks)}
dfx = df.copy()
dfx["w"] = dfx["yw"].map(week_ix)
dfx["si"] = dfx["StockCode"].map(sku_ix).astype(int)
orders = dfx.groupby("InvoiceNo").agg(
    w=("w", "min"), t0=("InvoiceDate", "min"),
    n_lines=("si", "size")).reset_index().rename(
        columns={"InvoiceNo": "ORDER_ID"})
lines = dfx[["InvoiceNo", "si", "w", "InvoiceDate"]].rename(
    columns={"InvoiceNo": "ORDER_ID"})

# Initial membership and placement use training data only.
train_velocity = X[:n_train].sum(0)
train_active = train_velocity > 0
initial_members = set(np.where(train_active)[0].tolist())
initial_local = np.asarray(sorted(initial_members), dtype=int)
initial_assign = np.full(n_sku, -1, dtype=int)
initial_assign[initial_local] = P.b1_abc(train_velocity[initial_local], lay)


def membership_schedule():
    """Return policy-independent weekly members, admissions, and evictions."""
    members = set(initial_members)
    schedule = {}
    for w in testw:
        required = set(np.where(X[w] > 0)[0].tolist())
        entrants = sorted(required - members)
        evicted = []
        need = max(0, len(members) + len(entrants) - lay.n_slots)
        if need:
            dormant = list(members - required)
            hist = X[max(0, w - 12):w].sum(0)
            dormant.sort(key=lambda s: (hist[s], F[w, s], str(skus[s])))
            if len(dormant) < need:
                raise RuntimeError("weekly required assortment exceeds layout capacity")
            evicted = dormant[:need]
            members.difference_update(evicted)
        members.update(entrants)
        assert required.issubset(members) and len(members) <= lay.n_slots
        schedule[w] = {
            "members": set(members), "entrants": entrants, "evicted": evicted,
            "required": required,
        }
    return schedule


schedule = membership_schedule()


def _signature(values):
    payload = "|".join(str(v) for v in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_admissions(assign, w):
    """Apply the common membership transition while preserving each policy's slots."""
    info = schedule[w]
    out = assign.copy()
    reclaimed = []
    for sku in info["evicted"]:
        if out[sku] >= 0:
            reclaimed.append(int(out[sku]))
            out[sku] = -1
    used = set(out[out >= 0].tolist())
    free = reclaimed + [
        int(s) for s in np.argsort(lay.slot_depot)
        if int(s) not in used and int(s) not in reclaimed
    ]
    if len(free) < len(info["entrants"]):
        raise RuntimeError("not enough slots for admissions")
    admitted = []
    for sku, slot in zip(info["entrants"], free):
        out[sku] = slot
        admitted.append((int(sku), int(slot)))
    actual_members = set(np.where(out >= 0)[0].tolist())
    assert actual_members == info["members"]
    assert len(np.unique(out[out >= 0])) == len(actual_members)
    return out, admitted


def local_problem(assign, forecast, budget):
    members = np.where(assign >= 0)[0]
    prob = ReslotProblem(assign[members], forecast[members], lay, budget,
                         0.0, 0.1, K=min(K, len(members)))
    return members, prob


def run(policy):
    assign = initial_assign.copy()
    rows = []
    for w in testw:
        before = assign.copy()
        assign, admitted = apply_admissions(assign, w)
        mandatory = len(schedule[w]["entrants"])
        pre_optional = assign.copy()

        if policy == "A1_open":
            members, prob = local_problem(assign, F[w], B)
            assign[members] = solve_a1(prob)
        elif policy == "reABC_open":
            members, prob = local_problem(assign, F[w], B)
            vel = X[max(0, w - 12):w].sum(0)[members]
            target = P.b1_abc(vel, lay)
            assign[members] = prob.budget_repair(target)
        elif policy != "B1_open":
            raise ValueError(policy)

        optional = int((assign != pre_optional).sum())
        assert optional <= B
        required = np.asarray(sorted(schedule[w]["required"]), dtype=int)
        assert np.all(assign[required] >= 0)
        metric = eval_week(lay, assign, lines, orders, w, do_des=True)
        metric.update(
            policy=policy, moves=optional,
            mandatory_admissions=mandatory,
            mandatory_evictions=len(schedule[w]["evicted"]),
            mandatory_budget_charge=0,
            optional_budget_limit=B,
            entrants_signature=_signature(schedule[w]["entrants"]),
            evictions_signature=_signature(schedule[w]["evicted"]),
            membership_signature=_signature(sorted(schedule[w]["members"])),
            admission_slots_signature=_signature(
                [f"{sku}:{slot}" for sku, slot in admitted]
            ),
            admission_mean_depot_m=(
                float(np.mean([lay.slot_depot[slot] for _, slot in admitted]))
                if admitted else np.nan
            ),
            slotted_skus=int((assign >= 0).sum()),
            total_skus=n_sku,
            previously_unseen_lines=int(
                X[w, ~train_active].sum()),
        )
        rows.append(metric)
        print(policy, w, "travel", round(metric["travel_total_km"], 3),
              "optional", optional, "admit", mandatory,
              "evict", len(schedule[w]["evicted"]), flush=True)
    return pd.DataFrame(rows)


result = pd.concat([run("B1_open"), run("A1_open"), run("reABC_open")],
                   ignore_index=True)
result.to_csv(OUT / "retail2_open_confirm.csv", index=False)

p = result.pivot(index="week", columns="policy", values="travel_total_km")
b1 = p["B1_open"]
summary = []
for policy in p:
    summary.append({
        "policy": policy,
        "mean_km": p[policy].mean(),
        "pct_vs_B1": 100 * (p[policy].sum() / b1.sum() - 1),
        "weeks_better_than_B1": int((p[policy] < b1).sum()),
        "worst_week_pct": float(
            (100 * (p[policy] / b1 - 1)).max()),
        "n_test_weeks": len(testw),
        "full_universe_skus": n_sku,
        "test_only_skus": int((~train_active & (X[n_train:].sum(0) > 0)).sum()),
        "test_demand_included_pct": 100.0,
        "mandatory_admissions_total": int(
            result[result.policy == policy]["mandatory_admissions"].sum()),
        "mandatory_evictions_total": int(
            result[result.policy == policy]["mandatory_evictions"].sum()),
    })
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "retail2_open_summary.csv", index=False)
print(summary.round(4).to_string(index=False), flush=True)
