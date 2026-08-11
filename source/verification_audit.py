"""Create explicit verification artifacts for the 2026-07-30 reruns."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exp" / "out"

# --- Exact solver semantics and replay signs ---
diag = pd.read_csv(OUT / "exact_actions_diagnostics.csv")
exact_diag = diag[diag["policy"] == "Exact_assignment_same_state"].copy()
weekly = pd.read_csv(OUT / "exact_actions_weekly.csv")
p = weekly.pivot(index="week", columns="policy", values="travel_total_km")
exact_cmp = p[["A1_pool", "Exact_assignment_same_state"]].copy()
exact_cmp["exact_minus_a1_km"] = (
    exact_cmp["Exact_assignment_same_state"] - exact_cmp["A1_pool"]
)
exact_cmp["exact_vs_a1_pct"] = 100 * (
    exact_cmp["Exact_assignment_same_state"] / exact_cmp["A1_pool"] - 1
)
exact_cmp["exact_route_better"] = exact_cmp["exact_minus_a1_km"] < 0
exact_cmp.reset_index().to_csv(OUT / "exact_replay_comparison.csv", index=False)

# --- Open-assortment per-week symmetry and budget accounting ---
open_df = pd.read_csv(OUT / "retail2_open_confirm.csv")
audit_rows = []
for week, frame in open_df.groupby("week"):
    audit_rows.append({
        "week": int(week),
        "n_policies": int(frame["policy"].nunique()),
        "entrants_identical": frame["entrants_signature"].nunique() == 1,
        "evictions_identical": frame["evictions_signature"].nunique() == 1,
        "membership_identical": frame["membership_signature"].nunique() == 1,
        "admission_counts_identical": frame["mandatory_admissions"].nunique() == 1,
        "eviction_counts_identical": frame["mandatory_evictions"].nunique() == 1,
        "mandatory_budget_charge_identical": (
            frame["mandatory_budget_charge"].nunique() == 1
        ),
        "mandatory_budget_charge_zero": (
            frame["mandatory_budget_charge"].eq(0).all()
        ),
        "optional_moves_within_B": (
            frame["moves"].le(frame["optional_budget_limit"]).all()
        ),
        "absolute_admission_slots_identical": (
            frame["admission_slots_signature"].nunique() == 1
        ),
    })
open_audit = pd.DataFrame(audit_rows)
open_audit.to_csv(OUT / "open_assortment_symmetry_audit.csv", index=False)

summary = {
    "verification_date": "2026-07-30",
    "zero_gap_meaning": "solver relative MIP bound gap",
    "exact_all_solver_gaps_zero": bool(
        exact_diag["solver_mip_gap"].eq(0).all()
    ),
    "exact_all_surrogate_gains_above_a1": bool(
        exact_diag["gain_over_warm"].gt(0).all()
    ),
    "exact_candidate_scope": sorted(
        exact_diag["candidate_scope"].dropna().unique().tolist()
    ),
    "exact_full_T_candidate_pool": bool(
        exact_diag["full_T_candidate_pool"].astype(str).str.lower().eq("true").all()
    ),
    "exact_old_head_only": bool(
        exact_diag["old_head_only"].astype(str).str.lower().eq("true").all()
    ),
    "exact_replay": (
        "A1 and exact replayed with identical eval_week machinery from the "
        "same weekly starting state; exact is not an independent trajectory"
    ),
    "exact_route_better_weeks": int(exact_cmp["exact_route_better"].sum()),
    "exact_route_worse_weeks": int((~exact_cmp["exact_route_better"]).sum()),
    "exact_vs_a1_aggregate_pct": float(
        100 * (
            p["Exact_assignment_same_state"].sum() / p["A1_pool"].sum() - 1
        )
    ),
    "exact_vs_a1_weekly_pct_min": float(exact_cmp["exact_vs_a1_pct"].min()),
    "exact_vs_a1_weekly_pct_max": float(exact_cmp["exact_vs_a1_pct"].max()),
    "open_all_weekly_entrants_identical": bool(
        open_audit["entrants_identical"].all()
    ),
    "open_all_weekly_evictions_identical": bool(
        open_audit["evictions_identical"].all()
    ),
    "open_all_weekly_membership_identical": bool(
        open_audit["membership_identical"].all()
    ),
    "open_all_mandatory_budget_charges_zero": bool(
        open_audit["mandatory_budget_charge_zero"].all()
    ),
    "open_all_optional_moves_within_B": bool(
        open_audit["optional_moves_within_B"].all()
    ),
    "open_absolute_admission_slots_identical_weeks": int(
        open_audit["absolute_admission_slots_identical"].sum()
    ),
    "open_total_test_weeks": int(len(open_audit)),
    "open_placement_interpretation": (
        "identical admission/eviction/member rule and zero mandatory budget "
        "charge; absolute entrant slots may differ under policy-specific layouts"
    ),
}
(OUT / "verification_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
