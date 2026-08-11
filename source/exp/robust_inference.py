"""Dependence-robust inference for weekly policy comparisons.

Reports circular moving-block bootstrap intervals at several block lengths,
Newey-West/HAC intervals for the mean weekly percentage effect, and descriptive
non-overlapping four-week block signs.  No independence-based Wilcoxon p-value
is used as the confirmatory claim.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / "out"
RNG_SEED = 20260730
N_BOOT = 50000


def circular_block_indices(n, block, rng):
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(N_BOOT, k))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets) % n
    return idx.reshape(N_BOOT, -1)[:, :n]


def hac_mean_ci(x, lag=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lag is None:
        lag = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    u = x - x.mean()
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    for h in range(1, lag + 1):
        gamma = float(np.dot(u[h:], u[:-h]) / n)
        lrv += 2 * (1 - h / (lag + 1)) * gamma
    se = np.sqrt(max(lrv, 0.0) / n)
    z = norm.ppf(0.975)
    return x.mean(), se, x.mean() - z * se, x.mean() + z * se, lag


def analyse(dataset, frame, treat, ref, week_col="week"):
    p = frame.pivot(index=week_col, columns="policy",
                    values="travel_total_km").sort_index()
    t, r = p[treat].to_numpy(), p[ref].to_numpy()
    weekly_pct = 100 * (t - r) / r
    point_ratio = 100 * (t.sum() / r.sum() - 1)
    point_weekly_mean = float(weekly_pct.mean())
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for block in sorted(set([1, 2, 4, min(6, len(t))])):
        idx = circular_block_indices(len(t), block, rng)
        # Match the manuscript's estimand: arithmetic mean of the weekly
        # percentage changes.  The aggregate ratio-of-sums is retained in a
        # separate column because the two weight weeks differently.
        boot = weekly_pct[idx].mean(axis=1)
        rows.append({
            "dataset": dataset, "comparison": f"{treat} vs {ref}",
            "method": "circular moving-block bootstrap",
            "block_length": block, "n_weeks": len(t),
            "estimate_pct": point_weekly_mean,
            "aggregate_ratio_pct": point_ratio,
            "ci_low_pct": np.quantile(boot, 0.025),
            "ci_high_pct": np.quantile(boot, 0.975),
            "prob_effect_below_zero": np.mean(boot < 0),
        })
    mean, se, lo, hi, lag = hac_mean_ci(weekly_pct)
    rows.append({
        "dataset": dataset, "comparison": f"{treat} vs {ref}",
        "method": "Newey-West HAC mean weekly percent",
        "block_length": lag, "n_weeks": len(t),
        "estimate_pct": mean, "aggregate_ratio_pct": point_ratio,
        "ci_low_pct": lo, "ci_high_pct": hi,
        "prob_effect_below_zero": np.nan,
    })

    block_rows = []
    for start in range(0, len(t), 4):
        stop = min(start + 4, len(t))
        eff = 100 * (t[start:stop].sum() / r[start:stop].sum() - 1)
        block_rows.append({
            "dataset": dataset, "comparison": f"{treat} vs {ref}",
            "block": len(block_rows) + 1,
            "week_start": p.index[start], "week_end": p.index[stop - 1],
            "n_weeks": stop - start, "effect_pct": eff,
            "favorable": bool(eff < 0),
        })
    return rows, block_rows


inference, blocks = [], []
jobs = [
    ("primary", OUT / "exact_actions_weekly.csv",
     [("A1_pool", "B1_ABC_static"),
      ("Exact_assignment_same_state", "B1_ABC_static")]),
    ("retail2_closed", OUT / "retail2_confirm.csv",
     [("A1_pool", "B1_ABC_static")]),
    ("retail2_open", OUT / "retail2_open_confirm.csv",
     [("A1_open", "B1_open"), ("A1_open", "reABC_open")]),
]
for dataset, path, comparisons in jobs:
    if not path.exists():
        print("skip missing", path, flush=True)
        continue
    df = pd.read_csv(path)
    for treat, ref in comparisons:
        r, b = analyse(dataset, df, treat, ref)
        inference.extend(r)
        blocks.extend(b)

inf = pd.DataFrame(inference)
blk = pd.DataFrame(blocks)
inf.to_csv(OUT / "robust_inference.csv", index=False)
blk.to_csv(OUT / "robust_four_week_blocks.csv", index=False)
print(inf.round(4).to_string(index=False), flush=True)
print(blk.round(4).to_string(index=False), flush=True)
