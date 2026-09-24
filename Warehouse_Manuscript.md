# Warehouse re-slotting manuscript: independent rerun report

**Run date:** 30 July 2026  
**Status:** Primary reproduction, exact local benchmark, closed- and open-assortment confirmation, routing/batching, staffing, layout sensitivity, and dependence-robust inference completed successfully.

## Executive result

| Analysis | Manuscript-style mean weekly change | Ratio of total distances | Favorable weeks | Four-week circular-block 95% CI |
|---|---:|---:|---:|---:|
| Primary Warehouse Science, A1 vs static ABC | −2.68% | −2.68% | 9/9 | [−3.31%, −1.97%] |
| Retail II closed assortment, A1 vs static ABC | −8.97% | −9.20% | 42/42 | [−9.93%, −7.93%] |
| Retail II open assortment, A1 vs static ABC | −15.83% | −16.55% | 42/42 | [−17.47%, −14.17%] |
| Retail II open assortment, A1 vs budgeted re-ABC | −8.55% | −8.94% | 42/42 | [−9.41%, −7.71%] |

The closed-assortment primary and Retail II weekly travel outputs reproduce the archived experiment CSVs exactly. The apparent −8.97% versus −9.20% discrepancy is not a rerun discrepancy: −8.97% is the arithmetic mean of weekly percentage changes used in the manuscript, while −9.20% is the ratio of total A1 distance to total B1 distance.

Input reconstruction was also checked before running policies. The Warehouse Science source yielded 2,079,011 pick rows, 130 pickers, 3,294 raw SKU codes, and the documented 3 January–31 May 2017 date range; preprocessing produced the same 3,240-SKU, 22-week arrays used by the archived code. The two-sheet Retail II workbook yielded 1,067,371 raw rows and 1,036,970 rows after the frozen filters, with the same 62/42 train/test split and gate statistics as the manuscript package.

## 1. Swap-accounted exact benchmark

The old benchmark was head-only and used a per-SKU budget proxy. I replaced it with a sparse exact assignment MIP over every weekly decision SKU and its current decision-slot permutation:

\[
\min_x \sum_{i,j} c_{ij}x_{ij}
\]

\[
\sum_j x_{ij}=1,\qquad
\sum_i x_{ij}=1,\qquad
\sum_{i\ne j}x_{ij}\le 100.
\]

The final constraint is the realized number of relocated SKUs. A two-SKU occupied-slot swap therefore costs two moves; no eviction is left uncharged. The action graph contains every identity arc, each SKU's 30 strongest pairwise partners and reciprocal arcs, and every arc used by the deployed A1 incumbent. This makes A1 feasible in the exact model while extending the neighborhood to cycles longer than two.

The scope needs to be stated precisely: this is broader than the old top-head benchmark because it includes all 1,684–1,701 weekly `ReslotProblem` decision SKUs, but it does **not** optimize over the complete `T` candidate-slot pool. It permutes the decision SKUs' current slots over the declared sparse arc set. It is therefore a certified optimum of that broad local surrogate formulation, not a global warehouse-layout optimum.

A1 and exact are paired from the identical A1-trajectory starting assignment each week, use the same forecast and \(B=100\), and are replayed by the identical `eval_week` S-shape route/DES machinery. The exact rows are a same-state weekly solver diagnostic, not a separately evolving exact-policy trajectory.

Across the nine primary test weeks:

- 1,684–1,701 decision SKUs and 86,494–90,241 assignment arcs were solved per week.
- All nine MIPs terminated optimal with a **solver-reported relative bound gap of 0.0**. This is the optimizer's own certificate and does not mean a zero difference from A1.
- Per-week solve times and node counts are retained in the diagnostics; runtime is treated as environment-dependent rather than a scientific effect.
- The exact solution's summed forecast-distance surrogate improvement is 2.31% larger than A1's summed improvement from the same states; the exact improvement is positive relative to A1 in every week.
- Despite the better surrogate, the exact assignments produce 0.15% more replayed route distance than A1 in aggregate. The weekly sign is mixed: exact routed travel is lower in 2/9 weeks and higher in 7/9, with weekly differences from −0.13% to +0.42%.

This is a useful methodological result: the residual issue is surrogate-to-route fidelity, not an unaccounted swap budget or a large local search gap. The manuscript should not describe surrogate optimality and routed-distance optimality as interchangeable.

## 2. Open-assortment Retail II robustness run

The original confirmatory run restricts evaluation to 4,253 training-active SKUs and excludes 628 test-only SKUs, which represent 20.86% of test lines. The new robustness run includes all 4,881 SKUs and 100% of test demand in the same 4,480-slot layout.

A policy-independent membership schedule admits every SKU known to be in the week's operational assortment. Once capacity is full, it evicts a dormant SKU with the smallest trailing-12-week demand. The entrants, evictions, and resulting membership set are identical across all three arms in every week. Mandatory receipt/admission work is reported separately and is charged zero against the \(B=100\) discretionary re-slotting budget in every arm; optional A1 and re-ABC changes are capped at 100. The run records 631 admissions and 404 evictions; the slotted assortment ranges from 4,253 to 4,480 SKUs.

The admission **rule** is identical, but absolute admission slots are not forcibly held identical after policy layouts diverge. Entrants use a scheduled evicted SKU's reclaimed slot first and then the nearest currently free slot, so the same evicted SKU can release different physical slots under different policy assignments. The audit finds identical absolute admission-slot mappings in 39/42 test weeks and differences in weeks 94, 95, and 101. This is symmetric rule and budget treatment, not a common-slot counterfactual. The open-assortment result should remain a post-hoc robustness analysis unless a stricter common-absolute-slot design is additionally required.

Results are favorable in every week:

- A1 vs static ABC: −15.83% mean weekly distance, range −22.62% to −7.73%.
- A1 vs budgeted re-ABC: −8.55%, range −13.57% to −4.66%.
- Budgeted re-ABC vs static ABC: −7.99%.

This design was constructed after examining the closed-assortment limitation. It should be reported as a post-hoc robustness analysis, not relabeled as preregistered confirmation. It also assumes that the weekly available assortment is operationally known before slotting; demand quantities remain forecast from past data.

## 3. Routing, batching, layout, and staffing

### Routing and FCFS batching

| Routing | Batch cap | A1 travel change vs B1 | Blocking change |
|---|---:|---:|---:|
| S-shape | 1 | −2.68% | −0.12% |
| Largest-gap | 1 | −2.70% | −1.94% |
| S-shape | 2 | −2.76% | +1.41% |
| S-shape | 4 | −2.94% | −5.23% |
| Largest-gap | 4 | −2.97% | −2.58% |

The travel conclusion survives every arm. Blocking does not have a stable direction and should remain a secondary outcome.

### Layout OFAT

| Layout cell | A1 travel change vs B1 |
|---|---:|
| Base | −2.68% |
| One block | −0.61% |
| Four blocks | −3.33% |
| Shorter-aisle geometry | −2.00% |
| Longer-aisle geometry | −2.66% |
| Center depot | −1.52% |
| Slot pitch 0.8 m | −1.76% |
| Slot pitch 1.25 m | −2.22% |

The two largest A1 travel main effects are block count and slot pitch. Across their full \(3\times3\) interaction grid, A1 remains favorable in all nine cells, ranging from −0.61% to −3.33%.

### Staffing

Across 5, 15, 25, 40, and 50 pickers, travel is unchanged at −2.68% because routing and slotting do not depend on staffing. Mean throughput time improves by 3.84%–9.46%, and p90 throughput improves by 4.84%–7.38%. Blocking changes sign (from +4.11% at five pickers to −3.58% at 50), again arguing against a uniform congestion claim.

## 4. Dependence-robust inference

The inference file reports:

- circular moving-block bootstrap intervals at block lengths 1, 2, 4, and 6;
- Newey–West/HAC intervals for the mean weekly percentage effect; and
- descriptive non-overlapping four-week block effects.

All four-week bootstrap intervals in the executive table are below zero. Newey–West intervals agree. Every non-overlapping four-week block is favorable for the primary, closed Retail II, open Retail II, and open A1-versus-re-ABC comparisons.

The primary dataset still has only nine test weeks. The robust intervals address serial dependence but do not manufacture independent information, so the small time dimension should remain an explicit limitation.

## 5. Recommended manuscript changes

1. Replace the old head-only optimality-gap paragraph with the swap-accounted exact assignment diagnostic. State clearly that it certifies the declared local surrogate problem, not the global routed problem.
2. Add the full-universe Retail II analysis as a labeled post-hoc robustness section and retain the original closed-assortment result as the preregistered confirmation.
3. Report the manuscript-style mean weekly percentage as the main estimand, with the ratio of total distances as a secondary weighted estimand.
4. Replace independence-based significance language with the moving-block and HAC intervals. The Wilcoxon calculation may remain descriptive if clearly qualified.
5. Keep travel as the central result. Report throughput as supportive and blocking as mixed.
6. Release the filtering log, exact MIP diagnostics, membership/admission rule, weekly outputs, dependency versions, and checksums with the revision.

## 6. Acceptance implication

These runs resolve the most consequential technical objections identified in the earlier review: unseen-SKU exclusion, swap accounting in the exact reference, final-solver sensitivity, and serial dependence. They also reveal one honest limitation—the surrogate optimum is not the routed-distance optimum—which should be discussed rather than hidden.

My judgment remains:

- **Current manuscript without these revisions:** about 45% probability of acceptance at *Computers & Industrial Engineering* after the normal review/revision cycle.
- **Carefully revised manuscript incorporating these results and reproducibility material:** roughly 60%–70%.

These are editorial judgments, not statistical probabilities. The open-assortment rule is post hoc, the primary horizon is short, and the study remains simulation-based; those factors prevent a higher estimate.

## 7. Reproduction order

Run from the reconstructed project root:

```bash
python prepare_inputs.py
python exp/prep.py
python exp/retail2_gates.py
python exp/retail2_pass2.py calib
python exp/retail2_pass2.py confirm
python exp/run_exact_actions.py
python exp/retail2_open.py
python exp/run_routing_arms.py
python exp/run_sens.py
python exp/robust_inference.py
```


