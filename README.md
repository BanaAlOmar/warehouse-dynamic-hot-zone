# Warehouse Dynamic Hot-Zone Identification — rerun and verification artifact

Replication artifact for forecast-driven warehouse re-slotting under a binding
relocation budget. It contains the portable experiment source, the compact
processed inputs, and the final result tables for the runs described in
[`Warehouse_Manuscript.md`](Warehouse_Manuscript.md).

## Directories

- `source/`: portable experiment source used for the reruns.
- `processed_inputs/`: compact derived arrays/tables needed to repeat policy evaluation without rebuilding the public raw datasets.
- `results/`: weekly results, exact-MIP diagnostics, sensitivity outputs, filtering/calibration metadata, and robust-inference tables.

## Fast rerun from bundled processed inputs

```bash
mkdir -p source/exp/out
cp processed_inputs/* source/exp/out/
cp results/calib_chosen.json results/retail2_gates.json \
   results/retail2_calib.json source/exp/out/
cd source
python exp/run_exact_actions.py
python exp/retail2_pass2.py confirm
python exp/retail2_open.py
python exp/run_routing_arms.py
python exp/run_sens.py
python exp/robust_inference.py
```

Install the pinned dependencies first (Python 3.12.13):

```bash
pip install -r requirements.txt
```

## Full rebuild from public raw inputs

Place `picks.csv`, `SKUs.csv`, and `online_retail_II.xlsx` in `source/data/`, then run:

```bash
cd source
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

The raw Warehouse Science and Online Retail II files are public but are not duplicated here.

## Key result files

- `results/exact_actions_diagnostics.csv`: exact sparse-assignment status, gaps, dimensions, objective gains, and solve times.
- `results/exact_replay_comparison.csv`: weekly routed-distance sign of exact versus A1.
- `results/retail2_open_confirm.csv`: full-universe weekly replay including all 628 test-only SKUs.
- `results/open_assortment_symmetry_audit.csv`: per-week admission, eviction, membership, slot, and budget symmetry checks.
- `results/routing_arms.csv`: routing and FCFS batching sensitivity.
- `results/sens_ofat.csv`, `sens_pickers.csv`, `sens_factorial.csv`: final-solver layout and staffing sensitivity.
- `results/robust_inference.csv`: moving-block bootstrap and HAC intervals.
- `results/robust_four_week_blocks.csv`: four-week circular-block effects per comparison.
- `results/verification_summary.json`: machine-readable answers to the gap, scope, replay, open-assortment, and weekly-sign questions.

## Licence

Code under MIT (`LICENSE`). Derived data, result tables and report text under
CC BY 4.0 (`LICENSE-DATA`). Neither raw source dataset is redistributed; see
`LICENSE-DATA` for third-party attribution, including Online Retail II
(Chen, D., 2012, UCI Machine Learning Repository, https://doi.org/10.24432/C5CG6D).

## Citation

Manuscript under review. Until a DOI is issued, cite this repository. A Zenodo
deposit will be minted at acceptance and the DOI added here and to the paper's
Data Availability statement.
