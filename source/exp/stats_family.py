"""Pre-registered inference: paired two-sided Wilcoxon signed-rank over WEEKLY deltas
(tests), paired bootstrap for CIs ONLY (never for p-values), Holm across the frozen
primary family. Reused verbatim for the sealed Retail II confirmatory run.
Usage: compute_family(piv, primary=[(name, colA, colB), ...], descriptive=[...], ref_col)
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

def _bootci(delta, seed=20260724, n=20000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (n, len(delta)))
    m = delta[idx].mean(1)
    return float(delta.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def _wilcoxon(a, b):
    stat, p = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    return float(p)

def _cliffs(a, b):
    d = a[:, None] - b[None, :]
    return float(np.sign(-d).sum() / (len(a) * len(b)))  # a<b favorable

def compute_family(piv, primary, descriptive, ref_col, seed=20260724):
    ref_mean = piv[ref_col].values.mean()
    rows = []
    for name, ca, cb in primary:
        a, b = piv[ca].values, piv[cb].values
        m, lo, hi = _bootci(a - b, seed)
        rows.append(dict(comparison=name, family='primary', delta=m, ci_lo=lo, ci_hi=hi,
                         pct=100 * m / ref_mean, weeks_favorable=int((a < b).sum()),
                         n_weeks=len(a), cliffs_delta=_cliffs(a, b), p_wilcoxon=_wilcoxon(a, b)))
    ps = [r['p_wilcoxon'] for r in rows]; order = np.argsort(ps); m_fam = len(rows); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (m_fam - rank))); rows[i]['p_holm'] = run
    for name, ca, cb in descriptive:
        a, b = piv[ca].values, piv[cb].values
        m, lo, hi = _bootci(a - b, seed)
        rows.append(dict(comparison=name, family='descriptive', delta=m, ci_lo=lo, ci_hi=hi,
                         pct=100 * m / ref_mean, weeks_favorable=int((a < b).sum()),
                         n_weeks=len(a), cliffs_delta=_cliffs(a, b),
                         p_wilcoxon=_wilcoxon(a, b), p_holm=np.nan))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    OUT = '/workspace/scratch/d6d77d1d2b0a/warehouse_runs/work/exp/out'
    d = pd.read_csv(f'{OUT}/pass1_solvers.csv')
    piv = d.pivot_table(index='week', columns='policy', values='travel_total_km')
    st = compute_family(piv,
        primary=[('A1 vs B1', 'A1_pool', 'B1_ABC_static'),
                 ('A1 vs reABC', 'A1_pool', 'reABC_budgeted')],
        descriptive=[('reABC vs B1', 'reABC_budgeted', 'B1_ABC_static')],
        ref_col='B1_ABC_static')
    st.to_csv(f'{OUT}/stats.csv', index=False)
    print(st.round(5).to_string(index=False))
