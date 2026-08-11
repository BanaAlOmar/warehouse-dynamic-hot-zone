"""Shared evaluation machinery: weekly replay of any policy trajectory."""
from pathlib import Path
import numpy as np, pandas as pd, time, json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp.core import Layout, des_replay, aisle_gini, WALK_SPEED, PICK_TIME
from exp import policies as P

OUT = str(ROOT / 'exp' / 'out')

def load_all():
    meta = json.load(open(f'{OUT}/forecast_meta.json'))
    skus = meta['skus']; sku_ix = {s: i for i, s in enumerate(skus)}
    F = np.load(f'{OUT}/forecast.npy')   # [week-1, sku] forecast FOR that week
    X = np.load(f'{OUT}/actual.npy')
    lines = pd.read_parquet(f'{OUT}/lines.parquet')
    orders = pd.read_parquet(f'{OUT}/orders.parquet')
    lines['si'] = lines['SKU'].map(sku_ix)
    master = pd.read_parquet(f'{OUT}/sku_master.parquet').set_index('SKU')
    unit_vol = np.array([master.loc[s, 'unit_vol'] for s in skus])
    imputed = np.array([master.loc[s, 'imputed'] for s in skus])
    return meta, skus, F, X, lines, orders, unit_vol, imputed


def week_orders(lines, orders, w):
    L = lines[lines['w'] == w]
    O = orders[orders['w'] == w].sort_values('t0').reset_index(drop=True)
    oix = {o: i for i, o in enumerate(O['ORDER_ID'])}
    L = L.copy(); L['oi'] = L['ORDER_ID'].map(oix)
    L = L.dropna(subset=['oi']).sort_values('oi')
    ptr = np.zeros(len(O) + 1, dtype=int)
    cnt = L.groupby('oi').size()
    ptr[np.asarray(cnt.index, dtype=int) + 1] = cnt.values
    ptr = np.cumsum(ptr)
    return O, L['si'].values.astype(int), ptr


def eval_week(layout, assign, lines, orders, w, n_pickers=25, do_des=True):
    """Replay week w under assignment. Returns metrics dict."""
    O, si, ptr = week_orders(lines, orders, w)
    slot = assign[si]
    oa, ob, oy = layout.slot_a[slot], layout.slot_b[slot], layout.slot_y[slot]
    t = time.time()
    dist = layout.route_lengths(oa, ob, oy, ptr, 'sshape')
    rt = time.time() - t
    res = {'week': w, 'orders': len(O), 'lines': len(si),
           'travel_total_km': dist.sum() / 1000,
           'travel_per_order_m': dist.mean(), 'route_calc_s': rt}
    # aisle load gini (actual demand)
    key = oa + layout.n_aisles * ob
    loads = np.zeros(layout.n_aisles * layout.n_blocks)
    np.add.at(loads, key, 1.0)
    res['aisle_gini'] = aisle_gini(loads)
    if do_des:
        t0s = (O['t0'] - O['t0'].min()).dt.total_seconds().values
        nl = np.diff(ptr)
        travel_s = dist / WALK_SPEED
        # aisle visit sequences
        seqs = []
        for i in range(len(O)):
            s, e = ptr[i], ptr[i + 1]
            seqs.append(list(dict.fromkeys(zip(ob[s:e], oa[s:e]))))
        tput, waits = des_replay(travel_s, nl, t0s, n_pickers, seqs)
        res['tput_mean_min'] = tput.mean() / 60
        res['tput_p90_min'] = np.percentile(tput, 90) / 60
        res['block_wait_total_h'] = waits.sum() / 3600
    return res


def run_policy_trajectory(name, layout, F, X, lines, orders, weeks, init_assign,
                          make_step, n_pickers=25, do_des=True):
    """make_step(prev_assign, w) -> new assign. Returns weekly metrics + moves."""
    assign = init_assign.copy()
    rows = []
    for w in weeks:
        t = time.time()
        new_assign = make_step(assign, w)
        opt_s = time.time() - t
        moves = int((new_assign != assign).sum())
        assert len(np.unique(new_assign)) == len(new_assign), f"{name} w{w}: duplicate slots!"
        assign = new_assign
        m = eval_week(layout, assign, lines, orders, w, n_pickers, do_des)
        m.update(policy=name, moves=moves, opt_time_s=round(opt_s, 2))
        rows.append(m)
    return pd.DataFrame(rows), assign
