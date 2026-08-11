"""Pass-1 additions: largest-gap routing comparison + FCFS batch picking.
Both reuse the existing Layout.route_lengths / des_replay machinery.

FCFS batching (Section 6 secondary variant, previously not run):
  orders arrive chronologically; group up to `cap` consecutive orders (by t0)
  into one pick tour; the tour's line set = union of the batch's lines;
  route length computed over the union (S-shape). This is the standard
  variable-time-window FCFS batch heuristic (de Koster et al. 2007).
Aisle-visit sequence for DES = union of visited (block,aisle) across the batch.
"""
import numpy as np, time
from exp.core import aisle_gini, WALK_SPEED, PICK_TIME, des_replay


def eval_week_routed(layout, assign, lines, orders, w, routing='sshape',
                     batch_cap=1, n_pickers=25, do_des=True):
    """Generalized weekly eval supporting routing policy and FCFS batching.
    batch_cap=1 -> single-order picking (identical to runner.eval_week)."""
    from exp.runner import week_orders
    O, si, ptr = week_orders(lines, orders, w)
    slot = assign[si]
    oa, ob, oy = layout.slot_a[slot], layout.slot_b[slot], layout.slot_y[slot]
    t0s = (O['t0'] - O['t0'].min()).dt.total_seconds().values
    n_orders = len(O)

    if batch_cap <= 1:
        groups = [(i, i + 1) for i in range(n_orders)]  # (line-order-start, end) as order indices
        batch_ptr = [(ptr[i], ptr[i + 1]) for i in range(n_orders)]
        batch_t0 = t0s
    else:
        # FCFS: consecutive orders (already t0-sorted) grouped in blocks of batch_cap
        order_idx = np.arange(n_orders)  # O is t0-sorted in week_orders
        groups = []
        batch_ptr = []
        batch_t0 = []
        i = 0
        while i < n_orders:
            j = min(i + batch_cap, n_orders)
            # union of line slices [ptr[i]:ptr[j]] is contiguous since lines sorted by oi
            batch_ptr.append((ptr[i], ptr[j]))
            batch_t0.append(t0s[i])          # release at first order's arrival (FCFS window)
            groups.append((i, j))
            i = j
        batch_t0 = np.array(batch_t0)

    nb = len(batch_ptr)
    # build CSR for route_lengths over batches
    seg_a, seg_b, seg_y = [], [], []
    bptr = np.zeros(nb + 1, dtype=int)
    for k, (s, e) in enumerate(batch_ptr):
        seg_a.append(oa[s:e]); seg_b.append(ob[s:e]); seg_y.append(oy[s:e])
        bptr[k + 1] = bptr[k] + (e - s)
    seg_a = np.concatenate(seg_a) if seg_a else np.array([], int)
    seg_b = np.concatenate(seg_b) if seg_b else np.array([], int)
    seg_y = np.concatenate(seg_y) if seg_y else np.array([], float)

    t = time.time()
    dist = layout.route_lengths(seg_a, seg_b, seg_y, bptr, routing)
    rt = time.time() - t

    n_lines_batch = np.diff(bptr)
    res = {'week': w, 'orders': n_orders, 'lines': len(si),
           'n_tours': nb, 'batch_cap': batch_cap, 'routing': routing,
           'travel_total_km': dist.sum() / 1000,
           'travel_per_order_m': dist.sum() / max(n_orders, 1),  # per-order to compare fairly
           'route_calc_s': rt}
    key = oa + layout.n_aisles * ob
    loads = np.zeros(layout.n_aisles * layout.n_blocks)
    np.add.at(loads, key, 1.0)
    res['aisle_gini'] = aisle_gini(loads)

    if do_des:
        travel_s = dist / WALK_SPEED
        seqs = []
        for s, e in batch_ptr:
            seqs.append(list(dict.fromkeys(zip(seg_b_slice(ob, s, e), seg_a_slice(oa, s, e)))))
        tput, waits = des_replay(travel_s, n_lines_batch, batch_t0, n_pickers, seqs)
        # throughput reported per underlying order-equivalent for comparability
        res['tput_mean_min'] = tput.mean() / 60
        res['tput_p90_min'] = np.percentile(tput, 90) / 60
        res['block_wait_total_h'] = waits.sum() / 3600
    return res


def seg_a_slice(oa, s, e):
    return oa[s:e]


def seg_b_slice(ob, s, e):
    return ob[s:e]
