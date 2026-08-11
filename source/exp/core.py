"""Layout generator, Dijkstra distances, S-shape/largest-gap routing, DES replay.
All geometry parameters explicit (Section 4.1 / Section 8 of the plan).
Distances in meters, times in seconds.
"""
import numpy as np, heapq
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

WALK_SPEED = 1.0          # m/s  (plan Section 6)
PICK_TIME = 10.0          # s per line, flow-rack anchor (plan Section 6)
AISLE_PITCH = 3.0         # m between aisle centerlines
SLOT_PITCH = 1.0          # m between slot faces along aisle
CROSS_W = 4.0             # m cross-aisle width


class Layout:
    def __init__(self, n_aisles=20, slots_per_side_block=43, n_blocks=2,
                 depot='corner', slot_pitch=None):
        self.n_aisles = n_aisles
        self.n_blocks = n_blocks
        self.spb = int(slots_per_side_block)
        self.pitch = slot_pitch if slot_pitch is not None else SLOT_PITCH
        self.aisle_len = self.spb * self.pitch
        self.depot = depot
        # slots: (aisle a, block b, side s, pos y). capacity = n_aisles*n_blocks*2*spb
        A, B, S, Y = np.meshgrid(np.arange(n_aisles), np.arange(n_blocks),
                                 np.arange(2), np.arange(self.spb), indexing='ij')
        self.slot_a = A.ravel(); self.slot_b = B.ravel()
        self.slot_y = Y.ravel().astype(float) * self.pitch + self.pitch / 2
        self.n_slots = self.slot_a.size
        # intersection graph: nodes = (aisle_gap g in 0..n_aisles, cross c in 0..n_blocks)
        # aisle i runs between cross c and c+1 at x = i*AISLE_PITCH
        nG, nC = n_aisles + 1, n_blocks + 1
        self.nG, self.nC = nG, nC
        N = nG * nC
        M = lil_matrix((N, N))
        nid = lambda g, c: g * nC + c
        for g in range(nG):
            for c in range(nC):
                if g + 1 < nG:  # along cross-aisle
                    M[nid(g, c), nid(g + 1, c)] = AISLE_PITCH
                    M[nid(g + 1, c), nid(g, c)] = AISLE_PITCH
                if c + 1 < nC:  # along aisle direction
                    d = self.aisle_len + CROSS_W
                    M[nid(g, c), nid(g, c + 1)] = d
                    M[nid(g, c + 1), nid(g, c)] = d
        self.D_int = dijkstra(M.tocsr())
        # depot node: front cross-aisle (c=0), corner g=0 or center
        self.depot_node = nid(0, 0) if depot == 'corner' else nid(nG // 2, 0)
        self.depot_x = 0.0 if depot == 'corner' else (nG // 2) * AISLE_PITCH
        # slot -> distance to depot via nearest aisle-end intersection (Dijkstra-based)
        a, b, y = self.slot_a, self.slot_b, self.slot_y
        d_front = self.D_int[self.depot_node, a * nC + b] + y + CROSS_W / 2
        d_back = self.D_int[self.depot_node, a * nC + b + 1] + (self.aisle_len - y) + CROSS_W / 2
        self.slot_depot = np.minimum(d_front, d_back)
        # prime order: slots sorted by distance to depot
        self.prime_rank = np.argsort(np.argsort(self.slot_depot))
        self.slot_x = a * AISLE_PITCH

    def route_lengths(self, order_aisle, order_block, order_y, order_ptr, policy='sshape'):
        """Vectorized-ish route length per order.
        order_* : concatenated per-line slot attributes; order_ptr: CSR-style offsets.
        S-shape: per visited block band: full traversal of each visited aisle,
        horizontal span 2*(max|x - depot_x| within band chain), serpentine approx.
        Largest-gap: 2*max_y per visited aisle instead of full traversal.
        """
        n_orders = len(order_ptr) - 1
        out = np.zeros(n_orders)
        for i in range(n_orders):
            s, e = order_ptr[i], order_ptr[i + 1]
            if s == e: continue
            aa, bb, yy = order_aisle[s:e], order_block[s:e], order_y[s:e]
            L = 0.0
            for b in np.unique(bb):
                m = bb == b
                aisles = np.unique(aa[m])
                # vertical component
                if policy == 'sshape':
                    v = len(aisles) * (self.aisle_len + CROSS_W)
                    if len(aisles) % 2 == 1:  # odd: last aisle turn-back
                        v += 0  # enter/exit same end handled by cross terms (approx.)
                else:  # largest-gap
                    v = 0.0
                    for A in aisles:
                        v += 2 * yy[m][aa[m] == A].max()
                # horizontal component: depot -> aisle span -> depot (within front cross of band b)
                xs = aisles * AISLE_PITCH
                h = (abs(xs.min() - self.depot_x) + (xs.max() - xs.min()) + abs(xs.max() - self.depot_x))
                # vertical approach to band b front cross-aisle
                approach = 2 * b * (self.aisle_len + CROSS_W)
                L += v + h + approach
            out[i] = L
        return out


def des_replay(travel_s, n_lines, t0_s, n_pickers=25, aisle_seq=None,
               pick_time=PICK_TIME, aisle_cap=3):
    """Event-driven replay: orders (chronological) -> next available picker.
    Service = travel_time + n_lines*pick_time, split across the order's aisle visits.
    Aisle blocking: each aisle band admits `aisle_cap` simultaneous pickers
    (43 m aisle / ~15 m headway); entry waits until a slot frees.
    Returns per-order throughput time and per-order blocking wait.
    """
    order_idx = np.argsort(t0_s)
    picker_free = [0.0] * n_pickers
    heapq.heapify(picker_free)
    aisle_occ = {}  # aisle -> heap of exit times (size <= aisle_cap)
    tput = np.zeros(len(t0_s)); waits = np.zeros(len(t0_s))
    for oi in order_idx:
        start = max(heapq.heappop(picker_free), t0_s[oi])
        wait = 0.0
        if aisle_seq is not None and aisle_seq[oi] is not None and len(aisle_seq[oi]):
            seg = (travel_s[oi] + n_lines[oi] * pick_time) / max(len(aisle_seq[oi]), 1)
            t = start
            for A in aisle_seq[oi]:
                occ = aisle_occ.setdefault(A, [])
                while occ and occ[0] <= t:
                    heapq.heappop(occ)
                if len(occ) >= aisle_cap:
                    free_at = heapq.heappop(occ)
                    if free_at > t:
                        wait += free_at - t
                        t = free_at
                heapq.heappush(occ, t + seg)
                t += seg
            finish = t
        else:
            finish = start + travel_s[oi] + n_lines[oi] * pick_time
        heapq.heappush(picker_free, finish)
        tput[oi] = finish - t0_s[oi]
        waits[oi] = wait
    return tput, waits


def aisle_gini(loads):
    x = np.sort(np.asarray(loads, dtype=float))
    if x.sum() == 0: return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
