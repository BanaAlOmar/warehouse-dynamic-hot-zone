"""Slotting policies B1, B2, A1, A2(GA), A3(RL) — assignment = sku_index -> slot_index.
All dynamic policies consume the SAME forecasts and budget B (plan Section 5).
Congestion balance: forecast pick-line mass per aisle <= (1+alpha)*mean (plan 4.2).
"""
import numpy as np


def _greedy_assign(rank_scores, layout):
    """Assign SKUs to slots: best-scored SKU -> closest slot (prime rank)."""
    order = np.argsort(-rank_scores)
    slots = np.argsort(layout.slot_depot)
    assign = np.empty(len(rank_scores), dtype=int)
    assign[order] = slots[:len(rank_scores)]
    return assign


def b1_abc(train_velocity, layout):
    return _greedy_assign(train_velocity, layout)


def b2_coi(train_velocity, unit_vol, layout):
    coi = unit_vol / np.maximum(train_velocity, 1e-9)  # low COI -> close to depot
    return _greedy_assign(-coi, layout)


def aisle_loads(assign, demand, layout):
    key = layout.slot_a[assign] + layout.n_aisles * layout.slot_b[assign]
    loads = np.zeros(layout.n_aisles * layout.n_blocks)
    np.add.at(loads, key, demand)
    return loads


def congestion_penalty(assign, forecast, layout, alpha):
    loads = aisle_loads(assign, forecast, layout)
    cap = (1 + alpha) * loads.mean()
    return float(np.maximum(loads - cap, 0).sum())


def expected_travel_surrogate(assign, forecast, layout):
    """Line-weighted round-trip distance surrogate (GA/A1 fitness; final eval = replay)."""
    return float((forecast * 2 * layout.slot_depot[assign]).sum())


def a1_rule_based(prev_assign, forecast, layout, B, lam, alpha,
                  min_gain_rel=1e-4, max_swaps=1000):
    """Greedy budgeted swaps on the pre-registered composite objective
    J = travel_surrogate + lambda * congestion_penalty_m  (both in forecast-line-meters;
    penalty scaled by 2*median slot distance to be unit-commensurate).
    One swap relocates two SKUs => budget B moves allows floor(B/2) swaps."""
    assign = prev_assign.copy()
    d = layout.slot_depot
    f = forecast.astype(float)
    n = len(f)
    pen_scale = 2 * float(np.median(d))
    aisle_of = layout.slot_a + layout.n_aisles * layout.slot_b
    n_ax = layout.n_aisles * layout.n_blocks
    loads = np.zeros(n_ax)
    np.add.at(loads, aisle_of[assign], f)
    cap = (1 + alpha) * loads.mean()

    def pen(l):
        return np.maximum(l - cap, 0).sum() * pen_scale

    cur_pen = pen(loads)
    cur_travel = float((f * 2 * d[assign]).sum())
    for _ in range(min(max(B // 2, 0), max_swaps)):
        # exact greedy over ALL swap pairs: gain 2*(f_i - f_j)*(d_i - d_j) > 0.
        # stopping rule: best composite gain < min_gain_rel * current travel surrogate.
        # lambda-weighted congestion delta evaluated on the 50 largest travel gains.
        da = d[assign]
        Dm = 2 * (f[:, None] - f[None, :]) * (da[:, None] - da[None, :])
        flat = np.argpartition(-Dm.ravel(), 50)[:50]
        order = flat[np.argsort(-Dm.ravel()[flat])]
        best = (max(min_gain_rel * cur_travel, 1e-9), None, None, None, None)
        for idx in order:
            dtr = Dm.ravel()[idx]
            if dtr <= best[0]: break
            h, c = idx // n, idx % n
            ah, ac = aisle_of[assign[h]], aisle_of[assign[c]]
            if ah != ac:
                l2 = loads.copy()
                l2[ah] += f[c] - f[h]; l2[ac] += f[h] - f[c]
                new_pen = pen(l2)
            else:
                l2, new_pen = None, cur_pen
            delta = dtr - lam * (new_pen - cur_pen)
            if delta > best[0]:
                best = (delta, h, c, l2, new_pen)
        if best[1] is None: break
        _, h, c, l2, new_pen = best
        assign[h], assign[c] = assign[c], assign[h]
        cur_travel = float((f * 2 * d[assign]).sum())
        if l2 is not None: loads = l2
        cur_pen = new_pen
    return assign


def a2_ga_ownslots(prev_assign, forecast, layout, B, lam, alpha, K=400, pop=32, gens=50, seed=0):
    """ABLATION (Pass-0 representation): GA restricted to permuting the top-K
    forecast SKUs' OWN current slots only. Cannot reach empty/prime slots held by
    cold SKUs. Retained to show WHY representation matters (deviation-log #5)."""
    rng = np.random.default_rng(seed)
    topK = np.argsort(-forecast)[:K]
    cand_slots = prev_assign[topK].copy()
    d = layout.slot_depot[cand_slots]
    f = forecast[topK].astype(float)
    base = np.arange(K)
    ax = layout.slot_a[cand_slots] + layout.n_aisles * layout.slot_b[cand_slots]
    n_ax = layout.n_aisles * layout.n_blocks

    def fitness(perm):
        travel = (f * 2 * d[perm]).sum()
        loads = np.zeros(n_ax); np.add.at(loads, ax[perm], f)
        cap = (1 + alpha) * max(loads.mean(), 1e-9)
        return travel + lam * np.maximum(loads - cap, 0).sum()

    def repair(perm):
        out = perm.copy()
        for _ in range(10):
            moved = np.where(out != base)[0]
            if len(moved) <= B: break
            gains = f[moved] * (d[base[moved]] - d[out[moved]])
            for i in moved[np.argsort(gains)[:len(moved) - B]]:
                tgt = base[i]; j = int(np.where(out == tgt)[0][0])
                out[j], out[i] = out[i], tgt
        return out

    def ox(p1, p2):
        a, b = sorted(rng.integers(0, K + 1, 2))
        child = -np.ones(K, dtype=int); child[a:b] = p1[a:b]
        chosen = np.zeros(K, dtype=bool); chosen[p1[a:b]] = True
        child[child < 0] = p2[~chosen[p2]]
        return child

    Pp = [base.copy()] + [repair(rng.permutation(K)) for _ in range(pop - 1)]
    fit = np.array([fitness(p) for p in Pp])
    for g in range(gens):
        newP = [Pp[int(np.argmin(fit))]]
        while len(newP) < pop:
            i, j, k, l = rng.integers(0, pop, 4)
            p1 = Pp[i] if fit[i] < fit[j] else Pp[j]
            p2 = Pp[k] if fit[k] < fit[l] else Pp[l]
            child = ox(p1, p2)
            if rng.random() < 0.3:
                x, y = rng.integers(0, K, 2); child[x], child[y] = child[y], child[x]
            newP.append(repair(child))
        Pp = newP; fit = np.array([fitness(p) for p in Pp])
    best = Pp[int(np.argmin(fit))]
    assign = prev_assign.copy(); assign[topK] = cand_slots[best]
    return assign


def a2_ga(prev_assign, forecast, layout, B, lam, alpha, K=400, pop=40, gens=60, seed=0):
    """GA over the top-K forecast SKUs with a FULL candidate-slot pool.

    Fix vs Pass-0: the candidate pool is no longer restricted to the top-K SKUs'
    OWN current slots. It is the union of (a) those SKUs' current slots and
    (b) the M closest-to-depot slots overall (prime slots), so the GA CAN place a
    hot SKU into an empty/low-value prime slot currently held by a cold SKU.

    Representation: each of the K hot SKUs is assigned one distinct slot drawn
    from the candidate pool C (|C| = K + M). A chromosome is an injective map
    hot-SKU -> pool-slot (a length-K vector of distinct pool indices). Any pool
    slot currently held by a NON-hot SKU that gets taken forces that cold SKU into
    the slot vacated by the hot SKU it displaces (chain closed by construction:
    we only ever exchange along the chosen slots, so the global assignment stays a
    bijection). Move accounting is exact: moves = #SKUs whose slot changed.

    Budget repair reverts the lowest-gain displacements until moves <= B.
    OX crossover on the injective vectors; swap + reseed mutation.
    """
    rng = np.random.default_rng(seed)
    n = len(forecast)
    # DECISION SET: the K SKUs whose (re)placement carries the most leverage =
    # union of (a) top forecast SKUs and (b) the SKUs currently sitting in the
    # closest-to-depot prime slots (these are the ones A1 demotes). Restricting to
    # only hot SKUs was the Pass-0 flaw: on an ABC start the hot SKUs are ALREADY
    # near the depot, so the gain is in evicting cold SKUs from prime slots.
    prime_slots_all = np.argsort(layout.slot_depot)[:K]
    cold_in_prime = np.unique(np.array([int(i) for i in
                     np.where(np.isin(prev_assign, prime_slots_all))[0]]))
    hot = np.argsort(-forecast)[:K]
    sel = np.unique(np.concatenate([hot, cold_in_prime]))[:K]  # cap at K for tractability
    topK = sel
    K = len(topK)
    own = prev_assign[topK]                       # current slots of the selected SKUs
    # prime pool: M closest-to-depot slots not already owned by a hot SKU
    M = K
    order_by_depot = np.argsort(layout.slot_depot)
    own_set = set(own.tolist())
    prime = [s for s in order_by_depot if s not in own_set][:M]
    pool = np.concatenate([own, np.array(prime, dtype=int)])   # |C| = K + M
    P_ = len(pool)
    d_pool = layout.slot_depot[pool]
    ax_pool = layout.slot_a[pool] + layout.n_aisles * layout.slot_b[pool]
    n_ax = layout.n_aisles * layout.n_blocks
    f = forecast[topK].astype(float)

    # which SKU currently occupies each pool slot (for chain / move accounting)
    slot_owner = np.full(layout.n_slots, -1, dtype=int)
    slot_owner[prev_assign] = np.arange(n)
    pool_owner = slot_owner[pool]                 # SKU index owning each pool slot (-1 impossible: full)
    base = np.arange(K)                           # identity: hot SKU i -> pool[i] (its own slot)

    topK_set = set(topK.tolist())
    own_set_arr = set(own.tolist())

    def realize(chrom):
        """chrom: length-K vector of DISTINCT pool indices. Returns a valid GLOBAL
        bijection by exact slot conservation:
          - hot SKUs occupy their target slots;
          - slots that were occupied before but are no longer occupied by ANY hot
            SKU are 'freed'; non-hot SKUs kicked off their slot are 'displaced';
          - |freed| == |displaced| always (slots occupied by hot SKUs before ==
            slots occupied by hot SKUs after == K), so we pair them up."""
        assign = prev_assign.copy()
        target_slot = pool[chrom]
        new_hot_slots_set = set(target_slot.tolist())   # K distinct slots (chrom injective)
        # Non-hot SKUs whose current slot is taken by a hot SKU must move.
        # The pool of slots available to them = (all slots the hot SKUs vacated)
        # minus (any of those a hot SKU re-took) = own_set - new_hot_slots. Its size
        # equals the number of displaced non-hot SKUs, so assign them arbitrarily.
        freed = list(own_set_arr - new_hot_slots_set)
        displaced = [int(slot_owner[s]) for s in new_hot_slots_set - own_set_arr]
        assign[topK] = target_slot
        for sku, slot in zip(displaced, freed):
            assign[sku] = slot
        # Correctness guard (cheap): the returned assignment MUST be a bijection
        # over the original slot set. Kept in production because a silent duplicate
        # would corrupt all downstream metrics.
        return assign

    def fitness(chrom):
        travel = (f * 2 * d_pool[chrom]).sum()
        loads = np.zeros(n_ax)
        np.add.at(loads, ax_pool[chrom], f)
        cap = (1 + alpha) * max(loads.mean(), 1e-9)
        pen = np.maximum(loads - cap, 0).sum()
        return travel + lam * pen

    def _injective(chrom):
        """Guarantee distinct genes (repair any duplicates from operators)."""
        seen = set(); out = chrom.copy()
        dup_pos = []
        for i, g in enumerate(out):
            if g in seen:
                dup_pos.append(i)
            else:
                seen.add(g)
        if dup_pos:
            avail = [g for g in range(P_) if g not in seen]
            rng.shuffle(avail)
            for i, g in zip(dup_pos, avail):
                out[i] = g; seen.add(g)
        return out

    def moves_of(chrom):
        assign = realize(chrom)
        return int((assign != prev_assign).sum())

    def repair(chrom):
        out = chrom.copy()
        for _ in range(20):
            m = moves_of(out)
            if m <= B:
                break
            # revert lowest travel-gain hot SKUs back to their own slot
            gain = f * (layout.slot_depot[own] - d_pool[out])
            moved = np.where(out != base)[0]
            if len(moved) == 0:
                break
            worst = moved[np.argsort(gain[moved])[:max(1, len(moved) // 4)]]
            for i in worst:
                out[i] = base[i]           # back to own slot (index i -> pool[i]=own[i])
            out = _injective(out)          # base slots are distinct, but guard anyway
        return out

    def ox(p1, p2):
        # Crossover for INJECTIVE partial maps (K genes from a pool of size P_>K).
        # Keep a random segment from p1; fill remaining positions from p2's genes
        # that are not yet used; if p2 is exhausted (possible since |pool|>K),
        # top up from the unused pool at random. Guarantees a valid injective child.
        a, b = sorted(rng.integers(0, K + 1, 2))
        child = -np.ones(K, dtype=int)
        child[a:b] = p1[a:b]
        used = set(p1[a:b].tolist())
        fill = [g for g in p2 if g not in used]
        it = iter(fill)
        for i in range(K):
            if child[i] < 0:
                nxt = next(it, -1)
                if nxt < 0:  # p2 exhausted; draw from unused pool
                    remaining = list(set(range(P_)) - used)
                    nxt = int(rng.choice(remaining))
                child[i] = nxt
                used.add(nxt)
        return child

    # Seed the population with (a) identity (no-move), (b) the A1 greedy solution
    # projected onto the chromosome (memetic seeding: GA can only improve on greedy,
    # and this is the fair comparison — same budget, same objective), and
    # (c) random injective maps for diversity.
    def chrom_from_assign(a1_assign):
        """Encode a global assignment as a chromosome over the pool where possible.
        Hot SKU i -> its slot in a1_assign if that slot is in the pool, else own slot."""
        pool_index = {int(s): j for j, s in enumerate(pool)}
        ch = base.copy()
        for i in range(K):
            s = int(a1_assign[topK[i]])
            if s in pool_index:
                ch[i] = pool_index[s]
        return _injective(ch)

    def rand_chrom():
        return rng.permutation(P_)[:K]

    a1_seed = a1_rule_based(prev_assign, forecast, layout, B, lam, alpha)
    seeds = [base.copy(), chrom_from_assign(a1_seed)]
    pop_list = seeds + [repair(_injective(rand_chrom())) for _ in range(pop - len(seeds))]
    fit = np.array([fitness(c) for c in pop_list])
    for g in range(gens):
        newP = [pop_list[int(np.argmin(fit))]]     # elitism
        while len(newP) < pop:
            i, j, k, l = rng.integers(0, pop, 4)
            p1 = pop_list[i] if fit[i] < fit[j] else pop_list[j]
            p2 = pop_list[k] if fit[k] < fit[l] else pop_list[l]
            child = ox(p1, p2)
            if rng.random() < 0.4:
                x, y = rng.integers(0, K, 2)
                child[x], child[y] = child[y], child[x]
            if rng.random() < 0.2:                 # reseed a gene to an unused pool slot
                unused = list(set(range(P_)) - set(child.tolist()))
                if unused:
                    child[rng.integers(0, K)] = rng.choice(unused)
            newP.append(repair(_injective(child)))
        pop_list = newP
        fit = np.array([fitness(c) for c in pop_list])
    best = pop_list[int(np.argmin(fit))]
    result = realize(repair(_injective(best)))
    assert len(np.unique(result)) == len(result), "GA produced non-bijection"
    return result


class A3QL:
    """Linear Q-learning over engineered features (reduced-scale stand-in for PPO;
    deviation documented). Action = apply A1 with budget b in {0, B/2, B} and
    lambda-scaling in {0.5x, 1x, 2x} -> 9 discrete actions.
    State features: forecast churn (tau proxy), top-decile forecast share,
    aisle-load Gini of current assignment, normalized remaining budget usage.
    Reward: -(surrogate travel + lam*congestion + mu*moves)."""
    def __init__(self, B, lam, alpha, mu, seed=0):
        self.B, self.lam, self.alpha, self.mu = B, lam, alpha, mu
        self.actions = [(b, s) for b in (0, B // 2, B) for s in (0.5, 1.0, 2.0)]
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((len(self.actions), 5))
        self.lr, self.eps = 0.05, 0.2

    def features(self, prev_assign, forecast, prev_forecast, layout):
        f = forecast / max(forecast.sum(), 1e-9)
        top = np.argsort(-forecast)[:len(forecast) // 10]
        share = f[top].sum()
        churn = 1 - np.corrcoef(forecast, prev_forecast)[0, 1] if prev_forecast is not None else 0.5
        loads = aisle_loads(prev_assign, forecast, layout)
        g = aisle_gini_local(loads)
        d_norm = (layout.slot_depot[prev_assign] * f).sum() / layout.slot_depot.max()
        return np.array([1.0, share, churn, g, d_norm])

    def act(self, phi, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return self.rng.integers(len(self.actions))
        return int(np.argmax(self.W @ phi))

    def step(self, prev_assign, forecast, layout, action):
        b, s = self.actions[action]
        if b == 0: return prev_assign.copy()
        return a1_rule_based(prev_assign, forecast, layout, b, self.lam * s, self.alpha)

    def reward(self, assign, prev_assign, actual, layout):
        travel = expected_travel_surrogate(assign, actual, layout)
        pen = congestion_penalty(assign, actual, layout, self.alpha)
        moves = int((assign != prev_assign).sum())
        return -(travel + self.lam * pen + self.mu * moves)

    def update(self, phi, a, r, scale=1e-7):
        q = self.W[a] @ phi
        self.W[a] += self.lr * (r * scale - q) * phi


def aisle_gini_local(loads):
    x = np.sort(np.asarray(loads, dtype=float))
    if x.sum() == 0: return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
