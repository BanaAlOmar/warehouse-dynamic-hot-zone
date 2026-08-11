"""Three search strategies over the SAME ReslotProblem: A1 greedy, A2 GA, MIP.
All consume prob.realize / prob.budget_repair / prob.objective so the feasible
set, budget accounting, and objective are identical by construction.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, time
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


# ----------------------------------------------------------------------------
# A1 : budgeted greedy over the shared pool
# ----------------------------------------------------------------------------
def solve_a1(prob, max_iter=2000, LEV=1600, TOPK=200, EMPTY=400):
    """Greedy: repeatedly move the decision SKU whose relocation to its best
    available candidate slot yields the largest objective decrease, until the
    budget binds or no positive move remains. Operates on the shared pool T.

    Neighborhood-size knobs (LEV, TOPK, EMPTY) control how much of the shared
    pool is searched each iteration. Defaults search the FULL pool (LEV=1600 =
    all decision SKUs; TOPK=200 candidate swaps; EMPTY=400 candidate slots).
    See Deviation D11: an interim commit clamped these to 400/40/60 "for speed,"
    which silently under-searched and understated A1 by ~0.35pp; the full-pool
    defaults are the correct, well-posed greedy and reproduce the Pass-1 figure
    at ~4.5s/week (deterministic; verified 7/9 weeks bit-identical to the run)."""
    lay = prob.lay
    assign = prob.prev.copy()
    d = lay.slot_depot
    T = prob.T
    occ = {int(s): i for i, s in enumerate(assign)}
    empties = set(range(lay.n_slots)) - set(assign.tolist())
    changed = set()
    f = prob.f

    def slot_gain(sku, tgt):
        return f[sku] * 2 * (d[assign[sku]] - d[tgt])  # travel decrease (lam=0 path)

    for _ in range(max_iter):
        if len(changed) >= prob.B:
            break
        best = (1e-6, None, None, None)   # (gain, sku, tgt, occupant)
        # For each decision SKU consider (a) the nearest empty/prime slots and
        # (b) SWAPS with any other decision SKU (the highest-leverage moves: a hot
        # SKU trading places with a cold SKU sitting closer to the depot). This is
        # the full swap neighborhood the exact greedy needs to close the MIP gap.
        # restrict swap matrix to the highest-leverage SKUs (slot-rank vs forecast-rank
        # mismatch) so the O(m^2) neighborhood is bounded at m~400 without losing useful
        # swaps; identical restriction principle used by the GA's GRASP seeding.
        Sall = prob.S
        srank = np.argsort(np.argsort(d[assign[Sall]]))
        frank = np.argsort(np.argsort(-f[Sall]))
        S = Sall[np.argsort(-np.abs(srank - frank))[:min(LEV, len(Sall))]]
        da = d[assign[S]]
        fS = f[S]
        # swap gain matrix over decision SKUs: 2*(f_i - f_j)*(d_i - d_j)
        G = 2 * (fS[:, None] - fS[None, :]) * (da[:, None] - da[None, :])
        np.fill_diagonal(G, -np.inf)
        _k = min(TOPK, G.size - 1)
        flat = np.argpartition(-G.ravel(), _k)[:_k]
        for idx in flat[np.argsort(-G.ravel()[flat])]:
            g = G.ravel()[idx]
            if g <= best[0]:
                break
            ii, jj = idx // len(S), idx % len(S)
            si, sj = int(S[ii]), int(S[jj])
            would = {s for s in changed}
            would.add(si); would.add(sj)
            if len(would) > prob.B:
                continue
            best = (g, si, int(assign[sj]), sj)   # occupant = sj (a swap)
        # also allow moving a SKU into a genuinely EMPTY prime slot (cost 1)
        for sku in S:
            cur = int(assign[sku])
            for tgt in T[:EMPTY]:
                tgt = int(tgt)
                if occ.get(tgt, -1) != -1 or tgt == cur:
                    continue
                g = slot_gain(sku, tgt)
                if g <= best[0]:
                    break
                would = set(changed); would.add(int(sku))
                if len(would) > prob.B:
                    continue
                best = (g, int(sku), tgt, -1)
        if best[1] is None:
            break
        _, sku, tgt, occupant = best
        src = int(assign[sku])
        if occupant >= 0 and occupant != sku:
            assign[sku], assign[occupant] = tgt, src
            occ[tgt], occ[src] = sku, occupant
            changed.add(sku); changed.add(occupant)
        else:
            assign[sku] = tgt
            occ[tgt] = sku; occ.pop(src, None)
            empties.discard(tgt); empties.add(src)
            changed.add(sku)
        # de-register reverted SKUs
        changed = {s for s in changed if assign[s] != prob.prev[s]}
    assign = prob.budget_repair(assign)
    return assign


# ----------------------------------------------------------------------------
# Exact local action benchmark with TRUE swap-accounted budget
# ----------------------------------------------------------------------------
def solve_exact_actions(prob, partners_per_sku=30, empty_per_sku=2,
                        time_limit=180, mip_rel_gap=0.0):
    """Solve a broad, action-specific local neighborhood exactly.

    The benchmark considers every decision SKU in ``prob.S`` and retains its
    ``partners_per_sku`` highest-gain pairwise swaps, deduplicated across SKUs.
    It also considers the best moves into genuinely empty candidate slots.
    Binary action variables are mutually exclusive by SKU/target slot, and the
    budget charges 2 for an occupied-slot swap and 1 for an empty-slot move.

    This is an exact maximum-gain selection over the declared action set, not a
    relaxation: the returned assignment is a bijection and its realized move
    count is constrained directly by the same B used by A1.  It is deliberately
    described as an "exact local action benchmark" because partner pruning keeps
    the MIP tractable; it is not a claim of global optimality over all possible
    warehouse assignments.
    """
    if prob.lam != 0:
        raise ValueError("exact action additivity currently requires lambda=0")

    d, f, prev = prob.lay.slot_depot, prob.f, prob.prev
    S = np.asarray(prob.S, dtype=int)
    m = len(S)
    ds, fs = d[prev[S]], f[S]
    swaps = {}

    # Top action-specific partners for every decision SKU.  Building one row at
    # a time avoids materializing a dense m x m matrix.
    keep = min(max(int(partners_per_sku), 1), max(m - 1, 1))
    for ii in range(m):
        gains = 2.0 * (fs[ii] - fs) * (ds[ii] - ds)
        gains[ii] = -np.inf
        jj_keep = np.argpartition(-gains, keep - 1)[:keep]
        for jj in jj_keep:
            g = float(gains[jj])
            if g <= 1e-9:
                continue
            a, b = sorted((int(S[ii]), int(S[jj])))
            old = swaps.get((a, b))
            if old is None or g > old:
                swaps[(a, b)] = g

    actions = [("swap", a, b, 2, g) for (a, b), g in swaps.items()]

    # Empty-target actions.  Their exact gain is additive as long as each SKU
    # and each empty target is selected at most once, enforced below.
    empty = np.asarray([int(s) for s in prob.T if prob.occ0[int(s)] < 0], dtype=int)
    if len(empty):
        empty = empty[np.argsort(d[empty])]
        ekeep = min(max(int(empty_per_sku), 1), len(empty))
        for sku in S:
            gains = 2.0 * f[sku] * (d[prev[sku]] - d[empty])
            top = np.argpartition(-gains, ekeep - 1)[:ekeep]
            for jj in top:
                g = float(gains[jj])
                if g > 1e-9:
                    actions.append(("empty", int(sku), int(empty[jj]), 1, g))

    if not actions:
        return prob.prev.copy(), {
            "status": "Optimal", "success": True, "objective_gain": 0.0,
            "mip_gap": 0.0, "solve_s": 0.0, "n_actions": 0,
            "n_swaps": 0, "n_empty_moves": 0,
        }

    n = len(actions)
    c = -np.asarray([a[4] for a in actions], dtype=float)
    costs = np.asarray([a[3] for a in actions], dtype=float)

    # Rows: one per participating SKU, one per empty target, then the budget.
    sku_rows = {int(s): i for i, s in enumerate(S)}
    empty_targets = sorted({a[2] for a in actions if a[0] == "empty"})
    empty_rows = {int(s): len(sku_rows) + i for i, s in enumerate(empty_targets)}
    budget_row = len(sku_rows) + len(empty_rows)
    rr, cc, vv = [], [], []
    for j, action in enumerate(actions):
        kind, a, b, _, _ = action
        rr.append(sku_rows[a]); cc.append(j); vv.append(1.0)
        if kind == "swap":
            rr.append(sku_rows[b]); cc.append(j); vv.append(1.0)
        else:
            rr.append(empty_rows[b]); cc.append(j); vv.append(1.0)
        rr.append(budget_row); cc.append(j); vv.append(costs[j])
    A = coo_matrix((vv, (rr, cc)),
                   shape=(budget_row + 1, n)).tocsr()
    ub = np.ones(budget_row + 1)
    ub[budget_row] = prob.B
    constraint = LinearConstraint(A, np.zeros_like(ub), ub)

    t0 = time.time()
    res = milp(c=c, integrality=np.ones(n), bounds=Bounds(0, 1),
               constraints=constraint,
               options={"time_limit": float(time_limit),
                        "mip_rel_gap": float(mip_rel_gap),
                        "presolve": True})
    solve_s = time.time() - t0
    if res.x is None:
        raise RuntimeError(f"exact-action MIP failed: {res.message}")

    chosen = np.where(res.x > 0.5)[0]
    assign = prev.copy()
    n_swaps = n_empty = 0
    for j in chosen:
        kind, a, b, _, _ = actions[int(j)]
        if kind == "swap":
            assign[a], assign[b] = assign[b], assign[a]
            n_swaps += 1
        else:
            assign[a] = b
            n_empty += 1

    moves = int((assign != prev).sum())
    assert moves <= prob.B
    assert len(np.unique(assign)) == len(assign)
    gain = float(prob.objective(prev) - prob.objective(assign))
    return assign, {
        "status": str(res.message), "success": bool(res.success),
        "objective_gain": gain,
        "mip_gap": float(getattr(res, "mip_gap", np.nan)),
        "mip_node_count": int(getattr(res, "mip_node_count", -1)),
        "solve_s": solve_s, "n_actions": n,
        "n_swaps": n_swaps, "n_empty_moves": n_empty, "moves": moves,
        "partners_per_sku": int(partners_per_sku),
        "empty_per_sku": int(empty_per_sku),
    }


def solve_exact_assignment(prob, warm_assign=None, partners_per_sku=30,
                           time_limit=300, mip_rel_gap=0.0):
    """Exact sparse assignment MIP over all decision SKUs and their own slots.

    In contrast to :func:`solve_exact_actions`, this formulation permits cycles
    longer than two.  The sparse arc set is built from each SKU's strongest
    pairwise partners, with reciprocal arcs, plus every identity arc.  If a
    deployed A1 solution is supplied as ``warm_assign``, all of its arcs are
    included, so the exact optimum is guaranteed to be no worse than A1 on the
    shared starting state and objective.

    The budget is ``sum(x[i,j] for i != j) <= B``.  Because candidate slots are
    exactly the decision SKUs' current slots, this is the realized number of
    relocated SKUs under the final permutation—there is no per-SKU proxy and no
    uncharged eviction.
    """
    if prob.lam != 0:
        raise ValueError("exact sparse assignment currently requires lambda=0")
    S = np.asarray(prob.S, dtype=int)
    m = len(S)
    prev_slots = prob.prev[S]
    d, f = prob.lay.slot_depot, prob.f
    ds, fs = d[prev_slots], f[S]
    keep = min(max(int(partners_per_sku), 1), max(m - 1, 1))

    arcs = {(i, i) for i in range(m)}
    for i in range(m):
        gains = 2.0 * (fs[i] - fs) * (ds[i] - ds)
        gains[i] = -np.inf
        js = np.argpartition(-gains, keep - 1)[:keep]
        for j in js:
            if gains[j] > 1e-9:
                arcs.add((i, int(j)))
                arcs.add((int(j), i))

    # Explicitly include the deployed A1 permutation, making it a feasible MIP
    # incumbent/certificate comparator even if an arc falls outside the pruning.
    if warm_assign is not None:
        slot_to_j = {int(s): j for j, s in enumerate(prev_slots)}
        for i, sku in enumerate(S):
            target = int(warm_assign[sku])
            if target not in slot_to_j:
                raise ValueError("warm assignment leaves the S-own-slot permutation")
            arcs.add((i, slot_to_j[target]))

    arc_list = sorted(arcs)
    n = len(arc_list)
    base = 2.0 * fs * ds
    c = np.asarray([
        2.0 * fs[i] * ds[j] - base[i] for i, j in arc_list
    ], dtype=float)

    # SKU equalities, slot equalities, and the exact realized-move budget.
    rr, cc, vv = [], [], []
    for k, (i, j) in enumerate(arc_list):
        rr.extend((i, m + j, 2 * m))
        cc.extend((k, k, k))
        vv.extend((1.0, 1.0, float(i != j)))
    A = coo_matrix((vv, (rr, cc)), shape=(2 * m + 1, n)).tocsr()
    lb = np.ones(2 * m + 1)
    ub = np.ones(2 * m + 1)
    lb[-1] = 0.0
    ub[-1] = prob.B

    t0 = time.time()
    res = milp(
        c=c, integrality=np.ones(n), bounds=Bounds(0, 1),
        constraints=LinearConstraint(A, lb, ub),
        options={"time_limit": float(time_limit),
                 "mip_rel_gap": float(mip_rel_gap), "presolve": True},
    )
    solve_s = time.time() - t0
    if res.x is None:
        raise RuntimeError(f"exact assignment MIP failed: {res.message}")

    chosen = np.where(res.x > 0.5)[0]
    target_j = np.full(m, -1, dtype=int)
    for k in chosen:
        i, j = arc_list[int(k)]
        target_j[i] = j
    if np.any(target_j < 0):
        raise RuntimeError("MIP returned incomplete assignment")
    assign = prob.prev.copy()
    assign[S] = prev_slots[target_j]
    moves = int((assign != prob.prev).sum())
    assert moves <= prob.B and len(np.unique(assign)) == len(assign)
    gain = float(prob.objective(prob.prev) - prob.objective(assign))
    warm_gain = np.nan
    if warm_assign is not None:
        warm_gain = float(prob.objective(prob.prev) - prob.objective(warm_assign))
        assert gain + 1e-5 >= warm_gain
    return assign, {
        "status": str(res.message), "success": bool(res.success),
        "objective_gain": gain, "warm_a1_gain": warm_gain,
        "gain_over_warm": gain - warm_gain,
        "gain_over_warm_pct": (
            100.0 * (gain - warm_gain) / warm_gain
            if np.isfinite(warm_gain) and warm_gain != 0 else np.nan
        ),
        # This is the optimizer's relative bound gap, not a gap to A1.
        "solver_mip_gap": float(getattr(res, "mip_gap", np.nan)),
        "mip_gap": float(getattr(res, "mip_gap", np.nan)),
        "mip_node_count": int(getattr(res, "mip_node_count", -1)),
        "solve_s": solve_s, "n_arcs": n, "n_sku": m,
        "moves": moves, "partners_per_sku": int(partners_per_sku),
        "candidate_scope": (
            "all ReslotProblem decision SKUs; current-SKU-slot permutation; "
            "top-partner reciprocal arcs plus all A1 incumbent arcs"
        ),
        "full_T_candidate_pool": False,
        "old_head_only": False,
        "a1_incumbent_feasible": bool(warm_assign is not None),
        "replay_scope": "paired same-state weekly diagnostic",
    }


# ----------------------------------------------------------------------------
# A2 : GA with DIRECT ASSIGNMENT encoding (gene i = slot for decision SKU i)
# ----------------------------------------------------------------------------
def solve_a2(prob, pop=40, gens=60, seed=0, warm=None, log_curve=False):
    """Direct-assignment GA over the shared pool.
    Chromosome: length-nS vector, gene i = index into prob.T for decision SKU i.
    Uniform crossover + slot-conflict repair + shared budget_repair.
    One seed is warm-started from `warm` (A1 solution) if provided.
    Returns (assign, curve) where curve = best objective per generation."""
    rng = np.random.default_rng(seed)
    T = prob.T
    nT, nS = prob.nT, prob.nS
    own_T = prob.own_T                      # each SKU's own slot as a T-index
    d_T = prob.d_T

    def repair_conflicts(ch):
        """Ensure genes map SKUs to DISTINCT T-slots (assignment feasibility)."""
        seen = {}
        dup = []
        for i, g in enumerate(ch):
            if g in seen:
                dup.append(i)
            else:
                seen[g] = i
        if dup:
            free = [g for g in range(nT) if g not in seen]
            rng.shuffle(free)
            for i, g in zip(dup, free):
                ch[i] = g; seen[g] = i
        return ch

    def to_assign(ch):
        return prob.budget_repair(prob.realize(T[ch]))

    def fitness(ch):
        return prob.objective(to_assign(ch))

    # init population.
    # Encoding note: because a random assignment over ~1900 slots is dominated by the
    # move-budget constraint (budget_repair would revert almost everything to prev,
    # collapsing to baseline), random seeds are initialized as *local perturbations of
    # the identity*: start everyone in their own slot and apply a random set of feasible
    # swaps among decision SKUs. This explores the budgeted neighborhood rather than the
    # infeasible global assignment space, which is the correct search space here.
    base = own_T.copy()                     # identity: everyone in own slot
    pop_list = [base.copy()]
    if warm is not None:
        # encode warm (A1) assignment: decision SKU -> its slot's T-index if in T
        wch = own_T.copy()
        for i, sku in enumerate(prob.S):
            s = int(warm[sku])
            if s in prob.T_index:
                wch[i] = prob.T_index[s]
        pop_list.append(repair_conflicts(wch))

    def grasp_seed(alpha_rcl):
        """GRASP-style randomized-greedy construction: build a budgeted swap sequence
        by, at each step, picking uniformly at random from the top-`alpha_rcl` fraction
        of the best-gain candidate swaps (restricted candidate list). Different RCL
        draws yield genuinely different strong local optima -> real population diversity.
        Returns a chromosome (T-indices per decision SKU)."""
        d = prob.lay.slot_depot
        assign = prob.prev.copy()
        # restrict the swap search to the highest-leverage SKUs: those whose slot
        # closeness rank and forecast rank disagree most (the drift/mismatch region).
        # This bounds the O(m^2) gain matrix to m~300 without losing the useful swaps.
        Sall = prob.S
        srank = np.argsort(np.argsort(d[assign[Sall]]))
        frank = np.argsort(np.argsort(-prob.f[Sall]))
        lever = np.abs(srank - frank)
        S = Sall[np.argsort(-lever)[:min(300, len(Sall))]]
        changed = set()
        for _ in range(prob.B):
            da = d[assign[S]]; fS = prob.f[S]
            G = 2 * (fS[:, None] - fS[None, :]) * (da[:, None] - da[None, :])
            np.fill_diagonal(G, -np.inf)
            k = 30
            flat = np.argpartition(-G.ravel(), k)[:k]
            flat = flat[G.ravel()[flat] > 0]
            if len(flat) == 0:
                break
            rcl = flat[np.argsort(-G.ravel()[flat])]
            rcl = rcl[:max(1, int(len(rcl) * alpha_rcl))]
            pick = int(rng.choice(rcl))
            ii, jj = pick // len(S), pick % len(S)
            si, sj = int(S[ii]), int(S[jj])
            would = set(changed); would.add(si); would.add(sj)
            if len(would) > prob.B:
                break
            assign[si], assign[sj] = assign[sj], assign[si]
            changed = {s for s in would if assign[s] != prob.prev[s]}
        # encode assignment -> chromosome (index over the FULL pool prob.S, not the
        # leverage-restricted subset S, so gene positions line up with the encoding)
        ch = own_T.copy()
        for i, sku in enumerate(prob.S):
            s = int(assign[sku])
            if s in prob.T_index:
                ch[i] = prob.T_index[s]
        return ch

    while len(pop_list) < pop:
        pop_list.append(repair_conflicts(grasp_seed(alpha_rcl=0.5)))
    fit = np.array([fitness(c) for c in pop_list])
    curve = [float(fit.min())]

    for g in range(gens):
        newP = [pop_list[int(np.argmin(fit))]]       # elitism
        while len(newP) < pop:
            i, j, k, l = rng.integers(0, pop, 4)
            p1 = pop_list[i] if fit[i] < fit[j] else pop_list[j]
            p2 = pop_list[k] if fit[k] < fit[l] else pop_list[l]
            mask = rng.random(nS) < 0.5              # uniform crossover
            child = np.where(mask, p1, p2).astype(int)
            if rng.random() < 0.4:                   # mutation: swap two genes (feasible)
                a, b = rng.integers(0, nS, 2)
                child[a], child[b] = child[b], child[a]
            newP.append(repair_conflicts(child))
        pop_list = newP
        fit = np.array([fitness(c) for c in pop_list])
        curve.append(float(fit.min()))
    best = pop_list[int(np.argmin(fit))]
    assign = to_assign(best)
    return (assign, curve) if log_curve else assign


# ----------------------------------------------------------------------------
# MIP : exact assignment benchmark on a SMALL instance over the shared pool
# ----------------------------------------------------------------------------
def solve_mip(prob, n_sku=200, time_limit=120, verbose=False):
    """Exact/near-exact benchmark: restrict to the top-`n_sku` decision SKUs and
    their candidate slots (from the shared pool T), solve the budgeted assignment
    as a MIP. Objective = line-weighted travel surrogate (lam=0 path, matching the
    calibrated setting). Returns (assign_partial_objective, bound, gap, status).

    Formulation (assignment with move budget):
      x[i,s] = 1 if SKU i placed in slot s (i in decision subset, s in T)
      min  sum_i sum_s f_i * 2 * d_s * x[i,s]
      s.t. sum_s x[i,s] = 1                          (each SKU one slot)
           sum_i x[i,s] <= 1                          (each slot one SKU)
           sum_i (1 - x[i, own_i]) <= B_sub           (move budget, per-SKU proxy)
    B_sub scales B by the subset's share of decision SKUs. This is a RELAXATION of
    the true swap-accounted budget (it charges 1 per moved SKU, not evictions), so
    its optimum is a LOWER BOUND on achievable travel => a valid optimality-gap
    reference for the heuristics restricted to the same subset.
    """
    try:
        import pulp
    except Exception as e:
        return None
    d = prob.lay.slot_depot
    f = prob.f
    # top n_sku decision SKUs by forecast
    Ssub = prob.S[np.argsort(-f[prob.S])[:n_sku]]
    # candidate slots = union of their own slots + nearest prime slots in T
    own = prob.prev[Ssub]
    cand = np.unique(np.concatenate([own, prob.T[:max(n_sku * 2, 400)]]))
    ci = {int(s): j for j, s in enumerate(cand)}
    B_sub = int(round(prob.B * n_sku / max(prob.nS, 1)))

    solver = None
    try:
        import gurobipy  # noqa
        solver = pulp.GUROBI_CMD(msg=verbose, timeLimit=time_limit)
        _ = solver.available()
        if not solver.available():
            solver = None
    except Exception:
        solver = None
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)

    prob_lp = pulp.LpProblem("reslot", pulp.LpMinimize)
    x = {}
    for i, sku in enumerate(Ssub):
        for s in cand:
            x[(i, int(s))] = pulp.LpVariable(f"x_{i}_{int(s)}", cat="Binary")
    # objective
    prob_lp += pulp.lpSum(f[sku] * 2 * d[s] * x[(i, int(s))]
                          for i, sku in enumerate(Ssub) for s in cand)
    # each SKU exactly one slot
    for i, sku in enumerate(Ssub):
        prob_lp += pulp.lpSum(x[(i, int(s))] for s in cand) == 1
    # each slot at most one SKU
    for s in cand:
        prob_lp += pulp.lpSum(x[(i, int(s))] for i in range(len(Ssub))) <= 1
    # move budget (per-SKU proxy): SKUs that leave their own slot
    prob_lp += pulp.lpSum(1 - x[(i, int(prob.prev[sku]))]
                          for i, sku in enumerate(Ssub)) <= B_sub
    t = time.time()
    prob_lp.solve(solver)
    solve_s = time.time() - t
    status = pulp.LpStatus[prob_lp.status]
    obj = pulp.value(prob_lp.objective)
    # bound: for CBC, best bound not always exposed; approximate with obj if optimal
    try:
        bound = prob_lp.solverModel.getObjBound() if hasattr(prob_lp, "solverModel") else obj
    except Exception:
        bound = obj
    # heuristic objectives on the SAME subset (travel surrogate over Ssub only)
    def sub_travel(assign):
        return float((f[Ssub] * 2 * d[assign[Ssub]]).sum())
    return {"status": status, "mip_obj_subset": obj, "bound": bound,
            "solve_s": solve_s, "B_sub": B_sub, "n_sku": n_sku,
            "Ssub": Ssub, "sub_travel": sub_travel}
