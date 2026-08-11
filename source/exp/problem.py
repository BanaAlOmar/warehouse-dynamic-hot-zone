"""SHARED re-slotting problem definition (Pass-1 methodological fix).

A single candidate pool and a single budget-accounting convention are defined
here and consumed IDENTICALLY by A1 (greedy), A2 (GA), and the MIP benchmark, so
their comparison is "three search strategies on ONE problem" rather than three
different problems. This is the well-posedness fix demanded in review.

------------------------------------------------------------------------------
FEASIBLE SET
------------------------------------------------------------------------------
Decision SKUs  S = union of:
    (a) the top-K forecast SKUs, and
    (b) the SKUs currently occupying the K nearest-to-depot ("prime") slots.
Reason for (b): on an ABC warm start the hot SKUs are ALREADY near the depot, so
the achievable gain lies in evicting COLD SKUs out of prime slots. A hot-only pool
(Pass-0) structurally cannot express that and is retained only as an ablation.

Candidate slots  T = union of:
    (a) the current slots of the decision SKUs, and
    (b) the K nearest-to-depot prime slots overall (occupied or empty), and
    (c) all genuinely EMPTY slots (n_slots - n_active), which cost less to fill.
Every algorithm may place any decision SKU into any candidate slot in T.

------------------------------------------------------------------------------
BUDGET ACCOUNTING (identical for A1 / A2 / MIP)
------------------------------------------------------------------------------
Budget B counts SLOT-CHANGES (physical relocations), the operationally metered
quantity. A relocation of one SKU from slot p to slot q costs:
    - 1 slot-change  if q is EMPTY (the SKU simply moves in);
    - 2 slot-changes if q is OCCUPIED (the occupant must be evicted to a free
      slot; that eviction is itself a relocation). We realize eviction as a SWAP
      when the occupant is also a decision SKU (net 2 slot-changes = 1 swap), and
      as a push-to-nearest-free-slot otherwise (also 2 slot-changes).
Thus: moves(assignment) = #SKUs whose slot changed vs the previous assignment.
This is exactly (new_assign != prev_assign).sum(), so the SAME move count is used
everywhere; the empty-vs-occupied distinction is captured automatically because
filling an empty slot changes 1 SKU's slot while a swap changes 2.

------------------------------------------------------------------------------
OBJECTIVE (identical everywhere)
------------------------------------------------------------------------------
J(assign) = sum_sku forecast[sku] * 2 * dist_to_depot[assign[sku]]
            + lam * congestion_penalty(assign, forecast, alpha)
The line-weighted round-trip travel surrogate + optional congestion term. Final
reported metric is always the DES/route replay, not J (J is the search fitness).
"""
import numpy as np


class ReslotProblem:
    def __init__(self, prev_assign, forecast, layout, B, lam, alpha, K=400, seed=0):
        self.prev = prev_assign.copy()
        self.f = forecast.astype(float)
        self.lay = layout
        self.B, self.lam, self.alpha, self.K = B, lam, alpha, K
        self.n = len(prev_assign)
        d = layout.slot_depot

        # ---- decision SKUs S ----
        prime_slots = np.argsort(d)[:K]
        occ = np.full(layout.n_slots, -1, dtype=int)
        occ[prev_assign] = np.arange(self.n)          # SKU occupying each slot (-1 = empty)
        self.occ0 = occ
        cold_in_prime = occ[prime_slots]
        cold_in_prime = cold_in_prime[cold_in_prime >= 0]
        hot = np.argsort(-self.f)[:K]
        S = np.unique(np.concatenate([hot, cold_in_prime]))
        self.S = S
        self.nS = len(S)

        # ---- candidate slots T ----
        own = prev_assign[S]
        empty_slots = np.setdiff1d(np.arange(layout.n_slots), prev_assign)  # genuinely empty
        T = np.unique(np.concatenate([own, prime_slots, empty_slots]))
        # order T by depot distance (prime first) for interpretable indexing
        self.T = T[np.argsort(d[T])]
        self.nT = len(self.T)
        self.d_T = d[self.T]
        self.is_empty_T = (occ[self.T] < 0)
        self.aisle_T = layout.slot_a[self.T] + layout.n_aisles * layout.slot_b[self.T]
        self.n_ax = layout.n_aisles * layout.n_blocks
        self.f_S = self.f[S]

        # convenience maps
        self.T_index = {int(s): j for j, s in enumerate(self.T)}
        self.own_T = np.array([self.T_index[int(s)] for s in own])  # each S SKU's own slot index in T

        # congestion cap uses full-assignment mean load (constant across search)
        loads0 = np.zeros(self.n_ax)
        np.add.at(loads0, layout.slot_a[prev_assign] + layout.n_aisles * layout.slot_b[prev_assign], self.f)
        self.cap = (1 + alpha) * loads0.mean()
        self.pen_scale = 2 * float(np.median(d))

    # ---- realize a decision (S-SKU -> slot in T) into a global bijection ----
    def realize(self, slot_of_S):
        """slot_of_S: length-nS array of ABSOLUTE slot ids (distinct) drawn from T.
        Returns a valid global assignment (bijection over the original slot set),
        realizing evictions as pushes to freed slots. Move count = SKUs changed."""
        assign = self.prev.copy()
        new = np.asarray(slot_of_S)
        assert len(np.unique(new)) == len(new), "decision slots not distinct"
        Sset = set(self.S.tolist())
        new_set = set(new.tolist())
        own = self.prev[self.S]
        own_set = set(own.tolist())
        # SKUs displaced = current occupants of target slots that are NOT decision SKUs
        occupants = self.occ0[new]
        displaced = [int(o) for o in occupants if o >= 0 and o not in Sset]
        # freed slots = decision SKUs' own slots no longer occupied by any decision SKU
        freed = list(own_set - new_set)
        # plus any empty target slots consumed do not free anything; conservation:
        # |displaced| == |freed| because #slots held by S before == #held after (== nS
        # only if no empty slots used). When empty slots are used, some decision SKUs
        # occupy previously-empty slots, freeing MORE own-slots than displaced count.
        # Handle generally: assign displaced SKUs to freed slots, leftover freed stay empty.
        assign[self.S] = new
        for sku, slot in zip(displaced, freed):
            assign[sku] = slot
        # any remaining displaced (shouldn't happen) -> leftover empties
        if len(displaced) > len(freed):
            leftover = [int(o) for o in displaced[len(freed):]]
            remaining_empty = list(set(range(self.lay.n_slots)) - set(assign.tolist()))
            for sku, slot in zip(leftover, remaining_empty):
                assign[sku] = slot
        assert len(np.unique(assign)) == len(assign), "realize produced non-bijection"
        return assign

    def moves(self, assign):
        return int((assign != self.prev).sum())

    def objective(self, assign):
        travel = float((self.f * 2 * self.lay.slot_depot[assign]).sum())
        if self.lam == 0:
            return travel
        loads = np.zeros(self.n_ax)
        ax = self.lay.slot_a[assign] + self.lay.n_aisles * self.lay.slot_b[assign]
        np.add.at(loads, ax, self.f)
        pen = np.maximum(loads - self.cap, 0).sum() * self.pen_scale
        return travel + self.lam * pen

    def budget_repair(self, assign):
        """Keep only the B highest-gain SKU relocations; revert the rest to prev.
        Rebuilt in one pass to guarantee a bijection (no iterative decollide).
        A relocation's 'gain' = f * (dist_prev - dist_new); reverting a low-gain
        SKU restores prev[sku], which is always free once we rebuild from prev."""
        d = self.lay.slot_depot
        moved = np.where(assign != self.prev)[0]
        if len(moved) <= self.B:
            return assign.copy()
        # Greedily accept relocations in descending travel-gain, rebuilding from
        # prev and charging the TRUE slot-change cost of each acceptance (1 if the
        # target is currently free in the working assignment, 2 if it evicts an
        # occupant). Stop when the next acceptance would exceed B slot-changes.
        gain = self.f[moved] * (d[self.prev[moved]] - d[assign[moved]])
        order = moved[np.argsort(-gain)]
        out = self.prev.copy()
        occ = {int(s): i for i, s in enumerate(out)}   # slot -> sku currently there
        empties = set(range(self.lay.n_slots)) - set(out.tolist())
        for sku in order:
            tgt = int(assign[sku])
            if out[sku] == tgt:
                continue
            src = int(out[sku])
            occupant = occ.get(tgt, -1)
            # simulate the op and accept only if it keeps total slot-changes <= B
            trial = out.copy()
            if occupant >= 0 and occupant != sku:
                trial[sku], trial[occupant] = tgt, src
            else:
                trial[sku] = tgt
            if int((trial != self.prev).sum()) > self.B:
                continue
            # commit
            if occupant >= 0 and occupant != sku:
                out[sku], out[occupant] = tgt, src
                occ[tgt], occ[src] = sku, occupant
            else:
                out[sku] = tgt
                occ[tgt] = sku
                occ.pop(src, None)
                empties.discard(tgt); empties.add(src)
        assert len(np.unique(out)) == len(out), "budget_repair non-bijection"
        assert int((out != self.prev).sum()) <= self.B
        return out
