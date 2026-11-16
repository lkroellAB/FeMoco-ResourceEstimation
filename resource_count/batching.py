"""
Lookup table batching options for RPrep + Rot.

Every method loads the rotation angles in batches of `lam` angles (output register lam*b_rot)
using a lookup table over Y = N + 2**n_R B addresses with swap parameter K.  Methods
differ only in where the swap ancilla come from and in how many lookups are needed:

    method             swap ancilla from        lookups
    'qrom'             none (K = 1)             J
    'clean'            clean budget             J
    'dirty_restricted' spin-up system           J + 1
    'dirty_full'       all system qubits        J + runs

with J = ceil((N-1)/lam).  The dirty variants pay one extra lookup (a full
batched SelectSwap with its Hadamard conjugation) per recycling run;
restricted access needs a single run, full access needs `runs` of them
because the rotations act on qubits that are simultaneously being borrowed.

Viability
---------
An option is viable when all four hold; a row lists the ones that fail
under 'rejected':

  'no system Z access'  a dirty method borrows system qubits, which needs Z
                        access to them (model.SYSTEM_Z_ACCESS).  The rep-code
                        layout has no access corridors for the system, so the
                        dirty methods are out there; in LDPC they are in.
  'swap source'         the borrowed qubits fit their source: (K-1)*lam*b_rot
                        <= N for restricted access, at least one batch per
                        recycling run for full access (full_access_runs).
                        For 'clean' the source is the compute zone, checked
                        by 'compute zone' below.
  'adder layout'        lam >= 2, unless 'clean' with K >= 2 (both codes).
  'compute zone'        clean ancilla + access corridors <= the budget.  The
                        output register (all K copies for 'clean') needs Z
                        access, so its corridors (model.access_corridors; none
                        in LDPC) sit in the compute zone beside it.  The flag
                        qubit has its own zone and is not part of it.

'System corridors' reports the corridors a dirty method would need under the
system qubits -- only the borrowed ones for restricted access, all of them
for full access, which changes which qubits it borrows from run to run.  They
are not in the compute zone; they show why the dirty methods are out wherever
the layout has none.

Scope of the costs
------------------
The Givens rotations themselves are NOT included as their cost only depends on lambda.
That is why this module's cost fields are named 'lookup Toffoli' and 'lookup rounds'
rather than 'Toffoli cost'/'Temporal cost': they cover the lookup table only, so
they are NOT comparable across different lam (the omitted per-batch work grows
with lam).  To rank whole options, cost them with high_level_cost.RPrep_Rot --
which is what optimize.search_rprep_rot does.

An option is a (method, K, lam, runs) tuple.  `batching_cost` turns one into a
row; high_level_cost.RPrep_Rot takes the same four values and applies that exact
option, reading the lookup count and the SelectSwap record straight off this
module so the two can never disagree about what a method means.  To pass a
chosen row on:  RPrep_Rot(..., **as_option(row)).
"""
import numpy as np
from resource_count import accounting as acc
from resource_count import layout

METHODS = ('qrom', 'clean', 'dirty_restricted', 'dirty_full')


def swap_budget(method, N, clean_budget=None, full_budget=None):
    '''Ancilla available to the swap network, i.e. the bound on (K-1)*lam*b_rot.'''
    if method == 'qrom':
        return 0
    if method == 'clean':
        return clean_budget if clean_budget is not None else np.inf
    if method == 'dirty_restricted':
        return N                      # spin-up system qubits
    if method == 'dirty_full':
        return full_budget if full_budget is not None else 2*N
    raise ValueError(f"unknown method {method!r}")


def full_access_runs(N, b_rot, K, lam, system=None):
    """
    Number of recycling runs for dirty SelectSwap with full access.

    Holding lam*b_rot output qubits with a K-wide swap network borrows
    (K-1)*b_rot*lam system qubits.  After J batches have been applied, J*lam
    system qubits carry rotations and can no longer be borrowed, so

        (K-1)*b_rot*lam = system - J*lam   =>   J = (system - (K-1)*b_rot*lam)/lam

    batches fit in one run.  With J' = ceil((N-1)/lam) batches needed overall,
    that is floor(J'/J) full runs plus a partial one for the remainder.

    Returns None when no batch fits, i.e. the option is infeasible.
    """
    if system is None:
        system = 2*N
    per_run = int(np.floor((system - (K - 1)*b_rot*lam)/lam))
    if per_run <= 0:
        return None
    needed = int(np.ceil((N - 1)/lam))
    full, rem = divmod(needed, per_run)
    return {'runs': full + (1 if rem else 0), 'batches per run': per_run,
            'batches needed': needed}


def batching_cost(N, B, R, b_rot, distances, method, lam, K=1, runs=None,
                  clean_budget=None, full_budget=None, *, model):
    '''
    Cost of one RPrep lookup table batching option, with the primitives of
    the cost model `model` (required, see high_level_cost.MODEL_INTERFACE).

    runs:          number of recycling runs.  Only used by 'dirty_full'; the
                   other methods fix it (1 for dirty_restricted, 0 for the rest).
    clean_budget:  the compute zone's size, in logical qubits: clean ancilla
                   plus access corridors (see the module docstring).  None
                   means unbounded.

    The returned row carries 'load', the SelectSwap record for one lookup, so
    that a caller costing the full routine (high_level_cost.RPrep_Rot) uses the
    very same record rather than rebuilding it from method/K/lam by hand.
    '''
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    n_r = int(np.ceil(np.log2(R)))
    Y = N + 2**n_r*B
    alpha = int(np.log2(K))
    if 2**alpha != K:
        raise ValueError(f"K={K} is not a power of two")

    # 'qrom' and K=1 are two names for the same thing -- a plain skew-tree
    # QROM with no swap network -- so each implies the other.  Both directions
    # are needed: without the second, method='qrom' with K>1 would silently
    # build a K-wide swap network, label it 'qrom', and then be reported
    # infeasible because 'qrom' has no swap ancilla budget.
    if K == 1:
        method = 'qrom'
    elif method == 'qrom':
        K, alpha = 1, 0

    variety = 'clean' if method in ('qrom', 'clean') else 'dirty'
    load = model.SelectSwap_cost(Y, lam*b_rot, alpha, **distances, variety=variety, batched=True)

    # the output register needs Z access: its corridors sit in the compute
    # zone beside it, for all K copies with 'clean'
    register = K*lam*b_rot if method == 'clean' else lam*b_rot
    corridors = model.access_corridors(register)
    compute_zone = load['Clean ancilla'] + corridors
    # the corridors a dirty method needs under the system qubits it borrows:
    # only those for restricted access, all of them for full access
    if method == 'dirty_restricted':
        system_corridors = model.access_corridors((K - 1)*lam*b_rot)
    elif method == 'dirty_full':
        system_corridors = model.access_corridors(layout.system_zone_size(N))
    else:
        system_corridors = 0

    J = int(np.ceil((N - 1)/lam))
    per_run, runs_ok = None, True
    if method == 'dirty_full' and runs is None:
        split = full_access_runs(N, b_rot, K, lam, full_budget)
        if split is None:
            runs, per_run, runs_ok = 0, 0, False
        else:
            runs, per_run = split['runs'], split['batches per run']
            # dirty full access: one extra SelectSwap per recycling run,
            # runs = ceil(J / per_run) for the J batches (full_access_runs)
    extra = {'qrom': 0, 'clean': 0, 'dirty_restricted': 1, 'dirty_full': runs}[method]
    lookups = J + extra

    # feasibility
    needed = (K - 1)*lam*b_rot
    budget = swap_budget(method, N, clean_budget, full_budget)
    swap_ok = needed <= budget and runs_ok
    system_access = variety == 'clean' or model.SYSTEM_Z_ACCESS
    fits = clean_budget is None or compute_zone <= clean_budget
    # Adder layout.  The phase gradient adder needs from the layout:
    #  pool:   enough free slots anywhere to move additional rotation angles away
    #  row:    enough slots in one row to hold the output set, the Fourier
    #          resource state and one ancilla set (3*b_rot total).  The output
    #          set and Fourier state are already placed, so one set remains.
    #  offset: the adder needs to be offset, fulfilled by the layout
    # Only the rule below is checked.  The number of free slots, in a row or
    # in total, is not modelled, so 'pool' and 'row' are assumed to hold.
    if (method == 'clean' and K >= 2) or lam >=2:
        adder_ok = True
    else:
        adder_ok = False

    # for 'clean' the swap source is the compute zone, so a clean option that
    # does not fit is reported once, as 'compute zone'
    rejected = tuple(reason for reason, ok in (
        ('no system Z access', system_access),
        ('swap source', swap_ok or method == 'clean'),
        ('adder layout', adder_ok),
        ('compute zone', fits)) if not ok)

    return {'method': method, 'K': K, 'lam': lam, 'runs': extra, 'J': J,
            'batches per run': per_run,
            'lookups': lookups, 'addresses': Y,
            # lookup table only -- see the module docstring on why these are
            # not comparable across lam
            'lookup Toffoli': lookups*acc.total(load, 'toffoli'),
            'lookup rounds': lookups*acc.total(load, 'rounds'),
            'Clean ancilla': load['Clean ancilla'],
            'Access corridors': corridors,
            'Compute zone': compute_zone,
            'System corridors': system_corridors,
            # the SelectSwap record for one lookup, so RPrep_Rot can apply this
            # exact option instead of rebuilding it
            'load': load,
            'swap ancilla needed': needed, 'swap ancilla available': budget,
            # each condition of the module docstring on its own, then together
            'feasible': swap_ok,
            'system access': system_access,
            'adder ok': adder_ok,
            'fits': fits,
            'rejected': rejected,
            'viable': not rejected}


def batching_table(N, B, R, b_rot, distances, *, lams, alpha_max, model, runs=None,
                   clean_budget=None, full_budget=None, methods=None,
                   include_rejected=False):
    '''
    Sweep lam and K over every method and return the viable options (see the
    module docstring), sorted by compute zone, then lookup Toffoli.  For
    'dirty_full' the number of recycling runs is derived from the borrowable
    budget unless `runs` is given (int, or a callable
    runs(N, B, K, lam, b_rot) -> int).

    lams:              the batch sizes to sweep, e.g. [2] or range(1, 9).
                       Required, no default, so the search range is always
                       visible at the call.
    alpha_max:         K = 2**alpha is swept for alpha = 1 .. alpha_max (K = 1
                       for 'qrom').  Required, no default, like `lams`.
    clean_budget:      the compute zone's size (clean ancilla + access
                       corridors).  None means unbounded.
    methods:           the methods to sweep.  None (default) means all of
                       METHODS; the model's layout decides which are viable.
    include_rejected:  also return the options that are not viable, each with
                       its reasons under 'rejected'.
    '''
    if methods is None:
        methods = METHODS
    rows = []
    for method in methods:
        Ks = [1] if method == 'qrom' else [2**a for a in range(1,alpha_max + 1)]
        for lam in lams:
            for K in Ks:
                g = runs(N, B, K, lam, b_rot) if callable(runs) else runs
                row = batching_cost(N, B, R, b_rot, distances, method, lam, K, g,
                                    clean_budget, full_budget, model=model)
                if row['viable'] or include_rejected:
                    rows.append(row)
    return sorted(rows, key=lambda r: (r['Compute zone'], r['lookup Toffoli']))


def best_option(N, B, R, b_rot, distances, ancilla_budget, *, lams, alpha_max, model,
                metric='lookup rounds', **kw):
    '''
    Cheapest viable option by `metric` whose compute zone (clean ancilla +
    access corridors) fits `ancilla_budget`, among the batch sizes `lams`
    and swap parameters up to 2**alpha_max (both required, see
    batching_table).
    `batching_table`'s own listing stays sorted by compute zone (for
    browsing options by footprint); this picks by cost instead.

    NOTE: the default metrics here cover the lookup table only, so this ranks
    lookups, not whole RPrep+Rot routines -- across different lam it will
    favour large batches because their per-batch work is the part left out.
    For a ranking over full routines use optimize.search_rprep_rot.
    '''
    rows = batching_table(N, B, R, b_rot, distances, lams=lams, alpha_max=alpha_max, model=model,
                          clean_budget=ancilla_budget, **kw)
    return min(rows, key=lambda r: r[metric]) if rows else None


def as_option(row):
    '''
    The (method, K, lam, runs) option of a batching row (or of an RPrep_Rot
    result), ready to pass on:  RPrep_Rot(..., **as_option(row)).
    '''
    return {'method': row['method'], 'K': row['K'],
            'lam': row['lam'] if 'lam' in row else row['size of batches'],
            'runs': row['runs']}

def show(rows, limit=None, max_toffoli=None, max_compute_zone=None):
    """
    Print a table of batching rows, with the reasons for any rejected one.

    max_toffoli:       drop options costing more than this many Toffolis
    max_compute_zone:  drop options whose compute zone (clean ancilla +
                       access corridors) is larger than this
    limit:             print at most this many rows after filtering
    """
    rows = [r for r in rows
            if (max_toffoli is None or r['lookup Toffoli'] <= max_toffoli)
            and (max_compute_zone is None or r['Compute zone'] <= max_compute_zone)]
    head = (f"{'method':<18}{'K':>3}{'lam':>5}{'runs':>6}{'lookups':>9}"
            f"{'Toffoli':>10}{'rounds':>10}{'clean':>7}{'corr':>6}{'zone':>6}"
            f"{'sys corr':>10}   rejected")
    print(head)
    print('-'*len(head))
    for r in rows[:limit]:
        print(f"{r['method']:<18}{r['K']:>3}{r['lam']:>5}{r['runs']:>6}{r['lookups']:>9}"
              f"{r['lookup Toffoli']:>10.0f}{r['lookup rounds']:>10.0f}"
              f"{r['Clean ancilla']:>7}{r['Access corridors']:>6}{r['Compute zone']:>6}"
              f"{r['System corridors']:>10}   {', '.join(r['rejected'])}")
    if not rows:
        print('(no options match)')
    return rows
