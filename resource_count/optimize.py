"""
Option search across routines that have more than one valid way to build a
SelectSwap lookup table (clean vs dirty ancilla, and -- for RPrep+Rot only --
recycling batching over several methods).  cost_functions.py and
high_level_cost.py stay pure cost evaluators for one fully-specified
configuration each; this module is the only place that enumerates
configurations, compares them, and picks one.

Every search returns {'chosen': <best cost record>, 'alternatives': [...]},
both ranked by `metric` (default 'Temporal cost', any name in
accounting.summary), so the cost of the roads not taken is always
inspectable, not just the winner.  search_rprep_rot also returns 'front',
the options no other option beats on both Toffoli cost and rounds.

Which options exist at all is the cost model's layout: the dirty methods and
the dirty prep variety borrow system qubits, which needs Z access to them
(model.SYSTEM_Z_ACCESS -- none in the rep-code layout, full in LDPC).
"""
import warnings

from resource_count import accounting as acc
from resource_count import high_level_cost as hlc
from resource_count import batching as bt
from resource_count import layout


def search_prep_variety(prep_fn, prep_args, ancilla_budget, distances, *, model, adjoint=False,
                        metric='Temporal cost', varieties=None, **zones):
    '''
    Compare OuterPrep/InnerPrep's SelectSwap varieties (clean / dirty).

    prep_fn:     hlc.OuterPrep or hlc.InnerPrep.
    prep_args:   prep_fn's positional arguments before ancilla_budget and distances, e.g.
                 (address, bit_precision) for OuterPrep, (RC, B, bit_precision)
                 for InnerPrep.
    zones:       the prep's required zone sizes, passed through: N, b_rot and
                 B (OuterPrep) or X (InnerPrep).  The dirty variety borrows
                 2*N system qubits.
    varieties:   the varieties to compare.  None (default) means the ones the
                 cost model's layout allows: 'dirty' only with Z access to the
                 system qubits (model.SYSTEM_Z_ACCESS), so 'clean' only for
                 the rep code; pass ('clean', 'dirty') to compare both anyway.
    '''
    if varieties is None:
        varieties = ('clean', 'dirty') if model.SYSTEM_Z_ACCESS else ('clean',)
    candidates = [prep_fn(*prep_args, ancilla_budget, distances, adjoint=adjoint,
                          variety=v, model=model, **zones) for v in varieties]
    candidates.sort(key=lambda r: acc.summary(r)[metric])
    return {'chosen': candidates[0], 'alternatives': candidates[1:]}


PREP_METHODS = ('qrom', 'clean', 'dirty')


def prep_options(prep_fn, prep_args, ancilla_budget, distances, *, model, alphas, adjoint=False,
                 varieties=('clean', 'dirty'), dirty_budget=None, **zones):
    '''
    Every SelectSwap option of OuterPrep/InnerPrep, as whole prep records:
    each variety at each K = 2**alpha forced at every level (alpha in
    `alphas`; see high_level_cost._sequential_prep).  K = 1 is a plain QROM,
    the same for both varieties, so it is costed once, as method 'qrom'.

    prep_fn, prep_args, zones:  as search_prep_variety; bit_precision is the
                    last of prep_args for both preps.
    dirty_budget:   borrowable system qubits for 'dirty', default 2N.

    Each record gets 'method' ('qrom', 'clean' or 'dirty'), 'Compute zone'
    (the output register -- all K copies for 'clean' -- plus its access
    corridors, at the largest K any level used) and, as batching.py,
    'rejected' / 'viable':
      'no system Z access'     'dirty' without model.SYSTEM_Z_ACCESS
      'swap source'            'dirty' borrowing more than dirty_budget
      'compute zone'           the compute zone over ancilla_budget
      'routine clean ancilla'  the prep's own clean ancilla over it
    Sorted by method, then K.
    '''
    b = prep_args[-1]
    if dirty_budget is None:
        dirty_budget = layout.system_zone_size(zones['N'])
    options = [('qrom', 'clean', 0)] + [(v, v, a) for v in varieties for a in alphas if a > 0]

    out = []
    for method, variety, alpha in options:
        # costing options over budget is the point here, not worth a warning each
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            r = prep_fn(*prep_args, ancilla_budget, distances, adjoint=adjoint, variety=variety,
                        alpha=alpha, model=model, dirty_budget=dirty_budget, **zones)
        K = r.get('K', 1)
        register = K*b if variety == 'clean' else b
        zone = register + model.access_corridors(register)
        rejected = tuple(reason for reason, ok in (
            ('no system Z access', variety == 'clean' or model.SYSTEM_Z_ACCESS),
            ('swap source', variety == 'clean' or (K - 1)*b <= dirty_budget),
            ('compute zone', ancilla_budget is None or zone <= ancilla_budget),
            ('routine clean ancilla', ancilla_budget is None
             or r['Clean ancilla'] <= ancilla_budget)) if not ok)
        out.append({**r, 'method': method, 'Compute zone': zone,
                    'rejected': rejected, 'viable': not rejected})
    return sorted(out, key=lambda r: (PREP_METHODS.index(r['method']), r['K']))


def show_prep_options(options, *, model, distances):
    '''
    Print prep_options: each option's Toffoli cost, rounds and logical
    cycles for the whole prep, then for its lookups alone (the 'lookups'
    record, the only part that depends on K) with their rounds per Toffoli,
    and its compute zone, with the reasons for any rejected one.
    '''
    head = (f"{'':12}{'whole prep':^28}  {'lookups only':^27}")
    cols = (f"{'method':<8}{'K':>4}{'Toffoli':>9}{'rounds':>10}{'cycles':>9}  "
            f"{'Toffoli':>8}{'rounds':>10}{'rnd/Toff':>9}{'zone':>6}   rejected")
    print(head)
    print(cols)
    print('-'*len(cols))
    for r in options:
        s = acc.summary(r)
        lt, lr = acc.total(r['lookups'], 'toffoli'), acc.total(r['lookups'], 'rounds')
        print(f"{r['method']:<8}{r['K']:>4}{s['Toffoli cost']:>9}{s['Temporal cost']:>10}"
              f"{s['Temporal cost']/model.rounds_per_cycle(**distances):>9.0f}  "
              f"{lt:>8}{lr:>10}{lr/lt if lt else 0:>9.1f}{r['Compute zone']:>6}"
              f"   {', '.join(r['rejected'])}")


def search_rprep_rot(N, B, R, C, bit_precision, distances, *, lams, alpha_max, model, ancilla_budget=None,
                     clean_budget=None, full_budget=None, metric='Temporal cost',
                     methods=None, include_rejected=False):
    '''
    Search RPrep+Rot's {method, K, lam} space.  batching.py's sweep costs the
    lookup table alone (its own scope, per its docstring); for each option it
    returns, this builds the corresponding *full* RPrep_Rot call, so every
    candidate here is a complete, comparable RPrep_Rot cost record, not just
    the lookup-table portion.  This is the ranking to trust across different
    lam -- batching.best_option's own ranking omits the per-batch work.

    ancilla_budget: the compute zone's size.  Only options whose lookup fits
                    it with its access corridors (batching's 'compute zone'
                    condition) are costed, and of those, candidates whose full
                    RPrep_Rot 'Clean ancilla' exceeds it are dropped too
                    (checked against the whole routine, not just the lookup
                    table's own share).
    clean_budget,
    full_budget:    passed through to batching.batching_table for the compute
                    zone and the 'dirty_full' method respectively.
                    `clean_budget` defaults to `ancilla_budget` -- they are the
                    same compute zone, and letting them differ would have the
                    sweep filter on one budget while the costing below checks
                    the other.
    lams,
    alpha_max:      the batch sizes to search, e.g. [2], and the swap
                    parameters K = 2**alpha up to 2**alpha_max.  Required, no
                    defaults, so the search range is always visible at the call.
    methods:        methods to search; None (default) means all of them, of
                    which the model's layout decides which are viable (see
                    batching.py).
    include_rejected:
                    also cost the options that are not viable, as whole
                    RPrep_Rot records (they cost what they would if built),
                    and return them under 'rejected'.

    Each candidate is RPrep_Rot's record, which carries its 'Compute zone'
    and, under 'rejected', why it is not viable (() when it is).
    Returns {'chosen', 'alternatives', 'front'}: 'front' holds the candidates
    no other candidate beats on both 'Toffoli cost' and 'Temporal cost'
    (pareto_front), ordered by `metric`.  None if nothing is viable within
    `ancilla_budget` -- unless include_rejected, which always returns the
    dict, with 'chosen' None when nothing is viable.
    '''
    if clean_budget is None:
        clean_budget = ancilla_budget
    rows = bt.batching_table(N, B, R, bit_precision, distances, lams=lams, model=model,
                             alpha_max=alpha_max, clean_budget=clean_budget,
                             full_budget=full_budget, methods=methods,
                             include_rejected=include_rejected)

    results, rejected = [], []
    for row in rows:
        # budget-exceeded is an expected, filtered-out outcome while exploring
        # candidates here, not a signal worth warning about per candidate
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            full = hlc.RPrep_Rot(N, B, R, C, bit_precision, distances, model=model,
                                 method=row['method'], K=row['K'], lam=row['lam'],
                                 runs=row['runs'], ancilla_budget=ancilla_budget,
                                 full_budget=full_budget)
        # the whole routine's clean ancilla must fit too, not just the lookup's
        if ancilla_budget is not None and full['Clean ancilla'] > ancilla_budget:
            full = {**full, 'rejected': full['rejected'] + ('routine clean ancilla',)}
        (rejected if full['rejected'] else results).append(full)

    results.sort(key=lambda r: acc.summary(r)[metric])
    rejected.sort(key=lambda r: (bt.METHODS.index(r['method']), r['K'], r['size of batches']))
    if not include_rejected:
        if not results:
            return None
        return {'chosen': results[0], 'alternatives': results[1:],
                'front': pareto_front(results)}
    return {'chosen': results[0] if results else None, 'alternatives': results[1:],
            'front': pareto_front(results), 'rejected': rejected}


def pareto_front(records, metrics=('Toffoli cost', 'Temporal cost')):
    '''
    The records that no other record beats: none is at most as large on
    every one of `metrics` (names in accounting.summary) and smaller on at
    least one.  Records with equal values are all kept.  Order is preserved.
    '''
    values = [tuple(acc.summary(r)[m] for m in metrics) for r in records]

    def beaten(v):
        return any(all(a <= b for a, b in zip(w, v)) and w != v for w in values)

    return [r for r, v in zip(records, values) if not beaten(v)]


def show_options(search_result, *, model, distances):
    '''
    Print every candidate of search_rprep_rot: its option, lookups, Toffoli
    cost, rounds and logical cycles (rounds / model.rounds_per_cycle(**distances)) of
    the whole RPrep+Rot, and its compute zone (clean ancilla + access
    corridors), with the trade-off front marked '*' and the chosen one '<-'.
    The rejected options (include_rejected=True) follow, with their reasons.
    '''
    chosen = search_result['chosen']
    front = [id(r) for r in search_result['front']]
    viable = ([chosen] if chosen else []) + search_result['alternatives']
    head = (f"{'':2}{'method':<18}{'K':>3}{'lam':>5}{'runs':>6}{'lookups':>9}{'Toffoli':>10}"
            f"{'rounds':>11}{'cycles':>10}{'clean':>7}{'corr':>6}{'zone':>6}   ")
    print(head)
    print('-'*len(head))
    for r in viable + search_result.get('rejected', []):
        s = acc.summary(r)
        mark = '*' if id(r) in front else ''
        note = '<-' if r is chosen else ', '.join(r['rejected'])
        print(f"{mark:<2}{r['method']:<18}{r['K']:>3}{r['size of batches']:>5}{r['runs']:>6}"
              f"{r['lookups']:>9}{s['Toffoli cost']:>10}{s['Temporal cost']:>11}"
              f"{s['Temporal cost']/model.rounds_per_cycle(**distances):>10.0f}"
              f"{r['Compute zone'] - r['Access corridors']:>7}{r['Access corridors']:>6}"
              f"{r['Compute zone']:>6}   {note}")
    if not viable:
        print('(no viable option)')


def show_alternatives(search_result, metric='Temporal cost', limit=None):
    '''
    Print the chosen option and every alternative's cost on `metric`, and how
    much more expensive each was, relative to the chosen one.
    '''
    chosen = search_result['chosen']
    base = acc.summary(chosen)[metric]
    label = chosen.get('method') or chosen.get('variety') or chosen['Subroutine']
    print(f"chosen: {label:<20}{metric}={base}")
    for alt in search_result['alternatives'][:limit]:
        alt_label = alt.get('method') or alt.get('variety') or alt['Subroutine']
        value = acc.summary(alt)[metric]
        extra = value - base
        pct = 100*extra/base if base else float('inf')
        print(f"  alt: {alt_label:<22}{metric}={value:<12}(+{extra}, +{pct:.1f}%)")
