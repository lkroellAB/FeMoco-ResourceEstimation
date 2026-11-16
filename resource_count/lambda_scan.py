"""
The fastest block encoding at each compute-zone size, for either cost model
-- a contained example, built on the rest of the package without changing
it.  Used by scripts/lambda_scan.py and the notebook's 'Growing the compute
zone' section.

Levels
------
RPrep+Rot loads the N - 1 rotation angles in J = ceil((N-1)/lam) batches of
lam angles (batching.py), so lam only matters through J: lam_levels lists,
for each J that some lam reaches, the smallest such lam (FeMoCo-54: lam = 2,
3, 4, 5, 6, 7, 8, 9, 11, 14, 18, 27, 53 for J = 27, 18, 14, 11, 9, 8, 7, 6,
5, 4, 3, 2, 1).  A larger lam with the same J only costs space.

Each level is a compute-zone budget, the zone a lam-angle output register
needs (compute_zone): lam*b_rot qubits and their access corridors
(model.access_corridors: one per two qubits in the rep code, none in LDPC).
At lam = 2 that is the settings' budget (48 / 32 for FeMoCo-54).

The best block encoding at a budget (best_at)
---------------------------------------------
Candidates: every RPrep+Rot option viable within the budget
(optimize.search_rprep_rot with batch sizes 1..lam and K up to 2**alpha_max,
so also the clean SelectSwap with fewer angles, or a smaller lam), each with
every prep variety pair the model allows (PREP_VARIETIES: clean only in the
rep code, which has no Z access to the system qubits).  The preps pick
their own K per level within the same budget.  alpha_max = 6 reaches every
viable K for both molecules (the largest viable is K = 64, FeMoCo-76's rep
code); keep 2**alpha_max below RPrep+Rot's address count (486 / 988), where
a clean SelectSwap counts no output copies (its QROM has one address) and
would pass the LDPC budget for free.

Choice, in two stages, by expected duration (error_accounting.failure):
  1. every candidate at the reference distances (the settings' chosen
     distances at the noise level);
  2. the `top` fastest of them over the whole distance scan
     (error_accounting.choose_distance, DISTANCE_SCAN, TIE); the shortest
     expected duration wins.
A candidate the model refuses at the reference distances (ValueError, e.g.
the LDPC angle storage check) is left out of stage 1.

Physical qubits: the chip with the level's compute zone
(error_accounting.physical_qubits with a plain-QROM option of the level's
lam), whatever option runs in it -- the zone is built for the level.

Durations: the run time and the expected duration carry `slack` (a factor,
passed in: the paper tables' error_accounting_latex.SLACK), as in the paper
tables: expected = slack*days/(1 - p_fail).  A common factor, so it does not
change the choice.
"""
import warnings

import numpy as np

from resource_count import accounting as acc
from resource_count import batching as bt
from resource_count import error_accounting as ea
from resource_count import high_level_cost as hl
from resource_count import layout
from resource_count import optimize as opt

# the subroutines of the round shares: BE_cost's breakdown names -> group
GROUPS = {'RPrep+Rot': 'RPrep+Rot', 'InnerPrep': 'InnerPrep', 'InnerPrep_dag': 'InnerPrep',
          'OuterPrep': 'OuterPrep', 'OuterPrep_dag': 'OuterPrep', 'Spin': 'other', 'Maj': 'other',
          'InnerRef': 'other', 'OuterRef': 'other'}
GROUP_ORDER = ('RPrep+Rot', 'InnerPrep', 'OuterPrep', 'other')


def lam_levels(N, lam_min):
    '''For each number of batches J = ceil((N-1)/lam) reached with lam >=
    lam_min, the smallest lam reaching it, ascending: [(lam, J)]'''
    levels = {}
    for lam in range(lam_min, N):
        levels.setdefault(int(np.ceil((N - 1)/lam)), lam)
    return sorted(((lam, J) for J, lam in levels.items()))


def compute_zone(model, lam, b_rot):
    '''the compute zone a lam-angle output register needs: lam*b_rot qubits
    and their access corridors'''
    return lam*b_rot + model.access_corridors(lam*b_rot)


def prep_varieties(model):
    '''the (outer_variety, inner_variety) pairs the model's layout allows:
    'dirty' borrows system qubits, which needs Z access to them'''
    kinds = ('clean', 'dirty') if model.SYSTEM_Z_ACCESS else ('clean',)
    return [(o, i) for o in kinds for i in kinds]


def candidates(system, model, budget, lam, distances, alpha_max):
    '''every block-encoding option viable within `budget`: the RPrep+Rot
    options of search_rprep_rot (batch sizes 1..lam), each with every prep
    variety pair'''
    p = system
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        found = opt.search_rprep_rot(p['N'], p['B'], p['R'], p['C'], p['b_rot'], distances,
                                     lams=range(1, lam + 1), alpha_max=alpha_max, model=model,
                                     ancilla_budget=budget)
    rprep = [] if found is None else [found['chosen']] + found['alternatives']
    return [dict(bt.as_option(r), outer_variety=o, inner_variety=i)
            for r in rprep for o, i in prep_varieties(model)]


def expected_at(system, model, budget, option, distances, noise):
    '''(expected duration, run) of one option at `distances`: the run time /
    (1 - p_fail), infinite once p_fail >= 1'''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        run = ea.total_time(system, budget, option, distances, model, ea.EPSILON, ea.ROUND_TIME)
    fail, *_ = ea.failure(run, distances, noise, model, ea.FACTORY_ROUTING[model])
    return (run['total days']/(1 - fail) if fail < 1 else np.inf), run


def round_shares(system, model, budget, option, distances):
    '''the block encoding's rounds by GROUPS, as shares of its rounds, and
    the BE_cost record'''
    p = system
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        be = hl.BE_cost(p['N'], p['B'], p['R'], p['C'], p['b_coef'], p['b_rot'], budget, distances,
                        model=model, buffer_budget=layout.buffer_zone_size(), **option)
    total = acc.total(be, 'rounds')
    shares = dict.fromkeys(GROUP_ORDER, 0.0)
    for r in be['breakdown']:
        shares[GROUPS[r['name']]] += r['calls']*r['rounds']/total
    assert abs(sum(shares.values()) - 1) < 1e-9
    return shares, be


def best_at(system, model, lam, J, reference, level, *, alpha_max, top, slack):
    '''The fastest block encoding at the compute zone of `lam` (see the
    module docstring), at noise level `level` (a key of NOISE), with stage 1
    at the `reference` distances.  Returns a dict of the level, the chosen
    option and distances, the run's totals, failures and expected duration
    (both durations times `slack`), the physical qubits and the round shares
    (round_shares).'''
    noise = ea.NOISE[level]
    budget = compute_zone(model, lam, system['b_rot'])
    ranked = []
    for option in candidates(system, model, budget, lam, reference, alpha_max):
        try:
            expected, _ = expected_at(system, model, budget, option, reference, noise)
        except ValueError:
            continue
        ranked.append((expected, option))
    if not ranked:
        raise ValueError(f'no viable option at lam = {lam} (compute zone {budget})')
    ranked.sort(key=lambda x: x[0])

    best = None
    for _, option in ranked[:top]:
        def run(distances, option=option):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                return ea.total_time(system, budget, option, distances, model, ea.EPSILON, ea.ROUND_TIME)
        _, chosen = ea.choose_distance(run, ea.DISTANCE_SCAN[model], noise, model, ea.FACTORY_ROUTING[model], ea.TIE)
        if best is None or chosen['expected'] < best[0]['expected']:
            best = (chosen, option)
    chosen, option = best
    distances = chosen['dist']

    _, run = expected_at(system, model, budget, option, distances, noise)
    shares, be = round_shares(system, model, budget, option, distances)
    zone = dict(method='qrom', K=1, lam=lam)            # the chip is built for the level's zone
    phys = ea.physical_qubits(system, model, zone, distances)
    # the chip's compute zone is the level's: w_comp (rep) / the compute qubits (LDPC)
    assert phys['params'].get('w_comp', phys['params'].get('n_comp')) == budget, (phys['params'], budget)
    return dict(lam=lam, J=J, budget=budget, option=option, distances=distances, slack=slack,
                days=slack*chosen['days'], fail=chosen['fail'], logical=chosen['logical'],
                magic=chosen['magic'], istate=chosen['istate'], expected=slack*chosen['expected'],
                phys=phys['total'],
                toffoli=run['total Toffoli'], istates=run['total i-state'],
                volume=run['total effective volume'], be_rounds=acc.total(be, 'rounds'),
                shares=shares, candidates=len(ranked))


def scan(system, model, settings, level, *, lam_max, alpha_max, top, slack):
    '''best_at for every level of lam_levels(N, 2) up to lam_max, with the
    settings' distances at `level` as the reference'''
    reference = settings[model]['distances'][level]
    return [best_at(system, model, lam, J, reference, level, alpha_max=alpha_max, top=top, slack=slack)
            for lam, J in lam_levels(system['N'], 2) if lam <= lam_max]


# ---- reading a scan --------------------------------------------------------------
def space_time_volume(row):
    '''a scan row's physical qubits x expected duration, in qubit-hours (the
    volume of Low et al., A Denser Planar Surface Code, arXiv:2605.30455:
    qubits x expected runtime)'''
    return row['phys']*row['expected']*24


def best_within(rows, qubits):
    '''the shortest expected duration of the rows with at most `qubits`
    physical qubits; infinite when none fits'''
    return min((r['expected'] for r in rows if r['phys'] <= qubits), default=np.inf)


def fewest_as_fast(points, days):
    '''the point of `points` ((physical qubits, expected days)) with the
    fewest qubits among those whose expected duration is at most `days`;
    None when none is that fast'''
    return min((p for p in points if p[1] <= days), default=None)


def footprint_crossover(rows_a, rows_b):
    '''Which scan is faster at a footprint: at a qubit budget, each one's
    best_within it, at every budget the two scans' qubit counts give.
    Returns (last, first): the largest budget at which b is strictly faster
    than a (None if never), and the smallest budget above it, from which on
    a is at least as fast (None if never)'''
    budgets = sorted({r['phys'] for r in rows_a + rows_b})
    b_faster = [q for q in budgets if best_within(rows_b, q) < best_within(rows_a, q)]
    last = max(b_faster, default=None)
    first = next((q for q in budgets if last is None or q > last), None)
    return last, first
