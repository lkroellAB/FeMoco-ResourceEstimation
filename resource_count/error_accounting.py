"""
Error accounting, distance choice and physical qubit count of a whole phase
estimation run, for either cost model.  Used by notebooks/visualization.ipynb
and scripts/error_accounting_latex.py, so both compute these the same way.

Error model
-----------
A run fails with probability (an upper bound, the three added)

    p_fail = eps_alg + eps_magic + eps_i

  eps_alg    a logical error: eps_rep + eps_LDPC (code_errors), each
             eps = 1 - (1 - p)**V for the code's rate p per logical
             qubit-cycle (LAYER_ERROR at the code's own distance,
             LAYER_DISTANCE) and its logical qubit-cycles V over the run
             (Total_time, converted by the cost model).  The rep code's V
             includes the factory routing (FACTORY_ROUTING[model] physical qubits
             of the rep-code layer, active the whole run, protected at
             d_rep); see layer_errors
  eps_magic  total Toffolis x p_magic
  eps_i      total i-states x p_i

with p_Z, p_magic and p_i set per noise level in NOISE.  The expected
duration is the run time / (1 - p_fail); infinite once p_fail >= 1.

Distance choice
---------------
choose_distance scans DISTANCE_SCAN[model] and picks the shortest expected
duration; among the distances within TIE of it, the smallest (fewest qubits).

Physical qubits
---------------
physical_qubits: the chip's physical qubit count from its global layout
(rep code: physical_qubits_rep; hybrid LDPC: physical_qubits_ldpc), with the
algorithmic core counted without ancilla and doubled, plus the factories
(FACTORIES) and their routing (FACTORY_ROUTING), each set per cost model
(LDPC has one |i> factory fewer) and the same at every distance.
"""
import numpy as np

from resource_count import high_level_cost as hl
from resource_count import layout
from resource_count import cost_functions as rep_code
from resource_count import cost_functions_ldpc as ldpc_code

# ---- the run's fixed parameters and the noise levels ----------------------
EPSILON = 1e-3          # target precision (Ha)
ROUND_TIME = 1e-6       # one round, in seconds
# per noise level: the phase-flip rate p_Z, the magic-state error per Toffoli
# and the i-state error per i-state
NOISE = {
    '5e-4': dict(p_Z=5e-4, p_magic=2.02e-13, p_i=9.43e-13),
    '1e-3': dict(p_Z=1e-3, p_magic=1.48e-11, p_i=3.0e-12),
}
# the magic-state factories' routing area, in physical qubits of the
# rep-code layer, taken as active for the whole run, per cost model.  LDPC
# needs one |i> factory fewer, so its routing has 10 rows instead of 12
# under the |C_X C_X X> factory and no 2*11
FACTORY_ROUTING = {rep_code: 2*(22*10 + 29*12 + 2*11),     # 1180
                   ldpc_code: 2*(22*10 + 29*10)}           # 1020
# the factories, per cost model: one |C_X C_X X> factory (1566) and the
# |i> factories (196 each), three in the rep code, two in LDPC
FACTORIES = {rep_code: 1566 + 3*196,                       # 2154
             ldpc_code: 1566 + 2*196}                      # 1958
# LDPC layout: the buffer sits 5 columns into the compute zone
BUFFER_OFFSET = 5

# ---- distance choice -------------------------------------------------------
# odd d_rep only: an even d_rep corrects no more errors than the odd one
# below it (logical_error_rep's exponent is floor((d+1)/2)) but costs more
DISTANCE_SCAN = {rep_code: [dict(d_rep=d) for d in range(11, 24, 2)],
                 ldpc_code: [dict(d_rep=dr, d_LDPC=dl) for dl in (12, 16, 22) for dr in range(11, 24, 2)]}
TIE = 0.005             # within 0.5% of the shortest, the smaller distances win


# ---- logical error rates ---------------------------------------------------
def logical_error_rep(d, p_Z, A=0.12, B=23, C=0.99):
    '''
    Repetition code: logical error per d code cycles at distance d and
    phase-flip probability p_Z,

        A*d*(B*p_Z)**(C*floor((d + 1)/2)).

    A, B and C are the fit of Ruiz et al., LDPC-cat codes for low-overhead
    quantum computing in 2D, Nat. Commun. 16, 1040 (2025),
    https://doi.org/10.1038/s41467-025-56298-8, Supplementary Table I.
    '''
    exponent = C*np.floor((d+1)/2)
    return A*d*(B*p_Z)**exponent


def logical_error_ldpc(d, p_Z):
    '''
    LDPC-cat code: logical error per d code cycles at distance d and
    phase-flip probability p_Z, of the same form as logical_error_rep,

        A*d*(B*p_Z)**(C*floor((d + 1)/2)),

    with A = 0.1, C = 1 and B fitted per distance (the paper, Section VI,
    Table XII).  Only d = 12, 16 and 22 are supported; any other d raises.
    '''
    A = 0.1
    C = 1
    exponent = C*np.floor((d+1)/2)
    if d == 12:
        B = 25.8
        return A*d*(B*p_Z)**exponent
    if d == 16:
        B = 29.8
        return A*d*(B*p_Z)**exponent
    if d == 22:
        B = 42.5 #Upper bound (the paper gives B <= 42.5)
        return A*d*(B*p_Z)**exponent
    raise ValueError(f"unsupported LDPC distance {d!r}")


# logical error per logical qubit-cycle, one function of (d, p_Z) per layer,
# and the distance (the key in a model's distances) that layer is protected
# by.  Only the layers a cost actually uses are evaluated (see layer_errors),
# so the rep-code model never asks for the LDPC rate
LAYER_ERROR = {'rep': logical_error_rep, 'ldpc': logical_error_ldpc}
LAYER_DISTANCE = {'rep': 'd_rep', 'ldpc': 'd_LDPC'}


def layer_errors(cost, distances, p_Z, model, factory_routing):
    '''The logical error budget, split by layer.  For each layer the cost uses
    (cost['layers'], from Total_time):
        'rate'                   logical error per logical qubit-cycle, at
                                 the layer's own distance:
                                 LAYER_ERROR[layer](distances[LAYER_DISTANCE[layer]], p_Z)
        'logical qubit-cycles'   the layer's volume over the whole run,
                                 converted by the cost model
        'expected errors'        rate x logical qubit-cycles
        'share'                  the entry's share of all logical qubit-cycles
    plus 'factory routing': `factory_routing` physical qubits in the rep-code
    layer, active for the whole run (cost['total rounds']), converted by
    `model` like any rep-layer volume and protected at d_rep.
    A layer without a rate raises a KeyError, so no layer is left out.'''
    split = {}
    for layer in cost['layers']:
        rate = LAYER_ERROR[layer](distances[LAYER_DISTANCE[layer]], p_Z)
        volume = cost[f'total logical qubit cycles [{layer}]']
        split[layer] = {'rate': rate, 'logical qubit-cycles': volume,
                        'expected errors': rate*volume}
    rate = LAYER_ERROR['rep'](distances['d_rep'], p_Z)
    volume = model.logical_qubit_cycles('rep', factory_routing*cost['total rounds'], **distances)
    split['factory routing'] = {'rate': rate, 'logical qubit-cycles': volume,
                                'expected errors': rate*volume}
    total = sum(e['logical qubit-cycles'] for e in split.values())
    for e in split.values():
        e['share'] = e['logical qubit-cycles']/total
    return split


# the error sources (layer_errors' entries) of each code: the factory routing
# sits in the rep-code layer, at the rep rate at d_rep
CODE_SOURCES = {'rep': ('rep', 'factory routing'), 'ldpc': ('ldpc',)}


def code_errors(cost, distances, p_Z, model, factory_routing):
    '''The probability of a logical error in each code the run uses:
        'rep'    1 - (1 - p_rep)**(V_rep + V_factory routing): the rep-code
                 layer and the factory routing, at the one rep rate (d_rep),
                 their logical qubit-cycles added
        'ldpc'   1 - (1 - p_LDPC)**V_LDPC (only when the cost uses the LDPC
                 layer)
    with the rates and volumes of layer_errors.  1 - (1 - p)**V is computed
    as -expm1(V*log1p(-p)), which never forms 1 - p or (1 - p)**V, so it is
    exact even for p ~ 1e-16 (1 - p itself cannot be stored that precisely).
    A source of no code raises, so none is left out.'''
    split = layer_errors(cost, distances, p_Z, model, factory_routing)
    unassigned = set(split) - {s for sources in CODE_SOURCES.values() for s in sources}
    if unassigned:
        raise KeyError(f"error sources of no code: {sorted(unassigned)}")
    errors = {}
    for code, sources in CODE_SOURCES.items():
        present = [s for s in sources if s in split]
        if not present:
            continue
        rates = {split[s]['rate'] for s in present}
        if len(rates) != 1:
            raise ValueError(f"{code}: its sources {present} have different rates {rates}")
        volume = sum(split[s]['logical qubit-cycles'] for s in present)
        errors[code] = -np.expm1(volume*np.log1p(-rates.pop()))
    return errors


def algo_error(cost, distances, p_Z, model, factory_routing):
    '''The algorithmic error eps_alg: the codes' logical error probabilities
    (code_errors) added, an upper bound, as the magic-state and i-state
    errors are added to it in failure.'''
    return sum(code_errors(cost, distances, p_Z, model, factory_routing).values())


def failure(cost, distances, noise, model, factory_routing):
    '''(total, logical, magic, i-state) failure probabilities of one run
    (cost: Total_time's record) at noise level `noise` (an entry of NOISE)'''
    logical = algo_error(cost, distances, noise['p_Z'], model, factory_routing)
    magic = cost['total Toffoli']*noise['p_magic']
    istate = cost['total i-state']*noise['p_i']
    return logical + magic + istate, logical, magic, istate


def run_volumes(cost, distances, model, factory_routing, factories):
    '''The whole run's duration and effective volumes (cost: Total_time's
    record), the volumes in logical qubit-cycles:
        'cycles'           the run's logical cycles (cost['total logical
                           cycles']: its rounds / model.rounds_per_cycle)
        'algorithm'        the algorithm's effective volume, each layer
                           converted at its own distance (cost['total
                           effective volume'])
        'factory routing'  `factory_routing` physical qubits of the rep-code
                           layer, active for the whole run, converted like any
                           rep-layer volume (as in layer_errors)
        'factories'        `factories` physical qubits, active for the whole
                           run, converted the same way (at d_rep, though the
                           factories distil at their own distances)
        'total'            the three volumes added
    and per layer L the run uses ('rep', 'ldpc'):
        'algorithm [L]'       the algorithm's volume in layer L (cost['total
                              logical qubit cycles [L]'])
        'logical qubits [L]'  the effective logical qubits of layer L: its
                              volume divided by its own logical cycles, the
                              run's rounds / the layer's distance
                              (LAYER_DISTANCE), i.e. the average number of
                              logical qubits in it.  The rep layer's volume
                              includes the factory routing and the factories
        'logical qubits'      the layers' effective logical qubits added'''
    rounds = cost['total rounds']
    out = {'cycles': cost['total logical cycles'],
           'algorithm': cost['total effective volume'],
           'factory routing': model.logical_qubit_cycles('rep', factory_routing*rounds, **distances),
           'factories': model.logical_qubit_cycles('rep', factories*rounds, **distances)}
    out['total'] = out['algorithm'] + out['factory routing'] + out['factories']
    for layer in cost['layers']:
        volume = cost[f'total logical qubit cycles [{layer}]']
        out[f'algorithm [{layer}]'] = volume
        if layer == 'rep':
            volume += out['factory routing'] + out['factories']
        out[f'logical qubits [{layer}]'] = volume/(rounds/distances[LAYER_DISTANCE[layer]])
    out['logical qubits'] = sum(out[f'logical qubits [{layer}]'] for layer in cost['layers'])
    return out


# ---- a run, and the distance choice ----------------------------------------
def total_time(system, budget, option, distances, model, epsilon, round_time):
    '''Total_time for `system` (N, B, R, C, b_coef, b_rot, be_const) with cost
    model `model`, its compute-zone budget and block-encoding options
    (method, K, lam, outer_variety, inner_variety), at `distances`'''
    p = system
    return hl.Total_time(p['N'], p['B'], p['R'], p['C'], p['b_coef'], p['b_rot'], budget,
                         layout.buffer_zone_size(), distances, p['be_const'],
                         epsilon=epsilon, round_time=round_time, model=model, **option)


def choose_distance(run, scan, noise, model, factory_routing, tie):
    '''Every distance in `scan` with its run time, failures and expected
    duration, and the chosen one: the shortest expected duration, or among
    those within `tie` of it the smallest distances.  run(distances) costs
    the run (Total_time's record); a distance the model refuses (ValueError,
    e.g. the LDPC angle storage check) is kept as infeasible, with the
    model's reason.  Returns (rows, chosen row).'''
    rows = []
    for dist in scan:
        try:
            cost = run(dist)
        except ValueError as e:
            rows.append(dict(dist=dist, infeasible=str(e)))
            continue
        fail, logical, magic, istate = failure(cost, dist, noise, model, factory_routing)
        days = cost['total days']
        factory = layer_errors(cost, dist, noise['p_Z'], model, factory_routing)['factory routing']['share']
        rows.append(dict(dist=dist, days=days, fail=fail, logical=logical, magic=magic, istate=istate, factory=factory,
                         expected=days/(1 - fail) if fail < 1 else np.inf))
    feasible = [r for r in rows if 'infeasible' not in r]
    best = min(r['expected'] for r in feasible)
    near = [r for r in feasible if r['expected'] <= best*(1 + tie)]
    chosen = min(near, key=lambda r: (max(r['dist'].values()), sum(r['dist'].values())))
    return rows, chosen


# ---- physical qubits ---------------------------------------------------------
def address_bits(system):
    """(n_X, n_{B+1}): the X and B register sizes, as BE_cost sets them"""
    n_r = int(np.ceil(np.log2(system['R'])))
    X = system['N'] + 2**n_r*system['C']
    return int(np.ceil(np.log2(X))), int(np.ceil(np.log2(system['B'] + 1)))


def physical_qubits_rep(system, option, d_rep):
    """the rep code's physical qubits: the terms of Space_algo, then the total"""
    N, b_rot = system['N'], system['b_rot']
    n_X, n_B = address_bits(system)
    w_comp = int(np.ceil(3*option['lam']*b_rot/2))
    buffer = layout.buffer_zone_size()
    algo = [(f'compute, Fourier, address: ({w_comp} + {b_rot} + {n_X} + {n_B} + 3)(d+5)',
             (w_comp + b_rot + n_X + n_B + 3)*(d_rep + 5)),
            (f'system, top half: ({N} + 2)(d+4)', (N + 2)*(d_rep + 4)),
            (f'system, bottom half: ({N} + 2)d', (N + 2)*d_rep),
            (f'buffer: {buffer}d', buffer*d_rep)]
    space_algo = sum(n for _, n in algo)
    terms = algo + [('Space_algo', space_algo), ('2 Space_algo (with ancilla)', 2*space_algo),
                    ('factory routing', FACTORY_ROUTING[rep_code]), ('factories', FACTORIES[rep_code])]
    return dict(params=dict(n_X=n_X, n_B1=n_B, w_comp=w_comp), terms=terms,
                total=2*space_algo + FACTORY_ROUTING[rep_code] + FACTORIES[rep_code])


def compute_angles(option):
    """the rotation angles RPrep+Rot's lookup holds in the compute zone: all
    K copies with the clean SelectSwap, lam otherwise"""
    return option['K']*option['lam'] if option['method'] == 'clean' else option['lam']


def physical_qubits_ldpc(system, option, d_rep, d_LDPC):
    """the hybrid LDPC model's physical qubits: the terms of each layer, then the total"""
    N, b_rot = system['N'], system['b_rot']
    n_X, n_B = address_bits(system)
    chip = ldpc_code.chip_layout(d_LDPC)
    H, Delta_H = chip['H'], chip['base']
    n_comp = compute_angles(option)*b_rot           # compute qubits, after the b_rot phase gradient qubits
    l_address = int(np.ceil((n_X + n_B)/2))
    l_comp = max(int(np.ceil(3*b_rot/2)), int(np.ceil((b_rot + n_comp)/2)))
    L = l_address + l_comp + N + 3
    buffer = layout.buffer_zone_size()
    beyond = buffer - BUFFER_OFFSET                 # 22: routing columns beyond the compute zone
    ldpc_layer = 2*((H - 1)*L + Delta_H)
    rep = [(f'routing under the LDPC layer: HL + Delta_H = {H}*{L} + {Delta_H}', H*L + Delta_H),
           ('bottom routing: 2L', 2*L),
           (f'routing under the compute zone: 3({l_comp} + {beyond})', 3*(l_comp + beyond)),
           (f'buffer: {buffer}d', buffer*d_rep)]
    rep_layer = sum(n for _, n in rep)
    terms = [(f'LDPC layer: 2((H-1)L + Delta_H) = 2(({H}-1)*{L} + {Delta_H})', ldpc_layer)] + rep + \
            [('repetition-code layer', rep_layer), ('2 x repetition-code layer (with ancilla)', 2*rep_layer),
             ('factory routing', FACTORY_ROUTING[ldpc_code]), ('factories', FACTORIES[ldpc_code])]
    return dict(params=dict(n_X=n_X, n_B1=n_B, H=H, Delta_H=Delta_H, n_angles=compute_angles(option),
                            n_comp=n_comp, l_address=l_address, l_comp=l_comp, L=L),
                terms=terms, total=ldpc_layer + 2*rep_layer + FACTORY_ROUTING[ldpc_code] + FACTORIES[ldpc_code])


def physical_qubits(system, m, option, distances):
    """physical_qubits_rep or physical_qubits_ldpc, by cost model"""
    if m is rep_code:
        return physical_qubits_rep(system, option, **distances)
    if m is ldpc_code:
        return physical_qubits_ldpc(system, option, **distances)
    raise ValueError(f"no physical qubit count for {m.__name__}")
