"""
High-level composition of the SOS block encoding and phase estimation.

Every routine returns a cost record (see accounting.py), labelled with its
'Subroutine' name and its options.  Each is a sequence of steps; each step
lists what is BUSY in it, and everything else present is charged as idle
(layout.idle_except).  Read the totals with accounting.summary(record):
'Spatial cost' in physical qubits, 'Clean ancilla' in logical qubits (address
registers excluded), 'Temporal cost' in rounds, 'Toffoli cost' in CCZ
consumptions, 'i-state cost' in |i> injections, 'Qubit cycles' in physical
qubit-rounds (a flow, safe to sum -- unlike 'Spatial cost', which is a peak).
Those space and volume names are the 'rep' layer; a model that puts zones
in another layer L adds 'Spatial cost [L]', 'Qubit cycles [L]' and 'Idle
cycles [L]' in L's own unit.  Total_time converts every layer to logical
qubit-cycles with the model before adding them.
Warnings are raised when a budget is exceeded or when a lookup table degenerates.

The cost model
--------------
Every routine takes a required `model`: the module that costs the primitives
for one error-correcting code, e.g. `cost_functions` for the repetition code.
Nothing here imports a model itself, so switching code means passing a
different module.  A model must provide every name in MODEL_INTERFACE.

Distances
---------
Every routine takes `distances`: the model's code distances, a dict passed
through unchanged and unpacked into keywords (**distances) at each model
call.  Its keys are the model's own parameter names:
    rep code (cost_functions)        dict(d_rep=15)
    hybrid (cost_functions_ldpc)     dict(d_rep=15, d_LDPC=16)
Nothing here reads a distance itself.  A missing or extra key is refused by
the model function's own signature (a TypeError).
"""

import warnings
import numpy as np
from resource_count import accounting as acc
from resource_count import batching as bt
from resource_count import layout

# what high_level_cost.py, batching.py and optimize.py use from a cost model
MODEL_INTERFACE = (
    # primitives, each returning a cost record (accounting.py)
    'SelectSwap_cost', 'adder_cost', 'qc_comp_cost', 'qc_add_cost',
    'PlainSWAP_cost',
    'ref', 'phase_fixup_cost', 'qrom_ancilla',
    # the routines' other steps: shuffles, rotation, MBU, Maj's placeholder
    'prep_shuffle_cost', 'rot_shuffle_cost', 'rotation_cost', 'mbu_cost', 'maj_cost',
    # where each zone lives, and in what unit (layout.idle_except); each
    # layer's volume in logical qubit-cycles, and rounds per logical cycle
    # (Total_time)
    'zone_qubits', 'logical_qubit_cycles', 'rounds_per_cycle',
    # fallback timing, and which primitives still use it (printed by the report)
    'TOFF_TIME', 'FALLBACK',
    # access corridors for qubits that need Z access, and whether the system
    # qubits have it (i.e. can be borrowed as dirty ancilla) -- batching.py
    # and optimize.py
    'access_corridors', 'SYSTEM_Z_ACCESS',
)


def alpha_budget(ancilla_budget, target_bit, *, model):
    '''
    Largest alpha for a clean SelectSwap: the K copies of the output
    register (the base copy plus K-1 extras) all draw from the same
    compute-zone budget, together with the access corridors they need for Z
    access (model.access_corridors: one per two qubits in the rep code, none
    in LDPC), so K*target_bit + corridors <= ancilla_budget.  QROM's buffer
    need is a separate zone with its own budget -- it never competes with
    the compute zone, so it plays no part in this bound.
    '''

    def zone(K):
        return K*target_bit + model.access_corridors(K*target_bit)

    a = -1
    while zone(2**(a+1)) <= ancilla_budget:
        a += 1
    if a < 0:
        warnings.warn("budget cannot hold one Z-accessible output register")
        return 0
    return a


def SelectSwap_opt(address, target_bit, alpha_max, distances, variety='clean',
                   batched=False, metric='Temporal cost', *, model):
    '''
    Optimal swap split for a SelectSwap lookup table.

    alpha is capped by the address width: swapping more branches than the table
    has addresses is not a circuit.  Switch `metric` to 'Toffoli cost' if cycle time should not be considered.
    '''

    if address <= 1:
        return 0
    alpha_max = min(int(alpha_max), int(np.floor(np.log2(address))))
    if alpha_max < 0:
        return 0
    costs = [acc.summary(model.SelectSwap_cost(address, target_bit, a, **distances, variety=variety, batched=batched))[metric]
             for a in range(alpha_max + 1)]
    return int(np.argmin(costs))


def _sequential_prep(name, sizes, bit_precision, ancilla_budget, distances, adjoint, *,
                     model, N, b_rot, X, B, variety='clean', dirty_budget=None,
                     buffer_budget=None, alpha=None):
    '''
    Shared body of OuterPrep and InnerPrep: one SelectSwap plus one phase
    gradient addition per level.  Forward, all MBU phases combine into a single
    fixup at the end; for the adjoint each fixup is performed on the spot.
    Unlike RPrep+Rot, there is no recycling lookup table batching here -- each
    level is an independent, unbatched SelectSwap.

    Zones present throughout: system register (2N), Fourier state (b_rot),
    QPE, address zone (X and B registers + 2 selection qubits).  From each
    lookup until its MBU, also the loaded output register (bit_precision =
    b_coef qubits).  The flag qubit does not exist yet (Spin computes it).

    Per level L, one step after another; each lists what is BUSY (inside its
    own footprint), everything else present is idle:

      lookup    SelectSwap over L     busy: the ceil(log2 L) address bits it
                                            reads, the output register (its
                                            target); for the dirty variety
                                            also the system qubits it borrows
      shuffle   move the b_coef       busy: those b_coef Fourier qubits; the
                Fourier qubits in           loaded output register sits idle
                and back out
      addition  adder                 busy: b_coef Fourier qubits, the output
                                            register.  The rotated qubit is
                                            not part of it, so it is idle
      rotation  2 parity-controlled   busy: the b_coef Fourier qubits (the
                NOTs on the qubit           control), 1 address qubit (the
                being prepared              qubit being prepared).  One NOT
                                            runs while the output is live,
                                            the other after it is measured
                                            out; both are costed with the
                                            output live, so it sits idle
      MBU       measure the output    busy: the output register

    fixup (forward: once, over max(sizes); adjoint: after each level)
                                      busy: the address bits it reads (both
                                            branches' footprints contain them)

    The adder acts on the same qubits the lookup loaded, so the steps' space
    is the larger of them (seq), not their sum.

    variety:       'clean' draws the K-1 SelectSwap copies from `ancilla_budget`;
                    'dirty' borrows them from `dirty_budget` instead (e.g. the
                    system qubits) and restores them afterwards.
    dirty_budget:   borrowable ancilla.  Required when variety == 'dirty'.
    buffer_budget:  the buffer zone's fixed size.  Only checked (as a
                    warning), never used to constrain alpha -- the buffer
                    zone is separate from the compute zone `ancilla_budget`
                    bounds.
    N, b_rot, X, B: system size, Fourier resource state width and the X and
                    B register sizes, for the zones above.  Required.
    alpha:          None (default): each level picks its own K = 2**alpha
                    (SelectSwap_opt, within the budget).  An int forces
                    K = 2**alpha at every level, capped by the level's
                    address width, whatever the budget -- for comparing
                    options (alpha = 0 is a plain QROM); a budget it exceeds
                    is warned about, as always.
    The record's 'levels' lists (addresses, K) for each level's lookup, 'K'
    is the largest K any level used, and 'lookups' the cost of the lookups
    alone (one per level, one after another, no idle qubits): the only part
    of the prep that depends on K.
    '''
    name = name + ('_dag' if adjoint else '')
    # A single-address register has no levels to load: the state is already
    # the one basis state, so the whole routine is a no-op and costs nothing.
    if not sizes:
        return {**acc.empty(), 'Subroutine': name, 'K': 1, 'levels': [],
                'lookups': acc.empty()}

    if variety == 'dirty':
        alpha_max = int(np.floor(np.log2(1 + dirty_budget/bit_precision)))
    else:
        alpha_max = alpha_budget(ancilla_budget, bit_precision, model=model)
    add = model.adder_cost(bit_precision, **distances)

    zones = {'system': layout.system_zone_size(N),
             'fourier': layout.fourier_zone_size(b_rot),
             'QPE': layout.QPE_QUBITS,
             'address': layout.address_zone_size(X, B)}
    with_output = {**zones, 'output': bit_precision}

    def reads(L):
        '''address bits a lookup (or fixup) over L addresses reads'''
        return int(np.ceil(np.log2(L)))

    def fixup_step(L):
        fixup = model.phase_fixup_cost(L, **distances, ancilla_budget=ancilla_budget, buffer_budget=buffer_budget)
        return layout.idle_except(fixup, model, distances, zones, busy={'address': reads(L)})

    # move the b_coef Fourier qubits in and back out again; rotate the qubit
    # being prepared (2 parity-controlled NOTs); measure out the loaded
    # output register
    shuffle = model.prep_shuffle_cost(bit_precision, **distances)
    rotation = model.rotation_cost(bit_precision, **distances)
    mbu = model.mbu_cost(bit_precision, **distances)

    steps, loads, levels = [], [], []
    for L in sizes:
        if alpha is None:
            a = SelectSwap_opt(L, bit_precision, alpha_max, distances, variety=variety, model=model)
        else:
            # as SelectSwap_opt: no more branches than the table has addresses
            a = min(alpha, int(np.floor(np.log2(L)))) if L > 1 else 0
        levels.append((L, 2**a))
        load = model.SelectSwap_cost(L, bit_precision, a, **distances, variety=variety)
        loads.append(load)
        steps += [
            # a dirty lookup's borrowed system qubits are busy in it (no
            # more than the system holds, should dirty_budget exceed it)
            layout.idle_except(load, model, distances, with_output,
                               busy={'address': reads(L), 'output': bit_precision,
                                     'system': min(load['Dirty ancilla'], zones['system'])}),
            layout.idle_except(shuffle, model, distances, with_output,
                               busy={'fourier': bit_precision}),
            layout.idle_except(add, model, distances, with_output,
                               busy={'fourier': bit_precision, 'output': bit_precision}),
            # costed with the output still live (see the docstring)
            layout.idle_except(rotation, model, distances, with_output,
                               busy={'fourier': bit_precision, 'address': 1}),
            layout.idle_except(mbu, model, distances, with_output,
                               busy={'output': bit_precision}),
        ]
        if adjoint:
            steps.append(fixup_step(L))
    if not adjoint:
        steps.append(fixup_step(max(sizes)))

    # every step runs after the previous one: the swap registers are
    # uncomputed before the addition, so the ancilla demands are sequential
    # too (seq's max)
    cost = acc.seq(*steps)

    layout.check_feasibility(cost, name, ancilla_budget=ancilla_budget,
                             dirty_budget=dirty_budget if variety == 'dirty' else None,
                             buffer_budget=buffer_budget)

    return {**cost, 'Subroutine': name, 'ancilla_budget': ancilla_budget, 'variety': variety,
            'K': max(K for _, K in levels), 'levels': levels, 'lookups': acc.seq(*loads)}


def OuterPrep(address: int, bit_precision: int, ancilla_budget: int, distances: dict, *,
              model, N: int, b_rot: int, B: int, adjoint: bool = False,
              variety: str = 'clean', dirty_budget: int = None,
              buffer_budget: int = None, alpha: int = None):
    '''
    Pure state preparation of |x> over `address` addresses.
    Level i loads a table of 2**i entries, i = 0 .. n_X-1, reading the first
    i bits of X; the i=0 level is a classical constant and costs nothing in
    the lookup.

    N, b_rot, B:   system size, Fourier resource state width and the B
                   register's size, for the zones present (see
                   _sequential_prep).  Required.
    dirty_budget:  for variety == 'dirty', the borrowable ancilla; defaults to
                   2*N (the system qubits).
    buffer_budget: the buffer zone's fixed size, checked but never used to
                   constrain alpha (see _sequential_prep).
    alpha:         None picks K per level; an int forces K = 2**alpha (see
                   _sequential_prep).
    '''
    if address < 1:
        raise ValueError(f"address must be at least 1, got {address}")
    if variety == 'dirty' and dirty_budget is None:
        dirty_budget = 2*N
    n_address = int(np.ceil(np.log2(address)))
    sizes = [2**i for i in range(n_address)]
    return _sequential_prep('OuterPrep', sizes, bit_precision, ancilla_budget, distances,
                            adjoint, model=model, N=N, b_rot=b_rot, X=address, B=B,
                            variety=variety, dirty_budget=dirty_budget,
                            buffer_budget=buffer_budget, alpha=alpha)


def InnerPrep(RC: int, B: int, bit_precision: int, ancilla_budget: int, distances: dict, *,
              model, N: int, b_rot: int, X: int, adjoint: bool = False,
              variety: str = 'clean', dirty_budget: int = None,
              buffer_budget: int = None, alpha: int = None):
    '''
    Pure state preparation of |psi_{r,c}> for every index pair, so each of the
    n_B levels iterates over RC = 2**n_R * C addresses as well: level i reads
    (r, c) plus the first i bits of b, ceil(log2(RC * 2**i)) bits.

    N, b_rot, X:   system size, Fourier resource state width and the X
                   register's size (N + 2**n_R C), for the zones present (see
                   _sequential_prep).  Required.
    dirty_budget:  for variety == 'dirty', the borrowable ancilla; defaults to
                   2*N (the system qubits).
    buffer_budget: the buffer zone's fixed size, checked but never used to
                   constrain alpha (see _sequential_prep).
    alpha:         None picks K per level; an int forces K = 2**alpha (see
                   _sequential_prep).
    '''
    if RC < 1 or B < 1:
        raise ValueError(f"RC and B must be at least 1, got RC={RC}, B={B}")
    if variety == 'dirty' and dirty_budget is None:
        dirty_budget = 2*N
    n_b = int(np.ceil(np.log2(B+1)))
    sizes = [RC*2**i for i in range(n_b)]
    return _sequential_prep('InnerPrep', sizes, bit_precision, ancilla_budget, distances,
                            adjoint, model=model, N=N, b_rot=b_rot, X=X, B=B,
                            variety=variety, dirty_budget=dirty_budget,
                            buffer_budget=buffer_budget, alpha=alpha)


def Spin(address: int, system: int, distances: dict, b_rot: int, B: int, *, model):
    '''
    Controlled swap on the spin qubit, plus preparation of the comparison flag
    qubit |x >= 2**n_R C> reused in RPrep and in Spin_dag.
    The H gates from the description (2 here, 1 in Spin_dag) are costed in
    Maj (maj_cost's 3 H), where each model prices them in its own code.

    Zones present throughout: system register, Fourier state, QPE, address
    zone, flag qubit.  Two steps, comparator first; each lists what is BUSY
    (inside its own footprint), everything else is idle:

      comparator  busy:  X register, flag qubit (its output)
                  idle:  the rest -- so the spin control qubit is idle here
      swap        busy:  system register, spin control qubit
                  idle:  the rest -- so the flag (computed by the comparator,
                         unused here) is idle here

    Spin_dag runs the same two steps in reverse, so its flag is idle during
    the swap too, and its cost is the same.
    '''
    n_address = int(np.ceil(np.log2(address)))
    comp = model.qc_comp_cost(n_address, **distances)
    swap = model.PlainSWAP_cost(system, **distances)

    zones = {'system': layout.system_zone_size(system),
             'fourier': layout.fourier_zone_size(b_rot),
             'QPE': layout.QPE_QUBITS,
             'address': layout.address_zone_size(address, B),
             'flag': layout.FLAG_QUBITS}
    comp_step = layout.idle_except(comp, model, distances, zones,
                                   busy={'address': n_address, 'flag': layout.FLAG_QUBITS})
    swap_step = layout.idle_except(swap, model, distances, zones,
                                   busy={'system': layout.system_zone_size(system),
                                         'address': 1})
    cost = acc.seq(comp_step, swap_step)
    # Spin's own spin qubit, held throughout whichever of swap/comp is running
    cost['Clean ancilla'] += 1

    return {**cost, 'Subroutine': 'Spin'}


def Maj(address: int, B: int, distances: dict, *, model, N: int, b_rot: int):
    '''
    Majorana operator.  Placeholder: 2 comparators and 2 n_B-bit Toffolis per
    the subroutine table, plus 2 CCZ, 1 CZ and 3 H.  The 3 H are the ones
    the text puts in Spin (2) and Spin_dag (1): booked here, once per
    Spin/Spin_dag pair, so that each model's maj_cost prices them in its
    own code.

    Zones present throughout: system register (2N), Fourier state (b_rot),
    QPE, address zone (X and B registers + 2 selection qubits), flag qubit.
    Steps, one after another; each lists what is BUSY (inside its own
    footprint), everything else present is idle:

      comparator (twice)       busy: the X register
      the rest (model.maj_cost) busy: the B register.  2 n_B-bit Toffolis
        (2*n_B Toffolis,             plus the 2 CCZ, CZ and 3 H; strictly
        2 CCZ, CZ, 3 H)              the CCZ also use the flag and a
                                     selection qubit and the CZ the flag,
                                     but that is not resolved here.  Its
                                     footprint must contain the B register
    '''
    n_address = int(np.ceil(np.log2(address)))
    n_b = int(np.ceil(np.log2(B + 1)))
    comp = model.qc_comp_cost(n_address, **distances)
    hard_coded = model.maj_cost(n_b, **distances)

    zones = {'system': layout.system_zone_size(N),
             'fourier': layout.fourier_zone_size(b_rot),
             'QPE': layout.QPE_QUBITS,
             'address': layout.address_zone_size(address, B),
             'flag': layout.FLAG_QUBITS}
    comp_step = layout.idle_except(comp, model, distances, zones, busy={'address': n_address})
    cost = acc.seq(comp_step,
                   layout.idle_except(hard_coded, model, distances, zones, busy={'address': n_b}),
                   comp_step)
    # Clean ancilla: the larger of the comparator's and model.maj_cost's
    # (seq's max), plus Maj's own 2, held throughout
    cost['Clean ancilla'] += 2

    return {**cost, 'Subroutine': 'Maj'}


def Ref(name: str, distances: dict, *, model, N: int, b_rot: int, X: int, B: int,
        busy: dict, flag: bool):
    '''
    Reflection about the qubits in `busy` ({zone: logical qubits}), controls
    included: every one of them is in the reflection, and so inside its
    footprint (model.ref over their total).

    Zones present: system register (2N), Fourier state (b_rot), QPE, address
    zone (X and B registers + 2 selection qubits), and the flag qubit when
    `flag` is True.  Everything present and not busy is idle.

      InnerRef  busy: the B register and 1 selection qubit;  flag present
      OuterRef  busy: the X and B registers, both selection qubits and the
                2 QPE qubits (as controls);  no flag
    '''
    zones = {'system': layout.system_zone_size(N),
             'fourier': layout.fourier_zone_size(b_rot),
             'QPE': layout.QPE_QUBITS,
             'address': layout.address_zone_size(X, B)}
    if flag:
        zones['flag'] = layout.FLAG_QUBITS
    cost = layout.idle_except(model.ref(sum(busy.values()), **distances), model, distances, zones, busy)

    return {**cost, 'Subroutine': name}

def RPrep_Rot(N: int, B: int, R: int, C: int, bit_precision: int, distances: dict, *,
              model, method: str, K: int, lam: int,
              runs: int = None, ancilla_budget: int = None,
              full_budget: int = None, buffer_budget: int = None):
    '''
    Flattened index preparation, rotation angle loading and Givens rotations,
    for one fully specified lookup table batching option.

    This routine applies the option it is handed and never searches for one,
    and has no default option: method, K and lam must always be given.
    Choosing an option is batching.py's job (batching_table / best_option, for
    the lookup table alone) and optimize.search_rprep_rot's (for whole
    routines); pass the choice in with `**bt.as_option(row)`.  Everything
    about what a method means -- which SelectSwap variety it uses, how many
    lookups it needs, where its swap ancilla come from -- is read off
    batching.batching_cost rather than re-derived here, so the two cannot
    drift apart.

    method:         batching method, one of batching.METHODS.
    K:              swap parameter, a power of two.  K=1 degenerates every
                    method to a plain skew-tree QROM.
    lam:            rotation angles loaded per batch; output register lam*b_rot.
    runs:           recycling runs.  Only meaningful for 'dirty_full'; None
                    lets batching.py derive it from the borrowable budget.
    ancilla_budget: the compute zone's size in logical qubits: the lookup's
                    clean ancilla plus the access corridors of its output
                    register (warned about when exceeded; the record carries
                    'Compute zone').  Also serves as the 'clean' method's
                    bound on the swap network.
    full_budget:    borrowable ancilla for 'dirty_full', default 2*N.
    buffer_budget:  the buffer zone's fixed size, checked but never used to
                    constrain the option -- it's a separate zone from the
                    compute zone `ancilla_budget` bounds.

    Zones present throughout: system register (2N), Fourier state (b_rot),
    QPE, address zone (ceil(log2 max(X,Y)) + n_B + 2), flag qubit (computed
    in Spin).  From each lookup, also its loaded angles: lam angles of b_rot
    qubits each, each measured out (model.mbu_cost of b_rot qubits) right
    after its own rotation.  An angle is in the output register ('output'
    zone) while the lookup loads it, while it is being moved and during its
    own addition, rotation and MBU, and while it waits its turn -- except
    the second angle, which is parked ('parked' zone: the rep-code routing
    under the system qubits in LDPC, the same rep-code qubits as 'output'
    in the rep code) from the shuffle until the first addition is done, to
    make room for that addition's layout.

    Steps, one after another; each lists what is BUSY (inside its own
    footprint), everything else present is idle:

      flattening (start, undone at the end):
        QC-add of 2**n_R (B-C) on |x>   busy: the n_x bits of x, the flag
                                              (its control)
        swap |c> <-> |b> (plain swap    busy: c and b (n_b bits each), the
          of n_b pairs)                       flag (its control)
      each of the J batches:
        lookup (SelectSwap over Y)      busy: the ceil(log2 Y) address bits
                                              it reads, all lam angles; for
                                              the dirty methods also the
                                              (K-1)*lam*b_rot system qubits
                                              it borrows
        shuffle (once per batch:        busy: one angle (b_rot qubits; they
          model.rot_shuffle_cost is           move one at a time); the
          all of the batch's moves)           Fourier state idles, the
                                              second angle is parked, the
                                              other lam-2 idle in the
                                              output register
        then for each angle k = 1..lam:
          addition k                    busy: the Fourier state, angle k;
                                              the lam-k still to come idle
                                              in the output register (for
                                              k = 1 the second one is
                                              parked); earlier ones are
                                              measured out already
          rotation k (2 parity-         busy: the Fourier state (control),
            controlled NOTs, 2d rounds)       1 system qubit; angle k and
                                              the lam-k still to come idle
                                              as in addition k
          MBU k (model.mbu_cost of      busy: angle k; the lam-k still to
            b_rot qubits)                     come idle as in addition k
      extra lookups (dirty methods), each:
        lookup                          as in a batch
        then for each of its lam        busy: that angle; the ones not yet
          angles: MBU                         measured out idle in the
                                              output register
      fixup over Y                      busy: the address bits it reads

    The adder acts on the qubits the lookup loaded, so space is the largest
    single step (seq), not a sum.
    '''
    n_r = int(np.ceil(np.log2(R)))
    n_b = int(np.ceil(np.log2(B+1)))    # the B register holds B+1 values
    X = N + 2**n_r*C
    Y = N + 2**n_r*B
    n_x = int(np.ceil(np.log2(max(X, Y))))

    option = bt.batching_cost(N, B, R, bit_precision, distances, method, lam, K, runs, model=model,
                              clean_budget=ancilla_budget, full_budget=full_budget)
    # K=1 degenerates every method to 'qrom', so read the method back rather
    # than trusting the one that was passed in
    method = option['method']
    variety = 'clean' if method in ('qrom', 'clean') else 'dirty'
    load = option['load']
    J = option['J']
    n_loads = option['lookups']
    dirty_budget = (bt.swap_budget(method, N, ancilla_budget, full_budget)
                    if variety == 'dirty' else None)

    if not option['adder ok']:
        warnings.warn(f"RPrep+Rot: {method} with K={load['K']}, lam={lam} violates "
                      f"the adder layout condition (lam >= 2, unless clean with K >= 2)")

    if not option['feasible']:
        warnings.warn(f"RPrep+Rot: {method} with K={K}, lam={lam} needs "
                      f"{option['swap ancilla needed']} swap ancilla but only "
                      f"{option['swap ancilla available']} are available")

    # the compute zone holds the lookup's clean ancilla and the access
    # corridors of its output register (batching's 'compute zone' condition)
    if not option['fits']:
        warnings.warn(f"RPrep+Rot: {method} with K={load['K']}, lam={lam} needs a compute "
                      f"zone of {option['Compute zone']} ({option['Clean ancilla']} clean "
                      f"ancilla + {option['Access corridors']} access corridors) but the "
                      f"budget is {ancilla_budget}")

    b_rot = bit_precision

    # ---- the cost of each step, from the model --------------------------
    # flattening: controlled quantum-classical add of 2**n_R (B-C) on |x>,
    # then a swap of the |c> and |b> registers (n_b pairs) controlled on the
    # flag.  Undone at the end
    qc_add = model.qc_add_cost(n_x, **distances)
    swap = model.PlainSWAP_cost(n_b, **distances)
    # one batch's shuffle: all of its moves, once per batch (the angles move
    # one at a time; the Fourier state is moved back each time).  N, n_x, n_b: the LDPC model
    # checks the angle storage fits under the system and address columns
    shuffle = model.rot_shuffle_cost(b_rot, lam, **distances, N=N, n_x=n_x, n_b=n_b)
    # per angle: its addition, its rotation (on the Fourier state and 1
    # system qubit) and its MBU (its b_rot qubits measured out)
    add = model.adder_cost(b_rot, **distances)
    rotation = model.rotation_cost(b_rot, **distances)
    mbu = model.mbu_cost(b_rot, **distances)
    # the fixup comes after every angle has been measured out, so those
    # lam*b_rot logical qubits are free and available to it even when no
    # separate clean budget was given
    fixup = model.phase_fixup_cost(Y, **distances, ancilla_budget=max(ancilla_budget or 0, lam*b_rot),
                                   buffer_budget=buffer_budget)

    # ---- the zones present, and what is idle beside each step -----------
    zones = {'system': layout.system_zone_size(N),
             'fourier': layout.fourier_zone_size(b_rot),
             'QPE': layout.QPE_QUBITS,
             'address': layout.address_zone_size(max(X, Y), B),
             'flag': layout.FLAG_QUBITS}

    def angles(live, parked=0):
        '''the zones present with `live` angles in the output register and
        `parked` angles waiting their turn'''
        return {**zones, 'output': live*b_rot, 'parked': parked*b_rot}

    flatten = [
        layout.idle_except(qc_add, model, distances, zones, busy={'address': n_x, 'flag': 1}),
        layout.idle_except(swap, model, distances, zones, busy={'address': 2*n_b, 'flag': 1}),
    ]

    # the lookup, with all lam angles live.  A dirty lookup's borrowed
    # system qubits are busy in it; an infeasible option (warned about
    # above) borrows more than the system holds, so only what is there can
    # be busy
    borrowed = min(load['Dirty ancilla'], zones['system'])
    lookup = layout.idle_except(load, model, distances, angles(lam),
                                busy={'address': int(np.ceil(np.log2(Y))), 'output': lam*b_rot,
                                      'system': borrowed})

    # ---- one batch: lookup, shuffle (once), then each angle in turn -----
    # Only the second angle is parked (moved out of the output register to
    # make room for the first addition's layout), and only until the first
    # addition is done; angles 3..lam wait in the output register
    parked = 1 if lam >= 2 else 0
    batch = [
        lookup,
        # one angle moves at a time (busy); the second is parked, the
        # other lam-2 idle in the output register
        layout.idle_except(shuffle, model, distances, angles(lam - parked, parked),
                           busy={'output': b_rot}),
    ]
    for k in range(1, lam + 1):
        # angle k and the lam-k still to come are in the output register,
        # except the second angle, parked during the first addition; the
        # k-1 before it are measured out already
        present = angles(lam - parked, parked) if k == 1 else angles(lam - k + 1)
        batch += [
            layout.idle_except(add, model, distances, present,
                               busy={'fourier': b_rot, 'output': b_rot}),
            layout.idle_except(rotation, model, distances, present,
                               busy={'fourier': b_rot, 'system': 1}),
            layout.idle_except(mbu, model, distances, present,
                               busy={'output': b_rot}),
        ]

    # ---- one extra lookup (dirty methods only): lookup, then its lam
    # angles measured out one after another (the k-th MBU with lam-k+1
    # angles still in the output register)
    extra = [lookup] + [layout.idle_except(mbu, model, distances, angles(lam - k + 1),
                                           busy={'output': b_rot})
                        for k in range(1, lam + 1)]

    # ---- the whole routine ---------------------------------------------
    cost = acc.seq(
        *flatten,
        acc.repeat(acc.seq(*batch), J),
        acc.repeat(acc.seq(*extra), n_loads - J),
        layout.idle_except(fixup, model, distances, zones,
                           busy={'address': int(np.ceil(np.log2(Y)))}),
        *reversed(flatten))

    # every step runs after the previous one, so their ancilla is a peak
    # (seq's max), not a sum.  dirty_budget is already None for the clean
    # varieties, so no second guard
    layout.check_feasibility(cost, 'RPrep+Rot', ancilla_budget=ancilla_budget,
                             dirty_budget=dirty_budget, buffer_budget=buffer_budget)

    return {**cost, 'Subroutine': 'RPrep+Rot', 'size of batches': lam, 'K': load['K'],
            'method': method, 'runs': option['runs'], 'variety': variety,
            'lookups': n_loads, 'J': J, 'Compute zone': option['Compute zone'],
            'Access corridors': option['Access corridors'],
            # why the option is not viable (batching.py), () when it is
            'rejected': option['rejected']}


def BE_cost(N, B, R, C, b_coef, b_rot, ancilla_budget, distances, *,
            model, method, K, lam, outer_variety, inner_variety, runs=None,
            buffer_budget=None, full_budget=None):
    '''
    SOS block encoding.  Composite indices use the padded R register, so
    X = N + 2**n_R C and Y = N + 2**n_R B.

    method, K,
    lam, runs:     RPrep+Rot's lookup table batching option, passed straight
                   through (see RPrep_Rot and batching.py).  Required, no
                   default -- pass batching's choice with **bt.as_option(row).
                   Named explicitly rather than collected in **kwargs so that
                   what is tunable is visible from this signature.
    outer_variety,
    inner_variety: the SelectSwap variety of OuterPrep's and InnerPrep's
                   lookups (both calls of each, forward and adjoint), 'clean'
                   or 'dirty'.  Required, no default -- choose it beforehand,
                   e.g. with optimize.search_prep_variety, as for RPrep+Rot's
                   option.  The prep still picks K per level (fewest rounds,
                   up to its cap).  'dirty' borrows up to 2N system qubits
                   (the preps' dirty_budget), so it needs Z access to them
                   (model.SYSTEM_Z_ACCESS); a model without it refuses it.
    buffer_budget: the buffer zone's fixed size.  Passed explicitly to every
                   subroutine call below (rather than left to each one's own
                   default) so they all check against the same zone size.
    '''
    for name, variety in (('outer_variety', outer_variety), ('inner_variety', inner_variety)):
        if variety not in ('clean', 'dirty'):
            raise ValueError(f"{name} must be 'clean' or 'dirty', got {variety!r}")
        if variety == 'dirty' and not model.SYSTEM_Z_ACCESS:
            raise ValueError(f"{name}='dirty' borrows system qubits as swap ancilla, but this "
                             f"model has no Z access to them (model.SYSTEM_Z_ACCESS)")
    n_r = int(np.ceil(np.log2(R)))
    X = N + 2**n_r*C
    n_X = int(np.ceil(np.log2(X)))
    n_B = int(np.ceil(np.log2(B+1)))

    # N, b_rot and the X / B register sizes give each routine its zones, so
    # every idle qubit is counted; each prep's variety is the one given
    outer = OuterPrep(X, b_coef, ancilla_budget, distances, model=model, buffer_budget=buffer_budget,
                      variety=outer_variety, N=N, b_rot=b_rot, B=B)
    outer_dag = OuterPrep(X, b_coef, ancilla_budget, distances, model=model, adjoint=True,
                          buffer_budget=buffer_budget, variety=outer_variety,
                          N=N, b_rot=b_rot, B=B)
    inner = InnerPrep(2**n_r*C, B, b_coef, ancilla_budget, distances, model=model,
                      buffer_budget=buffer_budget, variety=inner_variety,
                      N=N, b_rot=b_rot, X=X)
    inner_dag = InnerPrep(2**n_r*C, B, b_coef, ancilla_budget, distances, model=model, adjoint=True,
                          buffer_budget=buffer_budget, variety=inner_variety,
                          N=N, b_rot=b_rot, X=X)
    spin = Spin(X, N, distances, b_rot=b_rot, B=B, model=model)
    maj = Maj(X, B, distances, model=model, N=N, b_rot=b_rot)
    rot = RPrep_Rot(N, B, R, C, b_rot, distances, model=model, method=method, K=K, lam=lam, runs=runs,
                    ancilla_budget=ancilla_budget, full_budget=full_budget,
                    buffer_budget=buffer_budget)
    # InnerRef: about B and 1 selection qubit, the flag is present.
    # OuterRef: about X, B, both selection qubits and the 2 QPE qubits (as
    # controls), the flag is uncomputed by then
    inner_ref = Ref('InnerRef', distances, model=model, N=N, b_rot=b_rot, X=X, B=B,
                    busy={'address': n_B + 1}, flag=True)
    outer_ref = Ref('OuterRef', distances, model=model, N=N, b_rot=b_rot, X=X, B=B,
                    busy={'address': n_X + n_B + 2, 'QPE': layout.QPE_QUBITS}, flag=False)

    # calls per block encoding, per the subroutine table
    calls = [(outer, 1), (outer_dag, 1), (inner, 2), (inner_dag, 2),
             (spin, 4), (maj, 2), (rot, 4), (inner_ref, 1), (outer_ref, 1)]

    # every subroutine call within one block encoding runs one after another
    # (never simultaneously): time and volume add, space and ancilla are the
    # largest single call
    be = acc.seq(*(acc.scale(p, n) for p, n in calls))

    layout.check_feasibility(be, 'BE', ancilla_budget=ancilla_budget,
                             buffer_budget=buffer_budget)

    # per subroutine: its name, its calls per block encoding, and its rounds,
    # Toffolis and i-states PER CALL (multiply by 'calls' for the block
    # encoding's share), for the report
    breakdown = [dict(name=p['Subroutine'], calls=n, rounds=acc.total(p, 'rounds'),
                      toffoli=acc.total(p, 'toffoli'), **{'i-states': acc.total(p, 'i-states')})
                 for p, n in calls]
    # (name, {layer: space}, {layer: the step that sets it}), per routine,
    # for the report
    space_breakdown = [(p['Subroutine'], p['space'], p['peak']) for p, _ in calls]
    return {**be, 'Subroutine': 'BlockEncoding', 'breakdown': breakdown,
            'space breakdown': space_breakdown}


def Total_time(N, B, R, C, b_coef, b_rot, ancilla_budget, buffer_budget, distances, be_const, *,
               epsilon, round_time, model, method, K, lam, outer_variety, inner_variety,
               runs=None, full_budget=None):
    '''
    Phase estimation total: calls = ceil(pi*be_const/(2*epsilon)) block
    encodings, with `be_const` the effective block-encoding constant from SOS
    and `epsilon` the target precision, both in Ha.  round_time is the
    duration of one round, in seconds.  epsilon and round_time are required,
    no default.

    method, K, lam, runs: RPrep+Rot's batching option, passed through BE_cost.
                          Required, no default.
    outer_variety,
    inner_variety:        the preps' SelectSwap variety, passed through
                          BE_cost.  Required, no default.

    Returns, besides 'calls' and the totals of rounds, Toffolis and i-states:
      'total days'             the run time, rounds x round_time
      'total logical cycles'   rounds / model.rounds_per_cycle(**distances)
      'layers'                 the layers the block encoding uses, 'rep' first
      per layer L, in L's own unit (unsuffixed: the 'rep' layer, as in
      accounting.summary; ' [L]' for every other layer):
        'total qubit cycles', 'total idle cycles', 'Spatial cost'
      per layer L, including 'rep':
        'total logical qubit cycles [L]'   model.logical_qubit_cycles of L's
                                           volume over all calls
      'total effective volume'  the sum of those over all layers -- the
                                quantity the error budget is set against
    '''
    calls = int(np.ceil(np.pi*be_const/(2*epsilon)))
    cost = BE_cost(N, B, R, C, b_coef, b_rot, ancilla_budget, distances, model=model,
                   method=method, K=K, lam=lam, runs=runs,
                   outer_variety=outer_variety, inner_variety=inner_variety,
                   buffer_budget=buffer_budget, full_budget=full_budget)
    be = acc.summary(cost)
    layers = acc.layers(cost)

    out = {'calls': calls,
           'total rounds': calls*be['Temporal cost'],
           'total logical cycles': calls*be['Temporal cost']/model.rounds_per_cycle(**distances),
           'total days': calls*be['Temporal cost']*round_time/(3600*24),
           'total Toffoli': calls*be['Toffoli cost'],
           'total i-state': calls*be['i-state cost'],
           'layers': layers,
           # space x time over the whole phase estimation, idle qubits
           # included, in the 'rep' layer's own unit
           'total qubit cycles': calls*be['Qubit cycles'],
           'total idle cycles': calls*be['Idle cycles'],
           'Spatial cost': be['Spatial cost'],
           'Clean ancilla': be['Clean ancilla'],
           'Dirty ancilla': be['Dirty ancilla'],
           'Buffer ancilla': be['Buffer ancilla']}
    for layer in layers:
        if layer != 'rep':
            out[f'total qubit cycles [{layer}]'] = calls*be[f'Qubit cycles [{layer}]']
            out[f'total idle cycles [{layer}]'] = calls*be[f'Idle cycles [{layer}]']
            out[f'Spatial cost [{layer}]'] = be[f'Spatial cost [{layer}]']
    # each layer converted to logical qubit-cycles first, then added
    logical = {layer: calls*model.logical_qubit_cycles(layer, acc.volume(cost, layer), **distances)
               for layer in layers}
    for layer, v in logical.items():
        out[f'total logical qubit cycles [{layer}]'] = v
    out['total effective volume'] = sum(logical.values())
    return out

def _primitive_table(parts, field, total, unit, width, model, col_width=14):
    '''
    Print one {'operation': share} table of `parts`, with `field(part)` the
    value per part (its rounds, its volume in one layer or its i-states).
    A part 'operation:step' is a step inside an operation and counts in that
    operation's row (e.g. 'fixup:hadamard' under 'fixup').
    Parts named in model.FALLBACK are timed at TOFF_TIME*d per Toffoli rather
    than from a real compilation; their operation's row is marked with a
    star, and the footnote counts those parts only (not the compiled steps
    in the same row).
    '''
    rows, fallback = {}, 0
    for name, p in parts.items():
        operation = name.split(':')[0]
        rows[operation] = rows.get(operation, 0) + field(p)
        if name in model.FALLBACK:
            fallback += field(p)
    print(f"{'operation':<18}{'':>6}{unit:>{col_width}}{'share':>8}")
    print('-'*width)
    for operation, val in sorted(rows.items(), key=lambda kv: -kv[1]):
        # idle categories carry qubit-cycles only, so they would show as a
        # row of zeros in the rounds and i-state tables
        if val == 0:
            continue
        star = '*' if operation in model.FALLBACK else ''
        share = 100*val/total if total else 0
        print(f"{operation + star:<18}{'':>6}{val:>{col_width}.0f}{share:>7.1f}%")
    print('-'*width)
    print(f"{'total':<18}{'':>6}{total:>{col_width}.0f}{100:>7.1f}%")
    if fallback:
        print(f"* not compiled, timed at {model.TOFF_TIME}d per Toffoli: "
              f"{fallback:.0f} {unit}, {100*fallback/total:.1f}%")


def _step_table(parts, total, width):
    '''
    Rounds of the steps inside operations ('operation:step'), by step type
    across every operation -- e.g. all Hadamards, wherever they are applied.
    These rounds are already counted in their operations' rows above.
    '''
    steps = {}
    for name, p in parts.items():
        if ':' in name:
            operation, step = name.split(':', 1)
            rounds, inside = steps.get(step, (0, set()))
            steps[step] = (rounds + p['rounds'], inside | {operation})
    if not steps:
        return
    print("steps inside operations, by type (already counted in their operations' rows)")
    print(f"{'step':<12}{'rounds':>12}{'share':>8}   inside")
    print('-'*width)
    for step, (rounds, inside) in sorted(steps.items(), key=lambda kv: -kv[1][0]):
        share = 100*rounds/total if total else 0
        print(f"{step:<12}{rounds:>12.0f}{share:>7.1f}%   {', '.join(sorted(inside))}")
    print('-'*width)


def report(be, *, model, width=44):
    '''
    Breakdown of a BE_cost result: by subroutine (rounds), and by operation
    on rounds, qubit-cycles (space x time, the additive space-impact metric),
    i-states and Toffolis.  Steps inside an operation ('fixup:hadamard')
    count in that operation's row; a separate table lists them by step type
    (all Hadamards, all reaction waits, ...).
    '''
    s = acc.summary(be)
    total = s['Temporal cost']

    print(f"{'subroutine':<18}{'calls':>6}{'rounds':>12}{'share':>8}")
    print('-'*width)
    for r in sorted(be['breakdown'], key=lambda r: -r['calls']*r['rounds']):
        rounds = r['calls']*r['rounds']
        print(f"{r['name']:<18}{r['calls']:>6}{rounds:>12}{100*rounds/total:>7.1f}%")
    print('-'*width)
    print(f"{'total':<18}{'':>6}{total:>12}{100:>7.1f}%")

    print()
    _primitive_table(be['parts'], lambda p: p['rounds'], total, 'rounds', width, model, col_width=12)

    print()
    _step_table(be['parts'], total, width)

    # one qubit-cycle table per layer, each in that layer's own unit
    for layer in acc.layers(be):
        print()
        if layer != 'rep':
            print(f"layer '{layer}' (its own unit)")
        total_qc = acc.volume(be, layer)
        _primitive_table(be['parts'], lambda p: p['volume'].get(layer, 0), total_qc,
                         'qubit-cyc', width, model)
        idle = acc.idle_volume(be, layer)
        if total_qc:
            print(f"idle qubits (space-time volume, no rounds): "
                  f"{idle:.0f} qubit-cyc, {100*idle/total_qc:.1f}%")

    print()
    _primitive_table(be['parts'], lambda p: p['i-states'], s['i-state cost'], 'i-states', width, model)

    print()
    _primitive_table(be['parts'], lambda p: p['toffoli'], s['Toffoli cost'], 'Toffolis', width, model)

    for layer in acc.layers(be):
        print()
        _space_table(be['space breakdown'], be['space'].get(layer, 0), layer)


def _space_table(space_breakdown, be_space, layer):
    '''
    For each routine with space in `layer`: its space, the step that sets
    it, and that step's own footprint plus the idle zones beside it (in the
    layer's own unit).  The largest routine sets the block encoding's space.
    '''
    width = 86
    if layer == 'rep':
        print("space per routine (physical qubits; excludes the unused part of the buffer zone)")
    else:
        print(f"space per routine in layer '{layer}' (its own unit)")
    print(f"{'routine':<15}{'space':>8}  {'set by':<12}{'footprint':>10}{'idle':>8}   idle by zone")
    print('-'*width)
    rows = [(name, space[layer], peak[layer]) for name, space, peak in space_breakdown
            if layer in space]
    seen = set()
    for name, space, peak in sorted(rows, key=lambda r: -r[1]):
        if name in seen:
            continue
        seen.add(name)
        idle = sum(peak['idle'].values())
        zones = ', '.join(f"{z} {q:,.0f}" for z, q in peak['idle'].items())
        mark = '  <- block encoding' if space == be_space else ''
        print(f"{name:<15}{space:>8,.0f}  {peak['step']:<12}{peak['footprint']:>10,.0f}"
              f"{idle:>8,.0f}   {zones}{mark}")
    print('-'*width)


if __name__ == '__main__':
    # a demo: FeMoCo-54 in the rep-code model, with the paper's settings at p_Z = 5e-4
    from resource_count import settings as st
    from resource_count import error_accounting as ea
    system, s = st.FEMOCO_54, st.SETTINGS_54[st.rep_code]
    args = [system[k] for k in ('N', 'B', 'R', 'C', 'b_coef', 'b_rot')]
    distances = s['distances']['5e-4']
    report(BE_cost(*args, s['budget'], distances, model=st.rep_code,
                   buffer_budget=layout.buffer_zone_size(), **s['option']),
           model=st.rep_code)
    print(Total_time(*args, s['budget'], layout.buffer_zone_size(), distances, system['be_const'],
                     epsilon=ea.EPSILON, round_time=ea.ROUND_TIME, model=st.rep_code, **s['option']))
