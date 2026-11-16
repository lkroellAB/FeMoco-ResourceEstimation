"""
Cost functions for relevant subroutines, in the repetition-code model.

Every primitive returns a cost record (see accounting.py).  Each step is an
`acc.op(name, rounds, space, ...)`, and steps that run one after another are
combined with `acc.seq`.

Conventions
-----------
Repetition-code (cat qubit) model: a logical patch is d physical qubits tall and
one wide, so footprints scale linearly in d.  Routing zones are a fixed number
of rows tall and do not scale with d.

space            {'rep': physical qubits}, stabilizer ancilla included.
                 Formulas count data qubits and are doubled (2*space);
                 footprints made of zone qubits use layout.physical (2d-1
                 per logical qubit), which already includes the ancilla.
                 A footprint contains only the buffer slots that step uses
                 itself.  The buffer zone is shared and sized for its largest
                 user (layout.buffer_zone_size()), so a routine's space
                 EXCLUDES the zone's unused slots: it is not the machine
                 size, which is computed by hand.  Unused buffer slots are
                 not idle qubits, so qubit-cycles never include them.
rounds           syndrome extraction rounds.
t_CXCXX          rounds to inject one |C_XC_XX> state (default 21, a parameter
                 of the primitives that use it), so a Toffoli step takes
                 max(d, t_CXCXX) rounds.
toffoli          CCZ / C_XC_XX consumptions (high-level quantity), on the
                 part that consumes them.
i-states         |i> state injections. If rough estimation 4*Toffoli (2 Hadamard gates using 4 fresh i states with 2 catalysts), changes if compilation available and
                 if additional Hadamard gates are used
'Clean ancilla'  logical qubits that must be available and clean, excluding the
                 address register.  Used for budget checking.
'Dirty ancilla'  logical qubits borrowed from some other register (e.g. the
                 system qubits) for the duration of the routine and restored
                 afterwards.  Does not draw from the clean ancilla budget.
                 Zero for routines that never borrow.

Where no compiled number for time is available the fallback is TOFF_TIME * d rounds per
Toffoli.

Code distance
-------------
One distance, d_rep (written d in the formulas' comments).  high_level_cost.py
passes a model's distances as a dict, unpacked into keywords, so this model
is called with distances = dict(d_rep=...) and refuses any other key (a
TypeError, e.g. for a d_LDPC meant for the hybrid model).

Zones
-----
Every zone (system, address, Fourier state, QPE, flag, a lookup's loaded
output) lives in the rep-code layer, counted in physical qubits.  `ZONE_LAYER` and `zone_qubits` are this
model's answer to "where does a zone live, and in what unit"; a different
cost model (e.g. LDPC) defines its own.  `logical_qubit_cycles` converts each
layer's volume to logical qubit-cycles, and `rounds_per_cycle` gives the
rounds in one logical cycle.

Access corridors and system access
----------------------------------
A rep-code logical qubit has X access only; a qubit that needs Z access needs
an access corridor next to it, one corridor per two qubits
(`access_corridors`).  A lookup's output register needs Z access, so its
corridors sit in the compute zone next to it and count against the compute
zone's budget.

The chosen layout has NO access corridors for the system qubits
(`SYSTEM_Z_ACCESS = False`), so nothing can borrow them as dirty swap
ancilla: the dirty RPrep+Rot batching methods and the preps' dirty SelectSwap
variety are ruled out by the layout.  batching.py still reports the
corridors they would need under the system ('System corridors') to show why.
"""

import numpy as np

from resource_count import accounting as acc
from resource_count import layout

TOFF_TIME = 3       # fallback rounds per Toffoli, in units of d
T_REACT = 10        # reaction time in rounds, independent of d
I_COUNT = 4         # Number of i-states for autocorrected brute force C_X C_X X to C_X C_X Z injection (2 for injection 2 for correction)
#Catalysis state gets reset between operations (adds `buff` i states per operation)

# categories whose timing is a TOFF_TIME*d placeholder rather than a real
# compilation, so the share of the runtime that is still guesswork is visible.
FALLBACK = ('comparator', 'reflection', 'maj', 'qc-add')

# the layout has no access corridors for the system qubits, so they cannot be
# borrowed as dirty swap ancilla (see the module docstring)
SYSTEM_Z_ACCESS = False

# which layer each zone lives in: all rep code in this model.  'output' is a
# lookup's loaded output register, from the lookup until it is measured out;
# 'parked' holds RPrep+Rot's loaded angles while they wait their turn (see
# high_level_cost.RPrep_Rot)
ZONE_LAYER = {'system': 'rep', 'address': 'rep', 'fourier': 'rep',
              'QPE': 'rep', 'flag': 'rep', 'output': 'rep', 'parked': 'rep'}


def zone_qubits(zone, logical, d_rep):
    '''
    (layer, qubits) for `logical` qubits of `zone`, with qubits in that
    layer's own unit: physical qubits for 'rep'.
    '''
    layer = ZONE_LAYER[zone]
    return layer, layout.physical(logical, d_rep)


def rounds_per_cycle(d_rep):
    '''Rounds in one logical cycle: d_rep.'''
    return d_rep


def access_corridors(qubits):
    '''
    Access corridors (logical slots) for `qubits` logical qubits that need Z
    access: one corridor per two qubits.
    '''
    return int(np.ceil(qubits/2))


def logical_qubit_cycles(layer, qubit_cycles, d_rep):
    '''
    Logical qubit-cycles (the effective volume) of `qubit_cycles` in `layer`,
    given in that layer's own unit.  'rep': physical qubit-rounds / (d(2d-1)):
    a logical qubit is 2d-1 physical qubits, and a logical cycle d rounds.
    This model has no other layer, so any other raises.
    '''
    if layer != 'rep':
        raise ValueError(f"the rep-code model has no layer {layer!r}")
    return qubit_cycles/(d_rep*(2*d_rep - 1))


def qrom_ancilla(address,buff: int = 2):
    '''
    Clean ancilla used by the skew-tree QROM over `address` addresses: its
    own working ancilla plus a fixed buffer (1 buffer equals 6 qubits). We assume that 
    buffer and clean ancilla sit side-by-side in the buffer zone
    '''
    address = int(np.ceil(address))
    if address <= 1:
        return 0
    return max(int(np.ceil(np.log2(address))) - 1, 0) + buff*6


def QROM_cost(address: int, target_bit: int, d_rep: int, route: int = 3, buff: int = 2, t_CXCXX = 21):
    '''
    Skew-tree QROM with classically absorbed corrections.
    Assumes C_XC_XX production rate of at least one per d rounds.
    A single-address table is a classical constant: Pauli only, zero cost.
    The target register requires Z access, which are NOT counted as clean ancilla
    '''
    # callers pass np.ceil(...) for a split address, which is an integral float;
    # coerce here so 'Toffoli cost'/'i-state cost' stay ints rather than
    # leaking np.float64 into every total downstream
    address = int(np.ceil(address))

    if address <= 1:
        return acc.empty()

    n_address = int(np.ceil(np.log2(address)))
    space = int((2*n_address + 3*np.ceil(target_bit/2))*(d_rep + route + 1) + buff*6*d_rep)
    time = int((address-1)*d_rep) + int((np.ceil(address/2) - n_address)*max(d_rep, t_CXCXX))

    # 2**x - x - 1 Ands for M = 2**x addresses (x address bits), 2 i-states each
    ands = address - n_address - 1
    i_state = ands*2 + buff #No C_X X phase correction, Hadamard with catalysis

    # clean = target_bit: the output register itself, fresh fungible
    # allocation -- not allocatable into buffer zone.  The n_address - 1
    # working ancilla sit in the 2*n_address term of `space` (they are also
    # in the buffer count, qrom_ancilla), so the volume includes them
    return acc.op('qrom', time, {'rep': 2*space},
                  toffoli=ands, i_states=i_state,
                  clean=target_bit, buffer=qrom_ancilla(address))


def adder_cost(bit: int, d_rep: int, buff: int = 3, t_CXCXX: int = 21):
    '''
    Quantum-quantum addition.
    Spatial cost counts data qubits, so it is doubled like every other footprint.
    The adder compilation needs bit free logical slots in addition to the summands.
    Time is a simplified count, 2*bit steps of max(d_rep, t_CXCXX) rounds: the
    reaction time wait and the MBU round are part of it, not tracked as steps
    of their own.
    '''
    if bit <= 0:
        return acc.empty()

    space = (d_rep + 4)*(3*bit + 9*buff) + 6*4 #Shifted buffer below: formula from Pappalardo et al., arXiv:2610.37839, plus 6*4 because the buffer sits further right in this layout
    t_surg = d_rep #One logical cycle (ignoring initialize and measurement)
    t_cad = max(t_surg, t_CXCXX)
    time = 2*bit*t_cad 

    i_state = 4*bit + buff   #I_COUNT = 4 per Toffoli (2 for the injection, 2 for the phase correction), plus the buff catalysis states

    # one step, the whole addition (reaction time and MBU included).  9
    # buffer slots per buff; layout.buffer_zone_size matches
    return acc.seq(
        acc.op('adder', time, {'rep': 2*space}, toffoli=bit, i_states=i_state,
               clean=bit, buffer=9*buff))


def qc_comp_cost(bit: int, d_rep: int):
    '''Quantum-classical comparison with 1 output ancilla.  Fallback timing.'''
    if bit <= 1:
        return acc.empty()
    buff = 2
    space = 4*d_rep*bit + 13*buff*d_rep
    time = TOFF_TIME*(bit - 1)*d_rep

    i_state = I_COUNT*(bit-1) + buff
    return acc.op('comparator', time, {'rep': 2*space}, toffoli=bit - 1,
                  i_states=i_state, clean=bit - 1, buffer = 13*buff)


def qc_add_cost(bit: int, d_rep: int):
    '''Quantum-classical addition.  Fallback timing.'''
    if bit <= 1:
        return acc.empty()

    buff = 2
    space = 2*d_rep*(2*bit - 1) + 13*buff*d_rep
    time = TOFF_TIME*(bit - 1)*d_rep

    i_state = I_COUNT*(bit-1) + buff
    return acc.op('qc-add', time, {'rep': 2*space}, toffoli=bit - 1,
                  i_states=i_state, clean=bit - 1, buffer = 13*buff)


def SWAP_cost(control: int, swapands: int, d_rep: int, t_CXCXX = 21, buff = 3, qubits: int = None):
    '''
    Controlled-SWAP network. Compiled with storing parity-CNOT corrections
    swapands: the Ctrl-Swaps (one Toffoli each), which set the time, the
    Toffolis and the i-states.
    qubits: the swapped qubits the network holds, which set its space.
    Default swapands; a SelectSwap over K registers of b qubits has
    (K-1)*b Ctrl-Swaps but holds K*b qubits.
    buff: the catalysis states.  With the 3 qubits storing the parity-CNOT
    corrections they take 8*buff + 3 buffer slots, and as they are reset
    between operations each adds one i-state.  i-states: 4 per Toffoli,
    buff, and `control` for the S corrections (worst case).
    '''
    if control <= 0 or swapands <= 0:
        return acc.empty()
    qubits = swapands if qubits is None else qubits

    # 8*buff buffer slots plus the 3 qubits storing the corrections, all in
    # the buffer zone's d rows
    space = (control+np.ceil(qubits*3/2))*(d_rep+4)+(8*buff+3)*d_rep
    time = swapands*max(d_rep, t_CXCXX)+T_REACT+(control+1)*d_rep

    i_state = 4*swapands + buff + control #Ctrl-Swap injection with catalysis + worst case S correction
    return acc.op('swap', time, {'rep': 2*space}, toffoli=swapands,
                  i_states=i_state, buffer=8*buff+3)

def PlainSWAP_cost(pairs: int, d_rep: int, t_CXCXX = 21, buff = 3):
    '''
    Swap of `pairs` pairs of qubits (2*pairs qubits), controlled on 1 qubit
    (taken into account in the space formula).  One Toffoli per pair.
    Called by:
      Spin             the 2N system qubits in N pairs, controlled on the
                       spin qubit
      RPrep_Rot        the flattening's swap of the |c> and |b> registers,
                       n_b pairs, controlled on the flag
    buff: the catalysis states, as in SWAP_cost: 8*buff + 3 buffer slots
    and buff i-states.  i-states: 4 per Toffoli, buff, and 1 for the S
    correction (one control qubit).
    '''
    if pairs <= 0:
        return acc.empty()
    space = (1 + 2*pairs)*(d_rep+4)+(8*buff+3)*d_rep
    time = pairs*max(d_rep, t_CXCXX)+T_REACT+2*d_rep  + pairs #Only one possible control qubit for S correction, pairs contribution from wandering access corridor
    i_state = 4*pairs + buff + 1 #last is for the S gate correction

    # the 3 correction qubits are buffer slots, as in SWAP_cost
    return acc.op('PlainSwap', time, {'rep': 2*space}, toffoli=pairs,
                  i_states=i_state, buffer=8*buff+3)

def SelectSwap_cost(address: int, target_bit: int, alpha: int, d_rep: int,
                    variety: str = 'clean', batched: bool = False,
                    n_select: int = None, n_swap: int = None):
    '''
    SelectSwap lookup table.  The Select part is the skew-tree QROM.

    alpha:     log2 of the swap parameter K, so K = 2**alpha.
    variety:   'clean'  K-1 extra copies of the output register are clean ancilla
               'dirty'  swap ancilla are borrowed and must be restored, so the
                        Select and Swap each run twice and the output register
                        is conjugated by a Hadamard layer
    batched:   True when this lookup is one of a run of consecutive lookups
               (recycling lookup table batching).  For the dirty variant
               consecutive lookups merge, so the per-batch cost drops to
               1 Select + 2 Swap (plus the Hadamard conjugation), and each
               recycling run
               pays one extra such lookup at its end (the extra lookups in
               RPrep_Rot, counted by batching.batching_cost).  Clean lookups
               measure their swap registers out, so nothing merges and
               batching does not change the per-lookup cost.

    Select / Swap application counts:
                        unbatched      batched
        clean           1 S + 1 Sw     1 S + 1 Sw   (J lookups)
        dirty           2 S + 4 Sw     1 S + 2 Sw   (J + runs lookups)

    n_select,
    n_swap:    override those counts explicitly.
    '''

    Alpha = 2**alpha
    labels = {'variety': variety, 'batched': batched, 'K': Alpha}

    if Alpha == 1:
        # No swap network, so nothing is borrowed and nothing has to be
        # restored: a plain QROM regardless of variety or batching.  n_swap is
        # meaningless here (there is no Swap to apply); n_select still applies.
        ns = 1 if n_select is None else n_select
        return {**acc.repeat(QROM_cost(address, target_bit, d_rep), ns), **labels}

    Select = QROM_cost(np.ceil(address/Alpha), Alpha*target_bit, d_rep)
    # (Alpha-1)*target_bit Ctrl-Swaps over all Alpha registers
    Swap = SWAP_cost(alpha, (Alpha - 1)*target_bit, d_rep, qubits=Alpha*target_bit)

    counts = {('clean', False): (1, 1), ('clean', True): (1, 1),
              ('dirty', False): (2, 4), ('dirty', True): (1, 2)}
    if (variety, batched) not in counts:
        raise ValueError(f"unknown variety {variety!r}")
    ns, nw = counts[(variety, batched)]
    ns = ns if n_select is None else n_select
    nw = nw if n_swap is None else n_swap
    steps = [acc.repeat(Select, ns), acc.repeat(Swap, nw)]

    # the swap network's buffer slots that the QROM's buffer region does not
    # hold (0 once the address is large enough), in d rows, doubled
    extra_buffer = 2*max(Swap['Buffer ancilla'] - qrom_ancilla(address), 0)*d_rep

    # the QROM's layout over all K copies of the output register, plus the
    # swap network's extra buffer slots.  For 'dirty' the K-1 extra copies
    # are borrowed system qubits: busy in the lookup, so inside its footprint
    space = QROM_cost(address, Alpha*target_bit, d_rep)['space']['rep'] + extra_buffer

    if variety == 'clean':
        # measure out the K-1 copies the swap network did not select (no
        # phase fixup of its own)
        steps.append(mbu_cost((Alpha - 1)*target_bit, d_rep))
        # all K copies of the output register are fresh/clean.  Not
        # Select['Clean ancilla']: with K >= address the Select has one
        # address, a classical constant that QROM_cost counts as empty
        anc = Alpha*target_bit
        dirty_anc = 0
    else:
        # swap ancilla are dirty, so they do not enter the clean budget.
        # Select's own output register is still Alpha*target_bit wide (the
        # skew-tree structure doesn't change with variety), but only the
        # base target_bit copy is freshly clean here -- the other
        # (Alpha-1)*target_bit is dirty-sourced, so Select['Clean ancilla']
        # would overstate it; keep this one explicit rather than delegating.
        anc = target_bit
        # a Hadamard layer over the output register on either side,
        # conjugating it; the layer is the same width whatever alpha is (and
        # alpha >= 1 here, since Alpha == 1 returned above).  Two Hadamards
        # at a time, d cycles per pair with catalysis, ceil(target_bit/2)
        # pairs per layer: 2 i-states per Hadamard and 2 fresh catalysts
        # per layer
        steps.append(acc.op('swap:hadamard', 2*int(np.ceil(target_bit/2))*d_rep, {'rep': space},
                            i_states=2*2*target_bit + 4))
        # the K-1 extra copies of the output register are borrowed from
        # elsewhere (e.g. the system qubits) instead of the clean budget
        dirty_anc = (Alpha - 1)*target_bit

    cost = acc.seq(*steps)
    # LAYOUT OVERRIDES (seq would take the larger of Select's and Swap's):
    # the SelectSwap layout has its own footprint, computed above
    acc.set_space(cost, {'rep': space}, 'SelectSwap')
    # and its own clean / dirty ancilla, also above: the clean ancilla come
    # only from the Select.  Swap's buffer slots (its 3 correction qubits
    # included) sit in the buffer part next to the QROM's ancilla; the ones
    # that do not fit are extra_buffer
    cost['Clean ancilla'], cost['Dirty ancilla'] = anc, dirty_anc
    # Buffer ancilla: seq's max is right -- Select and Swap share the same
    # physical buffer zone (sized to whichever needs more) and run one after
    # another
    return {**cost, **labels}


def phase_fixup_cost(address: int, d_rep: int, ancilla_budget: int = None, buffer_budget: int = None,
                     t_react: float = T_REACT, t_CXCXX: float = 21, split: str = None):
    '''
    Phase fixup, with the n-bit address split as n = low + high.
 
    split='space-space':  one Toffoli per product of at least two address
        bits within each half, 2**high - high - 1 + 2**low - low - 1 in all,
        each max(d, t_CXCXX) rounds; min(2**low, 2**high) - 1 Hadamards (full
        rank), d rounds each; 0.5d rounds per product of the high half; and
        high - 1 reaction times.  Its footprint: the products (its clean
        ancilla), the address bits they read and 6*buff buffer slots, 2d
        each.  A single step, no sequential recursion.
 
    split='space-time': One-hot encoding over
        `high` bits (SWAP network) requiring 2**high ancilla and a Select over 2**low requiring buffer space, applied
        sequentially until all phases are resolved.  No MBU, so the one-hot
        register is uncomputed by running the SWAP network backwards.
 
    split=None (default):  the faster split (fewer rounds) of those that fit:
        when the space-space split's clean ancilla fit `ancilla_budget`, both
        are costed and the one with fewer rounds is returned (space-space on
        a tie); otherwise the space-time split.  A budget of None means no
        clean ancilla are available for the fixup, so the space-time split is
        used.
    '''
    address = int(np.ceil(address))
    if address <= 2:
        return acc.empty()
 
    n_address = int(np.ceil(np.log2(address)))
    low = int(np.floor(n_address/2))
    high = n_address - low
 
    if split is None:
        #Do not need Z access for space-space (see the paper)
        # its clean ancilla: the products, as in the space-space branch
        space_space_anc = 2**high + 2**low - n_address - 2
        if ancilla_budget is None or space_space_anc > ancilla_budget:
            split = 'space-time'
        else:
            # both fit: cost both, keep the one with fewer rounds (min keeps
            # the first, space-space, on a tie)
            options = [phase_fixup_cost(address, d_rep, ancilla_budget=ancilla_budget,
                                        buffer_budget=buffer_budget, t_react=t_react, t_CXCXX=t_CXCXX, split=s)
                       for s in ('space-space', 'space-time')]
            return min(options, key=lambda r: acc.total(r, 'rounds'))
 
    if split == 'space-space':
        buff = 2
        toff = 2**high + 2**low - n_address-2     # one Toffoli per product of at least two bits
        hadamard = min(2**low, 2**high)-1 #Full rank
        clean_ancilla = toff
        # the clean ancilla, the address bits the fixup reads and the buffer slots
        space = d_rep*(clean_ancilla + n_address)+6*buff*d_rep

        injection_i_state = 2*toff + buff #Power product is implemented with wandering AND so no phase correction
        correction_i_state = 2*int(np.ceil(0.5*toff)) + 1 #Rough counting (half the corrections on average)
        hadamard_i_state = 2*hadamard + 1 #Assuming 1 catalyst

        time = (2**high + 2**low - (high + low + 2))*max(d_rep, t_CXCXX) + (2**low-1)*d_rep + int(np.ceil(0.5*(2**high-(high+1))*d_rep))+(high-1)*t_react

        # one step on its footprint (doubled for the physical qubits), holding the clean ancilla
        cost = acc.op('fixup', time, {'rep': 2*space}, toffoli=toff,
                   i_states=injection_i_state + correction_i_state + hadamard_i_state, clean=clean_ancilla, buffer = 6*buff)
        return {**cost, 'split': split}
 
    elif split != 'space-time':
        raise ValueError(f"unknown split {split!r}")
 
    cost = acc.empty()
    while address > 2:
        n_address = int(np.ceil(np.log2(address)))
 
        # The one-hot register is 2**high clean ancilla, so the ancilla demand
        # grows monotonically in `high` while the round cost is U-shaped
        # (unary encoding ~ 2*2**high*max(d, t_CXCXX) against Select ~ 1.5*d*2**low).
        # Pick the cheapest `high` that fits the budget; fall back to the
        # ancilla-minimal high=1 if nothing does.
        options = []
        for h in range(1, n_address):
            sel = QROM_cost(2**(n_address - h), 2**h, d_rep)
            anc = sel['Clean ancilla'] #Ancilla from QROM are costed in the buffer
            swap = SWAP_cost(h, 2**h, d_rep)
            buff = max(swap['Buffer ancilla'], sel['Buffer ancilla'])
            # 2*2**h Hadamards, two at a time at d rounds per pair (catalysis), as charged below
            t = (2*acc.total(swap, 'rounds')
                 + acc.total(sel, 'rounds') + 2**h*d_rep)
            if ancilla_budget is None or (anc <= ancilla_budget and (buffer_budget is None or buff <= buffer_budget)):
                options.append((t, h))
        high = min(options)[1] if options else 1
        low = n_address - high

        unary = SWAP_cost(high, 2**high, d_rep)
        select = QROM_cost(2**low, 2**high, d_rep)

        # the iteration's layout, for its PEAK SPACE only: the swap network
        # runs in the QROM's layout (the one-hot register is the QROM's
        # target, and the swap's buffer sits in the QROM's), plus the swap's
        # buffer slots the QROM's buffer region does not hold (d rows,
        # doubled), plus the `high` address bits controlling the swap
        # network, 2d each as in the space-space branch.  Summing the two
        # footprints would count the shared register twice.  Each swap
        # run's VOLUME still uses the swap network's own footprint; the
        # Hadamards use this one
        extra_buffer = 2*max(unary['Buffer ancilla'] - qrom_ancilla(2**low), 0)*d_rep
        footprint = select['space']['rep'] + extra_buffer + 2*d_rep*high #Multiplied by two for physical qubit count

        # every step is part of the fixup ('fixup:qrom', 'fixup:swap', ...),
        # so the report counts its whole duration under the fixup
        iteration = acc.seq(
            acc.within(select, 'fixup'),
            acc.within(unary, 'fixup'),
            # the swap network runs a second time to uncompute the one-hot
            # register: the same network on the same footprint, paying its
            # time, Toffolis and i-states again
            acc.within(unary, 'fixup'),
            # the Hadamard layers: 2*2**high Hadamards, two at a time at d
            # rounds per pair (catalysis), 2 i-states each and 2 catalysts per layer
            acc.op('fixup:hadamard', 2**high*d_rep, {'rep': footprint}, i_states=4*2**high + 4))
        # LAYOUT OVERRIDE (seq would take the larger of unary's and select's):
        # the iteration's own layout footprint, above
        acc.set_space(iteration, {'rep': footprint}, 'fixup')
        # the swap network and the Select act on the same ancilla, the
        # one-hot register (select's output register), so it is counted once.
        # The swap network's 3 correction qubits are buffer slots
        iteration['Clean ancilla'] = select['Clean ancilla']
        # Buffer ancilla: seq's max over select and unary is right (rough
        # costing, they share the buffer zone)

        cost = acc.seq(cost, iteration)
        address = 2**low

    return {**cost, 'split': split}

def ref(bit: int, d_rep: int):
    '''Reflection about `bit` qubits.  Fallback timing.'''
    if bit <= 0:
        return acc.empty()

    buff = 2
    space = 2*d_rep*bit + 13*buff*d_rep + max(bit-2,1)*d_rep*2
    time = TOFF_TIME*d_rep*max(bit-2,0)

    i_state = I_COUNT*max(bit-2,0) + buff
    # the reflection about `bit` qubits needs bit-2 Toffolis; none below 2
    return acc.op('reflection', time, {'rep': 2*space}, toffoli=max(bit-2,0),
                  i_states=i_state, clean=max(bit - 2, 0), buffer = 13*buff)


# Steps of the high_level_cost routines that are not one of the primitives
# above.  Every zone lives in the rep code here, so a footprint made of zone
# qubits is layout.physical of their number.

def prep_shuffle_cost(bit_precision: int, d_rep: int):
    '''
    OuterPrep/InnerPrep: move the bit_precision Fourier qubits next to the
    loaded output register and back out again.  
    Footprint: the moved qubits.
    '''
    time = 2*int(np.ceil(bit_precision/4))*d_rep
    return acc.op('shuffle', time, {'rep': layout.physical(bit_precision, d_rep)})


def rot_shuffle_cost(b_rot: int, lam: int, d_rep: int, *, N: int, n_x: int, n_b: int):
    '''
    RPrep+Rot: move the lam loaded angles, one angle (b_rot qubits) at a
    time; the Fourier state is moved back each time, with transversal CNOTs
    (the + lam in `time`).
    The moves use the 5 routing corridors of the global layout, so one move
    of b_rot qubits takes ceil(b_rot/5)*d_rep rounds: the first angle is
    moved out, then in, then any remaining angles are moved in.
    Footprint: the angle being moved.
    N (the system size) and n_x, n_b (the bits of the X and B registers)
    are part of the interface for the LDPC model's storage check; unused
    here.
    '''
    time = (4 + max(lam - 2, 0))*int(np.ceil(b_rot/5))*d_rep + lam  #lam accounts for the transversal CNOTs
    # the angles' MBU is RPrep_Rot's own step (mbu_cost, once per angle)
    return acc.op('shuffle', time, {'rep': layout.physical(b_rot, d_rep)})


def rotation_cost(b_rot: int, d_rep: int):
    '''
    One rotation, 2 parity-controlled NOTs with b_rot Fourier qubits as
    control, 2d rounds, on one target qubit:
      RPrep+Rot   a system qubit, b_rot = the whole Fourier state
      the preps   the address qubit being prepared, b_rot = b_coef
    Footprint 2d for the target qubit, plus the Fourier qubits.
    '''
    return acc.op('rotation', 2*d_rep, {'rep': layout.physical(b_rot, d_rep) + 2*d_rep})


def mbu_cost(qubits: int, d_rep: int):
    '''
    Measurement-based uncomputation: 1 round.  Footprint: the `qubits`
    measured out.  Called by _sequential_prep (the loaded output register),
    SelectSwap_cost (the clean variety's K-1 unselected copies),
    RPrep_Rot (once per loaded angle, b_rot qubits).
    '''
    return acc.op('mbu', 1, {'rep': layout.physical(qubits, d_rep)})


def maj_cost(n_b: int, d_rep: int):
    '''
    Maj's part other than its 2 comparators.  Placeholder: 2 n_B-bit Toffolis
    (fallback timing) plus 2 CCZ, 1 CZ and 3 H.  Footprint: 2*n_b + 2 slots
    (the B register, the n_b ancilla and 2 more, 2d each) plus 26 buffer
    slots, doubled.  n_b clean ancilla: n_b - 1 for the Toffoli ladder plus
    a flag live only here.
    '''
    buff =2 
    # 2 CCZ (3d each), the CZ (3d), the 3 H (d each), then the ladder
    time = (6 + 3 + 3)*d_rep + 2*n_b*TOFF_TIME*d_rep
    # I_COUNT per Toffoli: the 2*n_b of the ladder and the 2 CCZ; 8 for the
    # CZ (2) and the 3 H (2 each); buff catalysts
    i_state = I_COUNT*2*(n_b+1) + 8 + buff
    space =  2*(2*n_b+2)*d_rep + 13*buff*d_rep
    #Clean ancilla also includes an additional flag qubit 
    return acc.op('maj', time, {'rep': 2*space}, toffoli=2 + 2*n_b, i_states = i_state, clean=n_b, buffer = 13*buff)


def factories_required(cost, d, rate=1):
    '''
    Consistency check: a routine cannot consume magic states faster than they are
    produced.  `rate` is states per d rounds per factory.  Returns the number of
    parallel factories the quoted temporal cost implicitly assumes.
    '''
    rounds = acc.total(cost, 'rounds')
    if rounds <= 0:
        return 0.0
    return acc.total(cost, 'toffoli')*d/(rate*rounds)
