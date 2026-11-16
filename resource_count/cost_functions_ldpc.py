"""
Cost functions for the hybrid model.

Every zone (system, address, Fourier state, QPE, flag, loaded output) lives
in an LDPC code, except RPrep+Rot's parked angles, which wait their turn in
the rep-code routing layer under the system qubits; the buffer and routing
stay in the repetition code.  Pass
this module as `model` to the routines in high_level_cost.py, exactly as
cost_functions.py (the rep-code model) is passed.  It must provide every name
in high_level_cost.MODEL_INTERFACE; the test suite checks that it does.

Conventions (as in cost_functions.py; see accounting.py)
-----------
Every primitive returns a cost record, built with acc.op / acc.seq.

space            {layer: qubits}, one entry per layer the footprint uses,
                 in that layer's own unit:
                     'rep'   physical qubits, stabilizer ancilla included
                     'ldpc'  logical qubits (zone_qubits and
                             logical_qubit_cycles use the same unit)
                 Volume is computed per layer from it, and the layers are
                 only added after logical_qubit_cycles converts each.
rounds           syndrome extraction rounds, the one clock of both layers.
toffoli,
i-states         as in the rep code.
t_CXCXX          as in the rep code: rounds to inject one |C_XC_XX> state
                 (default 21).
'Clean ancilla', 'Dirty ancilla', 'Buffer ancilla'
                 logical qubits, as in the rep code.

BUSY QUBITS ARE INSIDE THE FOOTPRINT.  high_level_cost.py tells each step
which zone qubits are busy in it (listed under "Busy" in each docstring
below); those are NOT charged as idle, so this footprint must contain them,
in the layer ZONE_LAYER puts their zone in.  E.g. a lookup reading 9
address bits must put those 9 address qubits in its 'ldpc' footprint; if it
puts them in 'rep' instead they are counted in the wrong layer, and if it
leaves them out they are counted nowhere.  Idle qubits are added by
high_level_cost.py itself, through zone_qubits.

Code parameters
---------------
Two distances, one per layer:
  d_LDPC   the LDPC code's: sets the chip sizes (chip_size_LDPC), including
           the size of the rep-code routing layer under the chip, and the
           adder's layout.  Only d_LDPC = 12, 16 and 22 are supported; any
           other d_LDPC raises.
  d_rep    the repetition code's: every rep-code qubit (buffer rows, the
           routing qubits, parked angles) is initialized at d_rep, so the
           rep footprints and zone_qubits use it.
Time: a logical cycle is d = max(d_rep, d_LDPC) rounds (rounds_per_cycle),
so every step that is timed in units of d uses the larger distance.
"""

import numpy as np

from resource_count import accounting as acc
from resource_count import layout
from resource_count import cost_functions as rep      # the rep-code model, for the rep-layer parts


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# the same as the rep code's, taken from it so the two cannot drift apart:
# fallback rounds per Toffoli in units of d (printed by the report), the
# reaction time in rounds, and the i-states per C_X C_X X injection
TOFF_TIME = rep.TOFF_TIME
T_REACT = rep.T_REACT
I_COUNT = rep.I_COUNT

# categories (acc.op names) whose timing is a TOFF_TIME*d placeholder; the
# report marks them with a star
FALLBACK = ('comparator', 'reflection', 'maj', 'qc-add')         

# the system qubits live in LDPC, which has full access, so they can be
# borrowed as dirty swap ancilla: every batching method and prep variety is
# available
SYSTEM_Z_ACCESS = True

# which layer each zone lives in, 'rep' or 'ldpc'.  'output' is a lookup's
# loaded output register, from the lookup until it is measured out
ZONE_LAYER = {'system': 'ldpc',     # the target register
              'address': 'ldpc',    # X and B registers + 2 selection qubits
              'fourier': 'ldpc',
              'QPE': 'ldpc',
              'flag': 'ldpc',
              'output': 'ldpc',      #Compute zone
              # RPrep+Rot's angles while they wait their turn: moved out
              # into the rep-code routing layer under the system qubits
              # (rot_shuffle_cost checks they fit).  The routing is only
              # stabilized while angles are parked there, so only they count
              'parked': 'rep'}


# ---------------------------------------------------------------------------
# Layers: where a zone lives, its unit, and the conversion to logical cycles
# ---------------------------------------------------------------------------

def zone_qubits(zone, logical, d_rep, d_LDPC):
    '''
    (layer, qubits) for `logical` qubits of `zone`, in that layer's own unit.
    Used by layout.idle_except for every idle qubit.  d_LDPC is unused (an
    LDPC zone is counted in logical qubits) but taken, since every model
    function is called with the model's full distances.
    '''
    layer = ZONE_LAYER[zone]
    if layer == 'rep':
        return layer, layout.physical(logical, d_rep)
    if layer == 'ldpc':
        return layer, logical
    raise ValueError(f"unknown layer {layer!r} for zone {zone!r}")


def rounds_per_cycle(d_rep, d_LDPC):
    '''Rounds in one logical cycle: d = max(d_rep, d_LDPC), the d every
    step is timed in.  Total_time's 'total logical cycles' is the rounds
    divided by it.'''
    return max(d_rep, d_LDPC)


def access_corridors(qubits):
    '''
    Access corridors for `qubits` logical qubits that need Z access: none.
    Every register that needs Z access (a lookup's output, the system
    qubits) lives in LDPC, which has full access.
    Rep code: cost_functions.access_corridors (one per two qubits).
    '''
    return 0


def logical_qubit_cycles(layer, qubit_cycles, d_rep, d_LDPC):
    '''
    Logical qubit-cycles (the effective volume) of `qubit_cycles` in `layer`,
    given in that layer's own unit.  Total_time adds the layers only after
    this conversion.

    Each layer is divided by ITS OWN cycle (d_rep rounds for 'rep', d_LDPC
    rounds for 'ldpc'), not by rounds_per_cycle (the max): each layer's
    logical error rate is quoted per cycle of its own distance, so volume x
    rate = qubit-rounds x the error per round.  The longer steps (timed in
    the max) are already in the qubit-rounds.
    '''
    if layer == 'rep':
        return rep.logical_qubit_cycles('rep', qubit_cycles, d_rep)
    if layer == 'ldpc':
        return qubit_cycles/d_LDPC
    raise ValueError(f"unknown layer {layer!r}")

# the chip layout's constants per supported LDPC distance: H; base, the
# chip size's offset (chip_size_LDPC = H*L + base); and the adder's p_r and
# p_l.  Used by chip_size_LDPC (H, base), adder_cost (H, p_r, p_l),
# rot_shuffle_cost's storage check (H) and mbu_cost (H)
CHIP_LAYOUT = {12: dict(H=6, base=17, p_r=4, p_l=3),
               16: dict(H=7, base=26, p_r=5, p_l=4),
               22: dict(H=8, base=29, p_r=4, p_l=5)}


def chip_layout(d_LDPC):
    '''CHIP_LAYOUT's constants for d_LDPC; any other distance raises.'''
    if d_LDPC not in CHIP_LAYOUT:
        raise ValueError(f"unsupported LDPC distance {d_LDPC!r}")
    return CHIP_LAYOUT[d_LDPC]


def chip_size_LDPC(L,d):
    '''
    Fixed chip sizes for different lengths of chip (and distances):
    H*L + base (chip_layout).  L<17 possible since only part of smaller
    subroutine
    '''
    chip = chip_layout(d)
    if L==0:
        return 0
    return chip['H']*L + chip['base']


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def qrom_ancilla(address):
    '''
    Clean ancilla of the skew-tree QROM over `address` addresses.
    Called by: QROM_cost, SelectSwap_cost and phase_fixup_cost (buffer
    slots).
    Rep code: cost_functions.qrom_ancilla.
    '''
    return rep.qrom_ancilla(address)+1      #ancilla are in rep code layer with 1 extra routing corridor

def QROM_cost(address: int, target_bit: int, d_rep: int, d_LDPC: int, buff: int = 2, t_CXCXX = 21):
    '''
    Skew-tree QROM with classically absorbed corrections.
    Assumes C_XC_XX production rate of at least one per d rounds.
    A single-address table is a classical constant: Pauli only, zero cost.
    The target register requires Z access, which are NOT counted as clean ancilla
    '''

    address = int(np.ceil(address))

    if address <= 1:
        return acc.empty()


    n_address = int(np.ceil(np.log2(address)))
    space_LDPC = n_address+target_bit
    L = int(np.ceil(n_address/2))+int(np.ceil(target_bit/2)) #Length of LDPC layer
    LDPC_routing = chip_size_LDPC(L,d_LDPC)
    extra = qrom_ancilla(address) #Space occupied in buffer
    space_rep = LDPC_routing + extra*d_rep+ 2*max(L, extra) + max(extra -L, 0) #includes routing
    d = max(d_rep, d_LDPC)
    time = int((address-1)*d) + int((np.ceil(address/2) - n_address)*max(d, t_CXCXX))

    # 2**x - x - 1 Ands for M = 2**x addresses (x address bits), 2 i-states each
    ands = address - n_address - 1
    i_state = ands*2 + buff #No C_X X phase correction

    # clean = target_bit: the output register itself, fresh fungible
    # allocation -- not allocatable into buffer zone.  The n_address - 1
    # working ancilla sit in the buffer (`extra`, the buffer count
    # qrom_ancilla), so the volume includes them
    return acc.op('qrom', time, {'rep': 2*space_rep, 'ldpc': space_LDPC},
                  toffoli=ands, i_states=i_state,
                  clean=target_bit, buffer=extra)


def SelectSwap_cost(address: int, target_bit: int, alpha: int, d_rep: int, d_LDPC: int, 
                    variety: str = 'clean', batched: bool = False,
                    n_select: int = None, n_swap: int = None):
    '''
    SelectSwap lookup table over `address` addresses into a `target_bit`
    output register, swap parameter K = 2**alpha.
    Called by:
      _sequential_prep (OuterPrep/InnerPrep), target_bit = b_coef,
        unbatched (relies on batched=False)
      RPrep_Rot via batching.batching_cost, address = Y,
        target_bit = lam*b_rot, batched=True
      SelectSwap_opt, the alpha search, with the same arguments
    Busy:   address  the ceil(log2 address) bits it reads
            output   target_bit
    Must return the label 'K' (= 2**alpha): RPrep_Rot reads it.
    batching.batching_cost reads its 'Clean ancilla' and 'Buffer ancilla'.
    Rep code: cost_functions.SelectSwap_cost.

    Basically the same as the rep code version except that Hadamards are slower 
    if target is in LDPC (3d for the catalysis version)
    '''
    
    
    Alpha = 2**alpha
    labels = {'variety': variety, 'batched': batched, 'K': Alpha}
    d = max(d_rep, d_LDPC)    # one logical cycle, for the dirty Hadamard layer
    
    if Alpha == 1:
        ns = 1 if n_select is None else n_select
        return {**acc.repeat(QROM_cost(address, target_bit, d_rep, d_LDPC), ns), **labels}
    
    Select = QROM_cost(np.ceil(address/Alpha), Alpha*target_bit, d_rep, d_LDPC)
    # (Alpha-1)*target_bit Ctrl-Swaps over all Alpha registers
    Swap = SWAP_cost(alpha, (Alpha - 1)*target_bit, d_rep, d_LDPC, qubits=Alpha*target_bit)
    
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
    # are borrowed system qubits: busy in the lookup, so inside its
    # footprint, with the routing under them (chip_size_LDPC over all K
    # copies) -- which also holds the swap network's own routing
    space = QROM_cost(address, Alpha*target_bit, d_rep, d_LDPC)['space']
    space['rep'] += extra_buffer

    if variety == 'clean':
        # measure out the K-1 copies the swap network did not select (no
        # phase fixup of its own)
        steps.append(mbu_cost((Alpha - 1)*target_bit, d_rep, d_LDPC))
        # all K copies of the output register are fresh/clean.  Not
        # Select['Clean ancilla']: with K >= address the Select has one
        # address, a classical constant that QROM_cost counts as empty
        anc = Alpha*target_bit
        dirty_anc = 0
    else:
        anc = target_bit

        #Layer of Hadamards (it is a conjugation, so it needs to be 2 per clean output register)
        #For space we account the full QROM layout 
        #Hadamard gate on LDPC is costed as catalysis taking 3d cycles
        steps.append(acc.op('swap:hadamard', 2*target_bit*3*d, space,
                                i_states=2*2*target_bit+2))
            
        dirty_anc = (Alpha - 1)*target_bit
    
    cost = acc.seq(*steps)
    acc.set_space(cost, space, 'SelectSwap')
    cost['Clean ancilla'], cost['Dirty ancilla'] = anc, dirty_anc

    return {**cost, **labels}
    


def adder_cost(bit: int, d_rep: int, d_LDPC: int, buff: int = 3, t_CXCXX: int = 21):
    '''
    Phase gradient addition of a loaded `bit`-qubit value into the Fourier
    state.  Time is a simplified count, 2*bit steps of max(d, t_CXCXX)
    rounds (d = max(d_rep, d_LDPC)): the reaction time wait and the MBU round
    are part of it, not tracked as steps of their own.
    Called by: _sequential_prep (bit = b_coef), RPrep_Rot (bit = b_rot),
    batching.batching_cost (its 'Buffer ancilla').
    Busy:   fourier  bit
            output   bit
    Its 'Buffer ancilla' sizes the buffer zone: layout.buffer_zone_size()
    is 9*3 = 27, the adder's (buff = 3 in both models).

    ASSUMPTION: s_k are initialized appropriately in LDPC
    Rep code: cost_functions.adder_cost.
    '''

    if bit <= 0:
            return acc.empty()

    chip = chip_layout(d_LDPC)
    H, p_r, p_l = chip['H'], chip['p_r'], chip['p_l']

    L = int(np.ceil(3*bit/2))
    space_rep =  (L+p_r + p_l)*(H+1) + (9*buff)*d_rep + 4*(L+9*buff-5)
    d = max(d_rep, d_LDPC)
    t_surg = d #One logical cycle (ignoring initialize and measurement)
    t_cad = max(t_surg, t_CXCXX)
    time = 2*bit*t_cad
    
    i_state = 4*bit + buff   #I_COUNT = 4 per Toffoli (2 for the injection, 2 for the phase correction), plus the buff catalysis states
    
    return acc.seq(
        acc.op('adder', time, {'rep': 2*space_rep, 'ldpc': 3*bit}, toffoli=bit, i_states=i_state,
                   clean=bit, buffer=9*buff))
    



def qc_comp_cost(bit: int, d_rep: int, d_LDPC: int):
    '''
    Quantum-classical comparison of a `bit`-qubit register, 1 output qubit.
    Assumes address in LDPC, and puts flag into LDPC as well
    Called by: Spin (bit = n_X, output = the flag), Maj (bit = n_X, twice).
    Busy:   address  bit (the X register)
            flag     1 (Spin only; Maj's comparator has no flag busy)
    Rep code: cost_functions.qc_comp_cost (fallback timing, 'comparator').
    '''
    if bit <= 1:
            return acc.empty()

    buff = 2
    L_LDPC = int(np.ceil(bit/2))+1
    space_rep = chip_size_LDPC(L_LDPC,d_LDPC) + 2*bit*d_rep + 13*buff*d_rep
    d = max(d_rep, d_LDPC)
    time = TOFF_TIME*(bit - 1)*d
    
    i_state = I_COUNT*(bit-1) + buff
    return acc.op('comparator', time, {'rep': 2*space_rep, 'ldpc': bit +1}, toffoli=bit - 1,
                      i_states=i_state, clean=bit - 1, buffer = 13*buff)



def qc_add_cost(bit: int, d_rep: int, d_LDPC: int):
    '''
    Controlled quantum-classical addition on a `bit`-qubit register.
    Both control and address live in LDPC
    Called by: RPrep_Rot's flattening (and its reverse), bit = n_x, the
    address bits of x, controlled by the flag.
    Busy:   address  bit
            flag     1
    Rep code: cost_functions.qc_add_cost (fallback timing, 'qc-add').
    '''

    if bit <= 1:
        return acc.empty()

    buff = 2
    L_LDPC = int(np.ceil(bit/2))+1
    space_rep = 2*d_rep*bit + chip_size_LDPC(L_LDPC, d_LDPC) + 13*buff*d_rep
    d = max(d_rep, d_LDPC)
    time = TOFF_TIME*(bit - 1)*d

    i_state = I_COUNT*(bit-1) + buff
    return acc.op('qc-add', time, {'rep': 2*space_rep, 'ldpc': bit +1}, toffoli=bit - 1,
                  i_states=i_state, clean=bit - 1, buffer = 13*buff)


def SWAP_cost(control: int, swapands: int, d_rep: int, d_LDPC: int, buff: int =3, qubits: int = None):
    '''
    Controlled-SWAP network: `swapands` controlled swaps (one Toffoli
    each), controlled on `control` qubits.
    Used only inside this model, by SelectSwap_cost and phase_fixup_cost
    (whose own layout overrides set their space); the routines in
    high_level_cost.py use PlainSWAP_cost.
    swapands: the Ctrl-Swaps, which set the time, the Toffolis and the
    i-states.
    qubits: the swapped qubits the network holds in LDPC, which set its
    space.  Default swapands; a SelectSwap over K registers of b qubits has
    (K-1)*b Ctrl-Swaps but holds K*b qubits.
    buff: the catalysis states.  With the 3 qubits storing the parity-CNOT
    corrections they take 8*buff + 3 buffer slots, and as they are reset
    between operations each adds one i-state.  i-states: 4 per Toffoli,
    buff, and `control` for the S corrections (worst case).
    Rep code: cost_functions.SWAP_cost.
    '''
    if control <= 0 or swapands <= 0:
        return acc.empty()
    qubits = swapands if qubits is None else qubits

    L_LDPC = int(np.ceil(control/2)) + int(np.ceil(qubits/2))
    #One extra row of routing, in case the buffer sits too far to the right
    space_rep = chip_size_LDPC(L_LDPC, d_LDPC) + 2*max(L_LDPC, 8*buff + 3) + (8*buff +3)*d_rep
    d=max(d_rep, d_LDPC)
    time = swapands*2*d+T_REACT+(control+1)*d

    i_state = 4*swapands + buff + control #For the control, S correction worst case
    # the 3 correction qubits are buffer slots (8*buff + 3)
    return acc.op('swap', time, {'rep': 2*space_rep, 'ldpc': control + qubits}, toffoli=swapands,
                  i_states=i_state, clean=0, buffer=8*buff+3)


def PlainSWAP_cost(pairs: int, d_rep: int, d_LDPC: int, buff: int = 3):
    '''
    Swap of `pairs` pairs of qubits (2*pairs qubits), controlled on 1 qubit.
    One Toffoli per pair.
    Called by:
      Spin, pairs = N: the 2N system qubits, controlled on the spin qubit
        Busy:   system   2N (LDPC)
                address  1 (the spin control qubit) (LDPC)
      RPrep_Rot's flattening (and its reverse), pairs = n_b: the |c> and
      |b> registers, controlled on the flag
        Busy:   address  2*n_b (LDPC)
                flag     1 (LDPC)
    Since Z access is not a problem this could also be done with SWAP_cost.
    buff: the catalysis states, as in SWAP_cost: 8*buff + 3 buffer slots
    and buff i-states.  i-states: 4 per Toffoli, buff, and 1 for the S
    correction (one control qubit).
    Rep code: cost_functions.PlainSWAP_cost.
    '''
    if pairs <= 0:
        return acc.empty()
    space_LDPC = pairs + 1
    space_rep = chip_size_LDPC(space_LDPC, d_LDPC) + 2*max(space_LDPC, 8*buff + 3) + (8*buff + 3)*d_rep
    d = max(d_rep, d_LDPC)
    time = pairs*2*d+T_REACT+2*d #Only one possible control qubit for S correction
    i_state = 4*pairs + buff + 1 

    return acc.op('PlainSwap', time, {'rep': 2*space_rep, 'ldpc': 2*pairs +1}, toffoli=pairs,
                  i_states=i_state, clean=0, buffer=buff*8+3)


def ref(bit: int, d_rep: int, d_LDPC: int):
    '''
    Reflection about `bit` qubits, controls included.
    Called by: Ref, bit = the sum of its busy qubits:
      InnerRef  address n_B + 1                        (flag present, idle)
      OuterRef  address n_X + n_B + 2, QPE 2
    Busy:   those qubits, in their zones' layers
    Rep code: cost_functions.ref (fallback timing, 'reflection').
    '''
    if bit <= 0:
        return acc.empty()

    buff = 2
    # one LDPC column per qubit (an overcount, as in maj_cost), the bit - 2
    # ancilla in the rep layer, 2d each
    space = chip_size_LDPC(bit, d_LDPC) + 13*buff*d_rep + 2*max(bit - 2, 0)*d_rep
    d = max(d_rep, d_LDPC)
    time = TOFF_TIME*d*max(bit-2,0)
    
    i_state = I_COUNT*max(bit-2,0) + buff
    # the reflection about `bit` qubits needs bit-2 Toffolis; none below 2
    return acc.op('reflection', time, {'rep': 2*space, 'ldpc': bit}, toffoli=max(bit-2,0),
                    i_states=i_state, clean=max(bit - 2, 0), buffer=13*buff)


def phase_fixup_cost(address: int, d_rep: int, d_LDPC: int, ancilla_budget: int = None, buffer_budget: int = None,
                     t_react: float = T_REACT, t_CXCXX: float = 21, split: str = None):
    '''
    Phase fixup after the measurement-based uncomputation of a lookup over
    `address` addresses.
    Called by:
      _sequential_prep, address = L (each level for the adjoint, the largest
        for the forward prep), ancilla_budget = the clean budget
      RPrep_Rot, address = Y, ancilla_budget = max(clean budget, lam*b_rot)
      (the measured-out angles are free by then)
    ancilla_budget may be None (no clean ancilla available).
    split=None (default): the faster split (fewer rounds) of those that fit
    the budget, as in the rep code.
    split='space-space': as the rep code's, at d = max(d_rep, d_LDPC), with
      2d per Hadamard and 1.5d per product of the high half (its MBU takes
      d in LDPC).  The products (its clean ancilla) and the address bits
      sit in LDPC, the 6*buff buffer slots in the rep layer.
    Busy:   address  the ceil(log2 address) bits it reads
    Rep code: cost_functions.phase_fixup_cost (space-space or space-time
    split, the faster one that fits the budgets).
    '''
    address = int(np.ceil(address))
    if address <= 2:
        return acc.empty()
    
    n_address = int(np.ceil(np.log2(address)))
    low = int(np.floor(n_address/2))
    high = n_address - low
    
    if split is None:
        # its clean ancilla: the products, as in the space-space branch
        space_space_anc = 2**high + 2**low - n_address - 2
        if ancilla_budget is None or space_space_anc > ancilla_budget:
            split = 'space-time'
        else:
            # both fit: cost both, keep the one with fewer rounds (min keeps
            # the first, space-space, on a tie)
            options = [phase_fixup_cost(address, d_rep, d_LDPC, ancilla_budget=ancilla_budget,
                                        buffer_budget=buffer_budget, t_react=t_react, t_CXCXX=t_CXCXX, split=s)
                       for s in ('space-space', 'space-time')]
            return min(options, key=lambda r: acc.total(r, 'rounds'))
    
    if split == 'space-space':
        buff = 2
        toff = 2**high + 2**low - n_address-2     # one Toffoli per product of at least two bits
        hadamard = min(2**high,2**low)-1
        clean_ancilla = toff                    # the products
        # the address bits the fixup reads and the toff products (clean
        # ancilla) sit in LDPC; the buffer slots in the rep layer
        space_LDPC = toff + n_address
        L_LDPC = int(np.ceil(space_LDPC/2))
        space_rep = chip_size_LDPC(L_LDPC, d_LDPC) + 6*buff*d_rep

    
        injection_i_state = 2*toff + buff #Power product is implemented with wandering AND so no phase correction
        correction_i_state = 2*int(np.ceil(0.5*toff)) + 1 #Rough counting (half the corrections on average)
        hadamard_i_state = 2*hadamard + 1 #Technically multi phase gates not Hadamards

       
        d = max(d_LDPC, d_rep)
        footprint = {'rep': 2*space_rep, 'ldpc': space_LDPC}

        time = (2**high + 2**low - (high+low+2))*max(d, t_CXCXX)+ (2**low-1)*2*d + int(np.ceil((high-1)*t_react + (2**high-(high+1))*(d + 0.5*d)))
        cost = acc.op('fixup', time, footprint, toffoli=toff,
                   i_states=injection_i_state + correction_i_state+hadamard_i_state, clean=clean_ancilla, buffer=buff*6)
        return {**cost, 'split': split}
    
    elif split != 'space-time':
        raise ValueError(f"unknown split {split!r}")
    
    cost = acc.empty()
    d = max(d_rep, d_LDPC)
    while address > 2:
        n_address = int(np.ceil(np.log2(address)))
    
        # The one-hot register is 2**high clean ancilla, so the ancilla demand
        # grows monotonically in `high` while the round cost is U-shaped
        # (unary encoding ~ 2*2**high*max(d, t_CXCXX) against Select ~ 1.5*d*2**low).
        # Pick the cheapest `high` that fits the budget; fall back to the
        # ancilla-minimal high=1 if nothing does.
        options = []
        for h in range(1, n_address):
            sel = QROM_cost(2**(n_address - h), 2**h, d_rep, d_LDPC)
            anc = sel['Clean ancilla'] #Ancilla from QROM are costed in the buffer
            swap = SWAP_cost(h, 2**h, d_rep, d_LDPC)
            buff = max(swap['Buffer ancilla'], sel['Buffer ancilla'])
            # 2*2**h Hadamards at 3*d rounds each (catalysis)
            t = (2*acc.total(swap, 'rounds')
                 + acc.total(sel, 'rounds') + 2*2**h*3*d)
            if ancilla_budget is None or (anc <= ancilla_budget and (buffer_budget is None or buff <= buffer_budget)):
                options.append((t, h))
        high = min(options)[1] if options else 1
        low = n_address - high

        #output bits and address sit in LDPC
        unary = SWAP_cost(high, 2**high, d_rep, d_LDPC)
        select = QROM_cost(2**low, 2**high, d_rep, d_LDPC)
    
        extra_buffer = 2*max(unary['Buffer ancilla'] - qrom_ancilla(2**low), 0)*d_rep
        space_LDPC = select['space']['ldpc'] + high #High address bits still need to be accounted for
        space_rep = select['space']['rep'] + extra_buffer + 2*chip_size_LDPC(int(np.ceil(high/2)), d_LDPC) 
        #last part to account for more addresses and the physical qubit count

    
        # every step is part of the fixup ('fixup:qrom', 'fixup:swap', ...),
        # so the report counts its whole duration under the fixup
        iteration = acc.seq(
            acc.within(select, 'fixup'),
            acc.within(unary, 'fixup'),
            acc.within(unary, 'fixup'),
            # the Hadamard layers: 2*2**high Hadamards at 3d rounds each
            # (catalysis), 2 i-states each
            acc.op('fixup:hadamard', 2*2**high*3*d, {'rep': space_rep, 'ldpc': space_LDPC}, i_states=4*2**high + 2))
        # LAYOUT OVERRIDE (seq would take the larger of unary's and select's):
        # the iteration's own layout footprint, above
        acc.set_space(iteration, {'rep': space_rep, 'ldpc': space_LDPC}, 'fixup')
        # the swap network and the Select act on the same ancilla, the
        # one-hot register (select's output register), so it is counted once.
        # The swap network's 3 correction qubits are buffer slots
        iteration['Clean ancilla'] = select['Clean ancilla']
        # Buffer ancilla: seq's max over select and unary is right (rough
        # costing, they share the buffer zone)
    
        cost = acc.seq(cost, iteration)
        address = 2**low
    
    return {**cost, 'split': split}


# ---------------------------------------------------------------------------
# The routines' other steps
# ---------------------------------------------------------------------------

def prep_shuffle_cost(bit_precision: int, d_rep: int, d_LDPC: int):
    '''
    Fourier resource state is already stored in compute zone and rotation angles
    are loaded appropriately
    No move necessary, however, need to prep s_i in |+> which takes d cycles total
    Also after the addition the Fourier resource state needs to move back taking (bit_precision + 1)d
    
    Rep code: cost_functions.prep_shuffle_cost.
    '''
    d = max(d_LDPC, d_rep)
    time = (bit_precision+2)*d 
    L_LDPC = bit_precision #since qubits are separated by Fourier resource state and other output
    space_rep = chip_size_LDPC(L_LDPC, d_LDPC)

    return acc.op('shuffle', time, {'rep': 2*space_rep, 'ldpc': bit_precision})


def rot_shuffle_cost(b_rot: int, lam: int, d_rep: int, d_LDPC: int, *, N: int, n_x: int, n_b: int):
    '''
    Move rotation angle out with a CNOT and an M_X measurement (2d cycles)
    Need enough storage under system qubits
    N, n_x, n_b: the system size and the bits of the X and B registers
    (n_x = ceil(log2 max(X, Y)), n_b = ceil(log2(B+1)), as RPrep_Rot's
    address zone); checks that the angle storage fits in the routing layer
    under the system qubits and the address columns (physical data
    qubits).  Raises if not; never changes the cost.
    Moves one angle (b_rot qubits) at a time; the Fourier resource state is
    moved back every time.
    Busy:  output   b_rot
    Rep code: cost_functions.rot_shuffle_cost.

    Method, with d = max(d_LDPC, d_rep), one term of `time` each:
      3*d*b_rot                   moving the second rotation angle out (2*d*b_rot: a CNOT
                                  between LDPC and rep code, then M_X,
                                  leaving the qubit in |+>) and back in
                                  (d*b_rot)
      max(lam - 2, 0)*(b_rot+1)*d any other angle, moved in and out with
                                  CNOTs (d each, plus the uncomputation)
      lam*(b_rot + 1)*d           the Fourier resource state, moved back
                                  every time
    The angles' MBU is RPrep_Rot's own step (mbu_cost, once per angle), not
    part of this cost.
    '''
    d = max(d_LDPC, d_rep)
    time = 3*d*b_rot + max(lam - 2,0)*(b_rot+1)*d + lam*(b_rot +1)*d
    L_LDPC = b_rot #since qubits are separated by Fourier resource state and other output
    space_rep = chip_size_LDPC(L_LDPC, d_LDPC)

    #Check that there is enough space underneath the system qubits
    #Takes into account that the adder needs 4 extra routing columns to the right of the routine
    #Address columns are also available
    address_zone = np.ceil((n_x + n_b)/2)
    H = chip_layout(d_LDPC)['H']
    routing_qubits = H*(N+1-4) + H*(address_zone + 2)
    if b_rot*d_rep > routing_qubits:
        raise ValueError(f"Rotation angle storage {b_rot*d_rep} exceeds system qubit routing layer {routing_qubits}")

    return acc.op('shuffle', time, {'rep': 2*space_rep, 'ldpc': b_rot})


def rotation_cost(b_rot: int, d_rep: int, d_LDPC: int):
    '''
    One rotation: b_rot Fourier qubits as control on 1 target qubit.
    Called by:
      RPrep_Rot, b_rot = the whole Fourier state, target = a system qubit
      _sequential_prep, b_rot = b_coef, target = the address qubit being
      prepared
    Busy:   fourier  b_rot
            system or address   1   (both 'ldpc')
    Rep code: cost_functions.rotation_cost (2 parity-controlled NOTs, 2d
    rounds, footprint 2d for the system qubit plus the Fourier state).
    '''
    d = max(d_rep, d_LDPC)
    L_LDPC = int(np.ceil(3*b_rot/2)) #since qubits are separated by Fourier resource state and other output
    space_rep = chip_size_LDPC(L_LDPC, d_LDPC)

    return acc.op('rotation', 2*d, {'rep': 2*space_rep, 'ldpc': b_rot + 1})


def mbu_cost(qubits: int, d_rep: int, d_LDPC: int):
    '''
    Measurement-based uncomputation of `qubits` qubits.
    Called by: _sequential_prep (qubits = b_coef, the loaded output),
    SelectSwap_cost (the clean variety's K-1 unselected copies),
    RPrep_Rot (once per loaded angle, qubits = b_rot).
    One qubit at a time, d rounds each: each needs access to its Z logical
    operator in the routing underneath.  At most 2*(2H-1) are measured one
    after another (H from chip_layout); beyond that the rest are taken to be
    measured in parallel with them, which needs their Z logical operators
    far enough apart.
    Busy:   output   qubits
    Rep code: cost_functions.mbu_cost (1 round).
    '''
    #costed with 1 snake in rep code layer underneath
    H = chip_layout(d_LDPC)['H']
    time = min(qubits, 2*(2*H-1))*max(d_rep, d_LDPC)
    return acc.op('mbu', time, {'rep': 2*chip_size_LDPC(qubits,d_LDPC), 'ldpc': qubits})


def maj_cost(n_b: int, d_rep: int, d_LDPC: int):
    '''
    Maj's part other than its 2 comparators: 2 n_B-bit Toffolis plus 2 CCZ,
    1 CZ and 3 H.  'ldpc' footprint: the B register and 2 ancilla.  n_b
    clean ancilla: n_b - 1 for the Toffoli ladder plus a flag live only here.
    Busy:   address  n_b (the B register)
    Rep code: cost_functions.maj_cost (placeholder, fallback timing, 'maj').
    '''
    buff =2
    d = max(d_rep, d_LDPC)
    # 2 CCZ (3d each), the CZ (3d), the 3 H (d each), then the ladder
    time = (6 + 3 + 3)*d + 2*n_b*TOFF_TIME*d
    # I_COUNT per Toffoli: the 2*n_b of the ladder and the 2 CCZ; 8 for the
    # CZ (2) and the 3 H (2 each); buff catalysts
    i_state = I_COUNT*2*(n_b+1) + 8 + buff

    # one LDPC column per B-register qubit (an overcount, as in ref), the n_b
    # ancilla in the rep layer, 2d each
    space_rep = chip_size_LDPC(n_b,d_LDPC) + 13*buff*d_rep + 2*n_b*d_rep
    #Ancilla same as for rep code.
    return acc.op('maj', time, {'rep': 2*space_rep, 'ldpc': n_b + 2}, toffoli= 2 + 2*n_b, i_states = i_state, clean = n_b, buffer = 13*buff)
