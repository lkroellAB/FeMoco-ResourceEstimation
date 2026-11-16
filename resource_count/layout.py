"""
Physical layout: the fixed zones (system register, address register,
Fourier resource state, QPE and flag qubits, buffer, compute) and the checks
built on them.

Zone sizes
----------
`system_zone_size`, `address_zone_size`, `fourier_zone_size`, QPE_QUBITS and
FLAG_QUBITS give each zone's size in LOGICAL qubits.  They come from the
algorithm, so they are the same whichever code the zones are stored in.

Idle qubits
-----------
Three questions, answered in three places:

  how big is each zone?             here (logical qubits, model-independent)
  which qubits are busy, and when?  high_level_cost.py, step by step -- e.g.
                                    the system register is busy during Spin's
                                    swap (inside its footprint) but not during
                                    its comparator
  where does a zone live, and in    the cost model (cost_functions.zone_qubits
  what unit are its qubits counted? for rep code), passed in as `model`

Each routine lists the zones present once, and each step lists only what is
BUSY in it; `idle_except` charges everything else as idle, via
accounting.with_idle: the idle qubits add to the step's space and, for the
step's whole duration, to its volume.  A forgotten zone is therefore charged
as idle (an overcount), never silently dropped.

Budgets
-------
`check_feasibility` compares a routine's clean/dirty/buffer ancilla
against the zone budgets.

'Dirty ancilla' is NOT one of these zones -- there is no separate physical
"dirty zone".  Borrowing dirty ancilla means using up some of the System
zone's capacity, so how much dirty ancilla is available depends on how much
of the System zone is currently idle, which varies by routine.  RPrep+Rot's
own dirty budget is intentionally NOT covered here -- it already has a more
precise, batch-progress-aware mechanism in batching.py that this would
undercut, not improve, if duplicated with a cruder formula.
"""
import warnings

import numpy as np

from resource_count import accounting as acc

ZONES = (('Clean ancilla', 'clean', 'ancilla_budget'),
         ('Dirty ancilla', 'dirty', 'dirty_budget'),
         ('Buffer ancilla', 'buffer', 'buffer_budget'))

# The comparison flag qubit: computed in Spin (by its comparator), kept through
# RPrep+Rot and Maj, and uncomputed in Spin_dag -- idle whenever it is not
# being used.
FLAG_QUBITS = 1

# The two phase estimation qubits.
QPE_QUBITS = 2


def physical(logical, d):
    '''
    Physical qubits for logical qubits: 2d-1
    '''
    return (2*d-1)*logical


def idle_except(step, model, distances, zones, busy):
    '''
    `step`, with every zone in `zones` ({zone: logical qubits present}) idle
    beside it for its whole duration -- except the qubits listed in `busy`
    ({zone: logical qubits}), which are inside the step's own footprint and
    so already counted.  E.g.

        zones = {'system': 108, 'fourier': 16}
        idle_except(swap, cf, dict(d_rep=15), zones, busy={'system': 108})

    is the swap (whose footprint contains the system register) with the 16
    Fourier qubits idle beside it.

    For each zone:  idle = size - busy, converted by `model` (e.g.
    cost_functions) into the layer that zone lives in, and added with
    accounting.with_idle (space + idle, volume + idle x the step's rounds).

    A zone missing from `busy` is entirely idle -- forgetting one overcounts
    rather than undercounts.  A busy zone that is not in `zones`, or more busy
    qubits than the zone holds, raises: the table is wrong.
    '''
    for zone in busy:
        if zone not in zones:
            raise ValueError(f"busy zone {zone!r} is not one of the zones present: {sorted(zones)}")
    for zone, size in zones.items():
        idle = size - busy.get(zone, 0)
        if idle < 0:
            raise ValueError(f"{zone}: {busy[zone]} busy but the zone only holds {size}")
        step = acc.with_idle(step, zone, *model.zone_qubits(zone, idle, **distances))
    return step


def buffer_zone_size(buff=3):
    '''The buffer zone's fixed size: adder_cost's own buffer (9*buff), the largest.'''
    return 9*buff


def fourier_zone_size(b_rot):
    '''The Fourier resource state's fixed size.'''
    return b_rot


def address_zone_size(X, B):
    '''
    The address zone: the X register (X values), the B register (B+1
    values) and the 2 selection qubits.
    '''
    return int(np.ceil(np.log2(X))) + int(np.ceil(np.log2(B+1))) + 2


def system_zone_size(N):
    '''The system zone's fixed size.'''
    return 2*N


def check_feasibility(result, name, ancilla_budget=None, dirty_budget=None,
                      buffer_budget=None):
    '''
    Warn for each zone whose budget was given and is exceeded by `result`.
    Replaces repeating the same "if usage > budget: warn" three times per
    composite routine.
    '''
    budgets = {'ancilla_budget': ancilla_budget, 'dirty_budget': dirty_budget,
               'buffer_budget': buffer_budget}
    for field, label, budget_name in ZONES:
        budget = budgets[budget_name]
        usage = result[field]
        if budget is not None and usage > budget:
            warnings.warn(f"{name}: {label} ancilla {usage} exceeds budget {budget}")
