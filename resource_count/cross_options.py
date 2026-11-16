"""
The two cost models with their own and each other's lookup options -- a
contained example, built on the rest of the package without changing it.
Used by the notebook's 'Rep code and LDPC with each other's lookup options'
section and scripts/cross_options_latex.py.

The variants (cross_variants), all at CROSS_DISTANCES so the options are
compared at nearly the same distance in both models:
    'rep, chosen'          the rep code's own options (settings)
    'rep, dirty options'   LDPC's chosen options for the same molecule (dirty
                           SelectSwap in RPrep+Rot and InnerPrep), at the rep
                           code's compute-zone budget
    'LDPC, chosen'         LDPC's own options (settings)
    'LDPC, rep options'    the rep code's options in the LDPC model, at LDPC's
                           compute-zone budget

The rep code's dirty options are not possible in its layout as it is: they
borrow system qubits, which needs Z access to them (rep_code.SYSTEM_Z_ACCESS
is False, so BE_cost refuses them).  Here they are costed as a space
trade-off, with access corridors under all 2N system qubits, one per two
qubits (rep_code.access_corridors, N in all):
  - the cost model is REP_WITH_SYSTEM_Z_ACCESS, a copy of the rep model
    claiming Z access; rep_code itself is not changed;
  - the corridors cost no rounds;
  - they are in the physical qubits (physical_qubits): each half of the system
    zone ceil(3N/2) + 2 wide instead of N + 2;
  - taken as present for the whole run, they are in the rep layer's volume
    and so in the failure probability (add_rep_routing, cross_run): an upper
    bound when they are not needed all the time.
"""
import types

import numpy as np

from resource_count import cost_functions as rep_code
from resource_count import cost_functions_ldpc as ldpc_code
from resource_count import error_accounting as ea
from resource_count import layout

# every variant at these distances: rep code d_rep = 15, LDPC d_rep = 15 and
# d_LDPC = 16
CROSS_DISTANCES = {rep_code: dict(d_rep=15), ldpc_code: dict(d_rep=15, d_LDPC=16)}

# the rep model claiming Z access to the system qubits: a copy, its functions
# the rep model's own
REP_WITH_SYSTEM_Z_ACCESS = types.ModuleType('cost_functions_with_system_z_access')
REP_WITH_SYSTEM_Z_ACCESS.__dict__.update(vars(rep_code))
REP_WITH_SYSTEM_Z_ACCESS.SYSTEM_Z_ACCESS = True


def chip_model(variant):
    '''the cost model whose chip layout `variant` uses: the rep code's for
    the rep model claiming Z access, the variant's own model otherwise.  It
    sets the factories and their routing (error_accounting.FACTORIES,
    FACTORY_ROUTING) and the physical qubit count'''
    return rep_code if variant['model'] is REP_WITH_SYSTEM_Z_ACCESS else variant['model']


def system_corridors(system):
    '''the access corridors under all 2N system qubits: one per two qubits
    (rep_code.access_corridors), N in all'''
    return rep_code.access_corridors(layout.system_zone_size(system['N']))


def cross_variants(system, settings):
    '''The four variants for `system` with its `settings` (SETTINGS_54 /
    SETTINGS_76), as dicts of label, model, budget, option, distances and
    corridors (the access corridors added under the system qubits).  Each
    model runs the other's chosen options for the same molecule: the rep
    code LDPC's (its dirty lookups), LDPC the rep code's'''
    rep_s, ldpc_s = settings[rep_code], settings[ldpc_code]
    return [dict(label='rep, chosen', model=rep_code, budget=rep_s['budget'], option=rep_s['option'],
                 distances=CROSS_DISTANCES[rep_code], corridors=0),
            dict(label='rep, dirty options', model=REP_WITH_SYSTEM_Z_ACCESS, budget=rep_s['budget'],
                 option=ldpc_s['option'], distances=CROSS_DISTANCES[rep_code], corridors=system_corridors(system)),
            dict(label='LDPC, chosen', model=ldpc_code, budget=ldpc_s['budget'], option=ldpc_s['option'],
                 distances=CROSS_DISTANCES[ldpc_code], corridors=0),
            dict(label='LDPC, rep options', model=ldpc_code, budget=ldpc_s['budget'], option=rep_s['option'],
                 distances=CROSS_DISTANCES[ldpc_code], corridors=0)]


def add_rep_routing(cost, slots, distances, model):
    '''Total_time's record with `slots` logical slots of rep-code routing (2
    d_rep - 1 physical qubits each, layout.physical) present for the whole
    run, added to the rep layer's logical qubit-cycles and the effective
    volume, so the error model (error_accounting.layer_errors) counts them in
    the rep code's volume.  Zero slots return the record unchanged'''
    if not slots:
        return cost
    volume = model.logical_qubit_cycles('rep', layout.physical(slots, distances['d_rep'])*cost['total rounds'],
                                        **distances)
    return {**cost, 'total logical qubit cycles [rep]': cost['total logical qubit cycles [rep]'] + volume,
            'total effective volume': cost['total effective volume'] + volume}


def cross_run(system, variant, noise, factory_routing, epsilon, round_time):
    '''One variant's whole run at noise level `noise` (an entry of
    error_accounting.NOISE): Total_time's record with its corridors added
    for the whole run (add_rep_routing), its failure probabilities
    (error_accounting.failure) and its expected duration.  factory_routing:
    the variant's chip's, error_accounting.FACTORY_ROUTING[chip_model(variant)]'''
    v = variant
    cost = add_rep_routing(ea.total_time(system, v['budget'], v['option'], v['distances'], v['model'],
                                         epsilon, round_time),
                           v['corridors'], v['distances'], v['model'])
    fail, logical, magic, istate = ea.failure(cost, v['distances'], noise, v['model'], factory_routing)
    return dict(cost=cost, fail=fail, logical=logical, magic=magic, istate=istate,
                expected=cost['total days']/(1 - fail) if fail < 1 else np.inf)


def physical_qubits(system, variant):
    '''One variant's physical qubits (error_accounting.physical_qubits), with
    the dirty rep variant's access corridors added to the rep-code layout:
    each half of the system zone ceil(3N/2) + 2 wide instead of N + 2, i.e.
    ceil(3N/2) - N more columns on the top half's d+4 rows and the bottom
    half's d rows, doubled for the ancilla as the rest of the core'''
    v = variant
    total = ea.physical_qubits(system, chip_model(v), v['option'], v['distances'])['total']
    if v['model'] is REP_WITH_SYSTEM_Z_ACCESS:
        N, d = system['N'], v['distances']['d_rep']
        extra_columns = int(np.ceil(3*N/2)) - N
        total += 2*(extra_columns*(d + 4) + extra_columns*d)
    return total
