"""
Pinned numbers: today's exact output for a fixed set of calls.

Unlike test_invariants.py (which checks relationships that should hold however
the model is implemented), this file asserts literal values.  Its job is to
make every numerical change *visible*: if a refactor is meant to be cost
neutral, this file must stay untouched; if a change is meant to move numbers,
the diff to EXPECTED is the statement of what moved and by how much.

Regenerating
------------
When a change is intentional, re-run the calls in CASES and paste the new
numbers into EXPECTED / EXPECTED_TOTAL / EXPECTED_BATCHING / EXPECTED_TOTAL_LDPC /
EXPECTED_OPTIONS below.  Keep the
edit to those dicts -- if a *test* has to change too, that is a signal the
change is bigger than a re-pin.

Parameter sets are the paper's (resource_count/settings.py).
The budget is the compute zone's size, which depends on the code: the rep
code's holds the output register's access corridors, LDPC needs none.

    set  N   B   R   C   b_coef  b_rot   budget: rep code  LDPC
    A    54  27  10  27  9       16              48        32
    B    76  57  15  19  9       15              45        30

    A: n_R=4, X = 54 + 16*27 = 486, RC = 16*27 = 432, Y = 54 + 16*27 = 486
    B: n_R=4, X = 76 + 16*19 = 380, RC = 16*19 = 304, Y = 76 + 16*57 = 988

The rep code is pinned at d_rep = 20, the hybrid (LDPC) model ('ldpc_' cases,
EXPECTED_TOTAL_LDPC) at d_rep = d_LDPC = 12, 16 and 22 (its supported
LDPC distances), plus a few cases with the two distances apart.  Both use the
buffer zone's own size from layout.buffer_zone_size() as the buffer budget.
"""
import warnings

import pytest

from resource_count import accounting as acc
from resource_count import cost_functions as cf
from resource_count import cost_functions_ldpc as ldpc
from resource_count import high_level_cost as hlc
from resource_count import batching as bt
from resource_count import optimize as opt
from resource_count import layout

BUFFER = layout.buffer_zone_size()

# RPrep+Rot batch sizes searched in these tests: lam = 1 .. 8
LAMS = range(1, 9)
# swap parameters searched: K = 2**alpha up to 2**4 = 16
ALPHA_MAX = 4


CASES = {
    # --- leaf primitives --------------------------------------------------
    'QROM': lambda: cf.QROM_cost(486, 9, 20),
    'adder': lambda: cf.adder_cost(9, 20),
    'qc_comp': lambda: cf.qc_comp_cost(9, 20),
    'qc_add': lambda: cf.qc_add_cost(9, 20),
    'SWAP': lambda: cf.SWAP_cost(3, 4, 20),
    'ref': lambda: cf.ref(16, 20),
    'SelectSwap_clean': lambda: cf.SelectSwap_cost(486, 9, 2, 20, 'clean', False),
    'SelectSwap_clean_K1': lambda: cf.SelectSwap_cost(486, 9, 0, 20, 'clean', False),
    'SelectSwap_dirty_batched': lambda: cf.SelectSwap_cost(486, 32, 2, 20, 'dirty', True),
    'fixup_space_space': lambda: cf.phase_fixup_cost(486, 20, ancilla_budget=1000, split='space-space'),
    'fixup_space_time': lambda: cf.phase_fixup_cost(486, 20, ancilla_budget=0),
    # --- composites, parameter set A --------------------------------------
    # the preps always count their idle zones, so they take the zone sizes
    'OuterPrep_A': lambda: hlc.OuterPrep(486, 9, 48, dict(d_rep=20), buffer_budget=BUFFER,
                                         N=54, b_rot=16, B=27, model=cf),
    'OuterPrep_dag_A': lambda: hlc.OuterPrep(486, 9, 48, dict(d_rep=20), adjoint=True, buffer_budget=BUFFER,
                                             N=54, b_rot=16, B=27, model=cf),
    'OuterPrep_dirty_A': lambda: hlc.OuterPrep(486, 9, 48, dict(d_rep=20), variety='dirty', buffer_budget=BUFFER,
                                               N=54, b_rot=16, B=27, model=cf),
    'InnerPrep_A': lambda: hlc.InnerPrep(432, 27, 9, 48, dict(d_rep=20), buffer_budget=BUFFER,
                                         N=54, b_rot=16, X=486, model=cf),
    'InnerPrep_dag_A': lambda: hlc.InnerPrep(432, 27, 9, 48, dict(d_rep=20), adjoint=True, buffer_budget=BUFFER,
                                             N=54, b_rot=16, X=486, model=cf),
    'Maj_A': lambda: hlc.Maj(486, 27, dict(d_rep=20), N=54, b_rot=16, model=cf),
    'Spin_A_idle': lambda: hlc.Spin(486, 54, dict(d_rep=20), b_rot=16, B=27, model=cf),
    # RPrep+Rot option: the one in use, qrom with K = 1 and lam = 2,
    # written out here so the pins do not depend on any search
    'RPrep_Rot_A': lambda: hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20),
                                         method='qrom', K=1, lam=2,
                                         buffer_budget=BUFFER, model=cf),
    # the one dirty case: not feasible for the rep code, but the code path is
    # kept for other cost models, so its numbers stay pinned
    'RPrep_Rot_A_dirty': lambda: hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20),
                                               method='dirty_restricted', K=2, lam=2,
                                               buffer_budget=BUFFER, model=cf),
    'BE_A': lambda: hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=20),
                                method='qrom', K=1, lam=2,
                                buffer_budget=BUFFER, model=cf, outer_variety='clean', inner_variety='clean'),
    # --- composites, parameter set B --------------------------------------
    'BE_B': lambda: hlc.BE_cost(76, 57, 15, 19, 9, 15, 45, dict(d_rep=20),
                                method='qrom', K=1, lam=2,
                                buffer_budget=BUFFER, model=cf, outer_variety='clean', inner_variety='clean'),
    # --- the hybrid (LDPC) model: every function at d = 16 ----------------
    'ldpc_QROM': lambda: ldpc.QROM_cost(486, 9, 16, 16),
    'ldpc_adder': lambda: ldpc.adder_cost(9, 16, 16),
    'ldpc_qc_comp': lambda: ldpc.qc_comp_cost(9, 16, 16),
    'ldpc_qc_add': lambda: ldpc.qc_add_cost(9, 16, 16),
    'ldpc_SWAP': lambda: ldpc.SWAP_cost(3, 4, 16, 16),
    'ldpc_PlainSWAP': lambda: ldpc.PlainSWAP_cost(54, 16, 16),
    'ldpc_ref': lambda: ldpc.ref(16, 16, 16),
    'ldpc_SelectSwap_clean': lambda: ldpc.SelectSwap_cost(486, 9, 2, 16, 16, 'clean', False),
    'ldpc_SelectSwap_dirty_batched': lambda: ldpc.SelectSwap_cost(486, 32, 2, 16, 16, 'dirty', True),
    'ldpc_fixup_space_space': lambda: ldpc.phase_fixup_cost(486, 16, 16, ancilla_budget=1000, split='space-space'),
    'ldpc_fixup_space_time': lambda: ldpc.phase_fixup_cost(486, 16, 16, ancilla_budget=0),
    'ldpc_prep_shuffle': lambda: ldpc.prep_shuffle_cost(9, 16, 16),
    'ldpc_rot_shuffle': lambda: ldpc.rot_shuffle_cost(16, 2, 16, 16, N=54, n_x=9, n_b=5),
    'ldpc_rotation': lambda: ldpc.rotation_cost(16, 16, 16),
    'ldpc_mbu': lambda: ldpc.mbu_cost(9, 16, 16),
    'ldpc_maj': lambda: ldpc.maj_cost(5, 16, 16),
    # --- the hybrid (LDPC) model: routines, set A at d = 16 ---------------
    # at LDPC's own compute-zone budgets: 32 for set A, 30 for set B
    'ldpc_OuterPrep_A': lambda: hlc.OuterPrep(486, 9, 32, dict(d_rep=16, d_LDPC=16), buffer_budget=BUFFER,
                                              N=54, b_rot=16, B=27, model=ldpc),
    'ldpc_InnerPrep_A': lambda: hlc.InnerPrep(432, 27, 9, 32, dict(d_rep=16, d_LDPC=16), buffer_budget=BUFFER,
                                              N=54, b_rot=16, X=486, model=ldpc),
    'ldpc_Spin_A': lambda: hlc.Spin(486, 54, dict(d_rep=16, d_LDPC=16), b_rot=16, B=27, model=ldpc),
    'ldpc_Maj_A': lambda: hlc.Maj(486, 27, dict(d_rep=16, d_LDPC=16), N=54, b_rot=16, model=ldpc),
    'ldpc_RPrep_Rot_A': lambda: hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=16, d_LDPC=16),
                                              method='qrom', K=1, lam=2, ancilla_budget=32,
                                              buffer_budget=BUFFER, model=ldpc),
    'ldpc_BE_A_d16': lambda: hlc.BE_cost(54, 27, 10, 27, 9, 16, 32, dict(d_rep=16, d_LDPC=16),
                                         method='qrom', K=1, lam=2,
                                         buffer_budget=BUFFER, model=ldpc, outer_variety='clean', inner_variety='clean'),
    'ldpc_BE_B_d22': lambda: hlc.BE_cost(76, 57, 15, 19, 9, 15, 30, dict(d_rep=22, d_LDPC=22),
                                         method='qrom', K=1, lam=2,
                                         buffer_budget=BUFFER, model=ldpc, outer_variety='clean', inner_variety='clean'),
    # the LDPC block encoding with InnerPrep's lookups dirty (its search pick)
    'ldpc_BE_A_d16_inner_dirty': lambda: hlc.BE_cost(54, 27, 10, 27, 9, 16, 32, dict(d_rep=16, d_LDPC=16),
                                                     method='dirty_restricted', K=2, lam=2,
                                                     outer_variety='clean', inner_variety='dirty',
                                                     buffer_budget=BUFFER, model=ldpc),
    # --- the hybrid (LDPC) model with its two distances apart -------------
    # time in max(d_rep, d_LDPC), chips in d_LDPC, rep-code rows in d_rep
    'ldpc_QROM_drep15_dLDPC22': lambda: ldpc.QROM_cost(486, 9, 15, 22),
    'ldpc_adder_drep15_dLDPC22': lambda: ldpc.adder_cost(9, 15, 22),
    'ldpc_BE_A_drep13_dLDPC16': lambda: hlc.BE_cost(54, 27, 10, 27, 9, 16, 32, dict(d_rep=13, d_LDPC=16),
                                                    method='qrom', K=1, lam=2,
                                                    buffer_budget=BUFFER, model=ldpc, outer_variety='clean', inner_variety='clean'),
    'ldpc_BE_B_drep17_dLDPC12': lambda: hlc.BE_cost(76, 57, 15, 19, 9, 15, 30, dict(d_rep=17, d_LDPC=12),
                                                    method='qrom', K=1, lam=2,
                                                    buffer_budget=BUFFER, model=ldpc, outer_variety='clean', inner_variety='clean'),
}

EXPECTED = {
    'QROM': {
        'Spatial cost': 2064,
        'Temporal cost': 14614,
        'Toffoli cost': 476,
        'i-state cost': 954,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 20,
        'Qubit cycles': 30163296,
        'Idle cycles': 0,
    },
    'adder': {
        'Spatial cost': 2640,
        'Temporal cost': 378,
        'Toffoli cost': 9,
        'i-state cost': 39,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 997920,
        'Idle cycles': 0,
    },
    'qc_comp': {
        'Spatial cost': 2480,
        'Temporal cost': 480,
        'Toffoli cost': 8,
        'i-state cost': 34,
        'Clean ancilla': 8,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 1190400,
        'Idle cycles': 0,
    },
    'qc_add': {
        'Spatial cost': 2400,
        'Temporal cost': 480,
        'Toffoli cost': 8,
        'i-state cost': 34,
        'Clean ancilla': 8,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 1152000,
        'Idle cycles': 0,
    },
    'SWAP': {
        'Spatial cost': 1512,
        'Temporal cost': 174,
        'Toffoli cost': 4,
        'i-state cost': 22,
        'Clean ancilla': 0,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 263088,
        'Idle cycles': 0,
    },
    'ref': {
        'Spatial cost': 3440,
        'Temporal cost': 840,
        'Toffoli cost': 14,
        'i-state cost': 58,
        'Clean ancilla': 14,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 2889600,
        'Idle cycles': 0,
    },
    'SelectSwap_clean': {
        'Spatial cost': 4216,
        'Temporal cost': 4192,
        'Toffoli cost': 141,
        'i-state cost': 343,
        'Clean ancilla': 36,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 15707445,
        'Idle cycles': 0,
    },
    'SelectSwap_clean_K1': {
        'Spatial cost': 2064,
        'Temporal cost': 14614,
        'Toffoli cost': 476,
        'i-state cost': 954,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 20,
        'Qubit cycles': 30163296,
        'Idle cycles': 0,
    },
    'SelectSwap_dirty_batched': {
        'Spatial cost': 10840,
        'Temporal cost': 8366,
        'Toffoli cost': 306,
        'i-state cost': 1140,
        'Clean ancilla': 32,
        'Dirty ancilla': 96,
        'Buffer ancilla': 27,
        'Qubit cycles': 87140896,
        'Idle cycles': 0,
    },
    'fixup_space_space': {
        'Spatial cost': 2320,
        'Temporal cost': 1377,
        'Toffoli cost': 37,
        'i-state cost': 146,
        'Clean ancilla': 37,
        'Dirty ancilla': 0,
        'Buffer ancilla': 12,
        'Qubit cycles': 3194640,
        'Idle cycles': 0,
    },
    'fixup_space_time': {
        'Spatial cost': 1752,
        'Temporal cost': 16431,
        'Toffoli cost': 498,
        'i-state cost': 1236,
        'Clean ancilla': 2,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 21524464,
        'Idle cycles': 0,
    },
    'OuterPrep_A': {
        'Spatial cost': 8090,
        'Temporal cost': 14116,
        'Toffoli cost': 369,
        'i-state cost': 1087,
        'Clean ancilla': 22,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 104865120,
        'Idle cycles': 74424285,
    },
    'OuterPrep_dag_A': {
        'Spatial cost': 8090,
        'Temporal cost': 15507,
        'Toffoli cost': 400,
        'i-state cost': 1251,
        'Clean ancilla': 22,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 113776327,
        'Idle cycles': 81807492,
    },
    'OuterPrep_dirty_A': {
        'Spatial cost': 8333,
        'Temporal cost': 19132,
        'Toffoli cost': 544,
        'i-state cost': 1627,
        'Clean ancilla': 22,
        'Dirty ancilla': 27,
        'Buffer ancilla': 27,
        'Qubit cycles': 141308242,
        'Idle cycles': 94089099,
    },
    'InnerPrep_A': {
        'Spatial cost': 9103,
        'Temporal cost': 219441,
        'Toffoli cost': 7105,
        'i-state cost': 14903,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 1737395550,
        'Idle cycles': 1111395168,
    },
    'InnerPrep_dag_A': {
        'Spatial cost': 9103,
        'Temporal cost': 239210,
        'Toffoli cost': 7652,
        'i-state cost': 17221,
        'Clean ancilla': 37,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 1891435316,
        'Idle cycles': 1212375606,
    },
    'Maj_A': {
        'Spatial cost': 7706,
        'Temporal cost': 1800,
        'Toffoli cost': 28,
        'i-state cost': 126,
        'Clean ancilla': 10,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 13598640,
        'Idle cycles': 9537840,
    },
    'Spin_A_idle': {
        'Spatial cost': 7667,
        'Temporal cost': 1718,
        'Toffoli cost': 62,
        'i-state cost': 254,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 13136004,
        'Idle cycles': 4131348,
    },
    'RPrep_Rot_A': {
        'Spatial cost': 9225,
        'Temporal cost': 446194,
        'Toffoli cost': 13821,
        'i-state cost': 29906,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 3930756952,
        'Idle cycles': 2343022968,
    },
    'RPrep_Rot_A_dirty': {
        'Spatial cost': 10210,
        'Temporal cost': 312522,
        'Toffoli cost': 9313,
        'i-state cost': 28396,
        'Clean ancilla': 32,
        'Dirty ancilla': 32,
        'Buffer ancilla': 27,
        'Qubit cycles': 2996900822,
        'Idle cycles': 1318846230,
    },
    'BE_A': {
        'Spatial cost': 9225,
        'Temporal cost': 2736321,
        'Toffoli cost': 85723,
        'i-state cost': 186490,
        'Clean ancilla': 37,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 23240080115,
        'Idle cycles': 14180537397,
    },
    'BE_B': {
        'Spatial cost': 10836,
        'Temporal cost': 6183189,
        'Toffoli cost': 197753,
        'i-state cost': 415566,
        'Clean ancilla': 37,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 63863807531,
        'Idle cycles': 42688137141,
    },
    'ldpc_QROM': {
        'Spatial cost': 970,
        'Temporal cost': 12674,
        'Toffoli cost': 476,
        'i-state cost': 954,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 21,
        'Qubit cycles': 12293780,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 18,
        'Qubit cycles [ldpc]': 228132,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_adder': {
        'Spatial cost': 1520,
        'Temporal cost': 378,
        'Toffoli cost': 9,
        'i-state cost': 39,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 574560,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 27,
        'Qubit cycles [ldpc]': 10206,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_qc_comp': {
        'Spatial cost': 1544,
        'Temporal cost': 384,
        'Toffoli cost': 8,
        'i-state cost': 34,
        'Clean ancilla': 8,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 592896,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 10,
        'Qubit cycles [ldpc]': 3840,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_qc_add': {
        'Spatial cost': 1544,
        'Temporal cost': 384,
        'Toffoli cost': 8,
        'i-state cost': 34,
        'Clean ancilla': 8,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 592896,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 10,
        'Qubit cycles [ldpc]': 3840,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_SWAP': {
        'Spatial cost': 1080,
        'Temporal cost': 202,
        'Toffoli cost': 4,
        'i-state cost': 22,
        'Clean ancilla': 0,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 218160,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 7,
        'Qubit cycles [ldpc]': 1414,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_PlainSWAP': {
        'Spatial cost': 1906,
        'Temporal cost': 1770,
        'Toffoli cost': 54,
        'i-state cost': 220,
        'Clean ancilla': 0,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 3373620,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 109,
        'Qubit cycles [ldpc]': 192930,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_ref': {
        'Spatial cost': 2004,
        'Temporal cost': 672,
        'Toffoli cost': 14,
        'i-state cost': 58,
        'Clean ancilla': 14,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 1346688,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 16,
        'Qubit cycles [ldpc]': 10752,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_SelectSwap_clean': {
        'Spatial cost': 1330,
        'Temporal cost': 4408,
        'Toffoli cost': 141,
        'i-state cost': 343,
        'Clean ancilla': 36,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 4610180,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 45,
        'Qubit cycles [ldpc]': 178278,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_SelectSwap_dirty_batched': {
        'Spatial cost': 2158,
        'Temporal cost': 12402,
        'Toffoli cost': 306,
        'i-state cost': 1138,
        'Clean ancilla': 32,
        'Dirty ancilla': 96,
        'Buffer ancilla': 27,
        'Qubit cycles': 25471616,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 137,
        'Qubit cycles [ldpc]': 1649114,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_fixup_space_space': {
        'Spatial cost': 758,
        'Temporal cost': 1921,
        'Toffoli cost': 37,
        'i-state cost': 146,
        'Clean ancilla': 37,
        'Dirty ancilla': 0,
        'Buffer ancilla': 12,
        'Qubit cycles': 1456118,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 46,
        'Qubit cycles [ldpc]': 88366,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_fixup_space_time': {
        'Spatial cost': 1162,
        'Temporal cost': 15863,
        'Toffoli cost': 498,
        'i-state cost': 1220,
        'Clean ancilla': 2,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 14069306,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 11,
        'Qubit cycles [ldpc]': 132039,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_prep_shuffle': {
        'Spatial cost': 178,
        'Temporal cost': 176,
        'Toffoli cost': 0,
        'i-state cost': 0,
        'Clean ancilla': 0,
        'Dirty ancilla': 0,
        'Buffer ancilla': 0,
        'Qubit cycles': 31328,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 9,
        'Qubit cycles [ldpc]': 1584,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_rot_shuffle': {
        'Spatial cost': 276,
        'Temporal cost': 1312,
        'Toffoli cost': 0,
        'i-state cost': 0,
        'Clean ancilla': 0,
        'Dirty ancilla': 0,
        'Buffer ancilla': 0,
        'Qubit cycles': 362112,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 16,
        'Qubit cycles [ldpc]': 20992,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_rotation': {
        'Spatial cost': 388,
        'Temporal cost': 32,
        'Toffoli cost': 0,
        'i-state cost': 0,
        'Clean ancilla': 0,
        'Dirty ancilla': 0,
        'Buffer ancilla': 0,
        'Qubit cycles': 12416,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 17,
        'Qubit cycles [ldpc]': 544,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_mbu': {
        'Spatial cost': 178,
        'Temporal cost': 144,
        'Toffoli cost': 0,
        'i-state cost': 0,
        'Clean ancilla': 0,
        'Dirty ancilla': 0,
        'Buffer ancilla': 0,
        'Qubit cycles': 25632,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 9,
        'Qubit cycles [ldpc]': 1296,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_maj': {
        'Spatial cost': 1274,
        'Temporal cost': 672,
        'Toffoli cost': 12,
        'i-state cost': 58,
        'Clean ancilla': 5,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 856128,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 7,
        'Qubit cycles [ldpc]': 4704,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_OuterPrep_A': {
        'Spatial cost': 1520,
        'Temporal cost': 15986,
        'Toffoli cost': 375,
        'i-state cost': 1077,
        'Clean ancilla': 22,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 13719008,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 164,
        'Qubit cycles [ldpc]': 2504598,
        'Idle cycles [ldpc]': 2172750,
    },
    'ldpc_InnerPrep_A': {
        'Spatial cost': 1520,
        'Temporal cost': 198558,
        'Toffoli cost': 7116,
        'i-state cost': 14925,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 219194188,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 174,
        'Qubit cycles [ldpc]': 31567758,
        'Idle cycles [ldpc]': 25795776,
    },
    'ldpc_Spin_A': {
        'Spatial cost': 1906,
        'Temporal cost': 2154,
        'Toffoli cost': 62,
        'i-state cost': 254,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 3966516,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 143,
        'Qubit cycles [ldpc]': 308022,
        'Idle cycles [ldpc]': 111252,
    },
    'ldpc_Maj_A': {
        'Spatial cost': 1544,
        'Temporal cost': 1440,
        'Toffoli cost': 28,
        'i-state cost': 126,
        'Clean ancilla': 10,
        'Dirty ancilla': 0,
        'Buffer ancilla': 26,
        'Qubit cycles': 2041920,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 145,
        'Qubit cycles [ldpc]': 208032,
        'Idle cycles [ldpc]': 195648,
    },
    'ldpc_RPrep_Rot_A': {
        'Spatial cost': 2256,
        'Temporal cost': 436105,
        'Toffoli cost': 13851,
        'i-state cost': 29920,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 492693342,
        'Idle cycles': 30426624,
        'Spatial cost [ldpc]': 175,
        'Qubit cycles [ldpc]': 75297373,
        'Idle cycles [ldpc]': 58639534,
    },
    'ldpc_BE_A_d16': {
        'Spatial cost': 2256,
        'Temporal cost': 2653077,
        'Toffoli cost': 86377,
        'i-state cost': 188306,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 2970275380,
        'Idle cycles': 121706496,
        'Spatial cost [ldpc]': 175,
        'Qubit cycles [ldpc]': 444872613,
        'Idle cycles [ldpc]': 352235237,
    },
    'ldpc_BE_B_d22': {
        'Spatial cost': 3002,
        'Temporal cost': 7227227,
        'Toffoli cost': 200919,
        'i-state cost': 420950,
        'Clean ancilla': 30,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 10390213038,
        'Idle cycles': 267453120,
        'Spatial cost [ldpc]': 218,
        'Qubit cycles [ldpc]': 1539954794,
        'Idle cycles [ldpc]': 1279205826,
    },
    'ldpc_BE_A_d16_inner_dirty': {
        'Spatial cost': 2256,
        'Temporal cost': 2380837,
        'Toffoli cost': 59521,
        'i-state cost': 174146,
        'Clean ancilla': 32,
        'Dirty ancilla': 63,
        'Buffer ancilla': 27,
        'Qubit cycles': 3295551828,
        'Idle cycles': 121706496,
        'Spatial cost [ldpc]': 175,
        'Qubit cycles [ldpc]': 391266605,
        'Idle cycles [ldpc]': 244032973,
    },
    'ldpc_QROM_drep15_dLDPC22': {
        'Spatial cost': 954,
        'Temporal cost': 15818,
        'Toffoli cost': 476,
        'i-state cost': 954,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 21,
        'Qubit cycles': 15090372,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 18,
        'Qubit cycles [ldpc]': 284724,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_adder_drep15_dLDPC22': {
        'Spatial cost': 1512,
        'Temporal cost': 396,
        'Toffoli cost': 9,
        'i-state cost': 39,
        'Clean ancilla': 9,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 598752,
        'Idle cycles': 0,
        'Spatial cost [ldpc]': 27,
        'Qubit cycles [ldpc]': 10692,
        'Idle cycles [ldpc]': 0,
    },
    'ldpc_BE_A_drep13_dLDPC16': {
        'Spatial cost': 1998,
        'Temporal cost': 2653077,
        'Toffoli cost': 86377,
        'i-state cost': 188306,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 2621818324,
        'Idle cycles': 98150400,
        'Spatial cost [ldpc]': 175,
        'Qubit cycles [ldpc]': 444872613,
        'Idle cycles [ldpc]': 352235237,
    },
    'ldpc_BE_B_drep17_dLDPC12': {
        'Spatial cost': 2302,
        'Temporal cost': 6009234,
        'Toffoli cost': 200919,
        'i-state cost': 420950,
        'Clean ancilla': 30,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
        'Qubit cycles': 6734575288,
        'Idle cycles': 167634720,
        'Spatial cost [ldpc]': 218,
        'Qubit cycles [ldpc]': 1281023599,
        'Idle cycles [ldpc]': 1063405691,
    },
}

EXPECTED_TOTAL = {
    'calls': 33616,
    'total rounds': 91984166736,
    'total logical cycles': 4599208336.8,
    'total days': 1.0646315594444444,
    'total Toffoli': 2881664368,
    'total i-state': 6269047840,
    # idle is ~56% of the volume: the background zones are larger
    # than any single routine's compute zone, so this is the dominant term
    'total qubit cycles': 781238533145840,
    'total idle cycles': 476692945137552,
    # qubit cycles / d(2d-1), the rep code's conversion to logical qubit-cycles;
    # the rep layer is the only layer, so it is also the effective volume
    'total logical qubit cycles [rep]': 1001587863007.4872,
    'total effective volume': 1001587863007.4872,
    'Spatial cost': 9225,
    'Clean ancilla': 37,
    'Dirty ancilla': 0,
    'Buffer ancilla': 27,
}

# best_option at a compute zone of 57: qrom with lam = 3 (48 clean + 24
# corridors = 72) no longer fits once the corridors count
EXPECTED_BATCHING = {
    'rows': 39,
    'best method': 'qrom',
    'best K': 1,
    'best lam': 2,
    'best lookups': 27,
    'best lookup Toffoli': 12852,
    'best lookup rounds': 394578,
    'best Clean ancilla': 32,
    'best Access corridors': 16,
}


def _run(make):
    '''Budget-overflow warnings are part of the pinned behaviour, not failures.'''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return make()


def _fields(result):
    '''The pinned quantities of one result: its summary.'''
    return acc.summary(result)


@pytest.mark.parametrize('name', list(CASES), ids=list(CASES))
def test_cost_fields_are_unchanged(name):
    actual = _fields(_run(CASES[name]))
    expected = EXPECTED[name]
    # approx compares key sets too, so a composite that stopped reporting
    # 'Idle cycles' (or a leaf that started) fails here rather than silently
    assert actual == pytest.approx(expected)


TOTAL_CASE = lambda: hlc.Total_time(54, 27, 10, 27, 9, 16, 48, BUFFER, dict(d_rep=20), 21.4,
                                    epsilon=1e-3, round_time=1e-6,
                                    method='qrom', K=1, lam=2, model=cf, outer_variety='clean', inner_variety='clean')   # the option in use


def test_total_time_is_unchanged():
    total = _run(TOTAL_CASE)
    actual = {f: total[f] for f in EXPECTED_TOTAL}
    assert actual == pytest.approx(EXPECTED_TOTAL)
    assert total['layers'] == ['rep']


# the hybrid (LDPC) model's total, set A at its LDPC budget of 32, at each
# supported distance
TOTAL_LDPC_CASE = lambda d: hlc.Total_time(54, 27, 10, 27, 9, 16, 32, BUFFER, dict(d_rep=d, d_LDPC=d), 21.4,
                                         epsilon=1e-3, round_time=1e-6,
                                         method='qrom', K=1, lam=2, model=ldpc, outer_variety='clean', inner_variety='clean')

EXPECTED_TOTAL_LDPC = {
    12: {
        'calls': 33616,
        'total rounds': 75280846960,
        'total logical cycles': 6273403913.333333,
        'total days': 0.871306099074074,
        'total Toffoli': 2903649232,
        'total i-state': 6330094496,
        'total qubit cycles': 67638210701600,
        'total qubit cycles [ldpc]': 12641002895312,
        'total idle cycles': 2501056217088,
        'total idle cycles [ldpc]': 9985906064464,
        'Spatial cost': 1818,
        'Spatial cost [ldpc]': 175,
        'total logical qubit cycles [rep]': 245065980802.89853,
        'total logical qubit cycles [ldpc]': 1053416907942.6667,
        'total effective volume': 1298482888745.5652,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
    },
    16: {
        'calls': 33616,
        'total rounds': 89185836432,
        'total logical cycles': 5574114777,
        'total days': 1.0322434772222222,
        'total Toffoli': 2903649232,
        'total i-state': 6330094496,
        'total qubit cycles': 99848777174080,
        'total qubit cycles [ldpc]': 14954837758608,
        'total idle cycles': 4091285569536,
        'total idle cycles [ldpc]': 11840739726992,
        'Spatial cost': 2256,
        'Spatial cost [ldpc]': 175,
        'total logical qubit cycles [rep]': 201308018496.12903,
        'total logical qubit cycles [ldpc]': 934677359913,
        'total effective volume': 1135985378409.129,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
    },
    22: {
        'calls': 33616,
        'total rounds': 111638030064,
        'total logical cycles': 5074455912,
        'total days': 1.2921068294444444,
        'total Toffoli': 2903649232,
        'total i-state': 6330094496,
        'total qubit cycles': 157645676362592,
        'total qubit cycles [ldpc]': 18696476151936,
        'total idle cycles': 7253620678656,
        'total idle cycles [ldpc]': 14833056772864,
        'Spatial cost': 2898,
        'Spatial cost [ldpc]': 175,
        'total logical qubit cycles [rep]': 166644478184.55814,
        'total logical qubit cycles [ldpc]': 849839825088,
        'total effective volume': 1016484303272.5581,
        'Clean ancilla': 32,
        'Dirty ancilla': 0,
        'Buffer ancilla': 27,
    },
}


@pytest.mark.parametrize('d', [12, 16, 22])
def test_ldpc_total_time_is_unchanged(d):
    total = _run(lambda: TOTAL_LDPC_CASE(d))
    actual = {f: total[f] for f in EXPECTED_TOTAL_LDPC[d]}
    assert actual == pytest.approx(EXPECTED_TOTAL_LDPC[d])
    assert total['layers'] == ['rep', 'ldpc']


def test_batching_table_is_unchanged():
    rows = _run(lambda: bt.batching_table(54, 27, 10, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, model=cf))
    best = _run(lambda: bt.best_option(54, 27, 10, 16, dict(d_rep=20), 57, lams=LAMS, alpha_max=ALPHA_MAX, model=cf))
    actual = {'rows': len(rows),
              'best method': best['method'],
              'best K': best['K'],
              'best lam': best['lam'],
              'best lookups': best['lookups'],
              'best lookup Toffoli': best['lookup Toffoli'],
              'best lookup rounds': best['lookup rounds'],
              'best Clean ancilla': best['Clean ancilla'],
              'best Access corridors': best['Access corridors']}
    assert actual == pytest.approx(EXPECTED_BATCHING)


# every viable RPrep+Rot option for set A at each code's compute-zone budget
# (as in the notebook), as optimize.search_rprep_rot costs it (the whole
# routine): (method, K, lam, runs): (Toffoli cost, Temporal cost, compute zone)
EXPECTED_OPTIONS = {
    # rep code, d = 20, budget 48: no system access corridors, so no dirty
    # method; the output register's corridors (one per two qubits) count in
    # the zone
    'rep': {
        ('qrom', 1, 2, 0): (13779, 444431, 48),
        ('clean', 2, 1, 0): (14161, 461372, 48),
    },
    # LDPC, d = 16, budget 32: no corridors at all, every method available
    'ldpc': {
        ('dirty_restricted', 2, 2, 1): (9343, 415579, 32),
        ('dirty_full', 2, 2, 1): (9343, 415579, 32),
        ('qrom', 1, 2, 0): (13851, 436105, 32),
        ('dirty_full', 4, 2, 5): (10791, 493331, 32),
        ('clean', 2, 1, 0): (14233, 487671, 32),
    },
}


@pytest.mark.parametrize('name', list(EXPECTED_OPTIONS))
def test_set_a_options_are_unchanged(name):
    model, dist, budget = {'rep': (cf, dict(d_rep=20), 48), 'ldpc': (ldpc, dict(d_rep=16, d_LDPC=16), 32)}[name]
    r = _run(lambda: opt.search_rprep_rot(54, 27, 10, 27, 16, dist, lams=LAMS, alpha_max=ALPHA_MAX,
                                          ancilla_budget=budget, model=model))
    actual = {(c['method'], c['K'], c['size of batches'], c['runs']):
              (acc.summary(c)['Toffoli cost'], acc.summary(c)['Temporal cost'], c['Compute zone'])
              for c in [r['chosen']] + r['alternatives']}
    assert actual == EXPECTED_OPTIONS[name]
