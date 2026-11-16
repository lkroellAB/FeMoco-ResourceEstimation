"""
The hybrid (LDPC) cost model's own rules: where the zones live and in what
unit, how each layer converts to logical qubit-cycles, which distances it
supports, and a few footprints that must hold specific qubits.  The rules
every model shares (busy qubits inside footprints, the interface) are in
test_invariants.py; its pinned numbers are in test_regression.py.
"""
import warnings

import pytest

from resource_count import accounting as acc
from resource_count import cost_functions as cf
from resource_count import cost_functions_ldpc as ldpc
from resource_count import high_level_cost as hlc
from resource_count import layout

# parameter set A at LDPC's compute-zone budget of 32 (see test_regression.py)
SET_A = (54, 27, 10, 27, 9, 16, 32)


def total(d_rep, d_LDPC):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hlc.Total_time(*SET_A, layout.buffer_zone_size(), dict(d_rep=d_rep, d_LDPC=d_LDPC),
                              21.4, epsilon=1e-3,
                              round_time=1e-6, model=ldpc, method='qrom', K=1, lam=2, outer_variety='clean', inner_variety='clean')


class TestZones:
    def test_every_zone_but_the_parked_angles_lives_in_ldpc(self):
        assert {z: l for z, l in ldpc.ZONE_LAYER.items() if l != 'ldpc'} == {'parked': 'rep'}

    @pytest.mark.parametrize('zone', sorted(set(ldpc.ZONE_LAYER) - {'parked'}))
    def test_ldpc_zones_are_counted_in_logical_qubits(self, zone):
        assert ldpc.zone_qubits(zone, 7, 16, 16) == ('ldpc', 7)

    def test_parked_angles_are_counted_in_physical_qubits(self):
        # rep-code logical qubits under the system: 2d-1 physical each
        assert ldpc.zone_qubits('parked', 7, 16, 16) == ('rep', layout.physical(7, 16))

    def test_full_access_needs_no_corridors(self):
        # the output register and the system qubits live in LDPC
        assert ldpc.access_corridors(64) == 0
        assert ldpc.SYSTEM_Z_ACCESS is True
        # the rep code: one corridor per two qubits, none for the system
        assert cf.access_corridors(33) == 17
        assert cf.SYSTEM_Z_ACCESS is False


class TestConversion:
    def test_rep_layer_converts_exactly_as_in_the_rep_code(self):
        for d in (12, 16, 22):
            assert ldpc.logical_qubit_cycles('rep', 10**9, d, d) == \
                cf.logical_qubit_cycles('rep', 10**9, d)

    def test_ldpc_layer_is_logical_qubit_rounds_per_cycle(self):
        # logical qubits x rounds, divided by the d rounds of one cycle
        assert ldpc.rounds_per_cycle(16, 16) == 16
        assert ldpc.logical_qubit_cycles('ldpc', 1600, 16, 16) == 100

    @pytest.mark.parametrize('d_rep, d_LDPC', [(17, 12), (15, 22)])
    def test_each_layer_converts_with_its_own_distance(self, d_rep, d_LDPC):
        # each layer's error rate is per cycle of ITS OWN distance, so its
        # volume is divided by that distance's cycle, never by the max
        assert ldpc.logical_qubit_cycles('ldpc', 10**6, d_rep, d_LDPC) == 10**6/d_LDPC
        assert ldpc.logical_qubit_cycles('rep', 10**9, d_rep, d_LDPC) == \
            cf.logical_qubit_cycles('rep', 10**9, d_rep)

    @pytest.mark.parametrize('d_rep, d_LDPC', [(17, 12), (15, 22)])
    def test_a_logical_cycle_is_the_larger_distance(self, d_rep, d_LDPC):
        assert ldpc.rounds_per_cycle(d_rep, d_LDPC) == max(d_rep, d_LDPC)

    def test_unknown_layer_is_refused(self):
        with pytest.raises(ValueError, match='ldcp'):
            ldpc.logical_qubit_cycles('ldcp', 1, 16, 16)

    @pytest.mark.parametrize('d_rep, d_LDPC', [(12, 12), (16, 16), (22, 22), (17, 12), (15, 22)])
    def test_total_reports_both_layers_and_adds_them_only_converted(self, d_rep, d_LDPC):
        t = total(d_rep, d_LDPC)
        assert t['layers'] == ['rep', 'ldpc']
        assert t['total logical cycles'] == pytest.approx(t['total rounds']/max(d_rep, d_LDPC))
        assert t['total logical qubit cycles [ldpc]'] == pytest.approx(
            t['total qubit cycles [ldpc]']/d_LDPC)
        assert t['total logical qubit cycles [rep]'] == pytest.approx(
            t['total qubit cycles']/(d_rep*(2*d_rep - 1)))
        assert t['total effective volume'] == pytest.approx(
            t['total logical qubit cycles [rep]'] + t['total logical qubit cycles [ldpc]'])


class TestDistances:
    @pytest.mark.parametrize('d', [12, 16, 22])
    def test_supported_distances_run(self, d):
        assert total(d, d)['total effective volume'] > 0

    def test_any_other_ldpc_distance_is_refused(self):
        with pytest.raises(ValueError, match='distance'):
            ldpc.chip_size_LDPC(3, 20)
        with pytest.raises(ValueError, match='distance'):
            total(16, 20)

    def test_the_rep_distance_is_free(self):
        # only the LDPC chips come in fixed sizes
        assert total(20, 16)['total effective volume'] > 0

    def test_the_model_needs_both_distances_and_no_others(self):
        # the distances are unpacked into the model's own keywords, so a
        # missing or foreign one is refused by its signature
        with pytest.raises(TypeError, match='d_LDPC'):
            ldpc.adder_cost(16, **dict(d_rep=16))
        with pytest.raises(TypeError, match='d_LDPC'):
            cf.adder_cost(16, **dict(d_rep=16, d_LDPC=16))

    def test_time_follows_the_larger_distance_and_space_each_its_own(self):
        # the comparator: timed in max(d_rep, d_LDPC); its chip sized by
        # d_LDPC; its rep-code rows by d_rep; its LDPC qubits by neither
        a, b, c = (ldpc.qc_comp_cost(9, *ds) for ds in ((16, 16), (13, 16), (16, 12)))
        assert acc.total(a, 'rounds') == acc.total(b, 'rounds') == acc.total(c, 'rounds')
        assert a['space']['ldpc'] == b['space']['ldpc'] == c['space']['ldpc']
        L = 6       # the chip: ceil(9/2) + 1 columns
        assert a['space']['rep'] - b['space']['rep'] == 2*(2*9 + 26)*(16 - 13)
        assert a['space']['rep'] - c['space']['rep'] == 2*(ldpc.chip_size_LDPC(L, 16)
                                                           - ldpc.chip_size_LDPC(L, 12))

    @pytest.mark.parametrize('d', [12, 16, 22])
    def test_an_empty_chip_costs_nothing(self, d):
        assert ldpc.chip_size_LDPC(0, d) == 0


class TestFootprints:
    def test_shared_constants_are_the_rep_codes(self):
        assert (ldpc.TOFF_TIME, ldpc.T_REACT, ldpc.I_COUNT) == \
            (cf.TOFF_TIME, cf.T_REACT, cf.I_COUNT)

    def test_plain_swap_holds_both_halves_and_the_control(self):
        # the flattening's swap: n_b = 5 pairs (|c> and |b>) and the flag
        assert ldpc.PlainSWAP_cost(5, 16, 16)['space']['ldpc'] == 2*5 + 1
        # Spin's: the 2N system qubits and the spin qubit
        assert ldpc.PlainSWAP_cost(54, 16, 16)['space']['ldpc'] == 2*54 + 1

    def test_lookup_holds_the_address_bits_not_the_addresses(self):
        # 486 addresses are read through ceil(log2 486) = 9 bits
        assert ldpc.QROM_cost(486, 32, 16, 16)['space']['ldpc'] == 9 + 32

    @pytest.mark.parametrize('alpha', [1, 2, 3])
    def test_dirty_lookup_peak_holds_every_step_and_the_borrowed_copies(self, alpha):
        # the layout spans all K copies of the output (the K-1 borrowed system
        # qubits with the routing under them), so neither the Select nor the
        # swap network needs more, at any K
        target, d = 32, 16
        r = ldpc.SelectSwap_cost(486, target, alpha, d, d, 'dirty', True)
        K = 2**alpha
        select = ldpc.QROM_cost(486/K, K*target, d, d)['space']
        swap = ldpc.SWAP_cost(alpha, (K - 1)*target, d, d, qubits=K*target)['space']
        for layer in ('rep', 'ldpc'):
            assert r['space'][layer] >= max(select[layer], swap[layer])
        assert r['space']['ldpc'] >= 9 + K*target     # address bits + K copies

    def test_mbu_of_nothing_holds_nothing(self):
        # an MBU of no qubits needs no chip and no LDPC qubits
        assert ldpc.mbu_cost(0, 16, 16)['space'] == {'rep': 0, 'ldpc': 0}

    def test_angle_storage_must_fit_under_the_system_qubits(self):
        # b_rot*d = 16*16 = 256 physical data qubits, against the routing at
        # d = 16: 7*(N+1-4) under the system qubits (4 columns go to the
        # adder) plus 7*(ceil((n_x+n_b)/2) + 2) under the address columns
        ldpc.rot_shuffle_cost(16, 2, 16, 16, N=54, n_x=9, n_b=5)    # 357 + 63 = 420: fits
        with pytest.raises(ValueError, match='storage'):
            ldpc.rot_shuffle_cost(16, 2, 16, 16, N=10, n_x=9, n_b=5)  # 49 + 63 = 112 < 256

    def test_angle_storage_is_rep_qubits_under_an_ldpc_sized_routing(self):
        # the angles are stored at d_rep (b_rot*d_rep data qubits); the
        # routing under the chip is sized by d_LDPC: 7*(20+1-4) + 7*(7+2) =
        # 182 at d_LDPC = 16
        ldpc.rot_shuffle_cost(16, 2, 11, 16, N=20, n_x=9, n_b=5)      # 176 <= 182
        with pytest.raises(ValueError, match='storage'):
            ldpc.rot_shuffle_cost(16, 2, 13, 16, N=20, n_x=9, n_b=5)  # 208 > 182
