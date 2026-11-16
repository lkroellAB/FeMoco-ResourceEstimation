"""
error_accounting.py: the error model, the distance choice and the physical
qubit count, as the notebook and scripts/error_accounting_latex.py use them.
The physical qubit counts and the chosen distances are pinned at the paper's
settings (FeMoCo-54 and -76, resource_count.settings' SETTINGS_54 / SETTINGS_76).
"""
import warnings

import numpy as np
import pytest

from resource_count import cost_functions as cf
from resource_count import cost_functions_ldpc as ldpc
from resource_count import error_accounting as ea
from resource_count import settings as st

FEMOCO_54 = dict(N=54, B=27, R=10, C=27, b_coef=9, b_rot=16, be_const=21.4)
FEMOCO_76 = dict(N=76, B=57, R=15, C=19, b_coef=9, b_rot=15, be_const=43.7)
REP_OPTION = dict(method='qrom', K=1, lam=2, outer_variety='clean', inner_variety='clean')
LDPC_54_OPTION = dict(method='dirty_restricted', K=2, lam=2, outer_variety='clean', inner_variety='dirty')
LDPC_76_OPTION = dict(method='dirty_full', K=4, lam=2, outer_variety='clean', inner_variety='dirty')


class TestPhysicalQubits:
    """The global-layout counts, pinned at the paper's chosen distances."""

    @pytest.mark.parametrize('system, model, option, distances, total', [
        (FEMOCO_54, cf, REP_OPTION, dict(d_rep=15), 11192),
        (FEMOCO_54, cf, REP_OPTION, dict(d_rep=17), 12072),
        (FEMOCO_76, cf, REP_OPTION, dict(d_rep=15), 12568),
        (FEMOCO_76, cf, REP_OPTION, dict(d_rep=17), 13612),
        (FEMOCO_54, ldpc, LDPC_54_OPTION, dict(d_rep=15, d_LDPC=16), 6808),
        (FEMOCO_54, ldpc, LDPC_54_OPTION, dict(d_rep=17, d_LDPC=22), 7280),
        (FEMOCO_76, ldpc, LDPC_76_OPTION, dict(d_rep=15, d_LDPC=16), 7462),
        (FEMOCO_76, ldpc, LDPC_76_OPTION, dict(d_rep=19, d_LDPC=22), 8130),
    ])
    def test_total(self, system, model, option, distances, total):
        assert ea.physical_qubits(system, model, option, distances)['total'] == total

    def test_factories_per_model(self):
        # the rep code: three |i> factories; LDPC one fewer, and its routing
        # 10 rows instead of 12 under the |C_X C_X X> factory, without 2*11
        assert ea.FACTORY_ROUTING == {cf: 1180, ldpc: 1020}
        assert ea.FACTORIES == {cf: 1566 + 3*196, ldpc: 1566 + 2*196} == {cf: 2154, ldpc: 1958}

    def test_noise_matches_the_paper(self):
        # p_magic: Table III, third layout (1.48e-11 at p_Z = 1e-3, 2.02e-13 at 5e-4).
        # p_i: Table XIV, d_c = 7 medium protocol (3.0e-12 at 1e-3, 9.4e-13 at 5e-4;
        # the code keeps one more digit at 5e-4)
        assert ea.NOISE == {'5e-4': dict(p_Z=5e-4, p_magic=2.02e-13, p_i=9.43e-13),
                            '1e-3': dict(p_Z=1e-3, p_magic=1.48e-11, p_i=3.0e-12)}

    def test_terms_add_up(self):
        # the rep code: 2 Space_algo + factory routing + factories
        p = ea.physical_qubits_rep(FEMOCO_54, REP_OPTION, d_rep=15)
        algo = dict(p['terms'])['Space_algo']
        assert p['total'] == 2*algo + ea.FACTORY_ROUTING[cf] + ea.FACTORIES[cf] == 2*3929 + 3334
        # LDPC: the LDPC layer + 2 x the repetition-code layer + its factory routing + its factories
        p = ea.physical_qubits_ldpc(FEMOCO_54, LDPC_54_OPTION, d_rep=15, d_LDPC=16)
        ldpc_layer, rep_layer = p['terms'][0][1], dict(p['terms'])['repetition-code layer']
        assert p['total'] == ldpc_layer + 2*rep_layer + ea.FACTORY_ROUTING[ldpc] + ea.FACTORIES[ldpc]
        assert ea.FACTORY_ROUTING[ldpc] + ea.FACTORIES[ldpc] == 2978

    def test_ldpc_compute_zone_grows_with_the_angles_held(self):
        # clean SelectSwap holds K*lam angles: 2*2*16 = 64 compute qubits,
        # more than fit beside the Fourier state in ceil(3*16/2) = 24 columns
        clean = dict(LDPC_54_OPTION, method='clean')
        p = ea.physical_qubits_ldpc(FEMOCO_54, clean, d_rep=15, d_LDPC=16)['params']
        assert p['n_comp'] == 64 and p['l_comp'] == (16 + 64)//2

    def test_unknown_model_refused(self):
        with pytest.raises(ValueError, match='no physical qubit count'):
            ea.physical_qubits(FEMOCO_54, ea.layout, REP_OPTION, dict(d_rep=15))


def eps(p, volume):
    '''1 - (1 - p)**V, for a rate p ~ 1e-16: ln(1 - p) = -p to 1e-16 relative,
    so 1 - exp(-p V), with expm1.  Written out directly, 1 - (1 - p)**V is
    off by about a percent, as 1 - p cannot be stored that precisely'''
    return -np.expm1(-p*volume)


class TestErrorModel:

    def test_ldpc_rate_only_at_supported_distances(self):
        with pytest.raises(ValueError, match='unsupported LDPC distance'):
            ea.logical_error_ldpc(14, 5e-4)

    def test_failure_is_the_three_added(self):
        cost = ea.total_time(FEMOCO_54, 48, REP_OPTION, dict(d_rep=15), cf, ea.EPSILON, ea.ROUND_TIME)
        noise = ea.NOISE['5e-4']
        total, logical, magic, istate = ea.failure(cost, dict(d_rep=15), noise, cf, ea.FACTORY_ROUTING[cf])
        assert total == logical + magic + istate
        assert magic == cost['total Toffoli']*noise['p_magic']
        assert istate == cost['total i-state']*noise['p_i']
        # the rep code alone: eps_alg is its error, the factory routing in its volume
        split = ea.layer_errors(cost, dict(d_rep=15), noise['p_Z'], cf, ea.FACTORY_ROUTING[cf])
        volume = split['rep']['logical qubit-cycles'] + split['factory routing']['logical qubit-cycles']
        assert ea.code_errors(cost, dict(d_rep=15), noise['p_Z'], cf, ea.FACTORY_ROUTING[cf]) == \
            {'rep': pytest.approx(eps(split['rep']['rate'], volume), rel=1e-9)}
        assert logical > -np.expm1(split['rep']['logical qubit-cycles']*np.log1p(-split['rep']['rate']))

    def test_ldpc_codes_are_added(self):
        # eps_alg = eps_rep + eps_LDPC, each 1 - (1 - p)**V; the rep code's
        # volume includes the factory routing
        dist = dict(d_rep=15, d_LDPC=16)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cost = ea.total_time(FEMOCO_54, 32, LDPC_54_OPTION, dist, ldpc, ea.EPSILON, ea.ROUND_TIME)
        p_Z = ea.NOISE['5e-4']['p_Z']
        split = ea.layer_errors(cost, dist, p_Z, ldpc, ea.FACTORY_ROUTING[ldpc])
        codes = ea.code_errors(cost, dist, p_Z, ldpc, ea.FACTORY_ROUTING[ldpc])
        v_rep = split['rep']['logical qubit-cycles'] + split['factory routing']['logical qubit-cycles']
        assert codes['rep'] == pytest.approx(eps(split['rep']['rate'], v_rep), rel=1e-9)
        assert codes['ldpc'] == pytest.approx(eps(split['ldpc']['rate'], split['ldpc']['logical qubit-cycles']),
                                              rel=1e-9)
        assert ea.algo_error(cost, dist, p_Z, ldpc, ea.FACTORY_ROUTING[ldpc]) == codes['rep'] + codes['ldpc']
        # pinned at the paper's distances
        assert codes['rep'] == pytest.approx(0.000341, abs=5e-7)
        assert codes['ldpc'] == pytest.approx(0.003190, abs=5e-7)


class TestDistanceChoice:
    """resource_count.settings' chosen distances are the distance choice's,
    for every molecule, model and noise level -- the paper's tables and the
    notebook all use them."""

    @pytest.mark.parametrize('molecule', list(st.MOLECULES))
    @pytest.mark.parametrize('model', [cf, ldpc])
    def test_settings_are_the_choice(self, molecule, model):
        system, settings = st.MOLECULES[molecule]
        s, cache = settings[model], {}

        def run(dist):
            key = tuple(sorted(dist.items()))
            if key not in cache:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cache[key] = ea.total_time(system, s['budget'], s['option'], dist, model,
                                               ea.EPSILON, ea.ROUND_TIME)
            return cache[key]
        for level, noise in ea.NOISE.items():
            rows, c = ea.choose_distance(run, ea.DISTANCE_SCAN[model], noise, model, ea.FACTORY_ROUTING[model], ea.TIE)
            assert c['dist'] == s['distances'][level], (molecule, s['name'], level)
            # nothing feasible is more than TIE shorter
            assert all(r['expected'] >= c['expected']/(1 + ea.TIE) for r in rows if 'infeasible' not in r)

    def test_pinned_femoco_54(self):
        # the paper's FeMoCo-54 distances, pinned independently of the settings
        assert st.SETTINGS_54[cf]['distances'] == {'5e-4': dict(d_rep=15), '1e-3': dict(d_rep=17)}
        assert st.SETTINGS_54[ldpc]['distances'] == {'5e-4': dict(d_rep=15, d_LDPC=16),
                                                     '1e-3': dict(d_rep=17, d_LDPC=22)}


class TestRunVolumes:
    """run_volumes: the run's logical cycles and effective volumes."""

    def test_parts_and_total(self):
        dist = dict(d_rep=15, d_LDPC=16)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cost = ea.total_time(FEMOCO_54, 32, LDPC_54_OPTION, dist, ldpc, ea.EPSILON, ea.ROUND_TIME)
        v = ea.run_volumes(cost, dist, ldpc, ea.FACTORY_ROUTING[ldpc], ea.FACTORIES[ldpc])
        assert v['cycles'] == cost['total rounds']/max(dist.values())
        assert v['algorithm'] == cost['total effective volume']
        # the factory routing's volume is the one the error model uses
        split = ea.layer_errors(cost, dist, ea.NOISE['5e-4']['p_Z'], ldpc, ea.FACTORY_ROUTING[ldpc])
        assert v['factory routing'] == split['factory routing']['logical qubit-cycles']
        # the factories: the same conversion, in proportion to their qubits
        assert v['factories'] == pytest.approx(v['factory routing']*ea.FACTORIES[ldpc]/ea.FACTORY_ROUTING[ldpc], rel=1e-12)
        assert v['total'] == v['algorithm'] + v['factory routing'] + v['factories']

    @pytest.mark.parametrize('model, budget, option, dist', [
        (cf, 48, REP_OPTION, dict(d_rep=15)),
        (ldpc, 32, LDPC_54_OPTION, dict(d_rep=17, d_LDPC=22)),
    ])
    def test_logical_qubits_per_layer(self, model, budget, option, dist):
        # each layer's volume over its own logical cycles (rounds / its
        # distance); the rep layer's volume includes the factories
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cost = ea.total_time(FEMOCO_54, budget, option, dist, model, ea.EPSILON, ea.ROUND_TIME)
        v = ea.run_volumes(cost, dist, model, ea.FACTORY_ROUTING[model], ea.FACTORIES[model])
        rounds = cost['total rounds']
        rep_volume = v['algorithm [rep]'] + v['factory routing'] + v['factories']
        assert v['logical qubits [rep]'] == pytest.approx(rep_volume*dist['d_rep']/rounds, rel=1e-12)
        if model is ldpc:
            assert v['logical qubits [ldpc]'] == pytest.approx(v['algorithm [ldpc]']*dist['d_LDPC']/rounds, rel=1e-12)
        assert v['logical qubits'] == sum(v[f'logical qubits [{L}]'] for L in cost['layers'])
        if model is cf:
            # one layer: the total volume over the run's logical cycles
            assert v['logical qubits'] == pytest.approx(v['total']/v['cycles'], rel=1e-12)
