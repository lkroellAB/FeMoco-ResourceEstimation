"""
Correctness/invariant tests: properties that should hold regardless of how the
cost functions are implemented internally (degenerate cases, boundary
agreement, error handling). Unlike test_regression.py, these are not pinned to
today's exact numbers -- they check real relationships in the model.
"""
import warnings

import numpy as np
import pytest

from resource_count import accounting as acc
from resource_count import cost_functions as cf
from resource_count import cost_functions_ldpc as ldpc
from resource_count import high_level_cost as hlc
from resource_count import batching as bt
from resource_count import layout


# the preps' required zone sizes (parameter set A: X = 486, B = 27)
OUTER = dict(N=54, b_rot=16, B=27)

# RPrep+Rot batch sizes searched in these tests: lam = 1 .. 8
LAMS = range(1, 9)
# swap parameters searched: K = 2**alpha up to 2**4 = 16
ALPHA_MAX = 4
INNER = dict(N=54, b_rot=16, X=486)

S = acc.summary


def vol(r, part):
    '''the 'rep' volume of one part of a record'''
    return r['parts'][part]['volume']['rep']


def _assert_zero(result):
    '''costs nothing: the empty record, apart from any labels'''
    assert {k: result[k] for k in acc.empty()} == acc.empty()


class TestDegenerateInputs:
    """A single-address lookup table, an empty register, etc. is a classical
    constant / no-op and must cost nothing."""

    def test_qrom_single_address(self):
        _assert_zero(cf.QROM_cost(1, 5, 20))
        _assert_zero(cf.QROM_cost(0, 5, 20))

    def test_adder_zero_bits(self):
        _assert_zero(cf.adder_cost(0, 20))
        _assert_zero(cf.adder_cost(-3, 20))

    def test_qc_comp_at_most_one_bit(self):
        _assert_zero(cf.qc_comp_cost(1, 20))
        _assert_zero(cf.qc_comp_cost(0, 20))

    def test_qc_add_at_most_one_bit(self):
        _assert_zero(cf.qc_add_cost(1, 20))

    def test_swap_no_control_or_no_swapands(self):
        _assert_zero(cf.SWAP_cost(0, 4, 20))
        _assert_zero(cf.SWAP_cost(3, 0, 20))

    def test_reflection_zero_bits(self):
        _assert_zero(cf.ref(0, 20))

    def test_phase_fixup_address_at_most_two(self):
        _assert_zero(cf.phase_fixup_cost(2, 20))
        _assert_zero(cf.phase_fixup_cost(1, 20))


class TestSelectSwapDegeneratesToQROM:
    """K=1 (alpha=0) means no swap network is used, so a clean, unbatched
    SelectSwap must reduce to a plain QROM load with a no-op swap."""

    def test_matches_plain_qrom_on_time_toffoli_space(self):
        address, target_bit, d = 50, 16, 20
        plain = cf.QROM_cost(address, target_bit, d)
        degenerate = cf.SelectSwap_cost(address, target_bit, 0, d, 'clean', False)

        for field in ('rounds', 'toffoli', 'i-states'):
            assert acc.total(degenerate, field) == acc.total(plain, field)
        assert degenerate['space'] == plain['space']
        assert degenerate['parts'] == plain['parts']

    def test_k_is_one(self):
        r = cf.SelectSwap_cost(50, 16, 0, 20, 'clean', False)
        assert r['K'] == 1


class TestSeqOnRealRecords:
    """Chaining two real primitives: their ancilla is the peak of the two,
    not the sum (the rule itself is tested in test_accounting.py)."""

    def test_ancilla_is_the_peak(self):
        load = cf.SelectSwap_cost(50, 16, 2, 20, 'dirty', True)
        fixup = cf.phase_fixup_cost(50, 20, ancilla_budget=1000)
        both = acc.seq(load, fixup)
        for k in acc.ANCILLA:
            assert both[k] == max(load[k], fixup[k])


class TestDirtySelectSwapVariety:
    """OuterPrep/InnerPrep (via _sequential_prep) support a dirty SelectSwap
    variety, using a fixed dirty budget (2*N) instead of drawing K-1 copies
    from the clean ancilla budget. Unlike RPrep+Rot, no batching applies."""

    def test_outer_prep_dirty_reports_nonzero_dirty_ancilla(self):
        # unbatched dirty SelectSwap pays 2 Select + 4 Swap, so alpha>0 only
        # pays off once the address is large relative to the target width --
        # true here (address=2000, bit_precision=4), not for small levels.
        big = dict(N=2000, b_rot=16, B=27)
        r = hlc.OuterPrep(2000, 4, 5700, dict(d_rep=20), variety='dirty', **big, model=cf)
        assert r['variety'] == 'dirty'
        assert r['Dirty ancilla'] > 0

    def test_outer_prep_dirty_degenerates_to_alpha_zero_for_small_levels(self):
        # for small/narrow levels, unbatched dirty's 4x swap multiplier
        # outweighs any Select savings, so the search correctly settles on
        # alpha=0 (no borrowing) rather than paying for a swap network.
        r = hlc.OuterPrep(54, 9, 57, dict(d_rep=20), variety='dirty', **OUTER, model=cf)
        assert r['Dirty ancilla'] == 0

    def test_outer_prep_clean_has_zero_dirty_ancilla(self):
        r = hlc.OuterPrep(54, 9, 57, dict(d_rep=20), variety='clean', **OUTER, model=cf)
        assert r['Dirty ancilla'] == 0

    def test_dirty_budget_defaults_to_twice_n(self):
        # a dirty_budget passed explicitly as 2*N must match the default
        explicit = hlc.OuterPrep(54, 9, 57, dict(d_rep=20), variety='dirty', dirty_budget=2*54, **OUTER, model=cf)
        via_n = hlc.OuterPrep(54, 9, 57, dict(d_rep=20), variety='dirty', **OUTER, model=cf)
        assert explicit == via_n

    def test_inner_prep_dirty_reports_nonzero_dirty_ancilla(self):
        r = hlc.InnerPrep(270, 57, 9, 57, dict(d_rep=20), variety='dirty', **INNER, model=cf)
        assert r['variety'] == 'dirty'
        assert r['Dirty ancilla'] > 0

    def test_tighter_dirty_budget_never_cheaper(self):
        # less borrowable ancilla can only restrict the swap split, never help
        big = dict(N=2000, b_rot=16, B=27)
        generous = hlc.OuterPrep(2000, 4, 5700, dict(d_rep=20), variety='dirty', dirty_budget=1000, **big, model=cf)
        tight = hlc.OuterPrep(2000, 4, 5700, dict(d_rep=20), variety='dirty', dirty_budget=0, **big, model=cf)
        assert S(tight)['Toffoli cost'] >= S(generous)['Toffoli cost']
        assert tight['Dirty ancilla'] == 0


class TestForcedPrepAlpha:
    """alpha forces K = 2**alpha at every level of a prep (capped by each
    level's address width); None leaves each level to pick its own."""

    def test_none_is_the_default(self):
        assert hlc.OuterPrep(486, 9, 48, dict(d_rep=20), alpha=None, **OUTER, model=cf) == \
            hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **OUTER, model=cf)

    def test_alpha_zero_is_a_plain_qrom(self):
        # no swap network anywhere: its lookups are QROMs, and the forward
        # fixup over 256 addresses fits the budget's space-space split
        r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), alpha=0, **OUTER, model=cf)
        assert r['K'] == 1
        assert 'swap' not in r['parts']

    @pytest.mark.parametrize('variety', ['clean', 'dirty'])
    def test_lookups_are_the_only_part_that_depends_on_k(self, variety):
        # 'lookups' is the prep's SelectSwaps alone (one per level, K capped
        # by the level); everything else costs the same whatever K is
        sizes = [2**i for i in range(9)]           # OuterPrep over 486 addresses
        rest = set()
        for alpha in range(4):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')     # large K is over budget: expected
                r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), alpha=alpha, variety=variety, **OUTER,
                                  model=cf)
            loads = acc.seq(*[cf.SelectSwap_cost(L, 9, min(alpha, int(np.log2(L))), 20, variety)
                              for L in sizes])
            for field in ('toffoli', 'rounds'):
                assert acc.total(r['lookups'], field) == acc.total(loads, field)
            rest.add((S(r)['Toffoli cost'] - acc.total(r['lookups'], 'toffoli'),
                      S(r)['Temporal cost'] - acc.total(r['lookups'], 'rounds')))
        assert len(rest) == 1

    def test_levels_record_each_lookups_addresses_and_k(self):
        # OuterPrep over 486 addresses: levels of 1, 2, ..., 256 addresses;
        # forced K = 4 is capped to 1 and 2 on the first two
        r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), alpha=2, **OUTER, model=cf)
        assert r['levels'] == [(1, 1), (2, 2)] + [(2**i, 4) for i in range(2, 9)]
        # chosen by the optimizer (alpha=None): within the budget's K
        r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **OUTER, model=cf)
        assert [L for L, _ in r['levels']] == [2**i for i in range(9)]
        assert max(K for _, K in r['levels']) <= 2**hlc.alpha_budget(48, 9, model=cf)

    def test_k_is_forced_and_capped_by_the_largest_level(self):
        assert hlc.OuterPrep(486, 9, 48, dict(d_rep=20), alpha=2, **OUTER, model=cf)['K'] == 4
        # OuterPrep's largest level has 256 addresses: K stops at 256
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert hlc.OuterPrep(486, 9, 48, dict(d_rep=20), alpha=10, **OUTER, model=cf)['K'] == 256


class TestBlockEncodingAncillaPeak:
    def test_be_cost_dirty_ancilla_is_peak_across_subroutines(self):
        # cross-check BE_cost's own combinator use against an independent
        # recomputation from its constituent calls
        N, B, R, C, b_coef, b_rot, budget, d = 76, 57, 15, 19, 9, 15, 75, 20
        be = hlc.BE_cost(N, B, R, C, b_coef, b_rot, budget, dict(d_rep=d),
                         method='dirty_restricted', K=2, lam=2, model=cf, outer_variety='clean', inner_variety='clean')

        n_r = int(np.ceil(np.log2(R)))
        X = N + 2**n_r*C
        n_X = int(np.ceil(np.log2(X)))
        n_B = int(np.ceil(np.log2(B+1)))
        children = [
            hlc.OuterPrep(X, b_coef, budget, dict(d_rep=d), N=N, b_rot=b_rot, B=B, model=cf),
            hlc.OuterPrep(X, b_coef, budget, dict(d_rep=d), adjoint=True, N=N, b_rot=b_rot, B=B, model=cf),
            hlc.InnerPrep(2**n_r*C, B, b_coef, budget, dict(d_rep=d), N=N, b_rot=b_rot, X=X, model=cf),
            hlc.InnerPrep(2**n_r*C, B, b_coef, budget, dict(d_rep=d), adjoint=True, N=N, b_rot=b_rot, X=X, model=cf),
            hlc.Spin(X, N, dict(d_rep=d), b_rot=b_rot, B=B, model=cf),
            hlc.Maj(X, B, dict(d_rep=d), N=N, b_rot=b_rot, model=cf),
            hlc.RPrep_Rot(N, B, R, C, b_rot, dict(d_rep=d), method='dirty_restricted', K=2,
                          lam=2, ancilla_budget=budget, model=cf),
            hlc.Ref('InnerRef', dict(d_rep=d), N=N, b_rot=b_rot, X=X, B=B,
                    busy={'address': n_B + 1}, flag=True, model=cf),
            hlc.Ref('OuterRef', dict(d_rep=d), N=N, b_rot=b_rot, X=X, B=B,
                    busy={'address': n_X + n_B + 2, 'QPE': 2}, flag=False, model=cf),
        ]
        expected = max(c['Dirty ancilla'] for c in children)
        assert be['Dirty ancilla'] == expected


class TestPhaseFixupSplitSelection:
    def test_ample_budget_prefers_space_space(self):
        r = cf.phase_fixup_cost(50, 20, ancilla_budget=1000)
        assert r['split'] == 'space-space'

    def test_no_budget_falls_back_to_space_time(self):
        r = cf.phase_fixup_cost(50, 20, ancilla_budget=None)
        assert r['split'] == 'space-time'

    def test_tight_budget_falls_back_to_space_time(self):
        # the space-space split for address=50 needs several clean ancilla;
        # a budget of 0 cannot hold it.
        r = cf.phase_fixup_cost(50, 20, ancilla_budget=0)
        assert r['split'] == 'space-time'

    def test_explicit_split_is_honoured(self):
        assert cf.phase_fixup_cost(50, 20, split='space-space')['split'] == 'space-space'
        assert cf.phase_fixup_cost(50, 20, split='space-time')['split'] == 'space-time'

    # the table sizes the routines' fixups run over (OuterPrep's levels,
    # InnerPrep's for both parameter sets, RPrep+Rot's Y), and the
    # distances the comparisons use
    FIXUP_ADDRESSES = (4, 8, 16, 32, 64, 128, 256, 304, 432, 486, 608, 864, 988,
                       1216, 1728, 2432, 3456, 4864, 6912, 9728)
    FIXUP_MODELS = [('rep', dict(d_rep=d)) for d in (11, 13, 15, 17, 21)] + \
                   [('ldpc', dict(d_rep=dr, d_LDPC=dl)) for dr, dl in
                    ((11, 12), (15, 16), (16, 16), (17, 22), (22, 22))]

    @pytest.mark.parametrize('model, distances', FIXUP_MODELS,
                             ids=[f"{m}-{'-'.join(map(str, d.values()))}" for m, d in FIXUP_MODELS])
    def test_default_takes_the_faster_split_that_fits(self, model, distances):
        # with the space-space split's clean ancilla as the budget both splits
        # fit, and the default returns the one with fewer rounds (MBU of the
        # products included); one ancilla less and only the space-time split
        # fits
        from resource_count import cost_functions_ldpc as ldpc
        m = {'rep': cf, 'ldpc': ldpc}[model]
        buffer = layout.buffer_zone_size()
        for address in self.FIXUP_ADDRESSES:
            budget = m.phase_fixup_cost(address, **distances, split='space-space')['Clean ancilla']
            rounds = {s: acc.total(m.phase_fixup_cost(address, **distances, ancilla_budget=budget,
                                                      buffer_budget=buffer, split=s), 'rounds')
                      for s in ('space-space', 'space-time')}
            chosen = m.phase_fixup_cost(address, **distances, ancilla_budget=budget, buffer_budget=buffer)
            assert acc.total(chosen, 'rounds') == min(rounds.values()), address
            assert chosen['split'] == min(rounds, key=rounds.get), address
            tight = m.phase_fixup_cost(address, **distances, ancilla_budget=budget - 1, buffer_budget=buffer)
            assert tight['split'] == 'space-time', address


class TestErrorHandling:
    def test_selectswap_rejects_unknown_variety(self):
        with pytest.raises(ValueError):
            cf.SelectSwap_cost(50, 16, 2, 20, variety='bogus')

    def test_phase_fixup_rejects_unknown_split(self):
        with pytest.raises(ValueError):
            cf.phase_fixup_cost(50, 20, split='bogus')


class TestWarnings:
    def test_alpha_budget_warns_and_clamps_when_budget_too_small(self):
        with pytest.warns(UserWarning):
            alpha = hlc.alpha_budget(ancilla_budget=5, target_bit=10, model=cf)
        assert alpha == 0

    def test_alpha_budget_silent_when_budget_sufficient(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            hlc.alpha_budget(ancilla_budget=100, target_bit=10, model=cf)

    def test_alpha_budget_counts_corridors_only_where_the_model_needs_them(self):
        # 48 holds K = 2 copies of 9 bits with their 9 corridors in the rep
        # code (K = 4 would be 36 + 18 = 54), but K = 4 without corridors in LDPC
        from resource_count import cost_functions_ldpc as ldpc
        assert hlc.alpha_budget(48, 9, model=cf) == 1
        assert hlc.alpha_budget(48, 9, model=ldpc) == 2

    def test_rprep_rot_warns_when_the_compute_zone_overflows(self):
        # qrom with lam = 3: 48 clean ancilla fit 48, but not with their 24
        # corridors
        with pytest.warns(UserWarning, match='compute zone of 72'):
            r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='qrom', K=1, lam=3,
                              ancilla_budget=48, model=cf)
        assert r['Clean ancilla'] <= 48
        assert r['Compute zone'] == 72

    def test_rprep_rot_compute_zone_needs_no_corridors_in_ldpc(self):
        from resource_count import cost_functions_ldpc as ldpc
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=16, d_LDPC=16), method='qrom', K=1, lam=3,
                              ancilla_budget=48, model=ldpc)
        assert r['Compute zone'] == 48


class TestQromAncilla:
    def test_boundary_values(self):
        # address<=1 is the classical-constant degenerate case: no buffer
        # needed either, since there is no QROM circuit at all
        assert cf.qrom_ancilla(0) == 0
        assert cf.qrom_ancilla(1) == 0
        # address>1: skew-tree working ancilla + the fixed 12-qubit buffer
        assert cf.qrom_ancilla(2) == 0 + 12
        assert cf.qrom_ancilla(3) == 1 + 12

    def test_monotonic_nondecreasing(self):
        values = [cf.qrom_ancilla(a) for a in range(1, 200)]
        assert all(b >= a for a, b in zip(values, values[1:]))


class TestFactoriesRequired:
    def test_zero_time_is_zero_factories(self):
        assert cf.factories_required(acc.empty(), 20) == 0.0

    def test_scales_inversely_with_rate(self):
        cost = cf.adder_cost(9, 20)
        base = cf.factories_required(cost, 20, rate=1)
        doubled_rate = cf.factories_required(cost, 20, rate=2)
        assert doubled_rate == pytest.approx(base / 2)


class TestBlockEncodingComposition:
    """BE_cost chains its routine calls one after another: its totals are the
    calls times each routine's own totals, its space and ancilla are the
    largest single routine's."""

    def test_totals_are_calls_times_routine_totals(self):
        be = hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=20), method='qrom', K=1, lam=3, model=cf, outer_variety='clean', inner_variety='clean')
        s = S(be)
        assert s['Temporal cost'] == sum(r['calls']*r['rounds'] for r in be['breakdown'])
        assert s['Toffoli cost'] == sum(r['calls']*r['toffoli'] for r in be['breakdown'])
        assert s['i-state cost'] == sum(r['calls']*r['i-states'] for r in be['breakdown'])

    def test_space_is_the_largest_routine(self):
        N, B, R, C, b_coef, b_rot, budget, d = 54, 27, 10, 27, 9, 16, 48, 20
        X, n_X, n_B = 486, 9, 5
        be = hlc.BE_cost(N, B, R, C, b_coef, b_rot, budget, dict(d_rep=d), method='qrom', K=1, lam=3, model=cf, outer_variety='clean', inner_variety='clean')
        routines = [
            hlc.OuterPrep(X, b_coef, budget, dict(d_rep=d), N=N, b_rot=b_rot, B=B, model=cf),
            hlc.InnerPrep(432, B, b_coef, budget, dict(d_rep=d), N=N, b_rot=b_rot, X=X, model=cf),
            hlc.InnerPrep(432, B, b_coef, budget, dict(d_rep=d), adjoint=True, N=N, b_rot=b_rot, X=X, model=cf),
            hlc.Spin(X, N, dict(d_rep=d), b_rot=b_rot, B=B, model=cf),
            hlc.Maj(X, B, dict(d_rep=d), N=N, b_rot=b_rot, model=cf),
            hlc.RPrep_Rot(N, B, R, C, b_rot, dict(d_rep=d), method='qrom', K=1, lam=3,
                          ancilla_budget=budget, model=cf),
            hlc.Ref('OuterRef', dict(d_rep=d), N=N, b_rot=b_rot, X=X, B=B,
                    busy={'address': n_X + n_B + 2, 'QPE': 2}, flag=False, model=cf),
        ]
        assert S(be)['Spatial cost'] == max(S(r)['Spatial cost'] for r in routines)


class TestSpaceBreakdown:
    """Every routine's space is its peak step's footprint plus the idle
    zones beside it -- a second, independent route to each space number."""

    N, B, R, C, b_coef, b_rot, budget, d, X = 54, 27, 10, 27, 9, 16, 48, 20, 486

    def routines(self):
        N, B, R, C, b_coef, b_rot, budget, d, X = (self.N, self.B, self.R, self.C, self.b_coef,
                                                     self.b_rot, self.budget, self.d, self.X)
        return [
            hlc.OuterPrep(X, b_coef, budget, dict(d_rep=d), N=N, b_rot=b_rot, B=B, model=cf),
            hlc.InnerPrep(432, B, b_coef, budget, dict(d_rep=d), adjoint=True, N=N, b_rot=b_rot, X=X, model=cf),
            hlc.Spin(X, N, dict(d_rep=d), b_rot=b_rot, B=B, model=cf),
            hlc.Maj(X, B, dict(d_rep=d), N=N, b_rot=b_rot, model=cf),
            hlc.RPrep_Rot(N, B, R, C, b_rot, dict(d_rep=d), method='qrom', K=1, lam=2, ancilla_budget=budget, model=cf),
            hlc.RPrep_Rot(N, B, R, C, b_rot, dict(d_rep=d), method='dirty_restricted', K=2, lam=2, model=cf),
            hlc.Ref('OuterRef', dict(d_rep=d), N=N, b_rot=b_rot, X=X, B=B,
                    busy={'address': 9 + 5 + 2, 'QPE': 2}, flag=False, model=cf),
            hlc.BE_cost(N, B, R, C, b_coef, b_rot, budget, dict(d_rep=d), method='qrom', K=1, lam=2, model=cf, outer_variety='clean', inner_variety='clean'),
        ]

    def test_footprint_plus_idle_is_the_space(self):
        for r in self.routines():
            p = r['peak']['rep']
            assert p['footprint'] + sum(p['idle'].values()) == r['space']['rep'], r['Subroutine']

    def test_spin_is_set_by_its_comparator(self):
        # the comparator's footprint (its 26 buffer slots included) plus the
        # system register, Fourier state, QPE and the address zone bar the 9
        # X bits it reads; the flag is its output, so busy
        # (see test_spin_space_is_the_larger_...)
        p = hlc.Spin(486, 54, dict(d_rep=20), b_rot=16, B=27, model=cf)['peak']['rep']
        assert p['step'] == 'comparator'
        assert p['footprint'] == cf.qc_comp_cost(9, 20)['space']['rep']
        assert p['idle'] == {'system': layout.physical(108, 20), 'fourier': layout.physical(16, 20),
                             'QPE': layout.physical(2, 20), 'address': layout.physical(16 - 9, 20)}

    def test_block_encoding_space_is_the_largest_routine_in_its_breakdown(self):
        be = self.routines()[-1]
        assert max(space['rep'] for _, space, _ in be['space breakdown']) == be['space']['rep']


class TestRPrepRotAppliesTheGivenOption:
    """RPrep_Rot applies the (method, K, lam, runs) option it is handed and
    never searches for one -- choosing an option belongs to batching.py and
    optimize.py. Everything the option implies (variety, lookup count, where
    the swap ancilla come from) is read off batching.batching_cost, so the
    two cannot disagree about what a method means."""

    OPTION = dict(method='dirty_restricted', K=4, lam=3)

    def test_given_k_is_applied_verbatim(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), **self.OPTION, model=cf)
        assert r['K'] == 4
        assert r['method'] == 'dirty_restricted'

    def test_qrom_method_forces_k_to_one(self):
        # 'qrom' and K=1 are the same thing, so each implies the other --
        # otherwise method='qrom', K=2 builds a swap network, calls it 'qrom',
        # and is then reported infeasible for having no swap ancilla budget
        with warnings.catch_warnings():
            warnings.simplefilter('error')   # must not warn about feasibility
            r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='qrom', K=2, lam=2, model=cf)
        assert r['K'] == 1
        assert r['method'] == 'qrom'

    def test_k_one_degenerates_every_method_to_qrom(self):
        # no swap network means nothing is borrowed and nothing has to be
        # restored, so the method label must come back as 'qrom'
        r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='dirty_full', K=1, lam=3, model=cf)
        assert r['K'] == 1
        assert r['method'] == 'qrom'
        assert r['variety'] == 'clean'
        assert r['Dirty ancilla'] == 0

    def test_lookup_count_matches_batching(self):
        # dirty_restricted pays one unmerged trailing Select; qrom pays none
        dirty = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='dirty_restricted',
                              K=2, lam=3, model=cf)
        qrom = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='qrom', K=1, lam=3, model=cf)
        assert dirty['lookups'] == dirty['J'] + 1
        assert qrom['lookups'] == qrom['J']

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='bogus', K=2, lam=3, model=cf)

    def test_non_power_of_two_k_is_rejected(self):
        with pytest.raises(ValueError):
            hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='clean', K=3, lam=3, model=cf)

    def test_infeasible_option_warns_rather_than_costing_silently(self):
        # K=16 at lam=3 needs 15*3*16 = 720 borrowed qubits, far past N=54
        with pytest.warns(UserWarning, match='swap ancilla'):
            hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='dirty_restricted',
                          K=16, lam=3, model=cf)


class TestIdleBackground:
    """Idle qubits are charged per step (layout.idle_except): everything
    present and not busy adds to the step's space and, as an 'idle *' part,
    to its volume.  Idle parts never add rounds, Toffolis or i-states."""

    ARGS = OUTER

    def test_prep_idle_accounting_is_mandatory(self):
        with pytest.raises(TypeError):
            hlc.OuterPrep(486, 9, 48, dict(d_rep=20), model=cf)
        with pytest.raises(TypeError):
            hlc.InnerPrep(432, 27, 9, 48, dict(d_rep=20), model=cf)

    def test_idle_is_counted(self):
        r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **self.ARGS, model=cf)
        assert S(r)['Idle cycles'] > 0

    def test_idle_shows_up_as_its_own_parts_categories(self):
        r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **self.ARGS, model=cf)
        idle = {k: v for k, v in r['parts'].items() if k.startswith('idle ')}
        assert set(idle) == {'idle system', 'idle address', 'idle QPE', 'idle fourier',
                             'idle output'}
        assert all(v['rounds'] == 0 and v['toffoli'] == 0 and v['i-states'] == 0
                   for v in idle.values())

    def test_system_zone_idles_for_the_whole_prep(self):
        r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **self.ARGS, model=cf)
        expected = layout.physical(layout.system_zone_size(54), 20)*S(r)['Temporal cost']
        assert vol(r, 'idle system') == expected

    # OuterPrep(486, b_coef=9): 9 levels, each with a shuffle, an addition
    # and a rotation
    LEVELS, B_COEF, D = 9, 9, 20

    def test_prep_output_register_is_idle_during_the_shuffle_and_rotation(self):
        # the rotation is costed with the output still live
        r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **self.ARGS, model=cf)
        shuffle = acc.total(cf.prep_shuffle_cost(self.B_COEF, self.D), 'rounds')
        rotation = acc.total(cf.rotation_cost(self.B_COEF, self.D), 'rounds')
        expected = layout.physical(self.B_COEF, self.D)*(shuffle + rotation)*self.LEVELS
        assert vol(r, 'idle output') == expected

    def test_prep_fourier_idles_b_rot_minus_b_coef_during_shuffle_addition_rotation(self):
        # the b_coef Fourier qubits are busy in the shuffle (being moved), in
        # the addition (inside the adder) and in the rotation (its control);
        # the full b_rot idles otherwise
        r = hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **self.ARGS, model=cf)
        shuffle = acc.total(cf.prep_shuffle_cost(self.B_COEF, self.D), 'rounds')
        rotation = acc.total(cf.rotation_cost(self.B_COEF, self.D), 'rounds')
        add_rounds = acc.total(cf.adder_cost(self.B_COEF, self.D), 'rounds')
        partial = self.LEVELS*(shuffle + add_rounds + rotation)
        expected = (layout.physical(16 - self.B_COEF, self.D)*partial
                    + layout.physical(16, self.D)*(S(r)['Temporal cost'] - partial))
        assert vol(r, 'idle fourier') == expected

    def test_prep_space_covers_the_fixup_and_what_idles_beside_it(self):
        # InnerPrep's forward fixup (over its largest level, 432*16
        # addresses) runs in the space-time branch, which reads its 13 address
        # bits; everything else present idles beside it
        r = hlc.InnerPrep(432, 27, 9, 48, dict(d_rep=20), **INNER, model=cf)
        fixup = cf.phase_fixup_cost(432*16, 20, 48)
        assert fixup['split'] == 'space-time'
        idle = 108 + 16 + layout.QPE_QUBITS + (layout.address_zone_size(486, 27) - 13)
        assert S(r)['Spatial cost'] >= fixup['space']['rep'] + layout.physical(idle, 20)

    def test_spin_idle_accounting_is_mandatory(self):
        with pytest.raises(TypeError):
            hlc.Spin(486, 54, dict(d_rep=20), model=cf)

    def test_spin_system_register_is_idle_during_the_comparator_only(self):
        # during the swap the system register is inside the swap's footprint;
        # during the comparator it sits idle beside it
        r = hlc.Spin(486, 54, dict(d_rep=20), b_rot=16, B=27, model=cf)
        comp_rounds = acc.total(cf.qc_comp_cost(9, 20), 'rounds')
        expected = layout.physical(layout.system_zone_size(54), 20)*comp_rounds
        assert vol(r, 'idle system') == expected

    def test_spin_space_is_the_larger_of_its_two_steps(self):
        # each step: its own footprint plus the zones idle beside it
        d, zone = 20, layout.address_zone_size(486, 27)
        swap = cf.PlainSWAP_cost(54, d)['space']['rep']
        comp = cf.qc_comp_cost(9, d)['space']['rep']
        swap_step = swap + layout.physical(16 + 2 + (zone - 1) + 1, d)
        comp_step = comp + layout.physical(108 + 16 + 2 + (zone - 9), d)
        r = hlc.Spin(486, 54, dict(d_rep=d), b_rot=16, B=27, model=cf)
        assert S(r)['Spatial cost'] == max(swap_step, comp_step)

    def test_rprep_rot_always_counts_idle(self):
        # RPrep+Rot already has N, B, R, C and b_rot in its signature, so
        # there is no configuration in which its idle goes uncounted
        r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='dirty_restricted',
                          K=2, lam=2, model=cf)
        assert S(r)['Idle cycles'] > 0

    def test_rprep_rot_system_register_idles_except_one_qubit_per_rotation(self):
        # the system register is live for the whole run and idle throughout,
        # except the one system qubit each rotation acts on (2d rounds each)
        d, lam = 20, 2
        r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=d), method='qrom', K=1, lam=lam, model=cf)
        rotating = r['J']*lam*2*d
        expected = (layout.physical(108, d)*(S(r)['Temporal cost'] - rotating)
                    + layout.physical(108 - 1, d)*rotating)
        assert vol(r, 'idle system') == expected

    @pytest.mark.parametrize('lam', [1, 2, 3, 5])
    def test_rprep_rot_angle_idling(self, lam):
        # per batch: the shuffle moves one angle at a time, so lam-1 idle;
        # then "lambda-1 rotation angles are idle during the first addition,
        # lambda-2 during the second and so on"; during the rotation that
        # follows addition k, angle k is costed as still present, so lam-k+1;
        # during its MBU after that rotation angle k is busy, so lam-k.
        # Only the second angle is parked, from the shuffle until the first
        # addition (its rotation and MBU) is done; every other idle angle
        # waits in the output register
        b_rot, d, N = 16, 20, 54
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = hlc.RPrep_Rot(N, 27, 10, 27, b_rot, dict(d_rep=d), method='qrom', K=1, lam=lam, model=cf)
        add_rounds = acc.total(cf.adder_cost(b_rot, d), 'rounds')
        shuffle_rounds = acc.total(cf.rot_shuffle_cost(b_rot, lam, d, N=N, n_x=9, n_b=5), 'rounds')
        mbu_rounds = acc.total(cf.mbu_cost(b_rot, d), 'rounds')
        p = 1 if lam >= 2 else 0        # the parked second angle
        parked = p*b_rot*(shuffle_rounds + add_rounds + 2*d + mbu_rounds)
        output = b_rot*((lam - 1 - p)*shuffle_rounds
                        + (lam - 1 - p)*add_rounds + (lam - p)*2*d + (lam - 1 - p)*mbu_rounds
                        + sum((lam - k)*add_rounds + (lam - k + 1)*2*d + (lam - k)*mbu_rounds
                              for k in range(2, lam + 1)))
        got = lambda part: r['parts'].get(part, {'volume': {'rep': 0}})['volume']['rep']
        assert got('idle parked') == layout.physical(1, d)*parked*r['J']
        assert got('idle output') == layout.physical(1, d)*output*r['J']
        # the angles' idle volume as a whole is the same however it splits:
        # lam-1 in the shuffle, lam-k (+1 in the rotation) for angle k
        whole = b_rot*((lam - 1)*shuffle_rounds
                       + sum((lam - k)*(add_rounds + mbu_rounds) + (lam - k + 1)*2*d for k in range(1, lam + 1)))
        assert got('idle parked') + got('idle output') == layout.physical(1, d)*whole*r['J']

    def test_ldpc_parks_angles_in_the_rep_layer(self):
        # parked angles wait in the rep layer; angle k, idle during its own
        # rotation, is back in the LDPC output register
        from resource_count import cost_functions_ldpc as ldpc
        b_rot, d, lam = 16, 16, 3
        r = hlc.RPrep_Rot(54, 27, 10, 27, b_rot, dict(d_rep=d, d_LDPC=d), method='qrom', K=1, lam=lam, model=ldpc)
        assert set(r['parts']['idle parked']['volume']) == {'rep'}
        assert set(r['parts']['idle output']['volume']) == {'ldpc'}

    @pytest.mark.parametrize('model', ['rep', 'ldpc'])
    def test_dirty_lookup_marks_its_borrowed_system_qubits_busy(self, model):
        # the (K-1)*lam*b_rot = 32 borrowed system qubits are inside the
        # lookup's footprint, not idle beside it
        from resource_count import cost_functions_ldpc as ldpc
        m, dist = {'rep': (cf, dict(d_rep=20)), 'ldpc': (ldpc, dict(d_rep=15, d_LDPC=16))}[model]
        r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dist, method='dirty_restricted', K=2, lam=2, model=m)
        load = bt.batching_cost(54, 27, 10, 16, dist, 'dirty_restricted', lam=2, K=2, model=m)['load']
        # the whole system idles throughout, except 1 qubit in each of the
        # J*lam rotations (2d rounds, d the logical cycle) and the 32
        # borrowed in each lookup
        d = m.rounds_per_cycle(**dist)
        layer, full = m.zone_qubits('system', 108, **dist)
        one, borrowed = m.zone_qubits('system', 1, **dist)[1], m.zone_qubits('system', 32, **dist)[1]
        expected = (full*S(r)['Temporal cost'] - one*r['J']*2*2*d
                    - borrowed*r['lookups']*acc.total(load, 'rounds'))
        assert r['parts']['idle system']['volume'][layer] == expected

    def test_maj_idle_accounting_is_mandatory(self):
        with pytest.raises(TypeError):
            hlc.Maj(486, 27, dict(d_rep=20), model=cf)

    def test_maj_flag_idles_throughout(self):
        # neither the comparators nor the hard-coded part keep the flag busy
        r = hlc.Maj(486, 27, dict(d_rep=20), N=54, b_rot=16, model=cf)
        assert vol(r, 'idle flag') == layout.physical(1, 20)*S(r)['Temporal cost']

    def test_maj_address_idle_follows_the_busy_register(self):
        # X (9 bits) busy in the two comparators, B (5 bits) in the rest
        d, zone = 20, layout.address_zone_size(486, 27)
        r = hlc.Maj(486, 27, dict(d_rep=d), N=54, b_rot=16, model=cf)
        comp_rounds = acc.total(cf.qc_comp_cost(9, d), 'rounds')
        rest = S(r)['Temporal cost'] - 2*comp_rounds
        expected = layout.physical(zone - 9, d)*2*comp_rounds + layout.physical(zone - 5, d)*rest
        assert vol(r, 'idle address') == expected

    # set A: n_X = 9, n_B = 5
    REF = dict(N=54, b_rot=16, X=486, B=27)

    def test_outer_ref_reflects_about_its_controls_too(self):
        # X, B, both selection qubits and the 2 QPE controls: 9 + 5 + 2 + 2,
        # bit - 2 Toffolis
        r = hlc.Ref('OuterRef', dict(d_rep=20), busy={'address': 9 + 5 + 2, 'QPE': 2},
                    flag=False, **self.REF, model=cf)
        assert S(r)['Toffoli cost'] == 18 - 2
        # the whole address zone and QPE are busy, and there is no flag
        assert {'idle address', 'idle QPE', 'idle flag'}.isdisjoint(r['parts'])

    def test_inner_ref_leaves_qpe_and_flag_idle(self):
        r = hlc.Ref('InnerRef', dict(d_rep=20), busy={'address': 5 + 1}, flag=True, **self.REF, model=cf)
        assert S(r)['Toffoli cost'] == 6 - 2
        assert vol(r, 'idle QPE') == layout.physical(2, 20)*S(r)['Temporal cost']
        assert vol(r, 'idle flag') == layout.physical(1, 20)*S(r)['Temporal cost']

    def test_every_routine_in_the_block_encoding_has_idle_qubits(self):
        N, B, R, C, b_coef, b_rot, budget, d = 54, 27, 10, 27, 9, 16, 48, 20
        X = 486
        for r in (hlc.OuterPrep(X, b_coef, budget, dict(d_rep=d), N=N, b_rot=b_rot, B=B, model=cf),
                  hlc.InnerPrep(432, B, b_coef, budget, dict(d_rep=d), N=N, b_rot=b_rot, X=X, model=cf),
                  hlc.Spin(X, N, dict(d_rep=d), b_rot=b_rot, B=B, model=cf),
                  hlc.Maj(X, B, dict(d_rep=d), N=N, b_rot=b_rot, model=cf),
                  hlc.RPrep_Rot(N, B, R, C, b_rot, dict(d_rep=d), method='qrom', K=1, lam=3, model=cf),
                  hlc.Ref('InnerRef', dict(d_rep=d), N=N, b_rot=b_rot, X=X, B=B,
                          busy={'address': 6}, flag=True, model=cf)):
            assert S(r)['Idle cycles'] > 0, r['Subroutine']

    def test_be_cost_idle_dominates_the_volume(self):
        # the background is larger than any single routine's compute zone, so
        # if idle were a rounding correction something would be wrong
        be = hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=20),
                         method='dirty_restricted', K=2, lam=2, model=cf, outer_variety='clean', inner_variety='clean')
        assert S(be)['Idle cycles'] > 0.25*S(be)['Qubit cycles']

    def test_idle_parts_never_add_rounds_or_i_states(self):
        for r in (hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **self.ARGS, model=cf),
                  hlc.Spin(486, 54, dict(d_rep=20), b_rot=16, B=27, model=cf),
                  hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20),
                                method='dirty_restricted', K=2, lam=2, model=cf)):
            for name, p in r['parts'].items():
                if name.startswith('idle '):
                    assert (p['rounds'], p['toffoli'], p['i-states']) == (0, 0, 0)


class TestBestOptionPicksCheapest:
    """best_option ranks on the lookup table alone ('lookup rounds' /
    'lookup Toffoli'), which is why those keys are named that way -- the
    per-batch work is not in them. Ranking whole routines is
    optimize.search_rprep_rot's job."""

    def test_cheaper_than_every_other_feasible_row(self):
        # the budget bounds the compute zone: clean ancilla plus corridors
        ancilla_budget = 57
        best = bt.best_option(54, 27, 10, 16, dict(d_rep=20), ancilla_budget, lams=LAMS, alpha_max=ALPHA_MAX, model=cf)
        feasible = [r for r in bt.batching_table(54, 27, 10, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, model=cf)
                    if r['Clean ancilla'] + r['Access corridors'] <= ancilla_budget]
        assert best['lookup rounds'] == min(r['lookup rounds'] for r in feasible)

    def test_respects_custom_metric(self):
        ancilla_budget = 57
        best = bt.best_option(54, 27, 10, 16, dict(d_rep=20), ancilla_budget, lams=LAMS, alpha_max=ALPHA_MAX, metric='lookup Toffoli', model=cf)
        feasible = [r for r in bt.batching_table(54, 27, 10, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, model=cf)
                    if r['Clean ancilla'] + r['Access corridors'] <= ancilla_budget]
        assert best['lookup Toffoli'] == min(r['lookup Toffoli'] for r in feasible)

    def test_lookup_costs_are_not_named_like_whole_routine_costs(self):
        # guards against the rename being undone: a bare 'Temporal cost' here
        # would be silently mistaken for a full RPrep+Rot cost
        row = bt.batching_cost(54, 27, 10, 16, dict(d_rep=20), 'dirty_restricted', lam=3, K=2, model=cf)
        assert 'Temporal cost' not in row
        assert 'Toffoli cost' not in row

    def test_none_when_nothing_fits(self):
        assert bt.best_option(54, 27, 10, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, ancilla_budget=0, model=cf) is None


class TestBatchingMethodsAndOption:
    """The cost model's layout decides which batching methods are viable (no
    dirty ones for the rep code, whose layout has no access corridors for the
    system qubits), and as_option passes a chosen row to RPrep_Rot
    unchanged."""

    def test_rep_code_has_no_viable_dirty_method(self):
        rows = bt.batching_table(54, 27, 10, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, model=cf)
        assert cf.SYSTEM_Z_ACCESS is False
        assert {r['method'] for r in rows} <= {'qrom', 'clean'}

    def test_rejected_dirty_rows_say_why(self):
        rows = bt.batching_table(54, 27, 10, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX,
                                 include_rejected=True, model=cf)
        dirty = [r for r in rows if r['method'].startswith('dirty')]
        assert dirty
        assert all('no system Z access' in r['rejected'] and not r['viable'] for r in dirty)

    def test_rejected_rows_list_every_failed_condition(self):
        # lam = 1 fails the adder layout; K = 16 borrows 15*16 = 240 > N = 54
        row = bt.batching_cost(54, 27, 10, 16, dict(d_rep=20), 'dirty_restricted', lam=1, K=16, model=cf)
        assert row['rejected'] == ('no system Z access', 'swap source', 'adder layout')
        # a clean option that does not fit is rejected once, for its compute zone
        row = bt.batching_cost(54, 27, 10, 16, dict(d_rep=20), 'clean', lam=2, K=2, clean_budget=48, model=cf)
        assert row['rejected'] == ('compute zone',)

    def test_dirty_corridors_under_the_system(self):
        # restricted access: only the borrowed (K-1)*lam*b_rot = 32 qubits;
        # full access: all 2N = 108 system qubits
        restricted = bt.batching_cost(54, 27, 10, 16, dict(d_rep=20), 'dirty_restricted', lam=2, K=2, model=cf)
        full = bt.batching_cost(54, 27, 10, 16, dict(d_rep=20), 'dirty_full', lam=2, K=2, model=cf)
        assert restricted['System corridors'] == 16
        assert full['System corridors'] == 54
        # neither is part of the compute zone: output register + its corridors
        assert restricted['Compute zone'] == full['Compute zone'] == 32 + 16

    def test_as_option_passes_the_chosen_row_unchanged(self):
        best = bt.best_option(54, 27, 10, 16, dict(d_rep=20), 57, lams=LAMS, alpha_max=ALPHA_MAX, model=cf)
        r = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), **bt.as_option(best), model=cf)
        assert (r['method'], r['K'], r['size of batches'], r['lookups']) == \
            (best['method'], best['K'], best['lam'], best['lookups'])
        # and an RPrep_Rot result gives back the same option
        assert bt.as_option(r) == bt.as_option(best)


class TestCleanAndBufferAreSeparateZones:
    """'Clean ancilla' and 'Buffer ancilla' are two independent zones with
    independent sizes, never summed into one number -- 'Clean ancilla' is
    the fungible/compute-zone demand only, 'Buffer ancilla' is the
    restricted-use buffer-zone demand only. Any computation of "how much is
    left for a different simultaneous purpose" checks each zone against its
    own budget separately, never a combined total."""

    def test_qrom_clean_is_the_output_register_not_the_buffer(self):
        # QROM's compute-zone need is its own output register (target_bit,
        # freshly allocated) -- entirely separate from its buffer need
        r = cf.QROM_cost(10, 5, 20)
        assert r['Clean ancilla'] == 5          # target_bit
        assert r['Buffer ancilla'] == cf.qrom_ancilla(10)
        assert r['Clean ancilla'] != r['Buffer ancilla']

    # a SWAP's buffer slots: 8 per catalysis state (buff = 3 by default) plus
    # the 3 qubits storing the corrections -- the whole buffer zone
    SWAP_BUFFER = 8*3 + 3

    def test_swap_needs_no_clean_ancilla(self):
        # SWAP doesn't allocate a new register -- it swaps existing ones --
        # and its 3 correction qubits are buffer slots, whatever the
        # register sizes
        for r in (cf.SWAP_cost(3, 4, 20), cf.SWAP_cost(10, 500, 20),
                  cf.PlainSWAP_cost(1, 20), cf.PlainSWAP_cost(76, 20),
                  ldpc.SWAP_cost(3, 4, 15, 16), ldpc.PlainSWAP_cost(76, 15, 16)):
            assert r['Clean ancilla'] == 0
            assert r['Buffer ancilla'] == self.SWAP_BUFFER

    @pytest.mark.parametrize('swap', [lambda **kw: cf.SWAP_cost(3, 21, 20, **kw),
                                      lambda **kw: ldpc.SWAP_cost(3, 21, 15, 16, **kw)],
                             ids=['rep', 'ldpc'])
    def test_swap_space_follows_the_held_qubits_not_the_ctrl_swaps(self, swap):
        # a SelectSwap over K = 8 registers of b = 3 qubits: 7*3 = 21
        # Ctrl-Swaps, but 8*3 = 24 qubits held.  The held qubits set the
        # space; the Ctrl-Swaps alone set rounds, Toffolis and i-states
        gates_only, held = swap(), swap(qubits=24)
        assert swap(qubits=21) == gates_only            # default: qubits = swapands
        for layer in gates_only['space']:
            assert held['space'][layer] > gates_only['space'][layer]
        for field in ('rounds', 'toffoli', 'i-states'):
            assert acc.total(held, field) == acc.total(gates_only, field)

    def test_selectswap_swap_step_holds_every_register(self):
        # the swap step inside SelectSwap is sized for all K registers
        d, target, alpha = 20, 9, 2
        K = 2**alpha
        r = cf.SelectSwap_cost(64, target, alpha, d, 'clean')
        swap = cf.SWAP_cost(alpha, (K - 1)*target, d, qubits=K*target)
        assert r['parts']['swap']['volume'] == swap['parts']['swap']['volume']

    @pytest.mark.parametrize('maj, d', [(lambda n_b: cf.maj_cost(n_b, 15), 15),
                                        (lambda n_b: ldpc.maj_cost(n_b, 15, 16), 16)],
                             ids=['rep', 'ldpc'])
    @pytest.mark.parametrize('n_b', [1, 5, 9])
    def test_maj_counts(self, maj, d, n_b):
        # I_COUNT per Toffoli (the 2*n_b of the ladder and the 2 CCZ), 2 for
        # the CZ, 2 for each of the 3 H, and the 2 catalysts.  Rounds: the
        # 2 CCZ and the CZ at 3d each, the 3 H at d each, the ladder at
        # TOFF_TIME*d per Toffoli (d = max(d_rep, d_LDPC))
        r = maj(n_b)
        assert acc.total(r, 'toffoli') == 2*n_b + 2
        assert acc.total(r, 'i-states') == cf.I_COUNT*(2*n_b + 2) + 2 + 3*2 + 2
        assert acc.total(r, 'rounds') == (2*3 + 3 + 3)*d + 2*n_b*cf.TOFF_TIME*d

    def test_swap_buffer_is_size_independent(self):
        small = cf.SWAP_cost(1, 1, 20)
        large = cf.SWAP_cost(10, 500, 20)
        assert small['Buffer ancilla'] == large['Buffer ancilla'] == self.SWAP_BUFFER

    def test_swap_buffer_fits_the_buffer_zone(self):
        assert self.SWAP_BUFFER <= layout.buffer_zone_size()

    @pytest.mark.parametrize('address', [4, 16, 128, 4096, 2**16])
    def test_selectswap_adds_the_swap_buffer_the_qrom_region_lacks(self, address):
        # the clean SelectSwap's layout is the QROM's plus the swap network's
        # buffer slots beyond the QROM's buffer region, d rows, doubled
        d, target, alpha = 20, 9, 1
        swap = cf.SWAP_cost(alpha, (2**alpha - 1)*target, d)
        missing = max(swap['Buffer ancilla'] - cf.qrom_ancilla(address), 0)
        layout_space = cf.QROM_cost(address, 2**alpha*target, d)['space']['rep']
        r = cf.SelectSwap_cost(address, target, alpha, d, 'clean')
        assert r['space']['rep'] == layout_space + 2*missing*d
        assert (missing > 0) == (address < 2**16)   # region n-1+12 holds the swap's 27 from n = 16

    def test_adder_clean_and_buffer_are_independent_fields(self):
        # adder has both a fungible portion (the working bits) and a
        # restricted buffer portion, reported separately -- not summed
        r = cf.adder_cost(9, 20)
        assert r['Clean ancilla'] == 9      # working bits only
        assert r['Buffer ancilla'] == 27    # 9*buff (buff = 3), independent of the above

    def test_adder_buffer_scales_as_9_times_buff(self):
        # 36 at the default buff=4 is not a coincidence -- it must track buff
        # when overridden, not stay pinned to the default's value
        assert cf.adder_cost(9, 20, buff=4)['Buffer ancilla'] == 36
        assert cf.adder_cost(9, 20, buff=1)['Buffer ancilla'] == 9
        assert cf.adder_cost(9, 20, buff=10)['Buffer ancilla'] == 90

    def test_selectswap_clean_is_exactly_the_output_register(self):
        # 'Clean ancilla' is already just the fungible output register --
        # no buffer subtraction needed, because none was ever added in
        address, target_bit, alpha, d = 50, 16, 2, 20
        r = cf.SelectSwap_cost(address, target_bit, alpha, d, 'clean', False)
        assert r['Clean ancilla'] == 2**alpha * target_bit

    def test_selectswap_dirty_clean_is_only_the_base_copy(self):
        # for dirty, only the one base copy is freshly clean; the K-1 extra
        # copies are dirty-sourced, not part of 'Clean ancilla' at all
        address, target_bit, alpha, d = 50, 16, 2, 20
        r = cf.SelectSwap_cost(address, target_bit, alpha, d, 'dirty', False)
        assert r['Clean ancilla'] == target_bit
        assert r['Dirty ancilla'] == (2**alpha - 1) * target_bit

    def test_selectswap_buffer_degenerates_to_zero_with_no_swap_network(self):
        # alpha=0 means K=1: no swap network at all, so no swap-related
        # buffer need, regardless of how big QROM's own buffer is
        r = cf.SelectSwap_cost(50, 16, 0, 20, 'clean', False)
        qrom_only = cf.QROM_cost(50, 16, 20)
        assert r['Buffer ancilla'] == qrom_only['Buffer ancilla']

    def test_alpha_budget_keeps_output_copies_within_budget(self):
        # alpha_budget only bounds the compute-zone (Clean) footprint -- the
        # buffer zone is separate and plays no part in this bound at all.
        # Round-trip it: the alpha it picks must produce a SelectSwap whose
        # *actual* Clean ancilla (computed independently) still fits.
        target_bit, address, d, budget = 16, 486, 20, 57
        alpha = hlc.alpha_budget(budget, target_bit, model=cf)
        result = cf.SelectSwap_cost(address, target_bit, alpha, d, 'clean', False)
        assert result['Clean ancilla'] <= budget
        # its buffer slots are in the buffer zone, bounded by that zone alone
        assert result['Buffer ancilla'] <= layout.buffer_zone_size()

    def test_phase_fixup_buffer_is_a_real_traceable_number(self):
        # phase_fixup_cost doesn't separate buffer for its own feasibility
        # check (rough costing, by design), but the field it reports must
        # still be a real, non-fabricated value -- Clean and Buffer are
        # independent now, so there's no ordering to expect between them,
        # only that Buffer reflects an actual SWAP/QROM contribution
        r = cf.phase_fixup_cost(50, 20, ancilla_budget=0)   # forces the loop
        assert r['Buffer ancilla'] >= 0
        assert r['Buffer ancilla'] >= 16 or r['Buffer ancilla'] == 0   # SWAP's fixed buffer, when a swap network runs at all


class TestBufferBudget:
    """buffer_budget is checked (as a warning) but never used to constrain
    alpha -- the buffer zone is separate from the compute zone
    ancilla_budget bounds, so tightening/loosening buffer_budget alone must
    never change the chosen alpha or any cost number, only whether a
    warning fires."""

    def test_outer_prep_warns_when_buffer_budget_exceeded(self):
        with pytest.warns(UserWarning, match='buffer ancilla'):
            hlc.OuterPrep(54, 9, 57, dict(d_rep=20), buffer_budget=1, **OUTER, model=cf)

    def test_outer_prep_silent_when_buffer_budget_sufficient(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            hlc.OuterPrep(54, 9, 57, dict(d_rep=20), buffer_budget=1000, **OUTER, model=cf)

    def test_outer_prep_buffer_budget_does_not_affect_cost(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tight = hlc.OuterPrep(54, 9, 57, dict(d_rep=20), buffer_budget=1, **OUTER, model=cf)
            loose = hlc.OuterPrep(54, 9, 57, dict(d_rep=20), buffer_budget=1000, **OUTER, model=cf)
            none = hlc.OuterPrep(54, 9, 57, dict(d_rep=20), **OUTER, model=cf)
        assert S(tight)['Temporal cost'] == S(loose)['Temporal cost'] == S(none)['Temporal cost']
        assert tight['Clean ancilla'] == loose['Clean ancilla'] == none['Clean ancilla']

    def test_rprep_rot_warns_when_buffer_budget_exceeded(self):
        with pytest.warns(UserWarning, match='buffer ancilla'):
            hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='dirty_restricted',
                          K=2, lam=3, buffer_budget=1, model=cf)

    def test_be_cost_rejects_a_second_conflicting_buffer_budget(self):
        # a caller cannot accidentally set a different buffer_budget for
        # RPrep_Rot than for OuterPrep/InnerPrep via **rprep -- BE_cost's own
        # named parameter is the only way in, and passing it twice conflicts
        rprep_kwargs = {'buffer_budget': 16}
        with pytest.raises(TypeError, match='multiple values'):
            hlc.BE_cost(76, 57, 15, 19, 9, 15, 75, dict(d_rep=20), method='qrom', K=1, lam=2,
                       buffer_budget=32, **rprep_kwargs, model=cf, outer_variety='clean', inner_variety='clean')

    def test_be_cost_rejects_an_unknown_rprep_option(self):
        # BE_cost names RPrep+Rot's option explicitly instead of forwarding
        # **kwargs, so a misspelled option is caught at the call rather than
        # silently ignored
        with pytest.raises(TypeError, match='varietyy'):
            hlc.BE_cost(76, 57, 15, 19, 9, 15, 75, dict(d_rep=20), method='qrom', K=1, lam=2,
                        varietyy='dirty', model=cf, outer_variety='clean', inner_variety='clean')

    def test_be_cost_warns_when_buffer_budget_exceeded(self):
        with pytest.warns(UserWarning, match='buffer ancilla'):
            hlc.BE_cost(76, 57, 15, 19, 9, 15, 75, dict(d_rep=20), method='qrom', K=1, lam=2,
                        buffer_budget=1, model=cf, outer_variety='clean', inner_variety='clean')

    @pytest.mark.parametrize('call', [
        lambda: hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), model=cf),
        lambda: hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=20), lam=2, model=cf),
        lambda: hlc.Total_time(54, 27, 10, 27, 9, 16, 48, 32, dict(d_rep=20), 21.4, epsilon=1e-3,
                               round_time=1e-6, method='qrom', model=cf),
    ], ids=['RPrep_Rot', 'BE_cost', 'Total_time'])
    def test_batching_option_has_no_default(self, call):
        # no call can silently fall back to some option nobody chose
        with pytest.raises(TypeError, match='required keyword-only'):
            call()


class TestSingleAddressRegister:
    """A single-address register has no levels to load -- the state is already
    that one basis state -- so OuterPrep/InnerPrep are a no-op and cost
    nothing. Previously this crashed on `max(sizes)` with `sizes` empty."""

    def test_outer_prep_single_address_is_free(self):
        _assert_zero(hlc.OuterPrep(1, 9, 57, dict(d_rep=20), **OUTER, model=cf))

    def test_inner_prep_b_one_is_a_one_qubit_register(self):
        # the B register holds B+1 values, so B=1 is two values (one qubit):
        # exactly one level over the RC addresses, not a no-op
        assert (hlc.InnerPrep(270, 1, 9, 57, dict(d_rep=20), **INNER, model=cf)
                == hlc._sequential_prep('InnerPrep', [270], 9, 57, dict(d_rep=20), adjoint=False,
                                        B=1, **INNER, model=cf))

    def test_subroutine_name_still_marks_the_adjoint(self):
        assert hlc.OuterPrep(1, 9, 57, dict(d_rep=20), **OUTER, model=cf)['Subroutine'] == 'OuterPrep'
        assert hlc.OuterPrep(1, 9, 57, dict(d_rep=20), adjoint=True, **OUTER, model=cf)['Subroutine'] == 'OuterPrep_dag'

    def test_zero_or_negative_address_is_rejected(self):
        # log2(0) would otherwise raise an opaque OverflowError deep inside
        with pytest.raises(ValueError):
            hlc.OuterPrep(0, 9, 57, dict(d_rep=20), **OUTER, model=cf)
        with pytest.raises(ValueError):
            hlc.InnerPrep(270, 0, 9, 57, dict(d_rep=20), **INNER, model=cf)


class TestCostModelIsAParameter:
    """The cost model (the code) is passed in, never imported by the
    model-independent modules, so switching code means passing a different
    module."""

    def test_rep_code_model_provides_the_whole_interface(self):
        missing = [name for name in hlc.MODEL_INTERFACE if not hasattr(cf, name)]
        assert missing == []

    def test_ldpc_model_provides_the_whole_interface(self):
        # filled in or not: every name exists, so nothing is missed
        from resource_count import cost_functions_ldpc as ldpc
        missing = [name for name in hlc.MODEL_INTERFACE if not hasattr(ldpc, name)]
        assert missing == []

    def test_ldpc_model_places_every_zone(self):
        from resource_count import cost_functions_ldpc as ldpc
        assert set(ldpc.ZONE_LAYER) == set(cf.ZONE_LAYER)

    @pytest.mark.parametrize('call', [
        lambda: hlc.Spin(486, 54, dict(d_rep=20), b_rot=16, B=27),
        lambda: hlc.OuterPrep(486, 9, 48, dict(d_rep=20), **OUTER),
        lambda: hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20), method='qrom', K=1, lam=2),
        lambda: hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=20), method='qrom', K=1, lam=2),
        lambda: bt.batching_table(54, 27, 10, 16, dict(d_rep=20), lams=[2], alpha_max=4),
    ], ids=['Spin', 'OuterPrep', 'RPrep_Rot', 'BE_cost', 'batching_table'])
    def test_leaving_out_the_model_is_an_error(self, call):
        with pytest.raises(TypeError, match='model'):
            call()

    @pytest.mark.parametrize('module', ['accounting', 'layout', 'high_level_cost',
                                        'batching', 'optimize'])
    def test_model_independent_modules_do_not_import_a_model(self, module):
        # only the `if __name__ == '__main__'` example may pick a model
        import pathlib, re
        import resource_count
        src = (pathlib.Path(resource_count.__file__).parent / f'{module}.py').read_text()
        body = src.split("if __name__ == '__main__'")[0]
        assert not re.search(r'^\s*(from|import)\s.*cost_functions', body, re.M)



def _two_layer_model():
    '''
    A STAND-IN, not an LDPC model: the rep-code model with the system and
    address zones moved to an 'ldpc' layer counted in logical qubits, whose
    logical qubit-cycles are its logical qubit-rounds / d.  It only checks
    that the routines carry a second layer through to the totals.  Like the
    rep code, it takes the one distance d_rep.
    '''
    import types
    m = types.SimpleNamespace(**{name: getattr(cf, name) for name in hlc.MODEL_INTERFACE})

    def zone_qubits(zone, logical, d_rep):
        return ('ldpc', logical) if zone in ('system', 'address') else cf.zone_qubits(zone, logical, d_rep)

    def logical_qubit_cycles(layer, qubit_cycles, d_rep):
        return qubit_cycles/d_rep if layer == 'ldpc' else cf.logical_qubit_cycles(layer, qubit_cycles, d_rep)

    m.zone_qubits, m.logical_qubit_cycles = zone_qubits, logical_qubit_cycles
    return m


class TestSecondLayer:
    """With a model that puts zones in a second layer, every total reports
    that layer under its own name, and the effective volume adds the layers
    only after each is converted to logical qubit-cycles."""

    D = 20
    ARGS = (54, 27, 10, 27, 9, 16, 48, layout.buffer_zone_size(), dict(d_rep=D), 21.4)
    OPTION = dict(epsilon=1e-3, round_time=1e-6, method='qrom', K=1, lam=2)

    def total(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return hlc.Total_time(*self.ARGS, model=model, **self.OPTION, outer_variety='clean', inner_variety='clean')

    def be(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=self.D), method='qrom', K=1, lam=2,
                               buffer_budget=layout.buffer_zone_size(), model=model, outer_variety='clean', inner_variety='clean')

    def test_both_layers_are_reported(self):
        t = self.total(_two_layer_model())
        assert t['layers'] == ['rep', 'ldpc']
        for key in ('total qubit cycles [ldpc]', 'total idle cycles [ldpc]', 'Spatial cost [ldpc]',
                    'total logical qubit cycles [rep]', 'total logical qubit cycles [ldpc]'):
            assert t[key] > 0

    def test_effective_volume_is_the_sum_of_the_converted_layers(self):
        t = self.total(_two_layer_model())
        assert t['total effective volume'] == pytest.approx(
            t['total logical qubit cycles [rep]'] + t['total logical qubit cycles [ldpc]'])
        assert t['total logical qubit cycles [ldpc]'] == pytest.approx(
            t['total qubit cycles [ldpc]']/self.D)
        assert t['total logical qubit cycles [rep]'] == pytest.approx(
            t['total qubit cycles']/(self.D*(2*self.D - 1)))

    def test_rep_only_effective_volume_is_its_converted_qubit_cycles(self):
        t = self.total(cf)
        assert t['layers'] == ['rep']
        assert t['total effective volume'] == pytest.approx(
            t['total qubit cycles']/(self.D*(2*self.D - 1)))

    def test_moved_idle_zones_are_neither_lost_nor_doubled(self):
        # the system and address idle volume leaves the rep layer and appears
        # in the ldpc layer, 1 logical qubit for every 2d-1 physical ones;
        # everything else is unchanged
        rep, two = self.be(cf), self.be(_two_layer_model())
        for part in ('idle system', 'idle address'):
            assert two['parts'][part]['volume'].get('rep', 0) == 0
            assert layout.physical(two['parts'][part]['volume']['ldpc'], self.D) == \
                rep['parts'][part]['volume']['rep']
        for part in set(rep['parts']) - {'idle system', 'idle address'}:
            assert two['parts'][part] == rep['parts'][part]

    def test_time_is_the_same_in_both_models(self):
        rep, two = self.total(cf), self.total(_two_layer_model())
        for key in ('total rounds', 'total logical cycles', 'total Toffoli', 'total i-state'):
            assert two[key] == rep[key]

    def test_report_prints_each_layer(self, capsys):
        hlc.report(self.be(_two_layer_model()), model=_two_layer_model())
        out = capsys.readouterr().out
        assert "layer 'ldpc' (its own unit)" in out
        assert "space per routine in layer 'ldpc'" in out

    def test_rep_code_model_refuses_another_layer(self):
        with pytest.raises(ValueError, match='ldpc'):
            cf.logical_qubit_cycles('ldpc', 1, 20)


class TestBusyQubitsAreInsideFootprints:
    """Every step lists the zone qubits that are BUSY in it; those are not
    charged as idle, so the step's own footprint has to hold them, in the
    layer their zone lives in.  If it holds fewer, the missing busy qubits are
    counted nowhere.  This runs whole block encodings in both models and
    checks every step on its way through layout.idle_except (the one place
    each step declares what is busy): footprint >= busy, layer by layer.

    It is a lower bound only: it catches a footprint too small for its busy
    qubits, not one that holds the right number of the wrong qubits.  Steps
    that take no rounds (e.g. a lookup over a single address) cost nothing,
    so they are skipped."""

    from resource_count import cost_functions_ldpc as ldpc
    SETS = {'A': (54, 27, 10, 27, 9, 16, 48), 'B': (76, 57, 15, 19, 9, 15, 45)}
    CASES = [
        (cf, 'A', dict(d_rep=20), 'qrom', 1), (cf, 'B', dict(d_rep=20), 'qrom', 1),
        (ldpc, 'A', dict(d_rep=12, d_LDPC=12), 'qrom', 1),
        (ldpc, 'A', dict(d_rep=16, d_LDPC=16), 'qrom', 1),
        (ldpc, 'A', dict(d_rep=22, d_LDPC=22), 'qrom', 1),
        (ldpc, 'B', dict(d_rep=12, d_LDPC=12), 'qrom', 1),
        (ldpc, 'B', dict(d_rep=16, d_LDPC=16), 'qrom', 1),
        (ldpc, 'B', dict(d_rep=22, d_LDPC=22), 'qrom', 1),
        # the two distances apart, either way round
        (ldpc, 'A', dict(d_rep=17, d_LDPC=12), 'qrom', 1),
        (ldpc, 'B', dict(d_rep=15, d_LDPC=22), 'qrom', 1),
        # the LDPC model allows the dirty methods, whose lookups hold the
        # borrowed system qubits; the rep code's (ruled out by its layout)
        # must still hold them if costed
        (ldpc, 'A', dict(d_rep=16, d_LDPC=16), 'dirty_restricted', 2),
        (ldpc, 'A', dict(d_rep=16, d_LDPC=16), 'dirty_full', 4),
        (ldpc, 'A', dict(d_rep=13, d_LDPC=16), 'dirty_full', 4),
        (cf, 'A', dict(d_rep=20), 'dirty_restricted', 2),
    ]

    @pytest.mark.parametrize('model, params, distances, method, K', CASES,
                             ids=[f"{m.__name__.split('.')[-1]}-{p}-"
                                  + '-'.join(f'{k}{v}' for k, v in dist.items()) + f"-{meth}"
                                  for m, p, dist, meth, _ in CASES])
    def test_no_step_holds_fewer_qubits_than_it_marks_busy(self, monkeypatch, model, params,
                                                           distances, method, K):
        original = layout.idle_except
        short = []

        def checked(step, model, distances, zones, busy):
            if acc.total(step, 'rounds') > 0:
                needed = {}
                for zone, logical in busy.items():
                    layer, qubits = model.zone_qubits(zone, logical, **distances)
                    needed[layer] = needed.get(layer, 0) + qubits
                for layer, qubits in needed.items():
                    held = step['space'].get(layer, 0)
                    if held < qubits:
                        name = next(iter(step['peak'].values()))['step']
                        short.append((name, layer, held, qubits, busy))
            return original(step, model, distances, zones, busy)

        monkeypatch.setattr(layout, 'idle_except', checked)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hlc.BE_cost(*self.SETS[params], distances, method=method, K=K, lam=2,
                        buffer_budget=layout.buffer_zone_size(), model=model, outer_variety='clean', inner_variety='clean')
        assert short == []


@pytest.mark.parametrize('model, method, K', [('rep', 'qrom', 1), ('ldpc', 'qrom', 1),
                                              ('ldpc', 'dirty_restricted', 2),
                                              ('ldpc', 'dirty_full', 4)])
def test_rprep_rot_measures_out_every_loaded_angle_once(model, method, K):
    # every lookup's lam angles -- the J batches' and the dirty methods'
    # extra lookups' -- are measured out once each, by RPrep_Rot's own MBU
    # steps; no other MBU (the phase fixup's own aside), and
    # none inside the shuffle
    from resource_count import cost_functions_ldpc as ldpc
    m, dist, budget = {'rep': (cf, dict(d_rep=15), 48),
                       'ldpc': (ldpc, dict(d_rep=16, d_LDPC=16), 32)}[model]
    lam, b_rot = 2, 16
    r = hlc.RPrep_Rot(54, 27, 10, 27, b_rot, dist, method=method, K=K, lam=lam,
                      ancilla_budget=budget, model=m)
    angles = lam*acc.total(m.mbu_cost(b_rot, **dist), 'rounds')    # one MBU per angle
    mbu = {n: p['rounds'] for n, p in r['parts'].items()
           if (n == 'mbu' or n.endswith(':mbu')) and n != 'fixup:mbu'}
    assert mbu == {'mbu': r['lookups']*angles}
    assert (r['lookups'] > r['J']) == (method != 'qrom')


@pytest.mark.parametrize('model', ['rep', 'ldpc'])
def test_clean_selectswap_measures_out_the_unselected_copies(model):
    # the K-1 copies the swap network did not select are measured out with
    # the model's own mbu_cost, so the lookup takes those rounds too
    from resource_count import cost_functions_ldpc as ldpc
    m, dist = {'rep': (cf, dict(d_rep=20)), 'ldpc': (ldpc, dict(d_rep=15, d_LDPC=16))}[model]
    target, alpha = 9, 1
    with_copies = m.SelectSwap_cost(64, target, alpha, **dist, variety='clean')
    mbu = m.mbu_cost((2**alpha - 1)*target, **dist)
    assert with_copies['parts']['mbu']['rounds'] == acc.total(mbu, 'rounds')
    # a plain QROM (K = 1) has no copies to measure out
    assert 'mbu' not in m.SelectSwap_cost(64, target, 0, **dist, variety='clean')['parts']


@pytest.mark.parametrize('model', ['rep', 'ldpc'])
@pytest.mark.parametrize('address, alpha', [(64, 1), (64, 6), (64, 9), (486, 9)])
def test_clean_selectswap_holds_all_k_copies_even_with_a_single_address_select(model, address, alpha):
    # with K >= address the Select loads one address (a classical constant,
    # empty in QROM_cost), but the K copies of the output register are still
    # clean ancilla -- otherwise a huge K fits any compute-zone budget
    m, dist = {'rep': (cf, dict(d_rep=20)), 'ldpc': (ldpc, dict(d_rep=15, d_LDPC=16))}[model]
    target = 16
    r = m.SelectSwap_cost(address, target, alpha, **dist, variety='clean')
    assert r['Clean ancilla'] == 2**alpha*target


class TestReportGroupsStepsIntoOperations:
    """Steps inside an operation ('fixup:hadamard', 'fixup:reaction', ...)
    count in that operation's row of the report; a separate table lists
    them by step type.  No number changes, only the grouping."""

    @staticmethod
    def printed(be, model, capsys):
        hlc.report(be, model=model)
        return capsys.readouterr().out

    @staticmethod
    def row(out, name):
        # the first table is rounds by subroutine; the operation tables follow
        for line in out.split('operation', 1)[1].splitlines():
            if line.split() and line.split()[0].rstrip('*') == name:
                return float(line.split()[1])
        raise AssertionError(f'no row {name!r}')

    def test_the_fixup_row_is_its_whole_duration(self, capsys):
        be = hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=15), method='qrom', K=1, lam=2,
                         buffer_budget=layout.buffer_zone_size(), model=cf, outer_variety='clean', inner_variety='clean')
        whole = sum(p['rounds'] for n, p in be['parts'].items() if n.split(':')[0] == 'fixup')
        assert any(n.startswith('fixup:') for n in be['parts'])
        assert self.row(self.printed(be, cf, capsys), 'fixup') == whole

    def test_space_time_fixup_steps_belong_to_the_fixup(self):
        r = cf.phase_fixup_cost(6912, 20, 48, 36)
        assert r['split'] == 'space-time'
        assert {n.split(':')[0] for n in r['parts']} == {'fixup'}

    def test_lookup_hadamards_belong_to_the_swap(self):
        r = cf.SelectSwap_cost(486, 32, 2, 20, 'dirty', True)
        assert 'swap:hadamard' in r['parts'] and 'hadamard' not in r['parts']

    def test_step_table_adds_every_hadamard(self, capsys):
        be = hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=15), method='qrom', K=1, lam=2,
                         buffer_budget=layout.buffer_zone_size(), model=cf, outer_variety='clean', inner_variety='clean')
        out = self.printed(be, cf, capsys)
        hadamards = sum(p['rounds'] for n, p in be['parts'].items() if n.endswith(':hadamard'))
        line = next(l for l in out.split('steps inside operations', 1)[1].splitlines()
                    if l.startswith('hadamard'))
        assert float(line.split()[1]) == hadamards

    def test_fallback_footnote_counts_only_the_fallback_steps(self, capsys):
        # the fixup row holds compiled Hadamards too; the footnote must not
        be = hlc.BE_cost(54, 27, 10, 27, 9, 16, 48, dict(d_rep=15), method='qrom', K=1, lam=2,
                         buffer_budget=layout.buffer_zone_size(), model=cf, outer_variety='clean', inner_variety='clean')
        out = self.printed(be, cf, capsys)
        fallback = sum(p['rounds'] for n, p in be['parts'].items() if n in cf.FALLBACK)
        note = next(l for l in out.splitlines() if l.startswith('* not compiled'))
        assert f"{fallback:.0f} rounds" in note


class TestPrepVarietyIsAnExplicitChoice:
    """The preps' SelectSwap variety is chosen beforehand and passed to
    BE_cost / Total_time, as RPrep+Rot's option is: no default, and a model
    without Z access to the system qubits refuses 'dirty'."""

    ARGS = (54, 27, 10, 27, 9, 16)
    OPTION = dict(method='qrom', K=1, lam=2)

    @pytest.mark.parametrize('missing', ['outer_variety', 'inner_variety'])
    def test_no_default(self, missing):
        given = {'outer_variety': 'clean', 'inner_variety': 'clean'}
        del given[missing]
        with pytest.raises(TypeError, match=missing):
            hlc.BE_cost(*self.ARGS, 48, dict(d_rep=20), model=cf, **self.OPTION, **given)

    def test_rep_code_refuses_dirty(self):
        with pytest.raises(ValueError, match='SYSTEM_Z_ACCESS'):
            hlc.BE_cost(*self.ARGS, 48, dict(d_rep=20), model=cf, **self.OPTION,
                        outer_variety='clean', inner_variety='dirty')

    def test_unknown_variety_is_refused(self):
        with pytest.raises(ValueError, match="'clean' or 'dirty'"):
            hlc.BE_cost(*self.ARGS, 48, dict(d_rep=20), model=cf, **self.OPTION,
                        outer_variety='clen', inner_variety='clean')

    def test_ldpc_inner_dirty_is_the_dirty_inner_prep(self):
        # the block encoding's InnerPrep calls are exactly InnerPrep with the
        # given variety: its routine breakdown carries their rounds
        from resource_count import cost_functions_ldpc as ldpc
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            be = hlc.BE_cost(*self.ARGS, 32, dict(d_rep=16, d_LDPC=16), model=ldpc, **self.OPTION,
                             outer_variety='clean', inner_variety='dirty',
                             buffer_budget=layout.buffer_zone_size())
            inner = hlc.InnerPrep(432, 27, 9, 32, dict(d_rep=16, d_LDPC=16), variety='dirty', model=ldpc, N=54, b_rot=16,
                                  X=486, buffer_budget=layout.buffer_zone_size())
        rounds = {r['name']: r['rounds'] for r in be['breakdown']}
        assert rounds['InnerPrep'] == acc.total(inner, 'rounds')
        assert inner['variety'] == 'dirty'
