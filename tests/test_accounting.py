"""
The combining rules in accounting.py, checked against hand-computed numbers.
"""
import pytest

from resource_count import accounting as acc
from resource_count import cost_functions as cf
from resource_count import layout


class TestOp:
    def test_volume_is_space_times_rounds_per_layer(self):
        c = acc.op('x', rounds=10, space={'rep': 4, 'ldpc': 3})
        assert c['parts']['x']['volume'] == {'rep': 40, 'ldpc': 30}

    def test_space_is_the_given_space(self):
        c = acc.op('x', rounds=10, space={'rep': 4, 'ldpc': 3})
        assert c['space'] == {'rep': 4, 'ldpc': 3}

    def test_ancilla_defaults_to_zero(self):
        c = acc.op('x', rounds=1, space={'rep': 1})
        assert (c['Clean ancilla'], c['Dirty ancilla'], c['Buffer ancilla']) == (0, 0, 0)


class TestSeq:
    """One step AFTER another: time-like fields add, space-like take the max."""
    A = acc.op('a', rounds=3, space={'rep': 4}, toffoli=1, i_states=2, clean=5, buffer=1)
    B = acc.op('b', rounds=5, space={'rep': 6, 'ldpc': 2}, toffoli=7, i_states=1, clean=2, dirty=3)

    def test_rounds_toffoli_i_states_add(self):
        c = acc.seq(self.A, self.B)
        assert acc.total(c, 'rounds') == 3 + 5
        assert acc.total(c, 'toffoli') == 1 + 7
        assert acc.total(c, 'i-states') == 2 + 1

    def test_volume_adds_per_layer(self):
        c = acc.seq(self.A, self.B)
        assert acc.volume(c, 'rep') == 3*4 + 5*6
        assert acc.volume(c, 'ldpc') == 5*2

    def test_space_is_max_per_layer(self):
        c = acc.seq(self.A, self.B)
        assert c['space'] == {'rep': max(4, 6), 'ldpc': 2}

    def test_ancilla_is_max(self):
        c = acc.seq(self.A, self.B)
        assert (c['Clean ancilla'], c['Dirty ancilla'], c['Buffer ancilla']) == (5, 3, 1)

    def test_same_name_parts_merge(self):
        c = acc.seq(self.A, self.A)
        assert list(c['parts']) == ['a']
        assert c['parts']['a']['rounds'] == 6
        assert c['parts']['a']['volume'] == {'rep': 24}

    def test_inputs_are_not_modified(self):
        before = repr(self.A)
        acc.seq(self.A, self.A)
        assert repr(self.A) == before

    def test_order_does_not_matter(self):
        assert acc.seq(self.A, self.B) == acc.seq(self.B, self.A)

    def test_empty_is_neutral(self):
        assert acc.seq(acc.empty(), self.A) == acc.seq(self.A)


class TestWithIdle:
    """Qubits idle BESIDE a step: rounds unchanged, space and volume grow."""
    STEP = acc.op('a', rounds=3, space={'rep': 4}, toffoli=1, i_states=2, clean=5)

    def test_rounds_toffoli_i_states_unchanged(self):
        c = acc.with_idle(self.STEP, 'system', 'rep', 10)
        assert acc.total(c, 'rounds') == 3
        assert acc.total(c, 'toffoli') == 1
        assert acc.total(c, 'i-states') == 2

    def test_space_adds(self):
        c = acc.with_idle(self.STEP, 'system', 'rep', 10)
        assert c['space'] == {'rep': 4 + 10}

    def test_volume_is_idle_qubits_times_the_steps_rounds(self):
        c = acc.with_idle(self.STEP, 'system', 'rep', 10)
        assert c['parts']['idle system'] == {'rounds': 0, 'toffoli': 0, 'i-states': 0,
                                             'volume': {'rep': 10*3}}
        assert acc.volume(c, 'rep') == 4*3 + 10*3

    def test_other_layer_is_kept_separate(self):
        c = acc.with_idle(self.STEP, 'system', 'ldpc', 10)
        assert c['space'] == {'rep': 4, 'ldpc': 10}
        assert acc.volume(c, 'rep') == 12
        assert acc.volume(c, 'ldpc') == 30

    def test_ancilla_unchanged(self):
        c = acc.with_idle(self.STEP, 'system', 'rep', 10)
        assert c['Clean ancilla'] == 5

    def test_step_is_not_modified(self):
        before = repr(self.STEP)
        acc.with_idle(self.STEP, 'system', 'rep', 10)
        assert repr(self.STEP) == before

    def test_nothing_idle_beside_an_empty_step(self):
        c = acc.with_idle(acc.empty(), 'system', 'rep', 10)
        assert c == acc.empty()

    def test_zero_qubits_is_unchanged(self):
        assert acc.with_idle(self.STEP, 'system', 'rep', 0) == self.STEP

    def test_idle_then_seq(self):
        # step A (3 rounds, 4 qubits) with 10 idle, then step B (5 rounds,
        # 6 qubits) with nothing idle: space max(14, 6), volume 12+30+30
        a = acc.with_idle(self.STEP, 'system', 'rep', 10)
        b = acc.op('b', rounds=5, space={'rep': 6})
        c = acc.seq(a, b)
        assert c['space'] == {'rep': 14}
        assert acc.total(c, 'rounds') == 8
        assert acc.volume(c, 'rep') == 4*3 + 10*3 + 6*5


class TestIdleExcept:
    """Every zone present is idle beside the step, except what the step's own
    footprint uses (`busy`)."""
    STEP = acc.op('a', rounds=3, space={'rep': 4})
    ZONES = {'system': 5, 'fourier': 1}

    def test_idle_is_zone_size_minus_busy(self):
        c = layout.idle_except(self.STEP, cf, dict(d_rep=20), self.ZONES, busy={'system': 3})
        assert c['space'] == {'rep': 4 + layout.physical(5 - 3, 20) + layout.physical(1, 20)}
        assert c['parts']['idle system']['volume'] == {'rep': layout.physical(2, 20)*3}
        assert c['parts']['idle fourier']['volume'] == {'rep': layout.physical(1, 20)*3}

    def test_fully_busy_zone_is_not_idle(self):
        c = layout.idle_except(self.STEP, cf, dict(d_rep=20), self.ZONES, busy={'system': 5})
        assert 'idle system' not in c['parts']

    def test_zone_missing_from_busy_is_entirely_idle(self):
        # forgetting a zone overcounts rather than drops it
        c = layout.idle_except(self.STEP, cf, dict(d_rep=20), self.ZONES, busy={})
        assert c['parts']['idle system']['volume'] == {'rep': layout.physical(5, 20)*3}

    def test_the_model_decides_the_layer(self):
        # a stand-in model that keeps the system register in LDPC, counted in
        # logical qubits -- idle_except itself must not care which
        class LdpcSystem:
            @staticmethod
            def zone_qubits(zone, logical, d_rep):
                return ('ldpc', logical) if zone == 'system' else cf.zone_qubits(zone, logical, d_rep)

        c = layout.idle_except(self.STEP, LdpcSystem, dict(d_rep=20), self.ZONES, busy={})
        assert c['space'] == {'rep': 4 + layout.physical(1, 20), 'ldpc': 5}
        assert c['parts']['idle system']['volume'] == {'ldpc': 5*3}

    def test_busy_zone_not_present_is_rejected(self):
        with pytest.raises(ValueError, match='not one of the zones'):
            layout.idle_except(self.STEP, cf, dict(d_rep=20), self.ZONES, busy={'flag': 1})

    def test_more_busy_than_the_zone_holds_is_rejected(self):
        with pytest.raises(ValueError, match='only holds'):
            layout.idle_except(self.STEP, cf, dict(d_rep=20), self.ZONES, busy={'system': 6})

    def test_zone_unknown_to_the_model_is_rejected(self):
        with pytest.raises(KeyError):
            layout.idle_except(self.STEP, cf, dict(d_rep=20), {'nonsense': 1}, busy={})


class TestRepeat:
    def test_is_seq_of_n_copies(self):
        a = acc.op('a', rounds=3, space={'rep': 4}, toffoli=1)
        assert acc.repeat(a, 3) == acc.seq(a, a, a)
        assert acc.total(acc.repeat(a, 3), 'rounds') == 9
        assert acc.repeat(a, 3)['space'] == {'rep': 4}

    def test_zero_times_is_empty(self):
        assert acc.repeat(acc.op('a', 3, {'rep': 4}), 0) == acc.empty()


class TestScale:
    """scale(cost, n) is repeat(cost, n), computed directly."""
    A = acc.seq(acc.op('a', rounds=3, space={'rep': 4}, toffoli=1, i_states=2, clean=5),
                acc.with_idle(acc.op('b', rounds=5, space={'rep': 6}), 'system', 'rep', 7))

    @pytest.mark.parametrize('n', [0, 1, 2, 4])
    def test_equals_repeat(self, n):
        assert acc.scale(self.A, n) == acc.repeat(self.A, n)

    def test_real_routine_equals_repeat(self):
        r = cf.SelectSwap_cost(50, 16, 2, 20, 'dirty', True)
        assert acc.scale(r, 4) == acc.repeat(r, 4)

    def test_input_is_not_modified(self):
        before = repr(self.A)
        acc.scale(self.A, 3)
        assert repr(self.A) == before


class TestSummary:
    def test_totals_are_the_sums_over_parts(self):
        r = cf.SelectSwap_cost(50, 16, 2, 20, 'dirty', True)
        s = acc.summary(r)
        assert s['Temporal cost'] == acc.total(r, 'rounds')
        assert s['Toffoli cost'] == acc.total(r, 'toffoli')
        assert s['i-state cost'] == acc.total(r, 'i-states')
        assert s['Spatial cost'] == r['space']['rep']
        assert s['Qubit cycles'] == acc.volume(r, 'rep')
        assert all(s[k] == r[k] for k in acc.ANCILLA)

    def test_idle_cycles_are_the_idle_parts_only(self):
        step = acc.with_idle(acc.op('a', rounds=3, space={'rep': 4}), 'system', 'rep', 10)
        s = acc.summary(step)
        assert s['Idle cycles'] == 10*3
        assert s['Qubit cycles'] == 4*3 + 10*3

    def test_empty_record(self):
        s = acc.summary(acc.empty())
        assert s['Spatial cost'] == s['Temporal cost'] == s['Qubit cycles'] == s['Idle cycles'] == 0

    def test_other_layers_get_their_own_names(self):
        # the unsuffixed names stay the 'rep' layer only; never a sum across
        # layers, whose units differ
        step = acc.with_idle(acc.op('x', rounds=3, space={'rep': 4, 'ldpc': 2}), 'system', 'ldpc', 5)
        s = acc.summary(step)
        assert (s['Spatial cost'], s['Qubit cycles'], s['Idle cycles']) == (4, 4*3, 0)
        assert (s['Spatial cost [ldpc]'], s['Qubit cycles [ldpc]'], s['Idle cycles [ldpc]']) == \
            (2 + 5, (2 + 5)*3, 5*3)

    def test_a_rep_only_record_has_no_suffixed_names(self):
        s = acc.summary(acc.op('x', rounds=3, space={'rep': 4}))
        assert not [k for k in s if '[' in k]


class TestLayers:
    def test_rep_first_then_by_name(self):
        c = acc.seq(acc.op('a', 1, {'zz': 1}), acc.op('b', 1, {'ldpc': 1, 'rep': 1}))
        assert acc.layers(c) == ['rep', 'ldpc', 'zz']

    def test_a_layer_with_volume_only_is_found(self):
        c = acc.empty()
        c['parts']['x'] = {'rounds': 1, 'toffoli': 0, 'i-states': 0, 'volume': {'ldpc': 3}}
        assert acc.layers(c) == ['ldpc']


class TestPeak:
    """'peak' says which step sets the space and how it splits: footprint +
    idle == space, in every layer, whatever was combined."""

    @staticmethod
    def consistent(c):
        return all(c['peak'][l]['footprint'] + sum(c['peak'][l]['idle'].values()) == s
                   for l, s in c['space'].items()) and set(c['peak']) == set(c['space'])

    def test_op_is_its_own_peak(self):
        c = acc.op('a', rounds=3, space={'rep': 4})
        assert c['peak'] == {'rep': {'step': 'a', 'footprint': 4, 'idle': {}}}

    def test_idle_is_recorded_beside_the_step(self):
        c = acc.with_idle(acc.op('a', 3, {'rep': 4}), 'system', 'rep', 10)
        assert c['peak'] == {'rep': {'step': 'a', 'footprint': 4, 'idle': {'system': 10}}}
        assert self.consistent(c)

    def test_idle_in_another_layer_has_no_footprint_there(self):
        c = acc.with_idle(acc.op('a', 3, {'rep': 4}), 'system', 'ldpc', 10)
        assert c['peak']['ldpc'] == {'step': 'a', 'footprint': 0, 'idle': {'system': 10}}
        assert self.consistent(c)

    def test_seq_takes_the_largest_steps_peak(self):
        small = acc.with_idle(acc.op('a', 3, {'rep': 4}), 'system', 'rep', 1)    # 5
        large = acc.op('b', 5, {'rep': 6})                                          # 6
        c = acc.seq(small, large, small)
        assert c['peak']['rep'] == {'step': 'b', 'footprint': 6, 'idle': {}}
        assert self.consistent(c)

    def test_seq_tie_keeps_the_first(self):
        c = acc.seq(acc.op('a', 1, {'rep': 6}), acc.op('b', 1, {'rep': 6}))
        assert c['peak']['rep']['step'] == 'a'

    def test_scale_keeps_the_peak(self):
        c = acc.with_idle(acc.op('a', 3, {'rep': 4}), 'system', 'rep', 10)
        assert acc.scale(c, 7)['peak'] == c['peak']

    def test_set_space_replaces_the_peak(self):
        c = acc.with_idle(acc.op('a', 3, {'rep': 4}), 'system', 'rep', 10)
        acc.set_space(c, {'rep': 99}, 'layout')
        assert c['peak'] == {'rep': {'step': 'layout', 'footprint': 99, 'idle': {}}}
        assert self.consistent(c)


class TestWithin:
    def test_parts_become_steps_of_the_operation(self):
        c = acc.seq(acc.op('qrom', 3, {'rep': 4}), acc.op('swap', 2, {'rep': 5}))
        w = acc.within(c, 'fixup')
        assert set(w['parts']) == {'fixup:qrom', 'fixup:swap'}
        assert w['parts']['fixup:qrom'] == c['parts']['qrom']

    def test_only_the_names_change(self):
        c = acc.op('qrom', 3, {'rep': 4}, toffoli=2, i_states=6, clean=1, buffer=7)
        w = acc.within(c, 'fixup')
        assert acc.summary(w) == acc.summary(c)
        assert 'qrom' in c['parts']                      # the input is not modified

    def test_a_step_cannot_be_nested_again(self):
        with pytest.raises(ValueError, match='already'):
            acc.within(acc.op('fixup:hadamard', 1, {'rep': 1}), 'other')
