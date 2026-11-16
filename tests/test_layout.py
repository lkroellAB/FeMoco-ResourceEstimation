"""
Tests for layout.py: zone sizes, the logical-to-physical conversion, and the
zone-budget check that high_level_cost.py's composite routines use.
"""
import warnings

import pytest

from resource_count import layout
from resource_count import cost_functions as cf


RESULT = {'Clean ancilla': 40, 'Dirty ancilla': 10, 'Buffer ancilla': 20}


class TestZoneSizes:
    def test_buffer_zone_matches_adder_costs_own_buffer(self):
        # tied to the same source of truth, not an independently-guessed number
        assert layout.buffer_zone_size() == cf.adder_cost(9, 20)['Buffer ancilla']
        assert layout.buffer_zone_size(buff=1) == cf.adder_cost(9, 20, buff=1)['Buffer ancilla']

    def test_fourier_zone_is_b_rot(self):
        assert layout.fourier_zone_size(16) == 16

    def test_system_zone_is_2n(self):
        assert layout.system_zone_size(54) == 108

    def test_address_zone_formula(self):
        # X=486, B=27: ceil(log2(486)) + ceil(log2(27)) + 2 = 9 + 5 + 2
        assert layout.address_zone_size(486, 27) == 16


class TestPhysical:
    def test_two_d_minus_one_per_logical_qubit(self):
        # a stored rep-code patch: d data + d-1 ancilla.  The fallback
        # primitives (ref, qc_comp, phase_fixup) use 4d per logical qubit
        # instead, which includes their routing.
        assert layout.physical(1, 20) == 2*20 - 1
        assert layout.physical(10, 20) == 10*(2*20 - 1)

    def test_zero_logical_is_zero_physical(self):
        assert layout.physical(0, 20) == 0


class TestCheckFeasibility:
    def test_silent_when_everything_fits(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            layout.check_feasibility(RESULT, 'Test', ancilla_budget=100,
                                     dirty_budget=100, buffer_budget=100)

    def test_silent_when_no_budgets_given(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            layout.check_feasibility(RESULT, 'Test')

    def test_warns_on_clean_overflow(self):
        with pytest.warns(UserWarning, match='clean ancilla'):
            layout.check_feasibility(RESULT, 'Test', ancilla_budget=10)

    def test_warns_on_dirty_overflow(self):
        with pytest.warns(UserWarning, match='dirty ancilla'):
            layout.check_feasibility(RESULT, 'Test', dirty_budget=5)

    def test_warns_on_buffer_overflow(self):
        with pytest.warns(UserWarning, match='buffer ancilla'):
            layout.check_feasibility(RESULT, 'Test', buffer_budget=5)

    def test_warns_once_per_exceeded_zone(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            layout.check_feasibility(RESULT, 'Test', ancilla_budget=1,
                                     dirty_budget=1, buffer_budget=1)
        assert len(w) == 3

    def test_message_includes_the_given_name(self):
        with pytest.warns(UserWarning, match='Test'):
            layout.check_feasibility(RESULT, 'Test', ancilla_budget=1)
