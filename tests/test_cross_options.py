"""
cross_options.py, the contained example comparing the lookup options across
the two models: the rep code's hypothetical dirty options, with access
corridors under the system qubits, and the rest of the package unchanged.
"""
import warnings

import pytest

from resource_count import cost_functions as cf
from resource_count import cross_options as xo
from resource_count import error_accounting as ea
from resource_count import settings as st


def dirty_variant():
    return next(v for v in xo.cross_variants(st.FEMOCO_54, st.SETTINGS_54) if v['label'] == 'rep, dirty options')


def test_each_model_runs_the_others_chosen_options():
    """per molecule: the rep code's dirty variant runs LDPC's chosen options,
    LDPC's cross variant the rep code's"""
    for system, settings in st.MOLECULES.values():
        variants = {v['label']: v for v in xo.cross_variants(system, settings)}
        assert variants['rep, dirty options']['option'] == settings[st.ldpc_code]['option']
        assert variants['LDPC, rep options']['option'] == settings[st.rep_code]['option']


def test_rep_code_itself_is_unchanged():
    assert xo.REP_WITH_SYSTEM_Z_ACCESS.SYSTEM_Z_ACCESS and not cf.SYSTEM_Z_ACCESS
    assert xo.REP_WITH_SYSTEM_Z_ACCESS.SelectSwap_cost is cf.SelectSwap_cost   # the model's own functions


def test_corridor_layout():
    # each system half ceil(3N/2) + 2 wide instead of N + 2: FeMoCo-54 at
    # d = 15 gains 2*27*(19 + 15) physical qubits
    v = dirty_variant()
    assert v['corridors'] == 54                     # one per two of the 108 system qubits
    plain = ea.physical_qubits(st.FEMOCO_54, cf, v['option'], v['distances'])['total']
    assert plain == 11192
    assert xo.physical_qubits(st.FEMOCO_54, v) == 11192 + 2*27*(19 + 15) == 13028


def test_corridors_count_in_the_volume_and_error():
    v = dirty_variant()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = xo.cross_run(st.FEMOCO_54, v, ea.NOISE['5e-4'], ea.FACTORY_ROUTING[xo.chip_model(v)], ea.EPSILON, ea.ROUND_TIME)
        bare = ea.total_time(st.FEMOCO_54, v['budget'], v['option'], v['distances'], v['model'],
                             ea.EPSILON, ea.ROUND_TIME)
    # 54 slots of 2d - 1 physical qubits for the whole run: 54 x rounds / d
    added = 54*bare['total rounds']/15
    assert r['cost']['total effective volume'] == pytest.approx(bare['total effective volume'] + added, rel=1e-12)
    assert r['cost']['total days'] == bare['total days']               # no rounds
    assert r['logical'] > ea.algo_error(bare, v['distances'], ea.NOISE['5e-4']['p_Z'], v['model'], ea.FACTORY_ROUTING[xo.chip_model(v)])


def test_other_variants_are_the_plain_runs():
    # no corridors: cross_run is the package's own run
    for v in xo.cross_variants(st.FEMOCO_54, st.SETTINGS_54):
        if v['corridors']:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = xo.cross_run(st.FEMOCO_54, v, ea.NOISE['5e-4'], ea.FACTORY_ROUTING[xo.chip_model(v)], ea.EPSILON, ea.ROUND_TIME)
            plain = ea.total_time(st.FEMOCO_54, v['budget'], v['option'], v['distances'], v['model'],
                                  ea.EPSILON, ea.ROUND_TIME)
        assert r['cost'] == plain
        assert xo.physical_qubits(st.FEMOCO_54, v) == \
            ea.physical_qubits(st.FEMOCO_54, v['model'], v['option'], v['distances'])['total']
