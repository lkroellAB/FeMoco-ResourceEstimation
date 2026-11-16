"""
Correctness tests for optimize.py: the search functions must actually return
the cheapest option (verified against an independent recomputation, not
trusted blindly), and must not leak warnings from candidates they discard.
"""
import warnings

import pytest

from resource_count import accounting as acc
from resource_count import cost_functions as cf
from resource_count import cost_functions_ldpc as ldpc
from resource_count import high_level_cost as hlc
from resource_count import batching as bt
from resource_count import optimize as opt


S = acc.summary

OUTER = dict(N=54, b_rot=16, B=27)

# RPrep+Rot batch sizes searched in these tests: lam = 1 .. 8
LAMS = range(1, 9)
# swap parameters searched: K = 2**alpha up to 2**4 = 16
ALPHA_MAX = 4
INNER = dict(N=54, b_rot=16, X=486)


BOTH = ('clean', 'dirty')


class TestSearchPrepVariety:
    def test_chosen_is_cheapest_of_the_candidates(self):
        r = opt.search_prep_variety(hlc.OuterPrep, (54, 9), 57, dict(d_rep=20), varieties=BOTH, **OUTER, model=cf)
        all_costs = [S(r['chosen'])['Temporal cost']] + [S(a)['Temporal cost'] for a in r['alternatives']]
        assert S(r['chosen'])['Temporal cost'] == min(all_costs)

    def test_default_uses_the_cost_models_varieties(self):
        # the rep-code layout has no system access corridors: clean only
        r = opt.search_prep_variety(hlc.OuterPrep, (54, 9), 57, dict(d_rep=20), **OUTER, model=cf)
        assert cf.SYSTEM_Z_ACCESS is False
        assert r['chosen']['variety'] == 'clean'
        assert r['alternatives'] == []

    def test_ldpc_default_compares_both_varieties(self):
        r = opt.search_prep_variety(hlc.OuterPrep, (54, 9), 57, dict(d_rep=16, d_LDPC=16), **OUTER, model=ldpc)
        assert {r['chosen']['variety']} | {a['variety'] for a in r['alternatives']} == set(BOTH)

    def test_works_for_inner_prep_too(self):
        r = opt.search_prep_variety(hlc.InnerPrep, (270, 57, 9), 57, dict(d_rep=20), varieties=BOTH, **INNER, model=cf)
        assert r['chosen']['variety'] in BOTH
        assert len(r['alternatives']) == 1

    def test_respects_custom_metric(self):
        r_time = opt.search_prep_variety(hlc.OuterPrep, (54, 9), 57, dict(d_rep=20),
                                         metric='Temporal cost', varieties=BOTH, **OUTER, model=cf)
        r_toff = opt.search_prep_variety(hlc.OuterPrep, (54, 9), 57, dict(d_rep=20),
                                         metric='Toffoli cost', varieties=BOTH, **OUTER, model=cf)
        all_by_toff = [r_toff['chosen']] + r_toff['alternatives']
        assert S(r_toff['chosen'])['Toffoli cost'] == min(S(c)['Toffoli cost'] for c in all_by_toff)
        # sanity: both metrics are well-defined even if they happen to agree here
        assert S(r_time['chosen'])['Temporal cost'] <= (S(r_time['alternatives'][0])['Temporal cost']
                                                       if r_time['alternatives'] else float('inf'))


class TestSearchRPrepRot:
    def test_chosen_is_cheapest_of_all_candidates(self):
        r = opt.search_rprep_rot(54, 27, 10, 27, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, ancilla_budget=57, model=cf)
        all_costs = [S(r['chosen'])['Temporal cost']] + [S(a)['Temporal cost'] for a in r['alternatives']]
        assert S(r['chosen'])['Temporal cost'] == min(all_costs)

    def test_chosen_respects_ancilla_budget(self):
        budget = 57
        r = opt.search_rprep_rot(54, 27, 10, 27, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, ancilla_budget=budget, model=cf)
        assert r['chosen']['Clean ancilla'] <= budget
        for alt in r['alternatives']:
            assert alt['Clean ancilla'] <= budget

    def test_chosen_full_record_has_a_method_tag(self):
        r = opt.search_rprep_rot(54, 27, 10, 27, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, ancilla_budget=57, model=cf)
        assert r['chosen']['method'] in bt.METHODS

    def test_none_when_budget_impossibly_tight(self):
        assert opt.search_rprep_rot(54, 27, 10, 27, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, ancilla_budget=0, model=cf) is None

    def test_no_budget_still_returns_every_feasible_method(self):
        r = opt.search_rprep_rot(54, 27, 10, 27, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, model=cf)
        methods_seen = {r['chosen']['method']} | {a['method'] for a in r['alternatives']}
        assert methods_seen <= set(bt.METHODS)

    def test_exploring_candidates_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')   # any warning here fails the test
            opt.search_rprep_rot(54, 27, 10, 27, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, ancilla_budget=57, model=cf)

    def test_chosen_record_is_a_real_rprep_rot_result(self):
        # the chosen record carries its own option back (method/K/lam/runs), so
        # replaying it is a direct call -- no translation step to get wrong
        r = opt.search_rprep_rot(54, 27, 10, 27, 16, dict(d_rep=20), lams=LAMS, alpha_max=ALPHA_MAX, ancilla_budget=57, model=cf)
        chosen = r['chosen']
        replay = hlc.RPrep_Rot(54, 27, 10, 27, 16, dict(d_rep=20),
                               method=chosen['method'], K=chosen['K'],
                               lam=chosen['size of batches'], runs=chosen['runs'],
                               ancilla_budget=57, model=cf)
        assert S(replay)['Temporal cost'] == S(chosen)['Temporal cost']
        assert S(replay)['Toffoli cost'] == S(chosen)['Toffoli cost']
        assert replay['lookups'] == chosen['lookups']


# (model, distances) for both cost models
MODELS = {'rep': (cf, dict(d_rep=20)), 'ldpc': (ldpc, dict(d_rep=16, d_LDPC=16))}


@pytest.mark.parametrize('name', list(MODELS))
class TestSearchRespectsTheComputeZone:
    """The budget bounds the compute zone -- the lookup's clean ancilla plus
    the access corridors of its output register -- not the clean ancilla
    alone, in either model."""

    BUDGET = 48

    def search(self, name):
        model, dist = MODELS[name]
        return opt.search_rprep_rot(54, 27, 10, 27, 16, dist, lams=LAMS, alpha_max=ALPHA_MAX,
                                    ancilla_budget=self.BUDGET, model=model)

    def test_every_candidate_fits_with_its_corridors(self, name):
        r = self.search(name)
        for c in [r['chosen']] + r['alternatives']:
            row = bt.batching_cost(54, 27, 10, 16, MODELS[name][1], **bt.as_option(c),
                                   model=MODELS[name][0])
            assert row['Clean ancilla'] + row['Access corridors'] <= self.BUDGET
            assert c['Compute zone'] == row['Compute zone']

    def test_front_is_not_beaten_on_both_toffoli_and_rounds(self, name):
        r = self.search(name)
        everyone = [r['chosen']] + r['alternatives']
        cost = lambda c: (S(c)['Toffoli cost'], S(c)['Temporal cost'])
        for f in r['front']:
            assert not any(cost(o)[0] <= cost(f)[0] and cost(o)[1] <= cost(f)[1]
                           and cost(o) != cost(f) for o in everyone)
        # and everything off the front is beaten by something on it
        for o in everyone:
            if all(o is not f for f in r['front']):
                assert any(cost(f)[0] <= cost(o)[0] and cost(f)[1] <= cost(o)[1]
                           for f in r['front'])

    def test_chosen_on_rounds_is_on_the_front(self, name):
        r = self.search(name)
        assert any(r['chosen'] is f for f in r['front'])


@pytest.mark.parametrize('name', list(MODELS))
def test_include_rejected_costs_every_option_without_changing_the_choice(name):
    model, dist = MODELS[name]
    budget = {'rep': 48, 'ldpc': 32}[name]       # set A's compute zone per code
    kw = dict(lams=LAMS, alpha_max=ALPHA_MAX, ancilla_budget=budget, model=model)
    plain = opt.search_rprep_rot(54, 27, 10, 27, 16, dist, **kw)
    full = opt.search_rprep_rot(54, 27, 10, 27, 16, dist, include_rejected=True, **kw)
    viable = [full['chosen']] + full['alternatives']
    # the same viable options and choice; every swept option appears once
    assert [bt.as_option(r) for r in viable] == \
        [bt.as_option(r) for r in [plain['chosen']] + plain['alternatives']]
    swept = bt.batching_table(54, 27, 10, 16, dist, lams=LAMS, alpha_max=ALPHA_MAX,
                              clean_budget=budget, include_rejected=True, model=model)
    assert len(viable) + len(full['rejected']) == len(swept)
    assert all(r['rejected'] == () for r in viable)
    assert all(r['rejected'] for r in full['rejected'])


class TestPrepOptions:
    """optimize.prep_options costs every prep SelectSwap option -- a plain
    QROM, then each variety at each forced K -- as a whole prep, with the
    same viability rules as batching.py."""

    def options(self, model, distances, budget):
        return opt.prep_options(hlc.InnerPrep, (432, 27, 9), budget, distances, model=model,
                                alphas=range(0, 4), **INNER)

    def test_one_qrom_then_each_variety_at_each_k(self):
        r = self.options(ldpc, dict(d_rep=16, d_LDPC=16), 32)
        assert [(o['method'], o['K']) for o in r] == \
            [('qrom', 1)] + [(v, K) for v in ('clean', 'dirty') for K in (2, 4, 8)]

    def test_rep_code_dirty_has_no_system_access(self):
        r = self.options(cf, dict(d_rep=20), 48)
        assert all('no system Z access' in o['rejected'] for o in r if o['method'] == 'dirty')

    def test_compute_zone_counts_the_models_corridors(self):
        # clean K = 2 holds 2*9 = 18 output qubits: + 9 corridors in the rep
        # code, none in LDPC
        rep = {(o['method'], o['K']): o for o in self.options(cf, dict(d_rep=20), 48)}
        ld = {(o['method'], o['K']): o for o in self.options(ldpc, dict(d_rep=16, d_LDPC=16), 32)}
        assert rep[('clean', 2)]['Compute zone'] == 27
        assert ld[('clean', 2)]['Compute zone'] == 18
        # K = 4 (36 qubits) is over LDPC's budget of 32; dirty K = 4 is not
        assert 'compute zone' in ld[('clean', 4)]['rejected']
        assert ld[('dirty', 4)]['viable']

    def test_matches_the_prep_costed_directly(self):
        o = {(o['method'], o['K']): o for o in self.options(ldpc, dict(d_rep=16, d_LDPC=16), 32)}[('dirty', 4)]
        direct = hlc.InnerPrep(432, 27, 9, 32, dict(d_rep=16, d_LDPC=16), variety='dirty', alpha=2, model=ldpc, **INNER)
        assert S(o) == S(direct)


def test_ldpc_needs_no_access_corridors():
    rows = bt.batching_table(54, 27, 10, 16, dict(d_rep=16, d_LDPC=16), lams=LAMS, alpha_max=ALPHA_MAX,
                             include_rejected=True, model=ldpc)
    assert all(r['Access corridors'] == 0 and r['System corridors'] == 0 for r in rows)
    assert all(r['Compute zone'] == r['Clean ancilla'] for r in rows)
