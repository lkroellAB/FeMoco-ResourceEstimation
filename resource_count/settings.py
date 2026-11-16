"""
The paper's systems and, per system and cost model, the settings every
result is computed with -- the one place they are set.  Read by
notebooks/visualization.ipynb, scripts/be_breakdown_latex.py and
scripts/error_accounting_latex.py.

FEMOCO_54, FEMOCO_76: the system, N, B, R, C, b_coef, b_rot, and be_const (the
SOS block-encoding constant).

SETTINGS_54, SETTINGS_76: per cost model (the model module, rep_code or
ldpc_code)
  'name'       the model's name in printouts
  'budget'     the compute-zone budget (the rep code's includes the output
               register's access corridors, LDPC needs none)
  'option'     the block-encoding options: RPrep+Rot's lookup (method, K, lam)
               and the preps' SelectSwap variety (outer_variety,
               inner_variety)
  'distances'  the distances chosen at each noise level ({level: distances},
               the levels of error_accounting.NOISE), each a dict of the
               model's own distance keywords: rep code d_rep; LDPC d_rep and
               d_LDPC, d_LDPC = 12, 16 or 22 only.  They are
               error_accounting.choose_distance's choice: the notebook's
               'Distance choice' cell checks them, scripts/error_accounting_latex.py
               stops if they differ, and tests/test_error_accounting.py
               checks them
  'scan'       the distances the notebook's error-accounting cells print
"""
from resource_count import cost_functions as rep_code
from resource_count import cost_functions_ldpc as ldpc_code

FEMOCO_54 = dict(N=54, B=27, R=10, C=27, b_coef=9, b_rot=16, be_const=21.4)
SETTINGS_54 = {
    rep_code:  dict(name='rep code', budget=48,
                    distances={'5e-4': dict(d_rep=15), '1e-3': dict(d_rep=17)},
                    scan=[dict(d_rep=d) for d in range(11, 22, 2)],
                    option=dict(method='qrom', K=1, lam=2,
                                outer_variety='clean', inner_variety='clean')),
    ldpc_code: dict(name='LDPC', budget=32,
                    distances={'5e-4': dict(d_rep=15, d_LDPC=16), '1e-3': dict(d_rep=17, d_LDPC=22)},
                    scan=[dict(d_rep=d, d_LDPC=d) for d in (12, 16, 22)],
                    # RPrep+Rot: optimize.search_rprep_rot's pick; preps:
                    # optimize.search_prep_variety's (dirty pays for InnerPrep's
                    # large tables, never for OuterPrep's)
                    option=dict(method='dirty_restricted', K=2, lam=2,
                                outer_variety='clean', inner_variety='dirty')),
}

FEMOCO_76 = dict(N=76, B=57, R=15, C=19, b_coef=9, b_rot=15, be_const=43.7)
SETTINGS_76 = {
    rep_code:  dict(name='rep code', budget=45,
                    distances={'5e-4': dict(d_rep=15), '1e-3': dict(d_rep=17)},
                    scan=[dict(d_rep=d) for d in range(11, 26, 2)],
                    option=dict(method='qrom', K=1, lam=2,
                                outer_variety='clean', inner_variety='clean')),
    ldpc_code: dict(name='LDPC', budget=30,
                    distances={'5e-4': dict(d_rep=15, d_LDPC=16), '1e-3': dict(d_rep=19, d_LDPC=22)},
                    scan=[dict(d_rep=d, d_LDPC=d) for d in (12, 16, 22)],
                    # RPrep+Rot: optimize.search_rprep_rot's pick; preps:
                    # optimize.search_prep_variety's (dirty pays for InnerPrep's
                    # large tables, never for OuterPrep's)
                    option=dict(method='dirty_full', K=4, lam=2,
                                outer_variety='clean', inner_variety='dirty')),
}

# the molecules by name, each (system, settings)
MOLECULES = {'FeMoCo-54': (FEMOCO_54, SETTINGS_54), 'FeMoCo-76': (FEMOCO_76, SETTINGS_76)}
