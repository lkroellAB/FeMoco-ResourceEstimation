"""
LaTeX table of one block encoding's cost breakdown, for the paper:
FeMoco-54 at the distances chosen at a noise level, with either cost model.

    pixi run python scripts/be_breakdown_latex.py rep 5e-4                # print it
    pixi run python scripts/be_breakdown_latex.py ldpc 1e-3 table.tex     # also write it

    rep    repetition-cat architecture
    ldpc   hybrid LDPC-cat architecture
    5e-4, 1e-3   the noise level (error_accounting.NOISE) whose chosen
                 distances the table is at

Two subtables:
  (a) rounds, Toffolis and i-states, by subroutine and then by primitive, in
      one tabular with a shared total (both blocks add up to the same totals);
  (b) logical qubit-cycles (effective volume) by primitive.  One column for
      the rep code; for LDPC one per layer (rep layer and LDPC layer), then
      their sum and its share.  Each layer is converted by the model: the rep
      layer's physical qubit-rounds / d_rep(2 d_rep - 1), the LDPC layer's
      logical qubit-rounds / d_LDPC.

Primitives (the code's operations) are grouped as high_level_cost.report
groups them ('operation:step' counts in 'operation'; model.FALLBACK parts mark
their row with a star), then merged further for the paper by MERGE.  The
primitives below 0.1% of the effective volume (the rows (b) would show as
'<0.1%') are combined into one 'Rest' row, in (b) and in (a) alike; the
captions list them.  Every total is checked against accounting.summary and
the volume against Total_time.

The system, its compute-zone budget, options and chosen distances are
resource_count.settings' (FEMOCO_54, SETTINGS_54), as in the notebook and
scripts/error_accounting_latex.py; the run's EPSILON and ROUND_TIME are
resource_count.error_accounting's.
"""
import sys
import warnings
from pathlib import Path

# make resource_count importable however the script is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resource_count import high_level_cost as hl                  # noqa: E402
from resource_count import cost_functions as rep_code             # noqa: E402
from resource_count import cost_functions_ldpc as ldpc_code       # noqa: E402
from resource_count import layout                                 # noqa: E402
from resource_count import accounting as acc                      # noqa: E402
from resource_count import error_accounting as ea                 # noqa: E402
from resource_count import settings as st                         # noqa: E402

# ---- the system and its settings (resource_count.settings), and per cost
# model the caption text.  'architecture' is filled in with the distances
SYSTEM, SETTINGS = st.FEMOCO_54, st.SETTINGS_54
# the table's label, per noise level (5e-4 keeps the label the paper uses)
LABEL_SUFFIX = {'5e-4': '', '1e-3': '_1e-3'}
CONFIGS = {
    'rep': dict(model=rep_code,
                architecture=r'the repetition-cat architecture at $d = {d_rep}$',
                volume=r'Effective logical volume $V_\mathrm{log,eff}$ by operation, i.e., physical qubit-cycles '
                       r'divided by $d(2d-1)$.'),
    'ldpc': dict(model=ldpc_code,
                 architecture=r'the LDPC-cat architecture at $d_\mathrm{{rep}} = {d_rep}$, '
                              r'$d_\mathrm{{LDPC}} = {d_LDPC}$',
                 volume=r"Effective logical volume $V_\mathrm{log,eff}$ by operation and layer, i.e., the repetition "
                        r"layer's physical qubit-cycles divided by $d_\mathrm{rep}(2d_\mathrm{rep}-1)$ and the LDPC "
                        r"layer's logical qubit-cycles divided by $d_\mathrm{LDPC}$. The share is of the total."),
}
COUNTS_CAPTION = (r'Number of syndrome extraction rounds, Toffoli gates and $\keti$ states by subroutine and by '
                  r'operation. The subroutine numbers take the number of calls (in brackets) into account, so '
                  r'that they make up the whole block encoding. Both subroutines and operations break down the '
                  r'same totals. Here, shuffle denotes the procedure to prepare the compute zone for the '
                  r'\textsc{Adder} layout{mbu}.')
ROTATION_NOTE = r'the parity-\textsc{Cnot} gates necessary for phase gradient addition'
# what MBU counts: in COUNTS_CAPTION when MBU is a row of its own, after the
# Rest list when Rest combines it
MBU_ROW_NOTE = ' and MBU refers to explicit uncomputation of lookup table output'
MBU_REST_NOTE = 'MBU counts explicit MBU of lookup table outputs'
# the lookups in the caption, from the block-encoding options (lookups_text)
METHOD_TEX = {'qrom': r'skew-tree \textsc{{Qrom}}', 'clean': r'clean \textsc{{SelectSwap}} ($K = {K}$)',
              'dirty_restricted': r'restricted dirty \textsc{{SelectSwap}} ($K = {K}$)',
              'dirty_full': r'full-access dirty \textsc{{SelectSwap}} ($K = {K}$)'}
PREP_TEX = {'outer_variety': r'\textsc{OuterPrep}', 'inner_variety': r'\textsc{InnerPrep}'}


def lookups_text(option):
    '''the caption's description of the lookups, from the block-encoding
    options: RPrep+Rot's method, K and lam, and the preps using dirty
    SelectSwap'''
    text = (r'with \textsc{Rprep + Rot} implemented by lookup table batching using '
            + METHOD_TEX[option['method']].format(K=option['K'])
            + rf' loading $\lambda = {option["lam"]}$ rotation angles at a time')
    dirty = [PREP_TEX[key] for key in ('outer_variety', 'inner_variety') if option[key] == 'dirty']
    if dirty:
        text += ', and ' + ' and '.join(dirty) + r' using dirty \textsc{SelectSwap}'
    return text
# in table (b)'s caption, when the idle output row is shown
OUTPUT_NOTE = 'Idle output qubits refer to already loaded rotation angles, idle during other operations.'

# ---- names in the paper ----------------------------------------------------
# primitives merged for the paper: the code's operation -> the row it joins
MERGE = {'PlainSwap': 'swap',               # both are SWAPs
         'idle parked': 'idle output'}      # angles waiting their turn and the live angle
SUBROUTINE_NAME = {'RPrep+Rot': r'\textsc{Rprep + Rot}', 'InnerPrep': r'\textsc{InnerPrep}',
                   'InnerPrep_dag': r'\textsc{InnerPrep}$^\dagger$', 'OuterPrep': r'\textsc{OuterPrep}',
                   'OuterPrep_dag': r'\textsc{OuterPrep}$^\dagger$', 'Spin': r'\textsc{Spin}',
                   'Maj': r'\textsc{Maj}', 'OuterRef': r'\textsc{OuterRef}', 'InnerRef': r'\textsc{InnerRef}'}
PRIMITIVE_NAME = {'qrom': r'\textsc{Qrom}', 'adder': r'\textsc{Adder}', 'fixup': r'\textsc{Phaseup}',
                  'shuffle': 'Shuffle', 'rotation': 'Rotation', 'swap': r'\textsc{Swap}',
                  'comparator': 'Comparator', 'qc-add': 'QC addition', 'maj': 'Majorana',
                  'reflection': 'Reflection', 'mbu': 'MBU'}
# the idle zones (layout.py): the address zone holds the X and B registers and
# the two spin-selection qubits; the flag qubit is a zone of its own
IDLE_NAME = {'system': 'System (idle)', 'fourier': 'Fourier state (idle)',
             'address': 'Address and spin qubits (idle)', 'QPE': 'QPE qubits (idle)',
             'output': 'Output (idle)', 'flag': 'Flag (idle)'}
# the primitives as the captions name them (lists of what 'Rest' combines)
CAPTION_NAME = {'qrom': r'\textsc{Qrom}', 'adder': r'\textsc{Adder}', 'fixup': r'\textsc{Phaseup}',
                'shuffle': 'shuffle', 'rotation': 'rotation', 'swap': r'\textsc{Swap}',
                'comparator': 'comparator', 'qc-add': 'quantum-classical-addition', 'reflection': r'\textsc{Ref}',
                'maj': 'remaining operations for Majorana application', 'mbu': 'MBU'}
LAYER_NAME = {'rep': 'Repetition layer', 'ldpc': 'LDPC layer'}


# the marks of estimated rows: '$^*$' when every part of the row is timed at
# TOFF_TIME*d per Toffoli (model.FALLBACK), PARTIAL_MARK when only some are.
# PARTIAL_NOTE says, per partly estimated primitive, what is estimated; a
# partly estimated primitive without a note stops the script
STAR, PARTIAL_MARK = '$^*$', r'$^{\ddagger}$'
PARTIAL_NOTE = {'fixup': r'marks \textsc{Phaseup}, of which only the Toffolis of the space-space split are '
                         'estimated this way'}


def estimate_marks(be, model):
    '''{primitive: mark} for the rows with parts in model.FALLBACK (after
    MERGE): STAR when all of the row's parts are, PARTIAL_MARK when the row
    also has compiled parts (e.g. the fixup's QROM, SWAP and Hadamard steps)'''
    estimated, compiled = set(), set()
    for name in be['parts']:
        op = name.split(':')[0]
        op = MERGE.get(op, op)
        (estimated if name in model.FALLBACK else compiled).add(op)
    marks = {op: PARTIAL_MARK if op in compiled else STAR for op in estimated}
    missing = [op for op, mark in marks.items() if mark == PARTIAL_MARK and op not in PARTIAL_NOTE]
    assert not missing, f'partly estimated primitives without a PARTIAL_NOTE: {missing}'
    return marks


def primitive_name(op, marks, names=PRIMITIVE_NAME):
    '''the row name of a primitive (or idle zone) with its mark (marks, from
    estimate_marks); names=CAPTION_NAME for a caption'''
    if op.startswith('idle '):
        zone = op[len('idle '):]
        return IDLE_NAME[zone] if names is PRIMITIVE_NAME else IDLE_NAME[zone].lower()
    return names[op] + marks.get(op, '')


def listed(names):
    '''"a", "a and b", "a, b and c"'''
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def grouped(be, field, model):
    '''{primitive: total of field(part)} over the block encoding's parts,
    largest first, without zero rows, and the total of the not-compiled
    (model.FALLBACK) parts: only those parts, so a partly estimated row (the
    fixup) contributes only its estimated parts.'''
    rows, fallback = {}, 0
    for name, part in be['parts'].items():
        op = name.split(':')[0]
        op = MERGE.get(op, op)
        rows[op] = rows.get(op, 0) + field(part)
        if name in model.FALLBACK:
            fallback += field(part)
    rows = {k: v for k, v in sorted(rows.items(), key=lambda kv: -kv[1]) if v != 0}
    return rows, fallback


def num(x):
    '''an integer with thin spaces between thousands, in math mode'''
    return '$' + f'{int(round(x)):,}'.replace(',', r'\,') + '$'


def sci(x):
    '''x as $m.mm\\times 10^{e}$ (three significant figures), in math mode.
    The mantissa comes from Python's own rounding ('.2e'), so 9.996e6 reads
    1.00e7, never 10.00e6.'''
    mantissa, exponent = f'{x:.2e}'.split('e')
    return f'${mantissa}\\times 10^{{{int(exponent)}}}$'


def below_shown(x, total):
    '''True when share(x, total) shows '<0.1%' (under 0.05% before rounding)'''
    return 100*x/total < 0.05


def share(x, total):
    return r'$<0.1\%$' if below_shown(x, total) else f'{100*x/total:.1f}\\%'


def tabular(blocks, total_row):
    '''blocks: [(header, rows)], each with its own header row; the body rows
    alternate grey and white, from grey in each block; one total row after
    them.  Header and total white, set off by \\hline.'''
    cols = len(blocks[0][0])
    out = [r'\begin{tabular}{' + '|'.join('c'*cols) + r'}\hline']
    for n_block, (header, rows) in enumerate(blocks):
        if n_block:
            out.append(r'    \hline')
        out.append('    ' + ' & '.join(header) + r' \\ \hline')
        out += ['    ' + (r'\rowcolor{gray!10} ' if n % 2 == 0 else '') + ' & '.join(row) + r' \\'
                for n, row in enumerate(rows)]
    out += [r'    \hline', '    ' + ' & '.join(total_row) + r' \\ \hline', r'\end{tabular}']
    return '\n'.join(out)


def subtable(body, width, caption, label):
    '''a subtable with its caption below the tabular, as in the paper'''
    return '\n'.join([rf'    \begin{{subtable}}[t]{{{width}}}', r'    \centering',
                      '\n'.join('    ' + line for line in body.splitlines()),
                      rf'    \caption{{{caption}}}\label{{{label}}}',
                      r'    \end{subtable}'])


def breakdown_latex(name, level):
    cfg = CONFIGS[name]
    model = cfg['model']
    settings = SETTINGS[model]
    distances, option, budget = settings['distances'][level], settings['option'], settings['budget']
    p, buffer_budget = SYSTEM, layout.buffer_zone_size()
    args = (p['N'], p['B'], p['R'], p['C'], p['b_coef'], p['b_rot'])
    be = hl.BE_cost(*args, budget, distances, model=model, buffer_budget=buffer_budget, **option)
    run = hl.Total_time(*args, budget, buffer_budget, distances, p['be_const'],
                        epsilon=ea.EPSILON, round_time=ea.ROUND_TIME, model=model, **option)
    s = acc.summary(be)
    total_rounds, total_toff, total_i = s['Temporal cost'], s['Toffoli cost'], s['i-state cost']
    label = f'tab:be_femoco54_{name}{LABEL_SUFFIX[level]}'
    marks = estimate_marks(be, model)          # $^*$ / $^{\ddagger}$ per estimated primitive

    # ---- logical qubit-cycles by primitive, per layer (table (b)) ----------
    layers = acc.layers(be)
    logical, fb_logical, idle = {}, 0, 0       # {layer: {primitive: logical qubit-cycles}}
    for layer in layers:
        convert = lambda v, layer=layer: model.logical_qubit_cycles(layer, v, **distances)
        volume, fb_volume = grouped(be, lambda part, layer=layer: part['volume'].get(layer, 0), model)
        assert abs(sum(volume.values()) - acc.volume(be, layer)) < 1e-6*acc.volume(be, layer)
        logical[layer] = {k: convert(v) for k, v in volume.items()}
        expected = run[f'total logical qubit cycles [{layer}]']
        assert abs(run['calls']*sum(logical[layer].values()) - expected) < 1e-6*expected
        fb_logical += convert(fb_volume)
        idle += convert(acc.idle_volume(be, layer))
    combined = {}
    for per_op in logical.values():
        for k, v in per_op.items():
            combined[k] = combined.get(k, 0) + v
    combined = dict(sorted(combined.items(), key=lambda kv: -kv[1]))
    total_logical = sum(combined.values())
    assert abs(run['calls']*total_logical - run['total effective volume']) < 1e-6*run['total effective volume']

    # the primitives combined into 'Rest', in both tables: those below 0.1%
    # of the effective volume, in (b)'s order
    rest = [k for k, v in combined.items() if below_shown(v, total_logical)]

    # ---- (a) by subroutine, then by primitive -------------------------------
    # BE_cost's breakdown: per subroutine, its calls and its rounds, Toffolis
    # and i-states per call
    subs = [(r['name'], r['calls'], r['calls']*r['rounds'], r['calls']*r['toffoli'], r['calls']*r['i-states'])
            for r in be['breakdown']]
    assert sum(r[2] for r in subs) == total_rounds
    assert sum(r[3] for r in subs) == total_toff
    assert sum(r[4] for r in subs) == total_i

    rounds, fb_rounds = grouped(be, lambda part: part['rounds'], model)
    toff, fb_toff = grouped(be, lambda part: part['toffoli'], model)
    ist, _ = grouped(be, lambda part: part['i-states'], model)
    assert sum(rounds.values()) == total_rounds
    assert sum(toff.values()) == total_toff and sum(ist.values()) == total_i
    assert set(toff) == set(ist) and set(toff) <= set(rounds)

    def counts(r, t, i):
        '''rounds, Toffolis and i-states with their shares; a dash where a
        primitive uses no resource states (e.g. the shuffle)'''
        cells = [num(r), share(r, total_rounds)]
        cells += [num(t), share(t, total_toff)] if t else ['--', '--']
        cells += [num(i), share(i, total_i)] if i else ['--', '--']
        return cells

    rest_a = [k for k in rest if k in rounds]       # an idle zone has no rounds, so is not in (a)
    columns = ['Rounds', 'Share', 'Toffolis', 'Share', r'$\keti$ states', 'Share']
    sub_rows = [[f'{SUBROUTINE_NAME[sub]} ($\\times {n}$)'] + counts(r, t, i)
                for sub, n, r, t, i in sorted(subs, key=lambda x: -x[2])]
    prim_rows = [[primitive_name(k, marks)] + counts(v, toff.get(k, 0), ist.get(k, 0))
                 for k, v in rounds.items() if k not in rest_a]
    if rest_a:
        prim_rows.append(['Rest'] + counts(sum(rounds[k] for k in rest_a), sum(toff.get(k, 0) for k in rest_a),
                                           sum(ist.get(k, 0) for k in rest_a)))
    table_a = tabular([(['Subroutine'] + columns, sub_rows), (['Operation'] + columns, prim_rows)],
                      ['Total', num(total_rounds), r'100\%', num(total_toff), r'100\%', num(total_i), r'100\%'])
    caption_a = COUNTS_CAPTION.replace('{mbu}', MBU_ROW_NOTE if 'mbu' in rounds and 'mbu' not in rest_a else '')
    if rest_a:
        caption_a += (f' Rest combines {listed([primitive_name(k, marks, CAPTION_NAME) for k in rest_a])}, i.e., '
                      rf'the operations with less than 0.1\% each of the effective volume in \zcref{{{label}_volume}}')
        notes = ([f'rotation refers to {ROTATION_NOTE}'] if 'rotation' in rest_a else []) \
            + ([MBU_REST_NOTE] if 'mbu' in rest_a else [])
        caption_a += (', where ' + ' and '.join(notes) if notes else '') + '.'
    if 'rotation' not in rest_a:
        caption_a += f' Rotation refers to {ROTATION_NOTE}.'

    # ---- (b) logical qubit-cycles by primitive ------------------------------
    # the volumes in scientific notation (sci), as they span several orders
    # of magnitude
    kept = [k for k in combined if k not in rest]
    if len(layers) == 1:
        rows_b = [[primitive_name(k, marks), sci(combined[k]), share(combined[k], total_logical)] for k in kept]
        if rest:
            v = sum(combined[k] for k in rest)
            rows_b.append(['Rest', sci(v), share(v, total_logical)])
        table_b = tabular([(['Operation', 'Logical qubit-cycles', 'Share'], rows_b)],
                          ['Total', sci(total_logical), r'100\%'])
    else:
        def per_layer(ops):
            '''each layer's logical qubit-cycles of the primitives `ops`; a
            dash where none of them uses that layer'''
            return [sci(sum(logical[L][k] for k in ops if k in logical[L]))
                    if any(k in logical[L] for k in ops) else '--' for L in layers]
        rows_b = [[primitive_name(k, marks)] + per_layer([k]) + [sci(combined[k]), share(combined[k], total_logical)]
                  for k in kept]
        if rest:
            v = sum(combined[k] for k in rest)
            rows_b.append(['Rest'] + per_layer(rest) + [sci(v), share(v, total_logical)])
        table_b = tabular([(['Operation'] + [LAYER_NAME[L] for L in layers] + ['Total', 'Share'], rows_b)],
                          ['Total'] + [sci(sum(logical[L].values())) for L in layers]
                          + [sci(total_logical), r'100\%'])
        assert layers == ['rep', 'ldpc'], layers      # the caption names these two
    caption_b = cfg['volume']
    if 'idle output' in kept:
        caption_b += ' ' + OUTPUT_NOTE
    caption_b += f' Idle qubits account for {share(idle, total_logical)} of the algorithmic effective volume.'
    if rest:
        caption_b += rf' Rest combines {listed([primitive_name(k, marks, CAPTION_NAME) for k in rest])}, each below 0.1\%.'

    partial = [op for op, mark in marks.items() if mark == PARTIAL_MARK]
    caption = (rf'Cost breakdown of one block encoding for FeMoco-54 in {cfg["architecture"].format(**distances)}, '
               rf'{lookups_text(option)}. '
               rf'{STAR} denotes operations whose duration is estimated at ${model.TOFF_TIME}d$ per Toffoli'
               + ''.join(f'; {PARTIAL_MARK} {PARTIAL_NOTE[op]}' for op in partial) + '. '
               + (r'These non-compiled parts' if partial else r'Such non-compiled operations')
               + rf' total {share(fb_rounds, total_rounds)} of rounds, '
               rf'{share(fb_logical, total_logical)} of logical qubit-cycles, '
               rf'and {share(fb_toff, total_toff)} of Toffoli gates.')
    return '\n'.join([
        r'\begin{table}[hbtp]', r'    \centering',
        subtable(table_a, r'\textwidth', caption_a, f'{label}_counts'),
        r'    \par\bigskip',
        subtable(table_b, r'\textwidth', caption_b, f'{label}_volume'),
        '', rf'    \caption{{{caption}}}', rf'    \label{{{label}}}',
        r'\end{table}'])


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in CONFIGS or sys.argv[2] not in LABEL_SUFFIX:
        sys.exit(f"usage: be_breakdown_latex.py {{{'|'.join(CONFIGS)}}} {{{'|'.join(LABEL_SUFFIX)}}} [out.tex]")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # budget warnings are not the point here
        latex = breakdown_latex(sys.argv[1], sys.argv[2])
    print(latex)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(latex + '\n')
