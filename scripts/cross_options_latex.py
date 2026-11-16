"""
LaTeX tables comparing the lookup options across the two models, for the
paper: each model with its own chosen options and with the other's
(resource_count.cross_options, a contained example: cross_variants), all at
the same distances (cross_options.CROSS_DISTANCES: d_rep = 15 for the rep
code, d_rep = 15, d_LDPC = 16 for LDPC), at p_Z = 5e-4.

    pixi run python scripts/cross_options_latex.py              # print them
    pixi run python scripts/cross_options_latex.py table.tex    # also write them

The variants:
    rep, chosen          the rep code's own options
    rep, dirty options   LDPC's chosen options for the same molecule (dirty
                         SelectSwap in RPrep+Rot and InnerPrep).  Not
                         possible in the rep-code layout as it is: it needs Z
                         access to the system qubits, i.e. access corridors
                         under them, one per two system qubits (N in all).
                         Costed with cross_options.REP_WITH_SYSTEM_Z_ACCESS;
                         the corridors are in its physical qubits (each system
                         half ceil(3N/2) + 2 wide) and, present for the whole
                         run, in its volume and failure probability (an upper
                         bound, cross_options.add_rep_routing)
    LDPC, chosen         LDPC's own options
    LDPC, rep options    the rep code's options in the LDPC model

Four tables:
  1. the primitive breakdown of one block encoding of MOLECULE, per variant
     of PRIMITIVE_VARIANTS (the two cross variants: the chosen ones are
     be_breakdown_latex.py's tables): rounds, Toffolis and i-states by
     primitive, as in be_breakdown_latex.py (its merges, marks and 'Rest'
     rule);
  2. the Hadamard and MBU steps of one block encoding of MOLECULE, per
     variant, by the operation they are part of;
  3. per molecule and variant, the failure probabilities, the physical qubits
     and the durations, as in error_accounting_latex.py (with its SLACK);
  4. the intro comparison: per architecture, without and with the system
     qubits as dirty ancillae, the physical qubits, Toffolis per run and
     expected duration (with SLACK) of both molecules; bold the chosen options
     (the error accounting tables' compilations).  Its caption names the
     faster rep-code configuration at the same footprint (FASTER_LAM), with
     the distance its distance choice gives.
"""
import sys
import warnings
from pathlib import Path

# make resource_count importable however the script is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resource_count import high_level_cost as hl                  # noqa: E402
from resource_count import accounting as acc                      # noqa: E402
from resource_count import layout                                 # noqa: E402
from resource_count import error_accounting as ea                 # noqa: E402
from resource_count import settings as st                         # noqa: E402
from resource_count import cross_options as xo                    # noqa: E402
from resource_count import lambda_scan                            # noqa: E402
# the paper tables' own names, merges, marks and formatting
from be_breakdown_latex import (grouped, estimate_marks, primitive_name, tabular, num, share,  # noqa: E402
                                below_shown, listed, CAPTION_NAME, STAR, PARTIAL_MARK,
                                PARTIAL_NOTE)
from be_breakdown_latex import sci                                          # noqa: E402
from error_accounting_latex import MOLECULE_TEX, MOLECULE_HEADER, MOLECULE_SHORT, sig2, days, SLACK, TABLES  # noqa: E402

MOLECULE = 'FeMoCo-54'          # the molecule of tables 1 and 2
LEVEL = '5e-4'                  # the noise level of table 3
VARIANT_TEX = {'rep, chosen': 'Rep code, chosen options',
               'rep, dirty options': r'Rep code, dirty \textsc{SelectSwap}',
               'LDPC, chosen': 'LDPC, chosen options',
               'LDPC, rep options': r'LDPC, with \textsc{Qrom} only lookup table batching'}
# the variants of table 1: the chosen options' breakdowns are be_breakdown_latex.py's tables
PRIMITIVE_VARIANTS = ('rep, dirty options', 'LDPC, rep options')
# short names in the comparison tables (2 and 3), defined in their captions
VARIANT_SHORT = {'rep, chosen': 'Rep', 'rep, dirty options': 'Rep, dirty',
                 'LDPC, chosen': 'LDPC', 'LDPC, rep options': 'LDPC, rep'}
# the dirty RPrep+Rot lookups: which system qubits they borrow (batching.py:
# restricted access the spin-up half, full access all of them)
ACCESS_TEX = {'dirty_restricted': 'on half the system qubits', 'dirty_full': 'on all system qubits'}


def check_rep_options(names):
    '''the captions describe the rep code's chosen options (run by 'LDPC,
    rep') as QROM lookup table batching with clean SelectSwap in InnerPrep:
    check that for the molecules `names`'''
    for name in names:
        _, settings = st.MOLECULES[name]
        o = settings[st.rep_code]['option']
        assert o['method'] == 'qrom' and o['inner_variety'] == 'clean', (name, o)


def short_note(names):
    '''the caption note defining the short variant names, for the molecules
    `names`: 'Rep, dirty' runs LDPC's chosen options of each molecule
    (cross_options.cross_variants), so its RPrep+Rot lookup is read from
    the settings'''
    lookups = []
    for name in names:
        system, settings = st.MOLECULES[name]
        o = next(v for v in xo.cross_variants(system, settings) if v['label'] == 'rep, dirty options')['option']
        # the note's wording: a dirty RPrep+Rot lookup and a dirty InnerPrep
        assert o['method'] in ACCESS_TEX and o['inner_variety'] == 'dirty', (name, o)
        lookups.append((name, rf"$K = {o['K']}$, $\lambda = {o['lam']}$ {ACCESS_TEX[o['method']]}"))
    check_rep_options(names)
    if len({text for _, text in lookups}) == 1:
        rprep = 'with ' + lookups[0][1]
    else:
        rprep = ' and '.join(f'with {text} for {MOLECULE_TEX[name]}' for name, text in lookups)
    return (r'Rep and LDPC refer to each architecture with its own chosen lookup options. Rep, dirty is the '
            r"repetition-cat architecture with the LDPC-cat architecture's lookup options for the same molecule: "
            r'dirty \textsc{SelectSwap} in \textsc{InnerPrep} and in \textsc{Rprep + Rot}, the latter '
            rf'{rprep}. Lastly, LDPC, rep is the LDPC architecture with the repetition code' + "'s option of "
            r'\textsc{Qrom} lookup table batching and clean \textsc{SelectSwap} in \textsc{InnerPrep}.')
# the steps compared in table 2: step type -> its name, and the host
# operation's name ('' for the standalone operation of that type)
STEP_TEX = {'hadamard': 'Hadamard', 'mbu': 'MBU'}
STANDALONE_TEX = {'mbu': 'of the rotation angles'}
CORRIDOR_NOTE = (r'The rep code with dirty \textsc{SelectSwap} needs $Z$ access to the system qubits it borrows, '
                 r'which the repetition-cat layout provides only with access corridors under them, one per two '
                 r'system qubits ($N$ in all)')
# the error table's version (table 3), which also says how the corridors count
CORRIDOR_ERROR_NOTE = (r'The rep code with dirty \textsc{SelectSwap} needs $Z$ access to the system qubits it '
                       r'borrows, which the repetition-cat layout provides only with access corridors in between '
                       r'them, one per two system qubits ($N$ in total): they are included in the physical qubits '
                       r'count and taken as present for the whole run in its volume to compute '
                       r'$\varepsilon_\mathrm{alg}$.')


def tag(name):
    '''a molecule's name in labels: FeMoCo-54 -> femoco54'''
    return name.lower().replace('-', '')


def distances_tex(v):
    d = v['distances']
    return (rf'$d = {d["d_rep"]}$' if 'd_LDPC' not in d
            else rf'$d_\mathrm{{rep}} = {d["d_rep"]}$, $d_\mathrm{{LDPC}} = {d["d_LDPC"]}$')


def block_encoding(system, v):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hl.BE_cost(system['N'], system['B'], system['R'], system['C'], system['b_coef'], system['b_rot'],
                          v['budget'], v['distances'], model=v['model'], buffer_budget=layout.buffer_zone_size(),
                          **v['option'])


def rest_of(be, v):
    '''the primitives combined into 'Rest', by be_breakdown_latex's rule:
    below 0.1% of the block encoding's effective volume'''
    m, combined = v['model'], {}
    for layer in acc.layers(be):
        volume, _ = grouped(be, lambda part, layer=layer: part['volume'].get(layer, 0), m)
        for k, x in volume.items():
            combined[k] = combined.get(k, 0) + m.logical_qubit_cycles(layer, x, **v['distances'])
    total = sum(combined.values())
    return [k for k, x in sorted(combined.items(), key=lambda kv: -kv[1]) if below_shown(x, total)]


# ---- 1. the primitive breakdown, per variant ---------------------------------
def primitive_tables(name):
    system, settings = st.MOLECULES[name]
    variants = [v for v in xo.cross_variants(system, settings) if v['label'] in PRIMITIVE_VARIANTS]
    assert [v['label'] for v in variants] == list(PRIMITIVE_VARIANTS), [v['label'] for v in variants]
    check_rep_options([name])           # VARIANT_TEX's 'LDPC, rep options'
    out, partial = [], set()
    for n, v in enumerate(variants):
        m = v['model']
        be = block_encoding(system, v)
        s = acc.summary(be)
        total_rounds, total_toff, total_i = s['Temporal cost'], s['Toffoli cost'], s['i-state cost']
        marks = estimate_marks(be, m)
        partial |= {op for op, mark in marks.items() if mark == PARTIAL_MARK}
        rounds, _ = grouped(be, lambda part: part['rounds'], m)
        toff, _ = grouped(be, lambda part: part['toffoli'], m)
        ist, _ = grouped(be, lambda part: part['i-states'], m)
        assert sum(rounds.values()) == total_rounds
        assert sum(toff.values()) == total_toff and sum(ist.values()) == total_i

        def counts(r, t, i):
            cells = [num(r), share(r, total_rounds)]
            cells += [num(t), share(t, total_toff)] if t else ['--', '--']
            cells += [num(i), share(i, total_i)] if i else ['--', '--']
            return cells

        rest = [k for k in rest_of(be, v) if k in rounds]
        rows = [[primitive_name(k, marks)] + counts(x, toff.get(k, 0), ist.get(k, 0))
                for k, x in rounds.items() if k not in rest]
        if rest:
            rows.append(['Rest'] + counts(sum(rounds[k] for k in rest), sum(toff.get(k, 0) for k in rest),
                                          sum(ist.get(k, 0) for k in rest)))
        body = tabular([(['Operation', 'Rounds', 'Share', 'Toffolis', 'Share', r'$\keti$ states', 'Share'], rows)],
                       ['Total', num(total_rounds), r'100\%', num(total_toff), r'100\%', num(total_i), r'100\%'])
        caption = f'{VARIANT_TEX[v["label"]]}, {distances_tex(v)}.'
        if v['corridors']:
            caption += ' Its access corridors cost no rounds.'
        if rest:
            caption += (f' Rest combines {listed([primitive_name(k, marks, CAPTION_NAME) for k in rest])}, '
                        r'each below 0.1\% of the effective volume.')
        label = f'tab:cross_{tag(name)}_{v["label"].replace(", ", "_").replace(" ", "_")}'
        out += ([r'    \par\bigskip'] if n else []) + [
            r'    \begin{subtable}[t]{\textwidth}', r'    \centering',
            '\n'.join('    ' + line for line in body.splitlines()),
            rf'    \caption{{{caption}}}\label{{{label}}}', r'    \end{subtable}']
    caption = (rf'Rounds, Toffoli gates and $\keti$ states of one block encoding for '
               rf'{MOLECULE_TEX[name]} by operation, with the lookup options of either architecture in both. '
               rf'{CORRIDOR_NOTE}. {STAR} denotes operations whose duration is estimated at '
               rf'${st.rep_code.TOFF_TIME}d$ per Toffoli'
               + ''.join(f'; {PARTIAL_MARK} {PARTIAL_NOTE[op]}' for op in sorted(partial)) + '.')
    return '\n'.join([r'\begin{table}[hbtp]', r'    \centering'] + out
                     + ['', rf'    \caption{{{caption}}}', rf'    \label{{tab:cross_{tag(name)}_primitives}}',
                        r'\end{table}'])


# ---- 2. the Hadamard and MBU steps, per variant ------------------------------
def step_table(name):
    system, settings = st.MOLECULES[name]
    variants = xo.cross_variants(system, settings)
    data = []
    for v in variants:
        be = block_encoding(system, v)
        total = acc.summary(be)['Temporal cost']
        steps = {}          # (step type, host operation or '') -> rounds
        for part_name, part in be['parts'].items():
            op, _, step = part_name.partition(':')
            for kind in STEP_TEX:
                if step == kind or (not step and op == kind):
                    host = op if step else ''
                    steps[(kind, host)] = steps.get((kind, host), 0) + part['rounds']
        data.append((v, total, steps))
    rows = []
    for kind, kind_tex in STEP_TEX.items():
        hosts = sorted({h for _, _, steps in data for (k, h) in steps if k == kind},
                       key=lambda h: (h != '', h))
        cells = lambda pick: sum((([num(x), share(x, total)] if x else ['--', '--'])
                                  for v, total, steps in data for x in [pick(steps)]), [])
        rows.append([kind_tex] + cells(lambda steps: sum(x for (k, _), x in steps.items() if k == kind)))
        for h in hosts:
            label = STANDALONE_TEX[kind] if h == '' else 'in ' + CAPTION_NAME[h]
            rows.append([rf'\quad {label}'] + cells(lambda steps, h=h: steps.get((kind, h), 0)))
    header = (['Step'] + [rf'\multicolumn{{2}}{{c{"|" if n < len(data) - 1 else ""}}}{{{VARIANT_SHORT[v["label"]]}}}'
                          for n, (v, _, _) in enumerate(data)])
    sub_header = [''] + ['Rounds', 'Share'] * len(data)
    body = [r'\begin{tabular}{' + '|'.join('c'*(1 + 2*len(data))) + r'}\hline',
            '    ' + ' & '.join(header) + r' \\',
            '    ' + ' & '.join(sub_header) + r' \\ \hline']
    body += ['    ' + (r'\rowcolor{gray!10} ' if n % 2 == 0 else '') + ' & '.join(r) + r' \\'
             for n, r in enumerate(rows)]
    body += [r'    \hline', '    ' + ' & '.join(['Block encoding'] + sum(([num(total), r'100\%']
                                                                         for _, total, _ in data), []))
             + r' \\ \hline', r'\end{tabular}']
    distances = ', '.join(sorted({distances_tex(v) for v, _, _ in data}))
    caption = (rf'Rounds of the Hadamard and MBU steps in one block encoding for {MOLECULE_TEX[name]}, by the '
               rf'operation they are part of, with the lookup options of either architecture in both ({distances}). '
               r"The share is of the block encoding's rounds. " + short_note([name]))
    return '\n'.join([r'\begin{table}[hbtp]', r'    \centering'] + ['    ' + line for line in body]
                     + [rf'    \caption{{{caption}}}', rf'    \label{{tab:cross_{tag(name)}_steps}}', r'\end{table}'])


# ---- 3. the error accounting, per molecule and variant -----------------------
def error_table(level):
    noise = ea.NOISE[level]
    header = [MOLECULE_HEADER, 'Variant', r'$d_\mathrm{rep}$', r'$d_\mathrm{LDPC}$',
              r'$\varepsilon_\mathrm{alg}$', r'$\varepsilon_{\ket{\CXCXX}}$', r'$\varepsilon_{\keti}$',
              'Physical', 'One-shot', r'$\langle t\rangle$']
    header_2 = [''] * 7 + ['qubits', '(days)', '(days)']
    out = [r'\begin{tabular}{' + '|'.join('c'*len(header)) + r'}\hline', '    ' + ' & '.join(header) + r' \\',
           '    ' + ' & '.join(header_2) + r' \\ \hline']
    molecules = list(st.MOLECULES)
    for n_mol, (name, (system, settings)) in enumerate(st.MOLECULES.items()):
        variants = xo.cross_variants(system, settings)
        for n, v in enumerate(variants):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = xo.cross_run(system, v, noise, ea.FACTORY_ROUTING[xo.chip_model(v)], ea.EPSILON, ea.ROUND_TIME)
            phys = xo.physical_qubits(system, v)
            one_shot = SLACK*r['cost']['total days']
            expected = one_shot/(1 - r['fail']) if r['fail'] < 1 else float('inf')
            d = v['distances']
            # the caption's corridors: N for the rep code with dirty options, none otherwise
            assert v['corridors'] == (system['N'] if v['label'] == 'rep, dirty options' else 0), (name, v['label'])
            cells = ([rf'\multirow{{-{len(variants)}}}{{*}}{{{MOLECULE_SHORT[name]}}}' if n == len(variants) - 1 else '',
                      VARIANT_SHORT[v['label']], str(d['d_rep']), str(d['d_LDPC']) if 'd_LDPC' in d else '--',
                      sig2(r['logical']), sig2(r['magic']), sig2(r['istate']), num(phys),
                      days(one_shot), days(expected)])
            line = '    ' + (r'\rowcolor{gray!10} ' if n_mol % 2 == 0 else '') + ' & '.join(cells) + r' \\'
            if n_mol == len(molecules) - 1 and n == len(variants) - 1:
                line += r' \hline'
            out.append(line)
        # no \hline between molecules: the grey/white blocks separate them,
        # as in the error accounting tables
    out.append(r'\end{tabular}')
    level_tex = {'5e-4': r'$p_Z = 5\times 10^{-4}$', '1e-3': r'$p_Z = 10^{-3}$'}[level]
    caption = (rf'Error accounting with the lookup options of either architecture in both, at {level_tex} for all '
               rf'variants. {CORRIDOR_ERROR_NOTE} '
               rf'The physical qubit count includes ancilla, factories and factory routing. The durations include '
               rf'${round(100*(SLACK - 1))}\%$ slack and the expected duration is the one-shot duration divided by '
               rf'$1-p_\mathrm{{fail}}$, with $p_\mathrm{{fail}} = \varepsilon_\mathrm{{alg}} + '
               rf'\varepsilon_{{\ket{{\CXCXX}}}} + \varepsilon_{{\keti}}$. ' + short_note(molecules))
    return '\n'.join([r'\begin{table}[hbtp]', r'    \centering'] + ['    ' + line for line in out]
                     + [rf'    \caption{{{caption}}}', r'    \label{tab:cross_error_accounting}', r'\end{table}'])


# ---- 4. the intro comparison: without and with dirty ancillae ----------------
ARCHITECTURE_TEX = {'rep': 'Repetition-cat', 'LDPC': 'LDPC-cat'}
# the caption's faster rep-code configuration at the same footprint: the
# chosen option with FASTER_LAM angles per batch, in that lam's compute zone
# (lambda_scan.compute_zone), for FASTER_MOLECULE (faster_rep checks it)
FASTER_MOLECULE, FASTER_LAM = 'FeMoCo-54', 3
FASTER_FIGURE = 'fig:tradeoff lambda sweep intro'      # the trade-off figure the caption points to


def faster_rep(level):
    '''the rep code's chosen option for FASTER_MOLECULE with lam =
    FASTER_LAM, at its own distance choice (error_accounting.choose_distance)
    at noise level `level`: its d_rep, checked to be faster than the chosen
    lam with no more physical qubits'''
    system, settings = st.MOLECULES[FASTER_MOLECULE]
    m, s = st.rep_code, settings[st.rep_code]
    option = dict(s['option'], lam=FASTER_LAM)
    budget = lambda_scan.compute_zone(m, FASTER_LAM, system['b_rot'])
    run = lambda dist: ea.total_time(system, budget, option, dist, m, ea.EPSILON, ea.ROUND_TIME)
    _, chosen = ea.choose_distance(run, ea.DISTANCE_SCAN[m], ea.NOISE[level], m, ea.FACTORY_ROUTING[m], ea.TIE)
    ref_dist = s['distances'][level]
    ref = ea.total_time(system, s['budget'], s['option'], ref_dist, m, ea.EPSILON, ea.ROUND_TIME)
    ref_fail, *_ = ea.failure(ref, ref_dist, ea.NOISE[level], m, ea.FACTORY_ROUTING[m])
    assert chosen['expected'] < ref['total days']/(1 - ref_fail), (chosen['expected'], ref['total days'])
    assert (ea.physical_qubits(system, m, option, chosen['dist'])['total']
            <= ea.physical_qubits(system, m, s['option'], ref_dist)['total'])
    return chosen['dist']['d_rep']


def uses_dirty(option):
    '''True when any lookup of `option` borrows the system qubits as dirty ancillae'''
    return (option['method'].startswith('dirty')
            or 'dirty' in (option['outer_variety'], option['inner_variety']))


def intro_table(level):
    noise = ea.NOISE[level]
    molecules = list(st.MOLECULES)
    rows = {}           # (architecture, dirty) -> [(qubits, Toffolis, expected duration, chosen)] per molecule
    lams = set()        # the entries' angles per batch: the caption's reference configuration
    for name in molecules:
        system, settings = st.MOLECULES[name]
        for v in xo.cross_variants(system, settings):
            lams.add(v['option']['lam'])
            chosen = v['label'].endswith('chosen')
            if chosen:
                # bold claims the error accounting tables' compilation: their distances
                assert v['distances'] == settings[v['model']]['distances'][level], (name, v['label'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = xo.cross_run(system, v, noise, ea.FACTORY_ROUTING[xo.chip_model(v)], ea.EPSILON, ea.ROUND_TIME)
            key =(v['label'].split(',')[0], uses_dirty(v['option']))
            # the expected duration as in error_table: the one-shot duration (with SLACK) over 1 - p_fail
            one_shot = SLACK*r['cost']['total days']
            expected = one_shot/(1 - r['fail']) if r['fail'] < 1 else float('inf')
            rows.setdefault(key, []).append((xo.physical_qubits(system, v), r['cost']['total Toffoli'],
                                             expected, chosen))
    # each architecture once without and once with dirty ancillae
    assert sorted(rows) == [(a, d) for a in ('LDPC', 'rep') for d in (False, True)], sorted(rows)

    def bold(cell, chosen):
        if not chosen:
            return cell
        return rf'$\mathbf{{{cell[1:-1]}}}$' if cell.startswith('$') else rf'\textbf{{{cell}}}'

    out = []
    for architecture in ('rep', 'LDPC'):
        for dirty in (False, True):
            cells = [ARCHITECTURE_TEX[architecture], 'yes' if dirty else 'no ']
            for qubits, toffolis, expected, chosen in rows[(architecture, dirty)]:
                cells += [bold(num(qubits), chosen), bold(sci(toffolis), chosen),
                          bold(r'$\infty$' if expected == float('inf') else f'{expected:.2f} d', chosen)]
            out.append('    ' + ' & '.join(cells) + r' \\')
    d_rep, d_ldpc = xo.CROSS_DISTANCES[xo.rep_code], xo.CROSS_DISTANCES[xo.ldpc_code]
    assert d_ldpc['d_rep'] == d_rep['d_rep']
    assert len(lams) == 1, lams
    lam, = lams
    level_tex = {'5e-4': r'$p_Z = 5\times10^{-4}$', '1e-3': r'$p_Z = 10^{-3}$'}[level]
    caption = (rf'Physical qubits, Toffoli count, and expected duration of a single phase-estimation run at '
               rf'{level_tex} for $d = {d_rep["d_rep"]}$, respectively $d_\text{{LDPC}} = {d_ldpc["d_LDPC"]}$ and '
               rf'$d_\text{{rep}} = {d_ldpc["d_rep"]}$, with and without using the system qubits as dirty '
               r'ancillae for data loading. Bold entries are the compilations of '
               rf'\zcref{{{TABLES["rep"]["label"]},{TABLES["ldpc"]["label"]}}}. All entries use the reference '
               rf'configuration, i.e. the smallest possible compute zone ($\lambda = {lam}$). For '
               rf'{MOLECULE_TEX[FASTER_MOLECULE]}, at the same footprint, the repetition-cat architecture admits a '
               rf'faster configuration with $\lambda = {FASTER_LAM}, '
               rf'd = {faster_rep(level)}$ (see \zcref{{{FASTER_FIGURE}}}).')
    header = ['Architecture', 'Dirty ancillae'] + ['Qubits', 'Toffolis', r'$\langle t\rangle$']*len(molecules)
    return '\n'.join(
        [r'\begin{table}[t]', r'  \centering',
         r'  \begin{tabular}{ll' + 'rrr'*len(molecules) + '}', r'    \toprule',
         '    & & ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{MOLECULE_TEX[n]}}}' for n in molecules) + r' \\',
         '    ' + ''.join(rf'\cmidrule(lr){{{3 + 3*i}-{5 + 3*i}}}' for i in range(len(molecules))),
         '    ' + ' & '.join(header) + r' \\', r'    \midrule'] + out
        + [r'    \bottomrule', r'  \end{tabular}', rf'  \caption{{{caption}}}', r'  \label{tab:intro-comparison}',
           r'\end{table}'])


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # budget warnings are not the point here
        latex = '\n\n'.join([primitive_tables(MOLECULE), step_table(MOLECULE), error_table(LEVEL),
                             intro_table(LEVEL)])
    print(latex)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(latex + '\n')
