"""
LaTeX tables of the error accounting, for the paper: per molecule and noise
level, the chosen distances, the three failure probabilities, the physical
qubit count, and the one-shot and expected durations, with either cost
model.

    pixi run python scripts/error_accounting_latex.py rep               # print it
    pixi run python scripts/error_accounting_latex.py ldpc table.tex    # also write it

    rep    repetition-cat architecture
    ldpc   hybrid LDPC-cat architecture

Everything comes from resource_count.error_accounting, as in the notebook:
the noise levels (NOISE), the distance choice (choose_distance over
DISTANCE_SCAN: the shortest expected duration, the smaller distances within
TIE), the failure probabilities (failure: eps_alg = eps_rep + eps_LDPC, the
rep code's volume including the factory routing, + magic-state + i-state,
all added) and the physical qubits
(physical_qubits).  The durations carry SLACK on top (initialization,
measurements, buffer staging); the expected duration is the one-shot
duration / (1 - p_fail).  The distance scan behind each choice is printed to
stderr.

The molecules, their compute-zone budgets and options are
resource_count.settings', as in the notebook.  The script stops if its
distance choice differs from the distances chosen there (settings'
'distances'), so this table, the block-encoding table and the notebook
always use the same distances.
"""
import math
import sys
import warnings
from pathlib import Path

# make resource_count importable however the script is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resource_count import cost_functions as rep_code             # noqa: E402
from resource_count import cost_functions_ldpc as ldpc_code       # noqa: E402
from resource_count import error_accounting as ea                 # noqa: E402
from resource_count import settings as st                         # noqa: E402

SLACK = 1.05            # 5% on the durations: initialization, measurements, buffer staging

MODEL = {'rep': rep_code, 'ldpc': ldpc_code}
MOLECULE_TEX = {'FeMoCo-54': 'FeMoco-54', 'FeMoCo-76': 'FeMoco-76'}    # the molecules as the captions name them
# the molecules in the error tables' first column, under the header MOLECULE_HEADER
MOLECULE_HEADER, MOLECULE_SHORT = 'FeMoco', {'FeMoCo-54': '54', 'FeMoCo-76': '76'}
P_Z_TEX = {'5e-4': r'$5\times 10^{-4}$', '1e-3': r'$10^{-3}$'}     # the noise levels as the table shows them

# ---- the table text --------------------------------------------------------
CAPTION_COMMON = (r'The physical qubit count includes ancilla and contributions from factories and factory '
                  r'routing. The duration accounts for ${slack}\%$ additional slack for initialization, '
                  r'measurements and buffer staging. The expected duration is computed by normalizing with '
                  r'$1-p_\mathrm{{fail}}$, where $p_\mathrm{{fail}} = \varepsilon_\mathrm{{alg}} + '
                  r'\varepsilon_{{\ket{{\CXCXX}}}} + \varepsilon_{{\keti}}$.')
TABLES = {
    'rep': dict(caption=r'Runtime and total physical qubit count for FeMoco-54 and FeMoco-76 for different '
                        r'phase-flip errors in the repetition-cat architecture. ' + CAPTION_COMMON,
                label='tab: Rep code error accounting',
                distance_cells=lambda dist: [str(dist['d_rep'])],
                header_rows=[[MOLECULE_HEADER, r'$p_Z$', r'$d$', r'$\varepsilon_\mathrm{alg}$',
                              r'$\varepsilon_{\ket{\CXCXX}}$', r'$\varepsilon_{\keti}$',
                              r'Physical', r'One-shot', r'$\langle t\rangle$'],
                             ['', '', '', '', '', '', 'qubits', '(days)', '(days)']]),
    'ldpc': dict(caption=r'Runtime and total physical qubit count for FeMoco-54 and FeMoco-76 for different '
                         r'phase-flip errors in the LDPC-cat architecture. ' + CAPTION_COMMON +
                         r' The algorithmic error contribution is obtained additively from the contribution of '
                         r'the repetition layer and the LDPC layer.',
                 label='tab: LDPC code error accounting',
                 distance_cells=lambda dist: [str(dist['d_LDPC']), str(dist['d_rep'])],
                 header_rows=[[MOLECULE_HEADER, r'$p_Z$', r'$d_\mathrm{LDPC}$', r'$d_\mathrm{rep}$',
                               r'$\varepsilon_\mathrm{alg}$', r'$\varepsilon_{\ket{\CXCXX}}$',
                               r'$\varepsilon_{\keti}$', r'Physical', r'One-shot', r'$\langle t\rangle$'],
                              ['', '', '', '', '', '', '', 'qubits', '(days)', '(days)']]),
}


def sig2(x):
    '''x to two significant figures, trailing zeros kept: 0.201 -> 0.20,
    0.000584 -> 0.00058, 0.0996 -> 0.10'''
    x = float(f'{x:.1e}')                      # round first, so 0.0996 becomes 0.1
    decimals = max(0, 1 - math.floor(math.log10(abs(x))))
    return f'{x:.{decimals}f}'


def num(x):
    '''an integer with thin spaces between thousands, in math mode'''
    return '$' + f'{int(round(x)):,}'.replace(',', r'\,') + '$'


def days(x):
    '''a duration in days, for a column headed (days)'''
    return r'$\infty$' if x == math.inf else f'{x:.2f}'


def error_rows(name):
    '''per molecule and noise level: the chosen distances, the failures, the
    physical qubits and the durations (with SLACK)'''
    m = MODEL[name]
    rows = []
    for molecule, (system, settings) in st.MOLECULES.items():
        s = settings[m]
        cache = {}

        def run(dist, system=system, s=s, cache=cache):
            key = tuple(sorted(dist.items()))
            if key not in cache:
                cache[key] = ea.total_time(system, s['budget'], s['option'], dist, m, ea.EPSILON, ea.ROUND_TIME)
            return cache[key]

        for level, noise in ea.NOISE.items():
            scan, c = ea.choose_distance(run, ea.DISTANCE_SCAN[m], noise, m, ea.FACTORY_ROUTING[m], ea.TIE)
            print(f'{molecule}, p_Z = {level}: chosen {c["dist"]}, expected {c["expected"]:.4f} days '
                  f'(before slack); scan: ' + ', '.join(
                      f'{tuple(r["dist"].values())} ' + ('refused' if 'infeasible' in r else f'{r["expected"]:.3f}')
                      for r in scan), file=sys.stderr)
            if c['dist'] != s['distances'][level]:
                sys.exit(f"{molecule}, {s['name']}, p_Z = {level}: the distance choice is {c['dist']}, but "
                         f"resource_count.settings has {s['distances'][level]} -- update the settings")
            phys = ea.physical_qubits(system, m, s['option'], c['dist'])['total']
            one_shot = SLACK*c['days']
            expected = one_shot/(1 - c['fail']) if c['fail'] < 1 else math.inf
            rows.append(dict(molecule=molecule, level=level, dist=c['dist'], logical=c['logical'],
                             magic=c['magic'], istate=c['istate'], phys=phys, one_shot=one_shot, expected=expected))
    return rows


def error_table_latex(name):
    t = TABLES[name]
    rows = error_rows(name)
    cols = len(t['header_rows'][0])
    out = [r'\begin{table}[hbpt]', r'    \centering',
           r'    \begin{tabular}{' + '|'.join('c'*cols) + r'}\hline']
    for header in t['header_rows']:
        out.append('       ' + ' & '.join(header) + r' \\')
    out[-1] += r' \hline'
    # one block of rows per molecule, alternately grey and white; the
    # molecule's name on its last row, spanning the block upwards
    molecules = list(st.MOLECULES)
    for row_n, r in enumerate(rows):
        block = [x for x in rows if x['molecule'] == r['molecule']]
        last = r is block[-1]
        grey = molecules.index(r['molecule']) % 2 == 0
        cells = ([rf'\multirow{{-{len(block)}}}{{*}}{{{MOLECULE_SHORT[r["molecule"]]}}}' if last else '',
                  P_Z_TEX[r['level']]]
                 + t['distance_cells'](r['dist'])
                 + [sig2(r['logical']), sig2(r['magic']), sig2(r['istate']), num(r['phys']),
                    days(r['one_shot']), days(r['expected'])])
        line = '       ' + (r'\rowcolor{gray!10} ' if grey else '') + ' & '.join(cells) + r' \\'
        if row_n == len(rows) - 1:
            line += r' \hline'
        out.append(line)
    # caption below the table, as in the paper
    out += [r'    \end{tabular}',
            r'    \caption{' + t['caption'].format(slack=round(100*(SLACK - 1))) + '}',
            rf'    \label{{{t["label"]}}}', r'\end{table}']
    return '\n'.join(out)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in TABLES:
        sys.exit(f"usage: error_accounting_latex.py {{{'|'.join(TABLES)}}} [out.tex]")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # budget warnings are not the point here
        latex = error_table_latex(sys.argv[1])
    print(latex)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(latex + '\n')
