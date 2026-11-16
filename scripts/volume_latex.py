"""
LaTeX table of the full algorithm's duration and effective volume, for the
paper: per molecule and noise level, at the chosen distances, the run's
logical cycles and its effective volumes (logical qubit-cycles) from the
algorithm, the factory routing and the factories, and their sum.

    pixi run python scripts/volume_latex.py rep               # print it
    pixi run python scripts/volume_latex.py ldpc table.tex    # also write it

    rep    repetition-cat architecture
    ldpc   hybrid LDPC-cat architecture

The numbers are error_accounting.run_volumes' (as in the notebook's 'Run
volumes' cell): the algorithm's from Total_time ('total effective volume',
each layer at its own distance), the factory routing (FACTORY_ROUTING) and
the factories (FACTORIES), the model's own, as physical qubits active for the whole run,
converted like rep-layer volume (physical qubit-rounds / d_rep(2 d_rep - 1)).
No slack is added.  The molecules, options and chosen distances are
resource_count.settings'.
"""
import sys
import warnings
from pathlib import Path

# make resource_count importable however the script is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resource_count import error_accounting as ea                 # noqa: E402
from resource_count import settings as st                         # noqa: E402
# the models and the molecules' and noise levels' names, as the error table has them
from error_accounting_latex import MODEL, MOLECULE_TEX, P_Z_TEX, num   # noqa: E402

# the value columns' headers, over two rows to keep the table narrow
COLUMNS = [('Logical', 'cycles'), ('Algorithm', ''), ('Factory', 'routing'), ('Factories', ''), ('Total', '')]
TABLES = {
    'rep': dict(caption=(r"Duration and effective volume of the full algorithm for FeMoco-54 and FeMoco-76 at "
                         r"the chosen distances for different phase-flip errors. The duration is given in logical "
                         r"cycles of $d$ rounds, the volumes in logical qubit-cycles, i.e., physical qubit-rounds "
                         r"divided by $d(2d-1)$. The algorithm's volume includes idle qubits and routing."),
                factory_at='',
                qubits_caption=(r" The effective logical qubits are the total volume divided by the number of "
                                r"logical cycles, i.e., the average number of logical qubits, the factory "
                                r"routing and the factories included."),
                # the effective logical qubit columns: (header, run_volumes' key)
                qubits=[('Logical qubits', 'logical qubits')],
                label='tab: Rep code volume',
                distance_header=[r'$d$'],
                distance_cells=lambda dist: [str(dist['d_rep'])]),
    'ldpc': dict(caption=(r"Duration and effective volume of the full algorithm for FeMoco-54 and FeMoco-76 at "
                          r"the chosen distances for different phase-flip errors in the LDPC-cat architecture. "
                          r"The duration is given in logical cycles of $\max(d_\mathrm{rep}, d_\mathrm{LDPC})$ "
                          r"rounds, the volumes in logical qubit-cycles: the algorithm's adds the repetition "
                          r"layer's physical qubit-rounds divided by $d_\mathrm{rep}(2d_\mathrm{rep}-1)$ and the "
                          r"LDPC layer's logical qubit-rounds divided by $d_\mathrm{LDPC}$, and includes idle "
                          r"qubits and routing."),
                 factory_at=r', at $d_\mathrm{rep}$',
                 qubits_caption=(r" The effective logical qubits are each layer's volume divided by that "
                                 r"layer's logical cycles (the rounds divided by its distance), i.e., the average "
                                 r"number of logical qubits in each layer; the repetition layer's include the "
                                 r"factory routing and the factories."),
                 qubits=[('Rep', 'logical qubits [rep]'), ('LDPC', 'logical qubits [ldpc]'),
                         ('Total', 'logical qubits')],
                 label='tab: LDPC code volume',
                 distance_header=[r'$d_\mathrm{LDPC}$', r'$d_\mathrm{rep}$'],
                 distance_cells=lambda dist: [str(dist['d_LDPC']), str(dist['d_rep'])]),
}


def caption_text(t, m):
    '''the table's caption: its own text, then the factories' (their
    physical qubits in cost model m, from error_accounting)'''
    return (t['caption'] + f' The factory routing ({num(ea.FACTORY_ROUTING[m])} physical qubits) and the factories '
            f'({num(ea.FACTORIES[m])} physical qubits) are taken as active for the whole run and converted like '
            f'repetition-layer volume{t["factory_at"]}. The total is the sum of the three volumes.'
            + t['qubits_caption'])


def sci(x):
    '''x as $m.mm\\times 10^{e}$ (three significant figures), in math mode'''
    mantissa, exponent = f'{x:.2e}'.split('e')
    return f'${mantissa}\\times 10^{{{int(exponent)}}}$'


def volume_rows(name):
    '''per molecule and noise level: the chosen distances and run_volumes'''
    m = MODEL[name]
    rows = []
    for molecule, (system, settings) in st.MOLECULES.items():
        s = settings[m]
        for level in ea.NOISE:
            dist = s['distances'][level]
            cost = ea.total_time(system, s['budget'], s['option'], dist, m, ea.EPSILON, ea.ROUND_TIME)
            rows.append(dict(molecule=molecule, level=level, dist=dist,
                             **ea.run_volumes(cost, dist, m, ea.FACTORY_ROUTING[m], ea.FACTORIES[m])))
    return rows


def volume_table_latex(name):
    t = TABLES[name]
    rows = volume_rows(name)
    header = ['Molecule', '$p_Z$'] + t['distance_header'] + [top for top, _ in COLUMNS]
    header_2 = [''] * (2 + len(t['distance_header'])) + [bottom for _, bottom in COLUMNS]
    # the effective logical qubits: one column over two header rows, or
    # several under one heading
    if len(t['qubits']) == 1:
        top, bottom = t['qubits'][0][0].split(' ', 1)
        header += [top]
        header_2 += [bottom]
    else:
        header += [rf'\multicolumn{{{len(t["qubits"])}}}{{c}}{{Effective logical qubits}}']
        header_2 += [label for label, _ in t['qubits']]
    n_cols = len(header_2)
    out = [r'\begin{table}[hbpt]', r'    \centering',
           r'    \begin{tabular}{' + '|'.join('c'*n_cols) + r'}\hline',
           '       ' + ' & '.join(header) + r' \\',
           '       ' + ' & '.join(header_2) + r' \\ \hline']
    # one block of rows per molecule, alternately grey and white; the
    # molecule's name on its last row, spanning the block upwards
    molecules = list(st.MOLECULES)
    for row_n, r in enumerate(rows):
        block = [x for x in rows if x['molecule'] == r['molecule']]
        last = r is block[-1]
        grey = molecules.index(r['molecule']) % 2 == 0
        cells = ([rf'\multirow{{-{len(block)}}}{{*}}{{{MOLECULE_TEX[r["molecule"]]}}}' if last else '',
                  P_Z_TEX[r['level']]] + t['distance_cells'](r['dist'])
                 + [sci(r[k]) for k in ('cycles', 'algorithm', 'factory routing', 'factories', 'total')]
                 + [num(r[key]) for _, key in t['qubits']])
        line = '       ' + (r'\rowcolor{gray!10} ' if grey else '') + ' & '.join(cells) + r' \\'
        if row_n == len(rows) - 1:
            line += r' \hline'
        out.append(line)
    # caption below the table, as in the paper
    out += [r'    \end{tabular}', r'    \caption{' + caption_text(t, MODEL[name]) + '}',
            rf'    \label{{{t["label"]}}}', r'\end{table}']
    return '\n'.join(out)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in TABLES:
        sys.exit(f"usage: volume_latex.py {{{'|'.join(TABLES)}}} [out.tex]")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # budget warnings are not the point here
        latex = volume_table_latex(sys.argv[1])
    print(latex)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(latex + '\n')
