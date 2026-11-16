"""
The RPrep+Rot lookup options' trade-off figure, for the paper: Toffoli count
against logical cycles of every option optimize.search_rprep_rot finds for
MOLECULE at LEVEL, a panel per cost model.  The chosen option is outlined and
its label set in near-black, the others' in light grey.

    pixi run python scripts/rprep_rot_tradeoff.py figs     # write figs/rprep_rot_tradeoff.pdf

The notebook's 'RPrep+Rot lookup options and trade-offs' section draws the
same figure with tradeoff_figure, for the system and noise level chosen
there.  Colours, markers and type are the paper figures' (figure_style.py,
PAPER_STYLE), as figs/lambda_sweep.pdf and figs/tradeoff_*.pdf draw them.
"""
import sys
from pathlib import Path

# make resource_count importable however the script is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                     # noqa: E402

from resource_count import accounting as acc           # noqa: E402
from resource_count import optimize as opt             # noqa: E402
from resource_count import settings as st              # noqa: E402
import figure_style as fs                              # noqa: E402

MOLECULE = 'FeMoCo-54'          # the paper's figure: this molecule ...
LEVEL = '5e-4'                  # ... at this noise level's chosen distances
LAMS, ALPHA_MAX = range(1, 5), 3  # search range: lam = 1..4, K up to 2**3
MODELS = {'rep code': st.rep_code, 'LDPC': st.ldpc_code}
NAME = 'rprep_rot_tradeoff.pdf'

# chart ink (figure_style.INK) on white, and the lookup methods as the chart
# palette's series S1-S4, each with its own marker too, so colour never
# carries identity alone.  'edge' is the marker edge: S1's yellow keeps a
# near-black one
INK = dict(fs.INK, surface='#ffffff')
METHOD_STYLE = {
    'qrom':             dict(color=fs.S1, edge=fs.OUTLINE, marker='o'),
    'clean':            dict(color=fs.S2, edge=fs.S2, marker='D'),
    'dirty_restricted': dict(color=fs.S3, edge=fs.S3, marker='^'),
    'dirty_full':       dict(color=fs.S4, edge=fs.S4, marker='s'),
}
# the architectures as the other paper figures name them (cross_options_latex.ARCHITECTURE_TEX)
PANEL = {'rep code': 'Repetition-cat', 'LDPC': 'LDPC-cat'}
PAPER_LABEL = {'qrom': 'Qrom', 'clean': 'Clean SelectSwap',
               'dirty_restricted': 'Dirty SelectSwap, restricted', 'dirty_full': 'Dirty SelectSwap, full'}
# the paper figures' type, ticks and axes (Computer Modern, ticks inside on
# all four sides, no grid), as lambda_scan.py draws them.  PRINT is kept as
# the name the notebook uses
PRINT = fs.PAPER_STYLE
LABEL_SIZE = 8          # the point labels, in points, as lambda_scan.LABEL_SIZE
TEX = {'d_rep': r'd_\mathrm{rep}', 'd_LDPC': r'd_\mathrm{LDPC}'}


def codes(molecule, level):
    '''per cost model (MODELS' names): its model module, the distances chosen
    for `molecule` at `level` and its compute-zone budget, from settings'''
    system, settings = st.MOLECULES[molecule]
    return {code: dict(model=m, distances=settings[m]['distances'][level], budget=settings[m]['budget'],
                       system=system)
            for code, m in MODELS.items()}


def search(c):
    '''optimize.search_rprep_rot for one entry of codes()'''
    p = c['system']
    return opt.search_rprep_rot(p['N'], p['B'], p['R'], p['C'], p['b_rot'], c['distances'],
                                lams=LAMS, alpha_max=ALPHA_MAX,
                                ancilla_budget=c['budget'], model=c['model'])


def distance_tex(distances):
    return '$' + r',\ '.join(f'{TEX[k]}={v}' for k, v in distances.items()) + '$'


def option_text(r):
    return rf"$\lambda={r['size of batches']}$" if r['method'] == 'qrom' else \
        rf"$K={r['K']},\ \lambda={r['size of batches']}$"


def tradeoff_figure(results, codes):
    '''Toffoli count against logical cycles, both in thousands, a panel per
    cost model with shared axes; `results` {code: search_rprep_rot's result},
    `codes` {code: dict(model=..., distances=...)}.  Coinciding options share
    a label.  Drawn inside PRINT; returns the figure.'''
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    def cycle_of(code):
        return codes[code]['model'].rounds_per_cycle(**codes[code]['distances'])

    def point(r, cycle):
        '''(Toffoli count, logical cycles), both in thousands; `cycle` is the
        rounds in one logical cycle (model.rounds_per_cycle)'''
        s = acc.summary(r)
        return s['Toffoli cost']/1e3, s['Temporal cost']/cycle/1e3

    pts = [point(r, cycle_of(c)) for c, res in results.items() if res
           for r in [res['chosen']] + res['alternatives']]
    xlim = (np.floor(min(x for x, _ in pts) - 0.3), np.ceil(max(x for x, _ in pts) + 1.9))   # room for the labels
    ylim = (np.floor(min(y for _, y in pts) - 0.6), np.ceil(max(y for _, y in pts) + 0.3))

    with plt.rc_context(PRINT):
        fig, axes = plt.subplots(1, len(codes), figsize=(7.0, 3.0), sharex=True, sharey=True)
        for k, (ax, code) in enumerate(zip(axes, codes)):
            res, cycle = results[code], cycle_of(code)
            everyone = [res['chosen']] + res['alternatives']
            # options: colour and marker by method; coinciding options share a label
            labels = {}
            for r in sorted(everyone, key=lambda r: r['method'] != 'dirty_full'):
                style, (x, y) = METHOD_STYLE[r['method']], point(r, cycle)
                chosen = r is res['chosen'] or point(res['chosen'], cycle) == (x, y)
                if r['method'] == 'dirty_full':
                    ax.scatter(x, y, s=64, marker='s', facecolor='none', edgecolor=style['color'],
                               linewidth=1.2, zorder=2)
                else:
                    ax.scatter(x, y, s=36, marker=style['marker'], color=style['color'],
                               edgecolor=INK['primary'] if chosen or style['edge'] == fs.OUTLINE else 'white',
                               linewidth=0.9 if chosen else 0.5, zorder=3)
                labels.setdefault((x, y), (option_text(r), chosen))
            for (x, y), (text, chosen) in labels.items():
                # the chosen option's label in near-black, the others in light grey
                # (not bold, as Computer Modern math has no bold lambda)
                ax.annotate(text, (x, y),
                            xytext=(6, 3), textcoords='offset points', fontsize=LABEL_SIZE,
                            color=INK['primary'] if chosen else INK['muted'])
            ax.set_title(f"({'ab'[k]}) {PANEL[code]}, {distance_tex(codes[code]['distances'])}", loc='left')
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.xaxis.set_major_locator(MultipleLocator(1)); ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax.set_xlabel(r'Toffoli count ($\times 10^3$)')
        axes[0].set_ylabel(r'Logical cycles ($\times 10^3$)')

        present = [m for m in METHOD_STYLE if any(r['method'] == m for res in results.values() if res
                                                  for r in [res['chosen']] + res['alternatives'])]
        handles = [Line2D([], [], linestyle='', marker=METHOD_STYLE[m]['marker'],
                          markersize=8 if m == 'dirty_full' else 6,
                          markerfacecolor='none' if m == 'dirty_full' else METHOD_STYLE[m]['color'],
                          markeredgecolor=METHOD_STYLE[m]['edge'],
                          markeredgewidth=1.0 if m == 'dirty_full' else
                          (0.5 if METHOD_STYLE[m]['edge'] != METHOD_STYLE[m]['color'] else 0),
                          label=PAPER_LABEL[m])
                   for m in present]
        fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, 1.06), handletextpad=0.3, columnspacing=1.2)
        fig.tight_layout(w_pad=1.5)
    return fig


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: rprep_rot_tradeoff.py <output directory>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    chosen = codes(MOLECULE, LEVEL)
    fig = tradeoff_figure({code: search(c) for code, c in chosen.items()}, chosen)
    with plt.rc_context(PRINT):
        fig.savefig(out / NAME, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out / NAME}')
