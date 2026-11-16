"""
The fastest block encoding at each compute-zone size, i.e. per number of
RPrep+Rot batches J (lam = 2 up to N - 1), for both cost models: its
option, distances, run time, failure probabilities, expected duration,
physical qubits and the share of the block encoding's rounds in RPrep+Rot,
InnerPrep and OuterPrep (each with its adjoint).  What is chosen and how:
resource_count.lambda_scan.

    pixi run python scripts/lambda_scan.py 5e-4                  # print it
    pixi run python scripts/lambda_scan.py 1e-3 scan.json        # also write it
    pixi run python scripts/lambda_scan.py figures figs          # both levels, and the figures
    pixi run python scripts/lambda_scan.py replot figs           # the figures again, from figs' scans

'figures' scans both noise levels (NOISE_LEVELS, a few minutes each),
prints them and writes into the directory: the scans (scan_<level>.json)
and the figures as PDFs,
    tradeoff_<level>.pdf  expected duration against physical qubits, one
                          point per compute-zone size, a panel per molecule,
                          with the frontier of Low et al. (arXiv:2605.30455)
                          if SHOW_LOW26 (their data is not included: see
                          LOW26_MISSING)
                          (tradeoff_figure)
    lambda_sweep.pdf      against lam, a column per molecule: the failure
                          probability at both levels, and the round shares of
                          RPrep+Rot and InnerPrep at SHARE_LEVEL
                          (lambda_figure)
    share_ratio.pdf       InnerPrep's rounds per RPrep+Rot round against
                          lam, a panel per molecule, both levels
                          (share_ratio_figure)
'replot' redraws the PDFs from the scans 'figures' wrote, without scanning.

The durations carry the paper tables' SLACK (error_accounting_latex.py), so
they compare to them.  The figures use the paper's names
(error_accounting_latex's MOLECULE_TEX, P_Z_TEX; cross_options_latex's
ARCHITECTURE_TEX), the chart palette (figure_style.py: the rep code S1
yellow, LDPC S2 teal) and Computer Modern, as its LaTeX (FIG_STYLE: draw or save
a figure inside plt.rc_context(FIG_STYLE), as write_figures does, or its
tick labels fall back to the default style).
"""
import json
import sys
from pathlib import Path

# make resource_count importable however the script is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resource_count import lambda_scan as ls                      # noqa: E402
from resource_count import settings as st                         # noqa: E402
from error_accounting_latex import SLACK, MOLECULE_TEX, P_Z_TEX   # noqa: E402
from cross_options_latex import ARCHITECTURE_TEX                  # noqa: E402
import figure_style as fs                                         # noqa: E402

ALPHA_MAX = 6           # RPrep+Rot's swap parameters K up to 2**ALPHA_MAX: every viable K (lambda_scan)
TOP = 5                 # stage-1 candidates rescanned over the distances
MODELS = {'rep': st.rep_code, 'ldpc': st.ldpc_code}
NOISE_LEVELS = ('5e-4', '1e-3')     # the levels of 'figures'
SHOW_LOW26 = False      # 'figures' / 'replot': Low+26's frontier on the trade-off figures (needs LOW26_FILE)
SHARE_LEVEL = '5e-4'    # 'figures' / 'replot': the level of lambda_sweep.pdf's round shares

# ---- the figures -------------------------------------------------------------
# per cost model: its name in the paper, its series in the chart palette
# (figure_style: the rep code S1, LDPC S2) and its marker
FIG_CODES = {'rep': (ARCHITECTURE_TEX['rep'], 0, 'o'), 'ldpc': (ARCHITECTURE_TEX['LDPC'], 1, 's')}
FIG_LEVELS = {'5e-4': True, '1e-3': False}                    # share_ratio_figure: filled markers
FIG_SUBROUTINES = ('RPrep+Rot', 'InnerPrep')   # lambda_figure's round shares, each bundle labelled directly
# lambda_figure's failure-probability key: above the curves where KEY_ROOM
# times the largest p_fail stays within P_FAIL_TOP, else in the lower left,
# with the axis capped at P_FAIL_TOP
KEY_ROOM = 5.5
P_FAIL_TOP = 0.5
LOW26_LABEL = 'Low+26'  # the legend's name for Low+26's frontier
LOW26_SERIES = 5        # figure_style's S6, mid-grey
FIG_STYLE = fs.PAPER_STYLE

# ---- the comparison with Low+26 (optional data) -------------------------------
# Low+26: G. H. Low, W. J. Huggins, D. W. Berry, T. Khattar, A. F. White,
# N. C. Rubin and R. Babbush, A Denser Planar Surface Code (2026),
# arXiv:2605.30455.  Their numbers are not part of this repository: the
# comparison (SHOW_LOW26 here, the notebook's matched-runtime ratios) reads
# them from LOW26_FILE, which you transcribe from their paper yourself.
LOW26_FILE = Path(__file__).resolve().parent / 'low26_frontier.py'
LOW26_MISSING = f'''The comparison with Low et al. (arXiv:2605.30455) needs their Table XXVII,
which is not included in this repository.  To run it, transcribe the table's
FeMoco-54o54e and FeMoco-76o113e blocks into {LOW26_FILE.name} next to this
script, as two dicts keyed by 'FeMoCo-54' and 'FeMoCo-76', each a list of rows
    (physical qubits / 10^3, spacetime volume / (10^6 qubit hours),
     expected runtime / hours, one-shot runtime / hours, p_fail):
  FRONTIER   the table's main block (the space-time Pareto frontier)
  SELECTED   the rows highlighted below it'''


def load_low26():
    '''the module LOW26_FILE (FRONTIER, SELECTED), or None if it is not there'''
    try:
        import low26_frontier
    except ModuleNotFoundError:
        return None
    return low26_frontier
# tradeoff_figure labels, besides each curve's ends and distance changes, the
# levels nearest these lam (on a log scale)
LABEL_LAMBDAS = (4, 8, 16, 32)
LABEL_SIZE = 8          # the point labels, in points
# label places beside the point, as (dx, dy) offsets in points and alignment,
# in order of preference
LABEL_PLACES = {'rep': [(3.5, 3, 'left'), (3.5, -11, 'left'), (-3.5, 3, 'right'), (-3.5, -11, 'right'),
                        (6, -4, 'left'), (-6, -4, 'right'), (0, 6, 'center'), (0, -14, 'center')],
                'ldpc': [(-3.5, -11, 'right'), (-3.5, 3, 'right'), (3.5, -11, 'left'), (3.5, 3, 'left'),
                         (-6, -4, 'right'), (6, -4, 'left'), (0, -14, 'center'), (0, 6, 'center')]}
# a distance change's label, beside the segment between the two distances
STEP_PLACES = [(6, -4, 'left'), (-6, -4, 'right'), (6, 3, 'left'), (-6, 3, 'right'),
               (5, -9, 'left'), (-5, -9, 'right')]
RING = 10               # the reference ring's marker size, in points


def ring_places(places):
    '''`places` moved out of a point's reference ring'''
    return [(dx + RING/2*((dx > 0) - (dx < 0)), dy if dx else dy + RING/2*((dy > 0) - (dy < 0)), ha)
            for dx, dy, ha in places]


# farther places, for the labels that must be placed, joined to their point by a leader line
LEADER_PLACES = [(-14, 16, 'right'), (14, 16, 'left'), (0, 22, 'center'), (-14, -24, 'right'), (14, -24, 'left'),
                 (0, -28, 'center'), (-26, 30, 'right'), (26, 30, 'left'), (-26, -38, 'right'), (26, -38, 'left'),
                 (40, 6, 'left'), (40, 40, 'left'), (-40, 40, 'right'), (0, 44, 'center'), (40, -50, 'left'),
                 (-40, -50, 'right')]
LABEL_PAD = 1.5         # the clearance kept between labels, and to the frame, in points


def option_text(o):
    lookup = 'qrom' if o['method'] == 'qrom' else f"{o['method']} K={o['K']}"
    return f"{lookup} lam={o['lam']}, preps {o['outer_variety'][0]}/{o['inner_variety'][0]}"


def show(name, mname, rows):
    print(f'\n{name}, {mname}')
    head = (f"{'lam':>4}{'J':>4}{'zone':>6}  {'option':<36}{'dist':>7}{'days':>8}{'p_fail':>8}{'eps_alg':>9}"
            f"{'eps_CXCXX':>10}{'eps_i':>8}{'<t>':>8}{'physical':>10}{'Toffoli':>10}"
            f"{'RPrep+Rot':>10}{'InnerPrep':>10}{'OuterPrep':>10}")
    print(head)
    print('-'*len(head))
    for r in rows:
        dist = '/'.join(str(v) for v in r['distances'].values())
        sh = r['shares']
        print(f"{r['lam']:>4}{r['J']:>4}{r['budget']:>6}  {option_text(r['option']):<36}{dist:>7}"
              f"{r['days']:8.3f}{r['fail']:8.4f}{r['logical']:9.5f}{r['magic']:10.5f}{r['istate']:8.5f}"
              f"{r['expected']:8.3f}{r['phys']:>10,}{r['toffoli']:10.2e}"
              f"{100*sh['RPrep+Rot']:9.1f}%{100*sh['InnerPrep']:9.1f}%{100*sh['OuterPrep']:9.1f}%")


def scan_level(level):
    '''the scans of every molecule and cost model at `level`, printed:
    {'<molecule>|<model>': rows}'''
    out = {}
    for name, (system, settings) in st.MOLECULES.items():
        for mname, model in MODELS.items():
            rows = ls.scan(system, model, settings, level, lam_max=system['N'] - 1,
                           alpha_max=ALPHA_MAX, top=TOP, slack=SLACK)
            show(name, settings[model]['name'], rows)
            out[f'{name}|{mname}'] = rows
    print("\nzone: the compute-zone budget (lam*b_rot and its access corridors); preps: outer/inner "
          "variety (c clean, d dirty); days, <t>: with {:.0f}% slack; shares: of one block encoding's "
          "rounds, each with its adjoint".format(100*(SLACK - 1)))
    return out


def distance_text(distances):
    '''the distances of a label: d_rep, or d_rep/d_LDPC'''
    return r'$d$ = ' + '/'.join(str(v) for v in distances.values())


def step_text(before, after):
    '''the label of a distance change, e.g. "d: 15 -> 13"'''
    return (r'$d$: ' + '/'.join(str(v) for v in before.values()) + r' $\rightarrow$ '
            + '/'.join(str(v) for v in after.values()))


def distance_runs_text(rows):
    '''the distances along `rows` for a key titled with d: each run of equal
    distances once, in order, e.g. "15 -> 13".  Where each change happens is
    labelled on the curve itself (step_text), so the key stays short enough
    to fit inside its panel'''
    runs = []
    for r in rows:
        if not runs or runs[-1] != r['distances']:
            runs.append(r['distances'])
    return r' $\rightarrow$ '.join('/'.join(str(v) for v in d.values()) for d in runs)


def distance_steps(rows):
    '''(row before, row after) for every change of the distances along `rows`'''
    return [(a, b) for a, b in zip(rows, rows[1:]) if a['distances'] != b['distances']]


def midpoint(ax, a, b):
    '''the middle of the segment from a to b (data units) as drawn, in data units'''
    pa, pb = ax.transData.transform(a), ax.transData.transform(b)
    return tuple(ax.transData.inverted().transform((pa + pb)/2))


def sparse_levels(rows, targets):
    '''the rows nearest each lam in `targets` on a log scale, each once'''
    import numpy as np
    out = []
    for t in targets:
        r = min(rows, key=lambda r: abs(np.log(r['lam']/t)))
        if all(r is not o for o in out):
            out.append(r)
    return out


def code_plot(ax, x, y, code, **kw):
    '''a cost model's curve on `ax`, in its palette series and marker (figure_style.plot)'''
    _, series, marker = FIG_CODES[code]
    fs.plot(ax, x, y, series, marker=marker, **kw)


def code_handle(code, **kw):
    '''a cost model's legend handle (figure_style.handle)'''
    _, series, marker = FIG_CODES[code]
    return fs.handle(series, marker=marker, **kw)


def code_ink(code):
    '''the colour of a cost model's labels (figure_style.text_color)'''
    return fs.text_color(FIG_CODES[code][1])


def line_marks(ax, xy, radius_pt):
    '''boxes, in display units, covering the points `xy` (data) and the line
    through them, sampled every ~2 points'''
    import numpy as np
    scale = ax.figure.dpi/72
    line = [ax.transData.transform(p) for p in xy]
    marks = [(x - radius_pt*scale, y - radius_pt*scale, x + radius_pt*scale, y + radius_pt*scale)
             for x, y in line]
    for (x0, y0), (x1, y1) in zip(line, line[1:]):
        n = max(1, int(np.hypot(x1 - x0, y1 - y0)/(2*scale)))
        marks += [(x0 + (x1 - x0)*j/n - scale, y0 + (y1 - y0)*j/n - scale,
                   x0 + (x1 - x0)*j/n + scale, y0 + (y1 - y0)*j/n + scale) for j in range(n + 1)]
    return marks


def legend_marks(ax):
    '''the box of `ax`'s legend in display units, as place_labels' marks
    (none without a legend)'''
    legend = ax.get_legend()
    if legend is None:
        return []
    e = legend.get_window_extent(ax.figure.canvas.get_renderer())
    return [(e.x0, e.y0, e.x1, e.y1)]


def place_labels(ax, requests, points, marks):
    '''Label points of `ax`, once its layout is fixed.  requests: [(xy,
    text, color, places, forced)] in the order they are placed, xy in data
    units, places [(dx, dy, ha)] offsets in points in order of preference.
    Each label takes the first place inside the axes, clear of `marks` and
    the labels before it (boxes in display units, as their rendered text),
    and nearer its own point than any of `points` (display units), so it
    cannot be read as another point's.  A forced label without such a place
    takes the first of LEADER_PLACES inside the axes and clear, joined to its
    point by a leader line that passes no other point or label; failing that, the
    first place inside the axes, then its first place.  Any other label
    without a place stays out.  While a forced label ends up so, it moves to
    the front of the forced labels and they are placed again (at most once
    per forced label); the order with the fewest such labels is kept.'''
    import numpy as np
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    scale = ax.figure.dpi/72
    pad = LABEL_PAD*scale

    def overlaps(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    def leader_line(own, b):
        '''from just outside the point's marker to the nearest point of the
        label's box, in display units'''
        end = np.array([np.clip(own[0], b[0], b[2]), np.clip(own[1], b[1], b[3])])
        return own + (end - own)*min(1, 4*scale/np.hypot(*(end - own))), end

    def attempt(order):
        '''place `order`'s labels: (the artists drawn, the forced labels that
        found no clear place)'''
        taken = list(marks)
        labels = []             # the boxes of the labels placed
        artists, stuck = [], []
        for request in order:
            xy, text, color, places, forced = request
            own = ax.transData.transform(xy)
            others = [(q[0] - 3*scale, q[1] - 3*scale, q[0] + 3*scale, q[1] + 3*scale)
                      for q in points if not np.allclose(q, own)]
            tried = []
            for leader, (dx, dy, ha) in [(False, p) for p in places] + [(True, p) for p in LEADER_PLACES if forced]:
                note = ax.annotate(text, xy, xytext=(dx, dy), textcoords='offset points', ha=ha, va='bottom',
                                   fontsize=LABEL_SIZE, color=color)
                e = note.get_window_extent(renderer)
                b = (e.x0, e.y0, e.x1, e.y1)
                cx, cy = (b[0] + b[2])/2, (b[1] + b[3])/2
                nearest = min(points, key=lambda q: np.hypot(q[0] - cx, q[1] - cy))
                clear = (not any(overlaps(b, o) for o in taken)
                         and not any(overlaps((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), o) for o in labels))
                if leader and clear:
                    start, end = leader_line(own, b)
                    samples = [start + (end - start)*j/20 for j in range(21)]
                    clear = not any(o[0] <= x <= o[2] and o[1] <= y <= o[3]
                                    for x, y in samples for o in others + labels)
                tried.append(dict(note=note, box=b, leader=leader, clear=clear, nearest=np.allclose(nearest, own),
                                  inside=(frame.x0 + pad <= b[0] and b[2] <= frame.x1 - pad
                                          and frame.y0 + pad <= b[1] and b[3] <= frame.y1 - pad)))
            near = [t for t in tried if not t['leader']]
            chosen = next((t for t in near if t['inside'] and t['clear'] and t['nearest']), None)
            if chosen is None and forced:
                chosen = next((t for t in tried if t['leader'] and t['inside'] and t['clear']), None)
                if chosen is None:
                    stuck.append(request)
                    chosen = next((t for t in near if t['inside']), near[0])
            for t in tried:
                if t is not chosen:
                    t['note'].remove()
            if chosen is None:
                continue
            artists.append(chosen['note'])
            taken.append(chosen['box'])
            labels.append(chosen['box'])
            if chosen['leader']:
                line = ax.transData.inverted().transform(leader_line(own, chosen['box']))
                artists += ax.plot(line[:, 0], line[:, 1], '-', color=color, lw=0.5, zorder=1.5)
                taken += line_marks(ax, [tuple(p) for p in line], 0)
        return artists, stuck

    forced = [r for r in requests if r[4]]
    rest = [r for r in requests if not r[4]]
    best = None             # (number stuck, order)
    for _ in range(len(forced) + 1):
        artists, stuck = attempt(forced + rest)
        for a in artists:
            a.remove()
        if best is None or len(stuck) < best[0]:
            best = (len(stuck), forced + rest)
        if not stuck or stuck[0] is forced[0]:
            break
        forced = [stuck[0]] + [r for r in forced if r is not stuck[0]]
    attempt(best[1])


def tradeoff_figure(scans, level, *, low):
    '''Expected duration against physical qubits at `level`, one point per
    compute-zone size, a panel per molecule, both cost models; `scans` as
    scan_level returns them.  Each curve's lam = 2 point, the reference
    configuration of the paper's tables, is ringed.  low: also Low+26's
    frontier (LOW26_FILE's FRONTIER, expected runtime), in grey -- then
    both axes are logarithmic, as it spans 3 decades of runtime; otherwise
    only the qubit axis, as the qubits grow steeply at the largest zones.

    Labels (place_labels), in each curve's colour: lam = 2 with its
    distances; each distance change on the segment across it ("d: 15 -> 13");
    the last point; and, without `low`, the levels nearest LABEL_LAMBDAS and
    the first point after each change, where they fit.  Returns the
    figure.'''
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    low26 = load_low26() if low else None
    if low and low26 is None:
        raise FileNotFoundError(LOW26_MISSING)
    mols = list(st.MOLECULES)
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(1, len(mols), figsize=(7.0, 3.7 if low else 3.3))
        for k, (ax, mol) in enumerate(zip(axes, mols)):
            curves = {code: scans[f'{mol}|{code}'] for code in FIG_CODES}
            for code, rows in curves.items():
                code_plot(ax, [r['phys'] for r in rows], [r['expected'] for r in rows], code, zorder=3)
                ax.plot(rows[0]['phys'], rows[0]['expected'], 'o', ms=RING, mfc='none', mec=fs.OUTLINE, mew=0.9,
                        zorder=4)
            phys = [r['phys'] for rows in curves.values() for r in rows]
            days = [r['expected'] for rows in curves.values() for r in rows]
            ax.set_xscale('log')
            if low:
                frontier = low26.FRONTIER[mol]
                fs.plot(ax, [1e3*q for q, *_ in frontier], [t/24 for _, _, t, _, _ in frontier], LOW26_SERIES,
                        marker='D', lw=1.0, ms=3.0)
                phys += [1e3*q for q, *_ in frontier]
                days += [t/24 for _, _, t, _, _ in frontier]
                ax.set_yscale('log')
                ax.set_xlim(0.55*min(phys), 1.25*max(phys))
                ax.set_ylim(0.6*min(days), 1.7*max(days))
                ax.set_xlabel('Physical qubits')
            else:
                # room for the labels at the curves' ends
                ax.set_xlim(0.65*min(phys), 1.12*max(phys))
                ax.set_ylim(0, 1.25*max(days))
                ax.xaxis.set_major_locator(FixedLocator(
                    [t for t in (5e3, 7.5e3, 1e4, 1.5e4, 2e4, 3e4, 5e4, 7.5e4, 1e5) if t <= 1.12*max(phys)]))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1e3:g}'))
                ax.set_xlabel(r'Physical qubits ($\times 10^3$)')
            ax.set_title(f"({'ab'[k]}) {MOLECULE_TEX[mol]}, $p_Z$ = {P_Z_TEX[level]}", loc='left')
        axes[0].set_ylabel(r'Expected duration $\langle t\rangle$ (days)')
        handles = [code_handle(code) for code in FIG_CODES]
        names = [name for name, _, _ in FIG_CODES.values()]
        if low:
            handles.append(fs.handle(LOW26_SERIES, marker='D', lw=1.0, ms=3.0))
            names.append(LOW26_LABEL)
        handles.append(Line2D([], [], ls='none', marker='o', ms=RING, mfc='none', mec=fs.OUTLINE, mew=0.9))
        names.append(r'reference, $\lambda = 2$')
        axes[0].legend(handles, names, frameon=False, loc='upper right', title=r'labels: $\lambda$',
                       title_fontsize=8, alignment='right', handlelength=2.2)
        fig.tight_layout(w_pad=1.5)

        for ax, mol in zip(axes, mols):
            curves = {code: scans[f'{mol}|{code}'] for code in FIG_CODES}
            xy = {code: [(r['phys'], r['expected']) for r in rows] for code, rows in curves.items()}
            if low:
                xy['low'] = [(1e3*q, t/24) for q, _, t, _, _ in low26.FRONTIER[mol]]
            marks = [m for code, pts in xy.items() for m in line_marks(ax, pts, 2.5)]
            marks += [m for code in curves for m in line_marks(ax, [xy[code][0]], RING/2 + 0.5)]    # the rings
            marks += legend_marks(ax)
            forced, optional, steps = [], [], []
            for code, rows in curves.items():
                at = lambda r: (r['phys'], r['expected'])
                forced.append((at(rows[0]), f"{rows[0]['lam']} ({distance_text(rows[0]['distances'])})",
                               code_ink(code), ring_places(LABEL_PLACES[code]), True))
                for a, b in distance_steps(rows):
                    mid = midpoint(ax, at(a), at(b))
                    steps.append(mid)
                    forced.append((mid, step_text(a['distances'], b['distances']), code_ink(code), STEP_PLACES, True))
                    optional.append((at(b), f"{b['lam']}", code_ink(code), LABEL_PLACES[code], False))
                forced.append((at(rows[-1]), f"{rows[-1]['lam']}", code_ink(code), LABEL_PLACES[code], True))
                labelled = [rows[0], rows[-1]] + [b for _, b in distance_steps(rows)]
                optional += [(at(r), f"{r['lam']}", code_ink(code), LABEL_PLACES[code], False)
                             for r in sparse_levels(rows, LABEL_LAMBDAS) if all(r is not o for o in labelled)]
            points = [ax.transData.transform(p) for pts in xy.values() for p in pts]
            points += [ax.transData.transform(p) for p in steps]
            place_labels(ax, forced + ([] if low else optional), points, marks)
    return fig


def lambda_figure(scans_by_level, share_level):
    '''Against lam, a column per molecule, both cost models (palette colour,
    marker); `scans_by_level` {level: scans as scan_level returns them}.
    (a, b): the failure probability at every level (filled or open markers:
    FIG_LEVELS), on a log scale, with a key of each curve's code distances
    (distance_runs_text; a column per level; placed by KEY_ROOM and
    P_FAIL_TOP) and each change of them labelled on the segment across it
    ("d: 15 -> 13").  (c, d): the share of the block encoding's rounds in
    RPrep+Rot and in InnerPrep (each with its adjoint) at `share_level`, all
    markers filled (open markers mean the other level above), each bundle
    labelled directly.  Returns the figure.'''
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

    mols = list(st.MOLECULES)
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(2, len(mols), figsize=(7.0, 6.8), sharex='col')
        for col, mol in enumerate(mols):
            top = st.MOLECULES[mol][0]['N'] - 1
            fails = []
            for code in FIG_CODES:
                for level, scans in scans_by_level.items():
                    rows = scans[f'{mol}|{code}']
                    code_plot(axes[0, col], [r['lam'] for r in rows], [r['fail'] for r in rows], code,
                              filled=FIG_LEVELS[level])
                    fails += [r['fail'] for r in rows]
                rows = scans_by_level[share_level][f'{mol}|{code}']
                for sub in FIG_SUBROUTINES:
                    code_plot(axes[1, col], [r['lam'] for r in rows], [100*r['shares'][sub] for r in rows], code,
                              lw=1.2, ms=4.0)
            for row, ax in enumerate(axes[:, col]):
                ax.set_xscale('log')
                ax.set_xlim(1.15, 1.15*top)     # room left of lam = 2 for its distance labels
                ax.xaxis.set_major_locator(FixedLocator([t for t in (2, 3, 5, 10, 20, 50, 75) if t <= top]))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
                ax.set_title(f"({'abcd'[2*row + col]}) {MOLECULE_TEX[mol]}"
                             + (f", $p_Z$ = {P_Z_TEX[share_level]}" if row == 1 else ''), loc='left')
            # log scale: the levels' failure probabilities span two decades; room for the key
            # above the curves, or, where that would run past P_FAIL_TOP, in the lower left
            ax = axes[0, col]
            ax.set_yscale('log')
            if KEY_ROOM*max(fails) <= P_FAIL_TOP:
                low, high, key_loc = 0.6*min(fails), KEY_ROOM*max(fails), 'upper center'
            else:
                low, high, key_loc = min(fails)/4, P_FAIL_TOP, 'lower left'
            ax.set_ylim(low, high)
            ax.yaxis.set_major_locator(FixedLocator(
                [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1) if low <= t <= high]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            # the key: each curve's code distances, a column per level
            ax.legend([code_handle(code, filled=FIG_LEVELS[level]) for level in scans_by_level for code in FIG_CODES],
                      [f"{P_Z_TEX[level]}: {distance_runs_text(scans[f'{mol}|{code}'])}"
                       for level, scans in scans_by_level.items() for code in FIG_CODES],
                      ncol=len(scans_by_level), loc=key_loc, frameon=False, fontsize=8,
                      title=r'$p_Z$: code distance $d$', title_fontsize=8, handlelength=1.8,
                      handletextpad=0.5, columnspacing=0.9)
            # the share bundles, labelled directly: RPrep+Rot above its bundle, InnerPrep below its
            # own, at the first levels
            ax = axes[1, col]
            first = [r for code in FIG_CODES for r in scans_by_level[share_level][f'{mol}|{code}'][:3]]
            ax.text(2, 100*max(r['shares']['RPrep+Rot'] for r in first) + 4, 'RPrep+Rot', ha='left',
                    va='bottom', fontsize=10, color=fs.INK['primary'])
            ax.text(2, 100*min(r['shares']['InnerPrep'] for r in first) - 4, 'InnerPrep', ha='left',
                    va='top', fontsize=10, color=fs.INK['primary'])
            ax.set_ylim(0, 100)
            ax.set_xlabel(r'Angles per batch $\lambda$')
        axes[0, 0].set_ylabel(r'Failure probability $p_\mathrm{fail}$')
        axes[1, 0].set_ylabel("Share of the block encoding's rounds (%)")
        axes[1, 0].legend([code_handle(code, lw=1.2, ms=4.0) for code in FIG_CODES],
                          [name for name, _, _ in FIG_CODES.values()], frameon=False, loc='lower right')
        fig.tight_layout(w_pad=1.5, h_pad=1.0)

        for col, mol in enumerate(mols):
            ax = axes[0, col]
            at = lambda r: (r['lam'], r['fail'])
            curves = [(code, scans[f'{mol}|{code}']) for code in FIG_CODES for scans in scans_by_level.values()]
            marks = [m for _, rows in curves for m in line_marks(ax, [at(r) for r in rows], 2.5)] + legend_marks(ax)
            requests, steps = [], []
            for code, rows in curves:
                for a, b in distance_steps(rows):
                    mid = midpoint(ax, at(a), at(b))
                    steps.append(mid)
                    requests.append((mid, step_text(a['distances'], b['distances']), code_ink(code),
                                     STEP_PLACES, True))
            points = [ax.transData.transform(at(r)) for _, rows in curves for r in rows]
            points += [ax.transData.transform(p) for p in steps]
            place_labels(ax, requests, points, marks)
    return fig


def share_ratio_figure(scans_by_level):
    '''InnerPrep's rounds per RPrep+Rot round (both with their adjoints, per
    block encoding) against lam, a panel per molecule, both cost models
    (palette colour, marker) and every level of `scans_by_level` ({level:
    scans as scan_level returns them}; filled or open markers: FIG_LEVELS).
    Returns the figure.'''
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    mols = list(st.MOLECULES)
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(1, len(mols), figsize=(7.0, 3.0), sharey=True)
        for k, (ax, mol) in enumerate(zip(axes, mols)):
            for code in FIG_CODES:
                for level, scans in scans_by_level.items():
                    rows = scans[f'{mol}|{code}']
                    code_plot(ax, [r['lam'] for r in rows],
                              [r['shares']['InnerPrep']/r['shares']['RPrep+Rot'] for r in rows], code,
                              lw=1.2, ms=4.0, filled=FIG_LEVELS[level])
            ax.axhline(1, color=fs.OUTLINE, lw=0.7, ls=':')
            ax.text(0.03, 1.02, 'InnerPrep = RPrep+Rot', transform=ax.get_yaxis_transform(), fontsize=8.5,
                    color=fs.INK['secondary'], va='bottom')
            top = st.MOLECULES[mol][0]['N'] - 1
            ax.set_xscale('log')
            ax.set_xlim(1.8, 1.15*top)
            ax.xaxis.set_major_locator(FixedLocator([t for t in (2, 3, 5, 10, 20, 50, 75) if t <= top]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.set_ylim(0, 1.12)
            ax.set_title(f"({'ab'[k]}) {MOLECULE_TEX[mol]}", loc='left')
            ax.set_xlabel(r'Angles per batch $\lambda$')
        axes[0].set_ylabel('InnerPrep / RPrep+Rot rounds')
        axes[0].legend([code_handle(code, lw=1.2, ms=4.0) for code in FIG_CODES],
                       [name for name, _, _ in FIG_CODES.values()], frameon=False, loc='lower right')
        axes[-1].legend([Line2D([], [], ls='none', marker='o', ms=3.6, mew=0.8, mec=fs.OUTLINE,
                                mfc=fs.OUTLINE if FIG_LEVELS[level] else 'white') for level in scans_by_level],
                        [f'$p_Z$ = {P_Z_TEX[level]}' for level in scans_by_level], frameon=False,
                        loc='lower right')
        fig.tight_layout(w_pad=1.5)
    return fig


def write_figures(scans_by_level, out, *, low):
    '''the PDFs of the module docstring into directory `out`; low: Low+26's
    frontier on the trade-off figures (tradeoff_figure)'''
    import logging
    import matplotlib.pyplot as plt
    # fontTools warns about cmr10's header timestamps while embedding it; harmless
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    figures = {}
    for level, scans in scans_by_level.items():
        figures[f'tradeoff_{level}.pdf'] = tradeoff_figure(scans, level, low=low)
    figures['lambda_sweep.pdf'] = lambda_figure(scans_by_level, SHARE_LEVEL)
    figures['share_ratio.pdf'] = share_ratio_figure(scans_by_level)
    for name, fig in figures.items():
        with plt.rc_context(FIG_STYLE):
            fig.savefig(out / name)
        plt.close(fig)
        print(f'wrote {out / name}')


if __name__ == '__main__':
    if len(sys.argv) < 2 or (sys.argv[1] in ('figures', 'replot') and len(sys.argv) < 3):
        sys.exit('usage: lambda_scan.py {figures|replot} <out dir>\n'
                 '       lambda_scan.py <level> [scan.json]')
    if sys.argv[1] in ('figures', 'replot'):
        import matplotlib
        matplotlib.use('Agg')
        out = Path(sys.argv[2])
        if sys.argv[1] == 'figures':
            out.mkdir(parents=True, exist_ok=True)
            scans_by_level = {}
            for level in NOISE_LEVELS:
                print(f'\n==== p_Z = {level}')
                scans_by_level[level] = scan_level(level)
                (out / f'scan_{level}.json').write_text(json.dumps(scans_by_level[level], indent=1,
                                                                   default=float) + '\n')
        else:
            scans_by_level = {level: json.loads((out / f'scan_{level}.json').read_text())
                              for level in NOISE_LEVELS}
        write_figures(scans_by_level, out, low=SHOW_LOW26)
    else:
        level = sys.argv[1]
        out = scan_level(level)
        if len(sys.argv) > 2:
            Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float) + '\n')
