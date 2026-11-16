"""
The figures' colours and type, shared by scripts/lambda_scan.py and the
notebook.  Data series take the chart palette's colours in this order, one
per series, never cycled:

    S1  #FAC500  yellow       primary series: dashed line, circle markers.
                              1.6:1 on white, so its lines and markers carry a
                              #010102 outline, and text never takes it
    S2  #00676A  teal dark    second series
    S3  #984F91  brand purple third series
    S4  #FB6168  brand red    fourth series (RGB 251, 97, 104)
    S5  #010102  near-black   fifth series, reference lines
    S6  #BDBDBD  mid-grey     sixth series; areas in it carry a hatch

plot(ax, x, y, i) draws series i (its line, then its markers), handle(i) is
its legend handle (a tuple, which legends draw overlaid), area(i) and
text_color(i) give its patch keywords and text colour; PROP_CYCLE is the
same order as
matplotlib's default cycle; PAPER_STYLE is the paper figures' type
(Computer Modern, as the paper's LaTeX) with that cycle.
"""
from cycler import cycler
from matplotlib import patheffects

S1, S2, S3, S4, S5, S6 = '#FAC500', '#00676A', '#984F91', '#FB6168', '#010102', '#BDBDBD'
SERIES = (S1, S2, S3, S4, S5, S6)
OUTLINE = S5            # the yellow's outline, and reference lines
HATCH = '////'          # S6 areas
# neutral chart ink: text, axes and grid (not series colours)
INK = {'primary': S5, 'secondary': '#52514e', 'muted': '#898781', 'grid': '#e1e0d9', 'axis': '#c3c2b7'}


def outline(lw):
    '''the near-black outline under a yellow line of width `lw`: 0.3 points
    on each side'''
    return [patheffects.Stroke(linewidth=lw + 0.6, foreground=OUTLINE), patheffects.Normal()]


def styles(i, *, lw=1.4, ms=4.4, marker=None, filled=True):
    '''(line, markers): ax.plot keywords for series i's line (S1 dashed,
    with its outline) and for its markers (`marker`, S1 always circles, open
    with a white face if not `filled`; S1's with a near-black edge).  Two
    artists, as an outline on the markers would thicken their edges'''
    color = SERIES[i]
    if i == 0:              # a little heavier, so the yellow shows inside its outline
        lw, ms = 1.3*lw, ms + 0.6
    line = dict(color=color, lw=lw, ls='--' if i == 0 else '-')
    if i == 0:
        line['path_effects'] = outline(lw)
    marker = 'o' if i == 0 else marker
    markers = dict(ls='none', marker=marker, ms=ms, mew=0.7 if i == 0 else 0.8,
                   mfc=color if filled else 'white', mec=OUTLINE if i == 0 else color) if marker else None
    return line, markers


def plot(ax, x, y, i, *, lw=1.4, ms=4.4, marker=None, filled=True, zorder=2):
    '''series i on `ax`: its line, then its markers on top (styles)'''
    line, markers = styles(i, lw=lw, ms=ms, marker=marker, filled=filled)
    ax.plot(x, y, zorder=zorder, **line)
    if markers:
        ax.plot(x, y, zorder=zorder + 0.1, **markers)


def handle(i, *, lw=1.4, ms=4.4, marker=None, filled=True):
    '''series i's legend handle: its line and markers (styles), overlaid'''
    from matplotlib.lines import Line2D
    line, markers = styles(i, lw=lw, ms=ms, marker=marker, filled=filled)
    return (Line2D([], [], **line),) + ((Line2D([], [], **markers),) if markers else ())


def area(i):
    '''patch keywords for series i: S1 with the near-black edge, S6 hatched
    in white (the hatch takes the edge colour), so text on it stays legible'''
    return dict(facecolor=SERIES[i], edgecolor=OUTLINE if i == 0 else ('white' if i == 5 else SERIES[i]),
                hatch=HATCH if i == 5 else None, linewidth=0.6 if i in (0, 5) else 0)


def text_color(i):
    '''the colour of text naming series i: its own, but near-black for S1'''
    return OUTLINE if i == 0 else SERIES[i]


# matplotlib's default cycle, for a plot that sets no colour: the palette's
# colours, S1 dashed with near-black marker edges.  The cycle cannot carry
# S1's line outline (rcParams refuses path effects), so the figures draw
# their series with plot(ax, x, y, i)
PROP_CYCLE = (cycler(color=SERIES) + cycler(linestyle=['--', '-', '-', '-', '-', '-'])
              + cycler(markeredgecolor=[OUTLINE, S2, S3, S4, S5, S6]))

# sized for a figure printed at full text width (about 7 in, as drawn): the
# paper's own 10 pt for labels and titles, 9 pt for ticks and legends, and
# 1 pt axes and ticks, so neither reads faint in print
PAPER_STYLE = {'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
               'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
               'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10, 'legend.fontsize': 9,
               'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 1.0,
               'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
               'xtick.major.size': 4, 'ytick.major.size': 4, 'xtick.major.width': 1.0,
               'ytick.major.width': 1.0, 'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
               'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8,
               'axes.prop_cycle': PROP_CYCLE}
