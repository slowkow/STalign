'''Make the figures of docs/how_it_works.md ("How STalign works").

Run from anywhere; the script finds the repository from its own location::

    python docs/figures/make_figures.py                  # all figures, into docs/figures/
    python docs/figures/make_figures.py --only overview maps
    python docs/figures/make_figures.py --atlas path/to/aba_nissl.nrrd

What it uses

* The MERFISH tutorial data in docs/merfish_data: slice 2 replicate 3 (S2R3, the source)
  aligned to slice 2 replicate 2 (S2R2, the target), and slice 1 replicate 1 (S1R1) for the
  3D figure.  The cells are rasterized at 30 um pixels with blur=1.5 (as in the MERFISH
  tutorial) and aligned with LDDMM's default settings (affine pre-fit, 2000 iterations).
  Unlike the tutorial, the main fit gets no initial rotation: the pre-fit finds it.  The
  partial-tissue example (figure "mixture") starts from the tutorial's 45 degree rotation.
* STalign from this checkout: its C++ engine if built (``python setup.py build_ext
  --inplace``), else the torch engine.  Both give the same figures up to rounding.
* For the energy history and the intermediate states of a fit, the script runs the steps of
  STalign.LDDMM itself (``_lddmm_common.prepare_problem``, ``resolve_defaults`` and an
  engine) and takes snapshots of the engine; the final state is what ``STalign.LDDMM``
  returns.
* The 3D figure is a schematic with an illustrative oblique plane, not a fitted one.  If the
  Allen Nissl volume is available (``--atlas``, or ``docs/notebooks/aba_nissl.nrrd`` where
  the 3D tutorials download it), the atlas sampled on that plane is shown too.

Everything is deterministic (fixed subsamples, no random starts).  The run takes about
1.5 minutes with the C++ engine on 8 threads (STALIGN_NUM_THREADS sets the number).  PNGs
are saved with a 256-color palette to keep them small.

Requirements: numpy, pandas, matplotlib (with pillow), torch, and pynrrd for --atlas.
'''
import argparse
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('STALIGN_NUM_THREADS', '8')
os.environ.setdefault('OMP_NUM_THREADS', '8')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'docs' / 'merfish_data'
sys.path.insert(0, str(ROOT))
import STalign  # noqa: E402
from STalign import _lddmm_common as LC  # noqa: E402

torch.set_num_threads(int(os.environ['STALIGN_NUM_THREADS']))
OUT = HERE

# ----------------------------------------------------------------------------- style
# Okabe-Ito colors: the source is always blue, the target always vermillion.
SRC = '#0072B2'
TGT = '#D55E00'
ACCENT = '#009E73'   # bluish green: highlights (zoom box, removed tissue, exact inverse)
GREY = '#555555'     # deformation grids, arrows
LIGHT = '#BBBBBB'
DPI = 150

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.titlelocation': 'left', 'axes.titleweight': 'normal',
    'contour.negative_linestyle': 'solid',
})


def ramp(color, name):
    '''Colormap from white to `color` (images of one data set).'''
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF', color])


CM_SRC = ramp(SRC, 'src')
CM_TGT = ramp(TGT, 'tgt')
# residual fAI - J: blue where the deformed source has more, vermillion where the target has more
CM_RES = LinearSegmentedColormap.from_list('res', [TGT, '#FFFFFF', SRC])
CM_W = 'viridis'


def rgb(color):
    return np.array(matplotlib.colors.to_rgb(color))


def composite(a, b, ca=SRC, cb=TGT, strength=0.6):
    '''Two images on one grid as subtractive colors on white: a in blue, b in vermillion,
    overlap dark grey.  Each image is scaled by its 99.5th percentile.'''
    def unit(x):
        x = np.asarray(x, float)
        return np.clip(x / max(np.percentile(x, 99.5), 1e-12), 0, 1) * strength
    a, b = unit(a), unit(b)
    out = 1.0 - a[..., None] * (1 - rgb(ca)) - b[..., None] * (1 - rgb(cb))
    return np.clip(out, 0, 1)


def extent(X, Y):
    '''imshow extent for pixel centers X (columns) and Y (rows), shown with y up.'''
    dx, dy = X[1] - X[0], Y[1] - Y[0]
    return (X[0] - dx / 2, X[-1] + dx / 2, Y[0] - dy / 2, Y[-1] + dy / 2)


def show(ax, img, X, Y, **kw):
    kw.setdefault('interpolation', 'nearest')
    return ax.imshow(img, origin='lower', extent=extent(X, Y), **kw)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect('equal')


def scalebar(ax, length=1000.0, text='1 mm', loc=(0.06, 0.06), color='k'):
    '''A scale bar of `length` um in an image axes (loc: its left end, in axes fractions).'''
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0)
    y = y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + length], [y, y], color=color, lw=2, solid_capstyle='butt')
    ax.text(x + length / 2, y + 0.025 * (y1 - y0), text, ha='center', va='bottom', fontsize=7.5, color=color)


def label(ax, letter, title=None):
    '''Panel letter (bold) and title.'''
    ax.set_title((r'$\bf{' + letter + r'}$  ' + title) if title else r'$\bf{' + letter + '}$')


def note(ax, text, y=-0.02):
    '''Small text under an image axes.'''
    ax.text(0.0, y, text, transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color='#333333')


def subsample(n, k, seed=0):
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def scatter(ax, x, y, color, n=25000, s=0.6, alpha=0.35, seed=0, **kw):
    '''A fixed random subsample of the cells as small points.'''
    idx = subsample(len(x), n, seed)
    ax.scatter(np.asarray(x)[idx], np.asarray(y)[idx], s=s, c=color, alpha=alpha, lw=0, rasterized=True, **kw)


def legend_cells(ax, labels_colors, **kw):
    handles = [plt.Line2D([], [], ls='', marker='o', ms=5, color=c, label=t) for t, c in labels_colors]
    kw = dict(dict(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False,
                   handletextpad=0.1, columnspacing=1.0), **kw)
    ax.legend(handles=handles, **kw)


def save(fig, name):
    '''Save as a palette PNG (256 colors), which is several times smaller than RGB.'''
    path = OUT / f'{name}.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    with Image.open(path) as im:
        im = im.convert('RGB')
        pal = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        q = im.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG)
    # the most common color is the background: make it pure white again
    colors = np.array(q.getpalette(), dtype=np.uint8).reshape(-1, 3)
    colors[np.bincount(np.asarray(q).ravel(), minlength=len(colors)).argmax()] = 255
    q.putpalette(colors.ravel().tolist())
    q.save(path, optimize=True)
    print(f'  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} '
          f'({path.stat().st_size / 1024:.0f} kB)')


# ----------------------------------------------------------------------------- data
def read_cells(name):
    df = pd.read_csv(DATA / name, usecols=['center_x', 'center_y'])
    return df['center_x'].to_numpy(float), df['center_y'].to_numpy(float)


FILES = {
    'S2R3': 'datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz',
    'S2R2': 'datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz',
    'S1R1': 's1r1_metadata.csv.gz',
}
DX = 30.0     # raster pixel size (um), as in the MERFISH tutorial
BLUR = 1.5    # raster blur (pixels): kernel standard deviation 2 * 1.5 * 30 = 90 um


class Data:
    '''Cells, rasters and the main fit, computed once and shared by the figures.'''

    def __init__(self):
        self.xI, self.yI = read_cells(FILES['S2R3'])
        self.xJ, self.yJ = read_cells(FILES['S2R2'])
        self.XI, self.YI, self.I = STalign.rasterize(self.xI, self.yI, dx=DX, blur=BLUR, draw=0)
        self.XJ, self.YJ, self.J = STalign.rasterize(self.xJ, self.yJ, dx=DX, blur=BLUR, draw=0)
        self.gI, self.gJ = [self.YI, self.XI], [self.YJ, self.XJ]   # (row, col) = (y, x) axes
        self._fit = None

    @property
    def fit(self):
        if self._fit is None:
            t = time.perf_counter()
            self._fit = fit_with_history(self.gI, self.I, self.gJ, self.J, snapshots=(1, 20, 200))
            s = self._fit['settings']
            print(f'  main fit ({time.perf_counter() - t:.1f} s): niter {s["niter"]}, sigmaR {s["sigmaR"]:.4g}, '
                  f'epL {s["epL"]:.3g}, epT {s["epT"]:.3g}, epV {s["epV"]:.4g}; pre-fit start '
                  f'{self._fit["prefit"]["starts"][self._fit["prefit"]["best"]]:g} deg')
        return self._fit

    def cells_I(self):
        return np.stack([self.yI, self.xI], 1)   # (row, col) order, as LDDMM's xI

    def tutorial_guess(self):
        '''The MERFISH tutorial's initial affine (45 degree rotation about the source centroid,
        moved to the target centroid), as L, T in (row, col) order.'''
        th = np.radians(-45.0)
        L = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        cI = np.array([self.xI.mean(), self.yI.mean()])
        cJ = np.array([self.xJ.mean(), self.yJ.mean()])
        T = cJ - L @ cI
        return L[::-1, ::-1].copy(), T[::-1].copy()


# ----------------------------------------------------------------------------- fitting
_RESOLVE = ('prefit', 'sigma_disp', 'legacy_defaults', 'num_threads')


def fit_with_history(xI, I, xJ, J, snapshots=(), **kw):
    '''STalign.LDDMM(xI, I, xJ, J, **kw), keeping the energy history and the state after the
    iterations in `snapshots`.

    Runs the same steps as LDDMM (prepare_problem, resolve_defaults, an engine), so the
    final 'A', 'v' and weights are what LDDMM returns.  Returns a dict with 'A', 'v', 'xv',
    'WM', 'Esave' (E, EM, ER per iteration), 'prefit', 'settings', 'niter', 'K' (the
    smoothing kernel's Fourier symbol) and 'snaps' {n: (A, v, engine snapshot)}: A and v
    after n iterations (n = 0: after the pre-fit; no engine snapshot).
    '''
    params = inspect.signature(STalign.LDDMM).parameters
    prep = inspect.signature(LC.prepare_problem).parameters
    args = {k: p.default for k, p in params.items() if k in prep and p.default is not inspect.Parameter.empty}
    res = {k: params[k].default for k in _RESOLVE}
    backend = kw.pop('backend', params['backend'].default)
    for k, val in kw.items():
        if k not in res and k not in args:
            raise TypeError(f'unexpected argument {k}')
        (res if k in res else args)[k] = val
    problem = LC.prepare_problem(2, xI, I, xJ, J, **args)
    info = LC.resolve_defaults(problem, **res)
    engine = LC.select_engine(problem, backend, res['num_threads'])
    niter = problem.niter
    snaps = {0: (torch.as_tensor(info['prefit']['A']), torch.zeros(problem.v.shape, dtype=torch.float64), None)}
    done = 0
    for n in sorted(set(s for s in snapshots if 0 < s < niter) | {niter}):
        engine.run(n - done)
        done = n
        sn = engine.snapshot()
        A = torch.as_tensor(LC.affine(np.asarray(sn['L']), np.asarray(sn['T'])))
        snaps[n] = (A, torch.as_tensor(np.asarray(sn['v'])), sn)
    last = snaps[niter][2]
    return dict(A=torch.as_tensor(np.asarray(last['A_eval'])), v=torch.as_tensor(np.asarray(last['v_eval'])),
                xv=[torch.as_tensor(x) for x in problem.xv], WM=np.asarray(last['WM']),
                Esave=np.asarray(last['Esave']), prefit=info['prefit'], settings=info['settings'],
                niter=niter, snaps=snaps, K=problem.K)


def lddmm(xI, I, xJ, J, **kw):
    '''STalign.LDDMM without figures.'''
    return STalign.LDDMM(xI, I, xJ, J, draw=False, **kw)


def deformed_source(d, A, v, xv, gJ=None):
    '''The source image on the target grid under (A, v): I(phi^-1(A^-1 y)).'''
    out = STalign.transform_image_source_to_target(xv, v, A, d.gI, d.I, d.gJ if gJ is None else gJ)
    return out.numpy()[0]


def to_target(d, out, pts=None, **kw):
    '''Source cells (row, col) mapped to the target (exact inverse by default); returns x, y.'''
    pts = d.cells_I() if pts is None else pts
    q = STalign.transform_points_source_to_target(out['xv'], out['v'], out['A'], pts, **kw).numpy()
    return q[:, 1], q[:, 0]


def nonlinear_displacement(d, out):
    '''|A phi(x) - A x| over the source cells: how far the deformation moves them.'''
    x, y = to_target(d, out)
    A = out['A'].numpy()
    aff = d.cells_I() @ A[:2, :2].T + A[:2, 2]
    return np.hypot(x - aff[:, 1], y - aff[:, 0])


def grid_contours(ax, out, gJ, spacing=1000.0, **kw):
    '''The source grid lines (every `spacing` um) as they land in the target: contours of the
    target-to-source map.'''
    phii = STalign.build_transform(out['xv'], out['v'], out['A'], XJ=gJ, direction='b').numpy()
    kw = dict(dict(colors=GREY, linewidths=0.6), **kw)
    for c in range(2):
        lo, hi = np.floor(phii[..., c].min() / spacing), np.ceil(phii[..., c].max() / spacing)
        ax.contour(gJ[1], gJ[0], phii[..., c], levels=np.arange(lo, hi + 1) * spacing, **kw)


def tissue_box(x, y, margin=300.0):
    return (x.min() - margin, x.max() + margin, y.min() - margin, y.max() + margin)


def set_box(ax, box):
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])


def arrow_between(fig, ax0, ax1, y=0.5, text=None, yfig=None):
    '''A horizontal arrow from the right edge of ax0 to the left edge of ax1 (figure coords).'''
    b0, b1 = ax0.get_position(), ax1.get_position()
    x0, x1 = b0.x1 + 0.006, b1.x0 - 0.006
    yy = yfig if yfig is not None else b0.y0 + y * b0.height
    fig.patches.append(FancyArrowPatch((x0, yy), (x1, yy), transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=14, color=GREY, lw=1.2))
    if text:
        fig.text((x0 + x1) / 2, yy + 0.02, text, ha='center', va='bottom', fontsize=8, color=GREY)


# ============================================================================= figures
def fig_overview(d):
    '''Pipeline: cells -> rasters -> fitted map -> cells mapped onto the target.'''
    f = d.fit
    fig = plt.figure(figsize=(11, 3.7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 0.7, 1.2, 1.2], wspace=0.3, hspace=0.12)
    a = fig.add_subplot(gs[:, 0])
    b1 = fig.add_subplot(gs[0, 1])
    b2 = fig.add_subplot(gs[1, 1])
    c = fig.add_subplot(gs[:, 2])
    e = fig.add_subplot(gs[:, 3])
    box = (300, 9900, 0, 9800)

    scatter(a, d.xI, d.yI, SRC)
    scatter(a, d.xJ, d.yJ, TGT, seed=1)
    clean(a)
    set_box(a, box)
    label(a, 'a', 'Two sets of cells')
    legend_cells(a, [('source (S2R3)', SRC), ('target (S2R2)', TGT)])
    scalebar(a, loc=(0.04, 0.03))

    show(b1, d.I[0], d.XI, d.YI, cmap=CM_SRC)
    show(b2, d.J[0], d.XJ, d.YJ, cmap=CM_TGT)
    for ax, letter in ((b1, 'I'), (b2, 'J')):
        clean(ax)
        set_box(ax, box)
        ax.text(0.0, 0.95, letter, transform=ax.transAxes, fontsize=11, style='italic', va='top',
                color=SRC if letter == 'I' else TGT)
    label(b1, 'b', 'Rasterize')

    AI = deformed_source(d, f['A'], f['v'], f['xv'])
    show(c, composite(AI, d.J[0]), d.XJ, d.YJ)
    grid_contours(c, f, d.gJ)
    clean(c)
    set_box(c, box)
    label(c, 'c', r'Fit $A$ and $\varphi$, deform $I$')
    legend_cells(c, [('deformed source', SRC), ('target', TGT), ('both', '#5A5545')])

    x, y = to_target(d, f)
    scatter(e, d.xJ, d.yJ, TGT, seed=1)
    scatter(e, x, y, SRC, seed=2)
    clean(e)
    set_box(e, box)
    label(e, 'd', 'Move the source cells')
    legend_cells(e, [('aligned source', SRC), ('target', TGT)])

    fig.canvas.draw()
    mid = a.get_position().y0 + 0.5 * a.get_position().height
    arrow_between(fig, a, b1, yfig=mid)
    arrow_between(fig, b1, c, yfig=mid)
    arrow_between(fig, c, e, yfig=mid)
    save(fig, 'overview')


def fig_rasterization(d):
    '''Points -> density image; effect of the pixel size dx and of the kernel width.'''
    x, y = d.xI, d.yI
    zoom = (4600.0, 6100.0, 4200.0, 5700.0)   # 1.5 x 1.5 mm around the lateral ventricles
    fig = plt.figure(figsize=(10, 5.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.1, hspace=0.42)
    a = fig.add_subplot(gs[0, 0])
    scatter(a, x, y, SRC, n=40000)
    a.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False, ec=ACCENT, lw=1.5))
    clean(a)
    set_box(a, tissue_box(x, y))
    scalebar(a, 2000, '2 mm', loc=(0.04, 0.0))
    label(a, 'a', 'Source cells')

    b = fig.add_subplot(gs[1, 0])
    inz = (x > zoom[0]) & (x < zoom[1]) & (y > zoom[2]) & (y < zoom[3])
    b.scatter(x[inz], y[inz], s=1.5, c=SRC, lw=0)
    clean(b)
    set_box(b, zoom)
    for spine in b.spines.values():
        spine.set_visible(True)
        spine.set_color(ACCENT)
    label(b, 'b', f'Zoom: {inz.sum():,} cells')

    def density(dx, blur_um):
        X, Y, M = STalign.rasterize(x, y, dx=dx, blur_um=blur_um, draw=0)
        return X, Y, M[0] / dx ** 2 * 1e6   # cells per mm^2

    rows = [('c', 'Pixel size dx (kernel s.d. 45 µm)', [(15, 45), (30, 45), (60, 45)], 'dx = {dx} µm'),
            ('d', 'Kernel s.d. = 2 · blur · dx (dx = 30 µm)', [(30, 15), (30, 45), (30, 120)],
             's.d. {s} µm (blur = {blur:g})')]
    for r, (letter, header, cases, fmt) in enumerate(rows):
        imgs = [density(dx, s) for dx, s in cases]
        vmax = max(np.percentile(M[np.ix_((Y > zoom[2]) & (Y < zoom[3]), (X > zoom[0]) & (X < zoom[1]))], 99.5)
                   for X, Y, M in imgs)
        for k, ((dx, s), (X, Y, M)) in enumerate(zip(cases, imgs)):
            ax = fig.add_subplot(gs[r, k + 1])
            im = show(ax, M, X, Y, cmap=CM_SRC, vmin=0, vmax=vmax)
            clean(ax)
            set_box(ax, zoom)
            ax.set_title(fmt.format(dx=dx, s=s, blur=s / (2 * dx)), fontsize=8.5)
            if k == 0:
                ax.text(0.0, 1.17, r'$\bf{' + letter + r'}$  ' + header, transform=ax.transAxes, fontsize=9.5,
                        va='bottom')
        if r == 1:
            scalebar(ax, 250, '250 µm', loc=(0.06, 0.05))
        cax = fig.add_subplot(gs[r, 4])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('cells / mm²')
        cb.outline.set_visible(False)
    save(fig, 'rasterization')


def _flow_partial(xv, v, k, pts):
    '''Points after the first k of the nt forward steps (each the exact inverse of a backward
    step), without the affine.  v[:k] scaled by k/nt keeps each step v_t/nt.'''
    nt = v.shape[0]
    if k == 0:
        return pts
    return STalign.transform_points_source_to_target(xv, v[:k] * (k / nt), torch.eye(3, dtype=torch.float64),
                                                     pts).numpy()


def _grid_lines(box, spacing, n=200):
    '''Polylines (row, col) of a square grid over box = (x0, x1, y0, y1).'''
    xs = np.arange(np.ceil(box[0] / spacing), np.floor(box[1] / spacing) + 1) * spacing
    ys = np.arange(np.ceil(box[2] / spacing), np.floor(box[3] / spacing) + 1) * spacing
    lines = [np.stack([np.linspace(box[2], box[3], n), np.full(n, xx)], 1) for xx in xs]
    lines += [np.stack([np.full(n, yy), np.linspace(box[0], box[1], n)], 1) for yy in ys]
    return lines


def _draw_lines(ax, pts, sizes, **kw):
    kw = dict(dict(color=GREY, lw=0.7), **kw)
    for ln in np.split(pts, np.cumsum(sizes)[:-1]):
        ax.plot(ln[:, 1], ln[:, 0], **kw)


ARROW_GAIN = 3.0   # velocity arrows are drawn this many times longer than the displacement


def fig_transformation(d):
    '''x -> A phi(x): the flow of the source grid along v_0, v_1, v_2, then the affine.'''
    f = d.fit
    xv, v, A = f['xv'], f['v'], f['A']
    nt = v.shape[0]
    box = tissue_box(d.xI, d.yI, 300)
    lines = _grid_lines(box, 750.0)
    allpts = np.concatenate(lines)
    sizes = [len(ln) for ln in lines]

    # velocity arrows: every other velocity grid node (every a = 500 um) on source tissue
    Vy, Vx = np.meshgrid(xv[0].numpy(), xv[1].numpy(), indexing='ij')
    tissue = STalign.interp([torch.as_tensor(d.YI), torch.as_tensor(d.XI)], torch.as_tensor(d.I),
                            torch.as_tensor(np.stack([Vy, Vx])), padding_mode='zeros').numpy()[0]
    sel = np.zeros(tissue.shape, bool)
    sel[::2, ::2] = True
    sel &= tissue > 0.1 * float(d.I.max())

    fig, axs = plt.subplots(1, 5, figsize=(12.5, 2.9), gridspec_kw=dict(wspace=0.05))
    titles = ['grid $x$ (source space)'] + [rf'step {t + 1}: velocity $v_{t}$' for t in range(nt)]
    pos = [_flow_partial(xv, v, k, allpts) for k in range(nt + 1)]
    for k in range(nt + 1):
        ax = axs[k]
        show(ax, d.I[0], d.XI, d.YI, cmap=CM_SRC, alpha=0.5)
        if k > 0:
            _draw_lines(ax, pos[k - 1], sizes, color=LIGHT, lw=0.6)
        _draw_lines(ax, pos[k], sizes)
        if k > 0:
            vt = v[k - 1].numpy() / nt     # displacement of this step
            ax.quiver(Vx[sel], Vy[sel], vt[..., 1][sel], vt[..., 0][sel], angles='xy', scale_units='xy',
                      scale=1 / ARROW_GAIN, color='k', width=0.005, headwidth=3.5, headlength=4, zorder=5)
        clean(ax)
        set_box(ax, box)
        label(ax, 'abcd'[k], titles[k])
    note(axs[1], rf'arrows: $v_t/n_t$, drawn {ARROW_GAIN:g}$\times$ longer', y=-0.03)
    axs[nt].text(1.0, -0.03, r'$= \varphi(x)$', transform=axs[nt].transAxes, ha='right', va='top', fontsize=9)
    scalebar(axs[0], loc=(0.05, 0.03))
    # the affine: A phi(x) in the target space
    ax = axs[nt + 1]
    show(ax, d.J[0], d.XJ, d.YJ, cmap=CM_TGT, alpha=0.5)
    Q = pos[nt] @ A[:2, :2].numpy().T + A[:2, 2].numpy()
    _draw_lines(ax, Q, sizes)
    clean(ax)
    set_box(ax, tissue_box(d.xJ, d.yJ, 200))
    label(ax, 'e', r'then the affine: $A\,\varphi(x)$')
    ax.text(1.0, -0.03, 'target space', transform=ax.transAxes, ha='right', va='top', fontsize=9)
    disp = np.linalg.norm(pos[nt] - allpts, axis=1)
    on = STalign.interp([torch.as_tensor(d.YI), torch.as_tensor(d.XI)], torch.as_tensor(d.I),
                        torch.as_tensor(allpts.T[:, :, None]), padding_mode='zeros').numpy()[0, :, 0] > 0.1 * d.I.max()
    print(f'  phi moves grid points on tissue by {np.median(disp[on]):.0f} um (median), max {disp[on].max():.0f} um; '
          f'max |v_t|/nt on the drawn nodes {np.max(np.linalg.norm(v.numpy()[:, sel], axis=-1)) / nt:.0f} um')
    save(fig, 'transformation')


def fig_objective(d):
    '''Matching term: target, contrast-mapped deformed source, residual before/after.'''
    f = d.fit
    s0 = f['snaps'][1][2]          # the forward pass of iteration 0 (after the pre-fit, v = 0)
    s1 = f['snaps'][f['niter']][2]
    J = d.J[0]
    r0 = np.asarray(s0['fAI'])[0] - J
    r1 = np.asarray(s1['fAI'])[0] - J
    E0, E1 = f['Esave'][0], f['Esave'][-1]
    lim = np.percentile(np.abs(r0), 99.5)
    box = tissue_box(d.xJ, d.yJ, 200)
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.1), gridspec_kw=dict(wspace=0.14))
    show(axs[0], J, d.XJ, d.YJ, cmap=CM_TGT)
    label(axs[0], 'a', r'target $J$')
    show(axs[1], np.asarray(s1['fAI'])[0], d.XJ, d.YJ, cmap=CM_SRC, vmin=0, vmax=J.max())
    label(axs[1], 'b', r'$f(I\circ\varphi^{-1}\circ A^{-1})$')
    show(axs[2], r0, d.XJ, d.YJ, cmap=CM_RES, vmin=-lim, vmax=lim)
    label(axs[2], 'c', 'residual, affine only')
    note(axs[2], rf'iteration 0, $E_M$ = {E0[1]:.0f}')
    im = show(axs[3], r1, d.XJ, d.YJ, cmap=CM_RES, vmin=-lim, vmax=lim)
    label(axs[3], 'd', 'residual, affine + deformation')
    note(axs[3], rf'iteration {f["niter"]}, $E_M$ = {E1[1]:.0f}')
    for ax in axs:
        clean(ax)
        set_box(ax, box)
    scalebar(axs[0], loc=(0.04, 0.03))
    cb = fig.colorbar(im, ax=list(axs[2:]), fraction=0.03, pad=0.01, shrink=0.85)
    cb.set_label('residual (cells per pixel)')
    cb.outline.set_visible(False)
    print(f'  E_M {E0[1]:.0f} -> {E1[1]:.0f}; weights at the end: W_M in [{f["WM"].min():.3f}, {f["WM"].max():.3f}]')
    save(fig, 'objective')


def fig_regularization(d):
    '''The smoothing kernel K and the effect of the regularization strength.'''
    cases = [(20.0, 'strong'), (LC.SIGMA_DISP, 'default'), (500.0, 'weak')]
    fits = []
    for sd, _ in cases:
        t = time.perf_counter()
        o = d.fit if sd == LC.SIGMA_DISP else lddmm(d.gI, d.I, d.gJ, d.J, sigma_disp=sd)
        fits.append(o)
        print(f'  sigma_disp={sd:g}: sigmaR={o["settings"]["sigmaR"]:.3g} ({time.perf_counter() - t:.1f} s)')

    fig = plt.figure(figsize=(12, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1, 1, 1], wspace=0.12)
    a = fig.add_subplot(gs[0])
    # profiles of the kernel through its center, on a velocity grid of spacing a/2 as LDDMM uses
    for aa, ls in ((250.0, ':'), (500.0, '-'), (1000.0, '--')):
        n = 101
        grid = [np.arange(n) * (aa / 2)] * 2
        _, K, _ = LC.kernels(grid, aa, 2.0)
        k = np.fft.fftshift(np.fft.ifftn(K).real)
        prof = k[n // 2]
        r = (np.arange(n) - n // 2) * (aa / 2) / 1000.0
        a.plot(r, prof / prof.max(), ls, color='k', lw=1.3, label=f'a = {aa:.0f} µm' + (' (default)' if aa == 500 else ''))
    a.axhline(0, color=LIGHT, lw=0.8, zorder=0)
    a.set_xlim(-4, 4)
    a.set_ylim(-0.05, 1.55)
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel('distance (mm)')
    a.set_ylabel('K (scaled to 1 at 0)')
    a.legend(frameon=False, loc='upper left', fontsize=7.5, handlelength=2.2)
    label(a, 'a', r'smoothing kernel $K = (L^\dagger L)^{-1}$')

    box = tissue_box(d.xJ, d.yJ, 200)
    for k, ((sd, name), o) in enumerate(zip(cases, fits)):
        ax = fig.add_subplot(gs[k + 1])
        AI = deformed_source(d, o['A'], o['v'], o['xv'])
        show(ax, composite(AI, d.J[0]), d.XJ, d.YJ)
        grid_contours(ax, o, d.gJ, linewidths=0.5)
        disp = nonlinear_displacement(d, o)
        clean(ax)
        set_box(ax, box)
        label(ax, 'bcd'[k], name + rf': $\sigma_{{disp}}$ = {sd:.0f} µm')
        note(ax, rf'$\sigma_R$ = {o["settings"]["sigmaR"]:.2g}' + '\n' + rf'$\varphi$ moves cells by '
             f'{np.median(disp):.0f} µm\n(median), at most {disp.max():.0f} µm')
    save(fig, 'regularization')


def fig_optimization(d):
    '''Energy vs iteration and the alignment at a few iterations.'''
    f = d.fit
    E = f['Esave']
    it = np.arange(len(E))
    shots = [0, 20, 200, f['niter']]
    fig = plt.figure(figsize=(12, 3.3))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.7, 1, 1, 1, 1, 1], wspace=0.08)
    a = fig.add_subplot(gs[0])
    a.plot(it, E[:, 0], color='k', lw=1.6, label=r'$E = E_M + E_R$')
    a.plot(it, E[:, 1], color=SRC, lw=1.6, label=r'$E_M$ (matching)')
    a.plot(it, E[:, 2], color=ACCENT, lw=1.6, label=r'$E_R$ (regularization)')
    a.set_xscale('symlog', linthresh=10, linscale=0.5)
    a.set_xlim(0, len(E))
    a.set_ylim(0, E[:, 0].max() * 1.12)
    a.axvspan(50, 75, color='#EEEEEE', zorder=0)
    a.text(78, E[:, 0].max() * 1.1, r'$W_M$ re-estimated from it. 50' + '\n' + r'(0.5 $\to$ 1): $E_M$ rises',
           fontsize=7, color=GREY, va='top')
    for s in shots:
        i = min(s, len(E) - 1)
        a.plot([i], [E[i, 0]], 'o', ms=4, color='k')
    a.set_xlabel('iteration')
    a.set_ylabel('energy')
    a.legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.33))
    label(a, 'a', 'Energy during gradient descent')

    box = tissue_box(d.xJ, d.yJ, 200)
    ax = fig.add_subplot(gs[1])
    AI = deformed_source(d, torch.eye(3, dtype=torch.float64), None, None)
    show(ax, composite(AI, d.J[0]), d.XJ, d.YJ)
    clean(ax)
    set_box(ax, box)
    label(ax, 'b', 'start (identity)')
    for k, s in enumerate(shots):
        A, v, _ = f['snaps'][s]
        ax = fig.add_subplot(gs[k + 2])
        AI = deformed_source(d, A, v, f['xv'])
        show(ax, composite(AI, d.J[0]), d.XJ, d.YJ)
        clean(ax)
        set_box(ax, box)
        label(ax, 'cdef'[k], 'affine pre-fit' if s == 0 else f'iteration {s}')
    note(fig.axes[1], 'blue: source, vermillion: target, grey: both', y=-0.03)
    print(f'  E at iterations 0, 49, 75, {len(E) - 1}: {E[0, 0]:.0f}, {E[49, 0]:.0f}, {E[75, 0]:.0f}, {E[-1, 0]:.0f}')
    save(fig, 'optimization')


def fig_mixture(d):
    '''A target with missing tissue: the matching weights with and without mixture='em'.'''
    cut = 6300.0                                 # remove target cells with x > cut
    keep = d.xJ <= cut
    xJ, yJ = d.xJ[keep], d.yJ[keep]
    # same raster grid as the complete target: two extra cells of weight 0 at the corners
    XJ, YJ, J = STalign.rasterize(np.r_[xJ, d.xJ.min(), d.xJ.max()], np.r_[yJ, d.yJ.min(), d.yJ.max()],
                                  g=np.r_[np.ones(len(xJ)), 0.0, 0.0], dx=DX, blur=BLUR, draw=0)
    assert np.allclose(XJ, d.XJ) and np.allclose(YJ, d.YJ)
    gJ = [YJ, XJ]
    L, T = d.tutorial_guess()
    outs = {}
    for m in (None, 'em'):
        t = time.perf_counter()
        outs[m] = lddmm(d.gI, d.I, gJ, J, L=L, T=T, mixture=m)
        print(f'  partial target, mixture={m}: {time.perf_counter() - t:.1f} s')
    ref = d.fit
    rx, ry = to_target(d, ref)
    inside = rx <= cut - 300                      # cells that the complete fit puts in the kept part
    err = {}
    for m, o in outs.items():
        x, y = to_target(d, o)
        err[m] = np.hypot(x - rx, y - ry)[inside]
        W = o['WM'].numpy()
        print(f'  mixture={m}: W_M mean {W[:, XJ > cut + 200].mean():.2f} right of the cut, '
              f'{W[:, XJ < cut - 200].mean():.2f} left; distance to the complete fit: median '
              f'{np.median(err[m]):.0f} um, 90th percentile {np.percentile(err[m], 90):.0f} um')

    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(2, 3, wspace=0.06, hspace=0.25)
    box = tissue_box(d.xJ, d.yJ, 200)

    def cut_line(ax):
        ax.axvline(cut, color=ACCENT, lw=1.2, ls='--')

    a = fig.add_subplot(gs[0, 0])
    show(a, J[0], XJ, YJ, cmap=CM_TGT)
    a.add_patch(Rectangle((cut, box[2]), box[1] - cut, box[3] - box[2], fc=ACCENT, alpha=0.08, ec='none'))
    cut_line(a)
    a.text(cut + 150, box[3] - 300, 'cells\nremoved', color=ACCENT, va='top', fontsize=8)
    label(a, 'a', 'partial target')
    axW = []
    for k, m in enumerate((None, 'em')):
        ax = fig.add_subplot(gs[0, k + 1])
        im = show(ax, outs[m]['WM'].numpy(), XJ, YJ, cmap=CM_W, vmin=0, vmax=1)
        cut_line(ax)
        label(ax, 'bc'[k], f"matching weight $W_M$, mixture={m!r}")
        axW.append(ax)
    e = fig.add_subplot(gs[1, 0])
    AI = deformed_source(d, ref['A'], ref['v'], ref['xv'])
    show(e, composite(AI, d.J[0]), d.XJ, d.YJ)
    grid_contours(e, ref, d.gJ, linewidths=0.5)
    label(e, 'd', 'reference: fit to the complete target')
    for k, m in enumerate((None, 'em')):
        ax = fig.add_subplot(gs[1, k + 1])
        AI = deformed_source(d, outs[m]['A'], outs[m]['v'], outs[m]['xv'], gJ=gJ)
        show(ax, composite(AI, J[0]), XJ, YJ)
        grid_contours(ax, outs[m], gJ, linewidths=0.5)
        cut_line(ax)
        label(ax, 'ef'[k], f'fit to the partial target, mixture={m!r}')
        note(ax, f'source cells left of the cut: median distance\n{np.median(err[m]):.0f} µm '
                 'from their position in the reference')
    for ax in fig.axes:
        if ax.images:
            clean(ax)
            set_box(ax, box)
    scalebar(a, loc=(0.04, 0.03))
    fig.canvas.draw()
    p = axW[1].get_position()
    cax = fig.add_axes([p.x1 + 0.008, p.y0 + 0.1 * p.height, 0.008, 0.8 * p.height])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    save(fig, 'mixture')


def fig_maps(d):
    '''Images are pulled back (target-to-source sampling); cells are pushed forward (exact inverse).'''
    f = d.fit
    xv, v, A = f['xv'], f['v'], f['A']
    fig = plt.figure(figsize=(11.5, 6.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.95], wspace=0.14, hspace=0.25)
    boxJ = tissue_box(d.xJ, d.yJ, 200)
    boxI = tissue_box(d.xI, d.yI, 200)

    # images: a coarse lattice of target pixels and the points of the source they sample
    lat = (1500.0, 8700.0, 1500.0, 8700.0)
    lines = _grid_lines(lat, 600.0, n=120)
    sizes = [len(ln) for ln in lines]
    pts = np.concatenate(lines)
    nodes = np.stack(np.meshgrid(np.arange(1500.0, 8701, 600), np.arange(1500.0, 8701, 600), indexing='ij'), -1)
    nodes = nodes.reshape(-1, 2)
    b_lines = STalign.transform_points_target_to_source(xv, v, A, pts).numpy()
    b_nodes = STalign.transform_points_target_to_source(xv, v, A, nodes).numpy()
    ax = fig.add_subplot(gs[0, 0])
    show(ax, d.J[0], d.XJ, d.YJ, cmap=CM_TGT, alpha=0.55)
    _draw_lines(ax, pts, sizes, color=TGT, lw=0.6)
    ax.plot(nodes[:, 1], nodes[:, 0], 's', ms=2.2, color=TGT)
    clean(ax)
    set_box(ax, boxJ)
    label(ax, 'a', r'target pixels $y$ (target space)')
    ax2 = fig.add_subplot(gs[0, 1])
    show(ax2, d.I[0], d.XI, d.YI, cmap=CM_SRC, alpha=0.55)
    _draw_lines(ax2, b_lines, sizes, color=TGT, lw=0.6)
    ax2.plot(b_nodes[:, 1], b_nodes[:, 0], 's', ms=2.2, color=TGT)
    clean(ax2)
    set_box(ax2, boxI)
    label(ax2, 'b', r'where they sample $I$: $\varphi^{-1}(A^{-1}y)$')
    fig.canvas.draw()
    arrow_between(fig, ax, ax2, text='images:\nbackward map')

    # points: a subset of the source cells and their exact forward images
    idx = subsample(len(d.xI), 400, seed=3)
    P = d.cells_I()[idx]
    q = STalign.transform_points_source_to_target(xv, v, A, P).numpy()
    ax3 = fig.add_subplot(gs[1, 0])
    show(ax3, d.I[0], d.XI, d.YI, cmap=CM_SRC, alpha=0.55)
    ax3.plot(P[:, 1], P[:, 0], 'o', ms=2.5, color=SRC)
    clean(ax3)
    set_box(ax3, boxI)
    label(ax3, 'c', r'source cells $x$ (source space)')
    ax4 = fig.add_subplot(gs[1, 1])
    show(ax4, d.J[0], d.XJ, d.YJ, cmap=CM_TGT, alpha=0.55)
    ax4.plot(q[:, 1], q[:, 0], 'o', ms=2.5, color=SRC)
    clean(ax4)
    set_box(ax4, boxJ)
    label(ax4, 'd', r'where they go: $A\,\varphi(x)$')
    fig.canvas.draw()
    arrow_between(fig, ax3, ax4, text='cells:\nforward map')
    scalebar(ax, loc=(0.04, 0.03))

    # round trip |b(f(x)) - x| over all source cells, exact inverse vs forward Euler
    h = fig.add_subplot(gs[:, 2])
    allP = d.cells_I()
    floor = 1e-12
    for exact, col, name in ((True, ACCENT, 'exact inverse (default)'),
                             (False, GREY, 'forward Euler (exact_inverse=False)')):
        qq = STalign.transform_points_source_to_target(xv, v, A, allP, exact_inverse=exact)
        back = STalign.transform_points_target_to_source(xv, v, A, qq).numpy()
        e = np.linalg.norm(back - allP, axis=1)
        med = f'{np.median(e):.1g}' if np.median(e) >= 1e-3 else f'{np.median(e):.0e}'
        h.hist(np.maximum(e, floor), bins=np.logspace(-12, 3, 61), color=col, alpha=0.8,
               label=f'{name}\nmedian {med} µm, max {e.max():.2g} µm')
        print(f'  round trip, exact_inverse={exact}: median {np.median(e):.3g} um, max {e.max():.3g} um')
    h.set_xscale('log')
    h.set_xticks([1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3])
    h.set_xlabel(r'round-trip error $|b(f(x)) - x|$ (µm)')
    h.set_ylabel('source cells')
    h.set_ylim(0, h.get_ylim()[1] * 1.35)
    h.legend(frameon=False, loc='upper center', fontsize=7.5)
    label(h, 'e', 'Is the forward map the inverse?')
    save(fig, 'maps')


def _find_atlas(arg):
    for p in ([Path(arg)] if arg else []) + [ROOT / 'docs' / 'notebooks' / 'aba_nissl.nrrd']:
        if p.is_file():
            return p
    return None


def _rotation(axis, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    R = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def fig_slice3d(d, atlas_path=None):
    '''Schematic of LDDMM_3D_to_slice: the target slice is the plane z = 0, mapped into the atlas.'''
    path = _find_atlas(atlas_path)
    dxA = 50.0
    nA = (264, 160, 228)                # the Allen CCF at 50 um: AP, DV, ML
    vol = None
    if path is not None:
        import nrrd
        vol, hdr = nrrd.read(str(path))
        vol = vol.astype(np.float64)
        dxA = float(np.diag(hdr['space directions'])[0])
        nA = vol.shape
    xA = [np.arange(n) * dxA - (n - 1) * dxA / 2 for n in nA]
    # the target slice (S1R1) rasterized at 50 um and centred: target pixels (0, y, x)
    xs, ys = read_cells(FILES['S1R1'])
    X, Y, M = STalign.rasterize(xs, ys, dx=50.0, blur=1.0, draw=0)
    M = M[0]                            # rows run from dorsal to ventral, as in the atlas
    Yc, Xc = Y - Y.mean(), X - X.mean()
    # an illustrative plane: coronal through atlas slice 177 (as in the MERFISH-Allen tutorial),
    # tilted 6 deg about ML and 4 deg about DV, scaled by 0.9; target rows go along DV, columns
    # along ML.
    R = _rotation(2, 6) @ _rotation(1, 4)
    T = np.array([xA[0][177], 0.0, 0.0])

    def to_atlas(y, x):                 # target (0, y, x) -> atlas (AP, DV, ML)
        p = np.stack([np.zeros_like(y), y * 0.9, x * 0.9], -1)
        return p @ R.T + T

    fig = plt.figure(figsize=(11.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.06)
    ax = fig.add_subplot(gs[0], projection='3d')
    lo = [x[0] for x in xA]
    hi = [x[-1] for x in xA]
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    P3 = lambda P: (P[..., 0] / 1000, P[..., 2] / 1000, -P[..., 1] / 1000)   # plot axes: AP, ML, up
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:
                ax.plot(*P3(corners[[i, j]]), color=GREY, lw=0.8)
    s = 3
    Yg, Xg = np.meshgrid(Yc[::s], Xc[::s], indexing='ij')
    Pp = to_atlas(Yg, Xg)
    cols = CM_TGT(np.clip(M[::s, ::s] / np.percentile(M, 99.5), 0, 1))
    ax.plot_surface(*P3(Pp), facecolors=cols, rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)
    edge = np.array([[Yc[0], Xc[0]], [Yc[0], Xc[-1]], [Yc[-1], Xc[-1]], [Yc[-1], Xc[0]], [Yc[0], Xc[0]]])
    ax.plot(*P3(to_atlas(edge[:, 0], edge[:, 1])), color=TGT, lw=1.5)
    ax.set_xlabel('AP (mm)', labelpad=-7)
    ax.set_ylabel('ML (mm)', labelpad=-7)
    ax.set_zlabel('DV (mm)', labelpad=-9)
    ax.tick_params(pad=-3, labelsize=6.5)
    ax.set_box_aspect((nA[0], nA[2], nA[1]), zoom=1.05)
    ax.view_init(elev=18, azim=-28)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor('white')
        pane.set_edgecolor(LIGHT)
    ax.grid(False)
    ax.set_title(r'$\bf{a}$  atlas volume $I$ (box) and the slice plane', loc='left')

    ext = (Xc[0], Xc[-1], Yc[-1], Yc[0])
    b = fig.add_subplot(gs[1])
    b.imshow(M, cmap=CM_TGT, extent=ext, vmax=np.percentile(M, 99.5))
    clean(b)
    scalebar(b, 2000, '2 mm', loc=(0.04, 0.02))
    label(b, 'b', r'target slice $J(y, x)$, the plane $z = 0$')
    c = fig.add_subplot(gs[2])
    if vol is not None:
        Yf, Xf = np.meshgrid(Yc, Xc, indexing='ij')
        Pa = to_atlas(Yf, Xf)
        samp = STalign.interp3D([torch.as_tensor(x) for x in xA], torch.as_tensor(vol[None]),
                                torch.as_tensor(np.moveaxis(Pa, -1, 0)[:, None]), padding_mode='zeros').numpy()[0, 0]
        c.imshow(samp, cmap='Greys', extent=ext, vmax=np.percentile(samp, 99.5))
        label(c, 'c', r'atlas on that plane: $I(A^{-1}(0, y, x))$')
    else:
        c.text(0.5, 0.5, 'Allen atlas not available\n(pass --atlas)', ha='center', va='center',
               transform=c.transAxes, color=GREY)
        label(c, 'c', r'atlas on that plane: $I(A^{-1}(0, y, x))$')
    clean(c)
    save(fig, 'slice3d')


FIGURES = {
    'overview': fig_overview, 'rasterization': fig_rasterization, 'transformation': fig_transformation,
    'objective': fig_objective, 'regularization': fig_regularization, 'optimization': fig_optimization,
    'mixture': fig_mixture, 'maps': fig_maps, 'slice3d': fig_slice3d,
}


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', nargs='+', choices=sorted(FIGURES), help='make only these figures')
    ap.add_argument('--out', default=str(HERE), help='output directory (default: docs/figures)')
    ap.add_argument('--atlas', default=None, help='Allen Nissl volume (aba_nissl.nrrd) for the 3D figure')
    args = ap.parse_args()
    OUT = Path(args.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'STalign {STalign.__version__}, engine {STalign.backend_info()["auto"]}, '
          f'{os.environ["STALIGN_NUM_THREADS"]} threads')
    t0 = time.perf_counter()
    d = Data()
    for name in (args.only or FIGURES):
        t = time.perf_counter()
        print(f'{name}:')
        if name == 'slice3d':
            fig_slice3d(d, args.atlas)
        else:
            FIGURES[name](d)
        print(f'  {time.perf_counter() - t:.1f} s')
    print(f'done in {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
