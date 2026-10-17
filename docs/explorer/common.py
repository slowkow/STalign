'''Shared helpers for the STalign explorer asset scripts.

Everything here only *uses* STalign (main tree, read-only). The one non-public piece is
`run_frames`, which drives the same engine that STalign.LDDMM drives (prepare_problem ->
resolve_defaults -> select_engine) so that one fit can be snapshotted at several iteration
counts; `check_run_frames` verifies that its last frame equals a plain STalign.LDDMM call.
'''
import glob
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))  # docs/explorer
REPO = os.path.dirname(os.path.dirname(HERE))
ASSETS = os.path.join(HERE, 'build')  # everything the scripts compute; package.py picks what the page needs
sys.path.insert(0, REPO)

from STalign import STalign  # noqa: E402
from STalign import _lddmm_common as LC  # noqa: E402

os.environ.setdefault('OMP_NUM_THREADS', '8')
NUM_THREADS = 8
torch.set_num_threads(NUM_THREADS)

# display intensity scale shared by every overlay: raster value (cells per 30 um pixel,
# dx=30, blur=1.5 rasters) mapped linearly to [0, 1] with this upper limit, then clipped
VMAX_CELLS_PER_PIXEL = None  # set by tutorial_rasters (99.5th percentile of target pixels)


# ------------------------------------------------------------------ data
def load_cells():
    d = os.path.join(REPO, 'docs', 'merfish_data')
    f1 = glob.glob(os.path.join(d, '*Slice2_Replicate3_cell_metadata_S2R3.csv.gz'))[0]
    f2 = glob.glob(os.path.join(d, '*Slice2_Replicate2_cell_metadata_S2R2.csv.gz'))[0]
    df1, df2 = pd.read_csv(f1), pd.read_csv(f2)
    return (np.array(df1['center_x']), np.array(df1['center_y']),
            np.array(df2['center_x']), np.array(df2['center_y']))


def tutorial_initial_affine(xI, yI, xJ, yJ, theta_deg=45):
    '''Exactly the tutorial's initial affine: rotate by -45 deg about the source centroid,
    move the source centroid onto the target centroid. Returns the moved source points.'''
    theta0 = (np.pi/180)*-theta_deg
    L = np.array([[np.cos(theta0), -np.sin(theta0)], [np.sin(theta0), np.cos(theta0)]])
    s = L @ np.array([xI, yI])
    T = np.array([np.mean(xI) - np.cos(theta0)*np.mean(xI) + np.sin(theta0)*np.mean(yI) - (np.mean(xI)-np.mean(xJ)),
                  np.mean(yI) - np.sin(theta0)*np.mean(xI) - np.cos(theta0)*np.mean(yI) - (np.mean(yI)-np.mean(yJ))])
    return s[0] + T[0], s[1] + T[1]


def tutorial_problem():
    '''Source (S2R3 after the tutorial's initial affine) and target (S2R2), rasterized as in
    the tutorial (dx=30, blur=1.5).'''
    xI, yI, xJ, yJ = load_cells()
    xI2, yI2 = tutorial_initial_affine(xI, yI, xJ, yJ)
    XI, YI, I = STalign.rasterize(xI2, yI2, dx=30, blur=1.5, draw=0)
    XJ, YJ, J = STalign.rasterize(xJ, yJ, dx=30, blur=1.5, draw=0)
    global VMAX_CELLS_PER_PIXEL
    VMAX_CELLS_PER_PIXEL = float(np.percentile(J[0][J[0] > 0.05*J.max()], 99.5))
    return dict(xI=xI2, yI=yI2, xJ=xJ, yJ=yJ, XI=XI, YI=YI, I=I, XJ=XJ, YJ=YJ, J=J,
                xI_raw=xI, yI_raw=yI)


# ------------------------------------------------------------------ fitting
def _lddmm_kwargs(**kw):
    d = {k: p.default for k, p in inspect.signature(STalign.LDDMM).parameters.items()
         if p.default is not inspect.Parameter.empty}
    d.update(kw)
    return d


def run_frames(xI, I, xJ, J, frames, **kw):
    '''Run one LDDMM fit and return a result dict (as STalign.LDDMM returns) after each
    iteration count in `frames` (sorted, may include 0). Frame k is exactly what
    STalign.LDDMM(..., niter=k_max) holds after k iterations, i.e. what
    STalign.LDDMM(..., niter=k) returns (the fit is deterministic and the settings do not
    depend on niter). Also returns the energy history Esave (niter x 3: E, EM, ER) and info.'''
    a = _lddmm_kwargs(**kw)
    niter = max(frames)
    a['niter'] = niter
    keys = [k for k in inspect.signature(LC.prepare_problem).parameters if k not in
            ('dim', 'xI', 'I', 'xJ', 'J', 'landmark_map')]
    prob = LC.prepare_problem(2, xI, I, xJ, J, **{k: a[k] for k in keys if k in a})
    info = LC.resolve_defaults(prob, prefit=a['prefit'], sigma_disp=a['sigma_disp'],
                               legacy_defaults=a['legacy_defaults'], num_threads=a['num_threads'])
    eng = LC.select_engine(prob, a['backend'], a['num_threads'])
    out, done = {}, 0
    for f in sorted(frames):
        if f == 0:
            r = LC._initial_result(prob)
        else:
            eng.run(f - done)
            done = f
            r = LC._result(prob, eng.snapshot())
        r.update(info)
        out[f] = r
    E = np.asarray(eng.snapshot()['Esave'])[:niter, :3] if niter else np.zeros((0, 3))
    return out, E, info, type(eng).__name__


def check_run_frames(xI, I, xJ, J, frames_out, **kw):
    '''Max |difference| of A and v between the last frame and a plain STalign.LDDMM call.'''
    n = max(frames_out)
    ref = STalign.LDDMM(xI, I, xJ, J, niter=n, draw=False, num_threads=NUM_THREADS, **kw)
    r = frames_out[n]
    return float((ref['A'] - r['A']).abs().max()), float((ref['v'] - r['v']).abs().max())


# ------------------------------------------------------------------ transforms / metrics
def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def s2t(r, x, y):
    '''Source cells (x, y) -> target space with the fit r (exact inverse). Returns x, y.'''
    p = STalign.transform_points_source_to_target(r['xv'], r['v'], r['A'], np.stack([y, x], 1))
    p = to_np(p)
    return p[:, 1], p[:, 0]


def s2t_affine(r, x, y):
    p = to_np(STalign.transform_points_source_to_target(r['xv'], None, r['A'], np.stack([y, x], 1)))
    return p[:, 1], p[:, 0]


def t2s_affine(r, x, y):
    p = to_np(STalign.transform_points_target_to_source(r['xv'], None, r['A'], np.stack([y, x], 1)))
    return p[:, 1], p[:, 0]


def deformed_source_image(r, P):
    '''Source image deformed onto the target grid (1 channel, numpy).'''
    phiI = STalign.transform_image_source_to_target(r['xv'], r['v'], r['A'], [P['YI'], P['XI']], P['I'],
                                                    [P['YJ'], P['XJ']])
    return to_np(phiI)[0]


def jacobian_stats(r, P, mask):
    '''det of the Jacobian of the backward (target->source) map times det(A): the area change
    of the nonrigid part alone (1 = no change), over the target pixels in mask.'''
    phii = to_np(STalign.build_transform(r['xv'], r['v'], r['A'], XJ=[P['YJ'], P['XJ']], direction='b'))
    dy = float(P['YJ'][1] - P['YJ'][0]); dx = float(P['XJ'][1] - P['XJ'][0])
    d0_dy, d0_dx = np.gradient(phii[..., 0], dy, dx)
    d1_dy, d1_dx = np.gradient(phii[..., 1], dy, dx)
    det = (d0_dy*d1_dx - d0_dx*d1_dy) * abs(np.linalg.det(to_np(r['A'])[:2, :2]))
    d = det[mask]
    ld = np.log(np.clip(d, 1e-6, None))
    return dict(detJ_min=float(d.min()), detJ_max=float(d.max()), detJ_p01=float(np.percentile(d, 1)),
                detJ_p99=float(np.percentile(d, 99)), logdetJ_sd=float(ld.std()),
                folded_fraction=float(np.mean(d <= 0)))


def image_scores(phiI, J, mask):
    a, b = phiI[mask], J[mask]
    return dict(corr=float(np.corrcoef(a, b)[0, 1]), rmse_cells_per_pixel=float(np.sqrt(np.mean((a-b)**2))))


# ------------------------------------------------------------------ rendering
def _flip(img):
    # raster row 0 = smallest y. PNGs are written with the top row = largest y (y up, as the
    # tutorial's figures after invert_yaxis and as scatter plots).
    return img[::-1]


def overlay_png(path, target, source, vmax=None):
    '''Target magenta (R+B), deformed source green (G); white where both. Linear, clipped.'''
    vmax = vmax or VMAX_CELLS_PER_PIXEL
    t = np.clip(target/vmax, 0, 1); s = np.clip(source/vmax, 0, 1)
    rgb = np.stack([t, s, t], -1)
    Image.fromarray((_flip(rgb)*255 + 0.5).astype(np.uint8)).save(path, optimize=True)


def gray_png(path, img, vmax, vmin=0.0):
    g = np.clip((img - vmin)/(vmax - vmin), 0, 1)
    Image.fromarray((_flip(g)*255 + 0.5).astype(np.uint8), mode='L').save(path, optimize=True)


def weight_png(path, W):
    '''Matching weight WM in [0, 1] -> grayscale (black 0, white 1).'''
    gray_png(path, np.asarray(W), 1.0)


def extent_of(X, Y):
    dx = float(X[1]-X[0]); dy = float(Y[1]-Y[0])
    return [round(float(X[0])-dx/2, 1), round(float(X[-1])+dx/2, 1), round(float(Y[0])-dy/2, 1),
            round(float(Y[-1])+dy/2, 1)]


def r1(a):
    return [round(float(v), 1) for v in np.asarray(a).ravel()]


def sig(a, n=4):
    return [float(f'{float(v):.{n}g}') for v in np.asarray(a).ravel()]


def dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))


def rot_about(x, y, deg, cx, cy):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return cx + c*(x-cx) - s*(y-cy), cy + s*(x-cx) + c*(y-cy)
