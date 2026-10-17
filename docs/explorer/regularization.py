'''regularization/: the tutorial's MERFISH alignment (S2R3 -> S2R2, tutorial initial affine,
dx=30, blur=1.5 rasters) at several sigma_disp, snapshotted at several iterations.'''
import os
import time

import numpy as np

import common as C
from common import (ASSETS, STalign, deformed_source_image, dump, extent_of, image_scores, overlay_png,
                    run_frames, check_run_frames, s2t, sig, r1, to_np)

SIGMAS = [3.0, 10.0, 30.0, 94.03, 300.0, 1000.0]
FRAMES = [0, 25, 100, 400, 1000, 2000]
GRID_SPACING = 500.0  # um
GRID_STEP = 100.0     # um between samples along a grid line
OUT = os.path.join(ASSETS, 'regularization')


def tag(sd):
    return f'sd{int(round(sd)):04d}'


def affine_fit(p, q):
    '''Least-squares affine q ~ M [p;1]. Returns predicted q and the 2x2 linear part.'''
    X = np.c_[p, np.ones(len(p))]
    M, *_ = np.linalg.lstsq(X, q, rcond=None)
    return X @ M, M[:2, :2].T


def local_area_change(r, P, mask, Lls):
    '''det of the Jacobian of the source->target map, divided by the det of the best-fitting
    affine (Lls): 1 = the nonrigid part neither compresses nor expands. Computed from the
    backward map on the target grid (det D(phi^-1) = 1/det D(phi)).'''
    phii = to_np(STalign.build_transform(r['xv'], r['v'], r['A'], XJ=[P['YJ'], P['XJ']], direction='b'))
    dy = float(P['YJ'][1]-P['YJ'][0]); dx = float(P['XJ'][1]-P['XJ'][0])
    a, b = np.gradient(phii[..., 0], dy, dx)
    c, d = np.gradient(phii[..., 1], dy, dx)
    det_back = (a*d - b*c)[mask]          # source area per target area
    ratio = 1.0/(det_back*abs(np.linalg.det(Lls)))
    ld = np.log(np.clip(ratio, 1e-6, None))
    return dict(area_ratio_p01=float(np.percentile(ratio, 1)), area_ratio_p99=float(np.percentile(ratio, 99)),
                area_ratio_min=float(ratio.min()), area_ratio_max=float(ratio.max()),
                log_area_ratio_sd=float(ld.std()), folded_fraction=float(np.mean(det_back <= 0)))


def grid_lines(bbox):
    x0, x1, y0, y1 = bbox
    xs = np.arange(np.floor(x0/GRID_SPACING)*GRID_SPACING, x1 + 1e-9, GRID_SPACING)
    ys = np.arange(np.floor(y0/GRID_SPACING)*GRID_SPACING, y1 + 1e-9, GRID_SPACING)
    lines = []
    ty = np.arange(ys[0], ys[-1] + 1e-9, GRID_STEP)
    tx = np.arange(xs[0], xs[-1] + 1e-9, GRID_STEP)
    for x in xs:
        lines.append(('v', np.c_[np.full_like(ty, x), ty]))
    for y in ys:
        lines.append(('h', np.c_[tx, np.full_like(tx, y)]))
    return lines


def main():
    os.makedirs(OUT, exist_ok=True)
    P = C.tutorial_problem()
    xI_, xJ_ = [P['YI'], P['XI']], [P['YJ'], P['XJ']]
    J0 = P['J'][0]
    mask = J0 > 0.1*C.VMAX_CELLS_PER_PIXEL
    src = np.c_[P['xI'], P['yI']]

    # before the pre-fit: the source after the tutorial's affine only, resampled on the target grid
    eye = np.eye(3)
    I_init = to_np(STalign.transform_image_source_with_A(eye, xI_, P['I'], xJ_))[0]
    overlay_png(os.path.join(OUT, 'overlay_tutorial_affine.png'), J0, I_init)

    runs, grid_bbox, lines0 = [], None, None
    for sd in SIGMAS:
        t = time.time()
        fr, E, info, engine = run_frames(xI_, P['I'], xJ_, P['J'], FRAMES, draw=False,
                                         num_threads=C.NUM_THREADS, sigma_disp=sd)
        secs = time.time() - t
        dA, dv = check_run_frames(xI_, P['I'], xJ_, P['J'], fr, sigma_disp=sd)
        d = os.path.join(OUT, tag(sd))
        os.makedirs(d, exist_ok=True)
        # grid: an axis-aligned square grid in target coordinates placed on the source by the
        # iteration-0 affine (A0, the pre-fit result, identical for every sigma_disp), i.e. the
        # source-space points A0^-1 g, then carried to target space by each frame's full map.
        r0 = fr[0]
        x0, y0 = s2t(r0, P['xI'], P['yI'])
        if grid_bbox is None:
            grid_bbox = [min(x0.min(), P['xJ'].min()), max(x0.max(), P['xJ'].max()),
                         min(y0.min(), P['yJ'].min()), max(y0.max(), P['yJ'].max())]
            lines0 = grid_lines(grid_bbox)
        A0 = to_np(r0['A'])
        frames_meta, grid_json = [], {}
        for k in FRAMES:
            r = fr[k]
            phiI = deformed_source_image(r, P)
            png = f'overlay_it{k:04d}.png'
            overlay_png(os.path.join(d, png), J0, phiI)
            xk, yk = s2t(r, P['xI'], P['yI'])
            q = np.c_[xk, yk]
            moved = np.hypot(xk - x0, yk - y0)
            pred, Lls = affine_fit(src, q)
            resid = np.hypot(*(q - pred).T)
            # grid
            gl = []
            for kind, pts in lines0:
                # source-space position of target-grid point g under A0 (affine only)
                ps = STalign.transform_points_target_to_source(r0['xv'], None, A0, np.ascontiguousarray(pts[:, ::-1]))
                pt = to_np(STalign.transform_points_source_to_target(r['xv'], r['v'], r['A'], ps))[:, ::-1]
                gl.append(r1(pt))
            grid_json[str(k)] = gl
            m = dict(iteration=k, overlay=f'regularization/{tag(sd)}/{png}',
                     E=None if k == 0 else float(E[k-1, 0]), EM=None if k == 0 else float(E[k-1, 1]),
                     ER=None if k == 0 else float(E[k-1, 2]),
                     moved_since_it0_um=dict(mean=float(moved.mean()), median=float(np.median(moved)),
                                             p95=float(np.percentile(moved, 95))),
                     nonaffine_residual_um=dict(mean=float(resid.mean()), median=float(np.median(resid)),
                                                p95=float(np.percentile(resid, 95)), max=float(resid.max())),
                     **local_area_change(r, P, mask, Lls),
                     **{f'image_{a}': b for a, b in image_scores(phiI, J0, mask).items()})
            frames_meta.append(m)
        dump(os.path.join(d, 'grid.json'), dict(
            frames=FRAMES, kinds=[kind for kind, _ in lines0], lines=grid_json))
        dump(os.path.join(d, 'energy.json'), dict(
            iteration=list(range(1, len(E)+1)), E=sig(E[:, 0]), EM=sig(E[:, 1]), ER=sig(E[:, 2])))
        settings = {k: float(v) for k, v in info['settings'].items()}
        runs.append(dict(sigma_disp=sd, tag=tag(sd), settings=settings, seconds=round(secs, 2), engine=engine,
                         check_vs_LDDMM_max_abs_diff=dict(A=dA, v=dv), frames=frames_meta,
                         grid=f'regularization/{tag(sd)}/grid.json',
                         energy=f'regularization/{tag(sd)}/energy.json'))
        f = frames_meta[-1]
        print(f"sigma_disp={sd}: {secs:.1f}s sigmaR={settings['sigmaR']:.3g} epV={settings['epV']:.3g} "
              f"check dA={dA:.1e} dv={dv:.1e} | it2000 EM={f['EM']:.0f} ER={f['ER']:.0f} "
              f"resid={f['nonaffine_residual_um']['mean']:.1f}um area p1-p99 {f['area_ratio_p01']:.2f}-"
              f"{f['area_ratio_p99']:.2f} corr={f['image_corr']:.3f}", flush=True)

    pre = fr[0]['prefit']
    dump(os.path.join(OUT, 'summary.json'), dict(
        frames=FRAMES, sigmas=SIGMAS, extent_um=extent_of(P['XJ'], P['YJ']),
        image_shape=list(J0.shape), grid_bbox_um=r1(grid_bbox), grid_spacing_um=GRID_SPACING,
        grid_sample_step_um=GRID_STEP, vmax_cells_per_pixel=C.VMAX_CELLS_PER_PIXEL,
        overlay_tutorial_affine='regularization/overlay_tutorial_affine.png',
        prefit=dict(ran=bool(pre['ran']), A=to_np(pre['A']).round(6).tolist(),
                    starts_deg=[float(s) for s in pre.get('starts', [])],
                    energy=[float(e) for e in pre.get('energy', [])], best=int(pre.get('best', -1))),
        runs=runs))


if __name__ == '__main__':
    main()
