'''prefit/: the tutorial source rotated by theta about its centroid, aligned to the target with
prefit=False and prefit=True (other settings default, niter=1000).'''
import os
import time

import numpy as np

import common as C
from common import ASSETS, STalign, deformed_source_image, dump, extent_of, overlay_png, rot_about, s2t, to_np

ANGLES = [0, 5, 10] + list(range(15, 181, 15))  # multiples of 30 coincide with the pre-fit's 30-degree starts
NITER = 1000
OUT = os.path.join(ASSETS, 'prefit')
FAIL_UM = 50.0


def main():
    os.makedirs(OUT, exist_ok=True)
    P = C.tutorial_problem()
    xs, ys = P['xI'], P['yI']                 # tutorial source (after the tutorial's affine)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    xJ_ = [P['YJ'], P['XJ']]
    J0 = P['J'][0]

    def fit(theta, prefit):
        xr, yr = rot_about(xs, ys, theta, cx, cy)
        XI, YI, I = STalign.rasterize(xr, yr, dx=30, blur=1.5, draw=0)
        t = time.time()
        out = STalign.LDDMM([YI, XI], I, xJ_, P['J'], niter=NITER, prefit=prefit, draw=False,
                            num_threads=C.NUM_THREADS)
        secs = time.time() - t
        x, y = s2t(out, xr, yr)
        Q = dict(XI=XI, YI=YI, I=I, XJ=P['XJ'], YJ=P['YJ'])
        return out, x, y, Q, secs

    # reference: the unrotated source with the default pre-fit
    ref, xref, yref, _, _ = fit(0, True)
    rows = []
    for th in ANGLES:
        for pf in (False, True):
            out, x, y, Q, secs = fit(th, pf)
            err = np.hypot(x - xref, y - yref)
            name = f'theta{th:03d}_prefit{int(pf)}.png'
            overlay_png(os.path.join(OUT, name), J0, deformed_source_image(out, Q))
            if pf is False:  # the input (rotated source, identity map) - same for both modes
                inp = to_np(STalign.transform_image_source_with_A(np.eye(3), [Q['YI'], Q['XI']], Q['I'],
                                                                  xJ_))[0]
                overlay_png(os.path.join(OUT, f'theta{th:03d}_input.png'), J0, inp)
            pre = out['prefit']
            row = dict(theta_deg=th, prefit=pf, overlay=f'prefit/{name}', input_overlay=f'prefit/theta{th:03d}_input.png',
                       error_um=dict(mean=float(err.mean()), median=float(np.median(err)),
                                     p95=float(np.percentile(err, 95))),
                       success=bool(np.median(err) < FAIL_UM), seconds=round(secs, 2),
                       prefit_ran=bool(pre['ran']))
            if pre['ran']:
                A = to_np(pre['A'])
                # rotation angle of the pre-fit's linear part (target->source, so negate for source->target)
                row['prefit_best_start_deg'] = float(pre['starts'][pre['best']])
                row['prefit_rotation_deg'] = float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])))
            rows.append(row)
            print(f"theta={th:3d} prefit={pf!s:5}: err mean {err.mean():7.1f} median {np.median(err):7.1f} um "
                  f"({secs:.1f}s){' start %g' % row['prefit_best_start_deg'] if pre['ran'] else ''}", flush=True)
    ok = {pf: [r['theta_deg'] for r in rows if r['prefit'] == pf and r['success']] for pf in (False, True)}
    dump(os.path.join(OUT, 'summary.json'), dict(
        angles_deg=ANGLES, niter=NITER, extent_um=extent_of(P['XJ'], P['YJ']), image_shape=list(J0.shape),
        rotation_center_um=[round(cx, 1), round(cy, 1)], success_threshold_median_um=FAIL_UM,
        reference='theta=0, prefit=True (same settings); error = |T_theta(R_theta p) - T_ref(p)| per source cell',
        success_angles={'prefit_false': ok[False], 'prefit_true': ok[True]},
        results=rows))
    print('success angles', ok)


if __name__ == '__main__':
    main()
