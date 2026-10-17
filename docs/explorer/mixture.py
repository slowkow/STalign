'''mixture/: the target with ~30% of its cells cropped away (a straight cut), aligned with
mixture=None and mixture='em', compared with the alignment to the full target.'''
import os
import time

import numpy as np

import common as C
from common import ASSETS, STalign, deformed_source_image, dump, extent_of, gray_png, overlay_png, s2t, weight_png, to_np

CROP_FRACTION = 0.30
OUT = os.path.join(ASSETS, 'mixture')


def main():
    os.makedirs(OUT, exist_ok=True)
    P = C.tutorial_problem()
    xJ, yJ = P['xJ'], P['yJ']
    # remove the 30% of target cells with the largest x (a vertical cut: x > x_cut is gone)
    x_cut = float(np.quantile(xJ, 1 - CROP_FRACTION))
    keep = xJ <= x_cut
    # rasterize the kept cells on the full target's grid (weights 0/1 on all target cells), so
    # every image of this section shares the full target's grid and extent
    XJ, YJ, Jc = STalign.rasterize(xJ, yJ, g=keep.astype(float), dx=30, blur=1.5, draw=0)
    assert np.allclose(XJ, P['XJ']) and np.allclose(YJ, P['YJ'])
    xI_, xJ_ = [P['YI'], P['XI']], [YJ, XJ]
    gray_png(os.path.join(OUT, 'target_cropped.png'), Jc[0], C.VMAX_CELLS_PER_PIXEL)
    gray_png(os.path.join(OUT, 'target_full.png'), P['J'][0], C.VMAX_CELLS_PER_PIXEL)

    cases = [('full', 'none', P['J'], None), ('full', 'em', P['J'], 'em'),
             ('cropped', 'none', Jc, None), ('cropped', 'em', Jc, 'em')]
    res = {}
    for tgt, mname, J, mix in cases:
        t = time.time()
        out = STalign.LDDMM(xI_, P['I'], xJ_, J, mixture=mix, draw=False, num_threads=C.NUM_THREADS)
        secs = time.time() - t
        x, y = s2t(out, P['xI'], P['yI'])
        res[(tgt, mname)] = (out, x, y, secs)
        overlay_png(os.path.join(OUT, f'overlay_{tgt}_{mname}.png'), J[0], deformed_source_image(out, P))
        weight_png(os.path.join(OUT, f'WM_{tgt}_{mname}.png'), to_np(out['WM']))

    _, xr, yr, _ = res[('full', 'none')]
    inside_kept = xr <= x_cut          # source cells that the reference maps into the kept part
    margin = xr <= x_cut - 300.0       # ... at least 300 um from the cut
    rows = []
    for (tgt, mname), (out, x, y, secs) in res.items():
        err = np.hypot(x - xr, y - yr)
        WM = to_np(out['WM'])
        cut_pixels = XJ[None, :] > x_cut
        tissue = P['J'][0] > 0.1*C.VMAX_CELLS_PER_PIXEL
        row = dict(target=tgt, mixture=None if mname == 'none' else mname,
                   overlay=f'mixture/overlay_{tgt}_{mname}.png', WM=f'mixture/WM_{tgt}_{mname}.png',
                   seconds=round(secs, 2),
                   error_vs_full_reference_um=dict(
                       kept_region_mean=float(err[inside_kept].mean()),
                       kept_region_median=float(np.median(err[inside_kept])),
                       kept_region_300um_from_cut_mean=float(err[margin].mean()),
                       cropped_region_mean=float(err[~inside_kept].mean())),
                   fraction_of_source_cells_mapped_into_cropped_region=float(np.mean(x > x_cut)),
                   WM_mean_on_cropped_tissue=float(WM[cut_pixels & tissue].mean()),
                   WM_mean_on_kept_tissue=float(WM[~cut_pixels & tissue].mean()))
        rows.append(row)
        e = row['error_vs_full_reference_um']
        print(f"{tgt:7} mixture={mname:4}: kept-region err {e['kept_region_mean']:6.1f} um "
              f"(>300um from cut {e['kept_region_300um_from_cut_mean']:6.1f}), cropped-region {e['cropped_region_mean']:6.1f}; "
              f"source in cut {row['fraction_of_source_cells_mapped_into_cropped_region']:.3f}; "
              f"WM cut/kept {row['WM_mean_on_cropped_tissue']:.2f}/{row['WM_mean_on_kept_tissue']:.2f} ({secs:.1f}s)",
              flush=True)

    # align_cells (15 um rasters, coarse to fine) for comparison - numbers only
    cells = []
    ac = {}
    for tgt, sel in (('full', np.ones_like(keep)), ('cropped', keep)):
        for mix in (None, 'em'):
            t = time.time()
            o = STalign.align_cells(P['xI'], P['yI'], xJ[sel], yJ[sel], mixture=mix, num_threads=C.NUM_THREADS)
            ac[(tgt, mix)] = (s2t(o, P['xI'], P['yI']), time.time() - t)
    (xa, ya), _ = ac[('full', None)]
    kept_a = xa <= x_cut
    for (tgt, mix), ((x, y), secs) in ac.items():
        err = np.hypot(x - xa, y - ya)
        cells.append(dict(target=tgt, mixture=mix, seconds=round(secs, 2),
                          kept_region_mean_error_um=float(err[kept_a].mean()),
                          cropped_region_mean_error_um=float(err[~kept_a].mean()),
                          fraction_of_source_cells_mapped_into_cropped_region=float(np.mean(x > x_cut))))
        print('align_cells', cells[-1], flush=True)

    dump(os.path.join(OUT, 'summary.json'), dict(
        crop=dict(rule='target cells with x > x_cut removed', x_cut_um=round(x_cut, 1),
                  fraction_removed=float(1 - keep.mean()), n_kept=int(keep.sum()), n_total=int(len(keep))),
        extent_um=extent_of(XJ, YJ), image_shape=list(Jc[0].shape),
        target_full='mixture/target_full.png', target_cropped='mixture/target_cropped.png',
        reference='LDDMM to the full target with mixture=None (default settings, niter=2000)',
        results=rows, align_cells_reference='align_cells to the full target with mixture=None',
        align_cells=cells))


if __name__ == '__main__':
    main()
