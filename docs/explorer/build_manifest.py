'''Merge the per-section summary.json files into build/manifest.json, with the conventions.'''
import glob
import json
import os
import subprocess
import sys

import common as C
from common import ASSETS, REPO

import STalign as S


def load(name):
    with open(os.path.join(ASSETS, name, 'summary.json')) as f:
        return json.load(f)


def main():
    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    files = sorted(p for p in glob.glob(os.path.join(ASSETS, '**', '*'), recursive=True) if os.path.isfile(p))
    total = sum(os.path.getsize(p) for p in files)
    m = dict(
        title='STalign explorer assets',
        generated_by='docs/explorer/run_all.sh',
        stalign=dict(version=S.__version__, commit=commit, backend=S.backend_info()['auto'],
                     num_threads=C.NUM_THREADS),
        data=dict(
            source='MERFISH mouse brain, Slice 2 Replicate 3 (S2R3), docs/merfish_data, columns center_x, center_y',
            target='MERFISH mouse brain, Slice 2 Replicate 2 (S2R2)',
            source_initial_affine=('the tutorial\'s (docs/notebooks/merfish-merfish-alignment.ipynb): source cells '
                                   'rotated by -45 deg about their centroid and moved so their centroid is on the '
                                   'target\'s; every "source" below means these moved cells unless said otherwise'),
            tutorial_rasters='STalign.rasterize(x, y, dx=30, blur=1.5) for source and target (kernel sd = 2*blur*dx = 90 um)'),
        conventions=dict(
            units='micrometers (um) for every coordinate and distance',
            coordinates=('(x, y) = the cells\' center_x, center_y. JSON points are [x, y] or flat [x0, y0, x1, y1, ...], '
                         'rounded to 0.1 um. Only the internal STalign calls use (row, col) = (y, x) order.'),
            png_orientation=('top row of every PNG = largest y, left column = smallest x (y axis points up, as in the '
                             'tutorial figures after invert_yaxis and in ordinary scatter plots). A PNG with extent_um '
                             '[xmin, xmax, ymin, ymax] covers exactly that rectangle: pixel (row r, col c) is centred at '
                             'x = xmin + (c + 0.5)*dx, y = ymax - (r + 0.5)*dx. To draw JSON points on it: '
                             'px = (x - xmin)/(xmax - xmin)*width, py = (ymax - y)/(ymax - ymin)*height.'),
            overlay_colors=('RGB = (target, deformed source, target), each = clip(value / vmax, 0, 1) with one vmax '
                            'for all overlays (regularization.vmax_cells_per_pixel, the 99.5th percentile of the '
                            'target raster\'s tissue pixels): target only = magenta, source only = green, both = '
                            'white/gray, neither = black. Values are rasterized cell densities (cells per 30 um pixel).'),
            overlay_grid=('every overlay in regularization/, prefit/ and mixture/ is on the target\'s dx=30 grid '
                          '(same extent_um and size), so frames line up exactly when swapped'),
            weight_maps='WM PNGs: grayscale, black = 0, white = 1 (linear), on the same target grid',
            grayscale_rasters=('rasterize/ PNGs: cell density in cells per mm^2 (raster value / dx^2 * 1e6), linear '
                               'grayscale from 0 (black) to rasterize.vmax_cells_per_mm2 (white, clipped)'),
            transforms=('cells mapped source -> target with STalign.transform_points_source_to_target (exact inverse, '
                        'the default); images with transform_image_source_to_target'),
        ),
        rasterize=dict(load('rasterize'), notes=(
            'Target (S2R2) cells as they are in the data (no affine). blur is in pixels; the Gaussian kernel sd is '
            '2*blur*dx um, so the same blur is physically wider at a larger dx (blur_um=... keeps it fixed). Each '
            'image covers the cells\' bounding box expanded by 1.1, so extents differ slightly with dx: draw each at '
            'its extent_um (or simply at the same displayed size; the differences are < 1 pixel of dx=60). The '
            'full-size PNGs are at native resolution (one PNG pixel = one raster pixel: 15 um images are ~650x680); '
            'use CSS image-rendering: pixelated. *_zoom.png: a 1.5 x 1.5 mm window (zoom_extent_um, the pixel '
            'edges actually included) upscaled to 300x300 by nearest neighbour so the pixels are visible. '
            'cells_thumbnail.png: all target cells binned on a 360-px canvas over thumbnail_extent_um, dark = more '
            'cells. cells_target_subsample.json: 6000 random target cells; cells_target_zoom.json: every cell in '
            'the zoom window.')),
        regularization=dict(load('regularization'), notes=(
            'One LDDMM fit per sigma_disp (sigma_disp is LDDMM\'s prior displacement scale in um; sigmaR is derived '
            'from it, larger = more flexible; default 94.03). All other settings are LDDMM\'s defaults (niter 2000, '
            'a=500, nt=3, data-driven step sizes, mixture=None, prefit=\'auto\'). The pre-fit (a coarse affine search, '
            'identical for every sigma_disp) runs first, so iteration 0 is the affine pre-fit result with no '
            'deformation; overlay_tutorial_affine.png is the state before the pre-fit (the tutorial\'s affine '
            'alone). Frame k is the fit after k gradient-descent iterations of one run, exactly what '
            'STalign.LDDMM(..., niter=k) returns (A and v of the k-th iteration\'s forward pass; snapshotted from '
            'the same engine LDDMM uses; check_vs_LDDMM_max_abs_diff compares the last frame with a plain LDDMM '
            'call). The affine part keeps being optimized together with the deformation. '
            'grid.json: lines[k] is a list of polylines (flat [x0,y0,x1,y1,...], target um) for frame k; kinds '
            'says if each line was vertical (v) or horizontal (h). The lines are an axis-aligned square grid '
            '(grid_spacing_um, sampled every grid_sample_step_um) in target coordinates, placed on the source by '
            'the iteration-0 affine (so at iteration 0 it is exactly square), then carried to target space by '
            'frame k\'s full map (affine + deformation, exact inverse). A grid point therefore stays attached to '
            'the same piece of source tissue: the grid shows how the fit has moved and bent the source since '
            'iteration 0. energy.json: per iteration i = 1..niter, E = EM + ER, EM the image matching term, ER the '
            'regularization term (0.5/sigmaR^2 sum_t ||L v_t||^2), evaluated at that iteration\'s forward pass '
            '(frame k shows energies [k-1]). E is not monotone: the matching weights WM are re-estimated at '
            'iteration 50 and every 5 iterations after it, which changes the energy being minimized (with '
            'mixture=None WM starts at 0.5 everywhere and becomes ~1 at iteration 50, so EM about doubles there: '
            'the jump is a change of weights, not a worse alignment); ER is not '
            'comparable across sigma_disp (different sigmaR). Per-frame numbers: moved_since_it0_um (distance '
            'each source cell moved since iteration 0); nonaffine_residual_um (distance of each mapped cell from '
            'the least-squares affine fit of the whole map: the size of the truly nonlinear part); area_ratio_* '
            '(local area change of the map divided by that of the least-squares affine, over target tissue '
            'pixels; 1 = no compression or expansion, p01/p99 = 1st/99th percentiles); log_area_ratio_sd (a '
            'single smoothness/roughness number, 0 for an affine map); folded_fraction (pixels where the map '
            'folds; 0 everywhere here); image_corr / image_rmse (deformed source vs target over target tissue). '
            'sigma_disp 300 and 1000 are still deforming at iteration 2000 (nonaffine residual still growing); '
            'the others have converged by ~400-1000.')),
        prefit=dict(load('prefit'), notes=(
            'The tutorial source rotated by theta_deg counter-clockwise (y up) about rotation_center_um (its '
            'centroid), rasterized (dx=30, blur=1.5) and aligned to the target from the identity with prefit=False '
            'or prefit=True, niter=1000, other settings default. theta*_input.png is the starting state (rotated '
            'source, no transform). error_um: per source cell, the distance between where this fit puts it and '
            'where the reference fit of the unrotated source (theta=0, prefit=True) puts it; since the rotation is '
            'known, a correct fit must agree with the reference. success = median error < 50 um. The pre-fit tries '
            '12 starting rotations 30 deg apart, so multiples of 30 deg are exactly one of its starts; 15, 45, ... '
            'are halfway between starts. prefit_best_start_deg: the start that won, as STalign reports it (in its '
            'own sign convention: 330 = -30 deg; ties within 1% go to the start nearest 0, so a 30 deg rotation can '
            'be won by the 0 start and refined); prefit_rotation_deg: rotation of the affine the pre-fit hands to '
            'LDDMM, measured on A, which maps target to source coordinates (it comes out at theta + ~2 deg, the 2 '
            'deg being the residual rotation between the tutorial\'s affine and the best affine).')),
        mixture=dict(load('mixture'), notes=(
            'The target with the 30% of its cells of largest x removed (a vertical cut at crop.x_cut_um), '
            'rasterized on the full target\'s grid. LDDMM (default settings, niter 2000) to the full and to the '
            'cropped target, with mixture=None (LDDMM\'s default) and mixture=\'em\' (align_cells\' default). Each '
            'overlay shows the target it was fitted to (magenta). WM_*.png: the matching weight of each target '
            'pixel at the end of the fit (white = matched, black = treated as background/artifact). With mixture=None '
            'WM is 0.99999 everywhere on these images (the fixed variances sigmaM=1, sigmaB=2, sigmaA=5 never favour '
            'the background class at these intensities, 0-3.5 cells per pixel), so its WM PNGs are plain white: that '
            'is the result, not a rendering error. Errors are per '
            'source cell vs the reference fit (full target, mixture=None): kept_region = cells the reference maps '
            'to x <= x_cut (where the cropped target still has tissue), cropped_region = the others (no target '
            'tissue there to match, so any error there is extrapolation). '
            'fraction_of_source_cells_mapped_into_cropped_region: ~0.3 is right (the source still has that tissue, '
            'it should hang over the cut); much less means the fit squeezed the source into the remaining target. '
            'align_cells: the same comparison with STalign.align_cells (numbers only).')),
        size_bytes=total, n_files=len(files) + 1,
    )
    with open(os.path.join(ASSETS, 'manifest.json'), 'w') as f:
        json.dump(m, f, indent=1)
    print(f'manifest: {len(files)+1} files, {total/1e6:.2f} MB (before manifest)')


if __name__ == '__main__':
    sys.exit(main())
