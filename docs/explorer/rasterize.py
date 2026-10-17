'''rasterize/: the target cells (S2R2) rasterized at several pixel sizes and kernel widths.'''
import os

import numpy as np
from PIL import Image

import common as C
from common import ASSETS, STalign, dump, extent_of, gray_png, r1

DXS = [15, 30, 60]
BLURS = [0.5, 1.0, 2.0]
ZOOM = dict(cx=5200.0, cy=5900.0, half=750.0)  # 1.5 x 1.5 mm window, um
ZOOM_PX = 300
OUT = os.path.join(ASSETS, 'rasterize')
N_SUB = 6000


def main():
    os.makedirs(OUT, exist_ok=True)
    xI, yI, xJ, yJ = C.load_cells()
    imgs = {}
    for dx in DXS:
        for b in BLURS:
            X, Y, M = STalign.rasterize(xJ, yJ, dx=dx, blur=b, draw=0)
            imgs[(dx, b)] = (X, Y, M[0] / dx**2 * 1e6)  # cells per mm^2
    # shared display scale: 99.5th percentile of tissue pixels of the tutorial-like image (dx=30, blur=1)
    ref = imgs[(30, 1.0)][2]
    vmax = float(np.percentile(ref[ref > 0.05*ref.max()], 99.5))
    items = []
    for (dx, b), (X, Y, D) in imgs.items():
        name = f'dx{dx:02d}_blur{b:g}'
        gray_png(os.path.join(OUT, name + '.png'), D, vmax)
        # zoom window, nearest-neighbour upscaled so the pixels stay visible
        cols = (X >= ZOOM['cx']-ZOOM['half']) & (X <= ZOOM['cx']+ZOOM['half'])
        rows = (Y >= ZOOM['cy']-ZOOM['half']) & (Y <= ZOOM['cy']+ZOOM['half'])
        Z = D[np.ix_(rows, cols)]
        g = np.clip(Z/vmax, 0, 1)[::-1]
        Image.fromarray((g*255+0.5).astype(np.uint8), 'L').resize((ZOOM_PX, ZOOM_PX), Image.NEAREST).save(
            os.path.join(OUT, name + '_zoom.png'), optimize=True)
        tissue = D[D > 0.05*D.max()]
        items.append(dict(dx_um=dx, blur_px=b, kernel_sd_um=2*b*dx, png=f'rasterize/{name}.png',
                          zoom_png=f'rasterize/{name}_zoom.png', shape=list(D.shape), extent_um=extent_of(X, Y),
                          zoom_extent_um=extent_of(X[cols], Y[rows]), zoom_shape=list(Z.shape),
                          max_cells_per_mm2=float(D.max()), p995_tissue_cells_per_mm2=float(np.percentile(tissue, 99.5)),
                          saturated_fraction_of_tissue=float(np.mean(tissue > vmax))))
        print(name, D.shape, f'max {D.max():.0f} /mm2, saturated {np.mean(tissue > vmax):.3f}')

    # raw cell positions: scatter thumbnail (all cells) + subsample JSON
    X, Y, _ = imgs[(30, 1.0)]
    ext = extent_of(X, Y)
    W = 360
    H = int(round(W*(ext[3]-ext[2])/(ext[1]-ext[0])))
    acc = np.zeros((H, W))
    c = ((xJ-ext[0])/(ext[1]-ext[0])*W).astype(int).clip(0, W-1)
    r = ((ext[3]-yJ)/(ext[3]-ext[2])*H).astype(int).clip(0, H-1)  # top row = ymax
    np.add.at(acc, (r, c), 1)
    g = 1 - np.clip(acc/np.percentile(acc[acc > 0], 95), 0, 1)  # dark dots on white
    Image.fromarray((g*255+0.5).astype(np.uint8), 'L').save(os.path.join(OUT, 'cells_thumbnail.png'), optimize=True)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(xJ), N_SUB, replace=False)
    zm = (np.abs(xJ-ZOOM['cx']) <= ZOOM['half']) & (np.abs(yJ-ZOOM['cy']) <= ZOOM['half'])
    dump(os.path.join(OUT, 'cells_target_subsample.json'), dict(
        n_total=int(len(xJ)), n=N_SUB, x=r1(xJ[sub]), y=r1(yJ[sub])))
    dump(os.path.join(OUT, 'cells_target_zoom.json'), dict(
        window_um=[ZOOM['cx']-ZOOM['half'], ZOOM['cx']+ZOOM['half'], ZOOM['cy']-ZOOM['half'], ZOOM['cy']+ZOOM['half']],
        n=int(zm.sum()), x=r1(xJ[zm]), y=r1(yJ[zm])))
    dump(os.path.join(OUT, 'summary.json'), dict(
        vmax_cells_per_mm2=vmax, images=items, zoom=ZOOM, thumbnail='rasterize/cells_thumbnail.png',
        thumbnail_extent_um=ext, thumbnail_shape=[H, W], subsample='rasterize/cells_target_subsample.json',
        zoom_cells='rasterize/cells_target_zoom.json'))
    print('vmax', vmax, 'zoom cells', zm.sum())


if __name__ == '__main__':
    main()
