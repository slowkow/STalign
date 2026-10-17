# STalign explorer

An interactive page that shows what STalign does at each step, on the MERFISH data of the
tutorial `docs/notebooks/merfish-merfish-alignment.ipynb`. It covers rasterization, the affine
pre-fit, LDDMM at six regularization strengths and the mixture model. Every image and number on
the page comes from STalign fits computed ahead of time, so the page itself runs no Python.

## Files

- `docs/_extra/explorer/index.html`: the page. It is plain HTML, CSS and JavaScript with no
  build step. It loads MathJax from jsDelivr and its fonts from Google Fonts.
- `docs/_extra/explorer/assets/`: the precomputed images (PNG) and data (`data/*.js`).
- `docs/explorer/` (this directory): the scripts that compute them.
  - `rasterize.py`, `regularization.py`, `prefit.py` and `mixture.py` each run one experiment and
    write its images and a `summary.json` into `build/`.
  - `common.py` has their shared helpers.
  - `build_manifest.py` merges the summaries into `build/manifest.json`, with the conventions
    (units, image orientation, colors) written out.
  - `package.py` copies what the page uses into `docs/_extra/explorer/assets/`.
  - `build/` is not committed.

## Viewing and publishing

Open `docs/_extra/explorer/index.html` in a browser; it works straight from disk. The links in
its top bar point to the documentation pages next to it, so they work on the built site, not
from the source tree.

`sphinx-build docs _build` copies `docs/_extra/` into the site (`html_extra_path` in
`docs/conf.py`), so the page is published at `explorer/index.html`, next to the documentation:
https://jef.works/STalign/explorer/ once the docs are deployed.

## Recomputing the images and numbers

From the base directory of the repository, with the C++ engine built
(`python setup.py build_ext --inplace`):

```
docs/explorer/run_all.sh python
```

The Python environment needs torch, numpy, pandas, matplotlib and Pillow. The script refits
everything with the STalign of this checkout, which takes a few minutes, and replaces
`docs/_extra/explorer/assets/`.

Some numbers are also written into the text of `index.html`, for example "186 µm" and "1.4%" in
the mixture section. If the fits change, check the text against `build/manifest.json`.
`regularization.py` reads a fit's intermediate iterations through STalign's private engine
interface (`common.run_frames`). It checks that its last frame equals a plain `STalign.LDDMM`
call, and it may need updating when that interface changes.
