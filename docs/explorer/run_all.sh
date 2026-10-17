#!/usr/bin/env bash
# Recompute every image and number of the STalign explorer (docs/_extra/explorer/index.html).
#
# Usage: docs/explorer/run_all.sh [python]
#
# The python (default: python on PATH) needs torch, numpy, pandas, matplotlib and Pillow, and
# imports STalign from this checkout; build its C++ engine first (python setup.py build_ext
# --inplace), or the fits take several times longer.  The scripts write everything into
# docs/explorer/build/ (not committed); package.py then copies what the page uses into
# docs/_extra/explorer/assets/.  It takes a few minutes with the C++ engine.
set -euo pipefail
PY="${1:-python}"
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 STALIGN_NUM_THREADS=8
rm -rf build && mkdir -p build
for s in rasterize regularization prefit mixture; do
  echo "== $s"; "$PY" -W ignore "$s.py"
done
"$PY" -W ignore build_manifest.py
"$PY" package.py
