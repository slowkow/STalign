# How STalign works

This page explains, at a high level, what STalign computes when it aligns two tissue sections:
the ideas in plain language, each with its key equation, and where it lives in the code. It is
written for a first-year graduate student who wants to use STalign with understanding, and for
developers who want to build on it. The method is described in the paper
([Clifton et al. 2023](https://www.nature.com/articles/s41467-023-43915-7)); this page follows
what the code of version 2 does, which differs from the paper in a few places that are pointed
out below.

The figures use the data of the
[MERFISH tutorial](notebooks/merfish-merfish-alignment.ipynb): the cells of one coronal section of a
mouse brain (slice 2, replicate 3; the *source*, always blue) are aligned to the cells of a
neighbouring replicate (slice 2, replicate 2; the *target*, always vermillion). The script
`docs/figures/make_figures.py` draws them with STalign's own functions; run it to reproduce them.

To try the steps yourself, open the <a href="explorer/index.html">STalign explorer</a>, an
interactive page with the same data: change the pixel size and blur, rotate the source with and
without the affine pre-fit, step through a fit for six regularization strengths, and cut tissue
from the target to see what the mixture model does.

```{figure} figures/overview.png
:alt: Four panels: two overlapping cell clouds; their density images; the deformed source image with a curved grid; the aligned cells.
:width: 100%

**What STalign does.** (a) Two sets of cell positions in different coordinate systems.
(b) Each set is turned into an image of cell density. (c) STalign finds an affine map $A$ and a
smooth deformation $\varphi$ that make the deformed source image (blue) look like the target
image (vermillion; overlap is grey). The grid shows how straight lines of the source (every
1 mm) land in the target. (d) The same map moves every source cell into the target's
coordinates.
```

## The problem

We have two lists of cell positions, $x_i$ in the source and $y_j$ in the target, measured on
different sections, often rotated, shifted, stretched and locally distorted relative to each other
(tissue folds, tears, uneven shrinkage). We want a map $\Phi$ from source coordinates to target
coordinates such that every source cell lands where the corresponding tissue is in the target.

There is no one-to-one correspondence between the cells of two sections, so STalign does not
match cells. It matches *tissue*: it turns each set of cells into an image of cell density and
deforms one image until it looks like the other. The method is LDDMM (large deformation
diffeomorphic metric mapping), a registration method from computational anatomy
([Beg et al. 2005](https://doi.org/10.1023/B:VISI.0000043755.93987.aa)), with an intensity
transformation and a mixture model for missing tissue
([Tward et al. 2020](https://doi.org/10.3389/fnins.2020.00052)).

Three requirements shape it:

- **Smooth and invertible.** Neighbouring cells should stay neighbours, and nothing should fold
  over itself. Maps with these properties (smooth, with a smooth inverse) are called
  *diffeomorphisms*.
- **Large deformations.** Sections can differ by large rotations and displacements of hundreds of
  micrometers, so small-displacement approximations are not enough.
- **Different contrasts.** The target can be another kind of image (a histology image, an atlas),
  so the method must tolerate different brightness and contrast, and tissue missing on one side.

## From cells to images: rasterization

`rasterize` places a small Gaussian bump on every cell and adds them up on a grid of pixels of
size $dx$. Each cell adds a total of 1 (or its weight $w_i$, for example the expression of a
gene), so a pixel value is a number of cells:

$$
I(p) = \sum_i w_i\,\frac{k(p - x_i)}{\sum_{p' \in W_i} k(p' - x_i)},
\qquad k(d) = \exp\!\left(-\frac{|d|^2}{2\sigma^2}\right),
\qquad \sigma = 2\cdot \texttt{blur}\cdot dx .
$$

The kernel is evaluated on a window $W_i$ of $\pm\lceil 4\,\texttt{blur}\rceil$ pixels (about
$\pm 2\sigma$) around the cell, and normalized to sum 1 on it. The argument `blur` is in pixels;
`blur_um` gives the kernel's standard deviation $\sigma$ in micrometers instead, so the kernel
keeps its physical width when you change $dx$. Several blurs give several image channels.

```{figure} figures/rasterization.png
:alt: Cells of a coronal brain section, a zoom on 1.5 mm, and the density images of the zoom at three pixel sizes and three kernel widths.
:width: 100%

**Rasterization.** (a, b) The source cells and a 1.5 mm zoom. (c) Density images of the zoom at
pixel sizes 15, 30 and 60 µm with the same kernel. (d) At 30 µm pixels, kernels with standard
deviations of 15, 45 and 120 µm. Values are cells per mm².
```

The two choices trade detail against noise and speed. A kernel narrower than the typical spacing
of cells (here about 15 µm) gives a speckled image whose details do not exist in the other
section; a kernel much wider than the structures you want to align blurs them away. The cost of
each LDDMM iteration grows with the number of target pixels, so halving $dx$ makes the fit about
four times slower. The MERFISH tutorial uses $dx = 30$ µm and `blur=1.5` ($\sigma = 90$ µm);
`align_cells` uses $dx = 15$ µm and $\sigma = 30$ µm and then works coarse to fine (see
{ref}`below <one-call-align-cells>`).

## The transformation: an affine map and a flow

STalign writes the source-to-target map as a deformation followed by an affine map,

$$
\Phi(x) = A\,\varphi(x),
$$

where $A$ is an affine map (a $3 \times 3$ matrix in homogeneous coordinates with linear part
$L$ and translation $T$: rotation, scaling, shear and shift) and $\varphi$ is a diffeomorphism of
the source space. The affine part does the large, global motion; $\varphi$ does the local
corrections.

$\varphi$ is not stored as a displacement field. It is built by following a *velocity field*
$v_t$ for a short time, as particles follow a flow of water: textbook LDDMM defines $\varphi$ by
$\frac{d}{dt}\varphi_t(x) = v_t(\varphi_t(x))$ for $t \in [0, 1]$, and if $v_t$ is smooth, the
result is a diffeomorphism. STalign discretizes the time interval into $n_t$ steps (`nt`,
default 3), with one velocity field $v_0, \dots, v_{n_t-1}$ per step. Each field lives on a
regular grid (`xv`) of spacing $a/2$ (250 µm by default) that covers the source image enlarged by
the factor `expand` (2 in `LDDMM`), and is interpolated bilinearly in between. It is zero outside
that grid, so far from the tissue the map is purely affine.

```{figure} figures/transformation.png
:alt: A square grid over the source tissue is bent in three steps by arrows of velocity, then rotated and stretched into the target space.
:width: 100%

**The transformation model, with the fitted velocity fields.** (a) A square grid on the source.
(b–d) Each step moves the grid along one velocity field (arrows: the displacement $v_t/n_t$ of
the step, drawn 3 times longer; light lines: the grid before the step). After the three steps
the grid is $\varphi(x)$. (e) The affine map $A$ then takes it into the target space. On this
fit, $\varphi$ moves points of the tissue by 164 µm (median) and up to 722 µm.
```

The fit itself works in the other direction, from the target to the source, because to compare
images we need, for every target pixel $y$, the point of the source it corresponds to. This
*backward map* is computed by stepping back through the velocity fields:

$$
x^{(n_t)} = A^{-1} y, \qquad
x^{(t)} = x^{(t+1)} - \tfrac{1}{n_t}\,v_t\!\left(x^{(t+1)}\right)
\quad (t = n_t-1, \dots, 0), \qquad
\varphi^{-1}(A^{-1} y) = x^{(0)} .
$$

Each step is invertible as long as the velocity changes slowly compared with the step (its
spatial derivative times $1/n_t$ is below 1), which the regularization below enforces in practice.

## What counts as a good alignment: the energy

STalign finds $A$ and $v$ by minimizing an energy with two terms (three with landmarks):

$$
E(A, v) = E_M + E_R \;(+\, E_P).
$$

**Matching term.** For each target pixel $y$, take the source image at the backward map, apply
a contrast function $f$, and compare with the target:

$$
E_M = \sum_{y \in \text{target pixels}} W_M(y)\,
\frac{\left| f\!\left(I\!\left(\varphi^{-1}(A^{-1}y)\right)\right) - J(y) \right|^2}{2\sigma_M^2} .
$$

- $I(\cdot)$ is evaluated by bilinear interpolation (outside the source image, its border values
  are repeated).
- $f$ lets the two images differ in brightness and contrast. In the code it is linear in the
  channels of the source, separately for each target channel:
  $f(z) = c_0 + \sum_k c_k z_k$. The coefficients are refitted at every iteration by weighted
  least squares (with a small ridge penalty of 0.1) and treated as fixed when computing the
  gradient. For a nonlinear contrast, pass powers of the source as extra channels, as the 3D
  tutorials do with $(I - \bar I)^2$.
- $W_M(y) \in [0, 1]$ is the *matching weight* of the pixel (see
  [the mixture model](#matching-weights-and-missing-tissue)); $\sigma_M$ (`sigmaM`, default 1)
  sets how much a mismatch costs. $E_M$ is a sum, not a mean, over pixels, so it grows with the
  number of pixels.

```{figure} figures/objective.png
:alt: The target image, the contrast-mapped deformed source, and the residual maps before and after the deformation.
:width: 100%

**The matching term.** (a) The target $J$. (b) The deformed source after the contrast
function, at the end of the fit. (c, d) The residual $f(\cdot) - J$ with the affine map only
(after the pre-fit) and after 2000 iterations: blue where the deformed source has more cells,
vermillion where the target has more. The deformation removes the large structured errors at
the edges; what is left is mostly the cell-scale difference between two sections.
```

**Regularization term.** Without a penalty, the velocity could become arbitrarily wiggly to fit
noise. The penalty measures roughness with a differential operator:

$$
E_R = \frac{1}{2\sigma_R^2} \sum_{t=0}^{n_t-1} \int \left| L\,v_t(x) \right|^2 dx,
\qquad L = \left(\mathrm{Id} - a^2 \Delta\right)^p ,
$$

with $\Delta$ the Laplacian (computed with finite differences on the velocity grid, in Fourier
space), $a$ a length scale (`a`, default 500 µm) and $p$ a power (`p`, default 2). Velocity
fields that vary over distances shorter than about $a$ are expensive. $\sigma_R$ (`sigmaR`) sets
the overall strength. Unlike the textbook integral over $t \in [0, 1]$, the code sums over the
$n_t$ steps without a factor $1/n_t$, so changing `nt` at a fixed `sigmaR` changes the strength
of the regularization (the default `sigmaR` compensates for this).

**Landmark term.** With corresponding points $p_i$ (source) and $q_i$ (target), `LDDMM` adds
$E_P = \frac{1}{2\sigma_P^2} \sum_i \left| A\varphi(p_i) - q_i \right|^2$, where $A\varphi$ is the
exact inverse of the backward map (see [below](#two-directions-moving-images-and-moving-cells)),
so the landmarks are matched under the same map that later moves the cells.

## The regularization strength

The regularizer has a natural *smoothing kernel* $K = (L^\dagger L)^{-1}$, a bell-shaped kernel
of width about $a$ (figure, panel a). In Fourier space it is simply
$\hat K(\omega) = 1 / |\hat L(\omega)|^2$. It plays two roles: it defines which velocity fields
are "cheap" (sums of kernel-sized bumps), and it smooths the gradient during the optimization
(next section).

How strong the penalty should be depends on the units: $E_M$ grows with the number of pixels and
with the squared intensity of the images. Version 2 therefore sets $\sigma_R$ from a
displacement scale in micrometers, `sigma_disp` (default 94.03 µm), unless you give `sigmaR`:

$$
\sigma_R = \sigma_{disp}\,\sqrt{C_d\,\pi}\;a^{d/2}\,\sqrt{n_t}\;\frac{dx}{30\ \mu\mathrm{m}}\;\frac{\sigma_M}{s_J},
\qquad C_2 = 12,\; C_3 = 64 ,
$$

with $d$ the dimension, $dx$ the target pixel size measured in source units, and $s_J$ the
typical tissue intensity of the target. The first factors come from a probabilistic reading of
$E_R$: if the velocity were random with density proportional to $e^{-E_R}$, each of its
components would have variance $\sigma_R^2 / (C_d \pi a^d)$ at every point ($p = 2$), and the
average of $n_t$ independent steps would have standard deviation $\sigma_{disp}$. The last two
factors scale $1/\sigma_R^2$ like $E_M$ scales with the pixel size and the intensities, so the
balance between the two terms does not depend on them. In practice `sigma_disp` is a knob for
how far the deformation may move tissue: the default reproduces the old default
`sigmaR=5e5` for 30 µm images whose tissue intensity is about `sigmaM`.

```{figure} figures/regularization.png
:alt: Kernel profiles for three values of a, and the deformed grid and overlay for strong, default and weak regularization.
:width: 100%

**Regularization.** (a) The smoothing kernel through its center for three length scales $a$.
(b–d) The same alignment with `sigma_disp` = 20, 94 (default) and 500 µm. With strong
regularization the deformation is too stiff to fix the edges (blue and vermillion fringes);
weaker regularization lets the grid bend more. Below each panel: the resulting `sigmaR` and how
far $\varphi$ moves the source cells.
```

## Finding the best map: gradient descent

STalign minimizes $E$ by gradient descent, by default for 2000 iterations in 2D (`niter`). Each
iteration has a forward pass (backward map of every target pixel, deformed source, contrast fit,
residual, energy), a backward pass (the gradients of $E$, by the chain rule through the
interpolations and the $n_t$ steps), and an update:

$$
\begin{aligned}
L &\leftarrow L - \tfrac{\epsilon_L}{10}\,\frac{\partial E}{\partial L}, \qquad
T \leftarrow T - \tfrac{\epsilon_T}{10}\,\frac{\partial E}{\partial T}, \\
v_t &\leftarrow v_t - \epsilon_V \left( K * \nabla_{v_t} E_M + \frac{\Delta V}{\sigma_R^2}\, v_t \right) .
\end{aligned}
$$

- The velocity update is the gradient of $E$ in the metric defined by $L$ (a Sobolev
  gradient), as in LDDMM: the plain gradient of the matching term, which is as rough as the
  images, is smoothed with $K$. For the regularizer, smoothing its gradient with $K$ gives exactly
  $(\Delta V / \sigma_R^2)\,v_t$, with $\Delta V$ the area of a velocity grid cell, so each update
  also shrinks $v$ a little.
- The affine steps are divided by 10 while the deformation is being fitted, which is from the
  first iteration unless `diffeo_start` is set.

**Step sizes.** In 2D, version 2 chooses $\epsilon_L, \epsilon_T, \epsilon_V$ (`epL`, `epT`,
`epV`) from the images unless you give them. It scales the steps of version 1 by
$1.1\,\sigma_M^2 / G$, where

$$
G = \Big(\sum_c \operatorname{Var} J_c\Big)\;
\operatorname{mean}_k \frac{\sum_y \big|\nabla (I\circ A^{-1})_k(y)\big|^2}{\operatorname{Var}\,(I \circ A^{-1})_k}
$$

estimates the curvature of $E_M$ (sums over target channels $c$, source channels $k$ and target
pixels $y$, with the source under the starting affine). It then caps the steps where gradient
descent would be unstable. Gradient descent on a quadratic of curvature $c$ is stable for
$\epsilon c < 2$ and decreases the energy at every step for $\epsilon c \le 1$, so
$\epsilon_V \le 1 / (\lambda_v + \Delta V/\sigma_R^2)$, with $\lambda_v$ the largest eigenvalue of
the $K$-smoothed curvature of $E_M$ (and $E_P$) in $v$ at the starting affine, found by subspace
iteration; and $\epsilon_L, \epsilon_T$ are scaled together so that the largest eigenvalue of the
step-scaled affine curvature is at most 0.5. In 3D the steps are fixed (`epL=1e-6`, `epT=10`,
`epV=1e3`, capped for the regularizer), and an *affine step control* halves the step of an affine
parameter whose gradient changes sign and lets it grow back by 5% per iteration.

**Affine pre-fit.** Gradient descent only finds the nearest minimum, and a section rotated by
30° is far from it. Before LDDMM, version 2 runs a coarse search (`prefit='auto'`): the initial
affine is rotated about the target tissue's centroid in 12 steps of 30°, each start is fitted as
an affine map on images mean-pooled to about 80 pixels across, and the best one is refined at
twice that resolution and handed to LDDMM. In the overview example the source starts rotated by
about 45° and no initial guess is given; the pre-fit finds the rotation. In 3D the starts tilt
the slice plane by ±6° about two axes and shift it by ±300 µm. The pre-fit is skipped with
landmarks or a given velocity field, and it can fail on partial sections (as in the next
section, where we start from the tutorial's initial rotation instead). Give a rough initial
`L` and `T` (for example from a few landmarks with `L_T_from_points`) when in doubt.

```{figure} figures/optimization.png
:alt: Energy curves over 2000 iterations and five overlays of source and target from the start to the end of the fit.
:width: 100%

**Optimization.** (a) The energy and its two terms. The matching weights $W_M$ start at 0.5 and
are re-estimated every 5 iterations from iteration 50, which raises them to about 1 here; $E_M$
is a weighted sum, so it jumps up there even though the alignment keeps improving. $E_R$ grows as
the deformation builds up. (b–f) Overlays of the deformed source (blue) and the target
(vermillion; grey where they agree): the start, after the affine pre-fit, and after 20, 200 and
2000 iterations.
```

## Matching weights and missing tissue

Some target pixels should not be matched at all: tissue that is missing in one section,
bubbles, debris. STalign treats every target pixel as coming from one of three classes, and
weights it by the probability of the *matching* class:

$$
W_M(y) = \frac{\pi_M\,\mathcal{N}\!\left(J(y);\, f(\cdot),\, \sigma_M^2\right)}
{\pi_M\,\mathcal{N}\!\left(J(y);\, f(\cdot),\, \sigma_M^2\right)
+ \pi_A\,\mathcal{N}\!\left(J(y);\, \mu_A,\, \sigma_A^2\right)
+ \pi_B\,\mathcal{N}\!\left(J(y);\, \mu_B,\, \sigma_B^2\right)},
$$

where $f(\cdot)$ is the contrast-mapped deformed source at $y$, *background* (B) is a constant
intensity $\mu_B$, *artifact* (A) another constant $\mu_A$, and $\pi$ are the class
proportions; $W_A$ and $W_B$ are the other two fractions. This is the E-step of an
expectation–maximization algorithm, interleaved with the gradient descent.

- **Default (`mixture=None`, as in version 1).** The weights start at
  $W_M, W_A, W_B = 0.5, 0.1, 0.4$. Every 5 iterations the class means $\mu_A, \mu_B$ are
  re-estimated as weighted means of $J$ (unless `muA`, `muB` are given), and from iteration 50
  the weights are recomputed with the fixed widths `sigmaM`, `sigmaA` (5) and `sigmaB` (2).
  With these widths, on count images the matching class usually wins everywhere ($W_M \approx 1$
  on the complete MERFISH target).
- **`mixture='em'` (version 2).** $W_M$ starts at 1. From iteration 50, every 5 iterations, the
  matching variance is estimated from the residual,
  $v_M = \sum_y W_M |r(y)|^2 / (C \sum_y W_M)$ with $C$ the number of channels, the background
  variance is tied to it ($(\sigma_B/\sigma_M)^2 v_M$), $\mu_B$ is the median of the target's
  border, and $\pi_M$ is kept at least 0.5. Only the weights change; $E_M$ still uses $\sigma_M$.
  Where the target looks like background but the deformed source has tissue, $W_M$ drops to about
  0, so the fit no longer squeezes the source to cover the gap.

```{figure} figures/mixture.png
:alt: A target with its right part removed, the matching weight maps with and without the mixture model, and the resulting alignments.
:width: 100%

**Missing tissue.** (a) The target with all cells right of x = 6.3 mm removed. (b, c) The
matching weights after the fit. With `mixture='em'` the region where the source has tissue but
the target does not gets weight about 0 (dark), and so do a few ventricles and edges.
(d) For reference, the fit to the complete target. (e) Without the mixture model the source is
compressed into the remaining tissue; (f) with it, the source extends past the cut, as it
should. The source cells that the reference puts left of the cut are 80 µm (median) from their
reference positions in (e) and 40 µm in (f). Both partial fits start from the tutorial's 45°
initial rotation; from the identity, the affine pre-fit picks a wrong orientation for this
partial target.
```

The mixture model is not always better: CHANGES.md reports that it can make targets of uneven
cell density worse. `align_cells` uses it by default; turn it off with `mixture=None` when the
target section is complete.

## Two directions: moving images and moving cells

The fitted map is used in both directions, and they are computed differently.

- **Images** are moved by *pulling back*: to draw the source in the target's coordinates
  (`transform_image_source_to_target`), each target pixel $y$ looks up the source at
  $\varphi^{-1}(A^{-1}y)$. This only needs the backward map, the one the fit computes.
- **Cells** are moved by *pushing forward*: a source cell at $x$ goes to $A\varphi(x)$
  (`transform_points_source_to_target`). This needs the inverse of the backward map.

Inverting each backward step gives an implicit equation for the forward steps,

$$
x_0 = x, \qquad
x_{t+1} = x_t + \tfrac{1}{n_t}\, v_t(x_{t+1}) \quad (t = 0, \dots, n_t - 1), \qquad
A\varphi(x) = A\,x_{n_t},
$$

which version 2 solves by fixed-point iteration (3 to 9 sweeps per step on the tutorials).
Earlier versions took one explicit step, $x_{t+1} = x_t + v_t(x_t)/n_t$, which is not the exact
inverse; `exact_inverse=False` still gives it.

The other two functions follow the same logic: `transform_points_target_to_source` applies the
backward map, and `transform_image_target_to_source` samples the target at the forward map.

```{figure} figures/maps.png
:alt: A lattice of target pixels and the curved lattice of the points they sample in the source; source cells and where they land in the target; a histogram of round-trip errors.
:width: 100%

**Backward and forward maps.** (a, b) A lattice of target pixels and the points of the source
image they sample. (c, d) Source cells and where the forward map puts them. (e) Mapping every
source cell forward and back again: with the exact inverse the round trip returns to within
$10^{-5}$ µm; with the forward Euler approximation of earlier versions the error is 7 µm
(median) and up to 150 µm on this fit.
```

## Aligning a slice to a 3D atlas

`LDDMM_3D_to_slice` aligns a 3D volume, typically an atlas such as the Allen Common Coordinate
Framework, to a 2D section. The target image is regarded as the plane $z = 0$ of a 3D space: its
pixel $(y_1, y_2)$ is the point $(0, y_1, y_2)$. The map has the same form as in 2D, with a
$4 \times 4$ affine $A$ (which places, tilts and scales the plane in the atlas) and a 3D
diffeomorphism $\varphi$ built from 3D velocity fields. The matching term samples the atlas on
the curved surface that the target plane maps to:

$$
E_M = \sum_{(y_1, y_2)} W_M\,
\frac{\left| f\!\left(I\!\left(\varphi^{-1}(A^{-1}(0, y_1, y_2))\right)\right) - J(y_1, y_2) \right|^2}{2\sigma_M^2} .
$$

The regularizer is the 3D version (over all three axes; version 2 fixed a bug that applied it
over two), `sigmaR` defaults to the 3D formula above, the pre-fit searches over tilted and
shifted planes, and the affine step control is on. There is no landmark term and no
`mixture='em'` in 3D. `analyze3Dalign` then maps each cell of the slice into the atlas with the
backward map and reads its region label.

```{figure} figures/slice3d.png
:alt: A box representing the atlas volume with an oblique plane carrying the target slice; the target slice; the atlas resampled on the plane.
:width: 100%

**Slice to atlas (schematic).** (a) The atlas volume $I$ (box) and a slice plane
$A^{-1}(\{z = 0\})$ inside it, carrying the target slice. The plane here is illustrative
(coronal, tilted by 6° and 4°), not a fitted result. (b) The target slice $J$, a MERFISH
section. (c) The Allen Nissl atlas sampled on that plane, $I(A^{-1}(0, y_1, y_2))$: the image
that the matching term compares with (b) before the deformation.
```

(one-call-align-cells)=
## One call: `align_cells`

`align_cells(xI, yI, xJ, yJ)` runs the steps above for two sets of cells. It rasterizes both at
$dx = 15$ µm with a kernel of $\sigma = 30$ µm, then runs `LDDMM` coarse to fine: 500 iterations
on images mean-pooled by 4 (60 µm pixels), then 500 more on images pooled by 2 (30 µm), each
level starting from the previous level's $A$ and $v$ on the same velocity grid, with
`mixture='em'`. The coarse level is fast and finds the large motions; the finer level refines
them. Its result works with all the transform functions.

## Where this lives in the code

| Concept | Functions | Module |
|---|---|---|
| Rasterization | `rasterize`, `rasterize_with_signal`, `rasterizePCA` | `STalign/_raster.py`, `STalign/STalign.py` |
| One-call cell alignment | `align_cells` | `STalign/_cells.py` |
| Public fitting functions | `LDDMM`, `LDDMM_3D_to_slice` | `STalign/_lddmm.py` |
| Problem set-up: velocity grid (spacing $a/2$), kernel $\hat K$ and $|\hat L|^2$, mixture options | `prepare_problem`, `velocity_grid`, `kernels`, `mixture_options` | `STalign/_lddmm_common.py` |
| Data-driven defaults: pre-fit, $\sigma_R$, step sizes | `resolve_defaults`; `affine_prefit`, `physical_sigmaR`, `curvature_steps`, `stable_steps` | `STalign/_lddmm_common.py`, `STalign/_prefit.py` |
| The iteration: forward pass, gradient, Sobolev smoothing, weights, mixture EM | `TorchEngine` (`_Model`, `_Smooth2D`, `_Smooth3D`) | `STalign/_lddmm_torch.py` |
| The same iteration in C++ (CPU, float64) | `CppEngine` | `STalign/_lddmm_cpp.py`, `STalign/_cpp/` |
| Choice of engine | `select_engine`, `backend_info` | `STalign/_lddmm_common.py` |
| Backward map, exact inverse, moving images and points | `build_transform`, `build_transform3D`, `transform_points_*`, `transform_image_*` | `STalign/_transforms.py` |
| Atlas coordinates and labels of cells | `analyze3Dalign` | `STalign/_transforms.py` |
| Reference implementation of one iteration (tests) | | `tests/reference.py` |

Both engines compute the same iterations up to rounding; `tests/` checks them against
`tests/reference.py`, a plain autograd implementation that is the easiest place to read the
algorithm. Coordinates in `LDDMM` are in (row, column) order, that is `[Y, X]`, and velocity
fields have shape `(nt, *grid, d)` with components in the same order.

### Notation

| Symbol | Meaning | Argument (default) |
|---|---|---|
| $I$, $J$ | source and target images (channels first) | `I`, `J`, on pixel axes `xI`, `xJ` |
| $A = (L, T)$ | affine map, source to target | `L`, `T` or `A` (identity, then pre-fit) |
| $v_t$, $n_t$ | velocity fields and number of time steps | `v`, `xv`; `nt` (3) |
| $a$, $p$ | length scale and power of $L = (\mathrm{Id} - a^2\Delta)^p$ | `a` (500), `p` (2) |
| $\sigma_M$ | matching noise level | `sigmaM` (1) |
| $\sigma_R$ | regularization strength | `sigmaR` (from `sigma_disp` = 94.03 µm) |
| $\sigma_A$, $\sigma_B$, $\mu_A$, $\mu_B$ | widths and means of the artifact and background classes | `sigmaA` (5), `sigmaB` (2), `muA`, `muB` (estimated) |
| $\sigma_P$ | landmark noise level | `sigmaP` (20) |
| $\epsilon_L$, $\epsilon_T$, $\epsilon_V$ | step sizes | `epL`, `epT`, `epV` (from the images in 2D) |
| $W_M$, $W_A$, $W_B$ | per-pixel class weights | returned as `WM`, `WA`, `WB` |

## Further reading

- K. Clifton, M. Anant, G. Aihara, L. Atta, O. K. Aimiuwu, J. M. Kebschull, M. I. Miller,
  D. Tward, J. Fan. STalign: Alignment of spatial transcriptomics data using diffeomorphic metric
  mapping. *Nature Communications* 14, 8123 (2023).
  [doi:10.1038/s41467-023-43915-7](https://doi.org/10.1038/s41467-023-43915-7)
- M. F. Beg, M. I. Miller, A. Trouvé, L. Younes. Computing large deformation metric mappings via
  geodesic flows of diffeomorphisms. *International Journal of Computer Vision* 61, 139–157
  (2005). [doi:10.1023/B:VISI.0000043755.93987.aa](https://doi.org/10.1023/B:VISI.0000043755.93987.aa)
- M. I. Miller, A. Trouvé, L. Younes. On the metrics and Euler–Lagrange equations of
  computational anatomy. *Annual Review of Biomedical Engineering* 4, 375–405 (2002).
  [doi:10.1146/annurev.bioeng.4.092101.125733](https://doi.org/10.1146/annurev.bioeng.4.092101.125733)
- D. Tward, T. Brown, Y. Kageyama, et al. Diffeomorphic registration with intensity
  transformation and missing data: application to 3D digital pathology of Alzheimer's disease.
  *Frontiers in Neuroscience* 14, 52 (2020).
  [doi:10.3389/fnins.2020.00052](https://doi.org/10.3389/fnins.2020.00052)
- L. Younes. *Shapes and Diffeomorphisms*. Springer (2010; 2nd edition 2019): the mathematics of
  diffeomorphic registration in book form.
- [CHANGES.md](CHANGES.md): what changed in version 2 and the measurements behind the new
  defaults.
