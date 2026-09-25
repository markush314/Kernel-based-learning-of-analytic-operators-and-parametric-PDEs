# Kernel-based learning of analytic operators and parametric PDEs — code

Reproduces the experiments of Section 6 (Figure 1, Figures 2–4, Table 1).

## Notebooks

Run in this order. `01` is independent of the others; `02` must run before `03`.

| Notebook | What it does | Produces | Runtime |
|---|---|---|---|
| `01_scalar_krr.ipynb` | noiseless scalar KRR, four kernels, two analytic targets | Figure 1 | ~5 h |
| `02_pde_data_generation.ipynb` | FEM solves for the parametric diffusion problem | `*.npz` training/test data | ~4 h |
| `03_vkrr_learning_evaluation.ipynb` | vector-valued KRR on that data | Figures 2–4, Table 1 | ~7 h |

Every notebook writes its figures under the file names used in the paper and saves
its raw results as `.npz`, so the figures can be redrawn without re-running the fits.
All seeds are fixed.

## Requirements

* `01`, `03`: `numpy scipy scikit-learn matplotlib jax`
* `02`: legacy FEniCS (`dolfin`, 2019.x) — e.g. `apt install python3-dolfin`, or
  conda-forge. Not needed if you only want to rerun the learning part on existing data.

Notebook 03 keeps several `1000 × 103041` arrays in memory; allow ~4 GB for the
$N=900$ experiment.

## Notes

* The hyperparameter grids are shared between Sections 6.1 and 6.2 and are defined at
  the top of the first code cell of each notebook.
* Notebook 02 writes ~3 GB of `.npz` files. They are excluded by `.gitignore`;
  regenerate them rather than committing them.
