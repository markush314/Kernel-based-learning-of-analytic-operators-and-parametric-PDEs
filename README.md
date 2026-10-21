# Kernel-based learning of analytic operators and parametric PDEs — code

Reproduces the experiments of Section 7 (Figure 1, Figures 2–4, Table 1).

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

## Stored results

The raw results behind the paper are included, so that every number can be checked
without rerunning anything:

| File | Content | Used for |
|---|---|---|
| `scalar_P1_exp.npz`, `scalar_P1_poly.npz`, `scalar_P5_exp.npz`, `scalar_P5_poly.npz` | test errors of the 10 repetitions per kernel and $N$, selected hyperparameters (`sel_*`) | Figure 1 |
| `results_320.npz` | relative test errors of the 10 repetitions per kernel and $N$, pipeline floor | Figure 2, Table 1, paired $t$-tests |
| `results_fig3_320.npz` | same, fixed kernel parameters | Figure 3 |
| `results_pvar320.npz` | same, varying $P$ at $N=300$ | Figure 4 |
| `results_oracle_320.npz` | test-set oracle sweep, $N\in\{300,900\}$, 3 repetitions | oracle statement in Section 7.2 |

In notebook 03, the cells marked *from stored results* redraw Figures 2–4, Table 1
(including the paired $p$-values) and the oracle summary from these files; their
stored outputs were produced in this way. Notebook 01 contains the outputs of the
original run.

The PDE training and test data written by notebook 02 (~1.4 GB) are not part of the
repository. They can be regenerated with notebook 02 (fixed seeds) or downloaded
from [ARCHIVE LINK / DOI].

## Requirements

* `01`, `03`: `numpy scipy scikit-learn matplotlib jax` (versions in `requirements.txt`)
* `02`: legacy FEniCS (`dolfin`, 2019.x) — e.g. `apt install python3-dolfin`, or
  conda-forge. Not needed if you only want to rerun the learning part on existing data.

Notebook 03 keeps several `1000 × 103041` arrays in memory; allow ~4 GB for the
$N=900$ experiment.

## Environment of the reported runs

* **Notebook 02** (FEM data): Ubuntu, `python3-dolfin` 2019.2.0~legacy20240219 (apt),
  FFC 2019.2.0~legacy, UFL-legacy 2022.3.0, Python 3.12.3, NumPy 1.26.4.
* **Notebooks 01 and 03** (scalar KRR, Figure 1; vector-valued KRR, Figures 2–4,
  Table 1): Python 3.11.15, jax/jaxlib 0.10.2 on CPU, NumPy 2.4.4, SciPy 1.17.1,
  scikit-learn 1.8.0, matplotlib 3.10.9 (pinned in `requirements.txt`).
* Hardware: 2 CPU cores (Intel Xeon, 2.8 GHz), 7 GB RAM, no GPU. The runtimes in the
  table above refer to this machine.

## Notes

* The hyperparameter grids are shared between Sections 7.1 and 7.2 and are defined at
  the top of the first code cell of each notebook.
* Notebook 02 writes ~3 GB of `.npz` files. They are excluded by `.gitignore`;
  regenerate them rather than committing them. Only the small result files listed
  above are tracked.

## Citation

Release used for the paper: [tag, e.g. `v1.0-juq`] — archived at [Zenodo DOI].
