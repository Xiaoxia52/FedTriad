# FedTriad

FedTriad is a compact research implementation for personalized cross-silo
federated learning on medical image classification. It keeps patient images at
their originating clients and exchanges model states plus compact class-risk
statistics. This repository reproduces the method and ablations reported for
BloodMNIST, OrganAMNIST, and PathMNIST at their native 28 x 28 resolution.

Repository: <https://github.com/Xiaoxia52/FedTriad>

> **Privacy scope.** Keeping images local avoids centralizing raw records, but
> ordinary model/statistic exchange is not a formal privacy guarantee. Add
> secure aggregation or differential privacy before using this prototype with
> sensitive clinical data.

## Method

FedTriad contains two components:

1. **Parallel--Sequential--Local (PSL) complementary learning.** The parallel path
   `P` learns population consensus with sample-size-weighted FedAvg; one
   persistent sequential path `S` visits active hospitals in a rotating order; and
   every hospital owns a private persistent local path `L` that is never
   uploaded. The three paths use the same network architecture and ordinary
   end-to-end classification gradients.
2. **Class-risk-guided prediction fusion.** Each client uploads per-class negative
   log-likelihood sums for P/S/L and per-class counts. The server aggregates
   these `4C` scalars per client into a class-conditioned risk memory. At
   inference, historical class risk and current predictive uncertainty produce
   sample-wise P/S/L fusion weights.

The matched ablation matrix is `P`, `P+S`, `P+L`, `P+S+L Uniform`,
`PSL + class-agnostic risk`, and full FedTriad. See
[`docs/METHOD_AND_ABLATIONS.md`](docs/METHOD_AND_ABLATIONS.md) for the frozen
method and reporting contract.

## Environment

- Python 3.10 or newer
- PyTorch 2.x
- A CUDA-capable GPU is recommended for the complete 72-run matrix

Install PyTorch for the CUDA or CPU platform you intend to use, then install
the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

For development and tests:

```powershell
python -m pip install -r requirements-dev.txt
```

The 11-method comparison runner has a separate dependency set because its
FedPAC implementation uses SciPy. Install
[`baselines/unified_runner/requirements.txt`](baselines/unified_runner/requirements.txt)
from that directory only when reproducing the baseline suite.

## Data

Download the official MedMNIST v2 NPZ archives from the
[MedMNIST website](https://medmnist.com/) and place these files in `data/`:

```text
data/
  bloodmnist.npz
  organamnist.npz
  pathmnist.npz
```

The loader preserves the official train/validation/test split and refuses
silent resolution changes. Dataset archives are intentionally excluded from
this repository; see [`data/README.md`](data/README.md).

## Reproduce the complete experiment

First inspect the 72-job matrix without loading data or starting training:

```powershell
python -u .\RUN_FEDTRIAD_FINAL_GPU.py --preview --job 6
```

Run all training ablations and automatically perform the final uniform,
class-agnostic-risk, and global-class-risk evaluation:

```powershell
python -u .\RUN_FEDTRIAD_FINAL_GPU.py --job 6 --eval-device cuda:0
```

The command is restart-safe: after an interruption, run the identical command
again. Completed jobs are skipped, valid checkpoints are resumed, and stale
lock files are removed only after their owning process is confirmed absent.
`--job` is a global process-pool width and must be between 1 and 9. Reduce it if
your GPU does not have enough memory.

For training only, with optional seed or ablation selection:

```powershell
python -u .\RUN_FEDTRIAD_GPU.py --resume --job 6 --seeds 0 1 2
```

For offline re-evaluation of completed PSL checkpoints:

```powershell
python -u .\EVAL_DENSE_PSL_GLOBAL_RISK.py --device cuda:0
```

Training artifacts are written under `runs/`; final risk-evaluation summaries
are written under `output/final-global-class-risk/`. Both directories are
ignored by Git.

## Reproduce the reported extensions

The curve package is isolated under
[`experiments/curves/`](experiments/curves/). It uses the frozen PSL
partitions in that directory and accepts a reader-supplied `--data-dir`:

```powershell
python -u .\experiments\curves\RUN_CURVES.py --data-dir D:\datasets\medmnist --job 2 --preview
```

Remove `--preview` for the 18 CUDA curve runs. See
[`experiments/curves/README.md`](experiments/curves/README.md) for resume and
export details. The package uses the independent `fedtriad_curves` namespace;
it does not alter the base `fedtriad` imports.

The P-only and Three-P supplementary controls are launched from
[`RUN_SUPPLEMENT_C.py`](RUN_SUPPLEMENT_C.py). They require completed base
`p_only`/`psl_uniform` runs and a prepared cache, supplied with
`--base-runs`, `--cache-root`, and the reader's `--data-dir`; none of those
artifacts are included here:

```powershell
python -u .\RUN_SUPPLEMENT_C.py --preview `
  --data-dir D:\datasets\medmnist `
  --base-runs D:\fedtriad-runs\fedtriad_3datasets_300r_ablations `
  --cache-root D:\fedtriad-cache\medmnist
```

See [`experiments/fedtriad_supplement_c/README.md`](experiments/fedtriad_supplement_c/README.md)
for the matched and unmatched dimensions of the controls.

The checked-in curve summary at
[`figures/curves/source_data_mean_sd.csv`](figures/curves/source_data_mean_sd.csv)
can be redrawn with the parameterized plotting script in the same directory.
It is a small aggregate table, not a dataset or a training log.

## Tests

```powershell
python -m pytest -q
```

The tests cover the locked learning-rate schedule, the complete experiment
matrix, moved-project checkpoint reuse, PSL branch behavior, compressed risk
aggregation, and normalized risk-guided fusion.

## Repository layout

```text
experiments/curves/              Isolated 300-round validation-curve package
experiments/fedtriad_supplement_c/  P-only and Three-P control implementation
baselines/unified_runner/        Self-contained 11-method comparison runner
figures/curves/                  Parameterized plotter and aggregate curve CSV
fedtriad/                       Core method, data, metrics, runner, and suite
configs/                        Locked three-dataset 300-round configuration
docs/                           Method and ablation contract
tests/                          Unit tests
RUN_FEDTRIAD_FINAL_GPU.py       One-command training and final evaluation
RUN_FEDTRIAD_GPU.py             Training-only launcher
EVAL_DENSE_PSL_GLOBAL_RISK.py   Offline risk evaluation
SUMMARIZE_FEDTRIAD.py           Run-summary utility
```

## Citation

The accompanying manuscript is under preparation. If this code supports your
research, please cite the repository for now; a final BibTeX entry will be added
after publication.

## License

Released under the [MIT License](LICENSE).
