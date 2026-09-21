# FedTriad validation-curve package

This directory contains the reproducible 300-round PSL curve extension. It is
kept in the separate `fedtriad_curves` Python namespace so importing it cannot
replace or mutate the repository's main `fedtriad` package.

The package covers the six dataset/partition settings (BloodMNIST,
OrganAMNIST, and PathMNIST; Dirichlet alpha 0.5 and 0.1) with seeds 0, 1, and
2. It evaluates uniform P/S/L and full class-risk fusion at rounds
`1, 15, 30, ..., 300`. The frozen partition files in `protocol/` contain only
sample indices and checksums; no images, checkpoints, logs, or caches are
included.

## Requirements

Use the main repository environment plus `matplotlib` only if you also want to
plot the exported results. Install PyTorch for the CUDA or CPU platform you
intend to use; this package does not install dependencies automatically.

## Run with reader-supplied data

Place the three official MedMNIST NPZ files in a directory of your choice, or
pass that directory explicitly:

```powershell
python -B -u .\experiments\curves\RUN_CURVES.py `
  --data-dir D:\datasets\medmnist --job 2 --preview
```

`--preview` verifies the array fingerprints and all 18 frozen partitions but
does not train. Remove it to run the 300-round CUDA jobs. Re-run the same
command after an interruption to resume. Results are written below
`experiments/curves/runs/`, which is ignored by Git; `--export-only` creates
the per-seed and three-seed summary CSV files after all 18 jobs complete.

The default `--data-dir data` is repository-relative. The three expected files
are `bloodmnist.npz`, `organamnist.npz`, and `pathmnist.npz` (a matching
dataset-named subdirectory is also accepted by the loader).

The curve package is intentionally separate from the base runner. Do not copy
checkpoints between the base project and this directory: the implementation
fingerprint and frozen partition checks are designed to reject such mixing.
