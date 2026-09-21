# P-only and Three-P controls

This package reproduces the reported supplementary controls without shipping
datasets, checkpoints, logs, or prior run results. It is namespaced as
`experiments.fedtriad_supplement_c` and imports the repository's existing
`fedtriad` implementation explicitly; it does not replace or edit the base
runner.

The two controls are:

- `p_clip5`: one parallel P model with joint gradient-norm clipping at 5;
- `p3_ensemble_clip5`: three independently initialized P models, each trained
  by FedAvg with the same matched partition and joint clipping, with uniform
  probability averaging at inference.

Each setting covers BloodMNIST, OrganAMNIST, and PathMNIST, alpha 0.1/0.5,
seeds 0/1/2, and 300 rounds. The preflight matches the original P-only and
PSL runs by partition, client participation, batch order, model dimensions,
learning-rate schedule, and branch work. The three-P control does not match
communication, persistent-state storage, or deployable inference-model count;
those are recorded as unmatched dimensions rather than hidden assumptions.

## Inputs

Before launching the controls, train the base repository's `p_only` and
`psl_uniform` variants with the normal runner. The completed base run
directory and its prepared cache are required only for frozen partitions and
matched reference endpoints; they are not part of this repository.

Put the three official NPZ files in a reader-chosen directory. The default is
`data/`, but every path can be supplied explicitly:

```powershell
python -B -u .\RUN_SUPPLEMENT_C.py --preview `
  --data-dir D:\datasets\medmnist `
  --base-runs D:\fedtriad-runs\fedtriad_3datasets_300r_ablations `
  --cache-root D:\fedtriad-cache\medmnist `
  --risk-output D:\fedtriad-output\final-global-class-risk
```

The files may be directly under `--data-dir` or under matching dataset-named
subdirectories. `--preview` performs the preflight and queues 36 jobs but does
not train. Remove it to run on CUDA (`--job 1` or `--job 2`). Results default
to `runs/supplement_c_300r/` and remain ignored by Git. Re-run the same command
after an interruption to resume from the last checkpoint.

The reference risk-evaluation summaries default to
`<repository>/output/final-global-class-risk/`, matching the base runner's
default output. Use `--risk-output` when the base results live elsewhere; the
path is recorded in `queue.json` and is validated before any control starts.
Each of the 18 PSL references must bind to its run basename and the
partition-checked PSL run, validation-selected round, and the SHA-256 of its
selected checkpoint. Newer endpoint files may include a top-level
`partition_id`, which is checked when present; the published v2 summaries
derive that binding through `run_directory`. No reference files are bundled
here.

## Completion gate

Only `ALL_DONE.json` after all 36 jobs reach 300 rounds indicates a complete
supplement. `REPORT.md`, `per_run_comparison.csv`, and the two aggregate CSVs
are generated only after the completion gate. Short `--verify` runs are placed
in a separate validation output and are never included in those summaries.
