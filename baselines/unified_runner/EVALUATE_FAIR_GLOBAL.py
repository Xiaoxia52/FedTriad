"""Re-evaluate FedRCA's global model at the client-validation checkpoint.

The training runner keeps two FedRCA checkpoints:

* ``best.pt`` is selected by the same mean client-validation metric used by
  the comparison methods.
* ``best_global.pt`` is selected on the complete official validation split.

The original official FedRCA result uses the latter.  This utility evaluates
the server model stored in ``best.pt`` so the main comparison can use the same
checkpoint-selection information for every method.  Original result files are
never overwritten.
"""

import argparse
import json
from pathlib import Path

import torch

from fedrca.algorithms import Federation
from fedrca.config import Config
from fedrca.data import prepare
from fedrca.io import write_json
from fedrca.runner import (
    evaluate_external_models,
    evaluate_global_validation,
    evaluate_server_clients,
)


DEFAULT_ROOT = "runs/main_3datasets_11methods_300r_topology_rbf"
OUTPUT_NAME = "fair_global_client_selected.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def eligible_runs(root, datasets):
    selected = set(datasets or [])
    for result_path in sorted(Path(root).glob("*/results.json")):
        result = read_json(result_path)
        if result.get("status") != "complete" or result.get("algorithm") != "fedrca":
            continue
        if selected and result.get("dataset") not in selected:
            continue
        run_dir = result_path.parent
        if not (run_dir / "best.pt").exists():
            raise FileNotFoundError("Completed FedRCA run lacks best.pt: " + str(run_dir))
        yield run_dir, result


def evaluate_run(run_dir, result, device):
    values = read_json(run_dir / "config.json")
    values["device"] = device
    config = Config.from_dict(values)
    cache, _, _ = prepare(config)
    partition = read_json(run_dir / "partition.json")

    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    selected_round = int(checkpoint["selected_round"])
    if selected_round != int(result["selected_round"]):
        raise ValueError("best.pt/results.json selected-round mismatch: " + str(run_dir))

    federation = Federation(config, cache, partition, torch.device(device))
    federation.restore(checkpoint["federation"])
    official_validation = evaluate_global_validation(federation, cache, config)
    official_test = evaluate_external_models(
        federation,
        cache,
        config,
        [federation.server],
        "global_model_at_client_validation_selected_checkpoint",
    )
    local_test = evaluate_server_clients(
        federation, cache, partition, config, "local_test"
    )
    return {
        "status": "complete",
        "dataset": result["dataset"],
        "algorithm": "fedrca",
        "seed": result["seed"],
        "split_seed": result["split_seed"],
        "partition_id": result["partition_id"],
        "checkpoint": "best.pt",
        "selected_round": selected_round,
        "selection_metric": result["selection_metric"],
        "selection_score": result["best_validation_score"],
        "fairness_note": (
            "Checkpoint selected by the same mean client-validation metric used "
            "for the comparison methods; official validation and test are evaluation-only."
        ),
        "official_validation": official_validation,
        "official_test": official_test,
        "local_test_global": local_test,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fair FedRCA global evaluation at the client-validation checkpoint."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    runs = list(eligible_runs(root, args.datasets))
    if not runs:
        raise SystemExit("No completed FedRCA runs found under " + str(root))

    print("Fair global evaluation: %s completed FedRCA runs" % len(runs), flush=True)
    for run_dir, result in runs:
        output = run_dir / OUTPUT_NAME
        label = "%s alpha=%s seed=%s" % (
            result["dataset"], read_json(run_dir / "config.json")["alpha"], result["seed"]
        )
        if args.preview:
            print("[preview] %s -> %s" % (label, output), flush=True)
            continue
        if output.exists() and not args.overwrite:
            print("[skip] %s (already evaluated)" % label, flush=True)
            continue
        evaluated = evaluate_run(run_dir, result, args.device)
        write_json(output, evaluated)
        metrics = evaluated["official_test"]["metrics"]
        print(
            "[done] %s round=%s official_macro_f1=%.4f" % (
                label, evaluated["selected_round"], metrics["macro_f1"]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
