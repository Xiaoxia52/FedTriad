"""Offline evaluation of the frozen FedTriad Global Class-Risk Calibration."""

import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from .algorithm import Federation
from .config import Config
from .data import loader, prepare
from .io import write_json
from .metrics import aggregate_metrics, classification_metrics
from .risk_router import (
    class_agnostic_risk,
    fit_global_class_risk,
    fuse_risk,
    serializable_memory,
)
from .runner import configure_reproducibility


PROTOCOL_VERSION = "fedtriad-global-class-risk-v2"
POLICIES = ("uniform", "class_agnostic_risk", "global_class_risk")
METRIC_KEYS = ("accuracy", "macro_f1", "balanced_accuracy", "auroc", "nll")


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fuse_policies(probabilities, global_risk, agnostic_risk, config):
    outputs = {"uniform": probabilities.mean(axis=1)}
    weights = {}
    for name, risk in (
        ("class_agnostic_risk", agnostic_risk),
        ("global_class_risk", global_risk),
    ):
        outputs[name], weights[name] = fuse_risk(
            probabilities,
            risk,
            config.triad_risk_temperature,
            config.triad_risk_entropy_weight,
            config.triad_risk_weight_floor,
        )
    return outputs, weights


def _evaluate_client_indices(federation, cache, partition_indices, config, split,
                             global_risk, agnostic_risk):
    policy_metrics = {name: [] for name in POLICIES}
    branch_metrics = [[], [], []]
    weight_means = {name: [] for name in POLICIES[1:]}
    for client_id, indices in enumerate(partition_indices):
        labels, probabilities = federation.predict_client_branches(
            loader(cache, split, indices, config), client_id
        )
        outputs, weights = _fuse_policies(
            probabilities, global_risk, agnostic_risk, config
        )
        for name, values in outputs.items():
            policy_metrics[name].append(
                classification_metrics(labels, values, config.num_classes)
            )
        for branch_id in range(3):
            branch_metrics[branch_id].append(classification_metrics(
                labels, probabilities[:, branch_id], config.num_classes
            ))
        for name, values in weights.items():
            weight_means[name].append(values.mean(axis=0).tolist())
    result = {
        name: {"aggregate": aggregate_metrics(values), "per_client": values}
        for name, values in policy_metrics.items()
    }
    result["branches"] = {
        name: {"aggregate": aggregate_metrics(values), "per_client": values}
        for name, values in zip(("parallel", "serial", "local"), branch_metrics)
    }
    result["weight_mean_per_client"] = weight_means
    return result


def _evaluate_official_ensemble(federation, cache, config, global_risk,
                                agnostic_risk, split="test"):
    labels = np.load(Path(cache) / (split + "_labels.npy"), allow_pickle=False)
    indices = np.arange(len(labels), dtype=np.int64)
    totals = {name: None for name in POLICIES}
    weight_means = {name: [] for name in POLICIES[1:]}
    observed_labels = None
    for client_id in range(config.clients):
        current_labels, probabilities = federation.predict_client_branches(
            loader(cache, split, indices, config), client_id
        )
        if observed_labels is None:
            observed_labels = current_labels
        elif not np.array_equal(observed_labels, current_labels):
            raise RuntimeError("Official split labels changed between client predictions")
        outputs, weights = _fuse_policies(
            probabilities, global_risk, agnostic_risk, config
        )
        for name, values in outputs.items():
            scaled = values / config.clients
            totals[name] = scaled if totals[name] is None else totals[name] + scaled
        for name, values in weights.items():
            weight_means[name].append(values.mean(axis=0).tolist())
    result = {
        name: classification_metrics(observed_labels, totals[name], config.num_classes)
        for name in POLICIES
    }
    result.update({
        "split": "official_" + split,
        "weight_mean_per_local_model": weight_means,
        "test_tuning": False,
        "policy": "average calibrated P/S/L predictions across all client-local models",
    })
    return result


def _metric_deltas(reference, candidate):
    result = {}
    for key in METRIC_KEYS:
        before, after = reference.get(key), candidate.get(key)
        if before is not None and after is not None:
            result[key] = float(before - after if key == "nll" else after - before)
    return result


def _self_contained_config(config_values, directory, device):
    project_root = Path(__file__).resolve().parents[1]
    values = dict(config_values)
    dataset = values["dataset"]
    values.update({
        "data_file": str(project_root / "data" / (dataset + ".npz")),
        "cache_dir": str(project_root / ".cache" / "medmnist"),
        "output_dir": str(directory),
        "device": device,
        "workers": 0,
    })
    return Config.from_dict(values)


def evaluate_run(directory, output_directory, device="cpu", force=False):
    directory, output_directory = Path(directory), Path(output_directory)
    output_path = output_directory / (directory.name + ".json")
    if output_path.is_file() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("protocol_version") == PROTOCOL_VERSION:
            return "current", existing

    config_path = directory / "config.json"
    checkpoint_path = directory / "best_personal.pt"
    result_path = directory / "results.json"
    if not config_path.is_file() or not checkpoint_path.is_file() or not result_path.is_file():
        return "incomplete", None
    old_result = json.loads(result_path.read_text(encoding="utf-8"))
    config_values = json.loads(config_path.read_text(encoding="utf-8"))
    if old_result.get("status") != "complete":
        return "incomplete", None
    if config_values.get("triad_variant") != "psl_uniform":
        return "not_dense_psl", None

    config = _self_contained_config(config_values, directory, device)
    if device == "cpu":
        torch_device = torch.device("cpu")
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(device)
    else:
        raise ValueError("Requested evaluation device is unavailable: " + device)
    torch.set_num_threads(config.cpu_threads)
    configure_reproducibility(torch_device)

    cache, _, _ = prepare(config)
    partition = json.loads((directory / "partition.json").read_text(encoding="utf-8"))
    federation = Federation(config, cache, partition, torch_device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    federation.restore(checkpoint["federation"])

    train_probabilities, train_labels = [], []
    for client_id, indices in enumerate(partition["train"]):
        labels, probabilities = federation.predict_client_branches(
            loader(cache, "train", indices, config), client_id
        )
        train_probabilities.append(probabilities)
        train_labels.append(labels)
    memory = fit_global_class_risk(
        train_probabilities, train_labels, config.num_classes
    )
    agnostic = class_agnostic_risk(memory)

    validation = _evaluate_client_indices(
        federation, cache, partition["val"], config, "val",
        memory["global_risk"], agnostic,
    )
    local_test = _evaluate_client_indices(
        federation, cache, partition["local_test"], config, "train",
        memory["global_risk"], agnostic,
    )
    official_test = _evaluate_official_ensemble(
        federation, cache, config, memory["global_risk"], agnostic, split="test"
    )

    stored_uniform = old_result["local_test_personalized"]["aggregate"]
    recomputed_uniform = local_test["uniform"]["aggregate"]
    reproducibility_error = {
        key: abs(float(stored_uniform[key]) - float(recomputed_uniform[key]))
        for key in stored_uniform
        if key in recomputed_uniform
        and isinstance(stored_uniform[key], (int, float))
        and isinstance(recomputed_uniform[key], (int, float))
    }
    deltas = {}
    for scope, values in (
        ("client_validation_weighted", validation),
        ("local_test_weighted", local_test),
    ):
        deltas[scope] = {
            policy: _metric_deltas(
                values["uniform"]["aggregate"], values[policy]["aggregate"]
            ) for policy in POLICIES[1:]
        }
    deltas["official_test"] = {
        policy: _metric_deltas(official_test["uniform"], official_test[policy])
        for policy in POLICIES[1:]
    }

    output = {
        "protocol_version": PROTOCOL_VERSION,
        "risk_source": (
            "compressed per-client P/S/L by-class NLL sums and class counts from "
            "the training partition at the validation-selected personalized checkpoint"
        ),
        "privacy_boundary": (
            "the evaluator simulates federated aggregation; only 3xC NLL sums and "
            "C counts are mathematically required, never images or sample features"
        ),
        "no_validation_or_test_label_fitting": True,
        "run_directory": str(directory),
        "dataset": config.dataset,
        "alpha": config.alpha,
        "seed": config.seed,
        "selected_round": int(checkpoint.get("selected_round", federation.round)),
        "selection_metric": config.selection_metric,
        "checkpoint": {"file": checkpoint_path.name, "sha256": _checkpoint_sha256(checkpoint_path)},
        "risk_hyperparameters": {
            "temperature": config.triad_risk_temperature,
            "entropy_weight": config.triad_risk_entropy_weight,
            "weight_floor": config.triad_risk_weight_floor,
        },
        "memory": serializable_memory(memory),
        "class_agnostic_risk_table": agnostic.tolist(),
        "client_validation": validation,
        "local_test": local_test,
        "official_test_personalized_ensemble": official_test,
        "deltas_positive_is_better": deltas,
        "stored_uniform_reproduction_absolute_error": reproducibility_error,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_path, output)
    del federation
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return "updated", output


def _summary_row(result):
    row = {
        "dataset": result["dataset"], "alpha": result["alpha"],
        "seed": result["seed"], "selected_round": result["selected_round"],
    }
    for scope, values in (("val", result["client_validation"]),
                          ("local_test", result["local_test"])):
        for policy in POLICIES:
            aggregate = values[policy]["aggregate"]
            for metric in METRIC_KEYS:
                for statistic_name in ("mean", "weighted", "min", "std"):
                    row[f"{scope}_{policy}_{metric}_{statistic_name}"] = aggregate.get(
                        metric + "_" + statistic_name
                    )
    official = result["official_test_personalized_ensemble"]
    for policy in POLICIES:
        for metric in METRIC_KEYS:
            row[f"official_{policy}_{metric}"] = official[policy].get(metric)
    return row


def _write_csv(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _aggregate_rows(results):
    rows = []
    grouped = {}
    for result in results:
        grouped.setdefault((result["dataset"], result["alpha"]), []).append(result)
    for (dataset, alpha), entries in sorted(grouped.items()):
        row = {"dataset": dataset, "alpha": alpha, "seeds": len(entries)}
        for scope_name, accessor in (
            ("local_test", lambda item: item["local_test"]),
            ("official", lambda item: item["official_test_personalized_ensemble"]),
        ):
            for policy in POLICIES:
                for metric in METRIC_KEYS:
                    values = []
                    for item in entries:
                        scope = accessor(item)
                        value = (scope[policy]["aggregate"].get(metric + "_mean")
                                 if scope_name == "local_test" else scope[policy].get(metric))
                        if value is not None:
                            values.append(float(value))
                    row[f"{scope_name}_{policy}_{metric}_mean"] = (
                        statistics.mean(values) if values else None
                    )
                    row[f"{scope_name}_{policy}_{metric}_seed_std"] = (
                        statistics.stdev(values) if len(values) > 1 else None
                    )
        rows.append(row)
    return rows


def _write_report(output_directory, statuses, aggregate_rows):
    lines = [
        "# FedTriad final Global Class-Risk evaluation",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`",
        "",
        "Policies: Uniform P/S/L; Class-Agnostic Risk control; Global Class-Risk (full).",
        "Positive comparison evidence requires Global Class-Risk to improve over both controls.",
        "",
        f"Completed result files: {sum(statuses[key] for key in ('updated', 'current'))}",
        f"Incomplete checkpoints: {statuses['incomplete']}",
        "",
        "| Dataset | alpha | seeds | Local Macro-F1: Uniform / Agnostic / Global | Official Macro-F1: Uniform / Agnostic / Global |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        local = " / ".join(
            f"{row[f'local_test_{policy}_macro_f1_mean']:.4f}" for policy in POLICIES
        )
        official = " / ".join(
            f"{row[f'official_{policy}_macro_f1_mean']:.4f}" for policy in POLICIES
        )
        lines.append(
            f"| {row['dataset']} | {row['alpha']} | {row['seeds']} | {local} | {official} |"
        )
    lines.extend([
        "",
        "All values are evaluated without fitting on validation or test labels. Full numerical fields are in `summary.csv` and `aggregate.csv`.",
        "",
    ])
    (output_directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_root(root, output_directory, datasets=("bloodmnist", "organamnist", "pathmnist"),
                  device="cpu", force=False, expected_runs=18):
    root, output_directory = Path(root), Path(output_directory)
    statuses = {"updated": 0, "current": 0, "incomplete": 0, "not_dense_psl": 0}
    results = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if "_fedtriad_psl_uniform_" not in directory.name:
            continue
        if not any(directory.name.startswith(dataset + "_") for dataset in datasets):
            continue
        status, result = evaluate_run(directory, output_directory, device=device, force=force)
        statuses[status] += 1
        if result is not None:
            results.append(result)
        print("FedTriad GlobalRisk %s: %s" % (status, directory.name), flush=True)

    rows = [_summary_row(item) for item in sorted(
        results, key=lambda item: (item["dataset"], item["alpha"], item["seed"])
    )]
    aggregate_rows = _aggregate_rows(results)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "summary.csv", rows)
    _write_csv(output_directory / "aggregate.csv", aggregate_rows)
    evaluated = statuses["updated"] + statuses["current"]
    complete = evaluated == expected_runs and statuses["incomplete"] == 0
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "root": str(root), "datasets": list(datasets), "device": device,
        "expected_dense_psl_runs": expected_runs, "statuses": statuses,
        "evaluated_results": evaluated, "complete": complete,
    }
    write_json(output_directory / "manifest.json", manifest)
    _write_report(output_directory, statuses, aggregate_rows)
    if not complete:
        raise RuntimeError(
            "Final evaluation incomplete: expected %s dense PSL runs, evaluated %s"
            % (expected_runs, evaluated)
        )
    return manifest
