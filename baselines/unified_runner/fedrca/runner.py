import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch

from .algorithms import Federation
from .config import DATASETS
from .data import loader, prepare
from .io import replace_with_retry, write_json
from .metrics import (
    aggregate_metrics,
    classification_metrics,
    predict,
)
from .partition import build_partition


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    replace_with_retry(temporary, path)


def rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def restore_rng(value):
    torch.set_rng_state(value["torch"])
    np.random.set_state(value["numpy"])
    random.setstate(value["python"])


def environment(config, device):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    import scipy
    import sklearn
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "device": str(device),
        "git_commit": revision, "implementation_sha256": digest.hexdigest(),
        "cpu_threads": config.cpu_threads, "protocol_version": 2,
        "reproducibility_policy": (
            "cuDNN deterministic, benchmark disabled; CUDA adaptive-average-pool backward may not be bitwise deterministic"
            if device.type == "cuda" else "PyTorch deterministic algorithms enforced"
        ),
        "simulator": "single process; not a secure multi-hospital deployment",
    }


def configure_reproducibility(device):
    """Use strict CPU determinism and warning-free best-effort CUDA reproducibility."""
    if device.type == "cuda":
        # CUDA adaptive_avg_pool2d_backward has no deterministic implementation.
        # Enforcing deterministic algorithms with warn_only=True merely prints the
        # same warning every round. Keep deterministic cuDNN selection without
        # claiming bitwise determinism for unsupported CUDA operators.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(True)


def _process_is_running(process_id):
    if process_id is None or process_id <= 0:
        return False
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_run_lock(lock):
    """Acquire a run lock and automatically recover locks from dead processes."""
    for _ in range(2):
        try:
            descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            try:
                process_id = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                process_id = None
            if _process_is_running(process_id):
                raise RuntimeError(
                    "Run is already active in process %s: %s" % (process_id, lock)
                ) from error
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            continue
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        return
    raise RuntimeError("Could not recover stale run lock: " + str(lock))


def evaluate_clients(federation, cache, partition, config, which):
    split = "val" if which == "val" else "train"
    results = []
    for client_id, model in enumerate(federation.evaluation_models()):
        labels, probabilities = predict(
            model,
            loader(cache, split, partition[which][client_id], config),
            federation.device,
            federation.inference_prototypes(),
        )
        results.append(classification_metrics(labels, probabilities, DATASETS[config.dataset][1]))
    return {"aggregate": aggregate_metrics(results), "per_client": results}


def evaluate_server_clients(federation, cache, partition, config, which):
    split = "val" if which == "val" else "train"
    results = []
    for indices in partition[which]:
        labels, probabilities = predict(
            federation.server,
            loader(cache, split, indices, config),
            federation.device,
        )
        results.append(classification_metrics(labels, probabilities, DATASETS[config.dataset][1]))
    return {"aggregate": aggregate_metrics(results), "per_client": results}


def evaluate_external_models(federation, cache, config, models, policy):
    labels = np.load(Path(cache) / "test_labels.npy", allow_pickle=False)
    batches = loader(cache, "test", np.arange(len(labels)), config)
    probabilities = None
    for model in models:
        targets, prediction = predict(
            model, batches, federation.device, federation.inference_prototypes(),
        )
        probabilities = (prediction.astype(np.float64) / len(models)
                         if probabilities is None else probabilities + prediction / len(models))
    return {
        "policy": policy,
        "metrics": classification_metrics(targets, probabilities, DATASETS[config.dataset][1]),
        "note": "PathMNIST test is external-center; other tasks follow the official test split. No test tuning.",
    }


def evaluate_global_validation(federation, cache, config):
    """Evaluate the deployable global model on official validation data."""
    labels = np.load(Path(cache) / "val_labels.npy", allow_pickle=False)
    targets, probabilities = predict(
        federation.server,
        loader(cache, "val", np.arange(len(labels)), config),
        federation.device,
    )
    classes = DATASETS[config.dataset][1]
    metrics = classification_metrics(targets, probabilities, classes)
    return {
        "split": "official_validation",
        "selection_metric": config.selection_metric,
        "selected_score": metrics[config.selection_metric],
        "metrics": metrics,
        "test_tuning": False,
    }


def evaluate_external(federation, cache, config):
    if config.algorithm in ("fedavg", "fedprox", "fedsol", "cwt", "fedseq"):
        return evaluate_external_models(
            federation, cache, config, [federation.server], "global_model"
        )
    return evaluate_external_models(
        federation, cache, config, federation.evaluation_models(),
        "equal_probability_ensemble_of_client_models",
    )


def write_history(path, history):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        if history:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    replace_with_retry(temporary, path)


def run(config, resume=False):
    config.validate()
    if config.device == "cpu":
        device = torch.device("cpu")
    elif config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("Explicit CUDA requested but unavailable")
        device = torch.device(config.device)
    else:
        raise ValueError("device must be cpu or an explicit cuda device")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    configure_reproducibility(device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise FileExistsError("Output directory is not empty. Choose a new output or use --resume: " + str(output))
    if resume and not (output / "last.pt").exists():
        raise FileNotFoundError("--resume requires last.pt")
    lock = output / "run.lock"
    acquire_run_lock(lock)
    try:
        return _run_locked(config, device, output, resume)
    finally:
        lock.unlink(missing_ok=True)


def _run_locked(config, device, output, resume):
    started = time.perf_counter()
    cache, metadata, data_id = prepare(config)
    train_labels = np.load(cache / "train_labels.npy", allow_pickle=False)
    val_labels = np.load(cache / "val_labels.npy", allow_pickle=False)
    partition = build_partition(train_labels, val_labels, config, data_id)
    federation = Federation(config, cache, partition, device)
    provenance = environment(config, device)
    history, best, best_round, best_score = [], None, 0, -float("inf")
    best_global, best_global_round = None, 0
    best_global_score, best_global_validation = -float("inf"), None
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config_id"] != config.identity() or checkpoint["partition_id"] != partition["partition_id"]:
            raise ValueError("Resume configuration/data/partition mismatch")
        if checkpoint["implementation_sha256"] != provenance["implementation_sha256"]:
            raise ValueError("Implementation changed since checkpoint; start a new run")
        federation.restore(checkpoint["federation"])
        if federation.round > config.rounds:
            raise ValueError("Cannot resume to fewer rounds")
        history, best = checkpoint["history"], checkpoint["best"]
        best_round, best_score = checkpoint["best_round"], checkpoint["best_score"]
        best_global = checkpoint.get("best_global")
        best_global_round = checkpoint.get("best_global_round", 0)
        best_global_score = checkpoint.get("best_global_score", -float("inf"))
        best_global_validation = checkpoint.get("best_global_validation")
        restore_rng(checkpoint["rng"])

    write_json(output / "config.json", config.as_dict())
    write_json(output / "environment.json", provenance)
    write_json(output / "data_manifest.json", metadata)
    write_json(output / "partition.json", partition)
    write_json(output / "results.json", {"status": "running", "dataset": config.dataset,
                                          "algorithm": config.algorithm,
                                          "smoke_only": bool(config.smoke_samples)})

    def report(message):
        """Keep progress visible in both the terminal and a durable job log."""
        line = str(message)
        print(line, flush=True)
        with (output / "console.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def checkpoint_now():
        save_checkpoint(output / "last.pt", {
            "config_id": config.identity(), "partition_id": partition["partition_id"],
            "implementation_sha256": provenance["implementation_sha256"],
            "federation": federation.pack(), "history": history, "best": best,
            "best_round": best_round, "best_score": best_score,
            "best_global": best_global, "best_global_round": best_global_round,
            "best_global_score": best_global_score,
            "best_global_validation": best_global_validation,
            "rng": rng_state(),
        })

    if config.algorithm == "fedap":
        while federation.warmup_done < config.fedap_warmup_rounds:
            federation.warmup_step()
            checkpoint_now()
            report("FedAP warmup %s/%s" % (
                federation.warmup_done, config.fedap_warmup_rounds
            ))
        if federation.ap_weights is None:
            federation.initialize_ap()
            checkpoint_now()

    while federation.round < config.rounds:
        federation.train_round()
        if federation.round % config.eval_every == 0 or federation.round == config.rounds:
            validation = evaluate_clients(federation, cache, partition, config, "val")
            summary = validation["aggregate"]
            score = summary[config.selection_metric + "_mean"]
            if score > best_score:
                best_score, best_round, best = score, federation.round, copy.deepcopy(federation.pack())
                write_json(output / "best_validation.json", validation)
            global_validation = None
            if config.algorithm == "fedrca":
                global_validation = evaluate_global_validation(federation, cache, config)
                if global_validation["selected_score"] > best_global_score:
                    best_global_score = global_validation["selected_score"]
                    best_global_round = federation.round
                    best_global = copy.deepcopy(federation.pack())
                    best_global_validation = copy.deepcopy(global_validation)
                    write_json(output / "best_global_validation.json", global_validation)
            method = federation.last_diagnostics.get("summary", {})
            record = {
                "round": federation.round, "train_seconds": federation.train_seconds,
                "learning_rate": config.learning_rate(config.lr, federation.round - 1),
                "head_learning_rate": config.learning_rate(config.head_lr, federation.round - 1),
                "communication_bytes": federation.bytes_sent, "optimizer_steps": federation.steps,
                "train_examples": federation.examples,
                "statistic_examples": federation.statistic_examples,
                "fedrca_pixel_regions": method.get("pixel_regions"),
                "fedrca_topology_loss": method.get("mean_topology_loss"),
                "fedrca_topology_scale": method.get("topology_scale"),
                "global_val_score": (global_validation or {}).get("selected_score"),
                **{"val_" + key: value for key, value in summary.items()},
            }
            history.append(record)
            global_text = (" global_val_%s=%.4f" % (
                config.selection_metric, global_validation["selected_score"],
            )) if global_validation else ""
            report("%s/%s alpha=%s seed=%s round=%s personal_val_%s=%.4f%s" % (
                config.dataset, config.algorithm, config.alpha, config.seed,
                federation.round, config.selection_metric, score, global_text,
            ))
        write_json(output / "last_algorithm_diagnostics.json", federation.last_diagnostics)
        if config.algorithm == "fedrca":
            write_json(output / "round_diagnostics" / ("round_%04d.json" % federation.round),
                       federation.last_diagnostics)
        checkpoint_now()
        write_history(output / "history.csv", history)

    if best is None:
        raise RuntimeError("No validated checkpoint")
    if config.algorithm == "fedrca" and best_global is None:
        raise RuntimeError("No globally validated FedRCA checkpoint")
    all_rounds_summary = federation.method_summary()
    cost = {
        "communication_bytes": federation.bytes_sent,
        "optimizer_steps": federation.steps,
        "train_examples": federation.examples,
        "statistic_examples": federation.statistic_examples,
        "preprocessing_examples": federation.preprocessing_examples,
        "train_seconds": federation.train_seconds,
        "fedap_warmup_rounds": federation.warmup_done,
        "fedrca_warmup_rounds": min(config.fedrca_warmup_rounds, federation.round)
        if config.algorithm == "fedrca" else 0,
    }
    federation.restore(best)
    save_checkpoint(output / "best.pt", {
        "federation": best, "config": config.as_dict(), "selected_round": best_round,
    })
    if config.algorithm == "fedrca":
        save_checkpoint(output / "best_personalized.pt", {
            "federation": best, "config": config.as_dict(), "selected_round": best_round,
            "validation_score": best_score,
        })
    personalized_method_summary = federation.method_summary()
    local_test = evaluate_clients(federation, cache, partition, config, "local_test")
    local_test_global = None
    external_global = None
    external_personalized = None
    if config.algorithm == "fedrca":
        external_personalized = evaluate_external_models(
            federation, cache, config, federation.evaluation_models(),
            "equal_probability_ensemble_of_personalized_models",
        )
        federation.restore(best_global)
        global_method_summary = federation.method_summary()
        save_checkpoint(output / "best_global.pt", {
            "federation": best_global, "config": config.as_dict(),
            "selected_round": best_global_round,
            "validation_score": best_global_score,
            "global_validation": best_global_validation,
        })
        local_test_global = evaluate_server_clients(
            federation, cache, partition, config, "local_test"
        )
        external_global = evaluate_external_models(
            federation, cache, config, [federation.server], "trained_global_model"
        )
        external = evaluate_external_models(
            federation, cache, config, [federation.server],
            "validation_selected_global_model",
        )
    else:
        global_method_summary = None
        external = evaluate_external(federation, cache, config)
    variant = config.algorithm
    if variant == "fedap":
        variant = "FedAP-federated-reference" if config.fedap_mode == "reference" else "f-FedAP-BN-warmup"
    elif variant == "fedrca":
        variant = "FedRCA-" + config.fedrca_variant
    result = {
        "status": "complete", "dataset": config.dataset, "algorithm": config.algorithm,
        "variant": variant, "seed": config.seed, "split_seed": config.split_seed,
        "partition_id": partition["partition_id"], "smoke_only": bool(config.smoke_samples),
        "selected_round": best_round,
        "selected_personalized_round": best_round,
        "selected_global_round": best_global_round if config.algorithm == "fedrca" else None,
        "selection_metric": "client_validation_mean_" + config.selection_metric,
        "global_selection_metric": ("official_validation_" + config.selection_metric
                                    if config.algorithm == "fedrca" else None),
        "best_validation_score": best_score,
        "best_personalized_validation_score": best_score,
        "best_global_validation_score": (best_global_score
                                         if config.algorithm == "fedrca" else None),
        "global_validation": best_global_validation,
        "local_test": local_test,
        "local_test_global": local_test_global,
        "official_test": external,
        "official_test_global": external_global,
        "official_test_personalized_ensemble": external_personalized,
        "method_summary_all_rounds": all_rounds_summary,
        "method_summary_selected_checkpoint": personalized_method_summary,
        "method_summary_personalized_checkpoint": personalized_method_summary,
        "method_summary_global_checkpoint": global_method_summary,
        "cost_all_rounds": cost,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "cost_note": "Tensor payload estimate; evaluation, serialization and protocol overhead are excluded.",
        "limitations": [
            "synthetic clients; no verified patient-level hospital split",
            "pixel partitions stay local, but model updates still lack differential privacy and secure aggregation",
            "uniform medical backbone adaptation; not a reproduction of published benchmark scores",
            "RBF and clustering gains require full multi-seed experiments; smoke runs are engineering checks only",
        ],
    }
    write_json(output / "results.json", result)
    write_history(output / "history.csv", history)
    return result
