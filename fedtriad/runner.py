import copy
import ctypes
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

from .algorithm import Federation
from .config import DATASETS
from .data import loader, prepare
from .io import replace_with_retry, write_json
from .metrics import aggregate_metrics, classification_metrics
from .partition import build_partition


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    replace_with_retry(temporary, path)


def rng_state():
    value = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
             "python": random.getstate()}
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng(value):
    torch.set_rng_state(value["torch"])
    np.random.set_state(value["numpy"])
    random.setstate(value["python"])
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])


def environment(config, device):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": str(device), "git_commit": revision,
        "implementation_sha256": digest.hexdigest(), "cpu_threads": config.cpu_threads,
        "protocol_version": "fedtriad-final-2contrib-300r",
        "reproducibility_policy": (
            "deterministic cuDNN selection; unsupported CUDA kernels are not claimed bitwise deterministic"
            if device.type == "cuda" else "PyTorch deterministic algorithms enforced"
        ),
        "simulator": "single-machine cross-silo simulation; no secure aggregation",
    }


def configure_reproducibility(device):
    if device.type == "cuda":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        except (AttributeError, RuntimeError):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(True)


def _process_is_running(process_id):
    if process_id is None or process_id <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(process_id, 0)
        except (ProcessLookupError, OSError):
            return False
        except PermissionError:
            return True
        return True
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100000, False, process_id)
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        kernel32.CloseHandle(handle)


def acquire_run_lock(lock):
    for _ in range(2):
        try:
            descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            try:
                process_id = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                process_id = None
            if _process_is_running(process_id):
                raise RuntimeError("Run is already active in process %s: %s" %
                                   (process_id, lock)) from error
            lock.unlink(missing_ok=True)
            continue
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        return
    raise RuntimeError("Could not recover stale run lock: " + str(lock))


def evaluate_client_split(federation, cache, partition, config, which):
    split = "val" if which == "val" else "train"
    personal, global_results = [], []
    for client_id, indices in enumerate(partition[which]):
        labels, personal_probabilities, global_probabilities = federation.predict_client(
            loader(cache, split, indices, config), client_id
        )
        personal.append(classification_metrics(labels, personal_probabilities, config.num_classes))
        global_results.append(classification_metrics(labels, global_probabilities, config.num_classes))
    return (
        {"aggregate": aggregate_metrics(personal), "per_client": personal},
        {"aggregate": aggregate_metrics(global_results), "per_client": global_results},
    )


def evaluate_official(federation, cache, config, split="val", personalized=False):
    labels = np.load(Path(cache) / (split + "_labels.npy"), allow_pickle=False)
    batches = loader(cache, split, np.arange(len(labels)), config)
    if personalized:
        targets, probabilities = federation.predict_personalized_ensemble(batches)
        policy = "equal_probability_ensemble_of_client_personalized_predictions"
    else:
        targets, probabilities = federation.predict_global(batches)
        policy = "deployable_parallel_serial_global_prediction"
    return {
        "split": "official_" + split,
        "policy": policy,
        "metrics": classification_metrics(targets, probabilities, config.num_classes),
        "test_tuning": False,
    }


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
        raise FileExistsError("Output is not empty; choose a new output or use --resume: " + str(output))
    if resume and not (output / "last.pt").is_file():
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
    history = []
    best_personal = best_global_client = best_global_official = None
    best_personal_round = best_global_client_round = best_global_official_round = 0
    best_personal_score = best_global_client_score = best_global_official_score = -float("inf")
    best_personal_validation = best_global_client_validation = best_global_official_validation = None

    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config_id"] != config.identity() or checkpoint["partition_id"] != partition["partition_id"]:
            raise ValueError("Resume configuration/data/partition mismatch")
        if checkpoint["implementation_sha256"] != provenance["implementation_sha256"]:
            raise ValueError("Implementation changed since checkpoint; start a new output")
        federation.restore(checkpoint["federation"])
        if federation.round > config.rounds:
            raise ValueError("Cannot resume to fewer rounds")
        history = checkpoint["history"]
        best_personal = checkpoint["best_personal"]
        best_global_client = checkpoint["best_global_client"]
        best_global_official = checkpoint["best_global_official"]
        best_personal_round = checkpoint["best_personal_round"]
        best_global_client_round = checkpoint["best_global_client_round"]
        best_global_official_round = checkpoint["best_global_official_round"]
        best_personal_score = checkpoint["best_personal_score"]
        best_global_client_score = checkpoint["best_global_client_score"]
        best_global_official_score = checkpoint["best_global_official_score"]
        best_personal_validation = checkpoint["best_personal_validation"]
        best_global_client_validation = checkpoint["best_global_client_validation"]
        best_global_official_validation = checkpoint["best_global_official_validation"]
        restore_rng(checkpoint["rng"])

    write_json(output / "config.json", config.as_dict())
    write_json(output / "environment.json", provenance)
    write_json(output / "data_manifest.json", metadata)
    write_json(output / "partition.json", partition)
    write_json(output / "results.json", {
        "status": "running", "dataset": config.dataset, "algorithm": config.algorithm,
        "variant": config.triad_variant, "smoke_only": bool(config.smoke_samples),
    })

    def report(message):
        line = str(message)
        print(line, flush=True)
        with (output / "console.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def checkpoint_now():
        save_checkpoint(output / "last.pt", {
            "config_id": config.identity(), "partition_id": partition["partition_id"],
            "implementation_sha256": provenance["implementation_sha256"],
            "federation": federation.pack(), "history": history,
            "best_personal": best_personal, "best_global_client": best_global_client,
            "best_global_official": best_global_official,
            "best_personal_round": best_personal_round,
            "best_global_client_round": best_global_client_round,
            "best_global_official_round": best_global_official_round,
            "best_personal_score": best_personal_score,
            "best_global_client_score": best_global_client_score,
            "best_global_official_score": best_global_official_score,
            "best_personal_validation": best_personal_validation,
            "best_global_client_validation": best_global_client_validation,
            "best_global_official_validation": best_global_official_validation,
            "rng": rng_state(),
        })

    while federation.round < config.rounds:
        federation.train_round()
        if federation.round % config.eval_every == 0 or federation.round == config.rounds:
            personal_validation, global_client_validation = evaluate_client_split(
                federation, cache, partition, config, "val"
            )
            global_official_validation = evaluate_official(federation, cache, config, "val")
            personal_score = personal_validation["aggregate"][config.selection_metric + "_mean"]
            global_client_score = global_client_validation["aggregate"][config.selection_metric + "_mean"]
            global_official_score = global_official_validation["metrics"][config.selection_metric]

            if personal_score > best_personal_score:
                best_personal_score, best_personal_round = personal_score, federation.round
                best_personal = copy.deepcopy(federation.pack())
                best_personal_validation = copy.deepcopy(personal_validation)
                write_json(output / "best_personal_validation.json", personal_validation)
            if global_client_score > best_global_client_score:
                best_global_client_score, best_global_client_round = global_client_score, federation.round
                best_global_client = copy.deepcopy(federation.pack())
                best_global_client_validation = copy.deepcopy(global_client_validation)
                write_json(output / "best_global_client_validation.json", global_client_validation)
            if global_official_score > best_global_official_score:
                best_global_official_score, best_global_official_round = global_official_score, federation.round
                best_global_official = copy.deepcopy(federation.pack())
                best_global_official_validation = copy.deepcopy(global_official_validation)
                write_json(output / "best_global_official_validation.json", global_official_validation)

            summary = federation.last_diagnostics["summary"]
            history.append({
                "round": federation.round,
                "learning_rate": config.learning_rate(config.lr, federation.round - 1),
                "train_seconds": federation.train_seconds,
                "communication_bytes": federation.bytes_sent,
                "optimizer_steps": federation.optimizer_steps,
                "branch_optimizer_steps": federation.branch_optimizer_steps,
                "train_examples": federation.train_examples,
                "personal_val_score": personal_score,
                "global_client_val_score": global_client_score,
                "global_official_val_score": global_official_score,
                "parallel_ce": summary["mean_parallel_ce"],
                "serial_ce": summary["mean_serial_ce"],
                "local_ce": summary["mean_local_ce"],
            })
            report(
                "%s/%s alpha=%s seed=%s round=%s personal_val_%s=%.4f "
                "global_client_val_%s=%.4f official_val_%s=%.4f" % (
                    config.dataset, config.triad_variant, config.alpha, config.seed,
                    federation.round, config.selection_metric, personal_score,
                    config.selection_metric, global_client_score,
                    config.selection_metric, global_official_score,
                )
            )
        write_json(output / "last_algorithm_diagnostics.json", federation.last_diagnostics)
        checkpoint_now()
        write_history(output / "history.csv", history)

    if None in (best_personal, best_global_client, best_global_official):
        raise RuntimeError("One or more validation-selected checkpoints are missing")
    all_rounds_summary = federation.method_summary()
    cost = {
        "communication_bytes": federation.bytes_sent,
        "optimizer_steps": federation.optimizer_steps,
        "branch_optimizer_steps": federation.branch_optimizer_steps,
        "train_examples_across_branches": federation.train_examples,
        "train_seconds": federation.train_seconds,
        "parameter_payload_bytes_parallel": sum(value.numel() * value.element_size()
                                                for value in federation.parallel_server.state_dict().values()),
        "parameter_payload_bytes_serial": (sum(value.numel() * value.element_size()
                                              for value in federation.serial.state_dict().values())
                                           if federation.serial is not None else 0),
        "parameter_payload_bytes_private_per_client": (sum(value.numel() * value.element_size()
                                                           for value in federation.locals[0].state_dict().values())
                                                        if federation.locals else 0),
    }

    federation.restore(best_personal)
    personal_method_summary = federation.method_summary()
    save_checkpoint(output / "best_personal.pt", {
        "federation": best_personal, "config": config.as_dict(),
        "selected_round": best_personal_round, "validation_score": best_personal_score,
    })
    local_test_personalized, _ = evaluate_client_split(
        federation, cache, partition, config, "local_test"
    )
    official_test_personalized = evaluate_official(
        federation, cache, config, "test", personalized=True
    )

    federation.restore(best_global_client)
    global_client_method_summary = federation.method_summary()
    save_checkpoint(output / "best_global_client.pt", {
        "federation": best_global_client, "config": config.as_dict(),
        "selected_round": best_global_client_round,
        "validation_score": best_global_client_score,
    })
    _, local_test_global_client_selected = evaluate_client_split(
        federation, cache, partition, config, "local_test"
    )
    official_test_global_client_selected = evaluate_official(
        federation, cache, config, "test"
    )

    federation.restore(best_global_official)
    global_official_method_summary = federation.method_summary()
    save_checkpoint(output / "best_global_official.pt", {
        "federation": best_global_official, "config": config.as_dict(),
        "selected_round": best_global_official_round,
        "validation_score": best_global_official_score,
    })
    _, local_test_global_official_selected = evaluate_client_split(
        federation, cache, partition, config, "local_test"
    )
    official_test_global_official_selected = evaluate_official(
        federation, cache, config, "test"
    )

    result = {
        "status": "complete", "dataset": config.dataset, "algorithm": "fedtriad",
        "variant": "FedTriad-" + config.triad_variant,
        "seed": config.seed, "split_seed": config.split_seed,
        "partition_id": partition["partition_id"], "smoke_only": bool(config.smoke_samples),
        "selection_metric": config.selection_metric,
        "selected_personal_round": best_personal_round,
        "selected_global_client_round": best_global_client_round,
        "selected_global_official_round": best_global_official_round,
        "best_personal_validation_score": best_personal_score,
        "best_global_client_validation_score": best_global_client_score,
        "best_global_official_validation_score": best_global_official_score,
        "best_personal_validation": best_personal_validation,
        "best_global_client_validation": best_global_client_validation,
        "best_global_official_validation": best_global_official_validation,
        "local_test_personalized": local_test_personalized,
        "local_test_global_client_validation_selected": local_test_global_client_selected,
        "local_test_global_official_validation_selected": local_test_global_official_selected,
        "official_test_personalized_ensemble": official_test_personalized,
        "official_test_global_client_validation_selected": official_test_global_client_selected,
        "official_test_global_official_validation_selected": official_test_global_official_selected,
        "method_summary_all_rounds": all_rounds_summary,
        "method_summary_personal_checkpoint": personal_method_summary,
        "method_summary_global_client_checkpoint": global_client_method_summary,
        "method_summary_global_official_checkpoint": global_official_method_summary,
        "cost_all_rounds": cost,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "cost_note": "Tensor payload estimate; evaluation and serialization overhead are excluded.",
        "limitations": [
            "clients are synthetic Dirichlet partitions, not verified real hospitals",
            "high relative local responsibility indicates a client-specific candidate, not a clinical novelty label",
            "private models and class-risk summaries are not protected by differential privacy or secure aggregation",
            "personalized and deployable-global results must be compared with matching baseline categories",
        ],
    }
    write_json(output / "results.json", result)
    write_history(output / "history.csv", history)
    return result
