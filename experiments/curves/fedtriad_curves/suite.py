import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import statistics
import traceback

from .config import Config


def _complete_legacy_output(root, name, config):
    """Reuse a unique completed run after path-only project reorganization."""
    matches = []
    for directory in Path(root).glob(name + "_*"):
        result_path = directory / "results.json"
        config_path = directory / "config.json"
        if not result_path.is_file() or not config_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            saved = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if result.get("status") != "complete":
            continue
        required = {
            "dataset": config.dataset,
            "triad_variant": config.triad_variant,
            "seed": config.seed,
            "split_seed": config.split_seed,
            "alpha": config.alpha,
            "rounds": config.rounds,
            "clients": config.clients,
            "participation_rate": config.participation_rate,
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
        }
        if all(saved.get(key) == value for key, value in required.items()):
            matches.append(directory)
    if len(matches) > 1:
        raise RuntimeError("Ambiguous completed legacy runs for " + name)
    return matches[0] if matches else None


def _resume_mode(config, resume):
    if not resume:
        return False
    output = Path(config.output_dir)
    checkpoint = output / "last.pt"
    if checkpoint.is_file():
        return True
    if not output.exists() or not any(output.iterdir()):
        return False
    expected = "%s_fedtriad_%s_" % (config.dataset, config.triad_variant)
    if not output.name.startswith(expected):
        raise RuntimeError("Unsafe interrupted output path: " + str(output))
    entries = {path.name for path in output.iterdir()}
    if entries <= {"failure.log", "run.lock"}:
        from .runner import _process_is_running
        lock = output / "run.lock"
        try:
            process_id = int(lock.read_text(encoding="utf-8").strip()) if lock.exists() else None
        except (OSError, ValueError):
            process_id = None
        if _process_is_running(process_id):
            raise RuntimeError("Checkpoint-free job is active in process %s" % process_id)
        shutil.rmtree(output)
        return False
    result_path, config_path = output / "results.json", output / "config.json"
    if not result_path.is_file() or not config_path.is_file():
        raise RuntimeError("Refusing to replace unrecognized nonempty output: " + str(output))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    if result.get("status") != "running" or saved != config.as_dict():
        raise RuntimeError("Checkpoint-free output does not match this job: " + str(output))
    from .runner import _process_is_running
    lock = output / "run.lock"
    try:
        process_id = int(lock.read_text(encoding="utf-8").strip()) if lock.exists() else None
    except (OSError, ValueError):
        process_id = None
    if _process_is_running(process_id):
        raise RuntimeError("Checkpoint-free job is active in process %s" % process_id)
    shutil.rmtree(output)
    return False


def _execute_one(index, total, values, resume):
    config = Config.from_dict(values)
    result_path = Path(config.output_dir) / "results.json"
    if resume and result_path.is_file():
        try:
            if json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
                print("Skipped complete job %s/%s: %s" % (index, total, config.output_dir), flush=True)
                return config.output_dir, "complete"
        except (OSError, ValueError):
            pass
    print("Job %s/%s: %s" % (index, total, config.output_dir), flush=True)
    from .runner import run
    try:
        result = run(config, resume=_resume_mode(config, resume))
    except BaseException:
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return config.output_dir, result["status"]


def expand_suite(spec):
    allowed = {"base", "datasets", "variants", "seeds", "alphas", "output_root"}
    if set(spec) - allowed:
        raise ValueError("Unknown suite fields: %s" % sorted(set(spec) - allowed))
    configs = []
    for dataset, variant, seed, alpha in itertools.product(
            spec["datasets"], spec["variants"], spec["seeds"], spec["alphas"]):
        values = dict(spec.get("base", {}))
        values.update(spec["datasets"][dataset])
        values.update(dataset=dataset, algorithm="fedtriad", triad_variant=variant,
                      seed=seed, split_seed=seed, alpha=alpha)
        config = Config.from_dict(values)
        name = "%s_fedtriad_%s_a%s_seed%s" % (dataset, variant, alpha, seed)
        fingerprint = dict(config.as_dict())
        for key in ("output_dir", "cache_dir"):
            fingerprint.pop(key)
        suffix = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:8]
        config.output_dir = str(Path(spec["output_root"]) / (name + "_" + suffix))
        legacy = _complete_legacy_output(spec["output_root"], name, config)
        if legacy is not None:
            config.output_dir = str(legacy)
        configs.append(config)
    if len({config.output_dir for config in configs}) != len(configs):
        raise ValueError("Duplicate suite jobs")
    return configs


def execute_suite(spec, resume=False, jobs=1):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    configs = expand_suite(spec)
    from .data import inspect_npz, locate, prepare
    for config in configs:
        info = inspect_npz(locate(config))
        if not config.allow_resize and info["arrays"]["train_images"]["shape"][1] != config.image_size:
            raise ValueError("Resolution mismatch in suite")
    prepared = set()
    for config in configs:
        key = (str(Path(locate(config)).resolve()), config.image_size,
               config.smoke_samples, str(Path(config.cache_dir).resolve()))
        if key not in prepared:
            prepare(config)
            prepared.add(key)
    if jobs == 1:
        for index, config in enumerate(configs, 1):
            _execute_one(index, len(configs), config.as_dict(), resume)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
        futures = [executor.submit(_execute_one, index, len(configs), config.as_dict(), resume)
                   for index, config in enumerate(configs, 1)]
        for future in as_completed(futures):
            output_dir, status = future.result()
            print("Completed: %s (%s)" % (output_dir, status), flush=True)


def summarize(root, output, include_smoke=False):
    groups = {}
    for path in sorted(Path(root).rglob("results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or (result.get("smoke_only") and not include_smoke):
            continue
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        key = (result["dataset"], result["variant"], config["alpha"])
        groups.setdefault(key, []).append(result)
    records = []
    for (dataset, variant, alpha), entries in sorted(groups.items()):
        row = {"dataset": dataset, "variant": variant, "alpha": alpha,
               "runs": len(entries)}
        sources = {
            "personal": "local_test_personalized",
            "global_client_selected": "local_test_global_client_validation_selected",
        }
        for prefix, source in sources.items():
            for metric in ("accuracy", "macro_f1", "balanced_accuracy", "auroc"):
                values = [item[source]["aggregate"][metric + "_mean"] for item in entries]
                values = [value for value in values if value is not None]
                row[prefix + "_" + metric + "_mean"] = statistics.mean(values) if values else None
                row[prefix + "_" + metric + "_seed_std"] = (
                    statistics.stdev(values) if len(values) > 1 else None
                )
        row["communication_bytes_mean"] = statistics.mean(
            item["cost_all_rounds"]["communication_bytes"] for item in entries
        )
        records.append(row)
    if not records:
        raise ValueError("No eligible completed runs")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return records
