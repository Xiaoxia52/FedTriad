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


def _resume_mode(config, resume):
    """Return checkpoint-resume mode, or safely restart a checkpoint-free shell."""
    if not resume:
        return False
    output = Path(config.output_dir)
    checkpoint = output / "last.pt"
    if checkpoint.is_file():
        return True
    if not output.exists() or not any(output.iterdir()):
        return False

    result_path = output / "results.json"
    config_path = output / "config.json"
    if not result_path.is_file() or not config_path.is_file():
        raise RuntimeError("Refusing to replace an unrecognized nonempty output: " + str(output))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if result.get("status") != "running" or saved_config != config.as_dict():
        raise RuntimeError("Checkpoint-free output does not match this interrupted job: " + str(output))
    lock = output / "run.lock"
    if lock.exists():
        try:
            process_id = int(lock.read_text(encoding="utf-8").strip())
            os.kill(process_id, 0)
        except (OSError, ValueError):
            process_id = None
        else:
            raise RuntimeError("Checkpoint-free job is still active in process %s" % process_id)
    expected_prefix = "%s_%s_" % (config.dataset, config.algorithm)
    if output.name == "" or not output.name.startswith(expected_prefix):
        raise RuntimeError("Unsafe checkpoint-free output path: " + str(output))
    shutil.rmtree(output)
    print("Restarting checkpoint-free interrupted job: %s" % output, flush=True)
    return False


def _execute_one(index, total, values, resume):
    """Top-level worker so Windows spawn can run independent CUDA jobs safely."""
    config = Config.from_dict(values)
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
    allowed = {"base", "datasets", "algorithms", "seeds", "alphas", "output_root"}
    if set(spec) - allowed:
        raise ValueError("Unknown suite fields: %s" % sorted(set(spec) - allowed))
    configs = []
    for dataset, algorithm, seed, alpha in itertools.product(
            spec["datasets"], spec["algorithms"], spec["seeds"], spec["alphas"]):
        values = dict(spec.get("base", {}))
        values.update(spec["datasets"][dataset])
        values.update(dataset=dataset, algorithm=algorithm, seed=seed, split_seed=seed, alpha=alpha)
        config = Config.from_dict(values)
        name = "%s_%s_a%s_seed%s" % (dataset, algorithm, alpha, seed)
        fingerprint = dict(config.as_dict())
        for key in ("output_dir", "cache_dir"):
            fingerprint.pop(key)
        suffix = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:8]
        config.output_dir = str(Path(spec["output_root"]) / (name + "_" + suffix))
        configs.append(config)
    if len({c.output_dir for c in configs}) != len(configs):
        raise ValueError("Duplicate suite jobs")
    return configs


def execute_suite(spec, execute=False, resume=False, jobs=1, skip_complete=False):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    configs = expand_suite(spec)
    if skip_complete:
        pending = []
        for config in configs:
            result_path = Path(config.output_dir) / "results.json"
            complete = False
            if result_path.is_file():
                try:
                    complete = json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete"
                except (OSError, ValueError):
                    pass
            if not complete:
                pending.append(config)
        skipped = len(configs) - len(pending)
        configs = pending
        print("Complete outputs skipped: %s; pending: %s" % (skipped, len(configs)), flush=True)
    if not execute:
        print(json.dumps({"jobs": len(configs), "execution": "preview only; no training",
                          "parallel_jobs": jobs,
                          "configs": [c.as_dict() for c in configs]}, ensure_ascii=False, indent=2))
        return
    if not configs:
        print("All suite outputs are already complete.", flush=True)
        return
    devices = {c.device for c in configs}
    if len(devices) != 1:
        raise ValueError("Use separate suites for different devices")
    if devices == {"cpu"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from .data import locate, inspect_npz, prepare
    for c in configs:
        info = inspect_npz(locate(c))
        if not c.allow_resize and info["arrays"]["train_images"]["shape"][1] != c.image_size:
            raise ValueError("Resolution mismatch in suite")
    # Build each disk cache once in the parent. Concurrent workers can then read
    # memory-mapped arrays without racing on cache creation.
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
        futures = [executor.submit(
            _execute_one, index, len(configs), config.as_dict(), resume
        ) for index, config in enumerate(configs, 1)]
        for future in as_completed(futures):
            output_dir, status = future.result()
            print("Completed: %s (%s)" % (output_dir, status), flush=True)


def summarize(root, output, include_smoke=False):
    groups, seen = {}, set()
    for path in sorted(Path(root).rglob("results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or (result.get("smoke_only") and not include_smoke):
            continue
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        env = json.loads((path.parent / "environment.json").read_text(encoding="utf-8"))
        data = json.loads((path.parent / "data_manifest.json").read_text(encoding="utf-8"))
        settings = dict(config)
        for key in ("seed", "split_seed", "output_dir", "cache_dir", "data_root", "data_file"):
            settings.pop(key)
        settings["implementation_sha256"] = env["implementation_sha256"]
        settings["data_arrays"] = data["arrays"]
        group = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        unique = (group, result["seed"], result["split_seed"])
        if unique in seen:
            raise ValueError("Duplicate seed run; do not double-count it: " + str(path))
        seen.add(unique)
        groups.setdefault(group, []).append((result, config))
    records = []
    for group, entries in groups.items():
        first, config = entries[0]
        row = {"group_id": group, "dataset": first["dataset"], "algorithm": first["algorithm"],
               "variant": first["variant"], "alpha": config["alpha"], "clients": config["clients"],
               "image_size": config["image_size"], "rounds": config["rounds"], "runs": len(entries),
               "smoke_only": first["smoke_only"]}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "auroc"):
            values = [r["local_test"]["aggregate"][metric + "_mean"] for r, _ in entries]
            values = [x for x in values if x is not None]
            row["local_" + metric + "_mean"] = statistics.mean(values) if values else None
            row["local_" + metric + "_seed_std"] = statistics.stdev(values) if len(values) > 1 else None
            row["local_" + metric + "_valid_runs"] = len(values)
        row["communication_bytes_mean"] = statistics.mean(r["cost_all_rounds"]["communication_bytes"] for r, _ in entries)
        records.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No eligible completed runs (smoke excluded by default)")
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("Wrote %s groups to %s" % (len(records), output))
    return records
