import hashlib
import json
import numpy as np


def build_partition(train_labels, val_labels, config, data_id):
    """Synthetic clients. Local test comes ONLY from official train, val stays val."""
    rng = np.random.default_rng(config.split_seed)
    classes = int(max(train_labels.max(), val_labels.max())) + 1
    for attempt in range(config.partition_attempts):
        pools = [[] for _ in range(config.clients)]
        for label in range(classes):
            indices = np.flatnonzero(train_labels == label)
            rng.shuffle(indices)
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(config.clients, config.alpha)))
            for client, part in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                pools[client].extend(part.tolist())
        trains, calibrations, tests = [], [], []
        for pool in pools:
            train, calibration, test = [], [], []
            pool = np.asarray(pool, dtype=np.int64)
            for label in range(classes):
                indices = pool[train_labels[pool] == label]
                rng.shuffle(indices)
                hold = min(len(indices) - 1, max(1, int(round(len(indices) * config.local_test_fraction)))) if len(indices) > 1 else 0
                test.extend(indices[:hold].tolist())
                remaining = indices[hold:]
                cal = (min(len(remaining) - 1,
                           max(1, int(round(len(remaining) * config.calibration_fraction))))
                       if config.calibration_fraction > 0 and len(remaining) > 1 else 0)
                calibration.extend(remaining[:cal].tolist())
                train.extend(remaining[cal:].tolist())
            trains.append(train)
            calibrations.append(calibration)
            tests.append(test)
        if any(len(t) < config.min_train_samples or len(np.unique(train_labels[t])) < config.min_classes
               or not tests[i] for i, t in enumerate(trains)):
            continue
        if config.calibration_fraction > 0:
            if any(len(calibrations[i]) < config.calibration_min_samples or
                   np.count_nonzero(np.bincount(
                       train_labels[calibrations[i]], minlength=classes
                   ) >= 2) < config.calibration_min_classes
                   for i in range(config.clients)):
                continue
        hist = np.array([np.bincount(train_labels[t], minlength=classes) for t in trains])
        vals = [[] for _ in range(config.clients)]
        for label in range(classes):
            indices = np.flatnonzero(val_labels == label)
            rng.shuffle(indices)
            probs = hist[:, label].astype(float)
            if probs.sum() == 0:
                probs[:] = 1
            counts = rng.multinomial(len(indices), probs / probs.sum())
            for client, part in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                vals[client].extend(part.tolist())
        if any(not v for v in vals):
            continue
        calibration_hist = np.array([
            np.bincount(train_labels[indices], minlength=classes) for indices in calibrations
        ])
        manifest = {"protocol": "official-train/client-train-calibration-localtest;official-val/client-val;official-test/external",
                    "group_isolation": "not verified: standard NPZ contains no patient/slide/lesion IDs",
                    "data_id": data_id, "split_seed": config.split_seed, "alpha": config.alpha,
                    "clients": config.clients, "attempt": attempt + 1,
                    "train": trains, "calibration": calibrations,
                    "local_test": tests, "val": vals,
                    "train_class_counts": hist.tolist(),
                    "calibration_class_counts": calibration_hist.tolist()}
        validate_partition(manifest, len(train_labels), len(val_labels))
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        manifest["partition_id"] = digest
        return manifest
    raise ValueError("No valid partition after %s attempts. Reduce clients/minimum constraints or increase alpha. "
                     "No silent fallback to an invalid partition." % config.partition_attempts)


def validate_partition(manifest, train_size, val_size):
    train_and_test = [i for parts in (manifest["train"], manifest.get("calibration", []),
                                      manifest["local_test"])
                      for p in parts for i in p]
    validation = [i for p in manifest["val"] for i in p]
    if sorted(train_and_test) != list(range(train_size)):
        raise ValueError("Train/local-test indices overlap, are missing, or out of bounds")
    if sorted(validation) != list(range(val_size)):
        raise ValueError("Validation indices overlap, are missing, or out of bounds")
