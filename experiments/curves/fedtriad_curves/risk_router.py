"""Label-free inference fusion from compressed branch-by-class risk statistics."""

import numpy as np


def fit_global_class_risk(client_probabilities, client_labels, classes):
    """Aggregate per-client P/S/L class-NLL sums without sharing samples.

    Each client contributes only a 3 x C loss-sum table and a C-vector of
    class counts.  The server obtains the same global table as if those
    sufficient statistics had been summed centrally.
    """
    global_sums = np.zeros((3, classes), dtype=np.float64)
    global_counts = np.zeros(classes, dtype=np.int64)
    client_sums = []
    client_counts = []
    for probabilities, labels in zip(client_probabilities, client_labels):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if probabilities.ndim != 3 or probabilities.shape[1:] != (3, classes):
            raise ValueError("Expected client probabilities with shape [N, 3, C]")
        if labels.ndim != 1 or len(probabilities) != len(labels):
            raise ValueError("Probability/label length mismatch")
        if len(labels) and (labels.min() < 0 or labels.max() >= classes):
            raise ValueError("Labels are outside the configured class range")

        sums = np.zeros((3, classes), dtype=np.float64)
        counts = np.bincount(labels, minlength=classes).astype(np.int64)
        if len(labels):
            selected = probabilities[
                np.arange(len(labels))[:, None], np.arange(3)[None, :], labels[:, None]
            ]
            nll = -np.log(np.clip(selected, 1e-8, 1.0))
            for class_id in np.flatnonzero(counts):
                sums[:, class_id] = nll[labels == class_id].sum(axis=0)
        global_sums += sums
        global_counts += counts
        client_sums.append(sums)
        client_counts.append(counts)

    fallback = np.full((3, classes), np.log(float(classes)), dtype=np.float64)
    global_risk = np.divide(
        global_sums,
        global_counts[None, :],
        out=fallback,
        where=global_counts[None, :] > 0,
    )
    return {
        "global_risk": global_risk,
        "global_loss_sums": global_sums,
        "global_counts": global_counts,
        "client_loss_sums": np.asarray(client_sums),
        "client_counts": np.asarray(client_counts),
        "uploaded_scalars_per_client": int(4 * classes),
        "source": "compressed_client_branch_by_class_nll_sums_and_class_counts",
    }


def class_agnostic_risk(memory):
    """Collapse class structure while retaining each branch's overall reliability."""
    sums = np.asarray(memory["global_loss_sums"], dtype=np.float64)
    counts = np.asarray(memory["global_counts"], dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        overall = np.full(3, np.log(float(sums.shape[1])), dtype=np.float64)
    else:
        overall = sums.sum(axis=1) / total
    return np.repeat(overall[:, None], sums.shape[1], axis=1)


def fuse_risk(probabilities, class_risk, temperature=1.0,
              entropy_weight=0.1, weight_floor=0.05):
    """Fuse P/S/L using expected class risk and current predictive uncertainty."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_risk = np.asarray(class_risk, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1] != 3:
        raise ValueError("Expected probabilities with shape [N, 3, C]")
    if class_risk.shape != probabilities.shape[1:]:
        raise ValueError("Expected class risk with shape [3, C]")
    if temperature <= 0 or not 0 <= weight_floor < 1 / 3:
        raise ValueError("Invalid risk-fusion temperature or weight floor")

    consensus = probabilities.mean(axis=1)
    expected_risk = consensus @ class_risk.T
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=2)
    total_risk = expected_risk + entropy_weight * entropy
    logits = -total_risk / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    if weight_floor:
        weights = weight_floor + (1.0 - 3.0 * weight_floor) * weights
    fused = (weights[:, :, None] * probabilities).sum(axis=1)
    fused /= fused.sum(axis=1, keepdims=True)
    return fused, weights


def serializable_memory(memory):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in memory.items()
    }
