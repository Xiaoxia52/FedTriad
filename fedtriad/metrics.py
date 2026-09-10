import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def classification_metrics(labels, probabilities, classes):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    if len(labels) == 0 or probabilities.shape != (len(labels), classes):
        raise ValueError("Empty/misaligned evaluation arrays")
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite predictions")
    predictions = probabilities.argmax(axis=1)
    cm = np.bincount(labels * classes + predictions, minlength=classes ** 2).reshape(classes, classes)
    support = cm.sum(axis=1)
    denom = support + cm.sum(axis=0)
    f1 = np.divide(2 * np.diag(cm), denom, out=np.zeros(classes, dtype=float), where=denom > 0)
    recall = np.divide(np.diag(cm), support, out=np.zeros(classes, dtype=float), where=support > 0)
    aucs = []
    for c in range(classes):
        target = labels == c
        aucs.append(float(roc_auc_score(target, probabilities[:, c])) if target.any() and not target.all() else None)
    valid = [a for a in aucs if a is not None]
    return {"n": len(labels), "accuracy": float((predictions == labels).mean()),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[support > 0].mean()),
            "auroc": float(np.mean(valid)) if valid else None, "auroc_valid_classes": len(valid),
            "auroc_per_class": aucs, "recall_per_class": [float(r) if s else None for r, s in zip(recall, support)],
            "class_support": support.tolist(), "confusion_matrix": cm.tolist(),
            "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean())}


def aggregate_metrics(metrics):
    result = {"clients": len(metrics), "samples": sum(m["n"] for m in metrics)}
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "auroc", "nll"):
        valid = [(m[key], m["n"]) for m in metrics if m[key] is not None]
        values = [v for v, _ in valid]
        result[key + "_mean"] = float(np.mean(values)) if values else None
        result[key + "_weighted"] = float(np.average(values, weights=[n for _, n in valid])) if values else None
        result[key + "_min"] = float(min(values)) if values else None
        result[key + "_std"] = float(np.std(values)) if values else None
        result[key + "_eligible_clients"] = len(values)
    return result


def predict(model, loader, device, prototypes=None):
    """Return labels and classifier (or native prototype) probabilities."""
    # Device transfers must stay outside inference mode.  Otherwise CUDA->CPU
    # conversion can replace BatchNorm buffers with inference tensors, which
    # later reject load_state_dict's in-place checkpoint restore.
    model.to(device).eval()
    ys, parts = [], []
    try:
        with torch.inference_mode():
            for images, labels in loader:
                features, logits = model(images.to(device))
                if prototypes is not None:
                    if not prototypes:
                        raise ValueError("FedProto requires prototypes before inference")
                    # FedProto native nearest-prototype evaluation, not its unused local linear head.
                    logits = torch.full_like(logits, -1e9)
                    for label, center in prototypes.items():
                        logits[:, int(label)] = -(features - center.to(device)).square().mean(dim=1)
                parts.append(logits.softmax(dim=1).cpu().numpy())
                ys.append(labels.numpy())
    finally:
        model.cpu()
    return np.concatenate(ys), np.concatenate(parts)
