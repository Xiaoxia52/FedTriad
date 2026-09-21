"""One-time, client-private pixel-space partitions used by FedRCA.

The partition is fitted once from each client's official training subset.  It is
never refreshed from learned features and is never uploaded to the server.
"""
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


@dataclass
class PixelRegionIndex:
    assignments: np.ndarray
    weights: np.ndarray
    summaries: list
    kmeans_fits: int


def _adaptive_k(sample_count, config):
    if config.fedrca_variant in ("no_regions", "single_region"):
        return 1
    capacity = max(1, sample_count // config.fedrca_pixel_min_cluster_samples)
    estimate = max(1, int(np.floor(
        sample_count / config.fedrca_pixel_samples_per_cluster + 0.5
    )))
    return min(sample_count, capacity, config.fedrca_pixel_max_clusters, estimate)


def _pixel_descriptors(images, size):
    tensor = torch.from_numpy(np.asarray(images, dtype=np.float32)).permute(0, 3, 1, 2)
    if tensor.shape[-2:] != (size, size):
        tensor = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
    return tensor.flatten(1).div_(255.0).numpy()


def build_pixel_region_index(cache, partition, config):
    """Fit deterministic PCA+K-means once per client and class.

    Returned arrays are indexed by the original official-train row id.  A
    region-balanced per-sample weight makes the fixed partition affect the
    optimization without sharing raw pixels, assignments, or region centers.
    """
    images = np.load(str(cache / "train_images.npy"), mmap_mode="r", allow_pickle=False)
    labels = np.load(str(cache / "train_labels.npy"), mmap_mode="r", allow_pickle=False)
    assignments = np.full(len(labels), -1, dtype=np.int16)
    weights = np.ones(len(labels), dtype=np.float32)
    summaries = []
    total_fits = 0

    for client_id, client_indices_value in enumerate(partition["train"]):
        client_indices = np.asarray(client_indices_value, dtype=np.int64)
        group_counts = []
        group_rows = []
        client_summary = {"client": client_id, "classes": {}, "groups": 0}
        for label in np.unique(labels[client_indices]).tolist():
            rows = client_indices[labels[client_indices] == label]
            k = _adaptive_k(len(rows), config)
            if k == 1:
                local_assignments = np.zeros(len(rows), dtype=np.int64)
            elif config.fedrca_variant == "random_regions":
                rng = np.random.default_rng(
                    config.seed * 1000003 + client_id * 1009 + int(label) * 101
                )
                local_assignments = np.arange(len(rows), dtype=np.int64) % k
                rng.shuffle(local_assignments)
            else:
                descriptors = _pixel_descriptors(
                    np.asarray(images[rows]), config.fedrca_pixel_descriptor_size
                )
                # Every local sample is represented; only exact duplicate pixel
                # vectors limit the number of meaningful clusters.
                k = min(k, len(np.unique(descriptors, axis=0)))
                if k == 1:
                    local_assignments = np.zeros(len(rows), dtype=np.int64)
                    counts = np.bincount(local_assignments, minlength=k)
                    assignments[rows] = local_assignments
                    for region, count in enumerate(counts.tolist()):
                        selected = rows[local_assignments == region]
                        group_rows.append(selected)
                        group_counts.append(count)
                    client_summary["classes"][str(int(label))] = {
                        "samples": int(len(rows)), "regions": int(k),
                        "supports": [int(value) for value in counts.tolist()],
                    }
                    client_summary["groups"] += int(k)
                    continue
                components = min(
                    config.fedrca_pixel_pca_components,
                    descriptors.shape[1],
                    len(descriptors) - 1,
                )
                if 0 < components < descriptors.shape[1]:
                    descriptors = PCA(
                        n_components=components,
                        svd_solver="randomized",
                        random_state=config.seed * 1000003 + client_id * 1009 + int(label),
                    ).fit_transform(descriptors)
                with threadpool_limits(limits=1, user_api="blas"):
                    local_assignments = KMeans(
                        n_clusters=k,
                        init="k-means++",
                        n_init=config.fedrca_kmeans_n_init,
                        random_state=config.seed * 1000003 + client_id * 1009 + int(label) * 101,
                    ).fit_predict(descriptors)
                total_fits += 1

            counts = np.bincount(local_assignments, minlength=k)
            assignments[rows] = local_assignments
            for region, count in enumerate(counts.tolist()):
                selected = rows[local_assignments == region]
                group_rows.append(selected)
                group_counts.append(count)
            client_summary["classes"][str(int(label))] = {
                "samples": int(len(rows)), "regions": int(k),
                "supports": [int(value) for value in counts.tolist()],
            }
            client_summary["groups"] += int(k)

        maximum = max(group_counts) if group_counts else 1
        if config.fedrca_variant != "no_regions":
            for rows, count in zip(group_rows, group_counts):
                raw = ((maximum + config.fedrca_region_balance_smoothing) /
                       (count + config.fedrca_region_balance_smoothing)) ** \
                      config.fedrca_region_balance_power
                weights[rows] = min(raw, config.fedrca_region_balance_max_ratio)
        if len(client_indices):
            weights[client_indices] /= float(weights[client_indices].mean())
        summaries.append(client_summary)

    if any(assignments[np.asarray(indices, dtype=np.int64)].min() < 0
           for indices in partition["train"]):
        raise RuntimeError("Pixel-region construction left training samples unassigned")
    return PixelRegionIndex(assignments, weights, summaries, total_fits)
