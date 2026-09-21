import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedrca.algorithms import (
    personal_class_weights,
    rbf_topology_loss,
    rbf_topology_probabilities,
    train_local,
)
from fedrca.config import Config
from fedrca.models import MedicalCNN
from fedrca.pixel_regions import build_pixel_region_index
from fedrca.runner import run


def test_rbf_topology_is_normalized_and_matches_identical_maps():
    torch.manual_seed(3)
    feature_map = torch.randn(4, 8, 7, 7)
    config = Config(clients=2, fedrca_topology_grid=4)
    probabilities = rbf_topology_probabilities(
        feature_map, torch.tensor(0.0), config.fedrca_topology_grid
    )
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4, 16), atol=1e-6)
    loss = rbf_topology_loss(
        [feature_map, feature_map], [feature_map, feature_map],
        torch.zeros(2), config,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_topology_alignment_excludes_classifier_head():
    torch.manual_seed(5)
    student = MedicalCNN(3, 8, width=4, feature_dim=8)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        teacher.encoder.conv1.weight.add_(0.1)
    images = torch.randn(6, 3, 28, 28)
    student_maps = student.forward_with_maps(images)[2]
    with torch.no_grad():
        teacher_maps = teacher.forward_with_maps(images)[2]
    loss = rbf_topology_loss(
        student_maps, teacher_maps, student.topology_log_bandwidth,
        Config(clients=2, width=4, feature_dim=8),
    )
    loss.backward()
    assert student.head.weight.grad is None
    assert student.encoder.conv1.weight.grad is not None
    assert torch.isfinite(student.encoder.conv1.weight.grad).all()


def test_one_time_pixel_partition_is_deterministic_and_private(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    images = np.zeros((24, 8, 8, 1), dtype=np.uint8)
    images[3:6, 4:, :, 0] = 255
    images[6:12, :, 4:, 0] = 255
    images[9:12, 4:, :, 0] = 255
    images[15:18, :, 4:, 0] = 255
    images[18:24, 4:, :, 0] = 255
    images[21:24, :, :4, 0] = 255
    labels = np.repeat([0, 1, 0, 1], 6).astype(np.int64)
    np.save(cache / "train_images.npy", images)
    np.save(cache / "train_labels.npy", labels)
    partition = {"train": [list(range(12)), list(range(12, 24))]}
    config = Config(
        clients=2, fedrca_pixel_max_clusters=2,
        fedrca_pixel_samples_per_cluster=3,
        fedrca_pixel_min_cluster_samples=2,
        fedrca_pixel_descriptor_size=4,
        fedrca_pixel_pca_components=4,
    )
    first = build_pixel_region_index(cache, partition, config)
    second = build_pixel_region_index(cache, partition, config)
    assert np.array_equal(first.assignments, second.assignments)
    assert np.allclose(first.weights, second.weights)
    assert first.kmeans_fits == second.kmeans_fits == 4
    assert all(summary["groups"] == 4 for summary in first.summaries)
    for indices in partition["train"]:
        assert first.weights[indices].mean() == pytest.approx(1.0)


def test_personal_class_balance_and_detached_private_head():
    config = Config(width=4, feature_dim=8, clients=2, lr=0.01)
    weights = personal_class_weights([100, 4, 0], config)
    assert weights[1] > weights[0] > 0
    assert weights[2] == 0

    torch.manual_seed(7)
    reference = MedicalCNN(3, 8, config.width, config.feature_dim)
    global_only = copy.deepcopy(reference)
    dual_head = copy.deepcopy(reference)
    personal_head = copy.deepcopy(reference.head)
    images = torch.rand(16, 3, 28, 28)
    labels = torch.arange(16) % 8
    batches = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)
    train_local(global_only, batches, config, "cpu", epochs=1)
    train_local(
        dual_head, batches, config, "cpu", epochs=1,
        personal_head=personal_head, class_weights=torch.ones(8),
    )
    assert all(torch.equal(left, right) for left, right in zip(
        global_only.state_dict().values(), dual_head.state_dict().values()
    ))


def _settings(path, tmp_path, output, rounds=2):
    return Config(
        dataset="bloodmnist", algorithm="fedrca", data_file=str(path),
        cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / output),
        clients=2, participation_rate=0.5, alpha=1.0,
        min_train_samples=8, rounds=rounds,
        width=4, feature_dim=8, batch_size=32, device="cpu", cpu_threads=2,
        smoke_samples=96, fedrca_warmup_rounds=0, fedrca_ramp_rounds=1,
        fedrca_pixel_min_cluster_samples=2, fedrca_pixel_samples_per_cluster=4,
        fedrca_pixel_max_clusters=2, fedrca_pixel_descriptor_size=8,
        fedrca_pixel_pca_components=4, fedrca_topology_grid=4,
    )


def test_fedrca_smoke_checkpoint_and_resume(medical_file, tmp_path):
    path = medical_file()
    full = _settings(path, tmp_path, "full")
    result = run(full)
    assert result["status"] == "complete"
    assert result["variant"] == "FedRCA-full"
    assert result["method_summary_all_rounds"]["active_rounds"] == 2
    assert result["method_summary_all_rounds"]["pixel_regions"] > 0
    assert result["method_summary_all_rounds"]["pixel_partition_builds"] == 1
    assert result["cost_all_rounds"]["statistic_examples"] == 0
    assert result["cost_all_rounds"]["preprocessing_examples"] > 0
    assert result["official_test"]["policy"] == "validation_selected_global_model"
    assert result["official_test_global"]["policy"] == "trained_global_model"
    assert result["official_test_personalized_ensemble"]["policy"] == "equal_probability_ensemble_of_personalized_models"
    assert result["local_test_global"] is not None
    assert result["selected_global_round"] > 0
    assert result["global_validation"]["test_tuning"] is False
    assert "official_test_rbf" not in result
    assert (Path(full.output_dir) / "best_personalized.pt").is_file()
    assert (Path(full.output_dir) / "best_global.pt").is_file()

    partial = _settings(path, tmp_path, "resumed", rounds=1)
    run(partial)
    partial.rounds = 2
    resumed = run(partial, resume=True)
    assert resumed["status"] == "complete"
    a = torch.load(Path(full.output_dir) / "last.pt", weights_only=False)
    b = torch.load(Path(partial.output_dir) / "last.pt", weights_only=False)
    assert a["federation"]["round"] == b["federation"]["round"] == 2
    assert a["federation"]["bytes_sent"] == b["federation"]["bytes_sent"]
    assert a["best_global_round"] == b["best_global_round"]
    assert a["best_global_validation"] == b["best_global_validation"]
    assert len(a["federation"]["personal_heads"]) == 2
    assert a["federation"]["pixel_kmeans_fits"] == b["federation"]["pixel_kmeans_fits"]
    assert np.array_equal(
        a["federation"]["pixel_region_assignments"],
        b["federation"]["pixel_region_assignments"],
    )
    for left, right in zip(a["federation"]["clients"], b["federation"]["clients"]):
        assert all(torch.equal(left[key], right[key]) for key in left)
