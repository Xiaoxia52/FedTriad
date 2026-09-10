import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fedtriad.algorithm import Federation
from fedtriad.config import Config, TRIAD_VARIANTS
from fedtriad.risk_router import (
    class_agnostic_risk,
    fit_global_class_risk,
    fuse_risk,
)
from fedtriad.suite import expand_suite


def test_locked_schedule_and_final_variants():
    config = Config(rounds=300)
    assert config.learning_rate(0.01, 179) == 0.01
    assert config.learning_rate(0.01, 180) == 0.001
    assert config.learning_rate(0.01, 255) == pytest.approx(0.0001)
    assert TRIAD_VARIANTS == ("p_only", "ps_uniform", "pl_uniform", "psl_uniform")


def _suite(tmp_path, variants):
    return {
        "base": {"rounds": 300},
        "datasets": {name: {} for name in ("bloodmnist", "organamnist", "pathmnist")},
        "variants": variants,
        "seeds": [0, 1, 2],
        "alphas": [0.5, 0.1],
        "output_root": str(tmp_path),
    }


def test_full_ablation_matrix_has_72_unique_jobs(tmp_path):
    configs = expand_suite(_suite(tmp_path, list(TRIAD_VARIANTS)))
    assert len(configs) == 72
    assert len({item.output_dir for item in configs}) == 72


def test_completion_matrix_has_36_entries_before_completed_runs_are_skipped(tmp_path):
    configs = expand_suite(_suite(tmp_path, ["pl_uniform", "psl_uniform"]))
    assert len(configs) == 36
    assert {item.triad_variant for item in configs} == {"pl_uniform", "psl_uniform"}


def test_complete_legacy_directory_is_reused_after_path_only_move(tmp_path):
    old = tmp_path / "bloodmnist_fedtriad_psl_uniform_a0.5_seed0_deadbeef"
    old.mkdir()
    config = Config(triad_variant="psl_uniform", rounds=300)
    saved = config.as_dict()
    saved["data_file"] = "../old-project/data/bloodmnist.npz"
    (old / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    (old / "results.json").write_text('{"status":"complete"}', encoding="utf-8")
    spec = {
        "base": {"rounds": 300, "data_file": "data/bloodmnist.npz"},
        "datasets": {"bloodmnist": {}},
        "variants": ["psl_uniform"], "seeds": [0], "alphas": [0.5],
        "output_root": str(tmp_path),
    }
    assert Path(expand_suite(spec)[0].output_dir) == old


def test_pl_has_parallel_and_private_paths_without_serial_path():
    config = Config(triad_variant="pl_uniform", clients=2, width=4, feature_dim=8)
    partition = {"train": [[0], [1]]}
    federation = Federation(config, Path("unused"), partition, torch.device("cpu"))
    assert federation.serial is None
    assert len(federation.locals) == 2
    parallel = torch.tensor([[2.0, 0.0]])
    local = torch.tensor([[0.0, 2.0]])
    personal = federation._mix((parallel, None, local), personalized=True)
    global_only = federation._mix((parallel, None, local), personalized=False)
    assert torch.allclose(personal, (parallel.softmax(1) + local.softmax(1)) / 2)
    assert torch.allclose(global_only, parallel.softmax(1))


def test_global_class_risk_uses_compressed_sums_and_agnostic_control():
    probabilities = [np.array([
        [[0.9, 0.1], [0.6, 0.4], [0.8, 0.2]],
        [[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]],
    ])]
    labels = [np.array([0, 1])]
    memory = fit_global_class_risk(probabilities, labels, classes=2)
    assert memory["global_risk"].shape == (3, 2)
    assert memory["uploaded_scalars_per_client"] == 8
    agnostic = class_agnostic_risk(memory)
    assert agnostic.shape == (3, 2)
    assert np.allclose(agnostic[:, 0], agnostic[:, 1])
    assert not np.allclose(memory["global_risk"][:, 0], memory["global_risk"][:, 1])


def test_risk_fusion_is_normalized_and_respects_weight_floor():
    probabilities = np.array([
        [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]],
        [[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]],
    ])
    risk = np.array([[0.1, 0.7], [0.4, 0.3], [0.6, 0.2]])
    fused, weights = fuse_risk(probabilities, risk, weight_floor=0.05)
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0.05)
