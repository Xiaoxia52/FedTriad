import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fedrca.config import ALGORITHMS, Config
from fedrca.runner import acquire_run_lock, configure_reproducibility
from fedrca.suite import _resume_mode, expand_suite
from fedrca.algorithms import fedseq_greedy_groups


def test_all_algorithms_and_variants_validate():
    for algorithm in ALGORITHMS:
        Config(algorithm=algorithm).validate()
    for variant in ("full", "no_rbf", "no_regions", "random_regions", "single_region"):
        Config(algorithm="fedrca", fedrca_variant=variant).validate()


def test_invalid_topology_bandwidth_and_unknown_key_rejected():
    with pytest.raises(ValueError, match="minimum"):
        Config(fedrca_topology_bandwidth_min=2.0,
               fedrca_topology_bandwidth_max=1.0).validate()
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_dict({"obsolete_old_setting": True})


def test_300_round_delayed_multistep_learning_rate_boundaries():
    config = Config(
        rounds=300,
        lr_schedule="multistep",
        lr_decay_fractions="0.6,0.85",
        lr_decay_gamma=0.1,
    )
    assert config.learning_rate(0.01, 0) == pytest.approx(0.01)
    assert config.learning_rate(0.01, 179) == pytest.approx(0.01)
    assert config.learning_rate(0.01, 180) == pytest.approx(0.001)
    assert config.learning_rate(0.01, 254) == pytest.approx(0.001)
    assert config.learning_rate(0.01, 255) == pytest.approx(0.0001)
    assert config.learning_rate(0.01, 299) == pytest.approx(0.0001)
    with pytest.raises(ValueError, match="lr_schedule"):
        Config(lr_schedule="cosine").validate()
    with pytest.raises(ValueError, match="lr_decay_fractions"):
        Config(lr_decay_fractions="0.0,0.75").validate()


def test_all_suite_files_expand_and_unified_main_is_exactly_198_jobs():
    config_root = Path(__file__).parents[1] / "configs"
    for path in sorted(config_root.rglob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        assert expand_suite(spec), path
    path = config_root / "main_3datasets_11methods.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    jobs = expand_suite(spec)
    assert len(jobs) == 198
    assert len({job.algorithm for job in jobs}) == 11
    assert {job.dataset for job in jobs} == {"bloodmnist", "organamnist", "pathmnist"}
    assert {job.device for job in jobs} == {"cuda:0"}
    assert {job.calibration_fraction for job in jobs} == {0.0}
    assert {job.rounds for job in jobs} == {300}
    assert {job.lr_schedule for job in jobs} == {"multistep"}
    assert {job.lr_decay_fractions for job in jobs} == {"0.6,0.85"}
    assert {job.fedrca_pixel_max_clusters for job in jobs} == {4}
    assert {job.fedrca_topology_grid for job in jobs} == {7}
    assert {job.participation_rate for job in jobs} == {0.6}


def test_serial_suite_and_fedseq_grouping_are_complete_and_deterministic():
    path = Path(__file__).parents[1] / "configs" / "serial_baselines_3datasets_300r.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    jobs = expand_suite(spec)
    assert len(jobs) == 36
    assert {job.algorithm for job in jobs} == {"cwt", "fedseq"}
    assert {job.rounds for job in jobs} == {300}
    assert {job.participation_rate for job in jobs} == {0.6}
    counts = np.asarray([[20, 0, 0], [0, 20, 0], [0, 0, 20], [10, 10, 0], [0, 10, 10]])
    groups = fedseq_greedy_groups(counts, 2)
    assert groups == fedseq_greedy_groups(counts, 2)
    assert sorted(sum(groups, [])) == list(range(5))
    assert max(map(len, groups)) <= 3


def test_reproducibility_policy_and_stale_lock_recovery(tmp_path):
    configure_reproducibility(torch.device("cuda:0"))
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    configure_reproducibility(torch.device("cpu"))
    assert torch.are_deterministic_algorithms_enabled()

    lock = tmp_path / "run.lock"
    lock.write_text("99999999", encoding="utf-8")
    acquire_run_lock(lock)
    assert int(lock.read_text(encoding="utf-8")) > 0


def test_resume_restarts_only_recognized_checkpoint_free_shell(tmp_path):
    output = tmp_path / "bloodmnist_fedrca_interrupted"
    config = Config(clients=2, output_dir=str(output))
    output.mkdir()
    (output / "config.json").write_text(
        json.dumps(config.as_dict()), encoding="utf-8"
    )
    (output / "results.json").write_text(
        json.dumps({"status": "running", "dataset": "bloodmnist",
                    "algorithm": "fedrca"}), encoding="utf-8"
    )
    assert _resume_mode(config, resume=True) is False
    assert not output.exists()

    unknown = tmp_path / "bloodmnist_fedrca_unknown"
    config.output_dir = str(unknown)
    unknown.mkdir()
    (unknown / "notes.txt").write_text("user file", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unrecognized"):
        _resume_mode(config, resume=True)
    assert (unknown / "notes.txt").read_text(encoding="utf-8") == "user file"
