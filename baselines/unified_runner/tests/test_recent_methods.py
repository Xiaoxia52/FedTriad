import json
from pathlib import Path

import pytest

from fedrca.config import Config
from fedrca.runner import run
from fedrca.suite import expand_suite


def _settings(path, tmp_path, algorithm):
    return Config(
        dataset="bloodmnist", algorithm=algorithm, data_file=str(path),
        cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / algorithm),
        clients=2, participation_rate=0.5, alpha=1.0, min_train_samples=8,
        rounds=1, width=4, feature_dim=8, batch_size=32, device="cpu",
        cpu_threads=2, smoke_samples=96, fedtgp_server_epochs=2,
        fedsimsup_supervisor_width=4, fedsimsup_supervisor_feature_dim=8,
    )


@pytest.mark.parametrize("algorithm", [
    "fedtgp", "fedsol", "fedsa", "cwfedavg", "fedsimsup",
])
def test_recent_method_end_to_end_smoke(medical_file, tmp_path, algorithm):
    config = _settings(medical_file(), tmp_path, algorithm)
    result = run(config)
    assert result["status"] == "complete"
    assert result["algorithm"] == algorithm
    assert result["method_summary_all_rounds"]["rounds"] == 1
    assert result["local_test"]["aggregate"]["macro_f1_mean"] >= 0
    checkpoint = Path(tmp_path / algorithm / "last.pt")
    assert checkpoint.is_file()
    config.rounds = 2
    resumed = run(config, resume=True)
    assert resumed["status"] == "complete"
    assert resumed["method_summary_all_rounds"]["rounds"] == 2


def test_unified_seed_zero_suite_is_66_matched_jobs():
    root = Path(__file__).parents[1]
    spec = json.loads((root / "configs" / "main_3datasets_11methods.json").read_text())
    spec["seeds"] = [0]
    jobs = expand_suite(spec)
    assert len(jobs) == 66
    assert {job.dataset for job in jobs} == {"bloodmnist", "organamnist", "pathmnist"}
    assert {job.alpha for job in jobs} == {0.1, 0.5}
    assert {job.participation_rate for job in jobs} == {0.6}
    grouped = {}
    for job in jobs:
        grouped.setdefault((job.dataset, job.alpha, job.seed), set()).add(job.split_seed)
    assert all(values == {0} for values in grouped.values())
