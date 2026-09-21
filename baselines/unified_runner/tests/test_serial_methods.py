from pathlib import Path

from fedrca.config import Config
from fedrca.runner import run


def _settings(path, tmp_path, algorithm):
    return Config(
        dataset="bloodmnist", algorithm=algorithm, data_file=str(path),
        cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / algorithm),
        clients=5, participation_rate=0.6, alpha=1.0,
        min_train_samples=8, rounds=1, local_epochs=1,
        width=4, feature_dim=8, batch_size=32, device="cpu", cpu_threads=2,
        smoke_samples=160, fedseq_superclients=2,
    )


def test_cwt_and_fedseq_train_the_protocol_matched_active_budget(medical_file, tmp_path):
    path = medical_file()
    for algorithm in ("cwt", "fedseq"):
        config = _settings(path, tmp_path, algorithm)
        result = run(config)
        assert result["status"] == "complete"
        assert result["official_test"]["policy"] == "global_model"
        assert result["cost_all_rounds"]["train_examples"] > 0
        checkpoint = Path(config.output_dir) / "last_algorithm_diagnostics.json"
        assert checkpoint.is_file()
        summary = result["method_summary_all_rounds"]
        assert summary["rounds"] == 1
        assert len(summary["last"]["active_clients_in_order"]) == 3
        if algorithm == "fedseq":
            groups = summary["superclient_groups"]
            assert sorted(sum(groups, [])) == list(range(5))
            assert result["cost_all_rounds"]["preprocessing_examples"] > 0
