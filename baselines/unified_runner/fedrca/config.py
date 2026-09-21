from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path


DATASETS = {
    "organsmnist": (1, 11),
    "organamnist": (1, 11),
    "bloodmnist": (3, 8),
    "pathmnist": (3, 9),
    "dermamnist": (3, 7),
}
ALGORITHMS = (
    "local", "fedavg", "fedprox", "fedrep", "ditto", "fedbn",
    "fedap", "fedproto", "fedpac", "fedtgp", "fedsol", "fedsa",
    "cwfedavg", "fedsimsup", "cwt", "fedseq", "fedrca",
)
FEDRCA_VARIANTS = ("full", "no_rbf", "no_regions", "random_regions", "single_region")


@dataclass
class Config:
    dataset: str = "bloodmnist"
    algorithm: str = "fedrca"
    data_root: str = "data"
    data_file: str = ""
    cache_dir: str = ".cache/medmnist"
    output_dir: str = "runs/default"
    image_size: int = 28
    allow_resize: bool = False
    clients: int = 5
    participation_rate: float = 1.0
    alpha: float = 0.5
    seed: int = 0
    split_seed: int = 0
    local_test_fraction: float = 0.2
    calibration_fraction: float = 0.0
    calibration_min_samples: int = 16
    calibration_min_classes: int = 2
    min_train_samples: int = 16
    min_classes: int = 2
    partition_attempts: int = 1000
    rounds: int = 50
    local_epochs: int = 1
    head_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    head_lr: float = 0.01
    lr_schedule: str = "constant"
    lr_decay_fractions: str = "0.5,0.75"
    lr_decay_gamma: float = 0.1
    momentum: float = 0.0
    weight_decay: float = 0.0005
    feature_dim: int = 128
    width: int = 32
    device: str = "cpu"
    workers: int = 0
    cpu_threads: int = 2
    eval_every: int = 1
    selection_metric: str = "macro_f1"
    prox_mu: float = 0.01
    ditto_lambda: float = 0.1
    prototype_lambda: float = 1.0
    fedtgp_lambda: float = 10.0
    fedtgp_server_epochs: int = 100
    fedtgp_server_lr: float = 0.01
    fedtgp_margin_threshold: float = 100.0
    fedsol_rho: float = 1.0
    fedsol_temperature: float = 3.0
    fedsa_anchor_lambda: float = 10.0
    fedsa_margin_lambda: float = 1.0
    fedsa_calibration_lambda: float = 1.0
    fedsa_anchor_momentum: float = 0.9999
    cwfedavg_wdr_lambda: float = 10.0
    fedsimsup_c: float = 40.0
    fedsimsup_gamma: float = 0.42857142857142855
    fedsimsup_supervisor_width: int = 12
    fedsimsup_supervisor_feature_dim: int = 48
    fedap_self_weight: float = 0.5
    fedap_warmup_rounds: int = 5
    fedap_mode: str = "reference"
    fedseq_superclients: int = 2

    # FedRCA: one-time local pixel partitions + homogeneous-model RBF topology KD.
    fedrca_variant: str = "full"
    fedrca_warmup_rounds: int = 5
    fedrca_ramp_rounds: int = 10
    fedrca_pixel_max_clusters: int = 4
    fedrca_pixel_samples_per_cluster: int = 64
    fedrca_pixel_min_cluster_samples: int = 8
    fedrca_pixel_descriptor_size: int = 28
    fedrca_pixel_pca_components: int = 64
    fedrca_kmeans_n_init: int = 10
    fedrca_topology_lambda: float = 0.1
    fedrca_topology_grid: int = 7
    fedrca_topology_bandwidth_init: float = 1.0
    fedrca_topology_bandwidth_min: float = 0.05
    fedrca_topology_bandwidth_max: float = 20.0
    fedrca_freeze_bn: bool = True
    fedrca_max_grad_norm: float = 5.0
    fedrca_personal_loss_weight: float = 1.0
    fedrca_class_balance_power: float = 0.5
    fedrca_class_balance_smoothing: float = 5.0
    fedrca_class_balance_max_ratio: float = 3.0
    fedrca_region_balance_power: float = 0.5
    fedrca_region_balance_smoothing: float = 5.0
    fedrca_region_balance_max_ratio: float = 3.0
    smoke_samples: int = 0

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
        defaults = cls()
        for key, value in values.items():
            expected = type(getattr(defaults, key))
            if expected is float and type(value) in (int, float):
                continue
            if type(value) is not expected:
                raise ValueError("Invalid type for %s: expected %s" % (key, expected.__name__))
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        if self.dataset not in DATASETS or self.algorithm not in ALGORITHMS:
            raise ValueError("Unsupported dataset/algorithm")
        positive_ints = (
            "clients", "rounds", "local_epochs", "head_epochs", "batch_size", "feature_dim",
            "width", "eval_every", "min_train_samples", "min_classes", "partition_attempts",
            "calibration_min_samples", "calibration_min_classes",
            "cpu_threads", "fedrca_ramp_rounds", "fedrca_pixel_max_clusters",
            "fedrca_pixel_samples_per_cluster", "fedrca_pixel_min_cluster_samples",
            "fedrca_pixel_descriptor_size", "fedrca_pixel_pca_components",
            "fedrca_kmeans_n_init", "fedrca_topology_grid",
            "fedtgp_server_epochs", "fedsimsup_supervisor_width",
            "fedsimsup_supervisor_feature_dim", "fedseq_superclients",
        )
        for name in positive_ints:
            if getattr(self, name) < 1:
                raise ValueError(name + " must be positive")
        if self.clients < 2:
            raise ValueError("At least two clients required")
        if self.fedseq_superclients > self.clients:
            raise ValueError("fedseq_superclients cannot exceed clients")
        if self.image_size not in (28, 64, 128, 224):
            raise ValueError("image_size must be 28, 64, 128, or 224")
        if self.min_classes > DATASETS[self.dataset][1]:
            raise ValueError("min_classes exceeds task class count")
        if not 0 < self.local_test_fraction < 0.5:
            raise ValueError("local_test_fraction must lie in (0, 0.5)")
        if not 0 <= self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must lie in [0, 0.5)")
        for name in ("alpha", "lr", "head_lr", "lr_decay_gamma",
                     "fedrca_topology_bandwidth_init", "fedrca_topology_bandwidth_min",
                     "fedrca_topology_bandwidth_max",
                     "fedrca_max_grad_norm", "fedtgp_server_lr",
                     "fedtgp_margin_threshold", "fedsol_rho", "fedsol_temperature",
                     "fedsimsup_c", "fedsimsup_gamma"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("prox_mu", "ditto_lambda", "prototype_lambda", "weight_decay",
                     "fedrca_topology_lambda"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        for name in ("fedtgp_lambda", "fedsa_anchor_lambda", "fedsa_margin_lambda",
                     "fedsa_calibration_lambda", "cwfedavg_wdr_lambda"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        for name in ("fedrca_personal_loss_weight", "fedrca_class_balance_power",
                     "fedrca_class_balance_smoothing", "fedrca_class_balance_max_ratio",
                     "fedrca_region_balance_power", "fedrca_region_balance_smoothing",
                     "fedrca_region_balance_max_ratio"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if min(self.fedrca_class_balance_max_ratio,
               self.fedrca_region_balance_max_ratio) < 1:
            raise ValueError("FedRCA balance maximum ratios must be >= 1")
        if not math.isfinite(self.participation_rate) or not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must lie in (0, 1]")
        if not 0 <= self.fedsa_anchor_momentum < 1:
            raise ValueError("fedsa_anchor_momentum must lie in [0, 1)")
        if not 0 < self.fedsimsup_gamma < 0.5:
            raise ValueError("fedsimsup_gamma must lie in (0, 0.5)")
        if not 0 <= self.momentum < 1 or not 0 <= self.fedap_self_weight <= 1:
            raise ValueError("Invalid momentum or FedAP self weight")
        if min(self.workers, self.smoke_samples, self.seed, self.split_seed,
               self.fedrca_warmup_rounds) < 0:
            raise ValueError("workers, smoke_samples, seeds and warmup must be nonnegative")
        if self.fedap_mode not in ("reference", "fedbn"):
            raise ValueError("fedap_mode must be reference or fedbn")
        if self.algorithm == "fedap" and self.fedap_warmup_rounds < 1:
            raise ValueError("FedAP requires a training-only federated warmup")
        if self.fedrca_variant not in FEDRCA_VARIANTS:
            raise ValueError("Unsupported fedrca_variant")
        if self.fedrca_topology_bandwidth_min > self.fedrca_topology_bandwidth_max:
            raise ValueError("fedrca topology bandwidth minimum exceeds maximum")
        if self.selection_metric not in ("accuracy", "macro_f1", "balanced_accuracy"):
            raise ValueError("Unsupported validation selection metric")
        if self.lr_schedule not in ("constant", "multistep"):
            raise ValueError("lr_schedule must be constant or multistep")
        if self.lr_decay_gamma >= 1:
            raise ValueError("lr_decay_gamma must lie in (0, 1)")
        self.decay_fractions()

    def decay_fractions(self):
        try:
            values = tuple(sorted({float(item.strip())
                                   for item in self.lr_decay_fractions.split(",")
                                   if item.strip()}))
        except ValueError as error:
            raise ValueError("lr_decay_fractions must be comma-separated numbers") from error
        if not values or any(not math.isfinite(value) or not 0 < value < 1
                             for value in values):
            raise ValueError("lr_decay_fractions must lie in (0, 1)")
        return values

    def learning_rate(self, base_lr, round_index):
        """Return the round-wise LR; round_index is zero based."""
        if round_index < 0:
            raise ValueError("round_index must be nonnegative")
        if self.lr_schedule == "constant":
            return base_lr
        decays = sum(round_index >= int(self.rounds * fraction)
                     for fraction in self.decay_fractions())
        return base_lr * self.lr_decay_gamma ** decays

    def as_dict(self):
        return asdict(self)

    def identity(self):
        values = self.as_dict()
        for key in ("output_dir", "cache_dir", "device", "workers", "cpu_threads", "rounds"):
            values.pop(key)
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def read_config(path):
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    return json.loads(path.read_text(encoding="utf-8-sig"))
