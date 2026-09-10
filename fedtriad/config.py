from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path


DATASETS = {
    "organamnist": (1, 11),
    "bloodmnist": (3, 8),
    "pathmnist": (3, 9),
}

TRIAD_VARIANTS = (
    "p_only",
    "ps_uniform",
    "pl_uniform",
    "psl_uniform",
)

LEGACY_UNUSED_FIELDS = {
    "router_lr", "triad_warmup_rounds", "triad_ramp_rounds",
    "triad_router_hidden", "triad_responsibility_temperature",
    "triad_loss_ema", "triad_loss_std_floor", "triad_quality_ratio",
    "triad_quality_temperature", "triad_fused_lambda", "triad_route_lambda",
    "triad_bridge_lambda", "triad_bridge_temperature",
    "triad_temperature_regularization",
    "triad_sparse_groups", "triad_sparse_active_groups",
    "triad_sparse_coverage_power", "triad_risk_shrinkage",
}


@dataclass
class Config:
    dataset: str = "bloodmnist"
    algorithm: str = "fedtriad"
    triad_variant: str = "psl_uniform"
    data_root: str = "data"
    data_file: str = ""
    cache_dir: str = ".cache/medmnist"
    output_dir: str = "runs/default"

    image_size: int = 28
    allow_resize: bool = False
    clients: int = 5
    participation_rate: float = 0.6
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

    rounds: int = 300
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    lr_schedule: str = "multistep"
    lr_decay_fractions: str = "0.6,0.85"
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

    triad_max_grad_norm: float = 5.0
    triad_risk_temperature: float = 1.0
    triad_risk_entropy_weight: float = 0.1
    triad_risk_weight_floor: float = 0.05
    smoke_samples: int = 0

    @classmethod
    def from_dict(cls, values):
        values = {key: value for key, value in values.items()
                  if key not in LEGACY_UNUSED_FIELDS}
        unknown = set(values) - {item.name for item in fields(cls)}
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
        if self.algorithm != "fedtriad":
            raise ValueError("This clean project supports only FedTriad")
        if self.dataset not in DATASETS:
            raise ValueError("Unsupported dataset")
        if self.triad_variant not in TRIAD_VARIANTS:
            raise ValueError("Unsupported FedTriad variant")
        positive_ints = (
            "clients", "rounds", "local_epochs", "batch_size", "feature_dim", "width",
            "eval_every", "min_train_samples", "min_classes", "partition_attempts",
            "calibration_min_samples", "calibration_min_classes", "cpu_threads",
        )
        for name in positive_ints:
            if getattr(self, name) < 1:
                raise ValueError(name + " must be positive")
        if self.clients < 2:
            raise ValueError("At least two clients are required")
        if self.image_size != 28:
            raise ValueError("The locked protocol uses native 28x28 data")
        if self.min_classes > DATASETS[self.dataset][1]:
            raise ValueError("min_classes exceeds task class count")
        if not 0 < self.local_test_fraction < 0.5:
            raise ValueError("local_test_fraction must lie in (0, 0.5)")
        if not 0 <= self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must lie in [0, 0.5)")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must lie in (0, 1]")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must lie in [0, 1)")
        if min(self.workers, self.smoke_samples, self.seed, self.split_seed) < 0:
            raise ValueError("workers, smoke_samples and seeds must be nonnegative")
        finite_positive = (
            "alpha", "lr", "lr_decay_gamma", "triad_max_grad_norm",
            "triad_risk_temperature",
        )
        for name in finite_positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        finite_nonnegative = (
            "weight_decay", "triad_risk_entropy_weight", "triad_risk_weight_floor",
        )
        for name in finite_nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if self.lr_schedule not in ("constant", "multistep"):
            raise ValueError("lr_schedule must be constant or multistep")
        if self.lr_decay_gamma >= 1:
            raise ValueError("lr_decay_gamma must lie in (0, 1)")
        if self.triad_risk_weight_floor * 3 >= 1:
            raise ValueError("triad_risk_weight_floor must be smaller than 1/3")
        if self.selection_metric not in ("accuracy", "macro_f1", "balanced_accuracy"):
            raise ValueError("Unsupported validation selection metric")
        self.decay_fractions()

    @property
    def num_classes(self):
        return DATASETS[self.dataset][1]

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
