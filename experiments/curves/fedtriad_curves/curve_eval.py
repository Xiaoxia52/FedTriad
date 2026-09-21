"""Additional read-only evaluation of current PSL states, preserving training RNG."""
import json
import time
from pathlib import Path
import numpy as np
from .algorithm import Federation
from .data import loader
from .io import write_json
from .risk_router import fit_global_class_risk, serializable_memory, class_agnostic_risk

ROUNDS = [1] + list(range(15, 301, 15))
BASE_ROUND = Federation.train_round

def evaluate_current(federation):
    from .runner import rng_state, restore_rng
    from .dense_psl_risk import _evaluate_client_indices
    config = federation.config
    output = Path(config.output_dir)
    path = output / 'curve_points' / ('round_%03d.json' % federation.round)
    saved_rng = rng_state()
    modules = [federation.parallel_server, *federation.parallel_clients,
               federation.serial, *federation.locals]
    modes = [m.training for m in modules]
    started = time.perf_counter()
    try:
        probabilities, labels = [], []
        for k, indices in enumerate(federation.partition['train']):
            y, p = federation.predict_client_branches(
                loader(federation.cache, 'train', indices, config), k)
            probabilities.append(p)
            labels.append(y)
        memory = fit_global_class_risk(probabilities, labels, config.num_classes)
        validation = _evaluate_client_indices(
            federation, federation.cache, federation.partition['val'], config,
            'val', memory['global_risk'], class_agnostic_risk(memory))
        write_json(path, {
            'protocol': 'fedtriad-curves-15r-v1', 'round': federation.round,
            'dataset': config.dataset, 'alpha': config.alpha, 'seed': config.seed,
            'partition_id': federation.partition['partition_id'],
            'risk_source': 'current_round_training_data_only',
            'temperature': config.triad_risk_temperature,
            'entropy_weight': config.triad_risk_entropy_weight,
            'weight_floor': config.triad_risk_weight_floor,
            'memory': serializable_memory(memory), 'client_validation': validation,
            'evaluation_seconds': time.perf_counter() - started,
        })
        print('FULL FedTriad round=%s client_val_macro_f1=%.6f' % (
            federation.round, validation['global_class_risk']['aggregate']['macro_f1_mean']), flush=True)
    finally:
        restore_rng(saved_rng)
        for module, mode in zip(modules, modes):
            module.train(mode)

def train_round_with_curve(self):
    # If a process died after saving a point but before its checkpoint, overwrite
    # that point when the resumed training reaches this round again.
    BASE_ROUND(self)
    if self.round in ROUNDS:
        evaluate_current(self)

def install():
    Federation.train_round = train_round_with_curve
