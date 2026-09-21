"""Single-P clip=5 and three independent P streams with joint clip=5."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from fedtriad.algorithm import train_client
from fedtriad.config import Config, DATASETS
from fedtriad.data import loader
from fedtriad.io import write_json
from fedtriad.metrics import aggregate_metrics, classification_metrics
from fedtriad.models import MedicalCNN, average, bytes_of, state
from fedtriad.runner import (
    acquire_run_lock, configure_reproducibility, restore_rng, rng_state,
    save_checkpoint, write_history,
)

PROTOCOL = 'scheme-c-v1'
VARIANTS = ('p_clip5', 'p3_ensemble_clip5')


class ParallelControl:
    def __init__(self, config, partition, cache, variant, device):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.config, self.partition, self.cache = config, partition, Path(cache)
        self.variant, self.device = variant, device
        self.members = 1 if variant == 'p_clip5' else 3
        self.init_seeds = [config.seed + 1000003 * i for i in range(self.members)]
        channels, classes = DATASETS[config.dataset]
        self.servers = []
        for seed in self.init_seeds:
            # Member 0 exactly matches the original P initialization seed.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = MedicalCNN(channels, classes, config.width, config.feature_dim)
            self.servers.append(model.to(device))
        self.copies = [copy.deepcopy(model) for model in self.servers]
        self.round = 0
        self.bytes_sent = self.optimizer_steps = self.branch_optimizer_steps = 0
        self.train_examples = 0
        self.train_seconds = 0.0
        self.history = []
        # The existing helper dispatches clipping through this field. All controls
        # explicitly select its finite-clipping branch; model topology comes only
        # from the actual model arguments below. The persisted variant stays clear.
        self.clip_config = replace(config, triad_variant='psl_uniform', triad_max_grad_norm=5.0)

    def active_clients(self):
        c = self.config
        rng = np.random.RandomState(c.seed * 10000019 + self.round * 100003 + 1709)
        count = max(1, int(round(c.clients * c.participation_rate)))
        return sorted(rng.choice(c.clients, size=count, replace=False).tolist())

    def train_round(self):
        started = time.perf_counter()
        c = self.config
        active = self.active_clients()
        pivot = self.round % c.clients
        order = sorted(active, key=lambda k: (k - pivot) % c.clients)
        anchors = [state(m) for m in self.servers]
        collected = [[] for _ in self.servers]
        sizes, diagnostics = [], []
        for k in order:
            for model, weights in zip(self.copies, anchors):
                model.load_state_dict(weights)
            seed = c.seed * 10000019 + self.round * 100003 + k * 101
            batches = loader(self.cache, 'train', self.partition['train'][k], c,
                             shuffle=True, seed=seed)
            d = train_client(self.copies[0],
                             self.copies[1] if self.members == 3 else None,
                             self.copies[2] if self.members == 3 else None,
                             batches, self.clip_config, self.device, self.round)
            for group, model in zip(collected, self.copies):
                group.append(state(model))
            sizes.append(len(self.partition['train'][k]))
            self.optimizer_steps += d['optimizer_steps']
            self.branch_optimizer_steps += d['branch_optimizer_steps']
            self.train_examples += d['examples'] * self.members
            diagnostics.append({
                'client': k, 'examples': d['examples'],
                'optimizer_steps': d['optimizer_steps'],
                'gradient_clipped_steps': d['gradient_clipped_steps'],
                'member_ce_sums': [d['parallel_ce_sum']] +
                    ([d['serial_ce_sum'], d['local_ce_sum']] if self.members == 3 else []),
            })
        for server, client_states in zip(self.servers, collected):
            server.load_state_dict(average(client_states, sizes))
        self.bytes_sent += 2 * len(active) * sum(bytes_of(m) for m in self.servers)
        self.train_seconds += time.perf_counter() - started
        self.round += 1
        result = {'round': self.round, 'active_clients': active, 'execution_order': order,
                  'joint_clip_norm': 5.0, 'clients': diagnostics,
                  'gradient_clipped_steps': sum(d['gradient_clipped_steps'] for d in diagnostics)}
        self.history.append(result)
        return result

    @torch.no_grad()
    def predict(self, batches):
        for m in self.servers:
            m.to(self.device).eval()
        ys, ps = [], []
        for images, labels in batches:
            images = images.to(self.device, non_blocking=True)
            probs = [m(images)[1].softmax(1) for m in self.servers]
            ps.append((sum(probs) / self.members).cpu().numpy())
            ys.append(labels.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    def pack(self):
        return {'variant': self.variant, 'members': self.members,
                'init_seeds': self.init_seeds, 'servers': [state(m) for m in self.servers],
                'round': self.round, 'bytes_sent': self.bytes_sent,
                'optimizer_steps': self.optimizer_steps,
                'branch_optimizer_steps': self.branch_optimizer_steps,
                'train_examples': self.train_examples, 'train_seconds': self.train_seconds,
                'history': copy.deepcopy(self.history)}

    def restore(self, packed):
        if packed['variant'] != self.variant or packed['init_seeds'] != self.init_seeds:
            raise ValueError('Control identity mismatch')
        if len(packed['servers']) != self.members:
            raise ValueError('Ensemble member count mismatch')
        for m, weights in zip(self.servers, packed['servers']):
            m.load_state_dict(weights)
        for key in ('round', 'bytes_sent', 'optimizer_steps', 'branch_optimizer_steps',
                    'train_examples', 'train_seconds', 'history'):
            setattr(self, key, copy.deepcopy(packed[key]))


def evaluate_local(model, config, partition, cache, which):
    source = 'val' if which == 'val' else 'train'
    values = []
    for indices in partition[which]:
        y, p = model.predict(loader(cache, source, indices, config))
        values.append(classification_metrics(y, p, config.num_classes))
    return {'aggregate': aggregate_metrics(values), 'per_client': values}


def implementation_id(project):
    # Resume protection for the implementation actually used by this experiment.
    digest = hashlib.sha256()
    paths = [*sorted(Path(__file__).resolve().parent.glob('*.py')),
             *sorted((Path(project) / 'fedtriad').glob('*.py'))]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def execute(job, output, project, device='cuda:0', stop_after=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    acquire_run_lock(lock)
    try:
        return _execute_locked(job, output, project, device, stop_after)
    finally:
        lock.unlink(missing_ok=True)


def _execute_locked(job, output, project, device, stop_after):
    identity = {k: job[k] for k in ('key', 'variant', 'config', 'partition_id', 'init_seeds')}
    identity['protocol'] = PROTOCOL
    code_id = implementation_id(project)
    result_path = output / 'results.json'
    identity_path = output / 'experiment.json'
    if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Output belongs to a different experiment: ' + str(output))
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding='utf-8'))
        if old.get('status') == 'complete':
            if old.get('identity') != identity or old.get('implementation_id') != code_id:
                raise ValueError('Completed result identity changed')
            return old
    cfg = Config.from_dict({**job['config'], 'device': device, 'workers': 0, 'cpu_threads': 2})
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    configure_reproducibility(dev)
    partition = json.loads(Path(job['partition_file']).read_text(encoding='utf-8'))
    if partition['partition_id'] != job['partition_id']:
        raise ValueError('Original partition changed')
    cache = Path(job['cache'])
    model = ParallelControl(cfg, partition, cache, job['variant'], dev)
    scores = []
    best_score, best_round, best_state, best_validation = -float('inf'), 0, None, None
    ckfile = output / 'last.pt'
    if ckfile.exists():
        ck = torch.load(ckfile, map_location='cpu', weights_only=False)
        if ck['identity'] != identity or ck['implementation_id'] != code_id:
            raise ValueError('Resume protocol/code identity changed')
        if ck['environment'] != {'torch': torch.__version__, 'numpy': np.__version__, 'device': device}:
            raise ValueError('Resume runtime changed; use the original Python/device')
        model.restore(ck['model'])
        scores = ck['scores']
        best_score, best_round = ck['best_score'], ck['best_round']
        best_state, best_validation = ck['best_state'], ck['best_validation']
        restore_rng(ck['rng'])
        del ck
    write_json(identity_path, identity)
    write_json(output / 'config.json', cfg.as_dict())

    def save():
        save_checkpoint(ckfile, {'identity': identity, 'implementation_id': code_id,
            'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'device': device},
            'model': model.pack(), 'scores': scores, 'best_score': best_score,
            'best_round': best_round, 'best_state': best_state,
            'best_validation': best_validation, 'rng': rng_state()})
        write_history(output / 'history.csv', scores)

    if not ckfile.exists():
        save()  # round-zero checkpoint allows exact restart after a first-round OOM.
    limit = cfg.rounds if stop_after is None else min(cfg.rounds, stop_after)
    while model.round < limit:
        d = model.train_round()
        validation = evaluate_local(model, cfg, partition, cache, 'val')
        score = validation['aggregate']['macro_f1_mean']
        if score > best_score:
            best_score, best_round = score, model.round
            best_state, best_validation = model.pack(), validation
        scores.append({'round': model.round,
            'learning_rate': cfg.learning_rate(cfg.lr, model.round - 1),
            'client_validation_macro_f1': score, 'best_round': best_round,
            'gradient_clipped_steps': d['gradient_clipped_steps'],
            'train_seconds': model.train_seconds,
            'branch_optimizer_steps': model.branch_optimizer_steps,
            'communication_bytes': model.bytes_sent})
        save()
        write_json(output / 'progress.json', {'status': 'running', 'round': model.round,
            'total_rounds': cfg.rounds, 'best_round': best_round, 'best_validation': best_score})
        print(f"{job['key']} round={model.round}/{cfg.rounds} val_F1={score:.6f} best={best_round}", flush=True)

    actual_rounds = model.round
    cost = {'communication_bytes': model.bytes_sent, 'train_seconds': model.train_seconds,
            'optimizer_steps': model.optimizer_steps,
            'branch_optimizer_steps': model.branch_optimizer_steps,
            'train_examples_across_branches': model.train_examples,
            'persistent_model_states': model.members,
            'inference_predictors': model.members,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else 0,
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(dev) if dev.type=='cuda' else 0}
    if best_state is None:
        raise RuntimeError('No validation-selected model')
    model.restore(best_state)
    save_checkpoint(output / 'best_personal.pt', {'model': best_state,
        'selected_round': best_round, 'validation_score': best_score, 'identity': identity})
    local_test = evaluate_local(model, cfg, partition, cache, 'local_test')
    indices = np.arange(len(np.load(cache/'test_labels.npy', allow_pickle=False)))
    y, p = model.predict(loader(cache, 'test', indices, cfg))
    official = classification_metrics(y, p, cfg.num_classes)
    completed = actual_rounds == cfg.rounds
    result = {'status': 'complete' if completed else 'validation_only',
        'identity': identity, 'implementation_id': code_id,
        'dataset': cfg.dataset, 'alpha': cfg.alpha, 'seed': cfg.seed, 'variant': job['variant'],
        'trained_rounds': actual_rounds, 'selected_round': best_round,
        'selected_validation': best_validation, 'local_test': local_test,
        'official_test': official, 'cost': cost,
        'prediction_policy': 'one P' if model.members==1 else 'uniform mean of three P softmax probabilities',
        'checkpoint_selection': 'mean client validation Macro-F1, strict improvement, first tie retained',
        'reference_run': job['reference_run'], 'partition_file': job['partition_file'],
        'numpy': np.__version__, 'torch': torch.__version__, 'device': device}
    write_json(result_path, result)
    write_json(output / 'progress.json', {'status': result['status'], 'round': actual_rounds,
        'total_rounds': cfg.rounds, 'best_round': best_round})
    return result
