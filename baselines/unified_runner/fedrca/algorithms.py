"""Federated baselines and FedRCA.

FedRCA aligns the spatial RBF topology of a frozen pre-round global encoder and
its homogeneous client encoder.  A one-time client-private pixel partition
balances that distillation across class/appearance subgroups.
"""
import copy
import math
import time

import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn
from torch.nn import functional as F

from .config import DATASETS
from .data import loader
from .models import MedicalCNN, ParallelEnsemble, average, bn_keys, bytes_of, load_subset, state
from .pixel_regions import build_pixel_region_index
from .recent_methods import (
    TrainableGlobalPrototypes,
    aggregate_classwise_models,
    class_distribution_weights,
    collect_class_prototypes,
    cosine_similarity_matrix,
    initialize_semantic_anchors,
    mix_classwise_models,
    train_cwfedavg,
    train_fedsa,
    train_fedsimsup,
    train_fedsol,
    update_fedsimsup_inactive,
    update_fedtgp,
    update_semantic_anchors,
)


def _trainable(name, phase):
    return phase == "all" or (phase == "head" and name.startswith("head.")) or (
        phase == "encoder" and not name.startswith("head.")
    )


def personal_class_weights(counts, config):
    """Fixed inverse-frequency weights over classes observed by one client."""
    counts = np.asarray(counts, dtype=float)
    observed = counts > 0
    if not observed.any():
        raise ValueError("Cannot weight an empty client")
    weights = np.zeros_like(counts)
    raw = ((counts[observed].max() + config.fedrca_class_balance_smoothing) /
           (counts[observed] + config.fedrca_class_balance_smoothing)) ** \
          config.fedrca_class_balance_power
    raw = np.minimum(raw, config.fedrca_class_balance_max_ratio)
    weights[observed] = raw
    weights /= np.dot(weights, counts) / counts.sum()
    return torch.tensor(weights, dtype=torch.float32)


def rbf_topology_probabilities(feature_map, log_bandwidth, grid_size,
                               bandwidth_min=0.05, bandwidth_max=20.0):
    """Convert within-feature-map spatial distances to row-wise RBF topology."""
    if feature_map.ndim != 4:
        raise ValueError("Topology features must have shape [B,C,H,W]")
    values = F.adaptive_avg_pool2d(feature_map, (grid_size, grid_size))
    nodes = F.normalize(values.flatten(2), dim=1).transpose(1, 2)
    distances = torch.cdist(nodes, nodes).square()
    bandwidth = log_bandwidth.exp().clamp(bandwidth_min, bandwidth_max)
    return F.softmax(-distances / bandwidth, dim=-1)


def rbf_topology_loss(student_maps, teacher_maps, log_bandwidth, config,
                      sample_weights=None):
    """Teacher-to-student KL over spatial RBF graphs; returns one scalar."""
    if len(student_maps) != len(teacher_maps) or len(student_maps) != len(log_bandwidth):
        raise ValueError("Topology stage count mismatch")
    per_stage = []
    for stage, (student_map, teacher_map) in enumerate(zip(student_maps, teacher_maps)):
        teacher_prob = rbf_topology_probabilities(
            teacher_map, log_bandwidth[stage], config.fedrca_topology_grid,
            config.fedrca_topology_bandwidth_min,
            config.fedrca_topology_bandwidth_max,
        ).detach()
        student_prob = rbf_topology_probabilities(
            student_map, log_bandwidth[stage], config.fedrca_topology_grid,
            config.fedrca_topology_bandwidth_min,
            config.fedrca_topology_bandwidth_max,
        )
        # Match Lite-MyoNet's KL(batchmean): sum over all graph rows and
        # neighbours, retain one loss per sample before region reweighting.
        per_sample = (teacher_prob * (
            teacher_prob.clamp_min(1e-12).log() -
            student_prob.clamp_min(1e-12).log()
        )).sum(dim=(-1, -2))
        per_stage.append(per_sample)
    losses = torch.stack(per_stage).mean(dim=0)
    if sample_weights is None:
        return losses.mean()
    sample_weights = sample_weights.to(losses.device, losses.dtype)
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)


def train_local(model, batches, config, device, epochs, phase="all", anchor=None,
                penalty=0.0, prototypes=None, lr=None, teacher=None,
                topology_scale=0.0, topology_sample_weights=None, freeze_bn=False,
                max_grad_norm=None, personal_head=None, class_weights=None,
                prototype_scale=None, keep_teacher_on_device=False):
    """Train one client and optionally apply FedRCA topology distillation."""
    model.to(device).train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_trainable(name, phase))
    if personal_head is not None:
        if phase != "all":
            raise ValueError("FedRCA personal head is supported only for full local training")
        personal_head.to(device).train()
    if phase == "head":
        model.encoder.eval()
    elif freeze_bn:
        for module in model.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    if teacher is not None:
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    optimizer_parameters = [p for p in model.parameters() if p.requires_grad]
    if personal_head is not None:
        optimizer_parameters.extend(personal_head.parameters())
    optimizer = torch.optim.SGD(
        optimizer_parameters,
        lr=config.lr if lr is None else lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    targets = {k: v.to(device) for k, v in (anchor or {}).items()}
    proto = {int(k): v.to(device) for k, v in (prototypes or {}).items()}
    balance = class_weights.to(device) if class_weights is not None else None
    loss_sum = global_ce_sum = personal_ce_sum = topology_sum = 0.0
    seen = steps = clipped_steps = 0

    for _ in range(epochs):
        for batch in batches:
            images, labels = batch[:2]
            sample_indices = batch[2] if len(batch) == 3 else None
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None and topology_scale:
                features, logits, student_maps = model.forward_with_maps(images)
            else:
                features, logits = model(images)
            global_ce = F.cross_entropy(logits, labels)
            personal_ce = features.new_zeros(())
            loss = global_ce
            if personal_head is not None:
                # The private classifier adapts to local class frequencies, but it
                # must not weaken or redirect the shared encoder/global head.  Its
                # detached feature input makes the two optimization paths explicit.
                personal_ce = F.cross_entropy(personal_head(features.detach()), labels, weight=balance)
                personal_weight = config.fedrca_personal_loss_weight
                loss = global_ce + personal_weight * personal_ce
            if penalty and targets:
                loss = loss + 0.5 * penalty * sum(
                    (p - targets[name]).square().sum()
                    for name, p in model.named_parameters() if p.requires_grad
                )
            if proto:
                reference = features.detach().clone()
                for label, center in proto.items():
                    reference[labels == label] = center
                scale = config.prototype_lambda if prototype_scale is None else prototype_scale
                loss = loss + scale * F.mse_loss(features, reference)

            topology = features.new_zeros(())
            if teacher is not None and topology_scale:
                with torch.no_grad():
                    teacher_maps = teacher.topology_maps(images)
                batch_weights = None
                if topology_sample_weights is not None:
                    if sample_indices is None:
                        raise ValueError("Topology sample weights require original sample indices")
                    batch_weights = topology_sample_weights[sample_indices.long()]
                topology = rbf_topology_loss(
                    student_maps, teacher_maps, model.topology_log_bandwidth,
                    config, batch_weights,
                )
                loss = loss + topology_scale * topology

            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite local objective at optimizer step %s" % (steps + 1))
            loss.backward()
            if max_grad_norm is not None:
                norm = torch.nn.utils.clip_grad_norm_(
                    optimizer_parameters,
                    max_grad_norm,
                    error_if_nonfinite=True,
                )
                clipped_steps += int(float(norm) > max_grad_norm)
            optimizer.step()
            batch_n = len(labels)
            loss_sum += float(loss.detach()) * batch_n
            global_ce_sum += float(global_ce.detach()) * batch_n
            personal_ce_sum += float(personal_ce.detach()) * batch_n
            topology_sum += float(topology.detach()) * batch_n
            seen += batch_n
            steps += 1

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.cpu()
    if personal_head is not None:
        personal_head.cpu()
    if teacher is not None and not keep_teacher_on_device:
        teacher.cpu()
    return {
        "loss_sum": loss_sum,
        "global_ce_sum": global_ce_sum,
        "personal_ce_sum": personal_ce_sum,
        "topology_sum": topology_sum,
        "examples": seen,
        "optimizer_steps": steps,
        "gradient_clipped_steps": clipped_steps,
    }


@torch.no_grad()
def feature_statistics(model, batches, device, classes, feature_dim):
    model.to(device).eval()
    counts = torch.zeros(classes, dtype=torch.float64)
    sums = torch.zeros(classes, feature_dim, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    for images, labels in batches:
        features = model(images.to(device))[0].cpu().double()
        for label in labels.unique():
            rows = features[labels == label]
            counts[label] += len(rows)
            sums[label] += rows.sum(dim=0)
            squares[label] += rows.square().sum(dim=0)
    model.cpu()
    means = sums / counts.clamp_min(1)[:, None]
    total = counts.sum()
    prior = counts / total
    h = prior[:, None] * means
    variance = ((squares.sum() / total) - (prior.square()[:, None] * means.square()).sum()) / total
    prototypes = {i: means[i].float() for i in range(classes) if counts[i] > 0}
    return {"counts": counts, "prototypes": prototypes, "h": h,
            "variance": max(float(variance), 0.0), "examples": int(total)}


def aggregate_prototypes(statistics, weighted):
    result = {}
    labels = sorted({key for item in statistics for key in item["prototypes"]})
    for label in labels:
        pairs = [(item["prototypes"][label], float(item["counts"][label]) if weighted else 1.0)
                 for item in statistics if label in item["prototypes"]]
        result[label] = sum(center * count for center, count in pairs) / sum(count for _, count in pairs)
    return result


def fedpac_weights(statistics):
    hs = np.stack([item["h"].numpy().reshape(-1) for item in statistics])
    variances = np.asarray([item["variance"] for item in statistics])
    clients = len(hs)
    rows, diagnostics = [], []
    for target in range(clients):
        delta = hs[target] - hs
        q = delta @ delta.T + np.diag(variances)
        q = (q + q.T) / 2
        scale = max(float(np.abs(q).max()), 1e-12)
        q /= scale
        answer = minimize(lambda a: float(a @ q @ a), np.full(clients, 1 / clients),
                          jac=lambda a: 2 * q @ a, method="SLSQP",
                          bounds=[(0.0, 1.0)] * clients,
                          constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1,
                                        "jac": lambda a: np.ones(clients)}],
                          options={"ftol": 1e-12, "maxiter": 1000})
        if not answer.success or not np.isfinite(answer.x).all():
            raise RuntimeError("FedPAC QP failed: " + answer.message)
        row = np.maximum(answer.x, 0)
        row /= row.sum()
        gradient = 2 * q @ row
        gap = float(row @ gradient - gradient.min())
        if gap > 1e-5:
            raise RuntimeError("FedPAC QP optimality gap exceeds tolerance: %s" % gap)
        rows.append(row)
        diagnostics.append({"objective": float(row @ q @ row) * scale,
                            "optimality_gap_scaled": gap})
    return np.stack(rows), diagnostics


@torch.no_grad()
def bn_statistics(model, batches, device):
    moments, handles = {}, []

    def hook(name):
        def collect(module, inputs):
            features = inputs[0].detach().double()
            axes = [0] + list(range(2, features.ndim))
            count = features.numel() // features.shape[1]
            mean = features.mean(dim=axes).cpu()
            m2 = features.var(dim=axes, unbiased=False).cpu() * count
            if name not in moments:
                moments[name] = [count, mean, m2]
            else:
                old_n, old_mean, old_m2 = moments[name]
                total = old_n + count
                delta = mean - old_mean
                moments[name] = [total, old_mean + delta * count / total,
                                 old_m2 + m2 + delta.square() * old_n * count / total]
        return collect

    model.to(device).eval()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            handles.append(module.register_forward_pre_hook(hook(name)))
    try:
        for images, _ in batches:
            model(images.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    if not moments:
        raise ValueError("FedAP requires BN layers")
    return [(mean, (m2 / count).clamp_min(0)) for count, mean, m2 in moments.values()]


def fedap_weights(statistics, self_weight):
    clients = len(statistics)
    if clients < 2:
        raise ValueError("FedAP needs at least two clients")
    result = np.zeros((clients, clients))
    for i in range(clients):
        for j in range(clients):
            if i != j:
                distance = sum(float(((a - c).square().sum() +
                                      (b.sqrt() - d.sqrt()).square().sum()).sqrt())
                               for (a, b), (c, d) in zip(statistics[i], statistics[j]))
                result[i, j] = 1 / max(distance, 1e-12)
        result[i] *= (1 - self_weight) / result[i].sum()
        result[i, i] = self_weight
    return result


def fedseq_greedy_groups(class_counts, group_count):
    """Build fixed, balanced, class-complementary FedSeq superclients.

    FedSeq's public implementation estimates client label distributions from a
    public exemplar set and greedily builds heterogeneous superclients.  The
    unified medical protocol has no public exemplar set, so this adaptation
    uses exact *training-only* label histograms.  That gives FedSeq stronger
    grouping information without touching validation or test labels.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 2 or len(counts) < 2:
        raise ValueError("FedSeq requires a client-by-class count matrix")
    if not 1 <= group_count <= len(counts):
        raise ValueError("Invalid FedSeq superclient count")
    groups = [[] for _ in range(group_count)]
    group_counts = np.zeros((group_count, counts.shape[1]), dtype=np.float64)
    group_sizes = np.zeros(group_count, dtype=np.float64)
    capacity = int(math.ceil(len(counts) / group_count))
    remaining = set(range(len(counts)))
    uniform = np.full(counts.shape[1], 1.0 / counts.shape[1])

    # Seed the groups with mutually distinct clients, then assign every
    # remaining client to the group whose combined distribution is closest to
    # uniform.  This is the finite-five-client counterpart of FedSeq's greedy
    # heterogeneous grouping rule.
    for group_id in range(group_count):
        candidates = []
        for client_id in remaining:
            distribution = counts[client_id] / max(counts[client_id].sum(), 1.0)
            if group_id == 0:
                score = float(np.square(distribution - uniform).sum())
            else:
                chosen = group_counts[:group_id].sum(axis=0)
                chosen /= max(chosen.sum(), 1.0)
                score = -float(np.dot(distribution, chosen) /
                               max(np.linalg.norm(distribution) * np.linalg.norm(chosen), 1e-12))
            candidates.append((score, -client_id, client_id))
        client_id = max(candidates)[2]
        groups[group_id].append(client_id)
        group_counts[group_id] += counts[client_id]
        group_sizes[group_id] += counts[client_id].sum()
        remaining.remove(client_id)

    while remaining:
        best = None
        for client_id in sorted(remaining):
            for group_id in range(group_count):
                if len(groups[group_id]) >= capacity:
                    continue
                combined = group_counts[group_id] + counts[client_id]
                distribution = combined / max(combined.sum(), 1.0)
                positive = distribution > 0
                divergence = float(np.sum(
                    distribution[positive] * np.log(distribution[positive] / uniform[positive])
                ))
                candidate = (-divergence, -group_sizes[group_id], -group_id,
                             -client_id, group_id, client_id)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            raise RuntimeError("FedSeq greedy grouping has no feasible assignment")
        group_id, client_id = best[-2:]
        groups[group_id].append(client_id)
        group_counts[group_id] += counts[client_id]
        group_sizes[group_id] += counts[client_id].sum()
        remaining.remove(client_id)
    return [sorted(group) for group in groups]


class Federation:
    def __init__(self, config, cache, partition, device):
        self.config, self.cache, self.partition, self.device = config, cache, partition, device
        channels, self.classes = DATASETS[config.dataset]
        self.server = MedicalCNN(channels, self.classes, config.width, config.feature_dim)
        with torch.no_grad():
            self.server.topology_log_bandwidth.fill_(
                math.log(config.fedrca_topology_bandwidth_init)
            )
        self.clients = [copy.deepcopy(self.server) for _ in range(config.clients)]
        self.personal = [copy.deepcopy(self.server) for _ in self.clients] if config.algorithm == "ditto" else []
        self.personal_heads = ([copy.deepcopy(self.server.head) for _ in self.clients]
                               if config.algorithm == "fedrca" else [])
        self.supervisors = ([MedicalCNN(
            channels, self.classes, config.fedsimsup_supervisor_width,
            config.fedsimsup_supervisor_feature_dim,
        ) for _ in self.clients] if config.algorithm == "fedsimsup" else [])
        self.tgp = (TrainableGlobalPrototypes(self.classes, config.feature_dim)
                    if config.algorithm == "fedtgp" else None)
        self.tgp_prototypes = {}
        self.semantic_anchors = (initialize_semantic_anchors(
            self.classes, config.feature_dim, device
        ) if config.algorithm == "fedsa" else None)
        self.local_prototypes = [{} for _ in self.clients]
        self.class_models = ([copy.deepcopy(self.server) for _ in range(self.classes)]
                             if config.algorithm == "cwfedavg" else [])
        self.prototypes = {}
        self.pixel_region_assignments = None
        self.pixel_region_weights = None
        self.pixel_region_summaries = []
        self.pixel_kmeans_fits = 0
        self.ap_weights = None
        self.warmup_done = self.round = self.bytes_sent = self.steps = self.examples = 0
        self.statistic_examples = self.preprocessing_examples = self.train_seconds = 0
        self.bn = bn_keys(self.server)
        self.head = {key for key in self.server.state_dict() if key.startswith("head.")}
        self.encoder = set(self.server.state_dict()) - self.head
        self.shared_nonbn = set(self.server.state_dict()) - self.bn
        self.sizes = [len(indices) for indices in partition["train"]]
        self.class_distributions = np.asarray(partition["train_class_counts"], dtype=float)
        self.class_distributions /= np.maximum(
            self.class_distributions.sum(axis=1, keepdims=True), 1.0
        )
        self.client_similarities = cosine_similarity_matrix(partition["train_class_counts"])
        self.fedseq_groups = (fedseq_greedy_groups(
            partition["train_class_counts"], config.fedseq_superclients
        ) if config.algorithm == "fedseq" else [])
        self.class_weights = ([personal_class_weights(counts, config)
                               for counts in partition["train_class_counts"]]
                              if config.algorithm == "fedrca" else [])
        self.last_diagnostics = {}
        self.method_history = []
        self.active_history = []
        if config.algorithm == "fedseq":
            # Count the one-time training-only class histograms as communicated
            # statistics.  No forward pass or validation/test example is used.
            self.bytes_sent += int(np.asarray(partition["train_class_counts"], dtype=np.int64).nbytes)
            self.preprocessing_examples += sum(self.sizes)

    def batches(self, client, round_number, phase=0, shuffle=True, return_indices=False):
        seed = self.config.seed * 10000019 + round_number * 100003 + client * 101 + phase
        return loader(self.cache, "train", self.partition["train"][client],
                      self.config, shuffle, seed, return_indices=return_indices)

    def _train(self, model, client, round_number, batch_phase=0, **kwargs):
        base_lr = kwargs.pop("lr", self.config.lr)
        kwargs["lr"] = self.config.learning_rate(base_lr, self.round)
        return_indices = bool(kwargs.pop("return_indices", False))
        statistics = train_local(model, self.batches(
            client, round_number, batch_phase, return_indices=return_indices
        ),
                                 self.config, self.device, **kwargs)
        self.steps += statistics["optimizer_steps"]
        self.examples += statistics["examples"]
        return statistics

    def active_clients(self):
        count = max(1, int(round(len(self.clients) * self.config.participation_rate)))
        rng = np.random.RandomState(
            self.config.seed * 10000019 + self.round * 100003 + 1709
        )
        return sorted(rng.choice(len(self.clients), size=count, replace=False).tolist())

    def sequential_active_clients(self):
        """Same sampled clients as active_clients, retaining seeded draw order."""
        count = max(1, int(round(len(self.clients) * self.config.participation_rate)))
        rng = np.random.RandomState(
            self.config.seed * 10000019 + self.round * 100003 + 1709
        )
        return rng.choice(len(self.clients), size=count, replace=False).tolist()

    def _aggregate(self, keys, client_ids=None, broadcast=True):
        client_ids = list(range(len(self.clients))) if client_ids is None else list(client_ids)
        combined = average([state(self.clients[index]) for index in client_ids],
                           [self.sizes[index] for index in client_ids], keys)
        load_subset(self.server, combined, keys)
        if broadcast:
            for client in self.clients:
                load_subset(client, combined, keys)
        self.bytes_sent += 2 * len(client_ids) * bytes_of(combined)

    def warmup_step(self):
        start = time.perf_counter()
        round_number = -self.config.fedap_warmup_rounds + self.warmup_done
        for client_id, client in enumerate(self.clients):
            self._train(client, client_id, round_number + 100000,
                        epochs=self.config.local_epochs)
        keys = self.shared_nonbn if self.config.fedap_mode == "fedbn" else set(self.server.state_dict())
        self._aggregate(keys)
        self.warmup_done += 1
        self.train_seconds += time.perf_counter() - start

    def initialize_ap(self):
        start = time.perf_counter()
        statistics = []
        if self.config.fedap_mode == "reference":
            for client_id in range(len(self.clients)):
                statistics.append(bn_statistics(
                    self.server, self.batches(client_id, 0, shuffle=False), self.device
                ))
                self.statistic_examples += self.sizes[client_id]
        else:
            for client in self.clients:
                statistics.append([
                    (module.running_mean.double().clone(), module.running_var.double().clone().clamp_min(0))
                    for module in client.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
                ])
        self.ap_weights = fedap_weights(statistics, self.config.fedap_self_weight)
        self.bytes_sent += sum(sum(a.numel() + b.numel() for a, b in item) * 8 for item in statistics)
        self.bytes_sent += self.ap_weights.nbytes
        self.train_seconds += time.perf_counter() - start

    def train_round(self):
        start = time.perf_counter()
        self.last_diagnostics = {}
        active = (self.sequential_active_clients()
                  if self.config.algorithm in ("cwt", "fedseq")
                  else self.active_clients())
        self.active_history.append(active)
        if self.config.algorithm == "fedrca":
            self._fedrca_round(active)
        elif self.config.algorithm == "fedtgp":
            self._fedtgp_round(active)
        elif self.config.algorithm == "fedsol":
            self._fedsol_round(active)
        elif self.config.algorithm == "fedsa":
            self._fedsa_round(active)
        elif self.config.algorithm == "cwfedavg":
            self._cwfedavg_round(active)
        elif self.config.algorithm == "fedsimsup":
            self._fedsimsup_round(active)
        elif self.config.algorithm == "cwt":
            self._cwt_round(active)
        elif self.config.algorithm == "fedseq":
            self._fedseq_round(active)
        else:
            self._baseline_round(active)
        self.round += 1
        self.train_seconds += time.perf_counter() - start

    def _cwt_round(self, active):
        """Partial-participation Cyclic Weight Transfer / vanilla SFL."""
        incoming = state(self.server)
        payload = bytes_of(incoming)
        for position, client_id in enumerate(active):
            client = self.clients[client_id]
            load_subset(client, incoming, set(incoming))
            self._train(client, client_id, self.round, batch_phase=1100 + position,
                        epochs=self.config.local_epochs)
            incoming = state(client)
        load_subset(self.server, incoming, set(incoming))
        # One transfer for each sequential edge, including the handoff from the
        # previous round's holder to the first active client.
        self.bytes_sent += len(active) * payload
        summary = {"round": self.round + 1, "active_clients_in_order": active,
                   "sequential_client_updates": len(active)}
        self.method_history.append(summary)
        self.last_diagnostics = {"method": "CWT", "summary": summary}

    def _fedseq_round(self, active):
        """FedSeq with fixed heterogeneous superclients and partial participation."""
        anchor = state(self.server)
        payload = bytes_of(anchor)
        active_positions = {client_id: position for position, client_id in enumerate(active)}
        chains = []
        endpoint_states, endpoint_weights = [], []
        for group in self.fedseq_groups:
            chain = sorted((client_id for client_id in group if client_id in active_positions),
                           key=active_positions.get)
            if not chain:
                continue
            incoming = anchor
            for position, client_id in enumerate(chain):
                client = self.clients[client_id]
                load_subset(client, incoming, set(incoming))
                self._train(client, client_id, self.round,
                            batch_phase=1200 + len(chains) * 100 + position,
                            epochs=self.config.local_epochs)
                incoming = state(client)
            chains.append(chain)
            endpoint_states.append(incoming)
            endpoint_weights.append(sum(self.sizes[client_id] for client_id in chain))
            # Server-to-first, within-chain handoffs, and last-to-server.
            self.bytes_sent += (len(chain) + 1) * payload
        combined = average(endpoint_states, endpoint_weights, set(anchor))
        load_subset(self.server, combined, set(combined))
        summary = {"round": self.round + 1, "active_clients_in_order": active,
                   "superclient_groups": copy.deepcopy(self.fedseq_groups),
                   "active_chains": chains, "endpoint_weights": endpoint_weights}
        self.method_history.append(summary)
        self.last_diagnostics = {"method": "FedSeq", "summary": summary,
                                 "grouping": "training-label-histogram oracle adaptation"}

    def _baseline_round(self, active):
        config, algorithm = self.config, self.config.algorithm
        anchor = state(self.server)
        statistics, qp_statistics = [], []
        if algorithm in ("fedavg", "fedprox", "ditto", "fedbn"):
            for client_id in active:
                load_subset(self.clients[client_id], anchor, set(anchor))
        elif algorithm in ("fedrep", "fedpac"):
            for client_id in active:
                load_subset(self.clients[client_id], anchor, self.encoder)
        for client_id in active:
            client = self.clients[client_id]
            if algorithm == "fedpac":
                qp_statistics.append(feature_statistics(
                    client, self.batches(client_id, self.round, shuffle=False), self.device,
                    self.classes, config.feature_dim
                ))
                self.statistic_examples += self.sizes[client_id]
            if algorithm in ("fedrep", "fedpac"):
                self._train(client, client_id, self.round, epochs=config.head_epochs,
                            phase="head", lr=config.head_lr)
                self._train(client, client_id, self.round, epochs=config.local_epochs,
                            phase="encoder", prototypes=self.prototypes if algorithm == "fedpac" else None)
            else:
                self._train(client, client_id, self.round, epochs=config.local_epochs,
                            anchor=anchor if algorithm == "fedprox" else None,
                            penalty=config.prox_mu if algorithm == "fedprox" else 0,
                            prototypes=self.prototypes if algorithm == "fedproto" else None)
            if algorithm == "ditto":
                self._train(self.personal[client_id], client_id, self.round, batch_phase=7,
                            epochs=config.local_epochs, anchor=anchor, penalty=config.ditto_lambda)
            if algorithm in ("fedproto", "fedpac"):
                statistics.append(feature_statistics(
                    client, self.batches(client_id, self.round, shuffle=False), self.device,
                    self.classes, config.feature_dim
                ))
                self.statistic_examples += self.sizes[client_id]
        if algorithm in ("fedavg", "fedprox", "ditto"):
            self._aggregate(set(anchor), active)
        elif algorithm == "fedbn":
            self._aggregate(self.shared_nonbn, active)
        elif algorithm in ("fedrep", "fedpac"):
            self._aggregate(self.encoder, active)
        elif algorithm == "fedap":
            if self.ap_weights is None:
                raise RuntimeError("FedAP warmup/statistics not initialized")
            snapshots = [state(model) for model in self.clients]
            for client_id, client in enumerate(self.clients):
                load_subset(client, average(snapshots, self.ap_weights[client_id].tolist(),
                                            self.shared_nonbn), self.shared_nonbn)
            self.bytes_sent += 2 * len(self.clients) * bytes_of(snapshots[0], self.shared_nonbn)
            self.last_diagnostics["aggregation_weights"] = self.ap_weights.tolist()
        if algorithm == "fedpac":
            weights, diagnostics = fedpac_weights(qp_statistics)
            snapshots = [state(self.clients[index]) for index in active]
            for local_target, client_id in enumerate(active):
                client = self.clients[client_id]
                load_subset(client, average(snapshots, weights[local_target].tolist(), self.head), self.head)
            self.bytes_sent += 2 * len(active) * bytes_of(snapshots[0], self.head)
            self.bytes_sent += sum(item["h"].numel() * 8 + 8 for item in qp_statistics)
            self.last_diagnostics = {"classifier_weights": weights.tolist(), "qp": diagnostics}
        if algorithm in ("fedproto", "fedpac"):
            self.prototypes = aggregate_prototypes(statistics, weighted=algorithm == "fedpac")
            self.bytes_sent += sum(len(item["prototypes"]) * (16 + config.feature_dim * 4)
                                   for item in statistics)
            self.bytes_sent += len(active) * len(self.prototypes) * (8 + config.feature_dim * 4)

        self.last_diagnostics.setdefault("active_clients", active)

    def _fedtgp_round(self, active):
        config = self.config
        uploaded = []
        for client_id in active:
            client = self.clients[client_id]
            statistics = self._train(
                client, client_id, self.round, epochs=config.local_epochs,
                prototypes=self.tgp_prototypes,
                prototype_scale=config.fedtgp_lambda,
            )
            prototypes, _ = collect_class_prototypes(
                client, self.batches(client_id, self.round, 901, shuffle=False),
                self.device, self.classes,
            )
            self.statistic_examples += self.sizes[client_id]
            uploaded.extend((prototype, label) for label, prototype in prototypes.items())
        lr = config.learning_rate(config.fedtgp_server_lr, self.round)
        self.tgp_prototypes, diagnostics = update_fedtgp(
            self.tgp, uploaded, config, self.classes, self.device, lr
        )
        model_bytes = bytes_of(state(self.clients[active[0]]))
        prototype_bytes = len(uploaded) * (8 + config.feature_dim * 4)
        download_bytes = len(active) * len(self.tgp_prototypes) * (8 + config.feature_dim * 4)
        # FedTGP communicates prototypes, not client network parameters.
        self.bytes_sent += prototype_bytes + download_bytes
        summary = {"round": self.round + 1, "active_clients": active,
                   "uploaded_prototypes": len(uploaded), **diagnostics}
        self.method_history.append(summary)
        self.last_diagnostics = {"method": "FedTGP", "summary": summary,
                                 "network_bytes_not_transmitted": model_bytes}

    def _fedsol_round(self, active):
        config = self.config
        server_state = state(self.server)
        for client_id in active:
            load_subset(self.clients[client_id], server_state, set(server_state))
            lr = config.learning_rate(config.lr, self.round)
            statistics = train_fedsol(
                self.clients[client_id], self.batches(client_id, self.round, 921),
                config, self.device, config.local_epochs, lr,
            )
            self.steps += statistics["optimizer_steps"]
            self.examples += statistics["examples"]
        self._aggregate(set(server_state), active)
        summary = {"round": self.round + 1, "active_clients": active,
                   "rho": config.fedsol_rho, "temperature": config.fedsol_temperature}
        self.method_history.append(summary)
        self.last_diagnostics = {"method": "FedSOL", "summary": summary}

    def _fedsa_round(self, active):
        config = self.config
        uploaded = []
        margins = []
        for client_id in active:
            lr = config.learning_rate(config.lr, self.round)
            statistics = train_fedsa(
                self.clients[client_id], self.batches(client_id, self.round, 941),
                config, self.device, config.local_epochs, lr,
                self.semantic_anchors, self.local_prototypes[client_id],
            )
            self.steps += statistics["optimizer_steps"]
            self.examples += statistics["examples"]
            margins.append(statistics["margin"])
            prototypes, counts = collect_class_prototypes(
                self.clients[client_id],
                self.batches(client_id, self.round, 942, shuffle=False),
                self.device, self.classes,
            )
            self.local_prototypes[client_id] = prototypes
            uploaded.append({"prototypes": prototypes,
                             "counts": torch.tensor([counts.get(i, 0)
                                                     for i in range(self.classes)])})
            self.statistic_examples += self.sizes[client_id]
        self.prototypes = aggregate_prototypes(uploaded, weighted=False)
        self.semantic_anchors = update_semantic_anchors(
            self.semantic_anchors, self.prototypes, config.fedsa_anchor_momentum
        )
        upload_count = sum(len(item["prototypes"]) for item in uploaded)
        self.bytes_sent += upload_count * (8 + config.feature_dim * 4)
        self.bytes_sent += len(active) * self.classes * config.feature_dim * 4
        summary = {"round": self.round + 1, "active_clients": active,
                   "mean_margin": float(np.mean(margins)),
                   "global_prototypes": len(self.prototypes)}
        self.method_history.append(summary)
        self.last_diagnostics = {"method": "FedSA", "summary": summary}

    def _cwfedavg_round(self, active):
        config = self.config
        # The paper uses classifier-weight norms as each client's estimated class
        # distribution, then creates a personalized mixture of class-wise models.
        for client_id, client in enumerate(self.clients):
            weights = class_distribution_weights(client)
            mix_classwise_models(client, self.class_models, weights.tolist())
        learned_weights = []
        for client_id in active:
            lr = config.learning_rate(config.lr, self.round)
            statistics = train_cwfedavg(
                self.clients[client_id], self.batches(client_id, self.round, 961),
                config, self.device, config.local_epochs, lr,
                self.class_distributions[client_id],
            )
            self.steps += statistics["optimizer_steps"]
            self.examples += statistics["examples"]
            learned_weights.append(class_distribution_weights(self.clients[client_id]))
        aggregate_classwise_models(
            self.class_models, [self.clients[index] for index in active],
            [self.sizes[index] for index in active], learned_weights,
        )
        payload = bytes_of(state(self.clients[active[0]]))
        self.bytes_sent += 2 * len(active) * payload
        summary = {"round": self.round + 1, "active_clients": active,
                   "classwise_models": self.classes,
                   "mean_estimated_distribution": np.mean(learned_weights, axis=0).tolist()}
        self.method_history.append(summary)
        self.last_diagnostics = {"method": "cwFedAvg", "summary": summary}

    def _fedsimsup_round(self, active):
        config = self.config
        for client_id in active:
            lr = config.learning_rate(config.lr, self.round)
            statistics = train_fedsimsup(
                self.clients[client_id], self.supervisors[client_id],
                lambda phase, client_id=client_id: self.batches(
                    client_id, self.round, phase
                ),
                config, self.device, lr,
            )
            self.steps += statistics["optimizer_steps"]
            self.examples += statistics["examples"]
        inactive_diagnostics = update_fedsimsup_inactive(
            self.clients, active, self.sizes, self.client_similarities,
            config, self.round,
        )
        payload = bytes_of(state(self.clients[active[0]]))
        self.bytes_sent += 2 * len(active) * payload
        summary = {"round": self.round + 1, "active_clients": active,
                   "inactive_clients": sorted(inactive_diagnostics),
                   "inactive_updates": inactive_diagnostics}
        self.method_history.append(summary)
        self.last_diagnostics = {
            "method": "FedSimSup", "summary": summary,
            "implementation": "paper equations 3-12; linked official repository unavailable",
        }

    def _fedrca_round(self, active):
        config = self.config
        if self.pixel_region_weights is None:
            index = build_pixel_region_index(self.cache, self.partition, config)
            self.pixel_region_assignments = index.assignments
            self.pixel_region_weights = torch.from_numpy(index.weights.copy())
            self.pixel_region_summaries = index.summaries
            self.pixel_kmeans_fits = index.kmeans_fits
            self.preprocessing_examples += sum(self.sizes)

        warmup = self.round < config.fedrca_warmup_rounds
        active_round = max(0, self.round - config.fedrca_warmup_rounds + 1)
        ramp = min(1.0, active_round / config.fedrca_ramp_rounds) if active_round else 0.0
        topology_scale = config.fedrca_topology_lambda * ramp
        if config.fedrca_variant == "no_rbf":
            topology_scale = 0.0

        teacher = (copy.deepcopy(self.server).to(self.device).eval()
                   if topology_scale else None)
        server_state = state(self.server)
        client_diagnostics = []
        for client_id in active:
            client = self.clients[client_id]
            # Global encoder and global head start from the same server state.
            # The client-specific head persists locally and is never uploaded.
            load_subset(client, server_state, set(server_state))
            statistics = self._train(
                client, client_id, self.round, batch_phase=71,
                epochs=config.local_epochs,
                teacher=teacher,
                topology_scale=topology_scale,
                topology_sample_weights=self.pixel_region_weights,
                return_indices=bool(topology_scale),
                freeze_bn=not warmup and config.fedrca_freeze_bn,
                max_grad_norm=config.fedrca_max_grad_norm,
                personal_head=self.personal_heads[client_id],
                class_weights=self.class_weights[client_id],
                keep_teacher_on_device=True,
            )
            client_diagnostics.append({
                "client": client_id,
                "topology_loss": statistics["topology_sum"] / max(statistics["examples"], 1),
                "global_ce": statistics["global_ce_sum"] / max(statistics["examples"], 1),
                "personal_ce": statistics["personal_ce_sum"] / max(statistics["examples"], 1),
                "gradient_clipped_steps": statistics["gradient_clipped_steps"],
            })

        if teacher is not None:
            teacher.cpu()

        combined = average([state(self.clients[index]) for index in active],
                           [self.sizes[index] for index in active], set(server_state))
        load_subset(self.server, combined, set(server_state))
        for client in self.clients:
            load_subset(client, combined, set(server_state))
        self.bytes_sent += 2 * len(active) * bytes_of(combined)

        summary = {
            "round": self.round + 1,
            "active_clients": active,
            "warmup": warmup,
            "ramp": ramp,
            "topology_scale": topology_scale,
            "pixel_partition_initialized": True,
            "pixel_kmeans_fits": self.pixel_kmeans_fits,
            "pixel_regions": sum(item["groups"] for item in self.pixel_region_summaries),
            "mean_topology_loss": float(np.mean([
                item["topology_loss"] for item in client_diagnostics
            ])),
            "mean_global_ce": float(np.mean([item["global_ce"] for item in client_diagnostics])),
            "mean_personal_ce": float(np.mean([item["personal_ce"] for item in client_diagnostics])),
            "bandwidths": self.server.topology_log_bandwidth.detach().exp().tolist(),
        }
        self.method_history.append(summary)
        self.last_diagnostics = {
            "method": "FedRCA",
            "variant": config.fedrca_variant,
            "summary": summary,
            "pixel_regions": copy.deepcopy(self.pixel_region_summaries),
            "clients": client_diagnostics,
            "privacy_boundary": "pixel clusters and assignments remain client-private; only model parameters are aggregated",
        }

    def evaluation_models(self):
        if self.config.algorithm in ("cwt", "fedseq"):
            return [self.server for _ in self.clients]
        if self.config.algorithm == "ditto":
            return self.personal
        if self.config.algorithm == "fedsimsup":
            return [ParallelEnsemble(client, supervisor)
                    for client, supervisor in zip(self.clients, self.supervisors)]
        if self.config.algorithm == "fedrca":
            models = []
            for client, personal_head in zip(self.clients, self.personal_heads):
                model = copy.deepcopy(client)
                model.head.load_state_dict(personal_head.state_dict())
                models.append(model)
            return models
        return self.clients

    def inference_prototypes(self):
        if self.config.algorithm == "fedtgp":
            return self.tgp_prototypes
        if self.config.algorithm in ("fedproto", "fedsa"):
            return self.prototypes
        return None

    def pack(self):
        return {
            "server": state(self.server),
            "clients": [state(model) for model in self.clients],
            "personal": [state(model) for model in self.personal],
            "personal_heads": [state(head) for head in self.personal_heads],
            "supervisors": [state(model) for model in self.supervisors],
            "tgp": state(self.tgp) if self.tgp is not None else None,
            "tgp_prototypes": copy.deepcopy(self.tgp_prototypes),
            "semantic_anchors": copy.deepcopy(self.semantic_anchors),
            "local_prototypes": copy.deepcopy(self.local_prototypes),
            "class_models": [state(model) for model in self.class_models],
            "prototypes": copy.deepcopy(self.prototypes),
            "pixel_region_assignments": copy.deepcopy(self.pixel_region_assignments),
            "pixel_region_weights": copy.deepcopy(self.pixel_region_weights),
            "pixel_region_summaries": copy.deepcopy(self.pixel_region_summaries),
            "pixel_kmeans_fits": self.pixel_kmeans_fits,
            "fedseq_groups": copy.deepcopy(self.fedseq_groups),
            "ap_weights": copy.deepcopy(self.ap_weights),
            "warmup_done": self.warmup_done,
            "round": self.round,
            "bytes_sent": self.bytes_sent,
            "steps": self.steps,
            "examples": self.examples,
            "statistic_examples": self.statistic_examples,
            "preprocessing_examples": self.preprocessing_examples,
            "method_history": copy.deepcopy(self.method_history),
            "active_history": copy.deepcopy(self.active_history),
            "last_diagnostics": copy.deepcopy(self.last_diagnostics),
            "train_seconds": self.train_seconds,
        }

    def restore(self, values):
        self.server.load_state_dict(values["server"])
        for model, weights in zip(self.clients, values["clients"]):
            model.load_state_dict(weights)
        for model, weights in zip(self.personal, values["personal"]):
            model.load_state_dict(weights)
        for head, weights in zip(self.personal_heads, values.get("personal_heads", [])):
            head.load_state_dict(weights)
        for model, weights in zip(self.supervisors, values.get("supervisors", [])):
            model.load_state_dict(weights)
        if self.tgp is not None and values.get("tgp") is not None:
            self.tgp.load_state_dict(values["tgp"])
        for model, weights in zip(self.class_models, values.get("class_models", [])):
            model.load_state_dict(weights)
        for key in ("prototypes", "ap_weights", "warmup_done", "round",
                    "bytes_sent", "steps", "examples", "statistic_examples",
                    "method_history", "train_seconds"):
            setattr(self, key, values[key])
        for key in ("tgp_prototypes", "semantic_anchors", "local_prototypes",
                    "active_history", "pixel_region_assignments",
                    "pixel_region_weights", "pixel_region_summaries",
                    "fedseq_groups"):
            if key in values:
                setattr(self, key, copy.deepcopy(values[key]))
        self.pixel_kmeans_fits = int(values.get("pixel_kmeans_fits", 0))
        self.preprocessing_examples = int(values.get("preprocessing_examples", 0))
        self.last_diagnostics = copy.deepcopy(values.get("last_diagnostics", {}))

    def method_summary(self):
        if self.config.algorithm != "fedrca":
            if self.config.algorithm in ("fedtgp", "fedsol", "fedsa", "cwfedavg", "fedsimsup",
                                         "cwt", "fedseq"):
                return {"method": self.config.algorithm,
                        "rounds": len(self.method_history),
                        "participation_rate": self.config.participation_rate,
                        "superclient_groups": (copy.deepcopy(self.fedseq_groups)
                                               if self.config.algorithm == "fedseq" else None),
                        "last": copy.deepcopy(self.method_history[-1])
                        if self.method_history else None}
            return None
        active = [item for item in self.method_history if not item["warmup"]]
        return {
            "variant": self.config.fedrca_variant,
            "rounds": len(self.method_history),
            "warmup_rounds": sum(item["warmup"] for item in self.method_history),
            "active_rounds": len(active),
            "pixel_regions": sum(item["groups"] for item in self.pixel_region_summaries),
            "pixel_kmeans_fits": self.pixel_kmeans_fits,
            "pixel_partition_builds": int(bool(self.pixel_region_summaries)),
            "mean_active_topology_loss": float(np.mean([
                item["mean_topology_loss"] for item in active
            ])) if active else None,
            "bandwidths": (active[-1]["bandwidths"] if active else
                           self.server.topology_log_bandwidth.detach().exp().tolist()),
            "design": "one-time client-private class-wise pixel PCA/K-means, region-balanced homogeneous-model RBF topology distillation, fully aggregated global head, detached private class-balanced heads",
        }
