"""Core operations for the recent comparison methods.

FedTGP, FedSOL, FedSA and cwFedAvg follow their public reference code.  The
FedSimSup implementation follows Eqs. (3)-(12) of the ICCV 2025 paper because
the repository linked by the paper is currently unavailable.
"""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import average, load_subset, state


class SAM(torch.optim.Optimizer):
    """Two-step SAM optimizer used by the official FedSOL implementation."""
    def __init__(self, params, rho, lr, momentum, weight_decay):
        defaults = dict(rho=rho, lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.base_optimizer = torch.optim.SGD(
            self.param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        norms = [p.grad.norm(2) for group in self.param_groups
                 for p in group["params"] if p.grad is not None]
        grad_norm = torch.norm(torch.stack(norms), 2) if norms else torch.tensor(0.0)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["old_p"] = parameter.data.clone()
                parameter.add_(parameter.grad * scale.to(parameter))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.data.copy_(self.state[parameter]["old_p"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()


def _set_bn_momentum(model, enabled):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if enabled and hasattr(module, "_fedsol_momentum"):
                module.momentum = module._fedsol_momentum
            elif not enabled:
                module._fedsol_momentum = module.momentum
                module.momentum = 0


def train_fedsol(model, batches, config, device, epochs, lr):
    """Official fixed FedSOL: KL perturbation followed by CE at perturbed weights."""
    model.to(device).train()
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = SAM(model.parameters(), config.fedsol_rho, lr,
                    config.momentum, config.weight_decay)
    seen = steps = 0
    loss_sum = 0.0
    temperature = config.fedsol_temperature
    for _ in range(epochs):
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            _set_bn_momentum(model, True)
            _, logits = model(images)
            with torch.no_grad():
                teacher_logits = teacher(images)[1]
                target = F.softmax(teacher_logits / temperature, dim=1)
            F.kl_div(F.log_softmax(logits / temperature, dim=1), target,
                     reduction="batchmean").backward()
            optimizer.first_step()
            _set_bn_momentum(model, False)
            loss = F.cross_entropy(model(images)[1], labels)
            loss.backward()
            optimizer.second_step()
            _set_bn_momentum(model, True)
            seen += len(labels)
            steps += 2
            loss_sum += float(loss.detach()) * len(labels)
    model.cpu()
    teacher.cpu()
    return {"loss_sum": loss_sum, "global_ce_sum": loss_sum,
            "personal_ce_sum": 0.0, "rbf_sum": 0.0, "geometry_sum": 0.0,
            "examples": seen, "optimizer_steps": steps,
            "gradient_clipped_steps": 0}


@torch.no_grad()
def collect_class_prototypes(model, batches, device, classes):
    model.to(device).eval()
    sums, counts = {}, {}
    for images, labels in batches:
        features = model(images.to(device))[0].cpu()
        for label in labels.unique().tolist():
            rows = features[labels == label]
            sums[label] = sums.get(label, torch.zeros_like(rows[0])) + rows.sum(0)
            counts[label] = counts.get(label, 0) + len(rows)
    model.cpu()
    return {label: sums[label] / counts[label] for label in sums}, counts


class TrainableGlobalPrototypes(nn.Module):
    def __init__(self, classes, feature_dim):
        super().__init__()
        self.embedding = nn.Embedding(classes, feature_dim)
        self.middle = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU())
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, labels):
        return self.fc(self.middle(self.embedding(labels)))


def update_fedtgp(generator, uploaded, config, classes, device, lr):
    """Train global prototypes with FedTGP adaptive-margin contrastive learning."""
    if not uploaded:
        return {}, {"server_loss": None, "class_gap": []}
    per_class = {}
    for prototype, label in uploaded:
        per_class.setdefault(label, []).append(prototype)
    means = {label: torch.stack(values).mean(0) for label, values in per_class.items()}
    finite_gaps = []
    gaps = torch.full((classes,), float("inf"), device=device)
    labels_present = sorted(means)
    for a, left in enumerate(labels_present):
        for right in labels_present[a + 1:]:
            distance = torch.norm(means[left].to(device) - means[right].to(device), p=2)
            gaps[left] = torch.minimum(gaps[left], distance)
            gaps[right] = torch.minimum(gaps[right], distance)
            finite_gaps.append(distance)
    fallback = min(finite_gaps).detach() if finite_gaps else torch.tensor(0.0, device=device)
    gaps[torch.isinf(gaps)] = fallback
    margin = min(float(gaps.max()), config.fedtgp_margin_threshold)

    generator.to(device).train()
    optimizer = torch.optim.SGD(generator.parameters(), lr=lr)
    prototypes = torch.stack([item[0] for item in uploaded]).to(device)
    targets = torch.tensor([item[1] for item in uploaded], dtype=torch.long, device=device)
    loss = prototypes.new_zeros(())
    for _ in range(config.fedtgp_server_epochs):
        generated = generator(torch.arange(classes, device=device))
        distances = torch.cdist(prototypes, generated)
        distances = distances + F.one_hot(targets, classes) * margin
        loss = F.cross_entropy(-distances, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    generator.eval()
    with torch.no_grad():
        values = generator(torch.arange(classes, device=device)).cpu()
    generator.cpu()
    return {label: values[label] for label in range(classes)}, {
        "server_loss": float(loss.detach()), "class_gap": gaps.cpu().tolist(),
        "margin": margin,
    }


def initialize_semantic_anchors(classes, feature_dim, device):
    projector = nn.Sequential(
        nn.Linear(feature_dim, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(),
        nn.Linear(feature_dim, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(),
    ).to(device)
    with torch.no_grad():
        anchors = projector(torch.randn(classes, feature_dim, device=device)).cpu()
    return anchors


def _minimum_gap(prototypes):
    if prototypes is None or len(prototypes) < 2:
        return -float("inf")
    values = torch.stack(list(prototypes.values()))
    distances = torch.cdist(values, values)
    distances.fill_diagonal_(float("inf"))
    return float(distances.min(dim=1).values.mean())


def train_fedsa(model, batches, config, device, epochs, lr, anchors, prior_prototypes):
    model.to(device).train()
    anchors = anchors.to(device)
    global_distances = torch.cdist(anchors, anchors)
    global_distances.fill_diagonal_(float("inf"))
    global_gap = float(global_distances.min(dim=1).values.mean())
    gap = max(global_gap, _minimum_gap(prior_prototypes))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    seen = steps = 0
    loss_sum = 0.0
    for _ in range(epochs):
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            features, logits = model(images)
            anchor_logits = model.head(anchors)
            calibration = F.cross_entropy(
                anchor_logits, torch.arange(len(anchors), device=device)
            )
            regularization = F.mse_loss(features, anchors[labels])
            distances = torch.cdist(features, anchors).clamp_min(1e-12)
            contrastive = F.cross_entropy(
                -(distances + F.one_hot(labels, len(anchors)) * gap), labels
            )
            loss = (F.cross_entropy(logits, labels)
                    + config.fedsa_calibration_lambda * calibration
                    + config.fedsa_anchor_lambda * regularization
                    + config.fedsa_margin_lambda * contrastive)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += len(labels)
            steps += 1
            loss_sum += float(loss.detach()) * len(labels)
    model.cpu()
    return {"loss_sum": loss_sum, "global_ce_sum": loss_sum,
            "personal_ce_sum": 0.0, "rbf_sum": 0.0, "geometry_sum": 0.0,
            "examples": seen, "optimizer_steps": steps,
            "gradient_clipped_steps": 0, "margin": gap}


def update_semantic_anchors(anchors, prototypes, momentum):
    result = anchors.clone()
    for label, prototype in prototypes.items():
        result[label] = momentum * result[label] + (1 - momentum) * prototype
    return result


def class_distribution_weights(model):
    norms = model.head.weight.detach().norm(dim=1).clamp_min(1e-12)
    return (norms / norms.sum()).cpu().numpy()


def mix_classwise_models(target, class_models, weights):
    keys = set(target.state_dict())
    mixed = average([state(model) for model in class_models], weights, keys)
    load_subset(target, mixed, keys)


def train_cwfedavg(model, batches, config, device, epochs, lr, target_distribution):
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    target = torch.tensor(target_distribution, dtype=torch.float32, device=device)
    seen = steps = 0
    loss_sum = 0.0
    for _ in range(epochs):
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)[1]
            norms = model.head.weight.norm(dim=1).clamp_min(1e-12)
            approximated = norms / norms.sum()
            wdr = torch.norm(target - approximated, p=2)
            loss = F.cross_entropy(logits, labels) + 0.5 * config.cwfedavg_wdr_lambda * wdr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += len(labels)
            steps += 1
            loss_sum += float(loss.detach()) * len(labels)
    model.cpu()
    return {"loss_sum": loss_sum, "global_ce_sum": loss_sum,
            "personal_ce_sum": 0.0, "rbf_sum": 0.0, "geometry_sum": 0.0,
            "examples": seen, "optimizer_steps": steps,
            "gradient_clipped_steps": 0}


def aggregate_classwise_models(class_models, clients, client_sizes, client_weights):
    snapshots = [state(model) for model in clients]
    keys = set(snapshots[0])
    matrix = np.asarray(client_weights, dtype=float) * np.asarray(client_sizes)[:, None]
    for label, class_model in enumerate(class_models):
        weights = matrix[:, label]
        if weights.sum() <= 0:
            weights = np.asarray(client_sizes, dtype=float)
        load_subset(class_model, average(snapshots, weights.tolist(), keys), keys)


def cosine_similarity_matrix(class_counts):
    values = np.asarray(class_counts, dtype=float)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    similarity = values @ values.T / np.maximum(norms @ norms.T, 1e-12)
    return np.clip(similarity, 0.0, 1.0)


def train_fedsimsup(inter_model, supervisor, batches_factory, config, device, lr):
    """Alternating optimization in FedSimSup Eqs. (4), (5), (11), and (12)."""
    def phase(train_model, fixed_model, phase_id):
        train_model.to(device).train()
        fixed_model.to(device).eval()
        for parameter in train_model.parameters():
            parameter.requires_grad_(True)
        for parameter in fixed_model.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.SGD(train_model.parameters(), lr=lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        local_seen = local_steps = 0
        local_loss = 0.0
        for _ in range(config.local_epochs):
            for images, labels in batches_factory(phase_id):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    fixed_logits = fixed_model(images)[1]
                logits = train_model(images)[1] + fixed_logits
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                local_seen += len(labels)
                local_steps += 1
                local_loss += float(loss.detach()) * len(labels)
        return local_seen, local_steps, local_loss

    seen_s, steps_s, loss_s = phase(supervisor, inter_model, 811)
    seen_i, steps_i, loss_i = phase(inter_model, supervisor, 812)
    for model in (inter_model, supervisor):
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.cpu()
    return {"loss_sum": loss_s + loss_i, "global_ce_sum": loss_s + loss_i,
            "personal_ce_sum": 0.0, "rbf_sum": 0.0, "geometry_sum": 0.0,
            "examples": seen_s + seen_i, "optimizer_steps": steps_s + steps_i,
            "gradient_clipped_steps": 0}


def update_fedsimsup_inactive(clients, active, sizes, similarities, config, round_index):
    """Paper Eqs. (7)-(10): update only clients absent from this round."""
    active = list(active)
    snapshots = [state(clients[index]) for index in active]
    keys = set(snapshots[0])
    selected_total = sum(sizes[index] for index in active)
    k = len(active)
    threshold = config.fedsimsup_c * config.rounds ** config.fedsimsup_gamma
    t = round_index + 1
    beta = 1.0 if t < threshold else (threshold / t) ** 2
    diagnostics = {}
    for client_id, client in enumerate(clients):
        if client_id in active:
            continue
        weights = np.asarray([similarities[client_id, index] for index in active], dtype=float)
        if weights.sum() <= 0:
            weights = np.asarray([sizes[index] for index in active], dtype=float)
        weights /= weights.sum()
        learned = average(snapshots, weights.tolist(), keys)
        old = state(client)
        lam = selected_total / (selected_total + k * sizes[client_id])
        alpha = lam * beta
        mixed = {}
        for key in keys:
            if old[key].is_floating_point():
                mixed[key] = (1 - alpha) * old[key] + alpha * learned[key]
            else:
                mixed[key] = learned[key]
        load_subset(client, mixed, keys)
        diagnostics[client_id] = {"alpha": alpha, "weights": weights.tolist()}
    return diagnostics
