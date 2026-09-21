import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from .config import DATASETS
from .data import loader
from .models import MedicalCNN, average, bytes_of, state


LOCAL_VARIANTS = {"pl_uniform", "psl_uniform"}
SERIAL_VARIANTS = {"ps_uniform", "psl_uniform"}


def train_client(parallel, serial, local, batches, config, device, round_index):
    """Train exactly the dense branches enabled by the selected ablation."""
    branches = [("parallel", parallel)]
    if serial is not None:
        branches.append(("serial", serial))
    if local is not None:
        branches.append(("local", local))
    models = [model for _, model in branches]
    for model in models:
        model.to(device).train()

    optimizer = torch.optim.SGD(
        [parameter for model in models for parameter in model.parameters()],
        lr=config.learning_rate(config.lr, round_index),
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    diagnostics = {
        "examples": 0,
        "optimizer_steps": 0,
        "branch_optimizer_steps": 0,
        "parallel_ce_sum": 0.0,
        "serial_ce_sum": None,
        "local_ce_sum": None,
        "gradient_clipped_steps": 0,
    }
    ce_sums = {
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name, _ in branches
    }
    clipped_steps = torch.zeros((), device=device, dtype=torch.long)

    for _ in range(config.local_epochs):
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for name, model in branches:
                _, logits = model(images)
                per_sample = F.cross_entropy(logits, labels, reduction="none")
                losses.append(per_sample)
                ce_sums[name] += per_sample.detach().double().sum()
            objective = sum(value.mean() for value in losses)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite FedTriad objective")
            objective.backward()
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            clip_limit = (
                float("inf")
                if config.triad_variant == "p_only"
                else config.triad_max_grad_norm
            )
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, clip_limit, error_if_nonfinite=True
            )
            clipped_steps += (norm > clip_limit).to(torch.long)
            optimizer.step()
            diagnostics["examples"] += len(labels)
            diagnostics["optimizer_steps"] += 1
            diagnostics["branch_optimizer_steps"] += len(branches)

    for name in ce_sums:
        diagnostics[name + "_ce_sum"] = float(ce_sums[name].cpu())
    diagnostics["gradient_clipped_steps"] = int(clipped_steps.cpu())
    if device.type != "cuda":
        for model in models:
            model.cpu()
    return diagnostics


class Federation:
    """Parallel consensus, serial transfer and persistent private paths."""

    def __init__(self, config, cache, partition, device):
        self.config, self.cache, self.partition, self.device = (
            config, cache, partition, device
        )
        channels, self.classes = DATASETS[config.dataset]
        self.parallel_server = MedicalCNN(
            channels, self.classes, config.width, config.feature_dim
        )
        self.parallel_clients = [
            copy.deepcopy(self.parallel_server) for _ in range(config.clients)
        ]
        self.serial = (
            copy.deepcopy(self.parallel_server)
            if config.triad_variant in SERIAL_VARIANTS
            else None
        )
        self.locals = (
            [copy.deepcopy(self.parallel_server) for _ in range(config.clients)]
            if config.triad_variant in LOCAL_VARIANTS
            else []
        )
        self.sizes = [len(indices) for indices in partition["train"]]
        self.round = 0
        self.bytes_sent = 0
        self.optimizer_steps = 0
        self.branch_optimizer_steps = 0
        self.train_examples = 0
        self.train_seconds = 0.0
        self.last_diagnostics = {}
        self.method_history = []
        self.active_history = []
        self.serial_order_history = []
        self.models_resident_on_device = device.type == "cuda"
        if self.models_resident_on_device:
            modules = [self.parallel_server, *self.parallel_clients]
            if self.serial is not None:
                modules.append(self.serial)
            modules.extend(self.locals)
            for module in modules:
                module.to(device)

    def batches(self, client_id):
        # Locked against the unified baseline protocol.
        seed = self.config.seed * 10000019 + self.round * 100003 + client_id * 101
        return loader(
            self.cache,
            "train",
            self.partition["train"][client_id],
            self.config,
            shuffle=True,
            seed=seed,
        )

    def active_clients(self):
        count = max(1, int(round(
            self.config.clients * self.config.participation_rate
        )))
        rng = np.random.RandomState(
            self.config.seed * 10000019 + self.round * 100003 + 1709
        )
        return sorted(rng.choice(
            self.config.clients, size=count, replace=False
        ).tolist())

    def serial_order(self, active):
        pivot = self.round % self.config.clients
        return sorted(
            active, key=lambda client_id: (client_id - pivot) % self.config.clients
        )

    def train_round(self):
        start = time.perf_counter()
        active = self.active_clients()
        order = self.serial_order(active)
        parallel_anchor = state(self.parallel_server)
        client_diagnostics = []

        for client_id in order:
            self.parallel_clients[client_id].load_state_dict(parallel_anchor)
            statistics = train_client(
                self.parallel_clients[client_id],
                self.serial,
                self.locals[client_id] if self.locals else None,
                self.batches(client_id),
                self.config,
                self.device,
                self.round,
            )
            self.optimizer_steps += statistics["optimizer_steps"]
            self.branch_optimizer_steps += statistics["branch_optimizer_steps"]
            branch_count = 1 + int(self.serial is not None) + int(bool(self.locals))
            self.train_examples += statistics["examples"] * branch_count
            client_diagnostics.append({"client": client_id, **statistics})

        weights = [self.sizes[client_id] for client_id in active]
        combined = average(
            [state(self.parallel_clients[client_id]) for client_id in active], weights
        )
        self.parallel_server.load_state_dict(combined)
        parallel_bytes = bytes_of(combined)
        self.bytes_sent += 2 * len(active) * parallel_bytes
        if self.serial is not None:
            # server -> first, client-to-client hops, last -> server
            self.bytes_sent += (len(active) + 1) * bytes_of(self.serial)

        count = max(sum(item["examples"] for item in client_diagnostics), 1)
        summary = {
            "round": self.round + 1,
            "variant": self.config.triad_variant,
            "active_clients": active,
            "serial_order": order if self.serial is not None else [],
            "mean_parallel_ce": sum(
                item["parallel_ce_sum"] for item in client_diagnostics
            ) / count,
            "mean_serial_ce": (
                sum(item["serial_ce_sum"] for item in client_diagnostics) / count
                if self.serial is not None
                else None
            ),
            "mean_local_ce": (
                sum(item["local_ce_sum"] for item in client_diagnostics) / count
                if self.locals
                else None
            ),
            "gradient_clipped_steps": sum(
                item["gradient_clipped_steps"] for item in client_diagnostics
            ),
        }
        self.last_diagnostics = {
            "method": "FedTriad",
            "summary": summary,
            "clients": client_diagnostics,
        }
        self.method_history.append(summary)
        self.active_history.append(active)
        self.serial_order_history.append(summary["serial_order"])
        self.round += 1
        self.train_seconds += time.perf_counter() - start

    def _forward_logits(self, images, client_id=None, include_local=False):
        _, parallel_logits = self.parallel_server(images)
        serial_logits = None
        local_logits = None
        if self.serial is not None:
            _, serial_logits = self.serial(images)
        if include_local and self.locals:
            if client_id is None:
                raise ValueError("Personalized inference requires a client id")
            _, local_logits = self.locals[client_id](images)
        return parallel_logits, serial_logits, local_logits

    @staticmethod
    def _mix(logits, personalized):
        parallel, serial, local = logits
        enabled = [parallel]
        if serial is not None:
            enabled.append(serial)
        if personalized and local is not None:
            enabled.append(local)
        return sum(F.softmax(value, dim=1) for value in enabled) / len(enabled)

    def _move_for_inference(self, client_id=None, all_locals=False):
        modules = [self.parallel_server]
        if self.serial is not None:
            modules.append(self.serial)
        if all_locals:
            modules.extend(self.locals)
        elif client_id is not None and self.locals:
            modules.append(self.locals[client_id])
        for module in modules:
            module.to(self.device).eval()
        return modules

    @torch.no_grad()
    def predict_client(self, batches, client_id):
        modules = self._move_for_inference(client_id=client_id)
        labels, personal_parts, global_parts = [], [], []
        include_local = self.config.triad_variant in LOCAL_VARIANTS
        for images, target in batches:
            logits = self._forward_logits(
                images.to(self.device, non_blocking=True), client_id, include_local
            )
            labels.append(target.numpy())
            personal_parts.append(
                self._mix(logits, personalized=True).cpu().numpy()
            )
            global_parts.append(
                self._mix(logits, personalized=False).cpu().numpy()
            )
        if not self.models_resident_on_device:
            for module in modules:
                module.cpu()
        return (
            np.concatenate(labels),
            np.concatenate(personal_parts),
            np.concatenate(global_parts),
        )

    @torch.no_grad()
    def predict_client_branches(self, batches, client_id):
        """Return P/S/L probabilities for dense PSL risk calibration."""
        if not self.locals or self.serial is None:
            raise RuntimeError("Three-branch prediction requires dense P/S/L")
        modules = self._move_for_inference(client_id=client_id)
        labels, parts = [], []
        for images, target in batches:
            logits = self._forward_logits(
                images.to(self.device, non_blocking=True), client_id, include_local=True
            )
            probabilities = torch.stack(
                [F.softmax(value, dim=1) for value in logits], dim=1
            )
            labels.append(target.numpy())
            parts.append(probabilities.cpu().numpy())
        if not self.models_resident_on_device:
            for module in modules:
                module.cpu()
        return np.concatenate(labels), np.concatenate(parts)

    @torch.no_grad()
    def predict_global(self, batches):
        modules = self._move_for_inference()
        labels, parts = [], []
        for images, target in batches:
            logits = self._forward_logits(
                images.to(self.device, non_blocking=True), include_local=False
            )
            labels.append(target.numpy())
            parts.append(self._mix(logits, personalized=False).cpu().numpy())
        if not self.models_resident_on_device:
            for module in modules:
                module.cpu()
        return np.concatenate(labels), np.concatenate(parts)

    @torch.no_grad()
    def predict_personalized_ensemble(self, batches):
        if not self.locals:
            return self.predict_global(batches)
        modules = self._move_for_inference(all_locals=True)
        labels, parts = [], []
        for images, target in batches:
            images = images.to(self.device, non_blocking=True)
            shared = self._forward_logits(images, include_local=False)
            total = None
            for local_model in self.locals:
                _, local_logits = local_model(images)
                probabilities = self._mix(
                    (shared[0], shared[1], local_logits), personalized=True
                )
                total = (
                    probabilities / len(self.locals)
                    if total is None
                    else total + probabilities / len(self.locals)
                )
            labels.append(target.numpy())
            parts.append(total.cpu().numpy())
        if not self.models_resident_on_device:
            for module in modules:
                module.cpu()
        return np.concatenate(labels), np.concatenate(parts)

    def pack(self):
        return {
            "parallel_server": state(self.parallel_server),
            "serial": state(self.serial) if self.serial is not None else None,
            "locals": [state(model) for model in self.locals],
            "round": self.round,
            "bytes_sent": self.bytes_sent,
            "optimizer_steps": self.optimizer_steps,
            "branch_optimizer_steps": self.branch_optimizer_steps,
            "train_examples": self.train_examples,
            "train_seconds": self.train_seconds,
            "last_diagnostics": copy.deepcopy(self.last_diagnostics),
            "method_history": copy.deepcopy(self.method_history),
            "active_history": copy.deepcopy(self.active_history),
            "serial_order_history": copy.deepcopy(self.serial_order_history),
        }

    def restore(self, values):
        self.parallel_server.load_state_dict(values["parallel_server"])
        if self.serial is not None:
            self.serial.load_state_dict(values["serial"])
        for model, weights in zip(self.locals, values["locals"]):
            model.load_state_dict(weights)
        for key in (
            "round",
            "bytes_sent",
            "optimizer_steps",
            "branch_optimizer_steps",
            "train_examples",
            "train_seconds",
        ):
            setattr(self, key, values[key])
        for key in (
            "last_diagnostics",
            "method_history",
            "active_history",
            "serial_order_history",
        ):
            setattr(self, key, copy.deepcopy(values[key]))

    def method_summary(self):
        return {
            "method": "FedTriad",
            "variant": self.config.triad_variant,
            "rounds": len(self.method_history),
            "parallel_path": True,
            "serial_path": self.serial is not None,
            "private_path": bool(self.locals),
            "dense_training": True,
            "last": (
                copy.deepcopy(self.method_history[-1])
                if self.method_history
                else None
            ),
            "design": (
                "parallel FedAvg consensus + rotating serial cross-hospital transfer + "
                "persistent private local path; optional post-training global class-risk "
                "calibration uses compressed branch-by-class loss statistics"
            ),
        }
