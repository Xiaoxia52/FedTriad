from collections import OrderedDict
import math
import torch
from torch import nn


class MedicalCNN(nn.Module):
    """Shared medical CNN used by FedRCA and every comparison baseline."""
    def __init__(self, channels, classes, width=32, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(channels, width, 3, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(width)), ("relu1", nn.ReLU()), ("pool1", nn.MaxPool2d(2)),
            ("conv2", nn.Conv2d(width, width * 2, 3, padding=1, bias=False)),
            ("bn2", nn.BatchNorm2d(width * 2)), ("relu2", nn.ReLU()), ("pool2", nn.MaxPool2d(2)),
            ("conv3", nn.Conv2d(width * 2, width * 4, 3, padding=1, bias=False)),
            ("bn3", nn.BatchNorm2d(width * 4)), ("relu3", nn.ReLU()),
            ("pool3", nn.AdaptiveAvgPool2d((2, 2))), ("flatten", nn.Flatten()),
            ("projection", nn.Linear(width * 16, feature_dim)), ("activation", nn.ReLU())]))
        self.head = nn.Linear(feature_dim, classes)
        # Two learnable RBF temperatures for pool2/relu3 topology.  They belong
        # to the shared encoder state, while the classifier head is never used
        # by the topology loss.
        self.topology_log_bandwidth = nn.Parameter(torch.full((2,), math.log(1.0)))

    def encode_with_maps(self, images):
        maps = []
        value = images
        for name, module in self.encoder.named_children():
            value = module(value)
            if name in ("pool2", "relu3"):
                maps.append(value)
        if len(maps) != 2:
            raise RuntimeError("MedicalCNN topology stages are incomplete")
        return value, maps

    def topology_maps(self, images):
        """Run only the convolutional stages needed by the RBF teacher."""
        maps = []
        value = images
        for name, module in self.encoder.named_children():
            value = module(value)
            if name in ("pool2", "relu3"):
                maps.append(value)
            if name == "relu3":
                break
        return maps

    def forward(self, images):
        features = self.encoder(images)
        return features, self.head(features)

    def forward_with_maps(self, images):
        features, maps = self.encode_with_maps(images)
        return features, self.head(features), maps


class ParallelEnsemble(nn.Module):
    """FedSimSup inference model: add supervisor and inter-learning logits."""
    def __init__(self, inter_model, supervisor):
        super().__init__()
        self.inter_model = inter_model
        self.supervisor = supervisor

    def forward(self, images):
        features, inter_logits = self.inter_model(images)
        _, supervisor_logits = self.supervisor(images)
        return features, inter_logits + supervisor_logits


def bn_keys(model):
    return {name + "." + key for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm) for key in module.state_dict()}


def state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_subset(model, values, keys):
    current = model.state_dict()
    with torch.no_grad():
        for key in keys:
            current[key].copy_(values[key])


def average(states, weights, keys=None):
    if not states or len(states) != len(weights) or sum(weights) <= 0:
        raise ValueError("Invalid aggregation inputs")
    weights = [float(w) / sum(weights) for w in weights]
    if any(w < 0 for w in weights):
        raise ValueError("Negative aggregation weight")
    output = {}
    for key in (states[0].keys() if keys is None else keys):
        first = states[0][key]
        if first.is_floating_point():
            output[key] = sum((s[key] * w for s, w in zip(states, weights)), torch.zeros_like(first))
        else:
            # BN counters are integers, not model parameters. Never average them as float.
            output[key] = torch.stack([s[key] for s in states]).max(dim=0).values
    return output


def bytes_of(values, keys=None):
    return sum(values[k].numel() * values[k].element_size() for k in (values if keys is None else keys))
