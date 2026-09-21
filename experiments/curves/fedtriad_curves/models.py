from collections import OrderedDict

import torch
from torch import nn


class MedicalCNN(nn.Module):
    """The exact 28x28 backbone used by the locked comparison protocol."""

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
            ("projection", nn.Linear(width * 16, feature_dim)), ("activation", nn.ReLU()),
        ]))
        self.head = nn.Linear(feature_dim, classes)

    def forward(self, images):
        features = self.encoder(images)
        return features, self.head(features)

def state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def average(states, weights):
    if not states or len(states) != len(weights) or sum(weights) <= 0:
        raise ValueError("Invalid aggregation inputs")
    normalized = [float(weight) / sum(weights) for weight in weights]
    result = {}
    for key in states[0]:
        first = states[0][key]
        if first.is_floating_point():
            result[key] = sum((item[key] * weight for item, weight in zip(states, normalized)),
                              torch.zeros_like(first))
        else:
            result[key] = torch.stack([item[key] for item in states]).max(dim=0).values
    return result


def bytes_of(module_or_state):
    values = module_or_state.state_dict() if isinstance(module_or_state, nn.Module) else module_or_state
    return sum(value.numel() * value.element_size() for value in values.values())
