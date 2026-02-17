from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FParamSpec:
    input_dim: int
    hidden_dim_1: int
    hidden_dim_2: int
    output_dim: int

    @property
    def param_dim(self) -> int:
        return (
            self.hidden_dim_1 * self.input_dim
            + self.hidden_dim_1
            + self.hidden_dim_2 * self.hidden_dim_1
            + self.hidden_dim_2
            + self.output_dim * self.hidden_dim_2
            + self.output_dim
        )

    def split(
        self,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if theta.dim() != 1:
            raise ValueError("theta must be a flat 1D tensor")
        if theta.numel() != self.param_dim:
            raise ValueError(f"invalid theta size: got {theta.numel()}, expected {self.param_dim}")

        offset = 0
        w1_size = self.hidden_dim_1 * self.input_dim
        b1_size = self.hidden_dim_1
        w2_size = self.hidden_dim_2 * self.hidden_dim_1
        b2_size = self.hidden_dim_2
        w3_size = self.output_dim * self.hidden_dim_2
        b3_size = self.output_dim

        w1 = theta[offset : offset + w1_size].view(self.hidden_dim_1, self.input_dim)
        offset += w1_size
        b1 = theta[offset : offset + b1_size].view(self.hidden_dim_1)
        offset += b1_size
        w2 = theta[offset : offset + w2_size].view(self.hidden_dim_2, self.hidden_dim_1)
        offset += w2_size
        b2 = theta[offset : offset + b2_size].view(self.hidden_dim_2)
        offset += b2_size
        w3 = theta[offset : offset + w3_size].view(self.output_dim, self.hidden_dim_2)
        offset += w3_size
        b3 = theta[offset : offset + b3_size].view(self.output_dim)

        return w1, b1, w2, b2, w3, b3


class ClassifierNet(nn.Module):
    def __init__(self, spec: FParamSpec) -> None:
        super().__init__()
        self.spec = spec
        self.fc1 = nn.Linear(spec.input_dim, spec.hidden_dim_1)
        self.fc2 = nn.Linear(spec.hidden_dim_1, spec.hidden_dim_2)
        self.fc3 = nn.Linear(spec.hidden_dim_2, spec.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1))
        return self.fc3(h2)


class HyperNet(nn.Module):
    def __init__(self, meta_dim: int, output_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)



def functional_mlp_forward(x: torch.Tensor, theta: torch.Tensor, spec: FParamSpec) -> torch.Tensor:
    w1, b1, w2, b2, w3, b3 = spec.split(theta)
    h1 = F.gelu(F.linear(x, w1, b1))
    h2 = F.gelu(F.linear(h1, w2, b2))
    return F.linear(h2, w3, b3)



def flatten_classifier_params(model: ClassifierNet) -> torch.Tensor:
    flat_parts = [
        model.fc1.weight.detach().reshape(-1),
        model.fc1.bias.detach().reshape(-1),
        model.fc2.weight.detach().reshape(-1),
        model.fc2.bias.detach().reshape(-1),
        model.fc3.weight.detach().reshape(-1),
        model.fc3.bias.detach().reshape(-1),
    ]
    return torch.cat(flat_parts, dim=0)



def load_classifier_params_from_flat(model: ClassifierNet, theta: torch.Tensor) -> None:
    spec = model.spec
    w1, b1, w2, b2, w3, b3 = spec.split(theta)
    with torch.no_grad():
        model.fc1.weight.copy_(w1)
        model.fc1.bias.copy_(b1)
        model.fc2.weight.copy_(w2)
        model.fc2.bias.copy_(b2)
        model.fc3.weight.copy_(w3)
        model.fc3.bias.copy_(b3)
