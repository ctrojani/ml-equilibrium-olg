from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_c_rate = nn.Parameter(torch.log(torch.tensor([[0.10]])))
        self.risky_share_raw = nn.Parameter(torch.tensor([[0.50]]))

    def forward(self, wealth):
        if wealth.ndim == 1:
            wealth = wealth.unsqueeze(-1)

        c_rate = torch.exp(self.log_c_rate).expand_as(wealth)
        risky_share = self.risky_share_raw.expand_as(wealth)

        consumption = c_rate * wealth
        risky_position = risky_share * wealth
        return consumption, risky_position, c_rate, risky_share


class PolicyNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        wealth_scale: float = 1.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.wealth_scale = float(max(wealth_scale, 1e-8))

        layers = []
        in_features = 1
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.Tanh())
            in_features = hidden_dim

        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.base = nn.Parameter(torch.zeros(1, 2))
        self.head = nn.Linear(in_features, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, wealth):
        if wealth.ndim == 1:
            wealth = wealth.unsqueeze(-1)

        x = (2.0 * wealth / self.wealth_scale) - 1.0
        features = self.trunk(x)
        raw = self.base + self.head(features)

        c_rate = F.softplus(raw[:, :1]) + 1e-6
        risky_share = raw[:, 1:2]

        consumption = c_rate * wealth
        risky_position = risky_share * wealth
        return consumption, risky_position, c_rate, risky_share
