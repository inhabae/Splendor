from __future__ import annotations

import torch
from torch import Tensor, nn

from .state_schema import ACTION_DIM, STATE_DIM


class MaskedPolicyValueNet(nn.Module):
    def __init__(self, input_dim: int = STATE_DIM, hidden_dim: int = 256, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head[0].bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(x)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return policy_logits, value
